"""
PPStructure processing tasks for advanced document layout analysis.
Uses PaddleOCR's PPStructure for layout detection, OCR, and structure analysis.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import tempfile
import shutil
from pathlib import Path
import time
import datetime
import warnings
import logging
import traceback
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Fix numpy compatibility issue with deprecated np.bool
try:
    # For numpy >= 1.20, np.bool is deprecated and removed
    # This creates a compatibility alias to avoid errors
    if not hasattr(np, 'bool'):
        np.bool = bool
        np.int = int
        np.float = float
        np.complex = complex
        np.object = object
        np.unicode = str
        np.str = str
    
    # Additional compatibility for older code that might use these
    warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
    warnings.filterwarnings("ignore", message=".*np.bool.*deprecated.*")
    
except Exception:
    pass  # If there are any issues with the compatibility fix, just continue

# Import PaddleOCR components conditionally to prevent errors in lite containers
try:
    from paddleocr import PPStructureV3
    from pdf2image import convert_from_bytes
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
    PADDLEPADDLE_AVAILABLE = True
except ImportError as e:
    # In lite containers without PaddlePaddle, these imports will fail
    # This is expected and handled gracefully
    PADDLEPADDLE_AVAILABLE = False
    # Use print since logger is not yet defined
    print(f"INFO: PaddlePaddle not available in this container: {e}")

# Import utilities
from tasks.utils import get_unix_timestamp, calculate_duration, format_duration, update_job_status, get_timestamp
from text_based_processor import TextBasedProcessor

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

logger = get_task_logger(__name__)

# Initialize PPStructureV3 pipeline globally to avoid reloading models
pipeline = None
pipeline_initialization_attempted = False

# Initialize TextBasedProcessor for Gemini integration
text_processor = TextBasedProcessor()

def ensure_pipeline_initialized():
    """
    Ensure PPStructure pipeline is initialized once per DOCUMENT worker process.
    This avoids reloading models for every document/chunk, but ONLY loads in document workers.
    Other workers (API, maintenance) will never load these heavy models.
    """
    global pipeline, pipeline_initialization_attempted
    
    if pipeline is not None:
        return pipeline
    
    # Remove the permanent failure logic - allow retries for bulk processing
    # if pipeline_initialization_attempted:
    #     # If we already tried and failed, don't keep trying
    #     if pipeline is None:
    #         raise RuntimeError("PPStructure pipeline initialization failed previously")
    #     return pipeline
    
    try:
        pipeline_initialization_attempted = True
        
        # Check if PaddlePaddle is available first
        if not PADDLEPADDLE_AVAILABLE:
            logger.info("🚫 PaddlePaddle not available - this is a lite container")
            return None
        
        # Smart worker detection - only load models when actually needed
        # Check if we're being called from a PPStructure task or document processing context
        import inspect
        
        # Look at the call stack to see if we're in a document processing task
        is_document_worker = False
        frame = inspect.currentframe()
        try:
            while frame:
                frame_info = inspect.getframeinfo(frame)
                filename = frame_info.filename
                function_name = frame.f_code.co_name
                
                # Check if we're being called from document processing functions
                if ('ppstructure' in filename.lower() or 
                    function_name in ['process_document_with_ppstructure', 'warmup_ppstructure']):
                    is_document_worker = True
                    break
                    
                frame = frame.f_back
        finally:
            del frame
        
        # Also check environment variables for worker queue assignment
        worker_queues = os.environ.get('CELERY_WORKER_QUEUES', '').lower()
        if ('chunk_queue' in worker_queues or 'documents' in worker_queues or 
            'ppstructure' in worker_queues):
            is_document_worker = True
        
        if is_document_worker:
            logger.info("Initializing PPStructure pipeline...")
        else:
            # This is likely an API worker or other non-document worker
            logger.info("Non-document worker - PPStructure models not loaded")
            # We still mark as attempted to avoid repeated checks
            return None
        
        # # DONOT TOUCH THIS - Initialize PPStructureV3 - this loads all models once
        # pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")


        # Import PaddleX and initialize pipeline
        from paddlex import create_pipeline
        
        pipeline = create_pipeline(
            pipeline="PP-StructureV3-lite.yaml",
            # device="gpu:0",
            # device="cpu",
            # use_hpip=True
        )
        logger.info("PPStructure pipeline initialized successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize PPStructure pipeline: {str(e)}")
        # Don't permanently mark as failed - allow retries for bulk processing
        pipeline_initialization_attempted = False  # Reset flag to allow retry
        
        # Force cleanup on failed initialization
        try:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
        except Exception:
            pass
        
        raise RuntimeError(f"PPStructure initialization failed: {str(e)}")

# Define colors for different region types
REGION_COLORS = {
    "text": (0, 0, 255),       # Red
    "title": (0, 255, 0),      # Green
    "paragraph_title": (255, 0, 0),  # Blue
    "list": (255, 255, 0),     # Cyan
    "table": (255, 0, 255),    # Magenta
    "figure": (0, 255, 255),   # Yellow
    "header": (128, 0, 128),   # Purple
    "footer": (128, 128, 0),   # Teal
    "unknown": (192, 192, 192) # Gray
}

# def initialize_pipeline():
#     """Initialize PPStructure pipeline with simple approach"""
#     global pipeline
#     try:
#         logger.info("Initializing PPStructureV3 pipeline...")
        
#         # Initialize PPStructureV3 - simple approach
#         pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
        
#         logger.info("PPStructureV3 pipeline initialized successfully")
#         return True
#     except Exception as e:
#         logger.error(f"Failed to initialize PPStructureV3 pipeline: {str(e)}")
#         return False

@shared_task(name='tasks.warmup_ppstructure')
def warmup_ppstructure():
    """
    Warmup task to initialize PPStructure models and prepare the pipeline.
    This should be called when workers start to preload models.
    
    Returns:
        dict: Warmup status and timing information
    """
    try:
        logger.info("🔥 Starting PPStructure warmup...")
        warmup_start = get_unix_timestamp()
        
        # Initialize the pipeline (this will cache it globally)
        pipeline_instance = ensure_pipeline_initialized()
        
        # Test with a small dummy image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        
        logger.info("🧪 Testing pipeline with dummy image...")
        test_result = pipeline_instance.predict(input=[test_image])
        
        warmup_end = get_unix_timestamp()
        warmup_duration = calculate_duration(warmup_start, warmup_end)
        
        logger.info(f"PPStructure warmup completed in {warmup_duration['formatted']}")
        
        return {
            "status": "success",
            "warmup_time": warmup_duration,
            "pipeline_ready": True,
            "pipeline_cached": True,
            "test_result_received": test_result is not None,
            "timestamp": get_timestamp()
        }
        
    except Exception as e:
        logger.error(f"❌ PPStructure warmup failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "pipeline_ready": False,
            "pipeline_cached": False,
            "timestamp": get_timestamp()
        }

def process_output_for_json(output):
    """Process the output from PPStructureV3 to make it JSON serializable"""
    if isinstance(output, list):
        return [process_output_for_json(item) for item in output]
    elif isinstance(output, dict):
        return {k: process_output_for_json(v) for k, v in output.items()}
    elif isinstance(output, np.ndarray):
        return output.tolist()
    elif hasattr(output, '__dict__'):
        # Convert custom objects to dictionaries
        result = {}
        for key, value in output.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = process_output_for_json(value)
        return result
    else:
        return output

def extract_layout_regions(ocr_results):
    """Extract layout regions from OCR results"""
    regions = []
    
    # Extract from layout detection results
    if 'layout_det_res' in ocr_results and 'boxes' in ocr_results['layout_det_res']:
        for box in ocr_results['layout_det_res']['boxes']:
            region = {
                'type': box.get('label', 'unknown'),
                'bbox': [int(c) for c in box.get('coordinate', [0, 0, 0, 0])],
                'score': box.get('score', 0)
            }
            regions.append(region)
    
    # If no regions found, try parsing_res_list
    if not regions and 'parsing_res_list' in ocr_results:
        for item in ocr_results['parsing_res_list']:
            region = {
                'type': item.get('block_label', 'unknown'),
                'bbox': item.get('block_bbox', [0, 0, 0, 0]),
                'content': item.get('block_content', '')
            }
            regions.append(region)
    
    return regions

def extract_image_from_region(image_path, bbox, output_path):
    """Extract a region from an image and save it as a separate image file"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        height, width = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Skip invalid regions
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox coordinates: {bbox}")
            return None
        
        # Extract region
        region_img = img[y1:y2, x1:x2]
        
        # Save region image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, region_img)
        
        # Return image bytes for Gemini processing
        _, img_bytes = cv2.imencode('.png', region_img)
        return img_bytes.tobytes()
        
    except Exception as e:
        logger.error(f"Error extracting image region: {str(e)}")
        return None

def draw_bounding_boxes(image_path, regions, output_path):
    """Draw bounding boxes and labels on the image"""
    try:
        # Validate input path
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return None
            
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # If no regions, just copy the original image
        if not regions:
            logger.info(f"No regions to draw for {image_path}, copying original image")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
            return output_path
            
        # Convert to RGB for PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (IOError, OSError):
            try:
                # Try alternative font paths
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
            except (IOError, OSError):
                font = ImageFont.load_default()
        
        # Draw boxes for each region
        for i, region in enumerate(regions):
            region_type = region.get('type', 'unknown')
            bbox = region.get('bbox')
            
            if bbox and len(bbox) >= 4:
                try:
                    # Ensure bbox coordinates are integers and within image bounds
                    height, width = img.shape[:2]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                    
                    # Clamp coordinates to image bounds
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    color = REGION_COLORS.get(region_type, REGION_COLORS['unknown'])
                    
                    # Convert color from BGR (OpenCV) to RGB (PIL)
                    color_rgb = (color[2], color[1], color[0])
                    
                    # Draw rectangle
                    draw.rectangle(
                        [(x1, y1), (x2, y2)], 
                        outline=color_rgb, 
                        width=2
                    )
                    
                    # Add label
                    label_text = f"{i+1}: {region_type}"
                    label_y = max(0, y1 - 25)  # Ensure label is visible
                    draw.text(
                        (x1, label_y), 
                        label_text, 
                        fill=color_rgb, 
                        font=font
                    )
                except Exception as e:
                    logger.warning(f"Error drawing box {i}: {str(e)}")
                    continue
        
        # Convert back to OpenCV format
        img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the result
        success = cv2.imwrite(output_path, img_result)
        if not success:
            logger.error(f"Failed to save image to: {output_path}")
            return None
            
        logger.info(f"Successfully created bounding box visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in draw_bounding_boxes: {str(e)}")
        return None



def get_pipeline():
    """
    Get the globally cached PPStructure pipeline instance.
    Models are loaded once per DOCUMENT worker process and reused for all documents/chunks.
    Returns None if called from non-document workers (API, maintenance) to save memory.
    """
    global pipeline
    
    if pipeline is not None:
        logger.debug("📋 Using cached PPStructure pipeline (models already loaded)")
        return pipeline
    
    # Initialize if not already done (only in document workers)
    pipeline_instance = ensure_pipeline_initialized()
    
    if pipeline_instance is None:
        logger.warning("⚠️ PPStructure pipeline not available - not in document worker context")
        raise RuntimeError("PPStructure pipeline only available in document workers")
    
    return pipeline_instance

def update_active_processes_worker(increment=1):
    """
    Update the count of active document processes from worker containers.
    
    Args:
        increment: +1 when starting a process, -1 when completing
    """
    try:
        import redis
        redis_client = redis.Redis.from_url(
            os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        # Get current count
        current_count = int(redis_client.get('active_document_processes') or 0)
        new_count = max(0, current_count + increment)
        
        # Update Redis
        redis_client.set('active_document_processes', new_count, ex=300)  # 5 minute expiry
        
        logger.info(f"📊 Worker updated active document processes: {new_count} (changed by {increment})")
        
    except Exception as e:
        logger.warning(f"Could not update Redis process count from worker: {str(e)}")

@shared_task(name='tasks.process_document_with_ppstructure')
def process_document_with_ppstructure(job_id, file_path, file_name, generate_summary=True, actual_start_page=1, 
                                     enable_visualizations=False, enable_table_extraction=True, 
                                     enable_figure_extraction=True, enable_chart_extraction=True, fast_mode=False,
                                     parallel_extraction=True, max_extraction_workers=8):
    """
    Process a document using PPStructure with optimized settings.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        generate_summary: Whether to generate summary (False for chunks, True for whole documents)
        actual_start_page: The actual starting page number (for continuous numbering across chunks)
        enable_visualizations: Whether to create visualization images (disabled by default for speed)
        enable_table_extraction: Whether to extract tables with Gemini AI
        enable_figure_extraction: Whether to extract figures with Gemini AI
        enable_chart_extraction: Whether to extract charts with Gemini AI
        fast_mode: If True, disables all optional features for maximum speed
        parallel_extraction: Whether to process tables/figures/charts in parallel (default: True)
        max_extraction_workers: Maximum number of parallel workers for extraction (default: 8)
        
    Returns:
        dict: Processing results including layout analysis, OCR, and extracted information
    """
    try:
        logger.info(f"🚀 Starting document processing in high-throughput worker container for job {job_id}")
        
        # Update active process count (increment on start)
        update_active_processes_worker(1)
        
        # Apply performance optimizations
        if fast_mode:
            enable_visualizations = False
            enable_table_extraction = False
            enable_figure_extraction = False
            enable_chart_extraction = False
            parallel_extraction = False  # Disable parallel processing in fast mode
            logger.info(f"🚀 Fast mode enabled - all optimizations applied for job {job_id}")
        
        # Log optimization settings
        opts = []
        if not enable_visualizations:
            opts.append("no visualizations")
        if not enable_table_extraction:
            opts.append("no tables")
        if not enable_figure_extraction:
            opts.append("no figures")
        if not enable_chart_extraction:
            opts.append("no charts")
        if parallel_extraction:
            opts.append(f"parallel extraction ({max_extraction_workers} workers)")
        
        if opts:
            logger.info(f"⚡ Performance optimizations: {', '.join(opts)} (high-throughput 6-container setup)")
        
        # Log container information for high-throughput setup
        container_id = os.environ.get('CONTAINER_ID', 'unknown')
        worker_queues = os.environ.get('CELERY_WORKER_QUEUES', 'unknown')
        logger.info(f"🏭 Container: {container_id} | Queue: {worker_queues} | Architecture: 6-container high-throughput")
        
        # Start overall timing
        overall_start_time = get_unix_timestamp()
        
        # Detect if this is a chunk based on filename
        is_chunk = "chunk_" in file_name
        chunk_id = None
        chunk_page_range = None
        if is_chunk:
            # Extract chunk_id from filename
            chunk_parts = file_name.split("_")
            if len(chunk_parts) >= 2:
                chunk_id = f"{chunk_parts[0]}_{chunk_parts[1]}"  # e.g., "chunk_0001"
                
                # Determine page range for chunk logging
                if "0000" in chunk_id:
                    chunk_page_range = "0-4"
                elif "0001" in chunk_id:
                    chunk_page_range = "5-9"
                else:
                    chunk_page_range = "unknown"
                
                # Log chunk processing start
                logger.info(f"🔄 Starting chunk processing for {chunk_id}, page range: {chunk_page_range}")
        
        # Create result directories
        result_dir = os.path.join("results", job_id)
        
        if is_chunk and chunk_id:
            # For chunks, save directly to main job folder structure
            images_dir = os.path.join(result_dir, "images")
            vis_dir = os.path.join(result_dir, "visualizations")
            tables_dir = os.path.join(result_dir, "tables")
            figures_dir = os.path.join(result_dir, "figures")
        else:
            # For single documents, use the standard structure
            images_dir = os.path.join(result_dir, "images")
            vis_dir = os.path.join(result_dir, "visualizations")
            tables_dir = os.path.join(result_dir, "tables")
            figures_dir = os.path.join(result_dir, "figures")
        
        # Create all result directories
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Process based on file type
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 📄 PPSTRUCTURE ANALYSIS LOGGING
        logger.info(f"📄 [PPSTRUCTURE] Document analysis:")
        logger.info(f"   📝 File: {file_name}")
        logger.info(f"   📄 Type: {file_ext}")
        logger.info(f"   🆔 Job ID: {job_id}")
        logger.info(f"   📦 Is chunk: {is_chunk}")
        if is_chunk and chunk_id:
            logger.info(f"   🆔 Chunk ID: {chunk_id}")
        logger.info(f"   📄 Starting page: {actual_start_page}")
        
        if file_ext == '.pdf':
            # Convert PDF to images
            logger.info(f"📄 [PPSTRUCTURE] Converting PDF to images...")
            images = convert_from_bytes(
                        open(file_path, "rb").read(),
                        dpi=100,
                        fmt='jpeg'
                    )
            
            total_pages = len(images)
            logger.info(f"📊 [PPSTRUCTURE] PDF converted to {total_pages} page images")
            
            # Save all page images with correct page numbering
            image_paths = []
            for i, image in enumerate(images):
                actual_page_num = actual_start_page + i  # Use actual page number for file naming
                image_path = os.path.join(images_dir, f"page_{actual_page_num}.jpg")
                image.save(image_path)
                image_paths.append(image_path)
                logger.info(f"   📄 Page {actual_page_num}: Saved as {os.path.basename(image_path)}")
        else:
            # Single image file - use actual page number
            logger.info(f"🖼️  [PPSTRUCTURE] Processing single image file")
            image_path = os.path.join(images_dir, f"page_{actual_start_page}.jpg")
            shutil.copy2(file_path, image_path)
            image_paths = [image_path]
            logger.info(f"   📄 Single page: Saved as {os.path.basename(image_path)}")
        
        # Validate image paths exist
        valid_image_paths = []
        for img_path in image_paths:
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                valid_image_paths.append(img_path)
                logger.info(f"   ✅ Valid image: {os.path.basename(img_path)} ({os.path.getsize(img_path):,} bytes)")
            else:
                logger.error(f"   ❌ Invalid image: {img_path}")
        
        if not valid_image_paths:
            raise ValueError("No valid images found for processing")
        
        image_paths = valid_image_paths
        
        # 🚀 PAGE PROCESSING PREPARATION LOGGING
        logger.info(f"🚀 [PPSTRUCTURE] Preparing page processing:")
        logger.info(f"   📊 Total valid pages: {len(image_paths)}")
        logger.info(f"   ⚡ Processing method: Sequential page-by-page")
        logger.info(f"   🧠 Pipeline: PPStructure with cached models")
        
        # Process each page
        all_ocr_text = []
        table_extractions = []
        figure_extractions = []
        chart_extractions = []
        
        # Global list to collect all extraction tasks for parallel processing
        all_extraction_tasks = []
        

        
        try:
            # 🔄 DOCUMENT-LEVEL PIPELINE INITIALIZATION
            # Reinitialize the entire pipeline once per document (chunk) before processing first page
            logger.info(f"🔄 [DOCUMENT] Initializing fresh pipeline for document processing")
            logger.info(f"   📦 Processing: {file_name}")
            logger.info(f"   🆔 Job ID: {job_id}")
            
            # Clear any existing pipeline to ensure fresh start
            global pipeline, pipeline_initialization_attempted
            if pipeline is not None:
                try:
                    del pipeline
                    pipeline = None
                    logger.info(f"   🧹 Cleared existing pipeline instance")
                except:
                    pass
            
            # Force garbage collection and CUDA cleanup for fresh start
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info(f"   💾 CUDA memory cleared for fresh document processing")
            except ImportError:
                pass
            
            # Initialize fresh pipeline for this document
            pipeline_instance = ensure_pipeline_initialized()
            logger.info(f"   ✅ Fresh pipeline initialized for document processing")
            
            # Validate all image paths as strings
            validated_paths = []
            for img_path in image_paths:
                if not isinstance(img_path, str):
                    logger.error(f"Image path must be string, got {type(img_path)}: {img_path}")
                    continue
                    
                if not os.path.exists(img_path):
                    logger.error(f"Image file does not exist: {img_path}")
                    continue
                    
                if os.path.getsize(img_path) == 0:
                    logger.error(f"Image file is empty: {img_path}")
                    continue
                    
                validated_paths.append(img_path)
            
            if not validated_paths:
                raise ValueError("No valid image paths found for processing")
            
            # Convert to absolute paths for better compatibility
            absolute_validated_paths = [os.path.abspath(str(path)) for path in validated_paths]
            
            # Process pages individually with the same pipeline instance (no mid-processing resets)
            all_outputs = []
            
            logger.info(f"🏭 [PPSTRUCTURE] Starting sequential page processing for {len(absolute_validated_paths)} pages")
            logger.info(f"   🧠 Pipeline: Single instance maintained throughout entire document")
                
            for i, img_path in enumerate(absolute_validated_paths):
                page_num = i + 1
                actual_page_num = actual_start_page + i
                
                # 📄 INDIVIDUAL PAGE LOGGING - START
                logger.info(f"📄 [PAGE-{page_num:02d}] Starting processing:")
                logger.info(f"   🆔 Page number: {actual_page_num}")
                logger.info(f"   📁 Image: {os.path.basename(img_path)}")
                logger.info(f"   📊 Progress: {page_num}/{len(absolute_validated_paths)} pages")
                
                # NO MORE 2-PAGE RESETS - Use same pipeline instance throughout document
                # The pipeline was initialized once at document start and stays alive
                
                try:
                    # Start timing for the entire page
                    page_total_start = get_unix_timestamp()
                    
                    # Light pre-processing CUDA synchronization (no cache clearing during processing)
                    cuda_start = get_unix_timestamp()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            # NO empty_cache() during processing to maintain performance
                    except ImportError:
                        pass
                    cuda_time = calculate_duration(cuda_start, get_unix_timestamp())
                    
                    # Start timing for PPStructure prediction
                    prediction_start = get_unix_timestamp()
                    
                    logger.info(f"   🧠 [PAGE-{page_num:02d}] Running PPStructure inference...")
                    single_output_raw = pipeline_instance.predict(input=[img_path])
                    
                    prediction_end = get_unix_timestamp()
                    prediction_time = calculate_duration(prediction_start, prediction_end)
                    
                    # Time the output conversion
                    conversion_start = get_unix_timestamp()
                    
                    # Convert generator to list if needed
                    if hasattr(single_output_raw, '__iter__') and not isinstance(single_output_raw, (list, tuple)):
                        single_output = list(single_output_raw)
                    else:
                        single_output = single_output_raw
                    
                    conversion_time = calculate_duration(conversion_start, get_unix_timestamp())
                    
                    # Calculate total page time
                    page_total_end = get_unix_timestamp()
                    total_time = calculate_duration(page_total_start, page_total_end)
                    
                    # ✅ INDIVIDUAL PAGE LOGGING - SUCCESS
                    if single_output and len(single_output) > 0:
                        all_outputs.append(single_output[0])
                        logger.info(f"✅ [PAGE-{page_num:02d}] Processing completed successfully:")
                        logger.info(f"   ⏱️  Total time: {total_time['formatted']}")
                        logger.info(f"   🧠 Prediction: {prediction_time['formatted']}")
                        logger.info(f"   🔄 Conversion: {conversion_time['formatted']}")
                        logger.info(f"   💾 CUDA ops: {cuda_time['formatted']}")
                        logger.info(f"   🆔 Page: {actual_page_num}")
                    else:
                        logger.warning(f"⚠️  [PAGE-{page_num:02d}] Empty results returned:")
                        logger.warning(f"   ⏱️  Time: {total_time['formatted']}")
                        logger.warning(f"   🆔 Page: {actual_page_num}")
                        all_outputs.append(None)
                    
                    # Light memory cleanup after each page (keep pipeline alive)
                    del single_output_raw, single_output
                    # NO gc.collect() during processing to avoid disrupting pipeline
                    
                except Exception as page_error:
                    page_total_end = get_unix_timestamp()
                    total_time = calculate_duration(page_total_start, page_total_end)
                    
                    # ❌ INDIVIDUAL PAGE LOGGING - ERROR
                    logger.error(f"❌ [PAGE-{page_num:02d}] Processing failed:")
                    logger.error(f"   ⚠️  Error: {str(page_error)}")
                    logger.error(f"   ⏱️  Time before failure: {total_time['formatted']}")
                    logger.error(f"   🆔 Page: {actual_page_num}")
                    logger.error(f"   📁 Image: {os.path.basename(img_path)}")
                    
                    all_outputs.append(None)
                    # NO gc.collect() during processing to avoid disrupting pipeline
                    continue
            
            # 🎯 PAGE PROCESSING SUMMARY
            successful_pages = len([o for o in all_outputs if o is not None])
            failed_pages = len(all_outputs) - successful_pages
            
            logger.info(f"🎯 [PPSTRUCTURE] Page processing completed:")
            logger.info(f"   ✅ Successful: {successful_pages}/{len(all_outputs)} pages")
            logger.info(f"   ❌ Failed: {failed_pages}/{len(all_outputs)} pages")
            logger.info(f"   📈 Success rate: {(successful_pages/len(all_outputs))*100:.1f}%")
            logger.info(f"   ⚡ Processing method: Single pipeline per document (chunk-based)")
            logger.info(f"   🧠 Pipeline lifecycle: Document-level initialization and cleanup")
            
        except Exception as e:
            logger.error(f"❌ [PPSTRUCTURE] Critical error in page processing pipeline: {str(e)}")
            all_outputs = [None] * len(image_paths)
        
        # Process results for each page
        processed_outputs = []
        
        for page_index, (image_path, output) in enumerate(zip(image_paths, all_outputs)):
            # Start timing for post-processing
            post_processing_start = get_unix_timestamp()
            
            # Calculate actual page number and page index within chunk
            actual_page_num = actual_start_page + page_index
            page_within_chunk = page_index + 1
            
            # Note: Progress updates now happen at batch level, not per page since we process all pages at once
            
            try:
                # Time the output processing
                output_processing_start = get_unix_timestamp()
                
                if output is not None:
                    # Process the output
                    if hasattr(output, 'save_to_json'):
                        # Use built-in method if available
                        result_dir_for_page = os.path.join(result_dir, "page_results")
                        os.makedirs(result_dir_for_page, exist_ok=True)
                        output.save_to_json(save_path=result_dir_for_page)
                        
                        # Try to find and load the saved JSON file
                        json_files = [f for f in os.listdir(result_dir_for_page) if f.endswith('_res.json')]
                        if json_files:
                            json_path = os.path.join(result_dir_for_page, json_files[0])
                            with open(json_path, 'r', encoding='utf-8') as f:
                                output_dict = json.load(f)
                            # Clean up the temp file
                            os.remove(json_path)
                        else:
                            # Fallback: convert output object to dict manually
                            output_dict = process_output_for_json(output)
                    else:
                        # Direct conversion if no save method
                        output_dict = process_output_for_json(output)
                        
                    # Validate that we got actual results
                    if not output_dict or (
                        'layout_det_res' in output_dict and 
                        not output_dict['layout_det_res'].get('boxes') and
                        'overall_ocr_res' in output_dict and
                        not output_dict['overall_ocr_res'].get('rec_texts')
                    ):
                        logger.warning(f"PPStructure returned empty results for page {actual_page_num}")
                        output_dict = {
                            'layout_det_res': {'boxes': []},
                            'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
                            'parsing_res_list': [],
                            'error': 'Empty results from PPStructure'
                        }
                else:
                    # Create empty structure for None results
                    logger.warning(f"No output for page {actual_page_num}, creating empty structure")
                    output_dict = {
                        'layout_det_res': {'boxes': []},
                        'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
                        'parsing_res_list': [],
                        'error': 'No output from PPStructure'
                    }
                
            except Exception as e:
                logger.error(f"Error post-processing page {actual_page_num}: {str(e)}")
                output_dict = {
                    'layout_det_res': {'boxes': []},
                    'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
                    'parsing_res_list': [],
                    'error': f'Post-processing failed: {str(e)}'
                }
            
            # Calculate output processing time
            output_processing_end = get_unix_timestamp()
            output_processing_time = calculate_duration(output_processing_start, output_processing_end)
            
            # Add timing information
            output_dict['output_processing_time'] = output_processing_time
            output_dict['processing_method'] = 'optimized_batch'
            output_dict['page_number'] = actual_page_num
            
            processed_outputs.append(output_dict)
            
            # Extract layout regions
            regions = extract_layout_regions(output_dict)
            
            # Create visualization with bounding boxes (conditional)
            if enable_visualizations:
                vis_path = os.path.join(vis_dir, f"page_{actual_page_num}_layout.jpg")
                layout_vis_result = draw_bounding_boxes(image_path, regions, vis_path)
                if layout_vis_result is None:
                    logger.warning(f"Failed to create layout visualization for page {actual_page_num}")
                else:
                    logger.info(f"Successfully created bounding box visualization: {vis_path}")
            # Visualization skipped if disabled
            
            # Extract OCR text
            ocr_text = []
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                ocr_text = output_dict['overall_ocr_res']['rec_texts']
                # Add to combined text with page marker using actual page number
                all_ocr_text.append(f"--- PAGE {actual_page_num} ---")
                all_ocr_text.extend(ocr_text)
            
            # Collect all extraction tasks for parallel processing
            extraction_tasks = []
            for j, region in enumerate(regions):
                region_type = region.get('type', '').lower()
                bbox = region.get('bbox')
                
                if region_type == 'table' and bbox and enable_table_extraction:
                    table_img_path = os.path.join(tables_dir, f"page_{actual_page_num}_table_{j+1}.png")
                    extraction_tasks.append({
                        'type': 'table',
                        'page': actual_page_num,
                        'region_id': j+1,
                        'image_path': image_path,
                        'bbox': bbox,
                        'output_path': table_img_path,
                        'context': f"Table from page {actual_page_num}"
                    })
                    
                elif region_type == 'figure' and bbox and enable_figure_extraction:
                    figure_img_path = os.path.join(figures_dir, f"page_{actual_page_num}_figure_{j+1}.png")
                    extraction_tasks.append({
                        'type': 'figure',
                        'page': actual_page_num,
                        'region_id': j+1,
                        'image_path': image_path,
                        'bbox': bbox,
                        'output_path': figure_img_path,
                        'context': f"Figure from page {actual_page_num}"
                    })
                
                elif region_type in ['chart', 'graph'] and bbox and enable_chart_extraction:
                    chart_img_path = os.path.join(figures_dir, f"page_{actual_page_num}_chart_{j+1}.png")
                    extraction_tasks.append({
                        'type': 'chart',
                        'page': actual_page_num,
                        'region_id': j+1,
                        'image_path': image_path,
                        'bbox': bbox,
                        'output_path': chart_img_path,
                        'context': f"Chart from page {actual_page_num}"
                    })
                
                # Skip debug messages for disabled extractions
            
            # Add tasks to the global list for batch parallel processing
            all_extraction_tasks.extend(extraction_tasks)
            
            # Calculate total post-processing time
            post_processing_end = get_unix_timestamp()
            post_processing_time = calculate_duration(post_processing_start, post_processing_end)
            
            # Log detailed timing for this page
            logger.info(f"📊 Page {actual_page_num} post-processing: {post_processing_time['formatted']} (output: {output_processing_time['formatted']})")
        
        # Process all extraction tasks in parallel (CPU-bound operations)
        if all_extraction_tasks:
            logger.info(f"🔄 Processing {len(all_extraction_tasks)} extractions (tables/figures/charts)")
            parallel_start_time = get_unix_timestamp()
            
            # Use ThreadPoolExecutor for parallel CPU processing
            max_workers = min(max_extraction_workers, len(all_extraction_tasks))  # Limit concurrent API calls
            extraction_lock = Lock()
            
            # Check if parallel processing is enabled
            if not parallel_extraction:
                max_workers = 1
            
            def process_extraction_task(task):
                """Process a single extraction task"""
                try:
                    # Extract image region
                    img_bytes = extract_image_from_region(
                        task['image_path'], 
                        task['bbox'], 
                        task['output_path']
                    )
                    
                    if img_bytes:
                        # Process with Gemini based on type
                        if task['type'] == 'table':
                            extracted_text = text_processor.process_table_image(img_bytes, task['context'])
                        elif task['type'] == 'figure':
                            extracted_text = text_processor.process_figure_image(img_bytes, task['context'])
                        elif task['type'] == 'chart':
                            extracted_text = text_processor.process_chart_image(img_bytes, task['context'])
                        else:
                            return None
                        
                        # Return formatted result
                        return {
                            'type': task['type'],
                            'page': task['page'],
                            'text': f"\n--- {task['type'].upper()} FROM PAGE {task['page']} ---\n{extracted_text}\n"
                        }
                    
                except Exception as e:
                    logger.error(f"Error processing {task['type']} from page {task['page']}: {str(e)}")
                    return None
            
            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(process_extraction_task, task): task 
                    for task in all_extraction_tasks
                }
                
                # Collect results as they complete
                completed_extractions = 0
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        # Thread-safe accumulation
                        with extraction_lock:
                            if result['type'] == 'table':
                                table_extractions.append(result['text'])
                            elif result['type'] == 'figure':
                                figure_extractions.append(result['text'])
                            elif result['type'] == 'chart':
                                chart_extractions.append(result['text'])
                        
                        completed_extractions += 1
                        if completed_extractions % 10 == 0:  # Log progress every 10 completions
                            logger.info(f"Completed {completed_extractions}/{len(all_extraction_tasks)} extractions")
            
            parallel_end_time = get_unix_timestamp()
            parallel_duration = calculate_duration(parallel_start_time, parallel_end_time)
            logger.info(f"Extraction completed in {parallel_duration['formatted']} - {completed_extractions} successful")
        
        # Combine all text for document summarization
        combined_text = "\n".join(all_ocr_text)
        
        # Add table, figure, and chart extractions
        if table_extractions:
            combined_text += "\n\n" + "\n".join(table_extractions)
        
        if figure_extractions:
            combined_text += "\n\n" + "\n".join(figure_extractions)
            
        if chart_extractions:
            combined_text += "\n\n" + "\n".join(chart_extractions)
        
        # Generate document summary using Gemini (only if requested)
        summary_result = {}
        if generate_summary and combined_text.strip():
            try:
                summary_result = text_processor.summarize_document_text(combined_text, file_name)
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                summary_result = {
                    "summary": "Summary generation failed",
                    "analysis": {"error": str(e)},
                    "usage_info": {"total_tokens": 0},
                    "estimated_cost": 0.0
                }
        elif not generate_summary:
            # For chunks, don't generate summary but provide placeholder
            summary_result = {
                "summary": None,
                "analysis": {},
                "usage_info": {"total_tokens": 0},
                "estimated_cost": 0.0
            }
        
        # Calculate performance metrics
        overall_end_time = get_unix_timestamp()
        processing_duration = calculate_duration(overall_start_time, overall_end_time)
        
        # Save results
        results_path = os.path.join(result_dir, "results.json")
        metrics_path = os.path.join(result_dir, "metrics.json")
        ppstructure_path = os.path.join(result_dir, "ppstructure_results.json")
        
        # Calculate average confidence score
        all_scores = []
        for output in processed_outputs:
            if 'overall_ocr_res' in output and 'rec_scores' in output['overall_ocr_res']:
                all_scores.extend(output['overall_ocr_res']['rec_scores'])
        
        average_confidence = sum(all_scores) / len(all_scores) if all_scores else 0
        average_confidence_formatted = f"{average_confidence:.1%}"
        
        # Save combined text to a separate file
        combined_text_path = os.path.join(result_dir, "combined_text.txt")
        with open(combined_text_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Prepare individual page results
        page_results = []
        for page_index, (image_path, output_dict) in enumerate(zip(image_paths, processed_outputs)):
            actual_page_num = actual_start_page + page_index
            
            # Get relative paths for web access
            relative_result_dir = f"results/{job_id}"
            
            # Extract OCR text for this page
            page_ocr_text = []
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                page_ocr_text = output_dict['overall_ocr_res']['rec_texts']
            
            # Extract layout regions for this page
            regions = extract_layout_regions(output_dict)
            
            page_result = {
                "page_number": actual_page_num,
                "image_path": f"{relative_result_dir}/images/page_{actual_page_num}.jpg",
                "layout_vis_path": f"{relative_result_dir}/visualizations/page_{actual_page_num}_layout.jpg",
                "regions": regions,
                "ocr_text": page_ocr_text,
                "json_path": f"{relative_result_dir}/page_results/page_{actual_page_num}_res.json" if hasattr(output_dict, 'save_to_json') else None
            }
            
            page_results.append(page_result)
        

    
        # Extract structured information from summary result
        extracted_info = {}
        if generate_summary and summary_result.get("extracted_info"):
            extracted_info = summary_result["extracted_info"]
        else:
            # Default structure for chunks
            extracted_info = {
                "key_dates": "Not available",
                "main_parties": "Not available", 
                "case_reference_numbers": "Not available",
                "full_analysis": "No summary generated for individual chunks"
            }
        
        # Prepare clean results.json
        clean_results = {
            "filename": file_name,
            "job_id": job_id,
            "processing_completed_at": datetime.datetime.now().isoformat(),
            "total_pages": len(image_paths),
            "summary": summary_result.get("summary", "Summary not available"),
            "date": summary_result.get("date", "undated"),
            "extracted_info": extracted_info,
            "combined_text_path": f"results/{job_id}/combined_text.txt",
            "combined_text": combined_text,
            "estimated_cost": summary_result.get("estimated_cost", 0.0),
            "token_usage": summary_result.get("usage_info", {}),
            "processing_time_seconds": processing_duration.get("seconds", 0),
            "average_confidence_formatted": average_confidence_formatted,
            "processing_method": "optimized_sequential"
        }
        
        # Save detailed PPStructure results to separate file
        ppstructure_results = {
            "job_id": job_id,
            "filename": file_name,
            "processing_completed_at": datetime.datetime.now().isoformat(),
            "total_pages": len(image_paths),
            "page_results": page_results,
            "detailed_page_outputs": processed_outputs,
            "processing_info": {
                "pages": len(image_paths),
                "processing_method": "optimized_sequential",
                "optimizations_enabled": {
                    "visualizations_disabled": not enable_visualizations,
                    "table_extraction": enable_table_extraction,
                    "figure_extraction": enable_figure_extraction,
                    "fast_mode": fast_mode
                },
                "images_dir": images_dir,
                "vis_dir": vis_dir,
                "tables_dir": tables_dir,
                "figures_dir": figures_dir
            }
        }
        
        # Prepare detailed metrics in separate file
        performance_metrics = {
            "job_id": job_id,
            "filename": file_name,
            "processing_start_time": overall_start_time,
            "processing_end_time": overall_end_time,
            "total_processing_time": processing_duration,
            "performance": {
                "total_pages": len(image_paths),
                "processing_time": processing_duration,
                "optimized_processing": True,
                "visualizations_disabled": not enable_visualizations
            },
            "confidence_metrics": {
                "average_confidence": average_confidence,
                "average_confidence_formatted": average_confidence_formatted,
                "total_text_elements": len(all_scores),
                "confidence_scores": all_scores[:100] if len(all_scores) > 100 else all_scores
            },
            "cost_info": {
                "estimated_cost": summary_result.get("estimated_cost", 0.0),
                "token_usage": summary_result.get("usage_info", {})
            }
        }
        
        # Save clean results to results.json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        # Save detailed metrics to metrics.json
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=2)
        
        # Save PPStructure results to ppstructure_results.json
        with open(ppstructure_path, 'w', encoding='utf-8') as f:
            json.dump(ppstructure_results, f, ensure_ascii=False, indent=2)
        
        # Get container information for logging
        container_id = os.environ.get('CONTAINER_ID', 'unknown')
        worker_queues = os.environ.get('CELERY_WORKER_QUEUES', 'unknown')
        
        # Log chunk completion if this is a chunk
        if is_chunk and chunk_id and chunk_page_range:
            logger.info(f"✅ Chunk {chunk_id} completed with {len(image_paths)} pages processed (page range: {chunk_page_range})")
        
        logger.info(f"✅ Chunk processing completed for job {job_id} on high-throughput container: {container_id}")
        logger.info(f"📊 Container performance: Queue={worker_queues} | Setup=6-container/4-concurrency | Total capacity=24 workers")
        logger.info(f"🧠 Pipeline management: Document-level lifecycle (initialized once, disposed after completion)")
        logger.info(f"⚡ Processing optimization: No mid-document resets, maximum memory efficiency")
        
        # Update active process count (decrement on completion)
        update_active_processes_worker(-1)
        
        # 🧹 DOCUMENT-LEVEL CLEANUP (Only after entire document is processed)
        # This is where we dispose of the pipeline and do final memory cleanup
        logger.info(f"🧹 [DOCUMENT] Starting final cleanup after document completion")
        try:
            # Dispose of the pipeline instance used for this document
            if 'pipeline_instance' in locals():
                del pipeline_instance
                logger.info(f"   🗑️  Pipeline instance disposed")
            
            # Clear the global pipeline reference for this worker and reset initialization flag
            if pipeline is not None:
                del pipeline
                pipeline = None
                logger.info(f"   🗑️  Global pipeline reference cleared")
            
            # Reset initialization flag so next document can initialize fresh
            pipeline_initialization_attempted = False
            logger.info(f"   🔄 Pipeline initialization flag reset for next document")
            
            # Final garbage collection after document completion
            import gc
            gc.collect()
            logger.info(f"   ♻️  Garbage collection completed")
            
            # Final CUDA memory cleanup after document completion
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info(f"   💾 Final CUDA memory cleanup completed")
            except ImportError:
                pass
            
            logger.info(f"✅ [DOCUMENT] Final cleanup completed - worker ready for next document")
            
        except Exception as cleanup_error:
            logger.warning(f"⚠️ [DOCUMENT] Cleanup warning: {str(cleanup_error)}")
        
        # Prepare final results for chunk (not document completion)
        final_results = clean_results.copy()
        final_results["performance"] = performance_metrics["performance"]
        
        return {
            "job_id": job_id,
            "status": "CHUNK_COMPLETED",  # Indicate this is a chunk completion, not document completion
            "chunk_id": chunk_id if is_chunk and chunk_id else f"chunk_{actual_start_page}",
            "filename": file_name,  # Include the chunk filename for merge identification
            "results_path": results_path,
            "message": f"Chunk processing completed: {len(image_paths)} pages",
            "performance": final_results["performance"],
            # Include the actual extracted data for use by the merge function
            "combined_text": combined_text,
            "extracted_text": combined_text,  # Alias for compatibility
            "average_confidence_formatted": average_confidence_formatted,
            "cost_info": performance_metrics["cost_info"],
            "processing_time": processing_duration,
            "total_pages": len(image_paths),
            "summary": summary_result.get("summary", None),
            "extracted_info": extracted_info
        }
        
    except Exception as e:
        logger.error(f"Error in PPStructure processing: {str(e)}")
        
        # Update active process count (decrement on failure)
        update_active_processes_worker(-1)
        
        # 🧹 DOCUMENT-LEVEL CLEANUP ON ERROR (Ensure cleanup even on failure)
        logger.info(f"🧹 [DOCUMENT] Starting cleanup after processing error")
        try:
            # Dispose of the pipeline instance used for this document
            if 'pipeline_instance' in locals():
                del pipeline_instance
                logger.info(f"   🗑️  Pipeline instance disposed (after error)")
            
            # Clear the global pipeline reference for this worker and reset initialization flag
            if pipeline is not None:
                del pipeline
                pipeline = None
                logger.info(f"   🗑️  Global pipeline reference cleared (after error)")
            
            # Reset initialization flag so next document can initialize fresh
            pipeline_initialization_attempted = False
            logger.info(f"   🔄 Pipeline initialization flag reset for next document")
            
            # Final garbage collection after error
            import gc
            gc.collect()
            logger.info(f"   ♻️  Garbage collection completed (after error)")
            
            # Final CUDA memory cleanup after error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info(f"   💾 CUDA memory cleanup completed (after error)")
            except ImportError:
                pass
            
            logger.info(f"✅ [DOCUMENT] Error cleanup completed - worker ready for next document")
            
        except Exception as cleanup_error:
            logger.warning(f"⚠️ [DOCUMENT] Error cleanup warning: {str(cleanup_error)}")
        
        # For chunk failures, update progress but don't mark entire job as FAILED
        # The merge task will determine final status based on chunk results
        try:
            import redis
            from tasks.utils import update_job_status, get_timestamp
            
            redis_client = redis.Redis.from_url(
                os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
            )
            
            update_job_status(redis_client, job_id, {
                'status': 'PROCESSING',  # Keep as PROCESSING - merge task will handle final failure
                'chunk_error': str(e),
                'chunk_failed': file_name,  # Track which chunk failed
                'message': f'Chunk {file_name} failed: {str(e)}',
                'updated_at': get_timestamp(),
                'container_id': os.environ.get('CONTAINER_ID', 'unknown')
            })
            
            logger.info(f"📊 Updated chunk failure in Redis (job still PROCESSING)")
            
        except Exception as update_error:
            logger.error(f"Error updating chunk failure status: {str(update_error)}")
        
        # Return chunk failure result instead of raising exception
        return {
            "job_id": job_id,
            "status": "CHUNK_FAILED",
            "chunk_id": file_name,
            "filename": file_name,
            "error": str(e),
            "message": f"Chunk processing failed: {str(e)}",
            "combined_text": "",
            "extracted_text": "",
            "total_pages": 0,
            "processing_time": {"seconds": 0},
            "summary": None,
            "extracted_info": {}
        }

@shared_task(name='tasks.merge_and_summarize_chunks')
def merge_and_summarize_chunks(chunk_results, job_id):
    """
    Merge chunk processing results and generate final summary.
    
    This task is called by Celery chord after all chunk processing tasks complete.
    It combines the text from all chunks and generates a final document summary.
    
    Args:
        chunk_results: List of results from individual chunk processing tasks (passed automatically by chord)
        job_id: Unique job identifier (passed as argument)
        
    Returns:
        dict: Final processing results with merged content and summary
    """
    try:
        logger.info(f"🔄 Starting merge and summarize for job {job_id}")
        
        # Initialize Redis client for status updates
        redis_client = redis.Redis.from_url(
            os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        # Log the number of chunk results received
        logger.info(f"📦 Received {len(chunk_results)} chunk results for job {job_id}")
        
        # Validate chunk results
        if not chunk_results:
            raise ValueError("No chunk results provided for merging")
        
        # Separate successful and failed chunks
        successful_chunks = []
        failed_chunks = []
        
        for i, chunk_result in enumerate(chunk_results):
            if chunk_result and isinstance(chunk_result, dict):
                if chunk_result.get('status') == 'CHUNK_FAILED':
                    failed_chunks.append(chunk_result)
                    logger.warning(f"   ❌ Chunk {i}: FAILED - {chunk_result.get('error', 'Unknown error')}")
                else:
                    successful_chunks.append((i, chunk_result))
                    logger.info(f"   ✅ Chunk {i}: SUCCESS - {chunk_result.get('filename', 'unknown')}")
            else:
                failed_chunks.append({"error": "Invalid chunk result", "chunk_index": i})
                logger.warning(f"   ⚠️ Chunk {i}: invalid result format")
        
        # Check if we have any successful chunks
        if not successful_chunks:
            raise ValueError("All chunks failed - no successful processing to merge")
        
        if failed_chunks:
            logger.warning(f"⚠️ {len(failed_chunks)} chunks failed, proceeding with {len(successful_chunks)} successful chunks")
        
        # Extract original document filename from the first successful chunk
        original_filename = "unknown_document"
        for _, chunk_result in successful_chunks:
            chunk_filename = chunk_result.get('filename', '')
            if chunk_filename.startswith('chunk_'):
                # Extract original filename: "chunk_0000_original.pdf" -> "original.pdf"
                parts = chunk_filename.split('_', 2)  # Split on first 2 underscores
                if len(parts) >= 3:
                    original_filename = parts[2]  # Everything after "chunk_XXXX_"
                    break
            else:
                original_filename = chunk_filename
                break
        
        logger.info(f"📄 Original document: {original_filename}")
        
        # CRITICAL: Sort chunks by chunk ID to ensure correct page order
        chunk_data = []
        for original_index, chunk_result in successful_chunks:
            filename = chunk_result.get('filename', '')
            # Extract chunk ID from filename (e.g., "chunk_0000" from "chunk_0000_file.pdf")
            chunk_id = "9999"  # Default high value if parsing fails
            if filename.startswith('chunk_'):
                try:
                    chunk_parts = filename.split('_')
                    if len(chunk_parts) >= 2:
                        chunk_id = chunk_parts[1]  # e.g., "0000" from "chunk_0000"
                except Exception as e:
                    logger.warning(f"Failed to parse chunk ID from {filename}: {e}")
            
            chunk_data.append({
                'chunk_id': chunk_id,
                'original_index': original_index,
                'result': chunk_result,
                'filename': filename
            })
            logger.info(f"   📦 Chunk {original_index}: ID={chunk_id}, filename={filename}")
        
        # Sort chunks by chunk ID to ensure correct page order
        chunk_data.sort(key=lambda x: x['chunk_id'])
        chunk_order_info = [f"{c['chunk_id']}({c['original_index']})" for c in chunk_data]
        logger.info(f"📋 Sorted chunks by ID: {chunk_order_info}")
        
        # Extract and combine text from all chunks in correct order
        all_combined_text = []
        total_pages = 0
        all_extracted_info = []
        total_processing_time = 0
        total_cost = 0.0
        
        for chunk_info in chunk_data:
            chunk_result = chunk_info['result']
            chunk_id = chunk_info['chunk_id']
            original_index = chunk_info['original_index']
            
            # Extract combined text from chunk
            chunk_text = chunk_result.get('combined_text', '') or chunk_result.get('extracted_text', '')
            if chunk_text:
                all_combined_text.append(chunk_text)
                logger.info(f"   ✅ Chunk {chunk_id} (orig index {original_index}): {len(chunk_text)} characters extracted")
            else:
                logger.warning(f"   ⚠️ Chunk {chunk_id} (orig index {original_index}): no text content found")
            
            # Accumulate metadata
            total_pages += chunk_result.get('total_pages', 0)
            if chunk_result.get('processing_time', {}).get('seconds'):
                total_processing_time += chunk_result.get('processing_time', {}).get('seconds', 0)
            if chunk_result.get('cost_info', {}).get('estimated_cost'):
                total_cost += chunk_result.get('cost_info', {}).get('estimated_cost', 0.0)
            
            # Collect extracted info (though chunks don't generate summaries)
            chunk_info_data = chunk_result.get('extracted_info', {})
            if chunk_info_data:
                all_extracted_info.append(chunk_info_data)
        
        # Combine all text into one document body IN CORRECT PAGE ORDER
        combined_text = '\n\n'.join(all_combined_text)
        logger.info(f"📄 Combined text length: {len(combined_text)} characters from {len(all_combined_text)} chunks")
        logger.info(f"📋 Chunks merged in correct page order: {[c['chunk_id'] for c in chunk_data]}")
        
        if not combined_text.strip():
            raise ValueError("No text content could be extracted from any chunks")
        
        # Generate final summary using text_processor
        logger.info(f"🧠 Generating final summary for job {job_id}")
        
        # Generate summary using TextBasedProcessor
        summary_result = text_processor.summarize_document_text(combined_text, original_filename)
        
        # Log completion
        final_summary = summary_result.get('summary', 'Summary not available')
        logger.info(f"🧠 Final summary generated for job {job_id}, total length: {len(final_summary)}")
        
        # Prepare final results directories
        result_dir = os.path.join("results", job_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save combined text to file
        combined_text_path = os.path.join(result_dir, "combined_text.txt")
        with open(combined_text_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Calculate average confidence (if available from chunks)
        average_confidence = 0.0
        confidence_count = 0
        for chunk_info in chunk_data:
            chunk_result = chunk_info['result']
            chunk_conf = chunk_result.get('average_confidence_formatted', '')
            if chunk_conf and chunk_conf != 'N/A':
                try:
                    # Extract percentage value
                    conf_value = float(chunk_conf.strip('%')) / 100.0
                    average_confidence += conf_value
                    confidence_count += 1
                except ValueError:
                    pass
        
        if confidence_count > 0:
            average_confidence = average_confidence / confidence_count
            average_confidence_formatted = f"{average_confidence:.1%}"
        else:
            average_confidence_formatted = "N/A"
        
        # Prepare clean results with ORIGINAL document filename
        clean_results = {
            "filename": original_filename,  # Use original filename, not chunk filename
            "job_id": job_id,
            "processing_completed_at": datetime.datetime.now().isoformat(),
            "total_pages": total_pages,
            "summary": final_summary,
            "date": summary_result.get("date", "undated"),
            "extracted_info": summary_result.get("extracted_info", {}),
            "combined_text_path": f"results/{job_id}/combined_text.txt",
            "combined_text": combined_text,
            "estimated_cost": summary_result.get("estimated_cost", total_cost),
            "token_usage": summary_result.get("token_usage", {}),
            "processing_time_seconds": total_processing_time,
            "average_confidence_formatted": average_confidence_formatted,
            "processing_method": "high_throughput_chunked",
            "num_chunks_processed": len(chunk_results),
            "num_chunks_successful": len(successful_chunks),
            "num_chunks_failed": len(failed_chunks),
            "chunk_order": [c['chunk_id'] for c in chunk_data]  # Record the order used
        }
        
        # Save results to files
        results_path = os.path.join(result_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        # Save metrics
        metrics_path = os.path.join(result_dir, "metrics.json")
        performance_metrics = {
            "job_id": job_id,
            "filename": original_filename,  # Use original filename
            "processing_end_time": get_unix_timestamp(),
            "total_processing_time": {"seconds": total_processing_time},
            "performance": {
                "total_pages": total_pages,
                "processing_time": {"seconds": total_processing_time},
                "high_throughput_processing": True,
                "num_chunks": len(chunk_results),
                "num_chunks_successful": len(successful_chunks),
                "num_chunks_failed": len(failed_chunks),
                "chunk_order": [c['chunk_id'] for c in chunk_data]
            },
            "confidence_metrics": {
                "average_confidence": average_confidence,
                "average_confidence_formatted": average_confidence_formatted
            },
            "cost_info": {
                "estimated_cost": summary_result.get("estimated_cost", total_cost),
                "token_usage": summary_result.get("token_usage", {})
            }
        }
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=2)
        
        # CRITICAL: Only NOW mark the job as COMPLETED (all chunks processed and merged)
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'message': f'Document "{original_filename}" processed successfully with {total_pages} pages using high-throughput chunking',
            'progress': 100,
            'results_path': results_path,
            'combined_text_path': combined_text_path,
            'total_pages': total_pages,
            'processing_completed_at': datetime.datetime.now().isoformat(),
            'average_confidence_formatted': average_confidence_formatted,
            'estimated_cost': summary_result.get("estimated_cost", total_cost),
            'processing_time_seconds': total_processing_time,
            'processing_method': 'high_throughput_chunked',
            'num_chunks_processed': len(chunk_results),
            'num_chunks_successful': len(successful_chunks),
            'num_chunks_failed': len(failed_chunks),
            'filename': original_filename,  # Store original filename
            'updated_at': get_timestamp()
        })
        
        logger.info(f"✅ Merge and summarize completed for job {job_id}")
        logger.info(f"📄 Final document: {original_filename} ({total_pages} pages)")
        
        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "filename": original_filename,  # Return original filename
            "results_path": results_path,
            "message": f'Document "{original_filename}" processed successfully with {total_pages} pages using high-throughput chunking',
            "total_pages": total_pages,
            "num_chunks_processed": len(chunk_results),
            "num_chunks_successful": len(successful_chunks),
            "num_chunks_failed": len(failed_chunks),
            "processing_method": "high_throughput_chunked",
            "summary": final_summary,
            "estimated_cost": summary_result.get("estimated_cost", total_cost)
        }
        
    except Exception as e:
        logger.error(f"Error in merge and summarize: {str(e)}")
        
        # CRITICAL: Only mark as FAILED here in the merge task
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Document merge and summarize failed: {str(e)}',
            'progress': 0,
            'updated_at': get_timestamp()
        })
        
        raise 
