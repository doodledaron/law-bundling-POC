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
    
    if pipeline_initialization_attempted:
        # If we already tried and failed, don't keep trying
        if pipeline is None:
            raise RuntimeError("PPStructure pipeline initialization failed previously")
        return pipeline
    
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
                    'document_tasks' in filename.lower() or
                    function_name in ['process_document_with_ppstructure', 'warmup_ppstructure', 'process_document']):
                    is_document_worker = True
                    break
                    
                frame = frame.f_back
        finally:
            del frame
        
        # Also check environment variables for worker queue assignment
        worker_queues = os.environ.get('CELERY_WORKER_QUEUES', '').lower()
        if 'documents' in worker_queues:
            is_document_worker = True
        
        # Log worker type detection
        worker_name = os.environ.get('CELERY_WORKER_NAME', 'unknown')
        logger.info(f"🔍 Worker detection: name='{worker_name}', queues='{worker_queues}', is_document_worker={is_document_worker}")
        
        if is_document_worker:
            logger.info("🔥 Document worker context detected - Initializing PPStructure pipeline...")
        else:
            # This is likely an API worker or other non-document worker
            logger.info("🚫 Non-document worker context - PPStructure models NOT loaded (saves memory)")
            # We still mark as attempted to avoid repeated checks
            return None
        
        # # DONOT TOUCH THIS - Initialize PPStructureV3 - this loads all models once
        # pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")


        # Import PaddleX and initialize pipeline
        from paddlex import create_pipeline
        
        logger.info("🚀 Initializing PPStructure with stable configuration...")
        pipeline = create_pipeline(
            pipeline="PP-StructureV3.yaml",
            device="gpu:0",
            use_hpip=True
        )
        logger.info("✅ PPStructure pipeline initialized successfully!")

        
        logger.info("✅ PPStructure pipeline initialized successfully and cached globally in document worker")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize PPStructure pipeline: {str(e)}")
        pipeline = None  # Reset to None on failure
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
        
        logger.info(f"✅ PPStructure warmup completed in {warmup_duration['formatted']} - Models cached for reuse")
        
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

@shared_task(name='tasks.process_document_with_ppstructure')
def process_document_with_ppstructure(job_id, file_path, file_name, generate_summary=True, actual_start_page=1, 
                                     enable_visualizations=False, enable_table_extraction=True, 
                                     enable_figure_extraction=True, enable_chart_extraction=True, fast_mode=False):
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
        
    Returns:
        dict: Processing results including layout analysis, OCR, and extracted information
    """
    try:
        # Apply performance optimizations
        if fast_mode:
            enable_visualizations = False
            enable_table_extraction = False
            enable_figure_extraction = False
            enable_chart_extraction = False
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
        
        if opts:
            logger.info(f"⚡ Performance optimizations: {', '.join(opts)} (estimated 30% faster)")
        
        # Start overall timing
        overall_start_time = get_unix_timestamp()
        
        # Detect if this is a chunk based on filename
        is_chunk = "chunk_" in file_name
        chunk_id = None
        if is_chunk:
            # Extract chunk_id from filename
            chunk_parts = file_name.split("_")
            if len(chunk_parts) >= 2:
                chunk_id = f"{chunk_parts[0]}_{chunk_parts[1]}"  # e.g., "chunk_0001"
        
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
        
        if file_ext == '.pdf':
            # Convert PDF to images
            images = convert_from_bytes(open(file_path, "rb").read())
            
            # Save all page images with correct page numbering
            image_paths = []
            for i, image in enumerate(images):
                actual_page_num = actual_start_page + i  # Use actual page number for file naming
                image_path = os.path.join(images_dir, f"page_{actual_page_num}.jpg")
                image.save(image_path)
                image_paths.append(image_path)
        else:
            # Single image file - use actual page number
            image_path = os.path.join(images_dir, f"page_{actual_start_page}.jpg")
            shutil.copy2(file_path, image_path)
            image_paths = [image_path]
        
        # Validate image paths exist
        valid_image_paths = []
        for img_path in image_paths:
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                valid_image_paths.append(img_path)
            else:
                logger.error(f"Invalid image file: {img_path}")
        
        if not valid_image_paths:
            raise ValueError("No valid images found for processing")
        
        image_paths = valid_image_paths
        
        # Process each page
        all_ocr_text = []
        table_extractions = []
        figure_extractions = []
        chart_extractions = []
        

        
        try:
            # Process images with PPStructureV3
            logger.info(f"Processing {len(image_paths)} images with PPStructure")
            
            # Get pipeline instance (uses globally cached models)
            pipeline_instance = get_pipeline()
            
            # Log whether we're using cached models or loading for first time
            if pipeline is not None:
                logger.info(f"🔄 Using cached PPStructure models (no reload needed)")
            else:
                logger.info(f"🔥 Loading PPStructure models for first time in this worker")
        
            
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
            
            # Update progress before PPStructure processing
            if is_chunk and chunk_id:
                update_job_status(redis_client, job_id, {
                    'message': f'🚀 Processing {chunk_id} - Running PPStructure on all {len(absolute_validated_paths)} pages in single batch',
                    'updated_at': get_timestamp(),
                    'chunk_id': chunk_id,
                    'total_pages_in_chunk': len(absolute_validated_paths),
                    'processing_stage': 'ppstructure_batch_processing'
                })
            
            # Process pages individually 
            logger.info(f"🔧 Processing {len(absolute_validated_paths)} pages individually")
            all_outputs = []
                
            for i, img_path in enumerate(absolute_validated_paths):
                page_num = i + 1
                logger.info(f"🔬 Processing page {page_num} individually ({page_num}/{len(absolute_validated_paths)}): {os.path.basename(img_path)}")
                
                # Pipeline reset every 2 pages for stability
                if page_num > 1 and (page_num - 1) % 2 == 0:
                    logger.info(f"🔄 Resetting pipeline after page {page_num - 1}")
                    try:
                        # Clear the current pipeline
                        del pipeline_instance
                        import gc
                        gc.collect()
                        
                        # Force CUDA cleanup if available
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        except ImportError:
                            pass
                        
                        # Reinitialize the pipeline
                        pipeline_instance = ensure_pipeline_initialized()
                        logger.info(f"✅ Pipeline reset completed before page {page_num}")
                        
                    except Exception as reset_error:
                        logger.error(f"⚠️ Pipeline reset failed: {reset_error}")
                        logger.info(f"🔄 Continuing with existing pipeline...")
                
                try:
                    # Update progress for individual page processing
                    if is_chunk and chunk_id:
                        update_job_status(redis_client, job_id, {
                            'message': f'🔬 {chunk_id} - Processing page {page_num} ({i+1}/{len(absolute_validated_paths)})',
                            'updated_at': get_timestamp(),
                            'chunk_id': chunk_id,
                            'current_page': page_num,
                            'total_pages_in_chunk': len(absolute_validated_paths),
                            'processing_stage': f'page_{page_num}'
                        })
                    
                    # Pre-processing CUDA synchronization for stability
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    
                    page_start_time = get_unix_timestamp()
                    logger.info(f"🚀 Processing page {page_num}...")
                    
                    single_output_raw = pipeline_instance.predict(input=[img_path])
                    logger.info(f"✅ Page {page_num} completed successfully")
                    
                    page_end_time = get_unix_timestamp()
                    
                    # Convert generator to list if needed
                    if hasattr(single_output_raw, '__iter__') and not isinstance(single_output_raw, (list, tuple)):
                        single_output = list(single_output_raw)
                    else:
                        single_output = single_output_raw
                    
                    if single_output and len(single_output) > 0:
                        all_outputs.append(single_output[0])
                        processing_time = calculate_duration(page_start_time, page_end_time)
                        logger.info(f"✅ Page {page_num} completed in {processing_time['formatted']}")
                    else:
                        logger.warning(f"⚠️ Page {page_num} returned empty results")
                        all_outputs.append(None)
                    
                    # Clear memory after each page
                    del single_output_raw, single_output
                    import gc
                    gc.collect()
                    
                except Exception as page_error:
                    logger.error(f"❌ Page {page_num} failed: {str(page_error)}")
                    all_outputs.append(None)
                    import gc
                    gc.collect()
                    continue
            
            logger.info(f"Processing completed: {len([o for o in all_outputs if o is not None])}/{len(all_outputs)} pages successful")
            
        except Exception as e:
            logger.error(f"Error in PPStructure processing: {str(e)}")
            all_outputs = [None] * len(image_paths)
        
        # Process results for each page
        processed_outputs = []
        
        for page_index, (image_path, output) in enumerate(zip(image_paths, all_outputs)):
            page_start_time = get_unix_timestamp()
            
            # Calculate actual page number and page index within chunk
            actual_page_num = actual_start_page + page_index
            page_within_chunk = page_index + 1
            
            # Note: Progress updates now happen at batch level, not per page since we process all pages at once
            
            try:
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
            
            # Add timing information
            page_end_time = get_unix_timestamp()
            output_dict['page_processing_time'] = calculate_duration(page_start_time, page_end_time)
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
            else:
                logger.debug(f"⚡ Skipped visualization for page {actual_page_num} (performance optimization)")
            
            # Extract OCR text
            ocr_text = []
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                ocr_text = output_dict['overall_ocr_res']['rec_texts']
                # Add to combined text with page marker using actual page number
                all_ocr_text.append(f"--- PAGE {actual_page_num} ---")
                all_ocr_text.extend(ocr_text)
            
            # Process tables, figures, and charts with Gemini AI (conditional) - ACCUMULATE across all pages
            for j, region in enumerate(regions):
                region_type = region.get('type', '').lower()
                bbox = region.get('bbox')
                
                if region_type == 'table' and bbox and enable_table_extraction:
                    # Extract table image and save to file
                    table_img_path = os.path.join(tables_dir, f"page_{actual_page_num}_table_{j+1}.png")
                    table_img_bytes = extract_image_from_region(image_path, bbox, table_img_path)
                    
                    if table_img_bytes:
                        # Process table with Gemini
                        context = f"Table from page {actual_page_num}"
                        table_text = text_processor.process_table_image(table_img_bytes, context)
                        
                        # Add to table extractions (ACCUMULATE, don't overwrite)
                        table_extractions.append(f"\n--- TABLE FROM PAGE {actual_page_num} ---\n{table_text}\n")
                    
                elif region_type == 'figure' and bbox and enable_figure_extraction:
                    # Extract figure image and save to file
                    figure_img_path = os.path.join(figures_dir, f"page_{actual_page_num}_figure_{j+1}.png")
                    figure_img_bytes = extract_image_from_region(image_path, bbox, figure_img_path)
                    
                    if figure_img_bytes:
                        # Process figure with Gemini
                        context = f"Figure from page {actual_page_num}"
                        figure_text = text_processor.process_figure_image(figure_img_bytes, context)
                        
                        # Add to figure extractions (ACCUMULATE, don't overwrite)
                        figure_extractions.append(f"\n--- FIGURE FROM PAGE {actual_page_num} ---\n{figure_text}\n")
                
                elif region_type in ['chart', 'graph'] and bbox and enable_chart_extraction:
                    # Extract chart image and save to file  
                    chart_img_path = os.path.join(figures_dir, f"page_{actual_page_num}_chart_{j+1}.png")
                    chart_img_bytes = extract_image_from_region(image_path, bbox, chart_img_path)
                    
                    if chart_img_bytes:
                        # Process chart with Gemini
                        context = f"Chart from page {actual_page_num}"
                        chart_text = text_processor.process_chart_image(chart_img_bytes, context)
                        
                        # Add to chart extractions (ACCUMULATE, don't overwrite)
                        chart_extractions.append(f"\n--- CHART FROM PAGE {actual_page_num} ---\n{chart_text}\n")
                
                elif region_type == 'table' and not enable_table_extraction:
                    logger.debug(f"⚡ Skipped table extraction on page {actual_page_num} (performance optimization)")
                elif region_type == 'figure' and not enable_figure_extraction:
                    logger.debug(f"⚡ Skipped figure extraction on page {actual_page_num} (performance optimization)")
                elif region_type in ['chart', 'graph'] and not enable_chart_extraction:
                    logger.debug(f"⚡ Skipped chart extraction on page {actual_page_num} (performance optimization)")
        
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
        
        logger.info(f"PPStructure processing completed for job {job_id}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Prepare final results
        final_results = clean_results.copy()
        final_results["performance"] = performance_metrics["performance"]
        
        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "results_path": results_path,
            "message": f"Document processed successfully with {len(image_paths)} pages",
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
        raise 
