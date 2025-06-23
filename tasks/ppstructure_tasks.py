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


        # Import PaddleX using the correct method
        from paddlex import create_pipeline
        
        # Initialize PP-StructureV3 with high-performance inference
        # Initialize with consistent GPU device usage to prevent segmentation faults
        try:
            logger.info("🚀 Attempting to initialize PPStructure with GPU device consistency...")
            pipeline = create_pipeline(
                pipeline="PP-StructureV3.yaml",
                use_hpip=True
            )
            logger.info("✅ High-performance GPU pipeline initialized successfully!")
            
        except Exception as hpi_error:
            logger.warning(f"⚠️ High-performance inference failed: {hpi_error}")
            logger.info("🔄 Falling back to standard GPU pipeline...")
            
            try:
                # Final fallback to standard pipeline with explicit GPU
                pipeline = create_pipeline(
                    pipeline="PP-StructureV3.yaml",
                )
                logger.info("✅ Standard GPU pipeline initialized successfully!")
                
            except Exception as std_error:
                logger.error(f"❌ All GPU attempts failed, trying CPU fallback: {std_error}")
                try:
                    # Last resort: CPU pipeline
                    pipeline = create_pipeline(
                        pipeline="PP-StructureV3",
                        device="cpu"
                    )
                    logger.info("✅ CPU pipeline initialized as last resort!")
                except Exception as cpu_error:
                    logger.error(f"❌ All pipeline initialization attempts failed: {cpu_error}")
                    raise RuntimeError(f"Complete pipeline initialization failure: {cpu_error}")

        
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
                                     enable_figure_extraction=True, enable_chart_extraction=True, fast_mode=False,
                                     processing_mode="auto"):
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
        processing_mode: Processing mode - "auto", "batch", "individual"
                        - "auto": Use batch for <=10 pages, sub-batch for >10 pages
                        - "batch": Process all pages in optimal batches
                        - "individual": Process one page at a time for debugging
        
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
            
            # Determine processing mode and log it
            logger.info(f"🔧 Processing mode: {processing_mode} for {len(absolute_validated_paths)} pages")
            
            # Process based on mode
            if processing_mode == "individual":
                logger.info(f"🔍 INDIVIDUAL PROCESSING MODE - Processing {len(absolute_validated_paths)} pages one by one for debugging")
                all_outputs = []
                
                for i, img_path in enumerate(absolute_validated_paths):
                    page_num = i + 1
                    logger.info(f"🔬 Processing page {page_num} individually ({page_num}/{len(absolute_validated_paths)}): {os.path.basename(img_path)}")
                    
                    # Pipeline reset every 10 pages to prevent memory corruption
                    if page_num > 1 and (page_num - 1) % 10 == 0:
                        logger.info(f"🔄 Resetting pipeline after page {page_num - 1} to prevent memory corruption")
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
                                'message': f'🔬 {chunk_id} - Individual processing: page {page_num} ({i+1}/{len(absolute_validated_paths)})',
                                'updated_at': get_timestamp(),
                                'chunk_id': chunk_id,
                                'current_page': page_num,
                                'total_pages_in_chunk': len(absolute_validated_paths),
                                'processing_stage': f'individual_page_{page_num}'
                            })
                        
                        # Analyze ALL pages to understand what makes page 19 different
                        logger.info(f"🔍 ANALYZING PAGE {page_num}:")
                        try:
                            import cv2
                            import numpy as np
                            from PIL import Image
                            
                            # Load and analyze the image
                            img_cv = cv2.imread(img_path)
                            img_pil = Image.open(img_path)
                            
                            if img_cv is not None and img_pil is not None:
                                height, width = img_cv.shape[:2]
                                file_size = os.path.getsize(img_path)
                                
                                # Calculate image statistics
                                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                                mean_intensity = np.mean(gray)
                                std_intensity = np.std(gray)
                                white_pixels = np.sum(gray > 240)
                                total_pixels = gray.size
                                white_ratio = white_pixels / total_pixels
                                
                                # Additional analysis for potential problematic elements
                                # Check for circular/curved elements that might cause issues
                                edges = cv2.Canny(gray, 50, 150)
                                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                # Count circular/complex contours
                                circular_contours = 0
                                large_contours = 0
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area > 100:  # Only consider meaningful contours
                                        large_contours += 1
                                        # Check if contour is approximately circular
                                        perimeter = cv2.arcLength(contour, True)
                                        if perimeter > 0:
                                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                                            if circularity > 0.7:  # Pretty circular
                                                circular_contours += 1
                                
                                logger.info(f"🔍 Page {page_num} Analysis:")
                                logger.info(f"   📐 Dimensions: {width}x{height}")
                                logger.info(f"   📁 File size: {file_size} bytes")
                                logger.info(f"   🎨 PIL mode: {img_pil.mode}")
                                logger.info(f"   🔢 Mean intensity: {mean_intensity:.2f}")
                                logger.info(f"   📊 Std intensity: {std_intensity:.2f}")
                                logger.info(f"   ⚪ White ratio: {white_ratio:.3f} ({white_ratio*100:.1f}%)")
                                logger.info(f"   🔵 Circular contours: {circular_contours}")
                                logger.info(f"   📦 Large contours: {large_contours}")
                                
                                # Memory cleanup before processing pages 15+ to prevent accumulation
                                if page_num >= 15:
                                    logger.info(f"🧹 Performing aggressive memory cleanup before page {page_num}")
                                    import gc
                                    import torch
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                    
                                    # Force garbage collection multiple times for stubborn memory
                                    for _ in range(3):
                                        gc.collect()
                                    
                                    logger.info(f"🧹 Memory cleanup completed for page {page_num}")
                                
                                # General warnings for all pages
                                if white_ratio > 0.95:
                                    logger.warning(f"⚠️ Page {page_num} has very high white space ratio: {white_ratio:.3f}")
                                if std_intensity < 10:
                                    logger.warning(f"⚠️ Page {page_num} has very low contrast: {std_intensity:.2f}")
                                if circular_contours > 5:
                                    logger.warning(f"⚠️ Page {page_num} has many circular elements: {circular_contours}")
                                if file_size > 1000000:  # > 1MB
                                    logger.warning(f"⚠️ Page {page_num} is very large: {file_size} bytes")
                            else:
                                logger.error(f"❌ Failed to load page {page_num} image for analysis")
                                
                        except Exception as analysis_error:
                            logger.error(f"❌ Failed to analyze page {page_num}: {analysis_error}")
                        
                        page_start_time = get_unix_timestamp()
                        
                        # Add detailed logging for which models are being used and when they fail
                        logger.info(f"🔧 Page {page_num} - Starting PPStructure pipeline with models:")
                        
                        # Try to extract model information from pipeline
                        try:
                            if hasattr(pipeline_instance, 'layout_model'):
                                logger.info(f"   📐 Layout Detection Model: {getattr(pipeline_instance.layout_model, 'model_name', 'Unknown')}")
                            if hasattr(pipeline_instance, 'table_model'):
                                logger.info(f"   📊 Table Detection Model: {getattr(pipeline_instance.table_model, 'model_name', 'Unknown')}")
                            if hasattr(pipeline_instance, 'ocr_model'):
                                logger.info(f"   🔤 OCR Model: {getattr(pipeline_instance.ocr_model, 'model_name', 'Unknown')}")
                        except Exception as model_info_error:
                            logger.warning(f"   ⚠️ Could not extract model info: {model_info_error}")
                        
                        logger.info(f"🚀 Page {page_num} - Calling pipeline.predict()...")
                        
                        try:
                            single_output_raw = pipeline_instance.predict(input=[img_path])
                            logger.info(f"✅ Page {page_num} - pipeline.predict() completed successfully")
                        except Exception as predict_error:
                            logger.error(f"💥 Page {page_num} - pipeline.predict() FAILED: {predict_error}")
                            logger.error(f"💥 Error type: {type(predict_error).__name__}")
                            logger.error(f"💥 This suggests the issue is in: {type(predict_error).__module__ if hasattr(predict_error, '__module__') else 'unknown module'}")
                            raise predict_error
                        
                        page_end_time = get_unix_timestamp()
                        
                        # Convert generator to list if needed
                        if hasattr(single_output_raw, '__iter__') and not isinstance(single_output_raw, (list, tuple)):
                            single_output = list(single_output_raw)
                        else:
                            single_output = single_output_raw
                        
                        if single_output and len(single_output) > 0:
                            all_outputs.append(single_output[0])
                            processing_time = calculate_duration(page_start_time, page_end_time)
                            logger.info(f"✅ Page {page_num} completed successfully in {processing_time['formatted']}")
                        else:
                            logger.warning(f"⚠️ Page {page_num} returned empty results")
                            all_outputs.append(None)
                        
                        # Clear memory after each page
                        del single_output_raw, single_output
                        import gc
                        gc.collect()
                        
                        # Additional memory monitoring and cleanup for pages 10+
                        if page_num >= 10:
                            # Monitor memory usage and trigger aggressive cleanup if needed
                            try:
                                import psutil
                                process = psutil.Process()
                                memory_info = process.memory_info()
                                memory_mb = memory_info.rss / 1024 / 1024
                                
                                logger.info(f"📊 Memory usage after page {page_num}: {memory_mb:.1f} MB")
                                
                                # If memory usage is high, trigger aggressive cleanup
                                if memory_mb > 2000:  # > 2GB
                                    logger.warning(f"⚠️ High memory usage detected ({memory_mb:.1f} MB) - triggering aggressive cleanup")
                                    
                                    # Force multiple garbage collection cycles
                                    for _ in range(5):
                                        gc.collect()
                                    
                                    # Clear CUDA cache if available
                                    try:
                                        import torch
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                            torch.cuda.synchronize()
                                    except ImportError:
                                        pass
                                    
                                    # Check memory again
                                    memory_info_after = process.memory_info()
                                    memory_mb_after = memory_info_after.rss / 1024 / 1024
                                    logger.info(f"📊 Memory usage after cleanup: {memory_mb_after:.1f} MB (freed {memory_mb - memory_mb_after:.1f} MB)")
                                    
                                    # If still high memory, consider switching to safer mode
                                    if memory_mb_after > 3000:  # > 3GB
                                        logger.error(f"💥 CRITICAL: Memory usage still very high ({memory_mb_after:.1f} MB) after page {page_num}")
                                        logger.error(f"💥 This may lead to segfaults on subsequent pages")
                                        
                                        # For remaining pages, we might want to use process isolation
                                        remaining_pages = len(absolute_validated_paths) - page_num
                                        if remaining_pages > 0:
                                            logger.warning(f"🛡️ Switching to process isolation for remaining {remaining_pages} pages to prevent crashes")
                                            
                                            # Process remaining pages in isolation
                                            for remaining_idx in range(page_num, len(absolute_validated_paths)):
                                                remaining_img_path = absolute_validated_paths[remaining_idx]
                                                remaining_page_num = remaining_idx + 1
                                                
                                                logger.info(f"🛡️ Processing page {remaining_page_num} in isolation due to high memory usage")
                                                isolated_result = _process_page_in_isolation(remaining_img_path, remaining_page_num, job_id)
                                                all_outputs.append(isolated_result)
                                            
                                            # Break out of the main loop since we've processed all remaining pages
                                            break
                                            
                            except ImportError:
                                # psutil not available, skip memory monitoring
                                pass
                            except Exception as memory_error:
                                logger.warning(f"⚠️ Memory monitoring failed: {memory_error}")
                        
                    except Exception as page_error:
                        logger.error(f"❌ CRITICAL: Page {page_num} failed with error: {str(page_error)}")
                        logger.error(f"❌ Page {page_num} path: {img_path}")
                        logger.error(f"❌ Page {page_num} exists: {os.path.exists(img_path)}")
                        logger.error(f"❌ Page {page_num} size: {os.path.getsize(img_path) if os.path.exists(img_path) else 'N/A'}")
                        all_outputs.append(None)
                        import gc
                        gc.collect()
                        
                        # For debugging, we might want to stop here
                        if processing_mode == "force_individual":
                            logger.error(f"💥 STOPPING at problematic page {page_num} for debugging")
                            break
                        
                        continue
                
                logger.info(f"🔍 Individual processing completed: {len([o for o in all_outputs if o is not None])}/{len(all_outputs)} pages successful")
                
            else:
                # Adaptive batch processing with size limits to prevent memory issues
                # Use smaller batches for documents with 15+ pages to prevent segfaults
                if len(absolute_validated_paths) >= 15:
                    batch_size = 5  # Smaller batches for large documents
                    logger.info(f"Large document detected ({len(absolute_validated_paths)} pages) - using smaller batch size of {batch_size}")
                else:
                    batch_size = min(10, len(absolute_validated_paths))  # Process max 10 pages at once
                
                if len(absolute_validated_paths) <= batch_size:
                    # Small batch - process all at once
                    logger.info(f"🚀 Processing all {len(absolute_validated_paths)} images at once with PPStructure")
                    
                    try:
                        # Process entire batch in one call
                        all_outputs_raw = pipeline_instance.predict(input=absolute_validated_paths)
                        
                        # Convert generator to list if needed
                        if hasattr(all_outputs_raw, '__iter__') and not isinstance(all_outputs_raw, (list, tuple)):
                            all_outputs = list(all_outputs_raw)
                        else:
                            all_outputs = all_outputs_raw
                        
                        if all_outputs:
                            logger.info(f"✅ Successfully processed all {len(all_outputs)} images in single call")
                            
                            # Update progress after successful batch processing
                            if is_chunk and chunk_id:
                                update_job_status(redis_client, job_id, {
                                    'message': f'✅ {chunk_id} - PPStructure completed, now extracting text and processing results',
                                    'updated_at': get_timestamp(),
                                    'chunk_id': chunk_id,
                                    'total_pages_in_chunk': len(absolute_validated_paths),
                                    'processing_stage': 'post_processing_results'
                                })
                        else:
                            logger.warning("⚠️ Empty output from PPStructure batch processing")
                            all_outputs = [None] * len(absolute_validated_paths)
                            
                    except Exception as batch_error:
                        raise batch_error  # Let it fall through to the fallback
                else:
                    # Large batch - process in smaller sub-batches
                    logger.info(f"🚀 Processing {len(absolute_validated_paths)} images in sub-batches of {batch_size} with PPStructure")
                    all_outputs = []
                    
                    try:
                        for i in range(0, len(absolute_validated_paths), batch_size):
                            end_idx = min(i + batch_size, len(absolute_validated_paths))
                            sub_batch = absolute_validated_paths[i:end_idx]
                            
                            # Update progress for sub-batch
                            if is_chunk and chunk_id:
                                batch_num = (i // batch_size) + 1
                                total_batches = (len(absolute_validated_paths) + batch_size - 1) // batch_size
                                update_job_status(redis_client, job_id, {
                                    'message': f'🚀 {chunk_id} - Processing sub-batch {batch_num}/{total_batches} ({len(sub_batch)} pages)',
                                    'updated_at': get_timestamp(),
                                    'chunk_id': chunk_id,
                                    'current_batch': batch_num,
                                    'total_batches': total_batches,
                                    'processing_stage': f'sub_batch_{batch_num}'
                                })
                            
                            logger.info(f"Processing sub-batch {i//batch_size + 1}: pages {i+1}-{end_idx}")
                            
                            # Process sub-batch
                            sub_outputs_raw = pipeline_instance.predict(input=sub_batch)
                            
                            # Convert generator to list if needed
                            if hasattr(sub_outputs_raw, '__iter__') and not isinstance(sub_outputs_raw, (list, tuple)):
                                sub_outputs = list(sub_outputs_raw)
                            else:
                                sub_outputs = sub_outputs_raw
                            
                            if sub_outputs:
                                all_outputs.extend(sub_outputs)
                                logger.info(f"✅ Sub-batch {i//batch_size + 1} completed: {len(sub_outputs)} images processed")
                            else:
                                # Add None placeholders for failed sub-batch
                                all_outputs.extend([None] * len(sub_batch))
                                logger.warning(f"⚠️ Sub-batch {i//batch_size + 1} returned empty results")
                            
                            # Clear memory after each sub-batch
                            del sub_outputs_raw, sub_outputs
                            import gc
                            gc.collect()
                            
                            # For large documents, add extra memory cleanup and stabilization
                            if len(absolute_validated_paths) >= 15:
                                import time
                                # Small delay to allow GPU memory to stabilize between sub-batches
                                time.sleep(0.5)
                                logger.debug(f"Memory stabilization completed for sub-batch {i//batch_size + 1}")
                        
                        # Final progress update
                        if is_chunk and chunk_id:
                            update_job_status(redis_client, job_id, {
                                'message': f'✅ {chunk_id} - All sub-batches completed, now extracting text and processing results',
                                'updated_at': get_timestamp(),
                                'chunk_id': chunk_id,
                                'total_pages_in_chunk': len(absolute_validated_paths),
                                'processing_stage': 'post_processing_results'
                            })
                        
                        logger.info(f"✅ Successfully processed all {len(all_outputs)} images using sub-batch method")
                        
                    except Exception as batch_error:
                        logger.warning(f"⚠️ Batch processing failed ({str(batch_error)}), falling back to individual processing...")
                        
                        # Update progress for fallback processing
                        if is_chunk and chunk_id:
                            update_job_status(redis_client, job_id, {
                                'message': f'⚠️ {chunk_id} - Batch failed, processing pages individually...',
                                'updated_at': get_timestamp(),
                                'chunk_id': chunk_id,
                                'total_pages_in_chunk': len(absolute_validated_paths),
                                'processing_stage': 'individual_fallback_processing'
                            })
                        
                        # Fallback: Process images individually with enhanced safety measures
                        all_outputs = []
                        consecutive_failures = 0
                        
                        for i, img_path in enumerate(absolute_validated_paths):
                            page_num = i + 1
                            
                            try:
                                # Update progress for individual page processing
                                if is_chunk and chunk_id:
                                    update_job_status(redis_client, job_id, {
                                        'message': f'📄 {chunk_id} - Individual processing: page {page_num}/{len(absolute_validated_paths)}',
                                        'updated_at': get_timestamp(),
                                        'chunk_id': chunk_id,
                                        'current_page_in_chunk': page_num,
                                        'total_pages_in_chunk': len(absolute_validated_paths),
                                        'processing_stage': f'individual_page_{page_num}'
                                    })
                                
                                # Use process isolation for problematic pages or after multiple failures
                                use_isolation = (
                                    consecutive_failures >= 2 or  # After 2 consecutive failures
                                    page_num >= 15  # For pages 15+ which are more prone to issues
                                )
                                
                                if use_isolation:
                                    logger.info(f"🛡️ Using process isolation for page {page_num} (consecutive_failures: {consecutive_failures})")
                                    result = _process_page_in_isolation(img_path, page_num, job_id)
                                    all_outputs.append(result)
                                    
                                    if result is not None:
                                        consecutive_failures = 0  # Reset failure counter on success
                                    else:
                                        consecutive_failures += 1
                                        
                                else:
                                    # Normal processing
                                    single_output_raw = pipeline_instance.predict(input=[img_path])
                                    
                                    # Convert generator to list if needed
                                    if hasattr(single_output_raw, '__iter__') and not isinstance(single_output_raw, (list, tuple)):
                                        single_output = list(single_output_raw)
                                    else:
                                        single_output = single_output_raw
                                    
                                    if single_output and len(single_output) > 0:
                                        all_outputs.append(single_output[0])
                                        consecutive_failures = 0  # Reset failure counter on success
                                    else:
                                        logger.warning(f"Empty output for page {page_num}")
                                        all_outputs.append(None)
                                        consecutive_failures += 1
                                    
                                    # Clear memory after each image
                                    del single_output_raw, single_output
                                    import gc
                                    gc.collect()
                                
                            except Exception as img_error:
                                logger.error(f"Error processing page {page_num}: {str(img_error)}")
                                consecutive_failures += 1
                                
                                # If we've had multiple failures, try process isolation as last resort
                                if consecutive_failures >= 3 and not use_isolation:
                                    logger.warning(f"🛡️ Multiple failures detected, trying process isolation for page {page_num}")
                                    try:
                                        result = _process_page_in_isolation(img_path, page_num, job_id)
                                        all_outputs.append(result)
                                        if result is not None:
                                            consecutive_failures = 0
                                    except Exception as isolation_error:
                                        logger.error(f"💥 Process isolation also failed for page {page_num}: {isolation_error}")
                                        all_outputs.append(None)
                                else:
                                    all_outputs.append(None)
                                
                                import gc
                                gc.collect()
                                continue
            
            # Update image_paths to match validated paths
            image_paths = validated_paths
            
            if not all_outputs:
                logger.warning("PPStructure returned empty results for all pages")
                all_outputs = [None] * len(image_paths)
            else:
                successful_count = sum(1 for output in all_outputs if output is not None)
                logger.info(f"PPStructure processing completed: {successful_count}/{len(all_outputs)} images processed successfully")
            
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

def _process_page_in_isolation(img_path, page_num, job_id):
    """
    Process a single page in a separate process to prevent segfaults from crashing the main worker.
    This is specifically designed to handle problematic pages like page 19.
    
    Args:
        img_path: Path to the image file
        page_num: Page number being processed
        job_id: Job identifier for logging
        
    Returns:
        dict: Processing result or None if failed
    """
    import multiprocessing
    import signal
    import sys
    
    def _isolated_page_processor(img_path, result_queue, error_queue):
        """
        The actual processing function that runs in isolation.
        """
        try:
            # Set up signal handlers to catch segfaults
            def signal_handler(signum, frame):
                error_queue.put(f"Signal {signum} received in isolated process")
                sys.exit(1)
            
            signal.signal(signal.SIGSEGV, signal_handler)
            signal.signal(signal.SIGFPE, signal_handler)
            signal.signal(signal.SIGABRT, signal_handler)
            
            # Initialize a fresh pipeline in this process
            logger.info(f"🔧 Initializing fresh pipeline in isolated process for page {page_num}")
            
            # Import here to avoid issues with multiprocessing
            from paddlex import create_pipeline
            
            # Create a minimal pipeline configuration
            isolated_pipeline = create_pipeline(
                pipeline="PP-StructureV3",
                # device="gpu:0",  # Let it auto-detect
                use_hpip=False  # Disable high-performance inference to reduce complexity
            )
            
            logger.info(f"🚀 Processing page {page_num} in isolated process...")
            
            # Process the single page
            result = isolated_pipeline.predict(input=[img_path])
            
            # Convert result to serializable format
            if hasattr(result, '__iter__') and not isinstance(result, (list, tuple)):
                result = list(result)
            
            if result and len(result) > 0:
                # Put the result in the queue
                result_queue.put(result[0])
                logger.info(f"✅ Page {page_num} processed successfully in isolated process")
            else:
                result_queue.put(None)
                logger.warning(f"⚠️ Page {page_num} returned empty result in isolated process")
                
        except Exception as e:
            error_msg = f"Isolated processing failed for page {page_num}: {str(e)}"
            logger.error(error_msg)
            error_queue.put(error_msg)
            result_queue.put(None)
    
    # Create queues for communication
    result_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()
    
    # Create and start the isolated process
    process = multiprocessing.Process(
        target=_isolated_page_processor,
        args=(img_path, result_queue, error_queue)
    )
    
    logger.info(f"🚀 Starting isolated process for page {page_num}")
    process.start()
    
    # Wait for the process to complete with timeout
    timeout_seconds = 120  # 2 minutes timeout
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        logger.error(f"💥 Isolated process for page {page_num} timed out after {timeout_seconds}s")
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()
        return None
    
    # Check if process completed successfully
    if process.exitcode == 0:
        # Get the result
        try:
            if not result_queue.empty():
                result = result_queue.get_nowait()
                logger.info(f"✅ Successfully retrieved result from isolated process for page {page_num}")
                return result
            else:
                logger.warning(f"⚠️ No result available from isolated process for page {page_num}")
                return None
        except Exception as e:
            logger.error(f"💥 Failed to retrieve result from isolated process for page {page_num}: {e}")
            return None
    else:
        # Process failed
        error_msg = "Unknown error"
        try:
            if not error_queue.empty():
                error_msg = error_queue.get_nowait()
        except:
            pass
        
        logger.error(f"💥 Isolated process for page {page_num} failed with exit code {process.exitcode}: {error_msg}")
        return None 