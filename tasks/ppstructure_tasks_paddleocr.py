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


# Import PaddleOCR components conditionally to prevent errors in lite containers
try:
    print("Importing PaddleOCR components---------------------------------")
    import sys
    print(f"Python version: {sys.version}")
    from paddleocr import PPStructure, PaddleOCR
    print("Imported PaddleOCR components")
    from pdf2image import convert_from_bytes
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
    PADDLEPADDLE_AVAILABLE = True
    
    # Remove problematic numpy compatibility fixes that can interfere with PaddlePaddle
    # The original numpy deprecation warnings are less problematic than segfaults
    warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
    warnings.filterwarnings("ignore", message=".*np.bool.*deprecated.*")
        
except ImportError as e:
    # In lite containers without PaddlePaddle, these imports will fail
    # This is expected and handled gracefully
    PADDLEPADDLE_AVAILABLE = False
    # Use print since logger is not yet defined
    print(f"INFO: PaddlePaddle not available in this container: {e}")
    # Create dummy numpy for compatibility
    np = None

# Import utilities
from tasks.utils import get_unix_timestamp, calculate_duration, format_duration, update_job_status, get_timestamp

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

logger = get_task_logger(__name__)

# Initialize PPStructureV3 pipeline globally to avoid reloading models
pipeline = None
pipeline_initialization_attempted = False


def check_system_resources():
    """Check system resources and limits that might cause segmentation faults"""
    try:
        import resource
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        logger.info(f"💾 System Memory: {memory.total/(1024**3):.1f}GB total, {memory.available/(1024**3):.1f}GB available")
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        logger.info(f"🖥️  CPU: {cpu_count} cores")
        
        # Check stack size limit
        stack_size = resource.getrlimit(resource.RLIMIT_STACK)
        logger.info(f"📚 Stack limit: {stack_size[0]/1024/1024:.1f}MB current, {stack_size[1]/1024/1024:.1f}MB max")
        
        # Check virtual memory limit  
        vmem_limit = resource.getrlimit(resource.RLIMIT_AS)
        if vmem_limit[0] != resource.RLIM_INFINITY:
            logger.info(f"🔒 Virtual memory limit: {vmem_limit[0]/1024/1024:.1f}MB")
        else:
            logger.info("🔓 Virtual memory: unlimited")
            
        # Warn if resources are low
        if memory.available < 2 * 1024**3:  # Less than 2GB
            logger.warning("⚠️  Low available memory - this may cause segmentation faults")
        
        if stack_size[0] < 8 * 1024 * 1024:  # Less than 8MB stack
            logger.warning("⚠️  Small stack size - this may cause segmentation faults in deep neural networks")
            
    except Exception as e:
        logger.warning(f"Could not check system resources: {e}")

def ensure_pipeline_initialized():
    """
    Ensure PPStructure pipeline is initialized once per worker process.
    This avoids reloading models for every document but handles initialization safely.
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
        
        logger.info("Initializing PPStructure pipeline...")
        
        # Check system resources that might cause segfaults
        check_system_resources()
        
        # Clear any existing memory before initialization
        import gc
        gc.collect()
        
        # Try to increase stack size if possible
        try:
            import resource
            current_stack = resource.getrlimit(resource.RLIMIT_STACK)
            if current_stack[0] < 16 * 1024 * 1024:  # If less than 16MB
                new_stack = min(16 * 1024 * 1024, current_stack[1])
                resource.setrlimit(resource.RLIMIT_STACK, (new_stack, current_stack[1]))
                logger.info(f"📚 Increased stack size to {new_stack/1024/1024:.1f}MB")
        except Exception as e:
            logger.warning(f"Could not increase stack size: {e}")
        

        pipeline_initialized = False
        try:
            pipeline = PPStructure(
                # paddlex_config=config_file, 
                # device="cpu"
            )
            # pipeline = PaddleOCR(use_angle_cls=True, lang="en")
            pipeline_initialized = True
        except Exception as e:
            logger.error(f"❌ Failed: {str(e)}")
        
        if not pipeline_initialized:
            raise RuntimeError("All PPStructure configurations failed - this may be a system compatibility issue")
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
        # test_result = pipeline_instance.predict(input=[test_image])
        test_result = pipeline_instance(test_image)
        
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
    """Extract layout regions from OCR results with enhanced detection"""
    regions = []
    
    # Debug: Log the complete structure
    logger.info(f"🔍 Extracting regions from keys: {list(ocr_results.keys())}")
    
    # Method 1: Extract from layout detection results
    if 'layout_det_res' in ocr_results and 'boxes' in ocr_results['layout_det_res']:
        logger.info(f"Found layout_det_res with {len(ocr_results['layout_det_res']['boxes'])} boxes")
        for box in ocr_results['layout_det_res']['boxes']:
            region = {
                'type': box.get('label', 'unknown'),
                'bbox': [int(c) for c in box.get('coordinate', [0, 0, 0, 0])],
                'score': box.get('score', 0)
            }
            regions.append(region)
    
    # Method 2: Try parsing_res_list
    if not regions and 'parsing_res_list' in ocr_results:
        logger.info(f"Found parsing_res_list with {len(ocr_results['parsing_res_list'])} items")
        for item in ocr_results['parsing_res_list']:
            region = {
                'type': item.get('block_label', 'unknown'),
                'bbox': item.get('block_bbox', [0, 0, 0, 0]),
                'content': item.get('block_content', '')
            }
            regions.append(region)
    
    # Method 3: Create regions from OCR text blocks (fallback)
    if not regions and 'res' in ocr_results:
        logger.info(f"Creating text regions from 'res' field with {len(ocr_results['res'])} items")
        for i, text_item in enumerate(ocr_results['res']):
            if isinstance(text_item, dict):
                bbox = None
                text_content = ""
                confidence = 0.0
                
                # Try different bbox field names
                if 'text_region' in text_item:
                    bbox = text_item['text_region']
                elif 'bbox' in text_item:
                    bbox = text_item['bbox']
                elif 'coordinates' in text_item:
                    bbox = text_item['coordinates']
                
                # Try different text field names
                if 'text' in text_item:
                    text_content = text_item['text']
                elif 'rec_res' in text_item and isinstance(text_item['rec_res'], dict):
                    text_content = text_item['rec_res'].get('text', '')
                    confidence = text_item['rec_res'].get('confidence', 0.0)
                
                # Handle different bbox formats
                if bbox and len(bbox) >= 4:
                    # Handle polygon format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    if isinstance(bbox[0], list) and len(bbox[0]) == 2:
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)
                    # Handle direct coordinates format: [x1, y1, x2, y2]
                    else:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    region = {
                        'type': 'text',
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'score': confidence,
                        'text': text_content
                    }
                    regions.append(region)
                    logger.info(f"  🎯 Created region {i+1}: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] text='{text_content[:30]}...'")
                else:
                    logger.warning(f"  ⚠️ Item {i+1}: No valid bbox found (bbox={bbox})")
    
    # Method 4: Try to extract from any bbox information in the results
    if not regions:
        logger.info(f"Attempting to extract regions from any bbox data")
        
        # Check for overall_ocr_res with dt_polys
        if 'overall_ocr_res' in ocr_results and 'dt_polys' in ocr_results['overall_ocr_res']:
            dt_polys = ocr_results['overall_ocr_res']['dt_polys']
            rec_texts = ocr_results['overall_ocr_res'].get('rec_texts', [])
            
            logger.info(f"Found {len(dt_polys)} text regions in overall_ocr_res")
            
            for i, poly in enumerate(dt_polys):
                if len(poly) >= 4:
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    text_content = rec_texts[i] if i < len(rec_texts) else ""
                    
                    region = {
                        'type': 'text',
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'score': 0.95,  # Default confidence for OCR text
                        'text': text_content
                    }
                    regions.append(region)
    
    logger.info(f"🎯 Extracted {len(regions)} regions for visualization")
    if regions:
        for i, region in enumerate(regions):
            logger.info(f"  Region {i+1}: {region['type']} at {region['bbox']}")
    
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
    Models are loaded once per worker process and reused for all documents/chunks.
    Returns None if pipeline cannot be initialized.
    """
    global pipeline
    
    if pipeline is not None:
        logger.debug("📋 Using cached PPStructure pipeline (models already loaded)")
        return pipeline
    
    try:
        # Initialize if not already done
        pipeline_instance = ensure_pipeline_initialized()
        
        if pipeline_instance is None:
            logger.warning("⚠️ PPStructure pipeline not available")
            raise RuntimeError("PPStructure pipeline initialization returned None")
        
        return pipeline_instance
        
    except Exception as e:
        logger.error(f"Failed to get PPStructure pipeline: {str(e)}")
        raise RuntimeError(f"PPStructure pipeline unavailable: {str(e)}")

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
            logger.info(f"⚡ Performance optimizations: {', '.join(opts)} (estimated 30-60% faster with parallel processing)")
        
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
        
        # Global list to collect all extraction tasks for parallel processing
        all_extraction_tasks = []
        

        
        try:
            # Get pipeline instance (uses globally cached models)
            logger.info("Getting PPStructure pipeline...")
            pipeline_instance = get_pipeline()
            logger.info("PPStructure pipeline retrieved successfully")
            
            # Simple path validation
            absolute_validated_paths = [os.path.abspath(str(path)) for path in image_paths]
            logger.info(f"Processing {len(absolute_validated_paths)} image paths")
            
            # Process pages individually with better error handling and memory management
            all_outputs = []
                
            for i, img_path in enumerate(absolute_validated_paths):
                page_num = i + 1
                
                try:
                    page_start_time = get_unix_timestamp()
                    
                    # Validate image file before processing
                    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                        logger.error(f"Invalid image file for page {page_num}: {img_path}")
                        all_outputs.append(None)
                        continue
                    
                    # Simple validation
                    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                        logger.error(f"Invalid image file for page {page_num}: {img_path}")
                        all_outputs.append(None)
                        continue
                    
                    # Process with the pipeline - use simpler approach
                    logger.info(f"Processing page {page_num}: {os.path.basename(img_path)}")
                    
                    # Call PPStructure pipeline
                    single_output_raw = pipeline_instance(img_path)
                    
                    page_end_time = get_unix_timestamp()
                    
                    # Convert generator to list if needed
                    if hasattr(single_output_raw, '__iter__') and not isinstance(single_output_raw, (list, tuple)):
                        single_output = list(single_output_raw)
                    else:
                        single_output = single_output_raw
                    
                    if single_output and len(single_output) > 0:
                        # Debug: Log the structure of the output to understand the format
                        output_sample = single_output[0]
                        logger.info(f"🔍 Page {page_num} output keys: {list(output_sample.keys()) if hasattr(output_sample, 'keys') else 'Not a dict'}")
                        
                        all_outputs.append(single_output[0])
                        processing_time = calculate_duration(page_start_time, page_end_time)
                        logger.info(f"✅ Page {page_num} completed in {processing_time['formatted']}")
                    else:
                        logger.warning(f"⚠️ Page {page_num} returned empty results")
                        all_outputs.append(None)
                    
                    # Clean up variables and temporary files
                    del single_output_raw
                    if 'single_output' in locals():
                        del single_output
                    
                    # Clean up any temporary resized images
                    if '_resized.jpg' in img_path and os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            logger.debug(f"Cleaned up temporary resized image: {img_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temporary image: {cleanup_error}")
                    
                    # Force garbage collection after each page
                    import gc
                    gc.collect()
                    
                except Exception as page_error:
                    logger.error(f"❌ Page {page_num} processing failed: {str(page_error)}")
                    all_outputs.append(None)
                    
                    # Clean up any temporary files for this page
                    if '_resized.jpg' in img_path and os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                    
                    # Force memory cleanup on error
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
            # Visualization skipped if disabled
            
            # Extract OCR text - handle multiple possible formats
            ocr_text = []
            
            # Debug: Log the output structure to understand the format
            logger.info(f"🔍 Page {actual_page_num} output_dict keys: {list(output_dict.keys())}")
            
            # ENHANCED DEBUG: Log the complete structure of the 'res' field for bounding box extraction
            if 'res' in output_dict and isinstance(output_dict['res'], list) and output_dict['res']:
                logger.info(f"🔍 DETAILED 'res' structure for page {actual_page_num}:")
                for i, item in enumerate(output_dict['res'][:2]):  # Log first 2 items
                    logger.info(f"  Item {i}: keys={list(item.keys()) if isinstance(item, dict) else 'not dict'}")
                    if isinstance(item, dict):
                        if 'text_region' in item:
                            logger.info(f"    text_region: {item['text_region']}")
                        if 'text' in item:
                            logger.info(f"    text: {item['text']}")
                        if 'rec_res' in item:
                            logger.info(f"    rec_res: {item['rec_res']}")
            
            # Try different possible OCR text extraction methods
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                ocr_text = output_dict['overall_ocr_res']['rec_texts']
                logger.info(f"📝 Found {len(ocr_text)} texts via overall_ocr_res")
            elif 'res' in output_dict and isinstance(output_dict['res'], list):
                # Try extracting from 'res' field (common PPStructure format)
                for item in output_dict['res']:
                    if 'text' in item:
                        ocr_text.append(item['text'])
                    elif 'rec_res' in item and 'text' in item['rec_res']:
                        ocr_text.append(item['rec_res']['text'])
                logger.info(f"📝 Found {len(ocr_text)} texts via res field")
            elif hasattr(output_dict, 'save_res') and output_dict.save_res:
                # Try extracting from save_res (another PPStructure format)
                for item in output_dict.save_res:
                    if hasattr(item, 'text'):
                        ocr_text.append(item.text)
                logger.info(f"📝 Found {len(ocr_text)} texts via save_res")
            else:
                # Fallback: try to find any text fields
                logger.warning(f"⚠️ Unknown PPStructure output format for page {actual_page_num}")
                
                # Try to extract any text we can find
                def extract_text_recursive(obj, texts=[]):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if 'text' in key.lower() and isinstance(value, (list, str)):
                                if isinstance(value, list):
                                    texts.extend([str(v) for v in value if v])
                                elif value:
                                    texts.append(str(value))
                            elif isinstance(value, (dict, list)):
                                extract_text_recursive(value, texts)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_text_recursive(item, texts)
                    return texts
                
                ocr_text = extract_text_recursive(output_dict)
                logger.info(f"📝 Found {len(ocr_text)} texts via recursive search")
            
            if ocr_text:
                # Add to combined text with page marker using actual page number
                all_ocr_text.append(f"--- PAGE {actual_page_num} ---")
                all_ocr_text.extend(ocr_text)
                logger.info(f"📝 Added {len(ocr_text)} text elements from page {actual_page_num}")
            else:
                logger.warning(f"⚠️ No OCR text found for page {actual_page_num}")
            
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
        
        # Process all extraction tasks in parallel (CPU-bound operations)
        if all_extraction_tasks:
            logger.info(f"Extracting {len(all_extraction_tasks)} regions as images (tables/figures/charts)")
            parallel_start_time = get_unix_timestamp()
            
            # Use ThreadPoolExecutor for parallel image extraction
            max_workers = min(max_extraction_workers, len(all_extraction_tasks))
            extraction_lock = Lock()
            
            # Check if parallel processing is enabled
            if not parallel_extraction:
                max_workers = 1
            
            # Lists to store extracted image metadata
            extracted_tables = []
            extracted_figures = []
            extracted_charts = []
            
            def extract_image_task(task):
                """Extract and save image region, return metadata"""
                try:
                    # Extract image region
                    img_bytes = extract_image_from_region(
                        task['image_path'], 
                        task['bbox'], 
                        task['output_path']
                    )
                    
                    if img_bytes and os.path.exists(task['output_path']):
                        # Return metadata about the saved image
                        return {
                            'type': task['type'],
                            'page': task['page'],
                            'region_id': task['region_id'],
                            'image_path': task['output_path'],
                            'context': task['context'],
                            'bbox': task['bbox'],
                            'extracted': True
                        }
                    else:
                        logger.warning(f"Failed to extract {task['type']} from page {task['page']}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Error extracting {task['type']} from page {task['page']}: {str(e)}")
                    return None
            
            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(extract_image_task, task): task 
                    for task in all_extraction_tasks
                }
                
                # Collect results as they complete
                completed_extractions = 0
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        # Thread-safe accumulation of extracted image metadata
                        with extraction_lock:
                            if result['type'] == 'table':
                                extracted_tables.append(result)
                            elif result['type'] == 'figure':
                                extracted_figures.append(result)
                            elif result['type'] == 'chart':
                                extracted_charts.append(result)
                        
                        completed_extractions += 1
                        if completed_extractions % 10 == 0:  # Log progress every 10 completions
                            logger.info(f"Extracted {completed_extractions}/{len(all_extraction_tasks)} images")
            
            parallel_end_time = get_unix_timestamp()
            parallel_duration = calculate_duration(parallel_start_time, parallel_end_time)
            logger.info(f"Image extraction completed in {parallel_duration['formatted']} - {completed_extractions} images saved")
            
            # Store extracted image metadata for processing by document_tasks.py
            extraction_metadata = {
                'tables': extracted_tables,
                'figures': extracted_figures, 
                'charts': extracted_charts,
                'total_extracted': completed_extractions,
                'extraction_time': parallel_duration
            }
        else:
            extraction_metadata = {
                'tables': [],
                'figures': [], 
                'charts': [],
                'total_extracted': 0
            }
        
        # Combine all OCR text for document summarization
        combined_text = "\n".join(all_ocr_text)
        
        # Note: Table, figure, and chart extractions will be processed by document_tasks.py
        # since it has access to Google Genai in Python 3.10
        # Summary generation is also handled entirely by document_tasks.py
        
        # PPStructure worker only handles OCR and layout detection
        # No summary generation in this worker - all AI processing moved to document_tasks.py
        summary_result = {
            "summary": None,
            "analysis": {"note": "Summary generation handled by document_tasks.py with Google AI"},
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
            
            # Extract OCR text for this page using the same logic as above
            page_ocr_text = []
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                page_ocr_text = output_dict['overall_ocr_res']['rec_texts']
            elif 'res' in output_dict and isinstance(output_dict['res'], list):
                for item in output_dict['res']:
                    if 'text' in item:
                        page_ocr_text.append(item['text'])
                    elif 'rec_res' in item and 'text' in item['rec_res']:
                        page_ocr_text.append(item['rec_res']['text'])
            else:
                # Use the same recursive extraction as above
                def extract_text_recursive(obj, texts=[]):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if 'text' in key.lower() and isinstance(value, (list, str)):
                                if isinstance(value, list):
                                    texts.extend([str(v) for v in value if v])
                                elif value:
                                    texts.append(str(value))
                            elif isinstance(value, (dict, list)):
                                extract_text_recursive(value, texts)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_text_recursive(item, texts)
                    return texts
                
                page_ocr_text = extract_text_recursive(output_dict)
            
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
        
        logger.info(f"Document processing completed for job {job_id}")
        
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
            "extracted_info": extracted_info,
            # Include extraction metadata for processing by document_tasks.py with Genai
            "extraction_metadata": extraction_metadata
        }
        
    except Exception as e:
        logger.error(f"Error in PPStructure processing: {str(e)}")
        raise 
