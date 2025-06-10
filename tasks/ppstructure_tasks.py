"""
PPStructure processing tasks for advanced document layout analysis.
Uses PaddleOCR's PPStructure for layout detection, OCR, and structure analysis.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import datetime
import random
import inspect
import warnings



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

# Import PaddleOCR components
from paddleocr import PPStructureV3
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont

# Import utilities
from tasks.utils import get_unix_timestamp, calculate_duration, format_duration
from text_based_processor import TextBasedProcessor

logger = get_task_logger(__name__)

# Initialize PPStructureV3 pipeline with error handling
pipeline = None
processing_lock = Lock()

# Initialize TextBasedProcessor for Gemini integration
text_processor = TextBasedProcessor()

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

def initialize_pipeline():
    """Initialize PPStructure pipeline with simple approach like working version"""
    global pipeline
    try:
        logger.info("Initializing PPStructureV3 pipeline...")
        
        # Initialize PPStructureV3 exactly like the working code - simple approach
        pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
        
        logger.info("PPStructureV3 pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PPStructureV3 pipeline: {str(e)}")
        return False

# DO NOT initialize pipeline at module load - use lazy loading only!
# Pipeline will be initialized only when actually needed by get_pipeline()

@shared_task(name='tasks.warmup_ppstructure')
def warmup_ppstructure():
    """
    Warmup task to pre-initialize PPStructure models.
    Call this when document workers start to preload models.
    """
    try:
        logger.info("Warming up PPStructure pipeline...")
        get_pipeline()  # This will initialize the pipeline
        logger.info("PPStructure pipeline warmup completed successfully")
        return {"status": "success", "message": "PPStructure models loaded and ready"}
    except Exception as e:
        logger.error(f"PPStructure warmup failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

def create_document_chunks(image_path, max_chunk_size=(2000, 2000), overlap=200):
    """
    Create chunks from a large document image for parallel processing.
    
    Args:
        image_path: Path to the image file
        max_chunk_size: Maximum size of each chunk (width, height)
        overlap: Overlap between chunks in pixels
        
    Returns:
        List of chunk information with paths and coordinates
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        height, width = image.shape[:2]
        max_width, max_height = max_chunk_size
        
        # If image is smaller than chunk size, no need to chunk
        if width <= max_width and height <= max_height:
            return [{
                'chunk_id': 'full',
                'path': image_path,
                'coordinates': {'x': 0, 'y': 0, 'width': width, 'height': height},
                'original_size': {'width': width, 'height': height}
            }]
        
        chunks = []
        chunk_id = 0
        
        # Create chunks directory
        base_dir = os.path.dirname(image_path)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        chunks_dir = os.path.join(base_dir, f"{basename}_chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Generate chunks with overlap
        y = 0
        while y < height:
            x = 0
            while x < width:
                # Calculate chunk boundaries
                x2 = min(x + max_width, width)
                y2 = min(y + max_height, height)
                
                # Extract chunk
                chunk = image[y:y2, x:x2]
                
                # Save chunk
                chunk_filename = f"chunk_{chunk_id:03d}.jpg"
                chunk_path = os.path.join(chunks_dir, chunk_filename)
                cv2.imwrite(chunk_path, chunk)
                
                # Store chunk info
                chunks.append({
                    'chunk_id': f"chunk_{chunk_id:03d}",
                    'path': chunk_path,
                    'coordinates': {'x': x, 'y': y, 'width': x2-x, 'height': y2-y},
                    'original_size': {'width': width, 'height': height}
                })
                
                chunk_id += 1
                
                # Move to next column
                if x2 >= width:
                    break
                x += max_width - overlap
                if x >= width:
                    break
            
            # Move to next row
            if y2 >= height:
                break
            y += max_height - overlap
            if y >= height:
                break
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        return []

def process_chunk_parallel(chunk_info):
    """Process a single chunk in parallel."""
    try:
        chunk_start_time = get_unix_timestamp()
        
        # Ensure chunk path is an absolute string path for Docker compatibility
        chunk_path = os.path.abspath(str(chunk_info['path']))
        logger.info(f"Processing chunk with absolute path: {chunk_path}")
        
        # Docker workaround: Load image as numpy array instead of using string path
        chunk_image = cv2.imread(chunk_path)
        if chunk_image is None:
            raise ValueError(f"Could not load chunk image: {chunk_path}")
            
        logger.info(f"Loaded chunk as numpy array: shape {chunk_image.shape}, dtype {chunk_image.dtype}")
        
        # Process the chunk using PPStructure with numpy array input
        result = get_pipeline().predict(input=[chunk_image])[0]
        
        # Process result to make it JSON serializable
        processed_result = process_output_for_json(result)
        
        # Add chunk metadata
        chunk_end_time = get_unix_timestamp()
        processed_result['chunk_info'] = chunk_info
        processed_result['processing_time'] = calculate_duration(chunk_start_time, chunk_end_time)
        processed_result['processing_method'] = 'chunked'
        
        return processed_result
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_info['chunk_id']}: {str(e)}")
        return {
            'chunk_info': chunk_info,
            'error': str(e),
            'processing_time': {'seconds': 0, 'formatted': '0s'},
            'processing_method': 'failed'
        }

def merge_chunk_results(chunk_results, original_size):
    """
    Merge results from multiple chunks back into a single result.
    
    Args:
        chunk_results: List of results from individual chunks
        original_size: Original document size {'width': w, 'height': h}
        
    Returns:
        Merged result in the same format as single document processing
    """
    try:
        merged_result = {
            'layout_det_res': {'boxes': []},
            'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
            'parsing_res_list': [],
            'processing_info': {
                'chunk_count': len(chunk_results),
                'merged_at': datetime.datetime.now().isoformat(),
                'chunk_timings': []
            }
        }
        
        total_processing_time = 0
        
        for chunk_result in chunk_results:
            if 'error' in chunk_result:
                continue
                
            chunk_info = chunk_result.get('chunk_info', {})
            chunk_coords = chunk_info.get('coordinates', {'x': 0, 'y': 0})
            offset_x = chunk_coords['x']
            offset_y = chunk_coords['y']
            
            # Add processing time
            chunk_time = chunk_result.get('processing_time', {}).get('seconds', 0)
            total_processing_time += chunk_time
            merged_result['processing_info']['chunk_timings'].append({
                'chunk_id': chunk_info.get('chunk_id', 'unknown'),
                'processing_time': chunk_result.get('processing_time', {}),
                'coordinates': chunk_coords
            })
            
            # Merge layout detection results
            if 'layout_det_res' in chunk_result and 'boxes' in chunk_result['layout_det_res']:
                for box in chunk_result['layout_det_res']['boxes']:
                    # Adjust coordinates to original image space
                    adjusted_box = box.copy()
                    if 'coordinate' in adjusted_box:
                        coord = adjusted_box['coordinate']
                        adjusted_box['coordinate'] = [
                            coord[0] + offset_x,
                            coord[1] + offset_y,
                            coord[2] + offset_x,
                            coord[3] + offset_y
                        ]
                    merged_result['layout_det_res']['boxes'].append(adjusted_box)
            
            # Merge OCR results
            if 'overall_ocr_res' in chunk_result:
                ocr_res = chunk_result['overall_ocr_res']
                
                if 'rec_boxes' in ocr_res:
                    for box in ocr_res['rec_boxes']:
                        adjusted_box = [
                            box[0] + offset_x,
                            box[1] + offset_y,
                            box[2] + offset_x,
                            box[3] + offset_y
                        ]
                        merged_result['overall_ocr_res']['rec_boxes'].append(adjusted_box)
                
                if 'rec_texts' in ocr_res:
                    merged_result['overall_ocr_res']['rec_texts'].extend(ocr_res['rec_texts'])
                
                if 'rec_scores' in ocr_res:
                    merged_result['overall_ocr_res']['rec_scores'].extend(ocr_res['rec_scores'])
            
            # Merge parsing results
            if 'parsing_res_list' in chunk_result:
                for item in chunk_result['parsing_res_list']:
                    adjusted_item = item.copy()
                    if 'block_bbox' in adjusted_item:
                        bbox = adjusted_item['block_bbox']
                        adjusted_item['block_bbox'] = [
                            bbox[0] + offset_x,
                            bbox[1] + offset_y,
                            bbox[2] + offset_x,
                            bbox[3] + offset_y
                        ]
                    merged_result['parsing_res_list'].append(adjusted_item)
        
        # Add overall timing info
        merged_result['processing_info']['total_chunk_processing_time'] = {
            'seconds': round(total_processing_time, 2),
            'formatted': format_duration(total_processing_time)
        }
        
        return merged_result
        
    except Exception as e:
        logger.error(f"Error merging chunk results: {str(e)}")
        return {
            'layout_det_res': {'boxes': []},
            'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
            'parsing_res_list': [],
            'error': f"Error merging results: {str(e)}"
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
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure coordinates are within image bounds
    height, width = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # Extract region
    region_img = img[y1:y2, x1:x2]
    
    # Save region image
    cv2.imwrite(output_path, region_img)
    
    # Return image bytes for Gemini processing
    _, img_bytes = cv2.imencode('.png', region_img)
    return img_bytes.tobytes()

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
            
        # Convert to RGB for PIL (exactly like working code)
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
                    
                    # Draw rectangle (exactly like working code)
                    draw.rectangle(
                        [(x1, y1), (x2, y2)], 
                        outline=color_rgb, 
                        width=2
                    )
                    
                    # Add label (exactly like working code)
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
        
        # Convert back to OpenCV format (exactly like working code)
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
    """Get the pipeline, initializing if needed"""
    global pipeline
    if pipeline is None:
        with processing_lock:
            if pipeline is None:  # Double-check pattern
                if not initialize_pipeline():
                    raise RuntimeError("PPStructure pipeline could not be initialized")
    return pipeline

@shared_task(name='tasks.process_document_with_ppstructure')
def process_document_with_ppstructure(job_id, file_path, file_name, generate_summary=True):
    """
    Process a document using PPStructure with intelligent chunking and parallel processing.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        generate_summary: Whether to generate summary (False for chunks, True for whole documents)
        
    Returns:
        dict: Processing results including layout analysis, OCR, and extracted information
    """
    try:
        # Initialize pipeline with lazy loading
        logger.info(f"Starting PPStructure processing for job {job_id}")
        
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
            # For chunks, create subdirectories within the main job folder
            images_dir = os.path.join(result_dir, "images", chunk_id)
            vis_dir = os.path.join(result_dir, "visualizations", chunk_id)
            tables_dir = os.path.join(result_dir, "tables", chunk_id)
            figures_dir = os.path.join(result_dir, "figures", chunk_id)
            chunk_results_dir = os.path.join(result_dir, "chunk_results", chunk_id)
            
            os.makedirs(chunk_results_dir, exist_ok=True)
        else:
            # For single documents, use the standard structure
            images_dir = os.path.join(result_dir, "images")
            vis_dir = os.path.join(result_dir, "visualizations")
            tables_dir = os.path.join(result_dir, "tables")
            figures_dir = os.path.join(result_dir, "figures")
        
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Process based on file type
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext == '.pdf':
            # Convert PDF to images - exactly like working code
            images = convert_from_bytes(open(file_path, "rb").read())
            
            # Save all page images - exactly like working code
            image_paths = []
            for i, image in enumerate(images):
                page_num = i + 1
                image_path = os.path.join(images_dir, f"page_{page_num}.jpg")
                image.save(image_path)
                image_paths.append(image_path)
        else:
            # Single image file - simplified approach
            image_path = os.path.join(images_dir, f"page_1.jpg")
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
        
        # Process each page directly (no internal image chunking)
        all_ocr_text = []
        table_extractions = []
        figure_extractions = []
        
        try:
            # Process all images at once with PPStructureV3 using string file paths (NO numpy arrays)
            logger.info(f"Processing {len(image_paths)} images with PPStructure")
            
            # Get pipeline instance
            pipeline_instance = get_pipeline()
            
            # Validate all image paths as strings (no numpy array loading)
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
            
            # Ensure paths are explicitly strings and convert to absolute paths for Docker compatibility
            absolute_validated_paths = []
            for path in validated_paths:
                abs_path = os.path.abspath(str(path))  # Ensure string and absolute path
                absolute_validated_paths.append(abs_path)
            
            # Try using file paths first (better for PPStructure filename handling)
            # Fall back to numpy arrays only if file path processing fails
            try:
                logger.info(f"Attempting PPStructure processing with file paths...")
                
                # Process images individually to prevent memory issues and SIGKILL
                all_outputs = []
                
                for i, img_path in enumerate(absolute_validated_paths):
                    try:
                        # Process single image with file path
                        single_output = pipeline_instance.predict(input=[img_path])
                        
                        if single_output and len(single_output) > 0:
                            all_outputs.append(single_output[0])
                        else:
                            logger.warning(f"Empty output for image {i+1}")
                            all_outputs.append(None)
                        
                        # Clear memory after each image
                        del single_output
                        import gc
                        gc.collect()
                        
                        # For chunks, add small delay to prevent rapid memory allocation
                        if is_chunk:
                            import time
                            time.sleep(0.1)  # 100ms delay between images in chunks
                            
                    except Exception as img_error:
                        logger.error(f"Error processing image {i+1} with file path: {str(img_error)}")
                        all_outputs.append(None)
                        # Clean up even on error
                        import gc
                        gc.collect()
                        continue
                        
                logger.info(f"File path processing completed successfully")
                
            except Exception as path_error:
                logger.warning(f"File path processing failed: {str(path_error)}, falling back to numpy arrays...")
                
                # Fallback: Docker environment workaround using numpy arrays
                # PaddleOCR in Docker has issues with string path processing
                numpy_images = []
                for img_path in absolute_validated_paths:
                    try:
                        img_array = cv2.imread(img_path)
                        if img_array is not None:
                            numpy_images.append(img_array)
                        else:
                            logger.error(f"Failed to load image as numpy array: {img_path}")
                            raise ValueError(f"Could not load image: {img_path}")
                    except Exception as img_error:
                        logger.error(f"Error loading image {img_path}: {str(img_error)}")
                        raise
                
                if not numpy_images:
                    raise ValueError("No valid images could be loaded as numpy arrays")
                    
                # Process images individually to prevent memory issues and SIGKILL
                all_outputs = []
                
                for i, numpy_image in enumerate(numpy_images):
                    try:
                        # Process single image with memory optimization
                        single_output = pipeline_instance.predict(input=[numpy_image])
                        
                        if single_output and len(single_output) > 0:
                            all_outputs.append(single_output[0])
                        else:
                            logger.warning(f"Empty output for image {i+1}")
                            all_outputs.append(None)
                        
                        # Clear memory after each image - aggressive cleanup
                        del single_output
                        del numpy_image  # Clear this specific image from memory
                        import gc
                        gc.collect()
                        
                        # For chunks, add small delay to prevent rapid memory allocation
                        if is_chunk:
                            import time
                            time.sleep(0.1)  # 100ms delay between images in chunks
                            
                    except Exception as img_error:
                        logger.error(f"Error processing image {i+1}: {str(img_error)}")
                        all_outputs.append(None)
                        # Clean up even on error
                        import gc
                        gc.collect()
                        continue
                
                # Clear numpy_images list to free memory
                del numpy_images
                import gc
                gc.collect()
            
            # Update image_paths to match validated paths
            image_paths = validated_paths
            
            if not all_outputs:
                logger.warning("PPStructure returned empty results for all pages")
                all_outputs = [None] * len(image_paths)  # Create empty placeholders
            else:
                successful_count = sum(1 for output in all_outputs if output is not None)
                logger.info(f"PPStructure processing completed: {successful_count}/{len(all_outputs)} images processed successfully")
            
        except Exception as e:
            logger.error(f"Error in PPStructure processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create empty results for all pages if processing fails
            all_outputs = [None] * len(image_paths)
        
        # Process results for each page
        processed_outputs = []
        for page_num, (image_path, output) in enumerate(zip(image_paths, all_outputs), 1):
            page_start_time = get_unix_timestamp()
            
            try:
                if output is not None:
                    # Process the output directly like working code
                    
                    # Convert output to dict format for processing
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
                        logger.warning(f"PPStructure returned empty results for page {page_num}")
                        output_dict = {
                            'layout_det_res': {'boxes': []},
                            'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
                            'parsing_res_list': [],
                            'error': 'Empty results from PPStructure'
                        }
                else:
                    # Create empty structure for None results
                    logger.warning(f"No output for page {page_num}, creating empty structure")
                    output_dict = {
                        'layout_det_res': {'boxes': []},
                        'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
                        'parsing_res_list': [],
                        'error': 'No output from PPStructure'
                    }
                
            except Exception as e:
                logger.error(f"Error post-processing page {page_num}: {str(e)}")
                # Create empty result structure for failed page
                output_dict = {
                    'layout_det_res': {'boxes': []},
                    'overall_ocr_res': {'rec_boxes': [], 'rec_texts': [], 'rec_scores': []},
                    'parsing_res_list': [],
                    'error': f'Post-processing failed: {str(e)}'
                }
            
            # Add timing information
            page_end_time = get_unix_timestamp()
            output_dict['page_processing_time'] = calculate_duration(page_start_time, page_end_time)
            output_dict['processing_method'] = 'batch_direct'
            output_dict['page_number'] = page_num
            
            processed_outputs.append(output_dict)
            
            # Extract layout regions
            regions = extract_layout_regions(output_dict)
            
            # Create visualization with bounding boxes
            vis_path = os.path.join(vis_dir, f"page_{page_num}_layout.jpg")
            layout_vis_result = draw_bounding_boxes(image_path, regions, vis_path)
            if layout_vis_result is None:
                logger.warning(f"Failed to create layout visualization for page {page_num}")
            
            # Extract OCR text
            ocr_text = []
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                ocr_text = output_dict['overall_ocr_res']['rec_texts']
                # Add to combined text with page marker
                all_ocr_text.append(f"--- PAGE {page_num} ---")
                all_ocr_text.extend(ocr_text)
            
            # Process tables and figures with Gemini AI
            for j, region in enumerate(regions):
                region_type = region.get('type', '').lower()
                bbox = region.get('bbox')
                
                if region_type == 'table' and bbox:
                    # Extract table image
                    table_img_path = os.path.join(tables_dir, f"page_{page_num}_table_{j+1}.png")
                    table_img_bytes = extract_image_from_region(image_path, bbox, table_img_path)
                    
                    # Process table with Gemini
                    context = f"Table from page {page_num}"
                    table_text = text_processor.process_table_image(table_img_bytes, context)
                    
                    # Add to table extractions
                    table_extractions.append(f"\n--- TABLE FROM PAGE {page_num} ---\n{table_text}\n")
                    
                elif region_type == 'figure' and bbox:
                    # Extract figure image
                    figure_img_path = os.path.join(figures_dir, f"page_{page_num}_figure_{j+1}.png")
                    figure_img_bytes = extract_image_from_region(image_path, bbox, figure_img_path)
                    
                    # Process figure with Gemini
                    context = f"Figure from page {page_num}"
                    figure_text = text_processor.process_figure_image(figure_img_bytes, context)
                    
                    # Add to figure extractions
                    figure_extractions.append(f"\n--- FIGURE FROM PAGE {page_num} ---\n{figure_text}\n")
        
        # Combine all text for document summarization
        combined_text = "\n".join(all_ocr_text)
        
        # Add table and figure extractions
        if table_extractions:
            combined_text += "\n\n" + "\n".join(table_extractions)
        
        if figure_extractions:
            combined_text += "\n\n" + "\n".join(figure_extractions)
        
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
                "summary": None,  # No summary for individual chunks
                "analysis": {},   # No entity extraction for chunks
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
        
        # Prepare individual page results in the format you prefer
        page_results = []
        for page_num, (image_path, output_dict) in enumerate(zip(image_paths, processed_outputs), 1):
            # Get relative paths for web access
            relative_result_dir = f"results/{job_id}"
            
            # Extract OCR text for this page
            page_ocr_text = []
            if 'overall_ocr_res' in output_dict and 'rec_texts' in output_dict['overall_ocr_res']:
                page_ocr_text = output_dict['overall_ocr_res']['rec_texts']
            
            # Extract layout regions for this page
            regions = extract_layout_regions(output_dict)
            
            page_result = {
                "page_number": page_num,
                "image_path": f"{relative_result_dir}/images/page_{page_num}.jpg",
                "layout_vis_path": f"{relative_result_dir}/visualizations/page_{page_num}_layout.jpg",
                "regions": regions,
                "ocr_text": page_ocr_text,
                "json_path": f"{relative_result_dir}/page_results/page_{page_num}_res.json" if hasattr(output_dict, 'save_to_json') else None
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
        
        # Prepare clean results.json like your reference code with extracted_info prominently displayed
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
            "processing_method": "individual_numpy_arrays"
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
                "processing_method": "individual_numpy_arrays",
                "docker_environment": True,
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
                "total_chunks_processed": 0,
                "pages_with_chunking": 0,
                "processing_time": processing_duration,
                "chunking_enabled": False,
                "individual_processing": True,
                "memory_optimization": True
            },
            "confidence_metrics": {
                "average_confidence": average_confidence,
                "average_confidence_formatted": average_confidence_formatted,
                "total_text_elements": len(all_scores),
                "confidence_scores": all_scores[:100] if len(all_scores) > 100 else all_scores  # Limit to first 100 for file size
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