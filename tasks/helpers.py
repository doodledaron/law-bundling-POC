"""
Helper/utility functions for document processing tasks.
Extracted from ppstructure_tasks.py for better modularity.
"""
import os
import json
import logging
import redis

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# Import PaddleOCR components conditionally to prevent errors in lite containers
try:
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
    PADDLEPADDLE_AVAILABLE = True
except ImportError:
    PADDLEPADDLE_AVAILABLE = False

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


def process_output_for_json(output):
    """Process the output from PPStructureV3 to make it JSON serializable"""
    if isinstance(output, list):
        return [process_output_for_json(item) for item in output]
    elif isinstance(output, dict):
        return {k: process_output_for_json(v) for k, v in output.items()}
    elif PADDLEPADDLE_AVAILABLE and isinstance(output, np.ndarray):
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
