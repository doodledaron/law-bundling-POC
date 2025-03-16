import os
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentRegion:
    """Class to store document region information"""
    region_type: str  # e.g., 'text', 'table', 'header', 'footer', 'logo', 'signature'
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in normalized coordinates
    confidence: float
    text: str = ""  # Will store OCR text for this region

class DocumentLayoutAnalyzer:
    """YOLOv8-based document layout analyzer for region detection"""
    
    REGION_TYPES = [
        'text', 'title', 'list', 'table', 'figure', 'header', 
        'footer', 'signature', 'logo', 'stamp', 'handwriting'
    ]
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize the document layout analyzer
        
        Args:
            model_path: Path to YOLOv8 model file. If None, will download from Ultralytics
            confidence_threshold: Minimum confidence threshold for detections
        """
        logger.info(f"Initializing DocumentLayoutAnalyzer with confidence threshold: {confidence_threshold}")
        self.confidence_threshold = confidence_threshold
        
        # Default to YOLOv8n if no model specified
        if model_path is None or not os.path.exists(model_path):
            logger.info("No model path provided or file not found. Using default YOLOv8n model.")
            self.model = YOLO('yolov8n.pt')
            logger.warning("Using general purpose YOLOv8n model, not specialized for document layout analysis.")
            logger.warning("For better results on legal documents, use a model fine-tuned on document layouts.")
            # We'll fine-tune the model for document layout analysis
            # This is a placeholder - in production, you'd use a pre-trained layout model
        else:
            logger.info(f"Loading custom model from: {model_path}")
            self.model = YOLO(model_path)
            
        # Ensure model directory exists
        os.makedirs('models', exist_ok=True)
        logger.info("Document layout analyzer initialized successfully")
    
    def detect_regions(self, image: np.ndarray) -> List[DocumentRegion]:
        """
        Detect document regions in an image
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            List of DocumentRegion objects
        """
        # Get image dimensions for logging
        h, w = image.shape[:2]
        logger.info(f"Detecting regions in image of size {w}x{h}")
        
        # Run YOLOv8 inference
        logger.debug("Running YOLOv8 inference")
        results = self.model(image, verbose=False)
        
        regions = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            logger.debug(f"YOLOv8 detected {len(boxes)} potential regions")
            
            for i, box in enumerate(boxes):
                # Get box coordinates (normalized)
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Skip low confidence detections
                if conf < self.confidence_threshold:
                    logger.debug(f"Skipping region {i} due to low confidence: {conf:.2f}")
                    continue
                
                # Map class ID to region type
                if cls_id < len(self.REGION_TYPES):
                    region_type = self.REGION_TYPES[cls_id]
                else:
                    region_type = "unknown"
                
                logger.debug(f"Detected region {i}: type={region_type}, confidence={conf:.2f}, bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                
                # Create DocumentRegion object
                region = DocumentRegion(
                    region_type=region_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )
                
                regions.append(region)
        
        logger.info(f"Detected {len(regions)} valid regions with confidence >= {self.confidence_threshold}")
        return regions
    
    def get_region_images(self, image: np.ndarray, regions: List[DocumentRegion]) -> Dict[int, np.ndarray]:
        """
        Extract region images from the original image
        
        Args:
            image: Original image
            regions: List of DocumentRegion objects
            
        Returns:
            Dictionary mapping region index to region image
        """
        logger.info(f"Extracting {len(regions)} region images")
        h, w = image.shape[:2]
        region_images = {}
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            
            # Convert normalized coordinates to absolute
            x1_abs = int(x1 * w)
            y1_abs = int(y1 * h)
            x2_abs = int(x2 * w)
            y2_abs = int(y2 * h)
            
            # Extract region image
            region_img = image[y1_abs:y2_abs, x1_abs:x2_abs]
            region_images[i] = region_img
            
            region_h, region_w = region_img.shape[:2]
            logger.debug(f"Extracted region {i}: type={region.region_type}, size={region_w}x{region_h}")
        
        return region_images
    
    def visualize_regions(self, image: np.ndarray, regions: List[DocumentRegion]) -> np.ndarray:
        """
        Visualize detected regions on the image
        
        Args:
            image: Original image
            regions: List of DocumentRegion objects
            
        Returns:
            Image with visualized regions
        """
        logger.info(f"Visualizing {len(regions)} regions")
        h, w = image.shape[:2]
        vis_image = image.copy()
        
        # Color map for different region types (BGR format)
        color_map = {
            'text': (0, 255, 0),      # Green
            'title': (0, 0, 255),     # Red
            'list': (255, 0, 0),      # Blue
            'table': (0, 255, 255),   # Yellow
            'figure': (255, 0, 255),  # Magenta
            'header': (255, 255, 0),  # Cyan
            'footer': (128, 0, 128),  # Purple
            'signature': (0, 165, 255), # Orange
            'logo': (255, 255, 255),  # White
            'stamp': (0, 128, 128),   # Teal
            'handwriting': (128, 128, 128) # Gray
        }
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            
            # Convert normalized coordinates to absolute
            x1_abs = int(x1 * w)
            y1_abs = int(y1 * h)
            x2_abs = int(x2 * w)
            y2_abs = int(y2 * h)
            
            # Get color for region type (default to white if not found)
            color = color_map.get(region.region_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1_abs, y1_abs), (x2_abs, y2_abs), color, 2)
            
            # Prepare label text
            label = f"{region.region_type} ({region.confidence:.2f})"
            
            # Calculate label size and position
            font_scale = 0.5
            font_thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Draw label background
            cv2.rectangle(vis_image, (x1_abs, y1_abs - label_h - 10), (x1_abs + label_w, y1_abs), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1_abs, y1_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
            
            logger.debug(f"Visualized region {i}: type={region.region_type}, bbox=({x1_abs}, {y1_abs}, {x2_abs}, {y2_abs})")
        
        logger.info("Region visualization completed")
        return vis_image
    
    def generate_document_map(self, regions: List[DocumentRegion]) -> Dict:
        """
        Generate a structured document map from regions
        
        Args:
            regions: List of DocumentRegion objects
            
        Returns:
            Dictionary with document structure information
        """
        logger.info(f"Generating document map with {len(regions)} regions")
        document_map = {
            "regions": []
        }
        
        for i, region in enumerate(regions):
            region_data = {
                "id": i,
                "type": region.region_type,
                "bbox": region.bbox,
                "confidence": region.confidence,
                "text": region.text
            }
            document_map["regions"].append(region_data)
            logger.debug(f"Added region {i} to document map: type={region.region_type}, text_length={len(region.text)}")
        
        logger.info("Document map generation completed")
        return document_map 