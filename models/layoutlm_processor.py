import os
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import paddleocr
from io import BytesIO
import base64
from pdf2image import convert_from_bytes
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
from typing import List, Dict, Tuple, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

class LayoutLMProcessor:
    """
    A processor class for document understanding using LayoutLM.
    This implementation is designed specifically for legal document processing.
    """
    def __init__(self, model_name: str = "microsoft/layoutlm-base-uncased"):
        """
        Initialize the LayoutLM processor with the specified model.
        
        Args:
            model_name: The name or path of the LayoutLM model to use
        """
        logger.info(f"Initializing LayoutLM processor with model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_name)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize PaddleOCR
        logger.info("Initializing PaddleOCR")
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
        logger.info("PaddleOCR initialized")
        
        # Fine-tuned model path (to be used after fine-tuning)
        self.fine_tuned_model_path = os.path.join("models", "layoutlm_legal_finetuned")
    
    def preprocess_document(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Process a document file (PDF or image) for LayoutLM input.
        
        Args:
            file_content: The binary content of the file
            
        Returns:
            A list of preprocessed pages with text, bounding boxes, and features
        """
        logger.info("Preprocessing document for LayoutLM")
        
        # Convert PDF to images if needed
        if file_content[:4] == b"%PDF":
            logger.info("Converting PDF to images")
            images = convert_from_bytes(file_content)
        else:
            # Process as image
            logger.info("Processing as image")
            image = Image.open(BytesIO(file_content))
            images = [image]
        
        processed_pages = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}")
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Get image dimensions
            width, height = image.size
            
            # Convert PIL Image to numpy array for PaddleOCR
            image_np = np.array(image)
            
            # Extract text and bounding boxes using PaddleOCR
            ocr_result = self.ocr.ocr(image_np, cls=True)
            
            words = []
            boxes = []
            
            if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                # Process OCR results
                for line in ocr_result[0]:
                    # PaddleOCR returns [box, (text, confidence)]
                    box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = line[1][0]  # text content
                    confidence = line[1][1]  # confidence score
                    
                    if len(text.strip()) > 0:
                        # Convert box coordinates to [x1, y1, x2, y2] format and normalize
                        x1 = min(point[0] for point in box)
                        y1 = min(point[1] for point in box)
                        x2 = max(point[0] for point in box)
                        y2 = max(point[1] for point in box)
                        
                        # LayoutLM expects normalized box coordinates
                        # Normalize coordinates to be between 0 and 1000
                        x1_norm = int(1000 * (x1 / width))
                        y1_norm = int(1000 * (y1 / height))
                        x2_norm = int(1000 * (x2 / width))
                        y2_norm = int(1000 * (y2 / height))
                        
                        # Ensure coordinates are within bounds
                        x1_norm = max(0, min(x1_norm, 1000))
                        y1_norm = max(0, min(y1_norm, 1000))
                        x2_norm = max(0, min(x2_norm, 1000))
                        y2_norm = max(0, min(y2_norm, 1000))
                        
                        words.append(text)
                        boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            if not words:
                logger.warning(f"No text found on page {i+1}")
                words = ["[NO_TEXT]"]
                boxes = [[0, 0, 0, 0]]
            
            # Encode tokens with bounding boxes
            encoded_inputs = self.tokenizer(
                words,
                padding="max_length",
                truncation=True,
                max_length=512,
                is_split_into_words=True,
                return_tensors="pt"
            )
            
            # We need to manually add bounding boxes to the encoded inputs
            word_ids = encoded_inputs.word_ids()
            bbox_inputs = []
            
            # Map token to its corresponding word bounding box
            for word_id in word_ids:
                if word_id is None:
                    bbox_inputs.append([0, 0, 0, 0])  # For special tokens like [CLS], [SEP], etc.
                else:
                    # If word_id is valid, use the corresponding bounding box
                    if word_id < len(boxes):
                        bbox_inputs.append(boxes[word_id])
                    else:
                        bbox_inputs.append([0, 0, 0, 0])  # Use default for out-of-range word_ids
            
            # Convert bounding boxes to tensor
            bbox_tensor = torch.tensor([bbox_inputs], dtype=torch.long)
            encoded_inputs["bbox"] = bbox_tensor
            
            processed_pages.append({
                "page_num": i+1,
                "image": image,
                "words": words,
                "boxes": boxes,
                "encoding": encoded_inputs
            })
        
        logger.info(f"Preprocessed {len(processed_pages)} pages")
        return processed_pages
    
    def extract_entities(self, processed_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract entities from preprocessed document pages.
        
        Args:
            processed_pages: List of preprocessed pages from preprocess_document
            
        Returns:
            Dictionary containing extracted entities and structured information
        """
        logger.info("Extracting entities using LayoutLM")
        
        # Switch model to evaluation mode
        self.model.eval()
        
        results = {}
        full_text = []
        
        with torch.no_grad():
            for page in processed_pages:
                # Move tensors to device
                input_ids = page["encoding"]["input_ids"].to(self.device)
                attention_mask = page["encoding"]["attention_mask"].to(self.device)
                token_type_ids = page["encoding"]["token_type_ids"].to(self.device)
                bbox = page["encoding"]["bbox"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    bbox=bbox
                )
                
                # Get predictions
                predictions = outputs.logits.argmax(dim=2)
                
                # Process predictions
                # Note: This is a placeholder - you would map predictions to entity labels
                # based on your specific fine-tuned model
                
                # Add page text to full text
                if page["words"]:
                    page_text = " ".join(page["words"])
                    full_text.append(page_text)
        
        results["full_text"] = "\n".join(full_text)
        
        # Extract legal entities (placeholder logic)
        # In a fully implemented system, this would use the actual predictions
        # from the fine-tuned model to identify entities
        
        results["extracted_entities"] = {
            "parties": [],
            "dates": [],
            "clauses": [],
            "signatures": []
        }
        
        logger.info("Entity extraction complete")
        return results
    
    def fine_tune_for_legal_documents(self, training_data_path: str, epochs: int = 5) -> None:
        """
        Fine-tune the LayoutLM model for legal document processing.
        
        Args:
            training_data_path: Path to the training data
            epochs: Number of training epochs
        """
        logger.info(f"Fine-tuning LayoutLM model for legal documents, epochs: {epochs}")
        
        # Placeholder for fine-tuning logic
        # A complete implementation would:
        # 1. Load and prepare legal document dataset
        # 2. Define training parameters and optimizer
        # 3. Train the model
        # 4. Save the fine-tuned model
        
        logger.info(f"Fine-tuning complete, saving model to {self.fine_tuned_model_path}")
        
        # Save the fine-tuned model
        os.makedirs(self.fine_tuned_model_path, exist_ok=True)
        self.model.save_pretrained(self.fine_tuned_model_path)
        self.tokenizer.save_pretrained(self.fine_tuned_model_path)
    
    def visualize_layout(self, processed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Visualize the detected layout by drawing bounding boxes on the images.
        
        Args:
            processed_pages: List of preprocessed pages from preprocess_document
            
        Returns:
            List of dictionaries with page images and their base64 representations
        """
        logger.info("Visualizing document layout")
        
        visualization_results = []
        
        for page in processed_pages:
            # Get the original image
            image = page["image"].copy()
            draw = ImageDraw.Draw(image)
            
            # Draw bounding boxes
            boxes = page["boxes"]
            words = page["words"]
            
            # Get image dimensions for denormalization
            width, height = image.size
            
            # Try to get a font for text
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                font = ImageFont.load_default()
            
            for i, (box, word) in enumerate(zip(boxes, words)):
                # Denormalize the coordinates
                x1, y1, x2, y2 = box
                
                # If coordinates are normalized (0-1000 range), convert back to pixels
                if max(box) <= 1000:
                    x1 = int(x1 * width / 1000)
                    y1 = int(y1 * height / 1000)
                    x2 = int(x2 * width / 1000)
                    y2 = int(y2 * height / 1000)
                
                # Draw rectangle
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                
                # Draw word identifier
                draw.text((x1, y1-15), word[:10], fill="blue", font=font)
            
            # Convert image to base64 for web display
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            visualization_results.append({
                "page_num": page["page_num"],
                "image_base64": img_str
            })
        
        logger.info(f"Visualized layout for {len(visualization_results)} pages")
        return visualization_results 