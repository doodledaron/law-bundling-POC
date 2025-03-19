from typing import Dict, List, Any, Optional
import logging
import paddleocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from models.layoutlm_processor import LayoutLMProcessor
from models.summarizer import Summarizer
from pdf2image import convert_from_bytes
from io import BytesIO
import torch
import colorsys
import re
from datetime import datetime
from transformers import pipeline
import os

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processing pipeline with all necessary components."""
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
        self.layout_processor = LayoutLMProcessor()
        self.summarizer = Summarizer(layoutlm_processor=self.layout_processor)
        
        # Initialize summarization pipeline
        self.summarization_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-xsum",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define patterns for common legal document fields
        self.patterns = {
            "date": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
            "party": r'(?:between|by and between)\s+([^,]+)',
            "address": r'\b\d+\s+[A-Za-z\s,]+(?:Street|St\.|Avenue|Ave\.|Boulevard|Blvd\.|Road|Rd\.|Lane|Ln\.|Drive|Dr\.)\b',
            "amount": r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            "reference": r'(?:Ref\.|Reference|Ref:)\s*([A-Z0-9-]+)',
            "section": r'§\s*\d+\.?\d*|Section\s+\d+\.?\d*|Article\s+\d+\.?\d*'
        }
        
        # Try to load a default font
        try:
            # Different options for font paths based on operating system
            font_paths = [
                # Windows paths
                "C:/Windows/Fonts/arial.ttf",
                # Linux paths
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/Arial.ttf",
                # MacOS paths
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            
            self.default_font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.default_font = ImageFont.truetype(font_path, 12)
                    logger.info(f"Using font: {font_path}")
                    break
                    
            if self.default_font is None:
                # As a last resort, use default font
                self.default_font = ImageFont.load_default()
                logger.info("Using default font")
        except Exception as e:
            logger.warning(f"Could not load font: {str(e)}")
            self.default_font = None
        
    def process_document(self, file_content: bytes) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_content: The binary content of the document
            
        Returns:
            Dictionary containing all processing results
        """
        try:
            # Step 1: Extract text and coordinates using PaddleOCR
            logger.info("Step 1: Extracting text and coordinates")
            ocr_results = self._extract_text_and_coordinates(file_content)
            
            # Step 2: Analyze layout using LayoutLM
            logger.info("Step 2: Analyzing document layout")
            layout_analysis = self._analyze_layout(ocr_results)
            
            # Step 3: Categorize and highlight sections
            logger.info("Step 3: Categorizing and highlighting sections")
            categorized_sections = self._categorize_sections(layout_analysis)
            
            # Step 4: Extract relevant information with context
            logger.info("Step 4: Extracting relevant information")
            extracted_info = self._extract_information(categorized_sections)
            
            # Step 5: Generate summary
            logger.info("Step 5: Generating summary")
            summary = self._generate_summary(extracted_info)
            
            return {
                "ocr_results": ocr_results,
                "layout_analysis": layout_analysis,
                "categorized_sections": categorized_sections,
                "extracted_info": extracted_info,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}", exc_info=True)
            raise
    
    def _extract_text_and_coordinates(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Extract text and coordinates using PaddleOCR.
        
        Args:
            file_content: The binary content of the document
            
        Returns:
            List of dictionaries containing OCR results for each page
        """
        try:
            # Convert PDF to images if needed
            if file_content[:4] == b"%PDF":
                logger.info("Converting PDF to images")
                images = convert_from_bytes(file_content, dpi=300)
            else:
                # Process as image
                logger.info("Processing as image")
                image = Image.open(BytesIO(file_content))
                images = [image]
            
            results = []
            for page_idx, image in enumerate(images):
                logger.info(f"Processing page {page_idx + 1}")
                
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Get image dimensions
                width, height = image.size
                
                # Store the original image
                results.append({
                    "page_idx": page_idx,
                    "width": width,
                    "height": height,
                    "image": image,  # Store the original image
                    "results": []
                })
                
                # Convert PIL Image to numpy array for PaddleOCR
                image_np = np.array(image)
                
                # Run OCR
                ocr_result = self.ocr.ocr(image_np, cls=True)
                
                if not ocr_result or len(ocr_result) == 0:
                    logger.warning(f"No OCR results for page {page_idx + 1}")
                    continue
                
                # Process OCR results
                for line in ocr_result[0]:
                    box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = line[1][0]  # text content
                    confidence = line[1][1]  # confidence score
                    
                    # Skip low confidence results
                    if confidence < 0.5:
                        continue
                    
                    # Convert box coordinates to [x1, y1, x2, y2] format
                    x1 = min(point[0] for point in box)
                    y1 = min(point[1] for point in box)
                    x2 = max(point[0] for point in box)
                    y2 = max(point[1] for point in box)
                    
                    # Normalize coordinates
                    x1_norm = x1 / width
                    y1_norm = y1 / height
                    x2_norm = x2 / width
                    y2_norm = y2 / height
                    
                    results[-1]["results"].append({
                        "text": text,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2],
                        "normalized_box": [x1_norm, y1_norm, x2_norm, y2_norm]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}", exc_info=True)
            raise
    
    def _analyze_layout(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze document layout using LayoutLM.
        
        Args:
            ocr_results: List of OCR results for each page
            
        Returns:
            Dictionary containing layout analysis results
        """
        try:
            layout_results = []
            
            for page in ocr_results:
                logger.info(f"Analyzing layout for page {page['page_idx'] + 1}")
                
                # Prepare input for LayoutLM
                words = []
                boxes = []
                
                for result in page['results']:
                    words.append(result['text'])
                    # Convert normalized coordinates to LayoutLM format (0-1000)
                    box = result['normalized_box']
                    boxes.append([
                        int(box[0] * 1000),
                        int(box[1] * 1000),
                        int(box[2] * 1000),
                        int(box[3] * 1000)
                    ])
                
                logger.info(f"Prepared {len(words)} words with bounding boxes for LayoutLM")
                
                # Get layout predictions
                logger.info("Calling LayoutLM processor for layout analysis")
                layout_predictions = self.layout_processor._analyze_layout_structure(
                    words=words,
                    boxes=boxes,
                    width=page['width'],
                    height=page['height']
                )
                
                # Log prediction details
                if layout_predictions:
                    logger.info(f"Received {len(layout_predictions)} layout predictions")
                    # Log the first few predictions as a sample
                    for i, pred in enumerate(layout_predictions[:5]):
                        logger.debug(f"Sample prediction {i+1}: {pred}")
                else:
                    logger.warning("No layout predictions received!")
                
                # Process predictions
                page_layout = {
                    "page_idx": page['page_idx'],
                    "width": page['width'],
                    "height": page['height'],
                    "words": words,
                    "boxes": boxes,
                    "layout_predictions": layout_predictions,
                    "image": page.get('image')  # Include the original image
                }
                
                layout_results.append(page_layout)
            
            logger.info(f"Layout analysis completed for {len(layout_results)} pages")
            document_structure = self._analyze_document_structure(layout_results)
            logger.info(f"Document structure analysis identified: {', '.join([f'{k}: {len(v)}' for k, v in document_structure.items()])}")
            
            return {
                "pages": layout_results,
                "document_structure": document_structure,
                "ocr_results": ocr_results  # Pass through the OCR results with original images
            }
            
        except Exception as e:
            logger.error(f"Error in layout analysis: {str(e)}", exc_info=True)
            raise
    
    def _analyze_document_structure(self, layout_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the overall document structure based on layout predictions.
        
        Args:
            layout_results: List of layout analysis results for each page
            
        Returns:
            Dictionary containing document structure analysis
        """
        structure = {
            "headers": [],
            "sections": [],
            "tables": [],
            "lists": [],
            "signatures": [],
            "dates": []
        }
        
        for page in layout_results:
            predictions = page['layout_predictions']
            
            # Group words by their predicted layout category
            for word_idx, pred in enumerate(predictions):
                # Get category from region_name or fall back to region_type
                if 'region_name' in pred:
                    category = pred['region_name']
                elif 'region_type' in pred:
                    # Get category name from region_type using the layout processor's region_categories
                    region_type = pred['region_type']
                    category = self.layout_processor.region_categories.get(region_type, "Unknown")
                else:
                    # Skip if prediction doesn't have required keys
                    logger.warning(f"Prediction missing required keys: {pred}")
                    continue
                
                word = page['words'][word_idx]
                box = page['boxes'][word_idx]
                
                if category == "Header":
                    structure["headers"].append({
                        "text": word,
                        "box": box,
                        "page": page['page_idx']
                    })
                elif category == "Text paragraph":
                    structure["sections"].append({
                        "text": word,
                        "box": box,
                        "page": page['page_idx']
                    })
                elif category == "Table":
                    structure["tables"].append({
                        "text": word,
                        "box": box,
                        "page": page['page_idx']
                    })
                elif category == "List item":
                    structure["lists"].append({
                        "text": word,
                        "box": box,
                        "page": page['page_idx']
                    })
                elif category == "Signature":
                    structure["signatures"].append({
                        "text": word,
                        "box": box,
                        "page": page['page_idx']
                    })
                elif category == "Date":
                    structure["dates"].append({
                        "text": word,
                        "box": box,
                        "page": page['page_idx']
                    })
        
        return structure
    
    def _categorize_sections(self, layout_analysis: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize and highlight different sections of the document.
        
        Args:
            layout_analysis: Dictionary containing layout analysis results
            
        Returns:
            Dictionary containing categorized and highlighted sections
        """
        try:
            categorized_sections = {
                "pages": [],
                "section_highlights": {}
            }
            
            # Generate distinct colors for different categories
            colors = self._generate_distinct_colors(len(layout_analysis['document_structure']))
            category_colors = dict(zip(layout_analysis['document_structure'].keys(), colors))
            logger.info(f"Generated colors for {len(category_colors)} categories: {list(category_colors.keys())}")
            
            # Process each page
            for page_idx, page in enumerate(layout_analysis['pages']):
                page_idx = page['page_idx']
                width = page['width']
                height = page['height']
                
                logger.info(f"Creating highlights for page {page_idx + 1}, size: {width}x{height}")
                
                # Create a new image for highlighting with transparency
                highlight_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(highlight_image)
                
                # Count highlighted areas by category
                highlight_counts = {}
                
                # Process each word and its prediction
                for word_idx, pred in enumerate(page['layout_predictions']):
                    # Get category from region_name or fall back to region_type
                    if 'region_name' in pred:
                        category = pred['region_name']
                    elif 'region_type' in pred:
                        region_type = pred['region_type']
                        category = self.layout_processor.region_categories.get(region_type, "Unknown")
                    else:
                        logger.warning(f"Prediction missing required keys: {pred}")
                        continue
                        
                    highlight_counts[category] = highlight_counts.get(category, 0) + 1
                    
                    box = page['boxes'][word_idx]
                    
                    # Convert normalized coordinates back to pixel coordinates
                    x1 = int(box[0] * width / 1000)
                    y1 = int(box[1] * height / 1000)
                    x2 = int(box[2] * width / 1000)
                    y2 = int(box[3] * height / 1000)
                    
                    # Draw highlight with appropriate color and transparency
                    color = category_colors.get(category, (0, 0, 0, 0))
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                    
                    # Draw a border around the box for better visibility
                    border_color = (color[0], color[1], color[2], 200)  # More opaque for border
                    border_width = 2  # Thicker border
                    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_width)
                
                logger.info(f"Page {page_idx + 1} highlights completed: {highlight_counts}")
                
                # Get the original image from OCR results
                original_image = None
                for ocr_page in layout_analysis.get('ocr_results', []):
                    if ocr_page['page_idx'] == page_idx:
                        original_image = ocr_page.get('image')
                        break
                
                if original_image:
                    # Create a composite image by overlaying highlights on the original image
                    composite_image = original_image.copy()
                    composite_image.paste(highlight_image, (0, 0), highlight_image)
                else:
                    # Fallback to white background if original image not found
                    composite_image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
                    composite_image.paste(highlight_image, (0, 0), highlight_image)
                
                # Store the highlighted page
                categorized_sections["pages"].append({
                    "page_idx": page_idx,
                    "highlight_image": composite_image
                })
            
            # Store section highlights for reference
            categorized_sections["section_highlights"] = {
                category: {
                    "color": color,
                    "items": items
                }
                for category, items in layout_analysis['document_structure'].items()
                for color in [category_colors.get(category, (0, 0, 0, 0))]
            }
            
            logger.info(f"Categorization complete: {len(categorized_sections['pages'])} pages processed")
            return categorized_sections
            
        except Exception as e:
            logger.error(f"Error in section categorization: {str(e)}", exc_info=True)
            raise
    
    def _generate_distinct_colors(self, n_colors: int) -> List[tuple]:
        """
        Generate n distinct colors using HSV color space.
        
        Args:
            n_colors: Number of distinct colors to generate
            
        Returns:
            List of RGBA color tuples
        """
        colors = []
        for i in range(n_colors):
            # Use golden ratio to generate distinct hues
            golden_ratio = 0.618033988749895
            hue = (i * golden_ratio) % 1.0
            saturation = 0.7  # Increased from 0.5 for more vibrant colors
            value = 0.95
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to RGBA with 0.5 alpha for better visibility (increased from 0.3)
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 128))
        
        return colors
    
    def _extract_information(self, categorized_sections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract relevant information with context awareness.
        
        Args:
            categorized_sections: Dictionary containing categorized sections
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            extracted_info = {
                "dates": [],
                "parties": [],
                "addresses": [],
                "amounts": [],
                "references": [],
                "sections": [],
                "contextual_info": {}
            }
            
            # Process each section type
            for section_type, section_data in categorized_sections["section_highlights"].items():
                items = section_data["items"]
                
                # Extract information based on section type
                for item in items:
                    text = item["text"]
                    
                    # Extract dates
                    dates = re.findall(self.patterns["date"], text)
                    extracted_info["dates"].extend(dates)
                    
                    # Extract parties
                    parties = re.findall(self.patterns["party"], text)
                    extracted_info["parties"].extend(parties)
                    
                    # Extract addresses
                    addresses = re.findall(self.patterns["address"], text)
                    extracted_info["addresses"].extend(addresses)
                    
                    # Extract amounts
                    amounts = re.findall(self.patterns["amount"], text)
                    extracted_info["amounts"].extend(amounts)
                    
                    # Extract references
                    references = re.findall(self.patterns["reference"], text)
                    extracted_info["references"].extend(references)
                    
                    # Extract sections
                    sections = re.findall(self.patterns["section"], text)
                    extracted_info["sections"].extend(sections)
                    
                    # Store contextual information
                    if section_type not in extracted_info["contextual_info"]:
                        extracted_info["contextual_info"][section_type] = []
                    
                    extracted_info["contextual_info"][section_type].append({
                        "text": text,
                        "page": item["page"],
                        "box": item["box"]
                    })
            
            # Remove duplicates while preserving order
            for key in ["dates", "parties", "addresses", "amounts", "references", "sections"]:
                extracted_info[key] = list(dict.fromkeys(extracted_info[key]))
            
            # Add metadata
            extracted_info["metadata"] = {
                "total_pages": len(categorized_sections["pages"]),
                "section_types": list(categorized_sections["section_highlights"].keys()),
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error in information extraction: {str(e)}", exc_info=True)
            raise
    
    def _generate_summary(self, extracted_info: Dict[str, Any]) -> str:
        """
        Generate a summary of the document.
        
        Args:
            extracted_info: Dictionary containing extracted information
            
        Returns:
            Generated summary text
        """
        try:
            # Prepare text for summarization
            summary_text = []
            
            # Add key information from headers
            if "headers" in extracted_info["contextual_info"]:
                header_text = " ".join(item["text"] for item in extracted_info["contextual_info"]["headers"])
                summary_text.append(header_text)
            
            # Add main content from sections
            if "sections" in extracted_info["contextual_info"]:
                section_text = " ".join(item["text"] for item in extracted_info["contextual_info"]["sections"])
                summary_text.append(section_text)
            
            # Add key details
            if extracted_info["dates"]:
                summary_text.append(f"Key dates: {', '.join(extracted_info['dates'])}")
            if extracted_info["parties"]:
                summary_text.append(f"Parties involved: {', '.join(extracted_info['parties'])}")
            if extracted_info["amounts"]:
                summary_text.append(f"Key amounts: {', '.join(extracted_info['amounts'])}")
            
            # Combine all text
            full_text = " ".join(summary_text)
            
            # Limit text length for summarization
            max_length = 1024
            if len(full_text) > max_length:
                full_text = full_text[:max_length] + "..."
            
            # Generate summary
            summary = self.summarization_pipeline(
                full_text,
                max_length=150,
                min_length=40,
                do_sample=False,
                truncation=True
            )[0]["summary_text"]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}", exc_info=True)
            return "Error generating summary. Please try again." 