"""
OCR processing tasks for legal documents.
Handles image and PDF processing for text extraction.
"""
import os
import cv2
import numpy as np
import redis
import paddleocr
from pdf2image import convert_from_bytes
from celery import shared_task
from celery.utils.log import get_task_logger

# Import utilities
from tasks.utils import (
    update_job_status, 
    get_timestamp, 
    preprocess_image,
    clean_text
)

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Initialize PaddleOCR - singleton pattern
_OCR_ENGINE = None

def get_ocr_engine():
    """Get or initialize OCR engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        logger.info("Initializing OCR engine")
        _OCR_ENGINE = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR_ENGINE

@shared_task(name='tasks.ocr.process_pdf')
def process_pdf(job_id, pdf_path):
    """
    Process a PDF file and extract text.
    
    Args:
        job_id: Unique job identifier
        pdf_path: Path to PDF file
        
    Returns:
        dict: OCR results
    """
    try:
        # Get OCR engine
        ocr = get_ocr_engine()
        
        # Convert PDF to images
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
            images = convert_from_bytes(pdf_content)
        
        full_text = ""
        confidence_scores = []
        
        # Process each page
        total_pages = len(images)
        logger.info(f"Processing PDF with {total_pages} pages")
        
        for i, image in enumerate(images):
            # Update progress
            update_job_status(redis_client, job_id, {
                'progress': int((i / total_pages) * 100),
                'message': f'Processing page {i+1}/{total_pages}',
                'updated_at': get_timestamp()
            })
            
            # Convert PIL image to OpenCV format
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            processed_img = preprocess_image(img)
            
            # Process with OCR
            try:
                result = ocr.ocr(processed_img, cls=True)
            except Exception as e:
                logger.error(f"OCR processing error on page {i+1}: {str(e)}")
                raise Exception(f"OCR processing error: {str(e)}")
            
            # Extract text and confidence
            page_text = ""
            for res in result:
                for line in res:
                    text = line[1][0]
                    conf = line[1][1]
                    page_text += text + " "
                    confidence_scores.append(conf)
            
            full_text += page_text + " "
            
            logger.debug(f"Completed processing page {i+1}/{total_pages}")
        
        # Clean text
        cleaned_text = clean_text(full_text)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        logger.info(f"PDF processing complete. Extracted {len(cleaned_text)} characters with {avg_confidence:.2%} confidence")
        
        return {
            'cleaned_text': cleaned_text,
            'average_confidence': avg_confidence,
            'num_pages': total_pages
        }
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

@shared_task(name='tasks.ocr.process_image')
def process_image(job_id, image_path):
    """
    Process an image file and extract text.
    
    Args:
        job_id: Unique job identifier
        image_path: Path to image file
        
    Returns:
        dict: OCR results
    """
    try:
        # Get OCR engine
        ocr = get_ocr_engine()
        
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError("Could not decode the image. Please try another file.")
        
        # Update progress
        update_job_status(redis_client, job_id, {
            'progress': 50,
            'message': 'Processing image',
            'updated_at': get_timestamp()
        })
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Process with OCR
        try:
            result = ocr.ocr(processed_img, cls=True)
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            raise Exception(f"OCR error: {str(e)}")
        
        # Extract text and confidence
        full_text = ""
        confidence_scores = []
        
        for res in result:
            for line in res:
                text = line[1][0]
                conf = line[1][1]
                full_text += text + " "
                confidence_scores.append(conf)
        
        # Clean text
        cleaned_text = clean_text(full_text)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        logger.info(f"Image processing complete. Extracted {len(cleaned_text)} characters with {avg_confidence:.2%} confidence")
        
        return {
            'cleaned_text': cleaned_text,
            'average_confidence': avg_confidence,
            'num_pages': 1
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise