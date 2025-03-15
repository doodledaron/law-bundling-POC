import paddleocr
import cv2
import numpy as np
import re
import pymupdf 
from transformers import pipeline
import logging
import os
import sys
from datetime import datetime
from celery_worker import celery_app
import pickle
import io

# Set up task-specific logger
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"tasks_{datetime.now().strftime('%Y%m%d')}.log")

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure basic logging to ensure all log handlers are initialized
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a file handler with immediate flush capability
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

flush_handler = FlushingFileHandler(log_filename, mode='a')
flush_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(flush_handler)

# Test the logger to verify it's working
logger.info("===== Tasks module initialized =====")
logger.info(f"Logging to file: {log_filename}")

# Initialize OCR and summarization models globally to reuse them
# This will be initialized once per worker
ocr = None
summarizer = None

def get_ocr():
    global ocr
    if ocr is None:
        logger.info("Initializing PaddleOCR in worker")
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
        logger.info("PaddleOCR initialization complete")
    return ocr

def get_summarizer():
    global summarizer
    if summarizer is None:
        logger.info("Initializing summarization model in worker")
        model_name = "facebook/bart-large-xsum"
        summarizer = pipeline("summarization", model=model_name)
        logger.info("Summarization model initialization complete")
    return summarizer

def preprocess_image(img):
    """
    Preprocess the image for better OCR results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get rid of the noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    return denoised

def clean_text(text):
    """
    Clean the extracted text for better regex matching
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters that might interfere with regex
    text = text.replace('\n', ' ').replace('\r', '')
    return text

def extract_nda_fields(text):
    """
    Extract relevant fields from NDA text using flexible regex patterns
    """
    logger.info("Extracting NDA fields from text")
    try:
        patterns = {
            # Match company name - fixed to remove trailing colon
            'company': r'between:\s+(.*?)(?=\s*:?\s*\("Discloser")',

            # Match recipient name - improved to get just the name
            'recipient': r'and\.\s+(.*?)(?=\s*:\s*\("Recipient")',

            # Match company address - unchanged as it works correctly
            'company_address': r'(?:business\s*at\s*)(.*?)(?:;)',

            # Match recipient address - unchanged as it works correctly
            'recipient_address': r'(?:residing\s*at\s*)(.*?)(?:\.)',

            # Match both initial duration and survival period
            'duration': r'period\s+of\s+(.*?)\s+years.*?additional\s+(.*?)\s+years',

            # Match governing law - fixed to capture full state law reference
            'governing_law': r'governed by and construed in accordance with the laws of the\.?\s*([^.]+?)(?:\.|$)',

            # Match confidential information - improved to capture full scope
            'confidential_info': r'information\s+relating\s+to\s+(.*?)(?=\s*\(the "Confidential Information"\))',

            # Match dates - improved format handling
            'dates': r'\b(?:February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
        }

        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            if field == 'dates':
                matches = re.findall(pattern, text, re.IGNORECASE)
                fields[field] = list(set(matches)) if matches else []  # Remove duplicates
                logger.debug(f"Found dates: {fields[field]}")
            elif field == 'duration':
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and match.groups():
                    initial_term = match.group(1).strip()
                    survival_period = match.group(2).strip()
                    fields[field] = f"{initial_term} years with {survival_period} years survival period"
                    logger.debug(f"Found duration: {fields[field]}")
                else:
                    fields[field] = "Not found"
                    logger.debug(f"Duration not found")
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                fields[field] = match.group(1).strip() if match and match.groups() else "Not found"
                # Clean up extra spaces and punctuation
                if fields[field] != "Not found":
                    fields[field] = re.sub(r'\s+', ' ', fields[field]).strip(' .,;:')  # Added ':' to strip
                    logger.debug(f"Found {field}: {fields[field]}")
                else:
                    logger.debug(f"{field} not found")

        # Post-processing for governing law to ensure complete phrase
        if fields['governing_law'] != "Not found":
            fields['governing_law'] = "laws of the " + fields['governing_law']

        logger.info(f"Field extraction complete. Found {sum(1 for v in fields.values() if v != 'Not found' and v)} fields")
        
        # Force flush logs
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()
            
        return fields

    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}", exc_info=True)
        # Force flush logs on error
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()
            
        return {
            'company': "Not found",
            'recipient': "Not found",
            'company_address': "Not found",
            'recipient_address': "Not found",
            'duration': "Not found",
            'governing_law': "Not found",
            'confidential_info': "Not found",
            'dates': []
        }

def generate_summary(text):
    """
    Generate a summary using a lightweight model.
    """
    logger.info("Generating summary of text")
    summarizer = get_summarizer()
    # Limit input text to prevent model overload
    text = ' '.join(text.split()[:512])  # Even shorter input for lightweight models
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False, truncation=True)
    logger.info("Summary generation complete")
    
    # Force flush logs
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
        
    return summary[0]['summary_text']

@celery_app.task(name="process_document", bind=True, max_retries=3)
def process_document(self, file_content, content_type, filename):
    """
    Process the document with OCR, extract fields, and generate summary
    """
    logger.info(f"Starting document processing task for {filename}")
    
    # Force flush initial log
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
    
    try:
        # Get OCR model instance
        ocr_model = get_ocr()
        
        # Process based on file type
        if content_type == "application/pdf":
            logger.info("Processing PDF file")
            pdf_document = pymupdf.open(stream=file_content, filetype="pdf")
            logger.info(f"PDF loaded with {len(pdf_document)} pages")
            
            # Force flush logs after PDF loading
            for handler in logger.handlers + logging.root.handlers:
                handler.flush()
            
            full_text = ""
            confidence_scores = []
            
            for page_num in range(len(pdf_document)):
                logger.info(f"Processing PDF page {page_num+1}/{len(pdf_document)}")
                page = pdf_document.load_page(page_num)
                
                # Render page to an image
                pix = page.get_pixmap(alpha=False)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array for OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.warning(f"Failed to convert page {page_num+1} to image")
                    # Force flush logs on warning
                    for handler in logger.handlers + logging.root.handlers:
                        handler.flush()
                    continue
                    
                # Preprocess image
                processed_img = preprocess_image(img)

                # Process the image using PaddleOCR
                try:
                    result = ocr_model.ocr(processed_img, cls=True)
                    logger.info(f"OCR completed for page {page_num+1}")
                except Exception as e:
                    logger.error(f"OCR processing error on page {page_num+1}: {str(e)}", exc_info=True)
                    # Force flush logs on error
                    for handler in logger.handlers + logging.root.handlers:
                        handler.flush()
                    raise e

                # Concatenate the OCR results into a full text string
                page_text = ""
                for res in result:
                    for line in res:
                        page_text += line[1][0] + " "
                        confidence_scores.append(line[1][1])
                
                logger.info(f"Extracted {len(page_text.split())} words from page {page_num+1}")
                full_text += page_text
                
                # Force flush logs after each page
                for handler in logger.handlers + logging.root.handlers:
                    handler.flush()
            
            # Close the document
            pdf_document.close()
            
        else:
            logger.info(f"Processing image file: {content_type}")
            # Convert image to OpenCV format
            nparr = np.frombuffer(file_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                error_message = "Image decode error: Could not decode the image."
                logger.error(error_message)
                # Force flush logs on error
                for handler in logger.handlers + logging.root.handlers:
                    handler.flush()
                raise ValueError(error_message)

            # Process single image
            processed_img = preprocess_image(img)
            try:
                result = ocr_model.ocr(processed_img, cls=True)
                logger.info("OCR completed for image")
            except Exception as e:
                logger.error(f"OCR processing error: {str(e)}", exc_info=True)
                # Force flush logs on error
                for handler in logger.handlers + logging.root.handlers:
                    handler.flush()
                raise e

            # Extract text and confidence scores
            full_text = ""
            confidence_scores = []
            for res in result:
                for line in res:
                    full_text += line[1][0] + " "
                    confidence_scores.append(line[1][1])
            
            logger.info(f"Extracted {len(full_text.split())} words from image")
            
            # Force flush logs after extraction
            for handler in logger.handlers + logging.root.handlers:
                handler.flush()
            
        # Clean the extracted text
        cleaned_text = clean_text(full_text)
        if confidence_scores:
            average_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            average_confidence = 0
            
        # Extract NDA fields and generate summary
        logger.info("Starting field extraction and summary generation")
        fields = extract_nda_fields(cleaned_text)
        summary = generate_summary(cleaned_text)
        
        logger.info("Task completed successfully")
        # Final log flush before returning results
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()
        
        # Return results
        return {
            "full_text": cleaned_text,
            "average_confidence": average_confidence,
            "summary": summary,
            **fields
        }
        
    except Exception as e:
        logger.error(f"Error in task process_document: {str(e)}", exc_info=True)
        # Force flush logs on error
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()
        
        # Retry the task with exponential backoff
        retry_countdown = 5 * (2 ** self.request.retries)  # 5s, 10s, 20s
        self.retry(exc=e, countdown=retry_countdown)