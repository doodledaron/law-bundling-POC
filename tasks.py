import logging
import numpy as np
import cv2
import pymupdf  # PyMuPDF
import re
import paddleocr
from transformers import pipeline
from celery import Celery
import sys
import os
from logging.handlers import RotatingFileHandler

# Configure Celery
celery_app = Celery('tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Create a custom handler that flushes after each log
class FlushingHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Set up logging with proper flushing
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, "document_processor.log")

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        FlushingFileHandler(log_file, mode='a'),
        FlushingHandler(sys.stdout)
    ]
)

# Create logger with explicit handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger to avoid duplicate logs

# Add explicit handlers to our logger
file_handler = FlushingFileHandler(log_file, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = FlushingHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Initialize PaddleOCR
logger.info("Initializing PaddleOCR")
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
logger.info("PaddleOCR initialized")

def flush_logs():
    """Helper function to ensure logs are flushed"""
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()

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
            # Match company name
            'company': r'between:\s+(.*?)(?=\s*:?\s*\("Discloser")',
            # Match recipient name
            'recipient': r'and\.\s+(.*?)(?=\s*:\s*\("Recipient")',
            # Match company address
            'company_address': r'(?:business\s*at\s*)(.*?)(?:;)',
            # Match recipient address
            'recipient_address': r'(?:residing\s*at\s*)(.*?)(?:\.)',
            # Match both initial duration and survival period
            'duration': r'period\s+of\s+(.*?)\s+years.*?additional\s+(.*?)\s+years',
            # Match governing law
            'governing_law': r'governed by and construed in accordance with the laws of the\.?\s*([^.]+?)(?:\.|$)',
            # Match confidential information
            'confidential_info': r'information\s+relating\s+to\s+(.*?)(?=\s*\(the "Confidential Information"\))',
            # Match dates
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
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                fields[field] = match.group(1).strip() if match and match.groups() else "Not found"
                # Clean up extra spaces and punctuation
                if fields[field] != "Not found":
                    fields[field] = re.sub(r'\s+', ' ', fields[field]).strip(' .,;:')
                    logger.debug(f"Found {field}: {fields[field]}")

        # Post-processing for governing law to ensure complete phrase
        if fields['governing_law'] != "Not found":
            fields['governing_law'] = "laws of the " + fields['governing_law']

        logger.info(f"Field extraction complete. Found {sum(1 for v in fields.values() if v != 'Not found' and (not isinstance(v, list) or len(v) > 0))} fields")
        flush_logs()
        return fields

    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}", exc_info=True)
        flush_logs()
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
    model_name = "facebook/bart-large-xsum"
    logger.info(f"Using summarization model: {model_name}")
    flush_logs()
    
    try:
        summarizer = pipeline("summarization", model=model_name)
        # Limit input text to prevent model overload
        text = ' '.join(text.split()[:512])
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False, truncation=True)
        logger.info("Summary generated successfully")
        flush_logs()
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        flush_logs()
        return "Failed to generate summary"

@celery_app.task(name="process_document")
def process_document_task(file_content, file_content_type, file_name):
    """
    Process a document and extract information
    """
    logger.info(f"Processing document: {file_name} ({file_content_type})")
    flush_logs()
    
    try:
        # Convert file to image and process it
        if file_content_type == "application/pdf":
            logger.info("Processing PDF file")
            flush_logs()
            
            # Use PyMuPDF to process PDF
            pdf_document = pymupdf.open(stream=file_content, filetype="pdf")
            logger.info(f"PDF loaded with {len(pdf_document)} pages")
            flush_logs()
            
            full_text = ""
            confidence_scores = []
            
            for page_num in range(len(pdf_document)):
                logger.info(f"Processing PDF page {page_num+1}/{len(pdf_document)}")
                flush_logs()
                
                page = pdf_document.load_page(page_num)
                
                # Render page to an image
                pix = page.get_pixmap(alpha=False)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array for OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.warning(f"Failed to convert page {page_num+1} to image")
                    flush_logs()
                    continue
                    
                # Preprocess image
                processed_img = preprocess_image(img)

                # Process the image using PaddleOCR
                result = ocr.ocr(processed_img, cls=True)
                logger.info(f"OCR completed for page {page_num+1}")
                flush_logs()

                # Concatenate the OCR results into a full text string
                page_text = ""
                for res in result:
                    for line in res:
                        page_text += line[1][0] + " "
                        confidence_scores.append(line[1][1])
                
                logger.info(f"Extracted {len(page_text.split())} words from page {page_num+1}")
                full_text += page_text
                flush_logs()
            
            # Close the document
            pdf_document.close()
            
            cleaned_text = clean_text(full_text)
            if confidence_scores:
                average_confidence = sum(confidence_scores) / len(confidence_scores)
                average_confidence_formatted = f"{average_confidence:.2%}"
                logger.info(f"Average OCR confidence: {average_confidence_formatted}")
            else:
                average_confidence_formatted = "N/A"
                logger.warning("No confidence scores available")
            flush_logs()
            
        else:
            logger.info(f"Processing image file: {file_content_type}")
            flush_logs()
            
            # Convert image to OpenCV format
            nparr = np.frombuffer(file_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                error_message = "Image decode error: Could not decode the image."
                logger.error(error_message)
                flush_logs()
                raise ValueError(error_message)

            # Process single image
            processed_img = preprocess_image(img)
            result = ocr.ocr(processed_img, cls=True)
            logger.info("OCR completed for image")
            flush_logs()

            # Extract text and confidence scores
            full_text = ""
            confidence_scores = []
            for res in result:
                for line in res:
                    full_text += line[1][0] + " "
                    confidence_scores.append(line[1][1])
            
            logger.info(f"Extracted {len(full_text.split())} words from image")
            cleaned_text = clean_text(full_text)
            
            if confidence_scores:
                average_confidence = sum(confidence_scores) / len(confidence_scores)
                average_confidence_formatted = f"{average_confidence:.2%}"
                logger.info(f"Average OCR confidence: {average_confidence_formatted}")
            else:
                average_confidence_formatted = "N/A"
                logger.warning("No confidence scores available")
            flush_logs()

        # Extract NDA fields
        logger.info("Extracting NDA fields from text")
        fields = extract_nda_fields(cleaned_text)
        logger.info(f"Extracted fields: {', '.join(fields.keys())}")
        flush_logs()

        # Generate summary
        logger.info("Generating text summary")
        # summary = generate_summary(cleaned_text)
        logger.info("Text summary generated")
        flush_logs()

        # Prepare result
        result = {
            "full_text": cleaned_text,
            "average_confidence_formatted": average_confidence_formatted,
            "summary": 'summary',
            **fields
        }

        logger.info(f"Document {file_name} processed successfully")
        flush_logs()
        return result

    except Exception as e:
        logger.error(f"Error processing document {file_name}: {str(e)}", exc_info=True)
        flush_logs()
        raise