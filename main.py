# Install dependencies: pip install fastapi uvicorn jinja2 paddleocr opencv-python numpy pymupdf transformers

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import paddleocr
import cv2
import numpy as np
import re
import fitz  # PyMuPDF
from transformers import pipeline
import logging
import os
import sys
from datetime import datetime
import tempfile
import uuid
import shutil
from pathlib import Path
from layoutlmv3.inference import process_document
import json
import time

# ------ LOGGING CONFIGURATION ------
# Ensure log directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"app_{datetime.now().strftime('%Y%m%d')}.log")

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # 'a' for append mode
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)  # Use module name for the logger

# Add a file handler with immediate flush capability
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

flush_handler = FlushingFileHandler(log_filename, mode='a')
flush_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(flush_handler)

# Test the logger to verify it's working
logger.info("===== Application starting up =====")
logger.info(f"Logging to file: {log_filename}")
# ------ END LOGGING CONFIGURATION ------

def generate_summary(text):
    """
    Generate a summary using a lightweight model.
    """
    model_name = "facebook/bart-large-xsum"
    logger.info("Generating summary of text")
    logger.info(f"Using summarization model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)
    # Limit input text to prevent model overload
    text = ' '.join(text.split()[:512])  # Even shorter input for lightweight models
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False, truncation=True)
    logger.info("Summary generation complete")
    return summary[0]['summary_text']

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize PaddleOCR - using en_PP-OCRv3_det (3.8M) model for English text detection
logger.info("Initializing PaddleOCR")
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
logger.info("PaddleOCR initialized")

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file.",
    "file_too_large": "File size exceeds 10MB limit."
}

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
        return fields

    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}", exc_info=True)
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

@app.get("/test_logging")
async def test_logging():
    """
    Test endpoint to verify logging functionality
    """
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Force flush
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
        
    return {"message": "Logging test complete, check the log file"}

# Create directories for storing results
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Define layoutlmv3 model paths
LAYOUTLMV3_MODEL_DIR = "layoutlmv3/model/best_model"
LAYOUTLMV3_PROCESSOR_DIR = "layoutlmv3/model"

# Add a helper function to run the LayoutLMv3 model
async def run_layoutlmv3_model(pdf_path: str):
    """
    Run the LayoutLMv3 model on the provided PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Dict with model results.
    """
    logger.info(f"Running LayoutLMv3 model on {pdf_path}")
    
    try:
        # Handle filenames with spaces by creating a temporary copy with a clean name
        original_pdf_path = pdf_path
        temp_filename = f"temp_file_{uuid.uuid4()}.pdf"
        temp_pdf_path = os.path.join(os.path.dirname(pdf_path), temp_filename)
        
        try:
            shutil.copy2(original_pdf_path, temp_pdf_path)
            logger.info(f"Created temporary file {temp_pdf_path} to handle spaces in filename")
            pdf_path = temp_pdf_path
        except Exception as e:
            logger.warning(f"Failed to create temporary file: {str(e)}. Continuing with original path.")
        
        # Generate a unique results directory
        result_id = str(uuid.uuid4())
        result_dir = os.path.join(RESULTS_DIR, result_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Get device (CPU or CUDA)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create args object for process_document
        class Args:
            def __init__(self):
                self.pdf_path = pdf_path
                self.model_dir = LAYOUTLMV3_MODEL_DIR
                self.processor_dir = LAYOUTLMV3_PROCESSOR_DIR
                self.output_dir = result_dir
                self.device = device
                # Lower the confidence threshold to catch more potential clauses
                self.confidence_threshold = 0.3
                self.dpi = 300
                self.num_workers = 0
        
        args = Args()
        
        # Process the document using the layoutlmv3 inference
        logger.info(f"Starting document processing with confidence threshold: {args.confidence_threshold}")
        results = process_document(args)
        
        # Format results for the UI
        formatted_results = {
            "clauses": {},
            "image_path": None
        }
        
        # Find the annotated images in the results directory
        logger.info(f"Looking for images in {result_dir}")
        image_files = [f for f in os.listdir(result_dir) if f.endswith('_annotated.png') or f.endswith('_annotated.jpg') or f.endswith('.jpg')]
        logger.info(f"Found {len(image_files)} image files: {', '.join(image_files[:5])}{'...' if len(image_files) > 5 else ''}")
        
        if image_files:
            # Use the first image for display
            image_path = os.path.join("results", result_id, image_files[0])
            formatted_results["image_path"] = f"/static/{image_path}"
            logger.info(f"Using image path: {formatted_results['image_path']}")
        
        # Get the extracted clauses from the JSON file
        clauses_file = os.path.join(result_dir, "extracted_clauses.json")
        logger.info(f"Looking for clauses file: {clauses_file}")
        
        if os.path.exists(clauses_file):
            with open(clauses_file, 'r', encoding='utf-8') as f:
                extracted_clauses = json.load(f)
                logger.info(f"Loaded clauses from JSON: {len(extracted_clauses)} categories")
                
            formatted_results["clauses"] = {}
            for category, texts in extracted_clauses.items():
                # Skip empty categories
                if not texts:
                    continue
                
                logger.info(f"Processing category '{category}' with {len(texts)} clauses")
                formatted_results["clauses"][category] = []
                for item in texts:
                    if isinstance(item, dict):
                        formatted_results["clauses"][category].append(item)
                    else:
                        # If it's just text, add with default confidence
                        formatted_results["clauses"][category].append({
                            "text": item,
                            "confidence": 0.7
                        })
        else:
            logger.warning(f"No clauses file found at {clauses_file}")
            # Add dummy clause for debugging if no clauses were found
            # This is just to verify the UI works even if no actual clauses were extracted
            if not formatted_results["clauses"]:
                logger.info("Adding dummy debug clause for UI testing")
                formatted_results["clauses"]["DEBUG"] = [{
                    "text": "No clauses detected in document. This is a debug message to verify the UI works.",
                    "confidence": 1.0
                }]
        
        logger.info(f"LayoutLMv3 processing complete, found {len(formatted_results['clauses'])} clause categories")
        
        # Clean up temporary file if created
        if pdf_path != original_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.info(f"Removed temporary file {temp_pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error running LayoutLMv3 model: {str(e)}", exc_info=True)
        raise e

# Define the LayoutLMv3 page route
@app.get("/layoutlmv3", response_class=HTMLResponse)
async def layoutlmv3_page(request: Request):
    """
    Render the LayoutLMv3 model testing page
    """
    logger.info("LayoutLMv3 page accessed")
    return templates.TemplateResponse("layoutlmv3.html", {"request": request})

# Define the LayoutLMv3 analysis endpoint
@app.post("/layoutlmv3_analyze", response_class=HTMLResponse)
async def layoutlmv3_analyze(request: Request, file: UploadFile = File(...)):
    """
    Process a document with the LayoutLMv3 model to extract legal clauses
    """
    logger.info(f"LayoutLMv3 analysis initiated: {file.filename} ({file.content_type})")
    
    try:
        # Read file contents
        try:
            contents = await file.read()
            logger.info(f"File read successfully: {len(contents)} bytes")
        except Exception as e:
            error_message = f"Error reading file: {str(e)}"
            logger.error(error_message, exc_info=True)
            return templates.TemplateResponse(
                "layoutlmv3.html",
                {"request": request, "error": error_message},
                status_code=500
            )

        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"File size exceeds limit: {len(contents)} bytes")
            return templates.TemplateResponse(
                "layoutlmv3.html",
                {"request": request, "error": ERROR_MESSAGES["file_too_large"]},
                status_code=400
            )

        # Check file type (only PDF)
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.content_type}")
            return templates.TemplateResponse(
                "layoutlmv3.html",
                {"request": request, "error": "Only PDF files are supported."},
                status_code=400
            )

        # Save the uploaded file temporarily
        file_id = str(uuid.uuid4())
        pdf_path = os.path.join(UPLOADS_DIR, f"{file_id}.pdf")
        
        with open(pdf_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File saved to {pdf_path}")
        
        # Process the document with LayoutLMv3
        try:
            results = await run_layoutlmv3_model(pdf_path)
            logger.info("LayoutLMv3 analysis completed successfully")
            
            # Return the results
            return templates.TemplateResponse(
                "layoutlmv3.html",
                {"request": request, "results": results}
            )
            
        except Exception as e:
            error_message = f"Error processing document with LayoutLMv3: {str(e)}"
            logger.error(error_message, exc_info=True)
            return templates.TemplateResponse(
                "layoutlmv3.html",
                {"request": request, "error": error_message},
                status_code=500
            )
            
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(error_message, exc_info=True)
        return templates.TemplateResponse(
            "layoutlmv3.html",
            {"request": request, "error": error_message},
            status_code=500
        )
    finally:
        # Clean up temporary files older than 1 hour
        try:
            cleanup_old_files(UPLOADS_DIR, max_age_hours=1)
            cleanup_old_files(RESULTS_DIR, max_age_hours=24)  # Keep results longer
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

# Helper function to clean up old files
def cleanup_old_files(directory, max_age_hours=1):
    """Remove files older than the specified age in hours"""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            # Check file age
            file_age = current_time - os.path.getmtime(item_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(item_path)
                    logger.debug(f"Removed old file: {item_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {item_path}: {str(e)}")
        elif os.path.isdir(item_path):
            # Check directory age and remove recursively if old
            dir_age = current_time - os.path.getmtime(item_path)
            if dir_age > max_age_seconds:
                try:
                    shutil.rmtree(item_path)
                    logger.debug(f"Removed old directory: {item_path}")
                except Exception as e:
                    logger.error(f"Failed to remove directory {item_path}: {str(e)}")
            
# Update the home page to add a link to the LayoutLMv3 page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main upload page
    """
    logger.info("Main page accessed")
    
    # Add link to LayoutLMv3 page
    features = [
        {
            "name": "NDA Analysis",
            "description": "Upload and analyze Non-Disclosure Agreements to extract key information.",
            "url": "/upload",
            "icon": "📄"
        },
        {
            "name": "LayoutLMv3 Legal Clause Detection",
            "description": "Detect and extract legal clauses using our fine-tuned LayoutLMv3 model.",
            "url": "/layoutlmv3",
            "icon": "🔍"
        }
    ]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": features
    })

# Define upload page route
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """
    Render the NDA upload page
    """
    logger.info("Upload page accessed")
    return templates.TemplateResponse("upload.html", {"request": request})

# Define upload endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Process uploaded file and extract NDA information
    """
    logger.info(f"File upload initiated: {file.filename} ({file.content_type})")
    try:
        # Read file contents
        try:
            contents = await file.read()
            logger.info(f"File read successfully: {len(contents)} bytes")
        except Exception as e:
            error_message = f"Error reading file: {str(e)}"
            logger.error(error_message, exc_info=True)
            return templates.TemplateResponse(
                "upload.html",
                {"request": request, "error": error_message},
                status_code=500
            )

        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"File size exceeds limit: {len(contents)} bytes")
            return templates.TemplateResponse(
                "upload.html",
                {"request": request, "error": ERROR_MESSAGES["file_too_large"]},
                status_code=400
            )

        # Check if a file was uploaded
        if not contents:
            logger.warning("Empty file")
            return templates.TemplateResponse(
                "upload.html",
                {"request": request, "error": ERROR_MESSAGES["empty_file"]},
                status_code=400
            )

        # Create a temporary directory for this upload
        logger.info("Creating temporary directory for file processing")
        upload_id = str(uuid.uuid4())
        temp_dir = os.path.join(UPLOADS_DIR, upload_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract text with OCR
        full_text = ""
        confidence_scores = []
        
        # Determine file type
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext in ['.pdf']:
            # For PDFs, save as is
            temp_file_path = os.path.join(temp_dir, f"upload{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            logger.info(f"Saved PDF to {temp_file_path}")
            
            # Convert PDF to images
            logger.info("Converting PDF to images")
            try:
                images = []
                pdf_doc = fitz.open(temp_file_path)
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                    pix.save(img_path)
                    images.append(img_path)
                    logger.info(f"Saved page {page_num+1} to {img_path}")
                
                pdf_doc.close()
                
                if not images:
                    logger.warning("No images extracted from PDF")
                    return templates.TemplateResponse(
                        "upload.html",
                        {"request": request, "error": "Could not extract any pages from the PDF."},
                        status_code=400
                    )
                
                # Process each page with OCR
                for img_path in images:
                    image = cv2.imread(img_path)
                    processed_img = preprocess_image(image)
                    
                    # Perform OCR
                    result = ocr.ocr(processed_img, cls=True)
                    logger.info(f"OCR completed for {img_path}")
                    
                    # Extract text and confidence scores
                    page_text = ""
                    for res in result:
                        for line in res:
                            page_text += line[1][0] + " "
                            confidence_scores.append(line[1][1])
                    
                    full_text += page_text + " "
                
            except Exception as e:
                error_message = f"Error processing PDF: {str(e)}"
                logger.error(error_message, exc_info=True)
                return templates.TemplateResponse(
                    "upload.html",
                    {"request": request, "error": error_message},
                    status_code=500
                )
                
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            # For images, save and load directly
            temp_file_path = os.path.join(temp_dir, f"upload{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            logger.info(f"Saved image to {temp_file_path}")
            
            try:
                image = cv2.imread(temp_file_path)
                if image is None:
                    logger.error(f"Failed to decode image: {temp_file_path}")
                    return templates.TemplateResponse(
                        "upload.html",
                        {"request": request, "error": ERROR_MESSAGES["decode_error"]},
                        status_code=400
                    )

                # Preprocess image
                processed_img = preprocess_image(image)
                
                # Perform OCR
                result = ocr.ocr(processed_img, cls=True)
                logger.info(f"OCR completed for {temp_file_path}")
                
                # Extract text and confidence scores
                for res in result:
                    for line in res:
                        full_text += line[1][0] + " "
                        confidence_scores.append(line[1][1])
                
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                logger.error(error_message, exc_info=True)
                return templates.TemplateResponse(
                    "upload.html",
                    {"request": request, "error": error_message},
                    status_code=500
                )
        else:
            logger.warning(f"Invalid file type: {file_ext}")
            return templates.TemplateResponse(
                "upload.html",
                {"request": request, "error": ERROR_MESSAGES["invalid_type"]},
                status_code=400
            )
        
        # Clean text
        cleaned_text = clean_text(full_text)
        logger.info(f"Extracted {len(cleaned_text.split())} words total")
        
        # Calculate average confidence
        if confidence_scores:
            average_confidence = sum(confidence_scores) / len(confidence_scores)
            average_confidence_formatted = f"{average_confidence:.2%}"
            logger.info(f"Average OCR confidence: {average_confidence_formatted}")
        else:
            average_confidence_formatted = "N/A"
            logger.warning("No confidence scores available")

        # Extract NDA fields
        logger.info("Extracting NDA fields from text")
        fields = extract_nda_fields(cleaned_text)
        logger.info(f"Extracted fields: {', '.join(fields.keys())}")

        # Generate summary
        summary = generate_summary(cleaned_text)
        logger.info("Text summary generated")

        logger.info("Document processing complete")
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "full_text": cleaned_text,
                "average_confidence": average_confidence_formatted,
                "summary": summary,
                **fields
            }
        )

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(error_message, exc_info=True)
        # Force flush logs
        for handler in logger.handlers:
            handler.flush()
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "error": error_message},
            status_code=500
        )

# Debug handler for uvicorn to force flush logs after each request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    # Force flush logs after each request
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
    return response
