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
from yolo.document_layout import DocumentLayoutAnalyzer, DocumentRegion
import json

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

# Initialize PaddleOCR
logger.info("Initializing PaddleOCR...")
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")

# Initialize YOLOv8 document layout analyzer
logger.info("Initializing YOLOv8 document layout analyzer...")

# Check if we have a specialized legal document model
legal_model_path = os.path.join('models', 'legal_layout_model.pt')
if os.path.exists(legal_model_path):
    logger.info(f"Using specialized legal document layout model: {legal_model_path}")
    layout_analyzer = DocumentLayoutAnalyzer(model_path=legal_model_path, confidence_threshold=0.25)
else:
    logger.info("Specialized legal document model not found, using default model")
    layout_analyzer = DocumentLayoutAnalyzer(confidence_threshold=0.25)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount results directory for accessing processed documents
app.mount("/results", StaticFiles(directory="results"), name="results")

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
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

# Define root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main upload page
    """
    logger.info("Main page accessed")
    return templates.TemplateResponse("index.html", {"request": request})


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
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )

        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"File size exceeds limit: {len(contents)} bytes")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "File size exceeds 10MB limit."},
                status_code=400
            )

        if not contents:
            logger.warning("Empty file uploaded")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["empty_file"]},
                status_code=400
            )

        # Validate file type
        if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
            logger.warning(f"Invalid file type: {file.content_type}")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["invalid_type"]},
                status_code=400
            )

        # Convert file to image
        try:
            if file.content_type == "application/pdf":
                logger.info("Processing PDF file")
                # Use PyMuPDF to process PDF
                import io
                pdf_document = fitz.open(stream=contents, filetype="pdf")
                logger.info(f"PDF loaded with {len(pdf_document)} pages")
                
                full_text = ""
                confidence_scores = []
                
                for page_num in range(len(pdf_document)):
                    logger.info(f"Processing PDF page {page_num+1}/{len(pdf_document)}")
                    page = pdf_document.load_page(page_num)
                    
                    # Render page to an image (PyMuPDF uses RGB format)
                    pix = page.get_pixmap(alpha=False)
                    img_data = pix.tobytes("png")
                    
                    # Convert to numpy array for OpenCV
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        logger.warning(f"Failed to convert page {page_num+1} to image")
                        continue
                        
                    # Preprocess image
                    processed_img = preprocess_image(img)

                    # Process the image using PaddleOCR
                    try:
                        result = ocr.ocr(processed_img, cls=True)
                        logger.info(f"OCR completed for page {page_num+1}")
                    except Exception as e:
                        error_message = f"OCR processing error: {str(e)}"
                        logger.error(error_message, exc_info=True)
                        return templates.TemplateResponse(
                            "index.html",
                            {"request": request, "error": error_message},
                            status_code=500
                        )

                    # Concatenate the OCR results into a full text string
                    page_text = ""
                    for res in result:
                        for line in res:
                            page_text += line[1][0] + " "
                            confidence_scores.append(line[1][1])
                    
                    logger.info(f"Extracted {len(page_text.split())} words from page {page_num+1}")
                    full_text += page_text
                
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
            else:
                logger.info(f"Processing image file: {file.content_type}")
                # Convert image to OpenCV format
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    error_message = f"Image decode error: Could not decode the image. Please try another file."
                    logger.error(error_message)
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "error": error_message},
                        status_code=400
                    )

                # Process single image
                processed_img = preprocess_image(img)
                try:
                    result = ocr.ocr(processed_img, cls=True)
                    logger.info("OCR completed for image")
                except Exception as e:
                    logger.error(f"OCR processing error: {str(e)}", exc_info=True)
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "error": ERROR_MESSAGES["ocr_error"]},
                        status_code=500
                    )

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

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Error processing the document: {str(e)}"},
                status_code=500
            )

        # Extract NDA fields
        logger.info("Extracting NDA fields from text")
        fields = extract_nda_fields(cleaned_text)
        logger.info(f"Extracted fields: {', '.join(fields.keys())}")

        # Generate summary
        summary = generate_summary(cleaned_text)
        logger.info("Text summary generated")

        # Force flush logs before returning
        logger.info("Processing complete, preparing response")
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()

        # Render template with results
        logger.info("Rendering results template")
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "full_text": cleaned_text,
                "average_confidence_formatted": average_confidence_formatted,
                "summary": summary,
                **fields
            },
        )

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message, exc_info=True)
        # Force flush logs
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()
        return templates.TemplateResponse(
            "index.html",
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

@app.post("/region_ocr", response_class=HTMLResponse)
async def region_ocr(request: Request, file: UploadFile = File(...)):
    """
    Process uploaded file using region-based OCR with YOLOv8
    """
    logger.info(f"Region-based OCR initiated for file: {file.filename} ({file.content_type})")
    try:
        # Read file contents
        try:
            contents = await file.read()
            logger.info(f"File read successfully: {len(contents)} bytes")
        except Exception as e:
            error_message = f"Error reading file: {str(e)}"
            logger.error(error_message, exc_info=True)
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )

        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"File size exceeds limit: {len(contents)} bytes")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "File size exceeds 10MB limit."},
                status_code=400
            )

        if not contents:
            logger.warning("Empty file uploaded")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["empty_file"]},
                status_code=400
            )

        # Validate file type
        if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
            logger.warning(f"Invalid file type: {file.content_type}")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["invalid_type"]},
                status_code=400
            )

        # Create a unique ID for this document processing job
        import uuid
        job_id = str(uuid.uuid4())
        results_dir = os.path.join("results", job_id)
        os.makedirs(results_dir, exist_ok=True)
        
        document_map = {"pages": []}

        # Process file based on type
        try:
            if file.content_type == "application/pdf":
                logger.info("Processing PDF file with region detection")
                # Use PyMuPDF to process PDF
                import io
                pdf_document = fitz.open(stream=contents, filetype="pdf")
                logger.info(f"PDF loaded with {len(pdf_document)} pages")
                
                for page_num in range(len(pdf_document)):
                    logger.info(f"Processing PDF page {page_num+1}/{len(pdf_document)}")
                    page = pdf_document.load_page(page_num)
                    
                    # Render page to an image (PyMuPDF uses RGB format)
                    pix = page.get_pixmap(alpha=False)
                    img_data = pix.tobytes("png")
                    
                    # Convert to numpy array for OpenCV
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        logger.warning(f"Failed to convert page {page_num+1} to image")
                        continue

                    # Step 1: Detect document regions using YOLOv8
                    regions = layout_analyzer.detect_regions(img)
                    logger.info(f"Detected {len(regions)} regions on page {page_num+1}")
                    
                    # Step 2: Get region images
                    region_images = layout_analyzer.get_region_images(img, regions)
                    
                    # Step 3: Perform OCR on each region
                    for i, region in enumerate(regions):
                        region_img = region_images[i]
                        
                        # Skip tiny regions
                        if region_img.shape[0] < 10 or region_img.shape[1] < 10:
                            logger.warning(f"Skipping tiny region {i} on page {page_num+1}")
                            continue
                        
                        # Process region using PaddleOCR
                        try:
                            result = ocr.ocr(region_img, cls=True)
                            
                            # Extract text from region
                            region_text = ""
                            for res in result:
                                for line in res:
                                    region_text += line[1][0] + " "
                                    
                            # Store OCR text in region
                            regions[i].text = region_text.strip()
                            logger.info(f"OCR completed for region {i} on page {page_num+1}")
                            
                        except Exception as e:
                            logger.error(f"OCR error for region {i} on page {page_num+1}: {str(e)}", exc_info=True)
                    
                    # Step 4: Create visualization of regions
                    vis_image = layout_analyzer.visualize_regions(img, regions)
                    
                    # Save visualization
                    vis_path = os.path.join(results_dir, f"page_{page_num+1}_regions.jpg")
                    cv2.imwrite(vis_path, vis_image)
                    
                    # Step 5: Generate document map for this page
                    page_map = layout_analyzer.generate_document_map(regions)
                    page_map["page_number"] = page_num + 1
                    document_map["pages"].append(page_map)
                
                # Close PDF document
                pdf_document.close()
                
            else:
                logger.info(f"Processing image file with region detection: {file.content_type}")
                # Convert image to OpenCV format
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    error_message = "Image decode error: Could not decode the image. Please try another file."
                    logger.error(error_message)
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "error": error_message},
                        status_code=400
                    )

                # Step 1: Detect document regions using YOLOv8
                regions = layout_analyzer.detect_regions(img)
                logger.info(f"Detected {len(regions)} regions in image")
                
                # Step 2: Get region images
                region_images = layout_analyzer.get_region_images(img, regions)
                
                # Step 3: Perform OCR on each region
                for i, region in enumerate(regions):
                    region_img = region_images[i]
                    
                    # Skip tiny regions
                    if region_img.shape[0] < 10 or region_img.shape[1] < 10:
                        logger.warning(f"Skipping tiny region {i}")
                        continue
                    
                    # Process region using PaddleOCR
                    try:
                        result = ocr.ocr(region_img, cls=True)
                        
                        # Extract text from region
                        region_text = ""
                        for res in result:
                            for line in res:
                                region_text += line[1][0] + " "
                                
                        # Store OCR text in region
                        regions[i].text = region_text.strip()
                        logger.info(f"OCR completed for region {i}")
                        
                    except Exception as e:
                        logger.error(f"OCR error for region {i}: {str(e)}", exc_info=True)
                
                # Step 4: Create visualization of regions
                vis_image = layout_analyzer.visualize_regions(img, regions)
                
                # Save visualization
                vis_path = os.path.join(results_dir, "image_regions.jpg")
                cv2.imwrite(vis_path, vis_image)
                
                # Step 5: Generate document map
                page_map = layout_analyzer.generate_document_map(regions)
                page_map["page_number"] = 1
                document_map["pages"].append(page_map)
            
            # Save document map to JSON file
            doc_map_path = os.path.join(results_dir, "document_map.json")
            with open(doc_map_path, 'w') as f:
                json.dump(document_map, f, indent=2)
                
            logger.info(f"Document map saved to {doc_map_path}")
            
            # Return template with visualization and document map
            return templates.TemplateResponse(
                "region_results.html",
                {
                    "request": request,
                    "job_id": job_id,
                    "document_map": document_map,
                    "message": f"Document processed with {sum(len(p['regions']) for p in document_map['pages'])} regions detected",
                }
            )
            
        except Exception as e:
            error_message = f"Error processing document: {str(e)}"
            logger.error(error_message, exc_info=True)
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )
            
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(error_message, exc_info=True)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )
