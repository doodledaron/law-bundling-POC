# Install dependencies: pip install fastapi uvicorn jinja2 paddleocr opencv-python numpy pymupdf transformers

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import traceback
import uuid
import shutil
import base64
from layoutlm.pipeline import DocumentProcessor
from io import BytesIO

# Try to import Summarizer, but make it optional
try:
    from models.summarizer import Summarizer
    summarizer_available = True
except ImportError:
    summarizer_available = False
    print("Warning: Summarizer module not available. Summaries will be disabled.")

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

# Initialize FastAPI app
app = FastAPI(title="Law Document Processing API", version="0.1.0")

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

# Initialize document processor
try:
    logger.info("Initializing Document Processor")
    document_processor = DocumentProcessor()
    logger.info("Document Processor initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Document Processor: {str(e)}", exc_info=True)
    document_processor = None

# Create directories for uploads and processed files if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mount static files
app.mount("/processed", StaticFiles(directory="processed"), name="processed")

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
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
    Process uploaded file and extract NDA information using new pipeline
    """
    logger.info(f"File upload initiated: {file.filename} ({file.content_type})")
    try:
        # Validate document processor availability
        if not document_processor:
            error_message = "Document processor is not available. Please check system configuration."
            logger.error(error_message)
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )
        
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

        # Process document using the new pipeline
        try:
            logger.info("Processing document using new pipeline")
            processing_results = document_processor.process_document(contents)
            
            # Extract results for template
            ocr_results = processing_results["ocr_results"]
            extracted_info = processing_results["extracted_info"]
            summary = processing_results["summary"]
            
            # Calculate average confidence score
            confidence_scores = []
            for page in ocr_results:
                for result in page["results"]:
                    confidence_scores.append(result["confidence"])
            
            if confidence_scores:
                average_confidence = sum(confidence_scores) / len(confidence_scores)
                average_confidence_formatted = f"{average_confidence:.2%}"
            else:
                average_confidence_formatted = "N/A"
            
            # Get full text from extracted info
            full_text = ""
            if "sections" in extracted_info["contextual_info"]:
                section_texts = [item["text"] for item in extracted_info["contextual_info"]["sections"]]
                full_text = " ".join(section_texts)
            
            # Create field mapping for compatibility with template
            fields = {
                "company": ", ".join(extracted_info["parties"][:1]) if extracted_info["parties"] else "Not found",
                "recipient": ", ".join(extracted_info["parties"][1:2]) if len(extracted_info["parties"]) > 1 else "Not found",
                "company_address": ", ".join(extracted_info["addresses"][:1]) if extracted_info["addresses"] else "Not found",
                "recipient_address": ", ".join(extracted_info["addresses"][1:2]) if len(extracted_info["addresses"]) > 1 else "Not found",
                "duration": "Not found",  # This would need more specific extraction logic
                "governing_law": "Not found",  # This would need more specific extraction logic
                "confidential_info": "Not found",  # This would need more specific extraction logic
                "dates": extracted_info["dates"] if extracted_info["dates"] else []
            }
            
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
                    "full_text": full_text,
                    "average_confidence_formatted": average_confidence_formatted,
                    "summary": summary,
                    **fields
                },
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

# Endpoint for processing with the new pipeline (same functionality as /upload but with JSON response)
@app.post("/process-document/")
async def process_document(
    file: UploadFile = File(...),
    extract_entities: bool = Form(True),
    summarize: bool = Form(True),
):
    """
    Process a legal document file (PDF, image) using the new pipeline.
    
    Args:
        file: The document file to process
        extract_entities: Whether to extract entities from the document
        summarize: Whether to generate a summary of the document
        
    Returns:
        JSON with processing results, including extracted entities and visualizations
    """
    logger.info(f"Received file: {file.filename}")
    
    try:
        # Validate file
        file_extension = os.path.splitext(file.filename)[1].lower()
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
        
        if file_extension not in supported_extensions:
            logger.error(f"Unsupported file type: {file_extension}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_extensions)}"},
            )
        
        # Check if document processor is available
        if not document_processor:
            return JSONResponse(
                status_code=500,
                content={"error": "Document processor is not available. Please check the system configuration."},
            )
        
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Create directory for processed results
        output_dir = os.path.join("processed", job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save file to uploads directory
        file_path = os.path.join("uploads", f"{job_id}{file_extension}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {file_path}")
        
        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Process document with new pipeline
        try:
            logger.info("Processing document with new pipeline...")
            processing_results = document_processor.process_document(file_content)
            
            # Extract key information from results
            ocr_results = processing_results["ocr_results"]
            layout_analysis = processing_results["layout_analysis"]
            categorized_sections = processing_results["categorized_sections"]
            extracted_info = processing_results["extracted_info"]
            summary = processing_results["summary"] if summarize else None
            
            # Save visualizations to output directory
            visualizations = []
            try:
                logger.info("Saving visualization images...")
                for page_idx, page in enumerate(categorized_sections["pages"]):
                    highlight_image = page["highlight_image"]
                    image_path = os.path.join(output_dir, f"page_{page_idx+1}.png")
                    highlight_image.save(image_path)
                    
                    # Convert image to base64 for response
                    with open(image_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Get OCR text for this page
                    page_ocr_text = []
                    if page_idx < len(ocr_results):
                        for result in ocr_results[page_idx]["results"]:
                            # Add text with its position and confidence
                            box = result["box"]
                            page_ocr_text.append({
                                "text": result["text"],
                                "confidence": round(float(result["confidence"]), 2),
                                "position": box[:2]  # Just take the top-left coordinate
                            })
                    
                    visualizations.append({
                        "page": page_idx+1,
                        "url": f"/processed/{job_id}/page_{page_idx+1}.png",
                        "image_base64": img_data,
                        "ocr_text": page_ocr_text
                    })
                logger.info(f"Created {len(visualizations)} visualizations")
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Get formatted entities for response
            formatted_entities = {}
            for key, values in extracted_info.items():
                if key != "metadata" and key != "contextual_info":
                    formatted_entities[key] = values
            
            # Prepare full text from sections
            full_text = ""
            if "sections" in extracted_info["contextual_info"]:
                section_texts = [item["text"] for item in extracted_info["contextual_info"]["sections"]]
                full_text = " ".join(section_texts)
            
            # Prepare results
            result = {
                "job_id": job_id,
                "filename": file.filename,
                "text_length": len(full_text),
                "processed_pages": len(ocr_results),
                "summary": summary,
                "entities": formatted_entities,
                "visualizations": visualizations,
                "metadata": extracted_info["metadata"]
            }
            
            logger.info(f"Document processing complete for {file.filename}")
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Document processing failed: {str(e)}"},
            )
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"},
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

# Add back LayoutLM endpoint for backward compatibility
@app.post("/process_with_layoutlm", response_class=HTMLResponse)
async def process_with_layoutlm(request: Request, file: UploadFile = File(...)):
    """
    Process a document using the new DocumentProcessor pipeline for backward compatibility.
    """
    logger.info(f"LayoutLM processing initiated for file: {file.filename} ({file.content_type})")

    # Check if document processor is available
    if not document_processor:
        error_message = "Document processor is not available. Please check system configuration."
        logger.error(error_message)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )

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
                {"request": request, "error": "File size exceeds the 10MB limit."},
                status_code=400
            )

        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        content_type = file.content_type.lower()
        
        if file_extension not in ['pdf', 'jpg', 'jpeg', 'png'] or not any(typ in content_type for typ in ['pdf', 'image']):
            logger.warning(f"Invalid file type: {content_type}, extension: {file_extension}")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["invalid_type"]},
                status_code=400
            )

        # Process the document using the new document processor
        try:
            # Process document
            logger.info("Processing document with new pipeline")
            processing_results = document_processor.process_document(contents)
            
            # Extract key information
            ocr_results = processing_results["ocr_results"]
            layout_analysis = processing_results["layout_analysis"]
            categorized_sections = processing_results["categorized_sections"]
            extracted_info = processing_results["extracted_info"]
            summary = processing_results["summary"]
            
            # Prepare visualizations with OCR text
            visualizations = []
            for page_idx, page in enumerate(categorized_sections["pages"]):
                # Convert highlight image to base64
                buffer = BytesIO()
                page["highlight_image"].save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Get OCR text for this page
                page_ocr_text = []
                if page_idx < len(ocr_results):
                    for result in ocr_results[page_idx]["results"]:
                        # Add text with its position and confidence
                        box = result["box"]
                        page_ocr_text.append({
                            "text": result["text"],
                            "confidence": f"{result['confidence']:.2%}",
                            "position": f"({box[0]:.0f}, {box[1]:.0f})"
                        })
                
                visualizations.append({
                    "page": page_idx + 1,
                    "image_base64": img_str,
                    "ocr_text": page_ocr_text
                })
            
            # Format entities for the template
            formatted_entities = {}
            for key, values in extracted_info.items():
                if key != "metadata" and key != "contextual_info":
                    formatted_entities[key] = values
            
            # Get full text
            full_text = ""
            if "sections" in extracted_info["contextual_info"]:
                section_texts = [item["text"] for item in extracted_info["contextual_info"]["sections"]]
                full_text = " ".join(section_texts)
            
            # Combine all results
            results = {
                "filename": file.filename,
                "file_type": content_type,
                "processed_pages": len(ocr_results),
                "extracted_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text,
                "summary": summary,
                "entities": formatted_entities,
                "visualizations": visualizations
            }
            
            logger.info(f"Document processing successful for {file.filename}")
            
            return templates.TemplateResponse(
                "layoutlm_results.html",
                {"request": request, "results": results},
                status_code=200
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

if __name__ == "__main__":
    logger.info("Starting Law Document Processing API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
