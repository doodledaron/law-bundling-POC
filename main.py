from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from datetime import datetime
import uuid
import redis
from celery.result import AsyncResult

# Import the Celery task
from tasks import celery_app, process_document_task

# ------ LOGGING CONFIGURATION ------
# Ensure log directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"app_{datetime.now().strftime('%Y%m%d')}.log")

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

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

# Initialize Redis client
try:
    redis_client = redis.Redis(host="redis", port=6379, db=0)
    redis_client.ping()  # Check connection
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Redis connection error: {str(e)}")
    redis_client = None

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
}

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
    Process uploaded file asynchronously and return job ID
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

        # Generate a unique ID for the job
        job_id = str(uuid.uuid4())
        
        # Queue the document processing task
        task = process_document_task.delay(contents, file.content_type, file.filename)
        
        # Store the task ID in Redis with a 24-hour expiration
        if redis_client:
            redis_client.setex(f"job:{job_id}", 86400, task.id)
            logger.info(f"Task ID stored in Redis with job ID: {job_id}")
        
        logger.info(f"Document processing queued with task ID: {task.id}")
        
        # Redirect to the status page
        return RedirectResponse(url=f"/status/{job_id}", status_code=303)

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

@app.get("/status/{job_id}", response_class=HTMLResponse)
async def check_status(request: Request, job_id: str):
    """
    Check the status of a document processing job, with support for chunked processing
    """
    logger.info(f"Checking status for job ID: {job_id}")
    
    # Get the job status from Redis
    if not redis_client:
        logger.error("Redis connection not available")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Queue system not available."},
            status_code=500
        )
    
    status = redis_client.get(f"status:{job_id}")
    
    if not status:
        logger.warning(f"Job ID not found: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Job not found or expired."},
            status_code=404
        )
    
    status = status.decode('utf-8')
    
    # If the job is successful, redirect to the result page
    if status == 'SUCCESS':
        logger.info(f"Job completed successfully: {job_id}")
        return RedirectResponse(url=f"/result/{job_id}", status_code=303)
    
    # If the job failed, show an error message
    if status == 'FAILED':
        logger.error(f"Job failed: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Document processing failed. Please try again."},
            status_code=500
        )
    
    # If the job is being processed in chunks, show progress
    if status == 'CHUNKED':
        total_chunks = int(redis_client.get(f"total_chunks:{job_id}").decode('utf-8'))
        
        # Count completed chunks
        completed_chunks = 0
        for i in range(total_chunks):
            chunk_status = redis_client.get(f"chunk_status:{job_id}:{i}")
            if chunk_status and chunk_status.decode('utf-8') == 'COMPLETED':
                completed_chunks += 1
        
        # Calculate progress percentage
        progress = int((completed_chunks / total_chunks) * 100)
        
        # Update the completed chunks counter
        redis_client.set(f"completed_chunks:{job_id}", str(completed_chunks))
        
        logger.info(f"Chunked job status: {completed_chunks}/{total_chunks} chunks completed ({progress}%)")
        
        # Show the status page with progress information
        return templates.TemplateResponse(
            "status.html",
            {
                "request": request, 
                "job_id": job_id, 
                "status": "PROCESSING", 
                "is_chunked": True,
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "progress": progress
            }
        )
    
    # For non-chunked jobs still in progress
    logger.info(f"Job status for {job_id}: {status}")
    return templates.TemplateResponse(
        "status.html",
        {"request": request, "job_id": job_id, "status": status, "is_chunked": False},
        status_code=200
    )

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def get_result(request: Request, job_id: str):
    """
    Get the result of a document processing job
    """
    logger.info(f"Getting result for job ID: {job_id}")
    
    # Get the job result from Redis
    if not redis_client:
        logger.error("Redis connection not available")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Queue system not available."},
            status_code=500
        )
    
    status = redis_client.get(f"status:{job_id}")
    
    if not status or status.decode('utf-8') != 'SUCCESS':
        logger.warning(f"Job not completed or not found: {job_id}")
        return RedirectResponse(url=f"/status/{job_id}", status_code=303)
    
    # Get the result data
    result_data = redis_client.get(f"result:{job_id}")
    
    if not result_data:
        logger.error(f"Result data not found for job: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Result data not found."},
            status_code=404
        )
    
    # Deserialize the result
    result = pickle.loads(result_data)
    
    # Render the result page
    logger.info(f"Rendering result for job ID: {job_id}")
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            **result
        },
        status_code=200
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