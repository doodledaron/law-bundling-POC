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
    Check the status of a document processing job
    """
    logger.info(f"Checking status for job ID: {job_id}")
    
    # Get the task ID from Redis
    if not redis_client:
        logger.error("Redis connection not available")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Queue system not available."},
            status_code=500
        )
    
    task_id = redis_client.get(f"job:{job_id}")
    
    if not task_id:
        logger.warning(f"Job ID not found: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Job not found or expired."},
            status_code=404
        )
    
    # Check the task status
    task = AsyncResult(task_id.decode('utf-8'), app=celery_app)
    status = task.status
    
    # If the task is successful, redirect to the result page
    if status == 'SUCCESS':
        logger.info(f"Job completed successfully: {job_id}")
        return RedirectResponse(url=f"/result/{job_id}", status_code=303)
    
    # If the task failed, show an error message
    if status == 'FAILURE':
        logger.error(f"Job failed: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Document processing failed. Please try again."},
            status_code=500
        )
    
    # If the task is still pending or running, show the status page
    logger.info(f"Job status for {job_id}: {status}")
    return templates.TemplateResponse(
        "status.html",
        {"request": request, "job_id": job_id, "status": status},
        status_code=200
    )

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def get_result(request: Request, job_id: str):
    """
    Get the result of a document processing job
    """
    logger.info(f"Getting result for job ID: {job_id}")
    
    # Get the task ID from Redis
    if not redis_client:
        logger.error("Redis connection not available")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Queue system not available."},
            status_code=500
        )
    
    task_id = redis_client.get(f"job:{job_id}")
    
    if not task_id:
        logger.warning(f"Job ID not found: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Job not found or expired."},
            status_code=404
        )
    
    # Get the task result
    task = AsyncResult(task_id.decode('utf-8'), app=celery_app)
    
    if task.status != 'SUCCESS':
        logger.warning(f"Job not completed: {job_id} (status: {task.status})")
        return RedirectResponse(url=f"/status/{job_id}", status_code=303)
    
    # Get the result
    result = task.result
    
    # Render the result page with the extracted information
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