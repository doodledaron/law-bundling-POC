from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from datetime import datetime
import uuid
from celery.result import AsyncResult
from celery_worker import celery_app
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import Redis

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

# Set up static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Connect to Redis
redis_client = Redis(host='redis', port=6379, db=2)

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
}

@app.on_event("startup")
async def startup():
    # Initialize FastAPI cache with Redis backend
    redis = Redis(host="redis", port=6379, db=2)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache:")
    logger.info("FastAPI Cache initialized with Redis backend")

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
    Process uploaded file using Celery task
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

        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Store task ID in Redis with initial status
        redis_client.set(f"task:{task_id}:status", "PENDING")
        
        # Submit the document processing task to Celery
        task = celery_app.send_task(
            "process_document",
            args=[contents, file.content_type, file.filename],
            task_id=task_id
        )
        
        logger.info(f"Celery task submitted with ID: {task_id}")
        
        # Redirect to the results page with the task ID
        return RedirectResponse(url=f"/results/{task_id}", status_code=303)
        
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

@app.get("/results/{task_id}", response_class=HTMLResponse)
async def get_results(request: Request, task_id: str):
    """
    Check the status of a task and display results when ready
    """
    logger.info(f"Checking results for task: {task_id}")
    
    # Get task result
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.ready():
        # If task is complete, get the result
        if task_result.successful():
            logger.info(f"Task {task_id} completed successfully")
            result = task_result.get()
            
            # Format confidence as percentage
            average_confidence = result.get('average_confidence', 0)
            average_confidence_formatted = f"{average_confidence:.2%}"
            
            # Return results template with data
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "full_text": result.get("full_text", "No text extracted"),
                    "average_confidence_formatted": average_confidence_formatted,
                    "summary": result.get("summary", "No summary generated"),
                    "company": result.get("company", "Not found"),
                    "recipient": result.get("recipient", "Not found"),
                    "company_address": result.get("company_address", "Not found"),
                    "recipient_address": result.get("recipient_address", "Not found"),
                    "duration": result.get("duration", "Not found"),
                    "governing_law": result.get("governing_law", "Not found"),
                    "confidential_info": result.get("confidential_info", "Not found"),
                    "dates": result.get("dates", [])
                }
            )
        else:
            # Task failed
            logger.error(f"Task {task_id} failed: {task_result.result}")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Processing failed: {task_result.result}"},
                status_code=500
            )
    else:
        # Task is still running, show progress page
        logger.info(f"Task {task_id} still processing, rendering waiting page")
        return templates.TemplateResponse(
            "waiting.html",
            {"request": request, "task_id": task_id}
        )

@app.get("/task-status/{task_id}")
async def task_status(task_id: str):
    """
    API endpoint to check task status
    """
    task_result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": task_result.status}

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