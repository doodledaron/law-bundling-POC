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
import redis
import pickle
import gc
from celery.result import AsyncResult
from celery import Celery, signals

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
app = FastAPI(title="Document Processing API",
              description="API for processing documents with memory optimization",
              version="1.0.0")

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

# Memory cleanup helper function
def background_memory_cleanup():
    """Clean up memory in the background"""
    # Run a full garbage collection
    collected = gc.collect()
    logger.info(f"Background memory cleanup collected {collected} objects")
    
    # Try to release memory to the OS
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        if hasattr(libc, 'malloc_trim'):
            libc.malloc_trim(0)
    except Exception:
        pass
    
    # Log memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        logger.info(f"Current memory usage after cleanup: {memory_mb:.2f} MB")
    except ImportError:
        pass

# Define root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, background_tasks: BackgroundTasks):
    """
    Render the main upload page
    """
    logger.info("Main page accessed")
    # Schedule a background memory cleanup
    background_tasks.add_task(background_memory_cleanup)
    return templates.TemplateResponse("index.html", {"request": request})

# Define upload endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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
        
        # Set initial status in Redis
        if redis_client:
            redis_client.set(f"status:{job_id}", "PENDING")
            logger.info(f"Initial status set for job ID: {job_id}")
        
        # Queue the document processing task
        # Pass the job_id to the task to ensure consistency
        task = process_document_task.delay(contents, file.content_type, file.filename, job_id)
        
        # Store the task ID in Redis with a 24-hour expiration
        if redis_client:
            redis_client.setex(f"job:{job_id}", 86400, task.id)
            logger.info(f"Task ID stored in Redis with job ID: {job_id}")
        
        logger.info(f"Document processing queued with task ID: {task.id}")
        
        # Schedule background cleanup to free memory after upload processing
        background_tasks.add_task(background_memory_cleanup)
        
        # Redirect to the status page
        return RedirectResponse(url=f"/status/{job_id}", status_code=303)

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Run garbage collection on error
        gc.collect()
        
        # Force flush logs
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()
            
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )

@app.get("/status/{job_id}", response_class=HTMLResponse)
async def check_status(request: Request, job_id: str, background_tasks: BackgroundTasks):
    """
    Serve the static status page HTML - all status data will be fetched via API
    """
    logger.info(f"Status page requested for job ID: {job_id}")
    
    # Basic validation that the job exists (to show a proper error page)
    if redis_client and not redis_client.exists(f"job:{job_id}") and not redis_client.exists(f"status:{job_id}"):
        logger.warning(f"Job ID not found: {job_id}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Job not found or expired."},
            status_code=404
        )
    
    # Schedule a background cleanup task
    background_tasks.add_task(background_memory_cleanup)
    
    # Serve the static status page
    return templates.TemplateResponse("status.html", {"request": request})

# Add this endpoint to your FastAPI app
@app.get("/api/status/{job_id}")
async def api_status(job_id: str, background_tasks: BackgroundTasks):
    """
    Get job status as JSON for the frontend to consume
    """
    logger.info(f"API status check for job ID: {job_id}")
    
    # Get the job status from Redis
    if not redis_client:
        logger.error("Redis connection not available")
        return {"error": "Queue system not available", "status": "ERROR"}
    
    status = redis_client.get(f"status:{job_id}")
    
    if not status:
        logger.warning(f"Job ID not found: {job_id}")
        return {"error": "Job not found or expired", "status": "NOT_FOUND"}
    
    status = status.decode('utf-8')
    
    # Basic response
    response = {
        "job_id": job_id,
        "status": status,
        "is_chunked": False,
    }
    
    # If the job is being processed in chunks, add progress information
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
        
        logger.info(f"Chunked job status API: {completed_chunks}/{total_chunks} chunks completed ({progress}%)")
        
        # Add chunk information to response
        response.update({
            "is_chunked": True,
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            "progress": progress
        })
    
    # Schedule a background cleanup task
    background_tasks.add_task(background_memory_cleanup)
    
    return response


@app.get("/result/{job_id}", response_class=HTMLResponse)
async def get_result(request: Request, job_id: str, background_tasks: BackgroundTasks):
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
    try:
        result = pickle.loads(result_data)
        
        # Schedule a background cleanup task after deserializing
        background_tasks.add_task(background_memory_cleanup)
        
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
    except Exception as e:
        logger.error(f"Error deserializing result: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Error processing result: {str(e)}"},
            status_code=500
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    # Check Redis connection
    redis_status = "UP" if redis_client and redis_client.ping() else "DOWN"
    
    # Simple memory stats
    memory_stats = {}
    try:
        import psutil
        process = psutil.Process()
        memory_stats = {
            "rss_mb": process.memory_info().rss / (1024 * 1024),
            "vms_mb": process.memory_info().vms / (1024 * 1024),
        }
    except ImportError:
        memory_stats = {"error": "psutil not installed"}
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "memory": memory_stats
    }

# Admin endpoint for memory cleanup
@app.get("/admin/cleanup-memory")
async def admin_cleanup_memory():
    """
    Admin endpoint to manually trigger memory cleanup.
    """
    try:
        # Get memory before
        memory_before = 0
        try:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        
        # Run cleanup
        collected = gc.collect()
        
        # Try to release memory to OS
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            if hasattr(libc, 'malloc_trim'):
                libc.malloc_trim(0)
        except Exception:
            pass
            
        # Get final memory
        memory_final = 0
        try:
            import psutil
            process = psutil.Process()
            memory_final = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
            
        return {
            "success": True,
            "collected_objects": collected,
            "memory_before_mb": round(memory_before, 2),
            "memory_final_mb": round(memory_final, 2),
            "memory_reduced_mb": round(memory_before - memory_final, 2),
            "memory_reduced_percent": round(((memory_before - memory_final) / memory_before) * 100, 2) if memory_before > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error during manual memory cleanup: {str(e)}")
        return {"success": False, "error": str(e)}

# Debug handler for uvicorn to force flush logs after each request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests and handle memory cleanup"""
    # Log request start
    logger.debug(f"Request: {request.method} {request.url.path}")
    
    # Process the request
    try:
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        
        # Run a light garbage collection after each request
        # This prevents memory buildup but isn't as aggressive as the full cleanup
        gc.collect(0)  # Only collect youngest generation
        
        return response
    finally:
        # Force flush logs after each request
        for handler in logger.handlers + logging.root.handlers:
            handler.flush()

# Shutdown event handler
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when shutting down"""
    logger.info("Application shutting down, cleaning up resources")
    gc.collect()  # Run garbage collection on shutdown
    logger.info("Shutdown cleanup complete")