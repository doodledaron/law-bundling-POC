"""
FastAPI application for law document processing system.
Handles HTTP endpoints and delegates processing to Celery tasks.
"""
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
import json
import redis
from celery import Celery

# Import tasks
from tasks.document_tasks import process_document
from tasks.utils import get_timestamp, update_job_status

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Initialize Celery app
celery_app = Celery(
    'law_doc_processing',
    broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Initialize FastAPI app
app = FastAPI(
    title="Law Document Processing API",
    description="API for processing legal documents with OCR and information extraction",
    version="1.0.0"
)

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
app.mount("/static", StaticFiles(directory="static"), name="static")

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
}

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("chunks", exist_ok=True)

# Define root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main upload page
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Define upload endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Process uploaded file by adding it to the processing queue
    """
    try:
        # Read file contents
        try:
            contents = await file.read()
        except Exception as e:
            error_message = f"Error reading file: {str(e)}"
            print(error_message)  # Log the error
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )

        # Check file size (limit to 10MB)
        # if len(contents) > 10 * 1024 * 1024:
        #     return templates.TemplateResponse(
        #         "index.html",
        #         {"request": request, "error": "File size exceeds 10MB limit."},
        #         status_code=400
        #     )

        if not contents:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["empty_file"]},
                status_code=400
            )

        # Validate file type
        if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["invalid_type"]},
                status_code=400
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save file to disk
        file_ext = os.path.splitext(file.filename)[1].lower()
        if not file_ext:
            # If no extension, infer from content type
            if file.content_type == "application/pdf":
                file_ext = ".pdf"
            elif file.content_type == "image/jpeg":
                file_ext = ".jpg"
            elif file.content_type == "image/png":
                file_ext = ".png"
        
        upload_path = os.path.join("uploads", f"{job_id}{file_ext}")
        with open(upload_path, "wb") as f:
            f.write(contents)
        
        # Initialize job status in Redis
        update_job_status(redis_client, job_id, {
            'status': 'PENDING',
            'filename': file.filename,
            'created_at': get_timestamp(),
            'message': 'Document uploaded, waiting for processing'
        })
        
        # Submit job to Celery -> the stage where it will process the document
        # - process_large_document : For PDFs or files > 5MB
        #     - Chunks the document
        #     - Processes chunks in parallel
        #     - Merges results at the end
        # - process_small_document : For smaller files
        #     - Processes the entire document directly
        #     - Extracts text, entities, and generates a summary

        task = process_document.delay(job_id, upload_path, file.filename)
        
        # Update job status with task ID
        update_job_status(redis_client, job_id, {
            'task_id': task.id,
            'updated_at': get_timestamp()
        })
        
        # Render submission confirmation
        return templates.TemplateResponse(
            "submitted.html",
            {
                "request": request,
                "job_id": job_id,
                "filename": file.filename
            }
        )
    
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)  # Log the error
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )

# Job status endpoint
@app.get("/job/{job_id}", response_class=HTMLResponse)
async def get_job_status(request: Request, job_id: str):
    """
    Get job status and results
    """
    try:
        # Get job status from Redis
        job_data = redis_client.get(f"job:{job_id}")
        
        if not job_data:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Job not found"},
                status_code=404
            )
        
        job_status = json.loads(job_data)
        
        # For completed jobs, show results
        if job_status.get('status') == 'COMPLETED' and 'results_path' in job_status:
            try:
                # Check if results file exists
                if not os.path.exists(job_status['results_path']):
                    raise FileNotFoundError(f"Results file not found: {job_status['results_path']}")
                
                # Read results from file
                with open(job_status['results_path'], 'r') as f:
                    results = json.load(f)
                
                # Render results template
                return templates.TemplateResponse(
                    "result.html",
                    {
                        "request": request,
                        "full_text": results.get("extracted_text", ""),
                        "average_confidence_formatted": results.get("average_confidence_formatted", "N/A"),
                        "summary": results.get("summary", ""),
                        **results.get("entities", {})
                    }
                )
            except Exception as e:
                job_status['error'] = f"Error retrieving results: {str(e)}"
                print(f"Error retrieving results for job {job_id}: {str(e)}")
        
        # For in-progress or failed jobs, show status
        return templates.TemplateResponse(
            "status.html",
            {
                "request": request,
                "job_id": job_id,
                "status": job_status
            }
        )
    
    except Exception as e:
        error_message = f"Error retrieving job status: {str(e)}"
        print(error_message)  # Log the error
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )

# API endpoint for status updates
@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str):
    """
    Get job status as JSON for AJAX updates
    """
    try:
        # Get job status from Redis
        job_data = redis_client.get(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_status = json.loads(job_data)
        return job_status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    # Check Redis connection
    try:
        redis_client.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # Check Celery status
    try:
        i = celery_app.control.inspect()
        active_workers = i.active()
        celery_status = "ok" if active_workers else "no active workers"
    except Exception as e:
        celery_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if redis_status == "ok" else "degraded",
        "redis": redis_status,
        "celery": celery_status,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))