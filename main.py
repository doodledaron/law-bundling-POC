"""
FastAPI application for law document processing system.
Handles HTTP endpoints and delegates processing to Celery tasks.
"""
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
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
app.mount("/results", StaticFiles(directory="results"), name="results")

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

def _fix_image_paths(page_results):
    """
    Fix image paths to be proper web URLs starting with /
    """
    fixed_results = []
    for page in page_results:
        fixed_page = page.copy()
        
        # Fix image_path
        if 'image_path' in fixed_page and fixed_page['image_path']:
            path = fixed_page['image_path']
            if not path.startswith('/'):
                fixed_page['image_path'] = '/' + path
        
        # Fix layout_vis_path
        if 'layout_vis_path' in fixed_page and fixed_page['layout_vis_path']:
            path = fixed_page['layout_vis_path']
            if not path.startswith('/'):
                fixed_page['layout_vis_path'] = '/' + path
        
        fixed_results.append(fixed_page)
    
    return fixed_results

def _fix_file_path(file_path):
    """
    Fix file path to be a proper web URL
    """
    if file_path and not file_path.startswith('/'):
        return '/' + file_path
    return file_path

# Define root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main upload page
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Bulk processing page
@app.get("/bulk", response_class=HTMLResponse)
async def bulk_processing_page(request: Request):
    """
    Render the bulk processing monitoring page
    """
    return templates.TemplateResponse("bulk_processing.html", {"request": request})

# Results list page
@app.get("/results-list", response_class=HTMLResponse)
async def results_page(request: Request):
    """
    Render the results list page
    """
    return templates.TemplateResponse("results_list.html", {"request": request})

# API endpoint to get all processed documents
@app.get("/api/results")
async def api_get_results():
    """
    Get all processed documents from results folder
    """
    try:
        results_dir = "results"
        documents = []
        
        if not os.path.exists(results_dir):
            return {"documents": []}
        
        # Scan results directory for job folders
        for job_folder in os.listdir(results_dir):
            job_path = os.path.join(results_dir, job_folder)
            
            # Skip if not a directory
            if not os.path.isdir(job_path):
                continue
                
            # Look for results.json in the job folder
            results_file = os.path.join(job_path, "results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    # Add job_id if not present
                    if 'job_id' not in doc_data:
                        doc_data['job_id'] = job_folder
                    
                    documents.append(doc_data)
                    
                except Exception as e:
                    print(f"Error reading results file {results_file}: {str(e)}")
                    continue
        
        # Sort by processing date (newest first)
        documents.sort(key=lambda x: x.get('processing_completed_at', ''), reverse=True)
        
        return {"documents": documents}
        
    except Exception as e:
        print(f"Error getting results: {str(e)}")
        return {"documents": [], "error": str(e)}

# Bulk upload endpoint
@app.post("/bulk-upload")
async def bulk_upload(request: Request, files: List[UploadFile] = File(...)):
    """
    Handle bulk document upload
    """
    try:
        if not files or len(files) == 0:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "No files were uploaded"},
                status_code=400
            )
        
        job_ids = []
        submitted_files = []
        errors = []
        
        for file in files:
            try:
                # Read file contents
                contents = await file.read()
                
                if not contents:
                    errors.append(f"{file.filename}: File is empty")
                    continue
                
                # Validate file type
                if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
                    errors.append(f"{file.filename}: Invalid file type")
                    continue
                
                # Generate job ID
                job_id = str(uuid.uuid4())
                job_ids.append(job_id)
                
                # Record job creation time
                job_created_time = get_timestamp()
                
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
                
                # Set umask for proper default permissions
                old_umask = os.umask(0o022)
                try:
                    with open(upload_path, "wb") as f:
                        f.write(contents)
                    # Ensure file is readable by all containers
                    os.chmod(upload_path, 0o644)
                finally:
                    os.umask(old_umask)
                
                # Initialize job status in Redis
                update_job_status(redis_client, job_id, {
                    'status': 'PENDING',
                    'filename': file.filename,
                    'created_at': job_created_time,
                    'message': 'Document uploaded, waiting for processing',
                    'progress': 0,
                    'bulk_upload': True,
                    'bulk_position': len(submitted_files) + 1,
                    'bulk_total': len(files)
                })
                
                # Submit job to Celery
                task = process_document.apply_async(
                    args=[job_id, upload_path, file.filename],
                    queue='documents'
                )
                
                # Update job status with task ID
                update_job_status(redis_client, job_id, {
                    'task_id': task.id,
                    'updated_at': get_timestamp()
                })
                
                submitted_files.append({
                    'filename': file.filename,
                    'job_id': job_id,
                    'task_id': task.id
                })
                
                print(f"Successfully submitted bulk job {job_id} for file {file.filename}")
                
            except Exception as e:
                error_msg = f"{file.filename}: {str(e)}"
                errors.append(error_msg)
                print(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        # If no files were successfully submitted
        if not job_ids:
            error_message = "No files could be processed. " + "; ".join(errors)
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=400
            )
        
        # Redirect to bulk processing page with job IDs
        job_ids_param = ",".join(job_ids)
        return templates.TemplateResponse(
            "bulk_processing.html",
            {
                "request": request,
                "job_ids": job_ids,
                "job_ids_json": json.dumps(job_ids),
                "submitted_files": submitted_files,
                "submitted_files_json": json.dumps(submitted_files),
                "errors": errors if errors else None
            }
        )
        
    except Exception as e:
        error_message = f"Bulk upload failed: {str(e)}"
        print(error_message)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )

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
        
        # Record job creation time
        job_created_time = get_timestamp()
        
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
        
        # Set umask for proper default permissions
        old_umask = os.umask(0o022)
        try:
            with open(upload_path, "wb") as f:
                f.write(contents)
            # Ensure file is readable by all containers
            os.chmod(upload_path, 0o644)
        finally:
            os.umask(old_umask)
        
        # Initialize job status in Redis
        update_job_status(redis_client, job_id, {
            'status': 'PENDING',
            'filename': file.filename,
            'created_at': job_created_time,
            'message': 'Document uploaded, waiting for processing',
            'progress': 0
        })
        
        # Submit job to Celery -> the stage where it will process the document
        # - process_large_document : For PDFs with >200 pages (chunking effectively disabled)
        #     - Chunks the PDF into 200-page chunks (rarely triggered)
        #     - Processes each chunk with PPStructure in parallel
        #     - Combines all chunk text and generates final summary with text_based_processor.py
        # - process_small_document : For PDFs ≤200 pages and all images (default path)
        #     - Processes the entire document directly with PPStructure
        #     - Generates summary using text_based_processor.py

        try:
            task = process_document.apply_async(
                args=[job_id, upload_path, file.filename],
                queue='documents'  # Explicitly specify the queue
            )
            
            # Update job status with task ID
            update_job_status(redis_client, job_id, {
                'task_id': task.id,
                'updated_at': get_timestamp()
            })
            
            print(f"Successfully submitted job {job_id} with task ID {task.id}")
            
        except Exception as e:
            print(f"Error submitting task for job {job_id}: {str(e)}")
            # Update job status on task submission failure
            update_job_status(redis_client, job_id, {
                'status': 'FAILED',
                'error': f'Task submission failed: {str(e)}',
                'message': 'Failed to submit processing task',
                'updated_at': get_timestamp()
            })
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Failed to submit task: {str(e)}"},
                status_code=500
            )
        
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
        
        # For chunked processing, check if all chunks are actually complete
        if job_status.get('status') == 'COMPLETED' and 'num_chunks' in job_status:
            # Verify that chunked processing is truly complete by checking results files
            results_dir = os.path.join("results", job_id)
            clean_results_path = os.path.join(results_dir, "results.json")
            
            # Only show results if the final results.json exists and is complete
            if not os.path.exists(clean_results_path):
                # Results not ready yet, show as still processing
                job_status['status'] = 'PROCESSING'
                job_status['message'] = 'Finalizing combined results and generating summary'
                job_status['progress'] = 90
        
        # For completed jobs, show results
        if job_status.get('status') == 'COMPLETED':
            try:
                # Try to load results from the new clean format first
                results_dir = os.path.join("results", job_id)
                clean_results_path = os.path.join(results_dir, "results.json")
                metrics_path = os.path.join(results_dir, "metrics.json")
                ppstructure_path = os.path.join(results_dir, "ppstructure_results.json")
                
                results = None
                metrics = {}
                ppstructure_results = {}
                
                # Try to load clean results first
                if os.path.exists(clean_results_path):
                    with open(clean_results_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                # Fallback to old results_path format
                elif 'results_path' in job_status and os.path.exists(job_status['results_path']):
                    with open(job_status['results_path'], 'r', encoding='utf-8') as f:
                        results = json.load(f)
                else:
                    raise FileNotFoundError("No results file found")
                
                # Load metrics if available
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                
                # Load PPStructure results if available  
                if os.path.exists(ppstructure_path):
                    with open(ppstructure_path, 'r', encoding='utf-8') as f:
                        ppstructure_results = json.load(f)
                
                if not results:
                    raise ValueError("Results file is empty or invalid")
                
                # For the new format, extract the needed data
                template_data = {
                    "request": request,
                    "job_id": job_id,
                    "results": results,
                    "metrics": metrics,
                    "ppstructure_results": ppstructure_results,
                    "filename": results.get("filename", "Unknown"),
                    "full_text": results.get("combined_text", ""),
                    "combined_text_path": _fix_file_path(results.get("combined_text_path", "")),
                    "average_confidence_formatted": results.get("average_confidence_formatted", "N/A"),
                    "summary": results.get("summary", ""),
                    "processing_time_seconds": float(results.get("processing_time_seconds", 0)),
                    "total_pages": int(results.get("total_pages", 0)),
                    "estimated_cost": float(results.get("estimated_cost", 0.0)),
                    "token_usage": results.get("token_usage", {}),
                    "processing_method": results.get("processing_method", "unknown"),
                    "date": results.get("date", "undated"),
                    # Extracted info fields
                    "extracted_info": results.get("extracted_info", {}),
                    # Page results from PPStructure file (with fixed paths)
                    "page_results": _fix_image_paths(ppstructure_results.get("page_results", [])),
                    # Performance data from metrics
                    "performance": metrics.get("performance", {}),
                    "confidence_metrics": metrics.get("confidence_metrics", {}),
                    "processing_details": ppstructure_results.get("processing_info", {})
                }
                
                # Render results template with new data
                return templates.TemplateResponse(
                    "result.html",
                    template_data
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

# API endpoint for status updates with enhanced loading support
@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str):
    """
    Get job status as JSON for AJAX updates with enhanced progress tracking
    """
    try:
        # Get job status from Redis
        job_data = redis_client.get(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_status = json.loads(job_data)
        
        # Add estimated completion time for better UX
        if job_status.get('status') == 'PROCESSING':
            # Add some helpful processing stage messages
            progress = job_status.get('progress', 0)
            
            # Check if this is chunked processing
            if 'num_chunks' in job_status:
                # Chunked processing stages
                num_chunks = job_status.get('num_chunks', 1)
                if progress < 20:
                    job_status['stage'] = f'Creating {num_chunks} document chunks'
                elif progress < 75:
                    # Calculate which chunk is being processed
                    chunk_progress = (progress - 30) / 45  # Chunk processing is 30-75%
                    current_chunk = max(1, int(chunk_progress * num_chunks) + 1)
                    job_status['stage'] = f'Processing chunk {current_chunk}/{num_chunks} (parallel batches)'
                elif progress < 90:
                    job_status['stage'] = f'All {num_chunks} chunks processed. Combining text and generating summary...'
                elif progress < 95:
                    job_status['stage'] = 'Summary generated. Saving final results...'
                else:
                    job_status['stage'] = 'Finalizing and preparing results display...'
            else:
                # Single document processing stages
                if progress < 30:
                    job_status['stage'] = 'Analyzing document structure'
                elif progress < 60:
                    job_status['stage'] = 'Performing OCR and layout detection'
                elif progress < 85:
                    job_status['stage'] = 'Extracting text and images'
                else:
                    job_status['stage'] = 'Generating summary and finalizing results'
        
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