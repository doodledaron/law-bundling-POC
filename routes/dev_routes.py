"""
Development routes for the law document processing system.
These endpoints are for internal development and testing only.
No API key authentication required.
"""
import os
import json
import uuid
import logging
from typing import List

from fastapi import APIRouter, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services.processing import (
    redis_client, submit_document_for_processing,
    get_system_load_info, get_parallel_processing_status,
    _fix_image_paths, _fix_file_path, MAX_DOCUMENT_WORKERS
)
from tasks.utils import get_timestamp, update_job_status

logger = logging.getLogger(__name__)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

dev_router = APIRouter(prefix="/dev", tags=["dev"])


# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
}


@dev_router.get("/", response_class=HTMLResponse)
async def dev_root(request: Request):
    """
    Render the main upload page (development interface)
    """
    return templates.TemplateResponse("index.html", {"request": request})

@dev_router.get("/bulk", response_class=HTMLResponse)
async def dev_bulk_processing_page(request: Request, jobs: str = None):
    """
    Render the bulk processing monitoring page with job IDs from URL parameters (development interface)
    """
    template_data = {"request": request}
    
    # Handle job IDs passed via URL parameters from submission confirmation
    if jobs:
        job_ids = jobs.split(',')
        template_data.update({
            "job_ids": job_ids,
            "job_ids_json": json.dumps(job_ids)
        })
        print(f"Bulk processing page loaded with {len(job_ids)} job IDs: {job_ids}")
    
    return templates.TemplateResponse("bulk_processing.html", template_data)

@dev_router.get("/results-list", response_class=HTMLResponse)
async def dev_results_page(request: Request):
    """
    Render the results list page (development interface)
    """
    return templates.TemplateResponse("results_list.html", {"request": request})


@dev_router.get("/api/results")
async def dev_api_get_results():
    """
    Get all processed documents from results folder (development interface)
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


@dev_router.post("/bulk-upload")
async def dev_bulk_upload(request: Request, files: List[UploadFile] = File(...)):
    """
    Handle bulk document upload with intelligent parallel distribution.
    
    Each uploaded document is assigned to the optimal container for parallel processing.
    This enables true document-level parallelism where multiple documents can be
    processed simultaneously across different containers.
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
                
                # Submit job for processing using Celery workers
                # Each document is processed by specialized worker containers
                task_result = submit_document_for_processing(job_id, upload_path, file.filename)
                
                # Update job status with processing info
                update_job_status(redis_client, job_id, {
                    'task_id': task_result.id,
                    'processing_mode': 'celery_workers',
                    'bulk_upload': True,
                    'bulk_position': len(submitted_files) + 1,
                    'bulk_total': len(files),
                    'updated_at': get_timestamp()
                })
                
                submitted_files.append({
                    'filename': file.filename,
                    'job_id': job_id,
                    'task_id': task_result.id
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
        
        # Redirect to bulk submission confirmation page (similar to single upload flow)
        return templates.TemplateResponse(
            "bulk_submitted.html",
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


@dev_router.post("/upload", response_class=HTMLResponse)
async def dev_upload_file(request: Request, file: UploadFile = File(...)):
    """
    Process uploaded file with intelligent container selection for optimal performance.
    
    The document is automatically assigned to the least loaded container, enabling
    efficient parallel processing when multiple users upload documents simultaneously.
    Large documents (>200 pages) will be processed using internal parallel chunking.
    """
    try:
        # Read file contents
        try:
            contents = await file.read()
        except Exception as e:
            error_message = f"Error reading file: {str(e)}"
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )

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
        
        # Submit document for processing using specialized worker containers
        try:
            # Get current system load for processing decisions
            load_info = get_system_load_info()
            
            # Submit document for processing using Celery workers
            task_result = submit_document_for_processing(job_id, upload_path, file.filename)
            
            # Update job status with processing info
            update_job_status(redis_client, job_id, {
                'task_id': task_result.id,
                'processing_mode': 'celery_workers',
                'system_load': load_info['load_level'],
                'active_processes': load_info['active_processes'],
                'max_processes': load_info['max_processes'],
                'updated_at': get_timestamp()
            })
            
            print(f"Successfully submitted job {job_id} to worker containers (Load: {load_info['load_level']}, Active: {load_info['active_processes']}/{load_info['max_processes']})")
            
        except Exception as e:
            print(f"Error submitting document to worker containers {job_id}: {str(e)}")
            # Update job status on submission failure
            update_job_status(redis_client, job_id, {
                'status': 'FAILED',
                'error': f'Worker submission failed: {str(e)}',
                'message': 'Failed to submit document to worker containers',
                'updated_at': get_timestamp()
            })

            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Failed to submit to worker containers: {str(e)}"},
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
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )


@dev_router.get("/job/{job_id}", response_class=HTMLResponse)
async def dev_get_job_status(request: Request, job_id: str):
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
                    # Mixed processing metrics
                    "num_ppstructure_chunks": results.get("num_ppstructure_chunks", 0),
                    "num_gemini_chunks": results.get("num_gemini_chunks", 0),
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


@dev_router.get("/api/parallel/status")
async def dev_get_parallel_status():
    """
    Get current status of parallel processing system for monitoring.
    
    This endpoint provides real-time visibility into the multi-level parallel processing system,
    showing document-level parallelism, system load, and processing capacity.
    """
    try:
        parallel_status = get_parallel_processing_status()
        
        return {
            "parallel_processing": parallel_status,
            "timestamp": get_timestamp(),
            "system_info": {
                "max_document_workers": MAX_DOCUMENT_WORKERS,
                "parallelism_levels": ["document_level", "chunk_level"],
                "adaptive_chunking": True,
                "load_aware_execution": True
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "parallel_processing": {},
            "timestamp": get_timestamp()
        }


@dev_router.get("/api/containers/status")
async def dev_get_container_status():
    """
    Legacy endpoint - redirects to parallel processing status (development interface).
    Maintained for backwards compatibility.
    """
    return await dev_get_parallel_status()
