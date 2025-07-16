
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
import logging
import redis
import shutil
from celery import Celery
from celery import chord
from pdf2image import convert_from_bytes

# Import tasks
from tasks.utils import get_timestamp, update_job_status

# Initialize logging
logger = logging.getLogger(__name__)

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

# Multi-level parallelism configuration
# Document-level: ProcessPoolExecutor for parallel document processing
# Chunk-level: ThreadPoolExecutor within each document process
import concurrent.futures
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global ProcessPoolExecutor for document-level parallelism
MAX_DOCUMENT_WORKERS = 12  # Maximum parallel documents
document_executor = None
document_executor_lock = threading.Lock()
active_document_processes = 0
active_processes_lock = threading.Lock()

def get_system_load_info():
    """
    Get current system load information for adaptive processing decisions.
    
    Returns:
        dict: System load metrics including CPU usage, active processes, and memory
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        global active_document_processes
        with active_processes_lock:
            current_active = active_document_processes
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'active_processes': current_active,
            'max_processes': MAX_DOCUMENT_WORKERS,
            'load_level': 'low' if current_active < 6 else 'high'
        }
    except Exception as e:
        logger.error(f"Error getting system load: {str(e)}")
        return {
            'cpu_percent': 50.0,
            'memory_percent': 50.0,
            'active_processes': 0,
            'max_processes': MAX_DOCUMENT_WORKERS,
            'load_level': 'medium'
        }


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

def get_document_executor():
    """
    Get or create the global ProcessPoolExecutor for document-level parallelism.
    
    This implements document-level parallelism where each document is processed
    in a separate process, allowing up to 12 documents to be processed simultaneously.
    
    Returns:
        ProcessPoolExecutor: Global executor for document processing
    """
    global document_executor
    
    with document_executor_lock:
        if document_executor is None:
            logger.info(f"🚀 Initializing ProcessPoolExecutor with {MAX_DOCUMENT_WORKERS} workers for document-level parallelism")
            document_executor = ProcessPoolExecutor(max_workers=MAX_DOCUMENT_WORKERS)
            
    return document_executor

def update_active_processes(increment=1):
    """
    Update the count of active document processes for load monitoring.
    
    Args:
        increment: +1 when starting a process, -1 when completing
    """
    global active_document_processes
    
    with active_processes_lock:
        active_document_processes = max(0, active_document_processes + increment)
        
        # Also update Redis for cross-process visibility
        try:
            redis_client.set('active_document_processes', active_document_processes, ex=300)  # 5 minute expiry
        except Exception as e:
            logger.warning(f"Could not update Redis process count: {str(e)}")
        
        logger.info(f"📊 Active document processes: {active_document_processes}/{MAX_DOCUMENT_WORKERS}")

def submit_document_for_processing(job_id, file_path, file_name):
    """
    Submit a document for processing using 5-page fixed chunking with high-throughput parallelism.
    
    Always splits the document into fixed 5-page chunks for standardized load balancing.
    Remaining pages (if total % 5 != 0) form a smaller final chunk.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file  
        file_name: Original file name
        
    Returns:
        AsyncResult: Celery chord result object
    """
    try:
        # Import tasks
        from tasks.ppstructure_tasks import process_document_with_ppstructure, merge_and_summarize_chunks
        
        # Get current system load for logging
        load_info = get_system_load_info()
        
        # Fixed chunk size for optimal load balancing
        PAGES_PER_CHUNK = 5
        
        # Read file and convert to chunks
        file_ext = os.path.splitext(file_name)[1].lower()
        chunk_files = []
        
        if file_ext == '.pdf':
            logger.info(f"📄 Converting PDF to images for 5-page chunking: {file_name}")
            # Convert PDF to images using convert_from_bytes
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            images = convert_from_bytes(pdf_bytes, dpi=100)
            
            total_pages = len(images)
            logger.info(f"📄 PDF converted to {total_pages} pages, creating 5-page chunks")
            
            # Calculate chunk ranges using fixed 5-page size
            chunk_ranges = []
            for start_page in range(0, total_pages, PAGES_PER_CHUNK):
                end_page = min(start_page + PAGES_PER_CHUNK - 1, total_pages - 1)
                chunk_ranges.append((start_page, end_page))
            
            num_chunks = len(chunk_ranges)
            logger.info(f"📦 Fixed 5-page chunking for {total_pages} pages: {num_chunks} chunks")
            logger.info(f"📦 Chunk ranges: {chunk_ranges}")
            
            # Create chunks directory
            chunks_dir = os.path.join("chunks", job_id)
            os.makedirs(chunks_dir, exist_ok=True)
            
            # Create chunk files for each page range
            for chunk_idx, (start_page, end_page) in enumerate(chunk_ranges):
                chunk_id = f"chunk_{chunk_idx:04d}"
                chunk_file_name = f"{chunk_id}_{file_name}"
                chunk_file_path = os.path.join(chunks_dir, chunk_file_name)
                
                # Extract pages for this chunk
                chunk_images = []
                for page_idx in range(start_page, end_page + 1):
                    if page_idx < len(images):
                        chunk_images.append(images[page_idx])
                
                if chunk_images:
                    # Save chunk as PDF or image
                    if len(chunk_images) == 1:
                        # Single page - save as image
                        chunk_images[0].save(chunk_file_path.replace('.pdf', '.jpg'))
                        chunk_file_path = chunk_file_path.replace('.pdf', '.jpg')
                        chunk_file_name = chunk_file_name.replace('.pdf', '.jpg')  # Fix: Update filename too
                    else:
                        # Multiple pages - save as PDF
                        chunk_images[0].save(chunk_file_path, save_all=True, append_images=chunk_images[1:])
                    
                    # Ensure proper file permissions
                    os.chmod(chunk_file_path, 0o644)
                    
                    chunk_files.append({
                        'chunk_id': chunk_id,
                        'chunk_file_path': chunk_file_path,
                        'chunk_file_name': chunk_file_name,
                        'start_page': start_page + 1,  # 1-based page numbering
                        'end_page': end_page + 1,
                        'actual_pages': len(chunk_images)
                    })
                    
                    pages_desc = f"pages {start_page + 1}-{end_page + 1}" if len(chunk_images) > 1 else f"page {start_page + 1}"
                    logger.info(f"📦 Created {chunk_id}: {pages_desc} ({len(chunk_images)} pages)")
                else:
                    logger.info(f"📦 Skipped {chunk_id}: no pages in range {start_page}-{end_page}")
        
        else:
            # Single image file - create one chunk
            logger.info(f"🖼️  Single image file, creating 1 chunk: {file_name}")
            
            chunks_dir = os.path.join("chunks", job_id)
            os.makedirs(chunks_dir, exist_ok=True)
            
            # For single image, create one chunk
            chunk_idx = 0
            chunk_id = f"chunk_{chunk_idx:04d}"
            chunk_file_name = f"{chunk_id}_{file_name}"
            chunk_file_path = os.path.join(chunks_dir, chunk_file_name)
            
            # Copy the original file
            shutil.copy2(file_path, chunk_file_path)
            os.chmod(chunk_file_path, 0o644)
            
            chunk_files.append({
                'chunk_id': chunk_id,
                'chunk_file_path': chunk_file_path,
                'chunk_file_name': chunk_file_name,
                'start_page': 1,  # Single page always starts at 1
                'end_page': 1,
                'actual_pages': 1
            })
            
            logger.info(f"📦 Created {chunk_id}: single image file")
        
        if not chunk_files:
            raise ValueError("No chunks could be created from the document")
        
        # Create Celery chord tasks for parallel chunk processing
        shared_queue = 'chunk_queue'  # Use shared queue for all chunk tasks
        
        # Create chunk processing tasks - all chunks processed in parallel
        chunk_tasks = []
        for chunk_info in chunk_files:
            chunk_task = process_document_with_ppstructure.s(
                job_id,
                chunk_info['chunk_file_path'],
                chunk_info['chunk_file_name'],
                generate_summary=False,  # No summary for individual chunks
                actual_start_page=chunk_info['start_page'],
                enable_visualizations=False,  # Disabled for performance
                enable_table_extraction=True,
                enable_figure_extraction=True,
                enable_chart_extraction=True,
                fast_mode=False,  # Keep advanced features enabled
                parallel_extraction=True,  # Enable parallel extraction within chunks
                max_extraction_workers=4  # Limit workers per chunk for resource control
            )
            chunk_tasks.append(chunk_task)
        
        # Create merge task that will combine results
        merge_task = merge_and_summarize_chunks.s(job_id)
        
        # Submit chord: run chunk tasks in parallel, then merge results
        num_chunks = len(chunk_files)
        logger.info(f"📤 Submitting {num_chunks} chunks for high-throughput processing (job {job_id})")
        logger.info(f"📤 Queue: {shared_queue} | Containers: 6 | Concurrency: 3 per container")
        logger.info(f"📤 Expected parallelism: up to {min(num_chunks, 18)} concurrent chunk processes")
        
        # Update job status to show chunking started
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': f'Document "{file_name}" split into {num_chunks} chunks (5 pages each) for high-throughput processing',
            'progress': 5,  # Small initial progress to show work started
            'num_chunks': num_chunks,
            'chunks_created': [chunk['chunk_id'] for chunk in chunk_files],
            'processing_mode': 'high_throughput_chunked',
            'chunk_size': PAGES_PER_CHUNK,
            'expected_parallelism': min(num_chunks, 18),  # 6 containers * 3 concurrency = 18 max
            'original_filename': file_name,  # Store original filename for merge task
            'updated_at': get_timestamp()
        })
        
        # Execute chord: chunk tasks in parallel, then merge
        chord_result = chord(chunk_tasks, merge_task).apply_async(
            queue=shared_queue,
            routing_key=shared_queue
        )
        
        logger.info(f"📋 Submitted document {job_id} ({file_name}) as {num_chunks}-chunk chord")
        logger.info(f"📋 Chunking strategy: {PAGES_PER_CHUNK} pages per chunk")
        logger.info(f"📋 Load: {load_info['load_level']}, Active: {load_info['active_processes']}/{load_info['max_processes']}")
        
        return chord_result
        
    except Exception as e:
        logger.error(f"Error submitting document for chunk processing: {str(e)}")
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': f'Chunk processing submission failed: {str(e)}',
            'message': 'Failed to submit document for chunk processing',
            'updated_at': get_timestamp()
        })
        raise

def get_parallel_processing_status():
    """
    Get current status of parallel document processing system.
    
    Returns:
        dict: System status including active processes, load metrics, and capacity
    """
    try:
        load_info = get_system_load_info()
        
        return {
            'active_processes': load_info['active_processes'],
            'max_processes': load_info['max_processes'],
            'available_capacity': load_info['max_processes'] - load_info['active_processes'],
            'cpu_percent': load_info['cpu_percent'],
            'memory_percent': load_info['memory_percent'],
            'load_level': load_info['load_level'],
            'parallelism_enabled': True,
            'processing_mode': 'ProcessPoolExecutor'
        }
    except Exception as e:
        logger.error(f"Error getting parallel processing status: {str(e)}")
        return {
            'active_processes': 0,
            'max_processes': MAX_DOCUMENT_WORKERS,
            'available_capacity': MAX_DOCUMENT_WORKERS,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'load_level': 'unknown',
            'parallelism_enabled': False,
            'processing_mode': 'error'
        }

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
async def bulk_processing_page(request: Request, jobs: str = None):
    """
    Render the bulk processing monitoring page with job IDs from URL parameters
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

# Bulk upload endpoint with parallel processing support
@app.post("/bulk-upload")
async def bulk_upload(request: Request, files: List[UploadFile] = File(...)):
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



# Single document upload endpoint with container load balancing
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
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
        # Worker container architecture:
        # 1. Document routing: Intelligent load balancing between worker containers
        # 2. Specialized workers: Containers with heavy dependencies (PaddlePaddle, pdf2image)
        # 3. Adaptive chunking within workers:
        #    - Documents >100 pages: Always chunk with dynamic sizing
        #    - Documents ≤100 pages: Chunk only if system load <6 processes
        # 4. Dynamic chunk sizes: 50 (≤100), 125 (101-250), 150 (251-500), 100 (>500)
        # 5. Load-aware execution: CPU and memory monitoring for optimal performance


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
        
        # CRITICAL: Validate COMPLETED status by checking if results files actually exist AND are complete
        # This prevents premature redirect when worker says COMPLETED but files aren't ready
        if job_status.get('status') == 'COMPLETED':
            results_dir = os.path.join("results", job_id)
            clean_results_path = os.path.join(results_dir, "results.json")
            
            # Check if results files exist AND are complete with valid content
            results_ready = False
            if os.path.exists(clean_results_path):
                try:
                    # Verify the results file has valid content and required fields
                    with open(clean_results_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)
                    
                    # Check for required fields that indicate processing is truly complete
                    required_fields = ['combined_text', 'total_pages', 'processing_completed_at']
                    has_required_fields = all(field in results_data for field in required_fields)
                    has_substantial_content = len(results_data.get('combined_text', '')) > 10
                    
                    if has_required_fields and has_substantial_content:
                        results_ready = True
                        logger.info(f"✅ Job {job_id} - Results validated and ready for display")
                    else:
                        logger.info(f"⚠️ Job {job_id} - Results file exists but incomplete: fields={has_required_fields}, content={has_substantial_content}")
                        
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"⚠️ Job {job_id} - Results file corrupted or incomplete: {str(e)}")
            
            # If results files don't exist or are incomplete, keep status as PROCESSING
            if not results_ready:
                logger.info(f"🔄 Job {job_id} marked COMPLETED but results not ready - keeping as PROCESSING")
                job_status['status'] = 'PROCESSING'
                job_status['message'] = 'Model inference complete. Finalizing results and generating summary...'
                job_status['progress'] = 95
        
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

# Parallel processing status monitoring endpoint
@app.get("/api/parallel/status")
async def get_parallel_status():
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

# Legacy container status endpoint (for backwards compatibility)
@app.get("/api/containers/status")
async def get_container_status():
    """
    Legacy endpoint - redirects to parallel processing status.
    Maintained for backwards compatibility.
    """
    return await get_parallel_status()

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
    
    # Check parallel processing system status
    try:
        parallel_status = get_parallel_processing_status()
        processing_status = "ok" if parallel_status['parallelism_enabled'] else "degraded"
    except Exception as e:
        processing_status = f"error: {str(e)}"
        parallel_status = {}
    
    return {
        "status": "healthy" if redis_status == "ok" and processing_status == "ok" else "degraded",
        "redis": redis_status,
        "parallel_processing": processing_status,
        "system_info": parallel_status,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

