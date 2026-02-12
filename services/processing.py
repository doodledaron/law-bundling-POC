"""
Core document processing services.
Handles document submission, chunking, parallel processing management,
and system load monitoring.
"""
import os
import logging
import threading
import math
import concurrent.futures
import psutil
import redis
from celery import Celery, chord
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf2image import convert_from_bytes

from tasks.utils import get_timestamp, update_job_status
from config import Config

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
        from tasks.ppstructure_tasks import process_document_with_ppstructure
        from tasks.gemini_tasks import process_document_with_gemini_only
        from tasks.merge_tasks import merge_and_summarize_chunks
        
        # Get current system load for logging
        load_info = get_system_load_info()
        
        # Fixed chunk size for optimal load balancing
        PAGES_PER_CHUNK = 5
        
        # Read file and convert to chunks
        file_ext = os.path.splitext(file_name)[1].lower()
        chunk_files = []
        
        if file_ext == '.pdf':
            logger.info(f"📄 Converting PDF to detect page count: {file_name}")
            # Convert PDF to images using convert_from_bytes
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            images = convert_from_bytes(pdf_bytes, dpi=100)
            
            total_pages = len(images)
            logger.info(f"📄 PDF has {total_pages} pages")
            
            # DECISION POINT: Choose processing method based on page count
            if total_pages < 8:
                logger.info(f"🤖 Document has {total_pages} pages (<7) - Using AI-only processing with merge")
                # Use single Gemini-only task followed by merge for consistency
                gemini_task = process_document_with_gemini_only.s(
                    job_id,
                    file_path,
                    file_name,
                    generate_summary=False,  # Let merge function handle summary
                    actual_start_page=1
                )
                
                # Create merge task that will handle final summary and completion
                merge_task = merge_and_summarize_chunks.s(job_id)
                
                # Update job status for Gemini-only processing with merge
                update_job_status(redis_client, job_id, {
                    'status': 'PROCESSING',
                    'message': f'Document "{file_name}" ({total_pages} pages) processing with AI text analysis',
                    'progress': 10,
                    'processing_mode': 'gemini_only_complete',
                    'total_pages': total_pages,
                    'num_chunks': 1,  # Single document treated as 1 chunk
                    'chunks_total': 1,
                    'original_filename': file_name,
                    'updated_at': get_timestamp()
                })
                
                # Execute as chord: Gemini task then merge (consistent with mixed processing)
                chord_result = chord([gemini_task], merge_task).apply_async(
                    queue='chunk_queue',
                    routing_key='chunk_queue'
                )
                
                logger.info(f"📋 Submitted small document {job_id} ({file_name}) for AI-only processing with merge")
                return chord_result
            
            else:
                logger.info(f"📄 Document has {total_pages} pages (≥7) - Using mixed processing (10% OCR, 90% AI)")
                logger.info(f"📄 Creating page-based chunks with max {PAGES_PER_CHUNK} pages per chunk")
            
                # Calculate page-based distribution (10% PPStructure, 90% Gemini-only)
                ppstructure_pages = max(1, math.ceil(total_pages * 0.10))  # At least 1 page for PPStructure
                gemini_pages = total_pages - ppstructure_pages
                
                logger.info(f"📦 Page-based distribution for {total_pages} pages:")
                logger.info(f"📦 OCR processing: {ppstructure_pages} pages (10%), AI processing: {gemini_pages} pages (90%)")
                
                # Create chunk ranges based on page allocation
                chunk_ranges = []
                current_page = 0
                
                # First create PPStructure chunks (up to PAGES_PER_CHUNK pages each)
                pages_remaining_ppstructure = ppstructure_pages
                while pages_remaining_ppstructure > 0:
                    chunk_size = min(PAGES_PER_CHUNK, pages_remaining_ppstructure)
                    end_page = current_page + chunk_size - 1
                    chunk_ranges.append((current_page, end_page, "ppstructure"))
                    current_page += chunk_size
                    pages_remaining_ppstructure -= chunk_size
                
                # NEW: Create ONE large Gemini-only chunk for 70% of pages (single container processing)
                if gemini_pages > 0:
                    end_page = current_page + gemini_pages - 1
                    chunk_ranges.append((current_page, end_page, "gemini_only_bulk"))
                    logger.info(f"📦 Creating single bulk Gemini chunk: pages {current_page + 1}-{end_page + 1} ({gemini_pages} pages)")
                    current_page += gemini_pages
                
                num_chunks = len(chunk_ranges)
                ppstructure_chunks = sum(1 for _, _, method in chunk_ranges if method == "ppstructure")
                gemini_chunks = sum(1 for _, _, method in chunk_ranges if method in ["gemini_only", "gemini_only_bulk"])
                
                logger.info(f"📦 Created {num_chunks} total chunks: {ppstructure_chunks} PPStructure chunks, {gemini_chunks} Gemini-only chunks")
                logger.info(f"📦 NEW STRATEGY: PPStructure uses {PAGES_PER_CHUNK}-page chunks, Gemini uses 1 bulk chunk for entire 70%")
                logger.info(f"📦 Chunk ranges: {[(start, end, method) for start, end, method in chunk_ranges]}")
            
                # Create chunks directory
                chunks_dir = os.path.join("chunks", job_id)
                os.makedirs(chunks_dir, exist_ok=True)
                
                # Create chunk files for each page range with processing method assignment
                for chunk_idx, (start_page, end_page, processing_method) in enumerate(chunk_ranges):
                    chunk_id = f"chunk_{chunk_idx:04d}"
                    chunk_file_name = f"{chunk_id}_{file_name}"
                    chunk_file_path = os.path.join(chunks_dir, chunk_file_name)
                    
                    # Processing method is already determined in chunk_ranges
                    
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
                            chunk_file_name = chunk_file_name.replace('.pdf', '.jpg')
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
                            'actual_pages': len(chunk_images),
                            'processing_method': processing_method
                        })
                        
                        pages_desc = f"pages {start_page + 1}-{end_page + 1}" if len(chunk_images) > 1 else f"page {start_page + 1}"
                        logger.info(f"📦 Created {chunk_id}: {pages_desc} ({len(chunk_images)} pages) - {processing_method}")
                    else:
                        logger.info(f"📦 Skipped {chunk_id}: no pages in range {start_page}-{end_page}")
        
        else:
            # Single image file - use Gemini-only processing (always < 7 pages)
            logger.info(f"🖼️  Single image file - Using Gemini-only processing with merge: {file_name}")
            
            # Use Gemini-only task for single image
            gemini_task = process_document_with_gemini_only.s(
                job_id,
                file_path,
                file_name,
                generate_summary=False,  # Let merge function handle summary
                actual_start_page=1
            )
            
            # Create merge task that will handle final summary and completion
            merge_task = merge_and_summarize_chunks.s(job_id)
            
            # Update job status for Gemini-only processing with merge
            update_job_status(redis_client, job_id, {
                'status': 'PROCESSING',
                'message': f'Single image "{file_name}" processing with AI text analysis',
                'progress': 10,
                'processing_mode': 'gemini_only_complete',
                'total_pages': 1,
                'num_chunks': 1,  # Single image treated as 1 chunk
                'chunks_total': 1,
                'original_filename': file_name,
                'updated_at': get_timestamp()
            })
            
            # Execute as chord: Gemini task then merge (consistent with all other processing)
            chord_result = chord([gemini_task], merge_task).apply_async(
                queue='chunk_queue',
                routing_key='chunk_queue'
            )
            
            logger.info(f"📋 Submitted single image {job_id} ({file_name}) for Gemini-only processing with merge")
            return chord_result
        
        if not chunk_files:
            raise ValueError("No chunks could be created from the document")
        
        # Create Celery chord tasks for mixed parallel chunk processing
        shared_queue = 'chunk_queue'  # Use shared queue for all chunk tasks
        
        # Create chunk processing tasks - mixed PPStructure and Gemini-only
        chunk_tasks = []
        ppstructure_count = 0
        gemini_count = 0
        
        for chunk_info in chunk_files:
            processing_method = chunk_info['processing_method']
            
            if processing_method == 'ppstructure':
                # PPStructure processing task
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
                ppstructure_count += 1
            elif processing_method in ['gemini_only', 'gemini_only_bulk']:
                # Gemini-only processing task (both regular and bulk)
                chunk_task = process_document_with_gemini_only.s(
                    job_id,
                    chunk_info['chunk_file_path'],
                    chunk_info['chunk_file_name'],
                    generate_summary=False,  # No summary for individual chunks
                    actual_start_page=chunk_info['start_page']
                )
                gemini_count += 1
            else:
                # Fallback for unknown processing methods
                logger.warning(f"Unknown processing method: {processing_method}, defaulting to Gemini-only")
                chunk_task = process_document_with_gemini_only.s(
                    job_id,
                    chunk_info['chunk_file_path'],
                    chunk_info['chunk_file_name'],
                    generate_summary=False,
                    actual_start_page=chunk_info['start_page']
                )
                gemini_count += 1
            
            chunk_tasks.append(chunk_task)
        
        # Create merge task that will combine results
        merge_task = merge_and_summarize_chunks.s(job_id)
        
        # Submit chord: run mixed chunk tasks in parallel, then merge results
        num_chunks = len(chunk_files)
        logger.info(f"📤 Submitting {num_chunks} chunks for mixed processing (job {job_id})")
        logger.info(f"📤 Distribution: {ppstructure_count} PPStructure, {gemini_count} Gemini-only")
        logger.info(f"📤 Queue: {shared_queue} | Containers: 6 | Concurrency: 3 per container")
        logger.info(f"📤 Expected parallelism: up to {min(num_chunks, 18)} concurrent chunk processes")
        
        # Update job status to show mixed chunking started
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': f'Document "{file_name}" split into {num_chunks} chunks for hybrid processing ({ppstructure_count} OCR chunks, 1 AI analysis chunk)',
            'progress': 5,  # Small initial progress to show work started
            'num_chunks': num_chunks,
            'chunks_total': num_chunks,  # Also store as chunks_total for consistency
            'ppstructure_chunks': ppstructure_count,
            'gemini_chunks': gemini_count,
            'chunks_created': [chunk['chunk_id'] for chunk in chunk_files],
            'processing_mode': 'mixed_processing_chunked',
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
        
        logger.info(f"📋 Submitted document {job_id} ({file_name}) as {num_chunks}-chunk mixed processing chord")
        logger.info(f"📋 Mixed strategy: {ppstructure_count} PPStructure + {gemini_count} Gemini-only chunks")
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
