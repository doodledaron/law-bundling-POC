"""
Relevance extraction processing service.
Mirrors services/processing.py but with reversed page allocation:
- First 90% of pages: Gemini (high-yield front-matter)
- Last 10% of pages: PPStructure (back pages)
"""
import os
import logging
import math
import redis
from celery import Celery, chord
from pdf2image import convert_from_bytes

from tasks.utils import get_timestamp, update_job_status

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


def submit_relevance_extraction(job_id, file_path, file_name):
    """
    Submit a document for relevance extraction using reversed page allocation.
    
    Strategy:
    - < 8 pages: 100% Gemini (same as full processing)
    - >= 8 pages: First 90% Gemini (front-matter), Last 10% PPStructure (back pages)
    
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
        from tasks.relevance_tasks import merge_and_generate_relevance
        
        # Fixed chunk size for optimal load balancing (same as full processing)
        PAGES_PER_CHUNK = 5
        
        # Read file and convert to chunks
        file_ext = os.path.splitext(file_name)[1].lower()
        chunk_files = []
        
        if file_ext == '.pdf':
            logger.info(f"📄 [RELEVANCE] Converting PDF to detect page count: {file_name}")
            # Convert PDF to images using convert_from_bytes
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            images = convert_from_bytes(pdf_bytes, dpi=100)
            
            total_pages = len(images)
            logger.info(f"📄 [RELEVANCE] PDF has {total_pages} pages")
            
            # DECISION POINT: Choose processing method based on page count
            if total_pages < 8:
                logger.info(f"🤖 [RELEVANCE] Document has {total_pages} pages (<8) - Using AI-only processing")
                # Use single Gemini-only task followed by relevance merge
                gemini_task = process_document_with_gemini_only.s(
                    job_id,
                    file_path,
                    file_name,
                    generate_summary=False,  # Let merge function handle relevance generation
                    actual_start_page=1
                )
                
                # Create relevance merge task
                merge_task = merge_and_generate_relevance.s(job_id)
                
                # Update job status for Gemini-only processing
                update_job_status(redis_client, job_id, {
                    'status': 'PROCESSING',
                    'message': f'Document "{file_name}" ({total_pages} pages) processing for relevance extraction',
                    'progress': 10,
                    'processing_mode': 'relevance_gemini_only',
                    'total_pages': total_pages,
                    'num_chunks': 1,
                    'chunks_total': 1,
                    'original_filename': file_name,
                    'updated_at': get_timestamp()
                })
                
                # Execute as chord: Gemini task then relevance merge
                chord_result = chord([gemini_task], merge_task).apply_async(
                    queue='chunk_queue',
                    routing_key='chunk_queue'
                )
                
                logger.info(f"📋 [RELEVANCE] Submitted small document {job_id} ({file_name}) for AI-only relevance extraction")
                return chord_result
            
            else:
                # REVERSED ALLOCATION: First 90% Gemini, Last 10% PPStructure
                logger.info(f"📄 [RELEVANCE] Document has {total_pages} pages (≥8) - Using reversed mixed processing")
                logger.info(f"📄 [RELEVANCE] First 90% pages: Gemini (front-matter), Last 10% pages: PPStructure")
                
                # Calculate reversed page-based distribution
                ppstructure_pages = max(1, math.ceil(total_pages * 0.10))  # Last 10%
                gemini_pages = total_pages - ppstructure_pages  # First 90%
                
                logger.info(f"📦 [RELEVANCE] Page-based distribution for {total_pages} pages:")
                logger.info(f"📦 [RELEVANCE] AI processing (first pages): {gemini_pages} pages (90%)")
                logger.info(f"📦 [RELEVANCE] OCR processing (last pages): {ppstructure_pages} pages (10%)")
                
                # Create chunk ranges based on REVERSED page allocation
                chunk_ranges = []
                current_page = 0
                
                # FIRST: Create ONE large Gemini-only chunk for first 90% of pages
                if gemini_pages > 0:
                    end_page = current_page + gemini_pages - 1
                    chunk_ranges.append((current_page, end_page, "gemini_only_bulk"))
                    logger.info(f"📦 [RELEVANCE] Creating Gemini bulk chunk: pages {current_page + 1}-{end_page + 1} ({gemini_pages} pages)")
                    current_page += gemini_pages
                
                # SECOND: Create PPStructure chunks for last 10% (up to PAGES_PER_CHUNK pages each)
                pages_remaining_ppstructure = ppstructure_pages
                while pages_remaining_ppstructure > 0:
                    chunk_size = min(PAGES_PER_CHUNK, pages_remaining_ppstructure)
                    end_page = current_page + chunk_size - 1
                    chunk_ranges.append((current_page, end_page, "ppstructure"))
                    current_page += chunk_size
                    pages_remaining_ppstructure -= chunk_size
                
                num_chunks = len(chunk_ranges)
                ppstructure_chunks = sum(1 for _, _, method in chunk_ranges if method == "ppstructure")
                gemini_chunks = sum(1 for _, _, method in chunk_ranges if method in ["gemini_only", "gemini_only_bulk"])
                
                logger.info(f"📦 [RELEVANCE] Created {num_chunks} total chunks: {gemini_chunks} Gemini chunks (first), {ppstructure_chunks} PPStructure chunks (last)")
                logger.info(f"📦 [RELEVANCE] REVERSED STRATEGY: Gemini gets first 90%, PPStructure gets last 10%")
                logger.info(f"📦 [RELEVANCE] Chunk ranges: {[(start, end, method) for start, end, method in chunk_ranges]}")
            
                # Create chunks directory
                chunks_dir = os.path.join("chunks", job_id)
                os.makedirs(chunks_dir, exist_ok=True)
                
                # Create chunk files for each page range with processing method assignment
                for chunk_idx, (start_page, end_page, processing_method) in enumerate(chunk_ranges):
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
                        logger.info(f"📦 [RELEVANCE] Created {chunk_id}: {pages_desc} ({len(chunk_images)} pages) - {processing_method}")
                    else:
                        logger.info(f"📦 [RELEVANCE] Skipped {chunk_id}: no pages in range {start_page}-{end_page}")
        
        else:
            # Single image file - use Gemini-only processing
            logger.info(f"🖼️  [RELEVANCE] Single image file - Using Gemini-only processing: {file_name}")
            
            # Use Gemini-only task for single image
            gemini_task = process_document_with_gemini_only.s(
                job_id,
                file_path,
                file_name,
                generate_summary=False,  # Let merge function handle relevance generation
                actual_start_page=1
            )
            
            # Create relevance merge task
            merge_task = merge_and_generate_relevance.s(job_id)
            
            # Update job status for Gemini-only processing
            update_job_status(redis_client, job_id, {
                'status': 'PROCESSING',
                'message': f'Single image "{file_name}" processing for relevance extraction',
                'progress': 10,
                'processing_mode': 'relevance_gemini_only',
                'total_pages': 1,
                'num_chunks': 1,
                'chunks_total': 1,
                'original_filename': file_name,
                'updated_at': get_timestamp()
            })
            
            # Execute as chord: Gemini task then relevance merge
            chord_result = chord([gemini_task], merge_task).apply_async(
                queue='chunk_queue',
                routing_key='chunk_queue'
            )
            
            logger.info(f"📋 [RELEVANCE] Submitted single image {job_id} ({file_name}) for Gemini-only relevance extraction")
            return chord_result
        
        if not chunk_files:
            raise ValueError("No chunks could be created from the document")
        
        # Create Celery chord tasks for reversed mixed parallel chunk processing
        shared_queue = 'chunk_queue'  # Use shared queue for all chunk tasks
        
        # Create chunk processing tasks - reversed mixed (Gemini first, PPStructure last)
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
                    fast_mode=False,
                    parallel_extraction=True,
                    max_extraction_workers=4
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
        
        # Create relevance merge task that will combine results and generate relevance
        merge_task = merge_and_generate_relevance.s(job_id)
        
        # Submit chord: run reversed mixed chunk tasks in parallel, then relevance merge
        num_chunks = len(chunk_files)
        logger.info(f"📤 [RELEVANCE] Submitting {num_chunks} chunks for reversed mixed processing (job {job_id})")
        logger.info(f"📤 [RELEVANCE] Distribution: {gemini_count} Gemini (first 90%), {ppstructure_count} PPStructure (last 10%)")
        logger.info(f"📤 [RELEVANCE] Queue: {shared_queue} | Expected parallelism: up to {min(num_chunks, 18)} concurrent chunk processes")
        
        # Update job status to show reversed mixed chunking started
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': f'Document "{file_name}" split into {num_chunks} chunks for relevance extraction (first {gemini_count} AI chunks, last {ppstructure_count} OCR chunks)',
            'progress': 5,
            'num_chunks': num_chunks,
            'chunks_total': num_chunks,
            'gemini_chunks': gemini_count,
            'ppstructure_chunks': ppstructure_count,
            'chunks_created': [chunk['chunk_id'] for chunk in chunk_files],
            'processing_mode': 'relevance_mixed_reversed',
            'chunk_size': PAGES_PER_CHUNK,
            'expected_parallelism': min(num_chunks, 18),
            'original_filename': file_name,
            'updated_at': get_timestamp()
        })
        
        # Execute chord: chunk tasks in parallel, then relevance merge
        chord_result = chord(chunk_tasks, merge_task).apply_async(
            queue=shared_queue,
            routing_key=shared_queue
        )
        
        logger.info(f"📋 [RELEVANCE] Submitted document {job_id} ({file_name}) as {num_chunks}-chunk reversed mixed processing chord")
        logger.info(f"📋 [RELEVANCE] Reversed strategy: {gemini_count} Gemini (first) + {ppstructure_count} PPStructure (last)")
        
        return chord_result
        
    except Exception as e:
        logger.error(f"Error submitting document for relevance extraction: {str(e)}")
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': f'Relevance extraction submission failed: {str(e)}',
            'message': 'Failed to submit document for relevance extraction',
            'updated_at': get_timestamp()
        })
        raise
