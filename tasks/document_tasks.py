"""
Main document processing tasks for the legal document processing system.
Handles document processing coordination and manages the processing flow.
"""
from celery import shared_task, chain, group
from celery.utils.log import get_task_logger
import os
import json
import redis
import datetime

# Import processing and chunking tasks
# NOTE: ppstructure_tasks is imported dynamically to prevent loading models in non-document workers
from tasks.chunking_tasks import (
    create_document_chunks, 
    update_chunk_status
)
from tasks.utils import (
    get_timestamp, 
    get_unix_timestamp,
    update_job_status, 
    update_job_timing,
    calculate_duration
)
from text_based_processor import TextBasedProcessor

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Initialize text processor
text_processor = TextBasedProcessor()

def _get_ppstructure_function():
    """
    Dynamically import the PPStructure function only when needed.
    This prevents loading models in workers that don't process documents.
    """
    try:
        from tasks.ppstructure_tasks import process_document_with_ppstructure
        return process_document_with_ppstructure
    except ImportError as e:
        logger.error(f"Failed to import PPStructure function: {str(e)}")
        raise RuntimeError("PPStructure processing not available in this worker type")

@shared_task(name='tasks.process_document')
def process_document(job_id, file_path, file_name):
    """
    Main document processing router - decides between chunked and direct processing.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    start_time = get_unix_timestamp()
    
    try:
        # Start timing for overall process
        update_job_timing(redis_client, job_id, 'overall', start_time=start_time)
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'filename': file_name,
            'message': 'Analyzing document structure and determining processing strategy',
            'updated_at': get_timestamp()
        })
        
        logger.info(f"Starting document analysis for job {job_id}, file: {file_name}")
        
        # Determine processing strategy based on PDF pages only
        if _should_chunk_document(file_path, file_name):
            logger.info(f"Large PDF detected for job {job_id} - using chunked processing")
            result = process_large_document(job_id, file_path, file_name)
        else:
            logger.info(f"Small document detected for job {job_id} - using direct processing")
            result = process_small_document(job_id, file_path, file_name)
        
        # Update job status with completion
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'message': 'Document processing completed successfully',
            'results_path': result['results_path'],
            'performance': result.get('performance', {}),
            'updated_at': get_timestamp()
        })
        
        # End timing
        end_time = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'overall', end_time=end_time)
        
        logger.info(f"Document processing completed for job {job_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Error processing document: {str(e)}',
            'updated_at': get_timestamp()
        })
        
        # End timing on failure
        update_job_timing(redis_client, job_id, 'overall', end_time=get_unix_timestamp())
        
        # Re-raise to allow Celery to handle the error
        raise


def process_large_document(job_id, file_path, file_name):
    """
    Process large PDFs using 20-page chunking and parallel processing.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the PDF file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Processing large PDF {job_id} with 20-page chunking")
        
        # Step 1: Create 20-page PDF chunks
        update_job_status(redis_client, job_id, {
            'message': 'Creating 20-page PDF chunks for parallel processing',
            'progress': 10,
            'updated_at': get_timestamp()
        })
        
        chunk_start = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'chunking', start_time=chunk_start)
        
        chunk_info = create_document_chunks(job_id, file_path, file_name)
        
        chunk_end = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'chunking', end_time=chunk_end)
        
        logger.info(f"Created {chunk_info['num_chunks']} PDF chunks for job {job_id}")
        
        # Step 2: Process chunks in parallel with memory controls
        update_job_status(redis_client, job_id, {
            'message': f'Processing {chunk_info["num_chunks"]} PDF chunks in parallel (controlled)',
            'progress': 30,
            'updated_at': get_timestamp()
        })
        
        processing_start = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'parallel_processing', start_time=processing_start)
        
        # Simplify chunk processing - process sequentially instead of parallel to avoid memory issues
        chunks = chunk_info['chunks']
        chunk_results = []
        
        logger.info(f"Processing {len(chunks)} chunks sequentially for better reliability")
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")
                
                # Process chunk directly without group() - more reliable
                chunk_result = process_chunk(job_id, chunk['chunk_id'], chunk['path'])
                
                if isinstance(chunk_result, dict):
                    # Use the continuous page numbers from chunk metadata
                    chunk_number = chunk_result.get('chunk_number', i)
                    
                    # Get actual page numbers from chunk metadata if available
                    if 'start_page' not in chunk_result or 'end_page' not in chunk_result:
                        # Calculate continuous page numbers: chunk 0 = pages 1-20, chunk 1 = pages 21-40, etc.
                        chunk_result['start_page'] = (chunk_number * 20) + 1
                        chunk_result['end_page'] = min((chunk_number + 1) * 20, chunk_info.get('total_pages', 999))
                    
                    chunk_results.append(chunk_result)
                    
                    # Update progress with individual chunk completion
                    progress = 30 + (45 * (i + 1) // len(chunks))
                    update_job_status(redis_client, job_id, {
                        'message': f'✅ Chunk {i + 1}/{len(chunks)} completed - Pages {chunk_result.get("start_page", "?")}-{chunk_result.get("end_page", "?")} processed successfully',
                        'progress': progress,
                        'updated_at': get_timestamp(),
                        'current_chunk': i + 1,
                        'total_chunks': len(chunks),
                        'chunk_progress': f"{i + 1}/{len(chunks)}"
                    })
                    
                    logger.info(f"✅ Successfully completed chunk {i + 1}/{len(chunks)} - pages {chunk_result.get('start_page', '?')}-{chunk_result.get('end_page', '?')}")
                else:
                    # Handle unexpected result format
                    chunk_number = i
                    error_result = {
                        'status': 'failed',
                        'error': f'Unexpected result format: {type(chunk_result)}',
                        'chunk_id': f'chunk_{chunk_number:04d}',
                        'chunk_number': chunk_number,
                        'start_page': (chunk_number * 20) + 1,
                        'end_page': (chunk_number + 1) * 20,
                        'combined_text': '',
                        'extracted_text': ''
                    }
                    chunk_results.append(error_result)
                    logger.warning(f"Chunk {i+1} returned unexpected format, continuing...")
                    
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i+1}: {str(chunk_error)}")
                chunk_number = i
                error_result = {
                    'status': 'failed',
                    'error': str(chunk_error),
                    'chunk_id': f'chunk_{chunk_number:04d}',
                    'chunk_number': chunk_number,
                    'start_page': (chunk_number * 20) + 1,
                    'end_page': (chunk_number + 1) * 20,
                    'combined_text': '',
                    'extracted_text': ''
                }
                chunk_results.append(error_result)
                continue
        
        processing_end = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'parallel_processing', end_time=processing_end)
        
        # Ensure we have all chunk results before proceeding - be more patient
        successful_chunks = [r for r in chunk_results if r.get('status') != 'failed']
        failed_chunks = [r for r in chunk_results if r.get('status') == 'failed']
        
        logger.info(f"Chunk processing complete: {len(successful_chunks)} successful, {len(failed_chunks)} failed out of {len(chunks)} total chunks")
        
        # Only fail if ALL chunks failed, not just some
        if len(successful_chunks) == 0:
            raise Exception(f"All {len(chunks)} chunks failed to process")
        elif len(failed_chunks) > 0:
            logger.warning(f"{len(failed_chunks)} chunks failed, but continuing with {len(successful_chunks)} successful chunks")
        
        # Step 3: Merge results and generate final summary (CRITICAL - don't mark complete until this is done)
        update_job_status(redis_client, job_id, {
            'message': f'All chunks processed ({len(successful_chunks)} successful). Combining text and generating comprehensive summary...',
            'progress': 80,
            'updated_at': get_timestamp()
        })
        
        merge_start = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'merging', start_time=merge_start)
        
        final_results = _merge_and_summarize_chunks(job_id, chunk_results, file_name)
        
        merge_end = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'merging', end_time=merge_end)
        
        # Update progress during file saving
        update_job_status(redis_client, job_id, {
            'message': 'Summary generated. Saving final results...',
            'progress': 90,
            'updated_at': get_timestamp()
        })
        
        # Save final results
        results_path = os.path.join('results', f'{job_id}_final_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # Also save in clean format to main job results folder
        main_job_results_dir = os.path.join("results", job_id)
        if os.path.exists(main_job_results_dir):
            # Save combined text to separate file
            combined_text_path = os.path.join(main_job_results_dir, "combined_text.txt")
            with open(combined_text_path, 'w', encoding='utf-8') as f:
                f.write(final_results["extracted_text"])
            
            # Create clean results similar to single document processing
            clean_chunk_results = {
                "filename": file_name,
                "job_id": job_id,
                "processing_completed_at": datetime.datetime.now().isoformat(),
                "total_pages": final_results["performance"]["total_pages"],
                "total_chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                "processing_method": "chunked_parallel",
                "summary": final_results["summary"],
                "date": final_results.get("date", "undated"),
                "extracted_info": final_results.get("extracted_info", {
                    "key_dates": "Not available",
                    "main_parties": "Not available", 
                    "case_reference_numbers": "Not available",
                    "full_analysis": final_results.get("extracted_text", "")
                }),
                "combined_text_path": f"results/{job_id}/combined_text.txt",
                "combined_text": final_results["extracted_text"],
                "estimated_cost": final_results["cost_info"]["estimated_cost"],
                "token_usage": final_results["cost_info"]["token_usage"],
                "processing_time_seconds": final_results["performance"]["total_processing_time"]["seconds"],
                "average_confidence_formatted": final_results["average_confidence_formatted"]
            }
            
            # Save clean results to main job folder
            clean_results_path = os.path.join(main_job_results_dir, "results.json")
            with open(clean_results_path, 'w', encoding='utf-8') as f:
                json.dump(clean_chunk_results, f, ensure_ascii=False, indent=2)
            
            # Create PPStructure results file for chunks
            ppstructure_chunk_results = {
                "job_id": job_id,
                "filename": file_name,
                "processing_completed_at": datetime.datetime.now().isoformat(),
                "total_pages": final_results["performance"]["total_pages"],
                "total_chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                "processing_method": "chunked_parallel",
                "chunk_results": chunk_results,
                "processing_info": {
                    "chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                    "failed_chunks": len([r for r in chunk_results if r.get('status') == 'failed']),
                    "chunk_size_pages": 20,
                    "processing_method": "chunked_parallel",
                    "docker_environment": True
                }
            }
            
            ppstructure_path = os.path.join(main_job_results_dir, "ppstructure_results.json")
            with open(ppstructure_path, 'w', encoding='utf-8') as f:
                json.dump(ppstructure_chunk_results, f, ensure_ascii=False, indent=2)
            
            # Create metrics file for chunks
            chunk_metrics = {
                "job_id": job_id,
                "filename": file_name,
                "processing_method": "chunked_parallel",
                "total_chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                "failed_chunks": len([r for r in chunk_results if r.get('status') == 'failed']),
                "total_processing_time": final_results["performance"]["total_processing_time"]["seconds"],
                "chunk_performance": final_results["performance"],
                "chunk_cost_breakdown": final_results["cost_info"],
                "chunk_details": chunk_results
            }
            
            metrics_path = os.path.join(main_job_results_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_metrics, f, ensure_ascii=False, indent=2)
        
        # Update progress before final completion
        update_job_status(redis_client, job_id, {
            'message': 'All files saved successfully. Processing complete!',
            'progress': 95,
            'updated_at': get_timestamp()
        })
        
        final_results['results_path'] = results_path
        
        logger.info(f"Large PDF processing completed for job {job_id}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error processing large document: {str(e)}")
        raise


def process_small_document(job_id, file_path, file_name):
    """
    Process small documents directly with PPStructure (no chunking).
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Processing small document {job_id} directly")
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'message': 'Processing document with PPStructure',
            'progress': 30,
            'updated_at': get_timestamp()
        })
        
        # Process directly with PPStructure (with summary generation)
        result = _get_ppstructure_function()(job_id, file_path, file_name, generate_summary=True)
        
        logger.info(f"Small document processing completed for job {job_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing small document: {str(e)}")
        raise


@shared_task(name='tasks.process_chunk')
def process_chunk(job_id, chunk_id, chunk_path):
    """
    Process an individual PDF chunk with PPStructure.
    
    Args:
        job_id: Unique job identifier
        chunk_id: Chunk identifier
        chunk_path: Path to the chunk PDF file
        
    Returns:
        dict: Chunk processing results WITH actual combined text
    """
    try:
        logger.info(f"Processing PDF chunk {chunk_id} for job {job_id}")
        
        # Load chunk metadata to get actual starting page number
        chunk_metadata = None
        job_chunk_dir = os.path.join('chunks', job_id)
        chunk_meta_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
        
        if os.path.exists(chunk_meta_path):
            try:
                with open(chunk_meta_path, 'r', encoding='utf-8') as f:
                    chunk_metadata = json.load(f)
                logger.info(f"Loaded chunk metadata: {chunk_metadata}")
            except Exception as e:
                logger.warning(f"Error loading chunk metadata: {str(e)}")
        
        # Get actual starting page number from metadata
        actual_start_page = 1  # Default fallback
        actual_end_page = 20   # Default fallback
        
        if chunk_metadata:
            actual_start_page = chunk_metadata.get('start_page', 1)
            actual_end_page = chunk_metadata.get('end_page', 20)
            logger.info(f"Chunk {chunk_id} covers actual pages {actual_start_page}-{actual_end_page}")
        
        # Update chunk status to processing
        update_chunk_status(job_id, chunk_id, 'processing')
        
        # Process chunk with PPStructure (NO summary generation for individual chunks)
        # Use the main job_id so all chunks go into the same folder
        chunk_filename = f"{chunk_id}_{os.path.basename(chunk_path)}"
        
        # Pass main job_id and actual starting page number to keep all chunks in the same folder
        result = _get_ppstructure_function()(
            job_id,  # Use main job_id instead of f"{job_id}_{chunk_id}"
            chunk_path, 
            chunk_filename,
            generate_summary=False,  # Important: Don't generate summary for individual chunks
            actual_start_page=actual_start_page  # Pass the actual starting page number
        )
        
        # Extract the actual combined text from the PPStructure result
        combined_text = ""
        if isinstance(result, dict):
            if 'combined_text' in result:
                combined_text = result['combined_text']
            elif 'extracted_text' in result:
                combined_text = result['extracted_text']
            
            # Try to load from results directory if not in direct result
            if not combined_text:
                results_dir = os.path.join("results", job_id)
                # Try to load chunk results from the results directory
                chunk_text_files = [
                    os.path.join(results_dir, "combined_text.txt"),
                    os.path.join(results_dir, chunk_id, "combined_text.txt"),
                    os.path.join(results_dir, "images", chunk_id, "combined_text.txt")  # Updated path without chunk_results
                ]
                
                for text_file in chunk_text_files:
                    if os.path.exists(text_file):
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                combined_text = f.read()
                            logger.info(f"Loaded combined text for chunk {chunk_id} from {text_file}: {len(combined_text)} characters")
                            break
                        except Exception as e:
                            logger.warning(f"Error reading {text_file}: {str(e)}")
                            continue
        
        # Prepare chunk result with the actual extracted text
        chunk_number = int(chunk_id.split('_')[1]) if '_' in chunk_id else 0
        chunk_result = {
            'chunk_id': chunk_id,
            'chunk_number': chunk_number,
            'status': 'completed',
            'combined_text': combined_text,
            'extracted_text': combined_text,  # Alias for compatibility
            'start_page': actual_start_page,
            'end_page': actual_end_page,
            'performance': result.get('performance', {}),
            'cost_info': result.get('cost_info', {'estimated_cost': 0.0}),
            'average_confidence_formatted': result.get('average_confidence_formatted', '0.0%'),
            'processing_time': result.get('processing_time', {'seconds': 0})
        }
        
        # Update chunk status to completed and save results
        update_chunk_status(job_id, chunk_id, 'completed', results=chunk_result)
        
        logger.info(f"Completed processing PDF chunk {chunk_id} for job {job_id} - extracted {len(combined_text)} characters")
        
        return chunk_result
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id} for job {job_id}: {str(e)}")
        
        # Update chunk status to failed
        update_chunk_status(job_id, chunk_id, 'failed', results={'error': str(e)})
        
        # Return error result
        chunk_number = int(chunk_id.split('_')[1]) if '_' in chunk_id else 0
        return {
            'chunk_id': chunk_id,
            'chunk_number': chunk_number,
            'status': 'failed',
            'error': str(e),
            'combined_text': '',
            'extracted_text': '',
            'start_page': actual_start_page,
            'end_page': actual_end_page,
            'performance': {},
            'cost_info': {'estimated_cost': 0.0}
        }


def _should_chunk_document(file_path, file_name):
    """
    Determine if a document should be chunked. 
    PDFs with more than 20 pages should be chunked into 20-page chunks.
    
    Args:
        file_path: Path to the document file
        file_name: Original file name
        
    Returns:
        bool: True if document should be chunked
    """
    try:
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Only chunk PDFs
        if file_ext != '.pdf':
            logger.info(f"Document {file_name} is not a PDF - will process directly")
            return False
        
        # Check PDF page count
        try:
            import fitz
            pdf_doc = fitz.open(file_path)
            page_count = pdf_doc.page_count
            pdf_doc.close()
            
            # Chunk PDFs with more than 20 pages into 20-page chunks
            if page_count > 20:
                logger.info(f"PDF has {page_count} pages (>20) - will chunk into 20-page chunks")
                return True
            else:
                logger.info(f"PDF has {page_count} pages (≤20) - will process directly")
                return False
                
        except Exception as e:
            logger.warning(f"Could not check PDF page count: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error determining chunking strategy: {str(e)}")
        # Default to direct processing if we can't determine
        return False


def _merge_and_summarize_chunks(job_id, chunk_results, file_name):
    """
    Merge chunk results and generate final summary using text_based_processor.py
    
    Args:
        job_id: Job identifier
        chunk_results: List of results from individual chunks
        file_name: Original filename
        
    Returns:
        dict: Final merged results with comprehensive summary
    """
    try:
        if not chunk_results:
            return {"status": "error", "message": "No chunk results to merge"}
        
        logger.info(f"Starting merge and summary for job {job_id} with {len(chunk_results)} chunks")
        
        # Combine all text from chunks - improved extraction
        combined_text_parts = []
        total_confidence_scores = []
        total_cost = 0.0
        total_pages = 0
        processing_times = []
        
        # Process each chunk result - extract ALL available text
        successful_chunks = 0
        
        # Sort chunks by chunk number to ensure proper order
        valid_chunk_results = [r for r in chunk_results if r.get('status') != 'failed']
        valid_chunk_results.sort(key=lambda x: x.get('chunk_number', x.get('start_page', 0)))
        
        for i, result in enumerate(valid_chunk_results):
            successful_chunks += 1
            chunk_num = result.get('chunk_number', i)
            start_page = result.get('start_page', (chunk_num * 20) + 1)
            end_page = result.get('end_page', (chunk_num + 1) * 20)
            
            # Extract combined text from chunk (this should be the full extracted text)
            chunk_text = ""
            if 'combined_text' in result and result['combined_text']:
                chunk_text = result['combined_text']
            elif 'extracted_text' in result and result['extracted_text']:
                chunk_text = result['extracted_text']
            elif isinstance(result, dict) and 'message' in result and 'Document processed successfully' in result['message']:
                # If this is a task result, try to load the actual results
                logger.info(f"Chunk {chunk_num + 1} returned task result, loading actual chunk results...")
                
                # Try to load the chunk results from the results directory
                chunk_results_dir = os.path.join("results", job_id)
                if os.path.exists(chunk_results_dir):
                    # Look for chunk-specific results or combined text
                    combined_text_path = os.path.join(chunk_results_dir, "combined_text.txt")
                    if os.path.exists(combined_text_path):
                        with open(combined_text_path, 'r', encoding='utf-8') as f:
                            chunk_text = f.read()
                        logger.info(f"Loaded chunk text from combined_text.txt: {len(chunk_text)} characters")
                    else:
                        # Try to find individual results files
                        results_path = os.path.join(chunk_results_dir, "results.json")
                        if os.path.exists(results_path):
                            with open(results_path, 'r', encoding='utf-8') as f:
                                chunk_data = json.load(f)
                                chunk_text = chunk_data.get('combined_text', '')
                            logger.info(f"Loaded chunk text from results.json: {len(chunk_text)} characters")
                        else:
                            logger.warning(f"No chunk results found for chunk {chunk_num + 1}")
                            continue
                else:
                    logger.warning(f"Results directory not found for chunk {chunk_num + 1}")
                    continue
            else:
                logger.warning(f"No text found in chunk {chunk_num + 1}")
                continue
            
            # Add chunk with clear separation and correct page numbers
            if chunk_text.strip():
                chunk_header = f"\n=== CHUNK {chunk_num + 1} (Pages {start_page}-{end_page}) ===\n"
                combined_text_parts.append(chunk_header + chunk_text.strip())
                logger.info(f"Added chunk {chunk_num + 1} (pages {start_page}-{end_page}): {len(chunk_text)} characters")
            
            # Collect confidence scores
            if 'average_confidence_formatted' in result:
                confidence_str = result['average_confidence_formatted'].replace('%', '')
                try:
                    confidence_val = float(confidence_str)
                    total_confidence_scores.append(confidence_val)
                except ValueError:
                    pass
            
            # Aggregate performance data
            performance = result.get('performance', {})
            if 'total_pages' in performance:
                total_pages += performance['total_pages']
            
            processing_time = result.get('processing_time', {})
            if 'seconds' in processing_time:
                processing_times.append(processing_time['seconds'])
            
            # Aggregate cost information
            cost_info = result.get('cost_info', {})
            if 'estimated_cost' in cost_info:
                total_cost += cost_info.get('estimated_cost', 0.0)
        
        # If we still don't have text, try to load ALL chunk results from the results directory
        if not combined_text_parts:
            logger.warning("No text found in chunk results, attempting to load from results directory...")
            
            # Try to collect all text from chunk subdirectories
            chunk_results_dir = os.path.join("results", job_id)
            if os.path.exists(chunk_results_dir):
                # Look for chunk subdirectories
                chunk_dirs = [d for d in os.listdir(chunk_results_dir) if d.startswith('chunk_') and os.path.isdir(os.path.join(chunk_results_dir, d))]
                chunk_dirs.sort()  # Sort to ensure proper order
                
                for chunk_dir in chunk_dirs:
                    chunk_path = os.path.join(chunk_results_dir, chunk_dir)
                    combined_text_path = os.path.join(chunk_path, "combined_text.txt")
                    
                    if os.path.exists(combined_text_path):
                        with open(combined_text_path, 'r', encoding='utf-8') as f:
                            chunk_text = f.read()
                        
                        if chunk_text.strip():
                            chunk_header = f"\n=== {chunk_dir.upper()} ===\n"
                            combined_text_parts.append(chunk_header + chunk_text.strip())
                            logger.info(f"Loaded text from {chunk_dir}: {len(chunk_text)} characters")
                            successful_chunks += 1
        
        # Combine all text for final document
        full_combined_text = '\n\n'.join(combined_text_parts)
        
        logger.info(f"Combined text from {successful_chunks} chunks: {len(full_combined_text)} characters total")
        
        if not full_combined_text.strip():
            logger.error("No text extracted from any chunks!")
            return {
                "status": "error", 
                "message": "No text could be extracted from any chunks",
                "extracted_text": "",
                "summary": "Summary generation failed - no text extracted"
            }
        
        # Generate final summary using ALL combined text from ALL chunks
        logger.info(f"Starting comprehensive summary generation for job {job_id} using {len(full_combined_text)} characters")
        
        try:
            summary_result = text_processor.summarize_document_text(full_combined_text, file_name)
            logger.info(f"✅ SUMMARY GENERATION COMPLETED for job {job_id} - Generated summary with {len(summary_result.get('summary', ''))} characters")
        except Exception as e:
            logger.error(f"Error generating final summary for job {job_id}: {str(e)}")
            summary_result = {
                "summary": f"Document processed with {successful_chunks} chunks. Summary generation failed: {str(e)}",
                "date": "undated",
                "extracted_info": {
                    "key_dates": "Not available", 
                    "main_parties": "Not available",
                    "case_reference_numbers": "Not available",
                    "full_analysis": f"Error generating analysis: {str(e)}"
                },
                "estimated_cost": 0.0,
                "usage_info": {"total_tokens": 0}
            }
        
        # Calculate overall metrics
        average_confidence = sum(total_confidence_scores) / len(total_confidence_scores) if total_confidence_scores else 0
        total_processing_time = sum(processing_times) if processing_times else 0
        
        # Create final results with proper structure
        final_results = {
            "extracted_text": full_combined_text,
            "average_confidence_formatted": f"{average_confidence:.1f}%",
            "summary": summary_result.get("summary", "Summary not available"),
            "date": summary_result.get("date", "undated"),
            "extracted_info": summary_result.get("extracted_info", {}),
            "performance": {
                "total_chunks": successful_chunks,
                "total_pages": total_pages,
                "total_processing_time": {
                    "seconds": total_processing_time,
                    "formatted": f"{total_processing_time:.1f}s"
                },
                "processing_method": "chunked_parallel"
            },
            "cost_info": {
                "estimated_cost": total_cost + summary_result.get("estimated_cost", 0.0),
                "token_usage": summary_result.get("usage_info", {}),
                "chunks_cost": total_cost,
                "summary_cost": summary_result.get("estimated_cost", 0.0)
            },
            "status": "completed",
            "job_id": job_id
        }
        
        logger.info(f"✅ Successfully merged {successful_chunks} chunks and generated final summary for job {job_id}")
        return final_results
        
    except Exception as e:
        logger.error(f"Error merging and summarizing chunks for job {job_id}: {str(e)}")
        return {
            "status": "error", 
            "message": f"Error merging results: {str(e)}",
            "extracted_text": "Error occurred during processing",
            "summary": "Summary generation failed",
            "extracted_info": {}
        } 