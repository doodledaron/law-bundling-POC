"""
Celery tasks for document processing.
Handles both small documents (≤100 pages) and large documents (>100 pages with chunking).
"""
import os
import json
import time
import gc
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
from celery import Celery, shared_task
from celery.utils.log import get_task_logger
from celery.exceptions import SoftTimeLimitExceeded
import redis

# Enhanced multiprocessing support for CUDA-safe parallel processing
import multiprocessing
import functools

# Set multiprocessing start method for cross-platform compatibility 
# (especially important for Windows and CUDA)
if __name__ != '__main__':
    # This ensures compatibility with Celery and Windows
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

# Text processing imports  
from tasks.utils import (
    get_timestamp, 
    update_job_status
)

# Import processing and chunking tasks
# NOTE: ppstructure_tasks is called via Celery to avoid loading models in this worker
from tasks.chunking_tasks import (
    create_document_chunks, 
    update_chunk_status
)
from tasks.utils import (
    get_unix_timestamp,
    update_job_timing,
    calculate_duration
)

# Import text processor for Google Genai processing (this worker has Google Genai dependencies)
try:
    from text_based_processor import TextBasedProcessor
    GENAI_AVAILABLE = True
except ImportError as e:
    logger.error(f"Google Genai not available: {str(e)}")
    GENAI_AVAILABLE = False
    TextBasedProcessor = None

logger = get_task_logger(__name__)

logger.info("🚀 Visualizations disabled for faster processing")

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Initialize text processor if available
text_processor = TextBasedProcessor() if GENAI_AVAILABLE else None

def cleanup_temporary_files(job_id, file_path, file_name):
    """
    Clean up temporary files after processing is complete.
    Keeps only essential results: results.json, metrics.json, ppstructure_results.json, and combined_text.txt
    
    Args:
        job_id: Job identifier
        file_path: Original file path
        file_name: Original file name
    """
    try:
        import shutil
        
        logger.info(f"Starting cleanup of temporary files for job {job_id}")
        
        # 1. Remove original uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                # logger.info(f"Removed uploaded file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {file_path}: {str(e)}")
        
        # 2. Remove chunk files and directory
        chunk_dir = os.path.join('chunks', job_id)
        if os.path.exists(chunk_dir):
            try:
                shutil.rmtree(chunk_dir)
                # logger.info(f"Removed chunk directory: {chunk_dir}")
            except Exception as e:
                logger.warning(f"Could not remove chunk directory {chunk_dir}: {str(e)}")
        
        # 3. Clean up results directory - keep only essential files
        results_dir = os.path.join('results', job_id)
        if os.path.exists(results_dir):
            # Files to keep
            keep_files = {
                'results.json',
                'metrics.json', 
                'ppstructure_results.json',
                'combined_text.txt'
            }
            
            # Remove temporary subdirectories
            temp_dirs = ['images', 'visualizations', 'tables', 'figures', 'page_results']
            for temp_dir in temp_dirs:
                temp_path = os.path.join(results_dir, temp_dir)
                if os.path.exists(temp_path):
                    try:
                        shutil.rmtree(temp_path)
                        # logger.info(f"Removed temporary directory: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove directory {temp_path}: {str(e)}")
            
            # Remove any other files except the ones we want to keep
            try:
                for item in os.listdir(results_dir):
                    item_path = os.path.join(results_dir, item)
                    if os.path.isfile(item_path) and item not in keep_files:
                        os.remove(item_path)
                        # logger.info(f"Removed temporary file: {item_path}")
                    elif os.path.isdir(item_path):
                        # Remove any remaining subdirectories (like chunk_xxxx dirs)
                        shutil.rmtree(item_path)
                        # logger.info(f"Removed temporary subdirectory: {item_path}")
            except Exception as e:
                logger.warning(f"Error cleaning results directory: {str(e)}")
        
        # 4. Clean up any empty parent directories
        try:
            # Remove chunks directory if empty
            chunks_parent = 'chunks'
            if os.path.exists(chunks_parent) and not os.listdir(chunks_parent):
                os.rmdir(chunks_parent)
                # logger.info(f"Removed empty chunks directory")
        except Exception as e:
            logger.warning(f"Could not remove empty directories: {str(e)}")
        
        logger.info(f"Cleanup completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error during cleanup for job {job_id}: {str(e)}")
        # Don't raise - cleanup failures shouldn't fail the job

def _call_ppstructure_worker_async(job_id, file_path, file_name, **kwargs):
    """
    Call PPStructure processing asynchronously via Celery, returning the AsyncResult.
    This avoids the Celery task-within-task violation.
    """
    from celery import current_app
    
    try:
        # Convert relative path to absolute path for cross-container access
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        logger.info(f"Submitting PPStructure task for job {job_id}, file: {file_path}")
        
        # Call PPStructure task in the dedicated PPStructure worker (async)
        ppstructure_task = current_app.send_task(
            'tasks.process_document_with_ppstructure',
            args=[job_id, file_path, file_name],
            kwargs=kwargs,
            queue='ppstructure'
        )
        
        return ppstructure_task
        
    except Exception as e:
        logger.error(f"Failed to submit PPStructure task: {str(e)}")
        raise RuntimeError(f"PPStructure task submission failed: {str(e)}")


def _wait_for_ppstructure_result(ppstructure_task, job_id, timeout=1800):
    """
    Wait for PPStructure task completion using polling instead of .get().
    This avoids the Celery violation while still getting the result.
    """
    import time
    
    start_time = time.time()
    poll_interval = 5  # Poll every 5 seconds
    
    logger.info(f"Waiting for PPStructure completion for job {job_id} (timeout: {timeout}s)")
    
    while time.time() - start_time < timeout:
        try:
            if ppstructure_task.ready():
                if ppstructure_task.successful():
                    result = ppstructure_task.result
                    logger.info(f"PPStructure completed successfully for job {job_id}")
                    return result
                else:
                    # Task failed
                    error = ppstructure_task.result
                    logger.error(f"PPStructure failed for job {job_id}: {error}")
                    raise RuntimeError(f"PPStructure processing failed: {error}")
            
            # Not ready yet, wait and poll again
            time.sleep(poll_interval)
            
        except Exception as e:
            logger.error(f"Error checking PPStructure task status: {str(e)}")
            raise RuntimeError(f"PPStructure monitoring failed: {str(e)}")
    
    # Timeout reached
    logger.error(f"PPStructure timeout after {timeout}s for job {job_id}")
    raise RuntimeError(f"PPStructure processing timeout after {timeout} seconds")

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
        # NOTE: Chunking is effectively disabled (threshold set to 200 pages)
        if _should_chunk_document(file_path, file_name):
            logger.info(f"Large PDF detected for job {job_id} - using chunked processing (>200 pages)")
            result = process_large_document(job_id, file_path, file_name)
        else:
            logger.info(f"Document detected for job {job_id} - using direct processing (≤200 pages, chunking disabled)")
            result = process_small_document(job_id, file_path, file_name)
        
        # Update job status with completion
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'message': 'Document processing completed successfully',
            'results_path': result.get('results_path', f'results/{job_id}_final_results.json'),
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
        
        # Clean up temporary files even on failure
        # try:
        #     cleanup_temporary_files(job_id, file_path, file_name)
        # except Exception as cleanup_error:
        #     logger.warning(f"Cleanup failed after main error: {str(cleanup_error)}")
        
        # Re-raise to allow Celery to handle the error
        raise


def process_large_document(job_id, file_path, file_name):
    """
    Process large PDFs by chunking into 20-page chunks and processing in parallel.
    Enhanced with comprehensive memory management and parallel processing.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the PDF file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Processing large document {job_id} with parallel chunking")
        
        # Step 1: Create chunks
        update_job_status(redis_client, job_id, {
            'message': 'Creating PDF chunks for parallel processing',
            'progress': 10,
            'updated_at': get_timestamp()
        })
        
        chunking_start = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'chunking', start_time=chunking_start)
        
        chunk_info = create_document_chunks(job_id, file_path, file_name)
        
        if not chunk_info or chunk_info.get('status') == 'error':
            raise Exception(f"Failed to create chunks: {chunk_info.get('message', 'Unknown error')}")
        
        chunking_end = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'chunking', end_time=chunking_end)
        
        # Store chunk info in Redis for coordination
        redis_client.set(f"chunk_info:{job_id}", json.dumps(chunk_info), ex=3600)
        
        logger.info(f"Created {chunk_info['num_chunks']} PDF chunks for job {job_id}")
        
        # Step 2: Process chunks in parallel with enhanced memory and coordination
        update_job_status(redis_client, job_id, {
            'message': f'Processing {chunk_info["num_chunks"]} PDF chunks in parallel with memory optimization',
            'progress': 30,
            'updated_at': get_timestamp(),
            'num_chunks': chunk_info['num_chunks'],
            'processing_method': 'parallel_optimized'
        })
        
        processing_start = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'parallel_processing', start_time=processing_start)
        
        chunks = chunk_info['chunks']
        
        # Enhanced parallel processing with memory management
        chunk_results = _process_chunks_parallel_enhanced(job_id, chunks)
        
        processing_end = get_unix_timestamp()
        update_job_timing(redis_client, job_id, 'parallel_processing', end_time=processing_end)
        
        # Validate chunk results
        successful_chunks = [r for r in chunk_results if r.get('status') != 'failed']
        failed_chunks = [r for r in chunk_results if r.get('status') == 'failed']
        
        logger.info(f"Parallel processing complete: {len(successful_chunks)} successful, {len(failed_chunks)} failed out of {len(chunks)} total chunks")
        
        # Only fail if ALL chunks failed
        if len(successful_chunks) == 0:
            raise Exception(f"All {len(chunks)} chunks failed to process")
        elif len(failed_chunks) > 0:
            logger.warning(f"{len(failed_chunks)} chunks failed, but continuing with {len(successful_chunks)} successful chunks")
        
        # Step 3: Merge results and generate final summary
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
        
        # Add results_path to final_results
        final_results['results_path'] = results_path
        
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
                "processing_completed_at": datetime.now().isoformat(),
                "total_pages": final_results["performance"]["total_pages"],
                "total_chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                "processing_method": "chunked_parallel_enhanced",
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
                "processing_completed_at": datetime.now().isoformat(),
                "total_pages": final_results["performance"]["total_pages"],
                "total_chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                "processing_method": "chunked_parallel_enhanced",
                "chunk_results": chunk_results,
                "processing_info": {
                    "chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                    "failed_chunks": len([r for r in chunk_results if r.get('status') == 'failed']),
                    "chunk_size_pages": 20,
                    "processing_method": "chunked_parallel_enhanced",
                    "docker_environment": True,
                    "parallel_optimization": True,
                    "memory_management": True
                }
            }
            
            ppstructure_path = os.path.join(main_job_results_dir, "ppstructure_results.json")
            with open(ppstructure_path, 'w', encoding='utf-8') as f:
                json.dump(ppstructure_chunk_results, f, ensure_ascii=False, indent=2)
            
            # Create metrics file for chunks
            chunk_metrics = {
                "job_id": job_id,
                "filename": file_name,
                "processing_method": "chunked_parallel_enhanced",
                "total_chunks": len([r for r in chunk_results if r.get('status') != 'failed']),
                "failed_chunks": len([r for r in chunk_results if r.get('status') == 'failed']),
                "total_processing_time": final_results["performance"]["total_processing_time"]["seconds"],
                "chunk_performance": final_results["performance"],
                "chunk_cost_breakdown": final_results["cost_info"],
                "chunk_details": chunk_results,
                "parallel_metrics": {
                    "chunks_processed_in_parallel": True,
                    "memory_optimization": True,
                    "redis_coordination": True
                }
            }
            
            metrics_path = os.path.join(main_job_results_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_metrics, f, ensure_ascii=False, indent=2)
        
        # Update job status to completed
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'message': 'Document processing completed successfully',
            'progress': 100,
            'updated_at': get_timestamp(),
            'results_path': results_path
        })
        
        # Clean up chunk info from Redis
        redis_client.delete(f"chunk_info:{job_id}")
        
        # Clean up temporary files (keep only essential results)
        # cleanup_temporary_files(job_id, file_path, file_name)
        
        logger.info(f"Large document processing completed for job {job_id}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error processing large document: {str(e)}")
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Processing failed: {str(e)}',
            'updated_at': get_timestamp()
        })
        
        # Clean up temporary files even on failure
        # try:
        #     cleanup_temporary_files(job_id, file_path, file_name)
        # except Exception as cleanup_error:
        #     logger.warning(f"Cleanup failed after error: {str(cleanup_error)}")
        
        # raise


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
        
        # Process directly with PPStructure (OCR and layout detection only)
        ppstructure_task = _call_ppstructure_worker_async(
            job_id, 
            file_path, 
            file_name, 
            generate_summary=False,  # Summary generation moved to document_tasks.py
            enable_visualizations=True,  # Disabled for speed
        )
        
        # Wait for PPStructure completion
        result = _wait_for_ppstructure_result(ppstructure_task, job_id)
        
        # Process extracted images with Genai (if any were extracted)
        if result and result.get('extraction_metadata'):
            update_job_status(redis_client, job_id, {
                'message': 'Processing extracted tables/figures/charts with Genai AI',
                'progress': 50,
                'updated_at': get_timestamp()
            })
            
            genai_extractions = process_extracted_images_with_genai(job_id, result['extraction_metadata'])
            
            # Add Genai extractions to the combined text
            if genai_extractions['total_processed'] > 0:
                additional_text = ""
                if genai_extractions['table_extractions']:
                    additional_text += "\n\n" + "\n".join(genai_extractions['table_extractions'])
                if genai_extractions['figure_extractions']:
                    additional_text += "\n\n" + "\n".join(genai_extractions['figure_extractions'])
                if genai_extractions['chart_extractions']:
                    additional_text += "\n\n" + "\n".join(genai_extractions['chart_extractions'])
                
                # Update the combined text in the result
                if 'combined_text' in result:
                    result['combined_text'] += additional_text
                if 'extracted_text' in result:
                    result['extracted_text'] += additional_text
                
                # Update the saved combined text file
                results_dir = os.path.join("results", job_id)
                combined_text_path = os.path.join(results_dir, "combined_text.txt")
                if os.path.exists(combined_text_path):
                    with open(combined_text_path, 'a', encoding='utf-8') as f:
                        f.write(additional_text)
                
                logger.info(f"Added {genai_extractions['total_processed']} Genai extractions to document text")
            else:
                logger.info("No additional extractions from Genai processing")
        
        # Generate document summary with Genai (moved from ppstructure_tasks.py)
        update_job_status(redis_client, job_id, {
            'message': 'Generating comprehensive document summary with Google AI',
            'progress': 80,
            'updated_at': get_timestamp()
        })
        
        combined_text = result.get('combined_text', '') if result else ''
        if combined_text.strip() and GENAI_AVAILABLE and text_processor is not None:
            try:
                logger.info(f"Starting summary generation for job {job_id} with {len(combined_text)} characters")
                summary_result = text_processor.summarize_document_text(combined_text, file_name)
                
                # Update the result with summary information
                if result:
                    result['summary'] = summary_result.get('summary', 'Summary not available')
                    result['extracted_info'] = summary_result.get('extracted_info', {
                        "key_dates": "Not available",
                        "main_parties": "Not available", 
                        "case_reference_numbers": "Not available",
                        "full_analysis": "Summary generation failed"
                    })
                    # Update cost information
                    if 'cost_info' in result:
                        result['cost_info']['estimated_cost'] += summary_result.get('estimated_cost', 0.0)
                        result['cost_info']['summary_cost'] = summary_result.get('estimated_cost', 0.0)
                    else:
                        result['cost_info'] = {
                            'estimated_cost': summary_result.get('estimated_cost', 0.0),
                            'summary_cost': summary_result.get('estimated_cost', 0.0),
                            'token_usage': summary_result.get('usage_info', {})
                        }
                    
                    # Update results.json with summary
                    results_dir = os.path.join("results", job_id)
                    results_path = os.path.join(results_dir, "results.json")
                    if os.path.exists(results_path):
                        with open(results_path, 'r', encoding='utf-8') as f:
                            saved_results = json.load(f)
                        
                        saved_results['summary'] = summary_result.get('summary', 'Summary not available')
                        saved_results['extracted_info'] = summary_result.get('extracted_info', {})
                        saved_results['estimated_cost'] += summary_result.get('estimated_cost', 0.0)
                        
                        with open(results_path, 'w', encoding='utf-8') as f:
                            json.dump(saved_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ Summary generation completed for job {job_id}")
                
            except Exception as e:
                logger.error(f"Error generating summary for job {job_id}: {str(e)}")
                if result:
                    result['summary'] = f"Summary generation failed: {str(e)}"
                    result['extracted_info'] = {
                        "key_dates": "Not available",
                        "main_parties": "Not available", 
                        "case_reference_numbers": "Not available",
                        "full_analysis": f"Summary generation error: {str(e)}"
                    }
        elif not GENAI_AVAILABLE:
            logger.warning(f"Google Genai not available for summary generation for job {job_id}")
            if result:
                result['summary'] = "Summary generation not available - Google Genai not configured"
                result['extracted_info'] = {
                    "key_dates": "Not available",
                    "main_parties": "Not available", 
                    "case_reference_numbers": "Not available",
                    "full_analysis": "Summary generation requires Google Genai"
                }
        else:
            logger.warning(f"No text available for summary generation for job {job_id}")
            if result:
                result['summary'] = "No text available for summary generation"
        
        # Clean up temporary files (keep only essential results)
        # cleanup_temporary_files(job_id, file_path, file_name)
        
        logger.info(f"Small document processing completed for job {job_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing small document: {str(e)}")
        
        # Clean up temporary files even on failure
        # try:
        #     cleanup_temporary_files(job_id, file_path, file_name)
        # except Exception as cleanup_error:
        #     logger.warning(f"Cleanup failed after error: {str(cleanup_error)}")
        
        # raise



def _extract_text_from_result(result, job_id, chunk_id):
    """
    Enhanced text extraction from PPStructure result with multiple fallback strategies.
    
    Args:
        result: PPStructure processing result
        job_id: Job identifier
        chunk_id: Chunk identifier
        
    Returns:
        str: Extracted text
    """
    combined_text = ""
    
    if isinstance(result, dict):
        # Try direct extraction from result
        if 'combined_text' in result:
            combined_text = result['combined_text']
        elif 'extracted_text' in result:
            combined_text = result['extracted_text']
        
        # Try loading from results directory with multiple paths
        if not combined_text:
            results_dir = os.path.join("results", job_id)
            chunk_text_files = [
                os.path.join(results_dir, "combined_text.txt"),
                os.path.join(results_dir, chunk_id, "combined_text.txt"),
                os.path.join(results_dir, "images", chunk_id, "combined_text.txt"),
                os.path.join(results_dir, "chunks", chunk_id, "combined_text.txt")
            ]
            
            for text_file in chunk_text_files:
                if os.path.exists(text_file):
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            combined_text = f.read()
                        logger.info(f"[Text Extraction] Loaded text for {chunk_id} from {text_file}: {len(combined_text)} chars")
                        break
                    except Exception as e:
                        logger.warning(f"[Text Extraction] Error reading {text_file}: {str(e)}")
                        continue
    
    if not combined_text:
        logger.warning(f"[Text Extraction] No text extracted for chunk {chunk_id}")
    
    return combined_text


def _process_chunks_parallel_enhanced(job_id, chunks):
    """
    Process chunks sequentially in the main process (CUDA-safe).
    Completely avoids multiprocessing to eliminate segmentation faults.
    
    Args:
        job_id: Job identifier
        chunks: List of chunk information
        
    Returns:
        list: Results from all chunks
    """
    logger.info(f"🚀 Starting pure sequential processing for {len(chunks)} chunks (no multiprocessing)")
    
    chunk_results = []
    total_chunks = len(chunks)
    
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{total_chunks}: {chunk['chunk_id']}")
        
        # Update status for current chunk with detailed progress
        progress = 30 + (40 * chunk_idx / total_chunks)
        update_job_status(redis_client, job_id, {
            'message': f'Processing chunk {chunk_idx + 1}/{total_chunks} ({chunk["chunk_id"]}) - pure sequential',
            'progress': int(progress),
            'updated_at': get_timestamp(),
            'current_chunk': chunk_idx + 1,
            'total_chunks': total_chunks,
            'processing_method': 'pure_sequential'
        })
        
        # Process single chunk directly in main process
        try:
            logger.info(f"📄 Starting processing for chunk {chunk['chunk_id']} ({chunk_idx + 1}/{total_chunks})")
            
            chunk_start_time = time.time()
            
            result = _process_single_chunk_internal(
                job_id,
                chunk['chunk_id'],
                chunk['path'],
                chunk_idx
            )
            
            chunk_processing_time = time.time() - chunk_start_time
            
            if result:
                chunk_results.append(result)
                text_length = len(result.get('combined_text', ''))
                logger.info(f"✅ Completed chunk {chunk['chunk_id']} ({chunk_idx + 1}/{total_chunks}) in {chunk_processing_time:.1f}s - extracted {text_length:,} characters")
            else:
                logger.warning(f"⚠️  Empty result for chunk {chunk['chunk_id']}")
                
        except Exception as chunk_error:
            logger.error(f"❌ Error processing chunk {chunk['chunk_id']}: {str(chunk_error)}")
            
            # Create error result
            error_result = {
                'chunk_id': chunk['chunk_id'],
                'chunk_number': chunk_idx,
                'status': 'failed',
                'error': str(chunk_error),
                'combined_text': '',
                'extracted_text': '',
                'start_page': (chunk_idx * 20) + 1,
                'end_page': (chunk_idx + 1) * 20,
                'performance': {},
                'cost_info': {'estimated_cost': 0.0}
            }
            chunk_results.append(error_result)
            
        # Memory cleanup after each chunk
        gc.collect()
        time.sleep(0.5)  # Brief pause to allow GPU memory cleanup
        
        # Log progress every chunk
        completed_chunks = len([r for r in chunk_results if r.get('status') != 'failed'])
        logger.info(f"📊 Progress: {completed_chunks}/{total_chunks} chunks completed successfully")
    
    successful_chunks = len([r for r in chunk_results if r.get('status') != 'failed'])
    logger.info(f"🎉 Pure sequential processing completed: {successful_chunks}/{total_chunks} chunks successful")
    
    return chunk_results


def _process_chunk_with_multiprocessing(job_id, chunk_data):
    """
    Process a single chunk in a separate process (CUDA-safe).
    This function runs in its own process with its own CUDA context.
    Includes comprehensive error handling for segmentation faults.
    
    Args:
        job_id: Job identifier  
        chunk_data: Dictionary with chunk_id, chunk_path, chunk_index
        
    Returns:
        dict: Chunk processing results
    """
    import signal
    import sys
    
    def signal_handler(signum, frame):
        logger.error(f"[Multiprocessing] Signal {signum} received in chunk processing")
        sys.exit(1)
    
    # Set up signal handlers for segmentation faults
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGFPE, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)
    
    try:
        chunk_id = chunk_data['chunk_id']
        chunk_path = chunk_data['chunk_path']
        chunk_index = chunk_data['chunk_index']
        
        logger.info(f"[Multiprocessing] Processing chunk {chunk_id} (index {chunk_index}) for job {job_id}")
        
        # Force garbage collection before processing
        import gc
        gc.collect()
        
        # Set environment variables for CUDA safety in multiprocessing
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure single GPU
        os.environ['PADDLE_TRAINERS_NUM'] = '1'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        # Additional environment variables for stability
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous CUDA operations
        os.environ['PADDLE_DISABLE_ASYNC_EXECUTOR'] = '1'  # Disable async execution
        
        # Each process initializes its own CUDA context with timeout
        import multiprocessing
        import time
        
        # Create a timeout wrapper
        def timeout_wrapper():
            try:
                return _process_single_chunk_internal(job_id, chunk_id, chunk_path, chunk_index)
            except Exception as e:
                logger.error(f"[Multiprocessing] Timeout wrapper caught: {str(e)}")
                raise
        
        # Use a simple timeout approach
        start_time = time.time()
        result = timeout_wrapper()
        processing_time = time.time() - start_time
        
        logger.info(f"[Multiprocessing] Completed chunk {chunk_id} in {processing_time:.2f}s")
        
        # Force cleanup
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"[Multiprocessing] Error processing chunk {chunk_data.get('chunk_id', 'unknown')}: {str(e)}")
        
        # Force cleanup on error
        try:
            import gc
            gc.collect()
        except:
            pass
            
        return {
            'chunk_id': chunk_data.get('chunk_id', 'unknown'),
            'chunk_number': chunk_data.get('chunk_index', 0),
            'status': 'failed',
            'error': f"Multiprocessing error: {str(e)}",
            'combined_text': '',
            'extracted_text': '',
            'start_page': (chunk_data.get('chunk_index', 0) * 20) + 1,
            'end_page': (chunk_data.get('chunk_index', 0) + 1) * 20,
            'performance': {},
            'cost_info': {'estimated_cost': 0.0}
        }


def _process_single_chunk_internal(job_id, chunk_id, chunk_path, chunk_index):
    """
    Process a single chunk internally using direct function call instead of Celery task.
    This runs within the ThreadPoolExecutor for internal parallelism.
    
    Args:
        job_id: Unique job identifier
        chunk_id: Chunk identifier
        chunk_path: Path to the chunk PDF file
        chunk_index: Index of chunk in processing order
        
    Returns:
        dict: Chunk processing results
    """
    try:
        logger.info(f"[Internal] Processing chunk {chunk_id} (index {chunk_index}) for job {job_id}")
        
        # Set up task monitoring
        task_start_time = time.time()
        
        # Load chunk metadata
        chunk_metadata = None
        job_chunk_dir = os.path.join('chunks', job_id)
        chunk_meta_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
        
        if os.path.exists(chunk_meta_path):
            try:
                with open(chunk_meta_path, 'r', encoding='utf-8') as f:
                    chunk_metadata = json.load(f)
                logger.info(f"[Internal] Loaded metadata for chunk {chunk_id}: {chunk_metadata}")
            except Exception as e:
                logger.warning(f"[Internal] Error loading chunk metadata: {str(e)}")
        
        # Get actual page numbers
        actual_start_page = 1
        actual_end_page = 20
        
        if chunk_metadata:
            actual_start_page = chunk_metadata.get('start_page', 1)
            actual_end_page = chunk_metadata.get('end_page', 20)
            logger.info(f"[Internal] Chunk {chunk_id} covers pages {actual_start_page}-{actual_end_page}")
        
        # Update chunk status
        update_chunk_status(job_id, chunk_id, 'processing_internal')
        
        # Update Redis with detailed chunk progress
        redis_client.setex(
            f"chunk_progress:{job_id}:{chunk_id}",
            300,  # 5 minute expiry
            json.dumps({
                'status': 'processing',
                'started_at': get_timestamp(),
                'worker_type': 'internal_parallel',
                'chunk_index': chunk_index,
                'pages': f"{actual_start_page}-{actual_end_page}"
            })
        )
        
        # Process with PPStructure (enhanced error handling)
        chunk_filename = f"{chunk_id}_{os.path.basename(chunk_path)}"
        
        try:
            # Update detailed progress
            redis_client.setex(
                f"chunk_progress:{job_id}:{chunk_id}",
                300,
                json.dumps({
                    'status': 'starting_ppstructure',
                    'message': f'Loading PPStructure models for pages {actual_start_page}-{actual_end_page}',
                    'progress': 10
                })
            )
            logger.info(f"[Internal] Starting PPStructure processing for chunk {chunk_id} (pages {actual_start_page}-{actual_end_page})")
            
            # Update progress
            redis_client.setex(
                f"chunk_progress:{job_id}:{chunk_id}",
                300,
                json.dumps({
                    'status': 'processing_pages',
                    'message': f'Processing {actual_end_page - actual_start_page + 1} pages with OCR and layout detection',
                    'progress': 50
                })
            )
            
            try:
                # Process chunk with memory monitoring and speed optimizations via PPStructure worker
                ppstructure_task = _call_ppstructure_worker_async(
                    job_id,
                    chunk_path, 
                    chunk_filename,
                    generate_summary=False,
                    actual_start_page=actual_start_page,
                    enable_visualizations=True  # Disabled for speed
                )
                
                # Wait for PPStructure completion
                result = _wait_for_ppstructure_result(ppstructure_task, job_id)
                
                # Process extracted images with Genai (if any were extracted)
                if result and result.get('extraction_metadata'):
                    # Update detailed progress
                    redis_client.setex(
                        f"chunk_progress:{job_id}:{chunk_id}",
                        300,
                        json.dumps({
                            'status': 'processing_genai',
                            'message': f'Processing extracted images with Genai AI for pages {actual_start_page}-{actual_end_page}',
                            'progress': 80
                        })
                    )
                    
                    genai_extractions = process_extracted_images_with_genai(job_id, result['extraction_metadata'])
                    
                    # Add Genai extractions to the combined text
                    if genai_extractions['total_processed'] > 0:
                        additional_text = ""
                        if genai_extractions['table_extractions']:
                            additional_text += "\n\n" + "\n".join(genai_extractions['table_extractions'])
                        if genai_extractions['figure_extractions']:
                            additional_text += "\n\n" + "\n".join(genai_extractions['figure_extractions'])
                        if genai_extractions['chart_extractions']:
                            additional_text += "\n\n" + "\n".join(genai_extractions['chart_extractions'])
                        
                        # Update the combined text in the result
                        if 'combined_text' in result:
                            result['combined_text'] += additional_text
                        if 'extracted_text' in result:
                            result['extracted_text'] += additional_text
                        
                        logger.info(f"Added {genai_extractions['total_processed']} Genai extractions to chunk {chunk_id}")
                    else:
                        logger.info(f"No additional extractions from Genai processing for chunk {chunk_id}")
                
                logger.info(f"[Internal] PPStructure completed for chunk {chunk_id}")
                
                # Update progress
                redis_client.setex(
                    f"chunk_progress:{job_id}:{chunk_id}",
                    300,
                    json.dumps({
                        'status': 'extracting_text',
                        'message': 'Extracting and combining text from OCR results',
                        'progress': 90
                    })
                )
                
            except Exception as proc_error:
                logger.error(f"[Internal] PPStructure processing failed for chunk {chunk_id}: {str(proc_error)}")
                raise Exception(f"PPStructure processing failed: {str(proc_error)}")
        except Exception as e:
            logger.error(f"[Internal] Error processing chunk {chunk_id}: {str(e)}")
            raise Exception(f"Error processing chunk {chunk_id}: {str(e)}")
        
        # Extract combined text with enhanced fallback
        combined_text = _extract_text_from_result(result, job_id, chunk_id)
        
        # Prepare chunk result
        chunk_number = int(chunk_id.split('_')[1]) if '_' in chunk_id else chunk_index
        chunk_result = {
            'chunk_id': chunk_id,
            'chunk_number': chunk_number,
            'chunk_index': chunk_index,
            'status': 'completed',
            'combined_text': combined_text,
            'extracted_text': combined_text,
            'start_page': actual_start_page,
            'end_page': actual_end_page,
            'performance': result.get('performance', {}),
            'cost_info': result.get('cost_info', {'estimated_cost': 0.0}),
            'average_confidence_formatted': result.get('average_confidence_formatted', '0.0%'),
            'processing_time': result.get('processing_time', {'seconds': 0}),
            'worker_info': {
                'worker_type': 'internal_parallel',
                'task_start_time': task_start_time,
                'task_end_time': time.time()
            }
        }
        
        # Update chunk status to completed
        update_chunk_status(job_id, chunk_id, 'completed_internal', results=chunk_result)
        
        # Update Redis progress
        redis_client.setex(
            f"chunk_progress:{job_id}:{chunk_id}",
            300,
            json.dumps({
                'status': 'completed',
                'completed_at': get_timestamp(),
                'text_length': len(combined_text),
                'worker_type': 'internal_parallel'
            })
        )
        
        # Memory cleanup
        del result
        gc.collect()
        
        processing_time = time.time() - task_start_time
        logger.info(f"[Internal] Completed chunk {chunk_id} in {processing_time:.2f}s - extracted {len(combined_text)} characters")
        
        return chunk_result
        
    except Exception as e:
        logger.error(f"[Internal] Error processing chunk {chunk_id}: {str(e)}")
        
        # Update chunk status to failed
        update_chunk_status(job_id, chunk_id, 'failed_internal', results={'error': str(e)})
        
        # Update Redis with failure
        redis_client.setex(
            f"chunk_progress:{job_id}:{chunk_id}",
            300,
            json.dumps({
                'status': 'failed',
                'failed_at': get_timestamp(),
                'error': str(e),
                'worker_type': 'internal_parallel'
            })
        )
        
        # Return error result
        chunk_number = int(chunk_id.split('_')[1]) if '_' in chunk_id else chunk_index
        return {
            'chunk_id': chunk_id,
            'chunk_number': chunk_number,
            'chunk_index': chunk_index,
            'status': 'failed',
            'error': str(e),
            'combined_text': '',
            'extracted_text': '',
            'start_page': actual_start_page if 'actual_start_page' in locals() else (chunk_index * 20) + 1,
            'end_page': actual_end_page if 'actual_end_page' in locals() else (chunk_index + 1) * 20,
            'performance': {},
            'cost_info': {'estimated_cost': 0.0},
            'worker_info': {
                'worker_type': 'internal_parallel',
                'error_time': time.time()
            }
        }


def _should_chunk_document(file_path, file_name):
    """
    Determine if a document should be chunked. 
    PDFs with more than 200 pages should be chunked into 200-page chunks.
    (Chunking effectively disabled for most documents)
    
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
            import fitz  # PyMuPDF - lazy import
            pdf_doc = fitz.open(file_path)
            page_count = pdf_doc.page_count
            pdf_doc.close()
            
            # Chunk PDFs with more than 200 pages into 200-page chunks (effectively disabled for most docs)
            if page_count > 200:
                logger.info(f"PDF has {page_count} pages (>200) - will chunk into 200-page chunks")
                return True
            else:
                logger.info(f"PDF has {page_count} pages (≤200) - will process directly (chunking disabled)")
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
            start_page = result.get('start_page', (chunk_num * 200) + 1)
            end_page = result.get('end_page', (chunk_num + 1) * 200)
            
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
        
        if GENAI_AVAILABLE and text_processor is not None:
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
        else:
            logger.warning(f"Google Genai not available for summary generation for job {job_id}")
            summary_result = {
                "summary": f"Document processed with {successful_chunks} chunks. Summary generation not available - Google Genai not configured.",
                "date": "undated",
                "extracted_info": {
                    "key_dates": "Not available", 
                    "main_parties": "Not available",
                    "case_reference_numbers": "Not available",
                    "full_analysis": "Summary generation requires Google Genai"
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
                "processing_method": "chunked_parallel_enhanced"
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

def process_extracted_images_with_genai(job_id, extraction_metadata):
    """
    Process extracted table, figure, and chart images with Google Genai.
    This runs in the document_tasks.py container which has Python 3.10 and Google Genai support.
    
    ARCHITECTURE NOTE:
    - PPStructure tasks (Python 3.9): OCR, layout detection, image extraction
    - Document tasks (Python 3.10): Genai processing, coordination, final assembly
    
    Args:
        job_id: Job identifier
        extraction_metadata: Metadata about extracted images from PPStructure
        
    Returns:
        dict: Text extractions for tables, figures, and charts
    """
    if not GENAI_AVAILABLE or text_processor is None:
        logger.warning(f"Google Genai not available for job {job_id} - skipping image processing")
        return {
            'table_extractions': [],
            'figure_extractions': [],
            'chart_extractions': [],
            'total_processed': 0,
            'error': 'Google Genai not available'
        }
    
    try:
        logger.info(f"Starting Genai processing for job {job_id}")
        
        table_extractions = []
        figure_extractions = []
        chart_extractions = []
        
        # Process tables
        if extraction_metadata.get('tables'):
            logger.info(f"Processing {len(extraction_metadata['tables'])} tables with Genai")
            for table_info in extraction_metadata['tables']:
                try:
                    if os.path.exists(table_info['image_path']):
                        with open(table_info['image_path'], 'rb') as f:
                            img_bytes = f.read()
                        
                        table_text = text_processor.process_table_image(img_bytes, table_info['context'])
                        formatted_text = f"\n--- TABLE FROM PAGE {table_info['page']} ---\n{table_text}\n"
                        table_extractions.append(formatted_text)
                        logger.info(f"✅ Processed table from page {table_info['page']}")
                    else:
                        logger.warning(f"Table image not found: {table_info['image_path']}")
                except Exception as e:
                    logger.error(f"Error processing table from page {table_info.get('page', 'unknown')}: {str(e)}")
                    continue
        
        # Process figures
        if extraction_metadata.get('figures'):
            logger.info(f"Processing {len(extraction_metadata['figures'])} figures with Genai")
            for figure_info in extraction_metadata['figures']:
                try:
                    if os.path.exists(figure_info['image_path']):
                        with open(figure_info['image_path'], 'rb') as f:
                            img_bytes = f.read()
                        
                        figure_text = text_processor.process_figure_image(img_bytes, figure_info['context'])
                        formatted_text = f"\n--- FIGURE FROM PAGE {figure_info['page']} ---\n{figure_text}\n"
                        figure_extractions.append(formatted_text)
                        logger.info(f"✅ Processed figure from page {figure_info['page']}")
                    else:
                        logger.warning(f"Figure image not found: {figure_info['image_path']}")
                except Exception as e:
                    logger.error(f"Error processing figure from page {figure_info.get('page', 'unknown')}: {str(e)}")
                    continue
        
        # Process charts
        if extraction_metadata.get('charts'):
            logger.info(f"Processing {len(extraction_metadata['charts'])} charts with Genai")
            for chart_info in extraction_metadata['charts']:
                try:
                    if os.path.exists(chart_info['image_path']):
                        with open(chart_info['image_path'], 'rb') as f:
                            img_bytes = f.read()
                        
                        chart_text = text_processor.process_chart_image(img_bytes, chart_info['context'])
                        formatted_text = f"\n--- CHART FROM PAGE {chart_info['page']} ---\n{chart_text}\n"
                        chart_extractions.append(formatted_text)
                        logger.info(f"✅ Processed chart from page {chart_info['page']}")
                    else:
                        logger.warning(f"Chart image not found: {chart_info['image_path']}")
                except Exception as e:
                    logger.error(f"Error processing chart from page {chart_info.get('page', 'unknown')}: {str(e)}")
                    continue
        
        total_processed = len(table_extractions) + len(figure_extractions) + len(chart_extractions)
        logger.info(f"Genai processing completed for job {job_id}: {total_processed} items processed")
        
        return {
            'table_extractions': table_extractions,
            'figure_extractions': figure_extractions,
            'chart_extractions': chart_extractions,
            'total_processed': total_processed
        }
        
    except Exception as e:
        logger.error(f"Error in Genai processing for job {job_id}: {str(e)}")
        return {
            'table_extractions': [],
            'figure_extractions': [],
            'chart_extractions': [],
            'total_processed': 0,
            'error': str(e)
        } 

 