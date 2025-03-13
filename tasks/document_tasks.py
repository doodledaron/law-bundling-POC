"""
Main document processing tasks for the legal document processing system.
Handles document processing coordination and manages the processing flow.
"""
from celery import shared_task, chain, group
from celery.utils.log import get_task_logger
import os
import json
import redis

# Import task modules
from tasks.ocr_tasks import process_pdf, process_image
from tasks.chunking_tasks import create_document_chunks, update_chunk_status
from tasks.extraction_tasks import extract_document_info, merge_extraction_results
from tasks.utils import (
    get_timestamp, 
    update_job_status, 
    extract_nda_fields, 
    generate_summary
)

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

@shared_task(name='tasks.process_document')
def process_document(job_id, file_path, file_name):
    """
    Process a document file and extract information.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    try:
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'filename': file_name,
            'message': 'Document processing started',
            'updated_at': get_timestamp()
        })
        
        # Get file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Check file size to determine if chunking is needed
        file_size = os.path.getsize(file_path)
        large_file_threshold = 5 * 1024 * 1024  # 5MB threshold (can be changed later)
        
        # For PDFs, we'll use chunking regardless of size since they can be multi-page
        if file_ext == '.pdf' or file_size > large_file_threshold:
            # Use chunking approach for large files
            return process_large_document(job_id, file_path, file_name)
        else:
            # Use direct processing for small files
            return process_small_document(job_id, file_path, file_name)
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Error processing document: {str(e)}',
            'updated_at': get_timestamp()
        })
        
        # Re-raise to allow Celery to handle the error
        raise


@shared_task(name='tasks.process_large_document')
def process_large_document(job_id, file_path, file_name):
    """
    Process a large document using the chunking approach.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    try:
        # Create chunks
        chunk_info = create_document_chunks.delay(job_id, file_path, file_name).get()
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': f'Processing {chunk_info["num_chunks"]} document chunks',
            'progress': 25,
            'updated_at': get_timestamp()
        })
        
        # Process each chunk in parallel with a group
        chunk_tasks = []
        for chunk in chunk_info['chunks']:
            # Create a task for each chunk
            chunk_id = chunk['chunk_id']
            chunk_path = chunk['path']
            
            # Create a task that processes this chunk
            task = process_document_chunk.s(job_id, chunk_id, chunk_path)
            chunk_tasks.append(task)
        
        # Create a group of tasks for parallel processing
        chunks_group = group(chunk_tasks)
        
        # Chain the group with final merging
        workflow = chain(
            chunks_group,
            finalize_document_processing.s(job_id, file_name)
        )
        
        # Execute the workflow
        result = workflow.apply_async()
        
        # Return the task ID for status tracking
        return {
            'job_id': job_id,
            'status': 'PROCESSING',
            'message': 'Large document processing started',
            'task_id': result.id
        }
        
    except Exception as e:
        logger.error(f"Error processing large document: {str(e)}")
        
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Error processing large document: {str(e)}',
            'updated_at': get_timestamp()
        })
        
        # Re-raise to allow Celery to handle the error
        raise


@shared_task(name='tasks.process_document_chunk')
def process_document_chunk(job_id, chunk_id, chunk_path):
    """
    Process a single document chunk.
    
    Args:
        job_id: Unique job identifier
        chunk_id: Chunk identifier
        chunk_path: Path to the chunk file
        
    Returns:
        dict: Chunk processing results
    """
    try:
        # Update chunk status
        update_chunk_status(job_id, chunk_id, 'processing')
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'message': f'Processing chunk {chunk_id}',
            'updated_at': get_timestamp()
        })
        
        # Get file extension
        file_ext = os.path.splitext(chunk_path)[1].lower()
        
        # Process chunk based on file type
        if file_ext == '.pdf':
            results = process_pdf(job_id, chunk_path)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            results = process_image(job_id, chunk_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Extract fields from the text
        cleaned_text = results.get('cleaned_text', '')
        entities = extract_nda_fields(cleaned_text)
        
        # Generate summary for this chunk
        summary = generate_summary(cleaned_text)
        
        # Combine results
        chunk_results = {
            'chunk_id': chunk_id,
            'job_id': job_id,
            'cleaned_text': cleaned_text,
            'average_confidence': results.get('average_confidence', 0),
            'entities': entities,
            'summary': summary
        }
        
        # Update chunk status with results
        update_chunk_status(job_id, chunk_id, 'completed', results=chunk_results)
        
        # Return results for the merge step
        return chunk_results
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
        
        # Update chunk status on failure
        update_chunk_status(job_id, chunk_id, 'failed', results={'error': str(e)})
        
        # Re-raise to allow Celery to handle the error
        raise


@shared_task(name='tasks.finalize_document_processing')
def finalize_document_processing(chunk_results, job_id, file_name):
    """
    Merge chunk results and finalize document processing.
    
    Args:
        chunk_results: List of results from chunk processing
        job_id: Unique job identifier
        file_name: Original file name
        
    Returns:
        dict: Final processing results
    """
    try:
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': 'Merging chunk results and finalizing processing',
            'progress': 90,
            'updated_at': get_timestamp()
        })
        
        # Merge chunk results
        merged_results = merge_extraction_results(job_id, chunk_results)
        
        # Add filename to results
        merged_results['filename'] = file_name
        
        # Format confidence score for display if not already formatted
        if 'average_confidence_formatted' not in merged_results:
            average_confidence = merged_results.get('average_confidence', 0)
            merged_results['average_confidence_formatted'] = f"{average_confidence:.2%}"
        
        # Save results to file
        results_path = os.path.join('results', f"{job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(merged_results, f)
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'results_path': results_path,
            'message': 'Processing completed successfully',
            'progress': 100,
            'updated_at': get_timestamp()
        })
        
        # Clean up temporary chunks
        try:
            import shutil
            chunks_dir = os.path.join('chunks', job_id)
            if os.path.exists(chunks_dir):
                shutil.rmtree(chunks_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up chunks directory: {str(e)}")
        
        return {
            'job_id': job_id,
            'status': 'COMPLETED',
            'results_path': results_path
        }
        
    except Exception as e:
        logger.error(f"Error finalizing document processing: {str(e)}")
        
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Error finalizing document processing: {str(e)}',
            'updated_at': get_timestamp()
        })
        
        # Re-raise to allow Celery to handle the error
        raise


@shared_task(name='tasks.process_small_document')
def process_small_document(job_id, file_path, file_name):
    """
    Process a small document directly without chunking.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        
    Returns:
        dict: Processing results
    """
    try:
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': 'Processing document directly',
            'progress': 25,
            'updated_at': get_timestamp()
        })
        
        # Process based on file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            results = process_pdf(job_id, file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            results = process_image(job_id, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Update progress
        update_job_status(redis_client, job_id, {
            'message': 'OCR processing complete, extracting information',
            'progress': 60,
            'updated_at': get_timestamp()
        })
        
        # Extract fields from processed text
        cleaned_text = results.get('cleaned_text', '')
        fields = extract_nda_fields(cleaned_text)
        
        # Generate summary
        summary = generate_summary(cleaned_text)
        
        # Update progress
        update_job_status(redis_client, job_id, {
            'message': 'Information extraction complete, finalizing results',
            'progress': 90,
            'updated_at': get_timestamp()
        })
        
        # Format confidence score for display
        average_confidence = results.get('average_confidence', 0)
        average_confidence_formatted = f"{average_confidence:.2%}"
        
        # Combine results
        final_results = {
            'job_id': job_id,
            'filename': file_name,
            'extracted_text': cleaned_text,
            'average_confidence': average_confidence,
            'average_confidence_formatted': average_confidence_formatted,
            'entities': fields,
            'summary': summary
        }
        
        # Save results to file
        results_path = os.path.join('results', f"{job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f)
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'results_path': results_path,
            'message': 'Processing completed successfully',
            'progress': 100,
            'updated_at': get_timestamp()
        })
        
        # Optional: Clean up the original uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up original file {file_path}: {str(e)}")
        
        return {
            'job_id': job_id,
            'status': 'COMPLETED',
            'results_path': results_path
        }
        
    except Exception as e:
        logger.error(f"Error processing small document: {str(e)}")
        
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Error processing document: {str(e)}',
            'updated_at': get_timestamp()
        })
        
        # Re-raise to allow Celery to handle the error
        raise