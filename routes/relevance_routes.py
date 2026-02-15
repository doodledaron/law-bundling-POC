"""
Relevance extraction API routes.
Mirrors api_routes.py but for relevance extraction endpoints.
"""
import os
import json
import uuid
import logging

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from services.auth import verify_api_key
from services.relevance_processing import (
    redis_client, submit_relevance_extraction
)
from tasks.utils import get_timestamp, update_job_status

logger = logging.getLogger(__name__)

relevance_router = APIRouter(prefix="/api/relevance", tags=["relevance"])


@relevance_router.post("", response_class=JSONResponse)
async def api_relevance_upload(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Upload document for relevance extraction.
    
    This endpoint processes documents to extract token-efficient legal relevance summaries.
    
    Args:
        file: PDF, JPEG, or PNG document
        
    Returns:
        JSON response with job_id for polling the processing status
    """
    try:
        # Validate file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, JPEG, and PNG are supported.")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job_created_time = get_timestamp()
        
        # Save file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if not file_ext:
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
            'message': 'Document uploaded, waiting for relevance extraction',
            'progress': 0
        })
        
        # Submit for relevance extraction
        task_result = submit_relevance_extraction(job_id, upload_path, file.filename)
        
        # Update job status with processing info
        update_job_status(redis_client, job_id, {
            'task_id': task_result.id,
            'processing_mode': 'relevance_extraction',
            'updated_at': get_timestamp()
        })
        
        logger.info(f"✅ [RELEVANCE] Successfully submitted job {job_id} for file {file.filename}")
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "submitted",
            "message": "Document submitted for relevance extraction",
            "polling_endpoint": f"/api/relevance/{job_id}",
            "estimated_completion_minutes": 2,
            "polling_interval_seconds": 15
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in relevance API upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@relevance_router.get("/{job_id}")
async def api_relevance_status(job_id: str, authenticated: bool = Depends(verify_api_key)):
    """
    Get relevance extraction job status and results.
    
    Returns processing status and relevance results when complete.
    """
    try:
        # Get job status from Redis
        job_data = redis_client.get(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_status = json.loads(job_data)
        
        # CRITICAL: Validate COMPLETED status by checking if relevance results files actually exist AND are complete
        if job_status.get('status') == 'COMPLETED':
            results_dir = os.path.join("results", job_id)
            relevance_results_path = os.path.join(results_dir, "relevance_results.json")
            
            # Check if results files exist AND are complete with valid content
            results_ready = False
            results_data = None
            
            if os.path.exists(relevance_results_path):
                try:
                    # Verify the results file has valid content and required fields
                    with open(relevance_results_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)
                    
                    # Check for required fields that indicate processing is truly complete
                    required_fields = ['relevances', 'document_type', 'total_pages', 'processing_completed_at']
                    has_required_fields = all(field in results_data for field in required_fields)
                    
                    # Valid relevance should have actual content
                    relevances = results_data.get('relevances', [])
                    has_substantial_content = len(relevances) > 0
                    has_valid_relevance = False
                    if relevances:
                        for rel in relevances:
                            relevance_text = rel.get('relevance_text', '')
                            if relevance_text and len(relevance_text) > 10 and 'Error during relevance generation' not in relevance_text:
                                has_valid_relevance = True
                                break
                    
                    if has_required_fields and has_substantial_content and has_valid_relevance:
                        results_ready = True
                        logger.info(f"✅ [RELEVANCE] Job {job_id} - Results validated and ready for display")
                    else:
                        logger.warning(f"⚠️ [RELEVANCE] Job {job_id} - Results file exists but incomplete or invalid")
                        
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"⚠️ [RELEVANCE] Job {job_id} - Results file corrupted or incomplete: {str(e)}")
            
            # If results files don't exist or are incomplete, keep status as PROCESSING
            if not results_ready:
                logger.info(f"🔄 [RELEVANCE] Job {job_id} marked COMPLETED but results not ready - keeping as PROCESSING")
                job_status['status'] = 'PROCESSING'
                job_status['message'] = 'Relevance extraction complete. Finalizing results...'
                job_status['progress'] = 95
            else:
                # PRODUCTION: Include actual relevance results in API response
                job_status.update({
                    'results': {
                        # PRIMARY RESULTS
                        'document_type': results_data.get('document_type', 'unknown'),
                        'relevances': results_data.get('relevances', []),  # Array of {relevance_text, pinpoints, evidence_quotes}
                        
                        # METADATA
                        'total_pages': results_data.get('total_pages', 0),
                        'pages_processed': results_data.get('pages_processed', {}),
                        'processing_completed_at': results_data.get('processing_completed_at', ''),
                        'token_usage': results_data.get('token_usage', {}),
                        'estimated_cost': results_data.get('estimated_cost', 0.0)
                    }
                })
                
                # Also try to load additional metrics if available
                metrics_path = os.path.join(results_dir, "relevance_metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        job_status['results']['metrics'] = metrics
                    except Exception as e:
                        logger.warning(f"⚠️ [RELEVANCE] Job {job_id} - Could not load metrics: {str(e)}")
        
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
                    job_status['stage'] = f'All {num_chunks} chunks processed. Generating relevance extraction...'
                elif progress < 95:
                    job_status['stage'] = 'Relevance extraction generated. Saving final results...'
                else:
                    job_status['stage'] = 'Finalizing and preparing results display...'
            else:
                # Single document processing stages
                if progress < 30:
                    job_status['stage'] = 'Analyzing document structure'
                elif progress < 60:
                    job_status['stage'] = 'Extracting text from document'
                elif progress < 85:
                    job_status['stage'] = 'Identifying relevant sections'
                else:
                    job_status['stage'] = 'Generating relevance extraction and finalizing results'
        
        # For production API, return only essential client data
        if job_status.get('status') == 'COMPLETED' and 'results' in job_status:
            # Return clean production response with only essential data
            clean_response = {
                'status': job_status['status'],
                'progress': job_status.get('progress', 100),
                'message': job_status.get('message', 'Relevance extraction completed successfully'),
                'filename': job_status.get('filename', job_status.get('original_filename', 'Unknown')),
                'created_at': job_status.get('created_at', ''),
                'processing_completed_at': job_status.get('processing_completed_at', ''),
                'results': job_status['results']  # This already contains clean data
            }
            return clean_response
        else:
            # For processing/failed jobs, return minimal status info
            clean_response = {
                'status': job_status.get('status', 'UNKNOWN'),
                'progress': job_status.get('progress', 0),
                'message': job_status.get('message', ''),
                'filename': job_status.get('filename', job_status.get('original_filename', 'Unknown')),
                'created_at': job_status.get('created_at', ''),
                'stage': job_status.get('stage', '')
            }
            
            # Add error info if failed
            if job_status.get('status') == 'FAILED' and job_status.get('error'):
                clean_response['error'] = job_status.get('error')
                
            return clean_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving relevance job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")
