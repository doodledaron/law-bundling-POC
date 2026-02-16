"""
Production API routes for the law document processing system.
Protected by API key authentication.
"""
import os
import json
import uuid
import logging

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse

from services.auth import verify_api_key
from services.processing import (
    redis_client, submit_document_for_processing, _fix_file_path
)
from tasks.utils import get_timestamp, update_job_status

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api", tags=["api"])


@api_router.post("/upload", response_class=JSONResponse)
async def api_upload_single(file: UploadFile = File(...), authenticated: bool = Depends(verify_api_key)):
    """
    Single document upload with JSON response for API integration.
    
    This endpoint provides the same functionality as bulk-upload but for a single document.
    Perfect for system-to-system integration where you need to process one document at a time.
    
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
            'message': 'Document uploaded, waiting for processing',
            'progress': 0
        })
        
        # Submit for processing using the same function as bulk upload
        task_result = submit_document_for_processing(job_id, upload_path, file.filename)
        
        # Update job status with processing info
        update_job_status(redis_client, job_id, {
            'task_id': task_result.id,
            'processing_mode': 'celery_workers',
            'updated_at': get_timestamp()
        })
        
        print(f"Successfully submitted API job {job_id} for file {file.filename}")
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "submitted",
            "message": "Document submitted for processing",
            "polling_endpoint": f"/api/job/{job_id}",
            "estimated_completion_minutes": 2,
            "polling_interval_seconds": 15
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in API upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@api_router.get("/job/{job_id}")
async def api_job_status(job_id: str, authenticated: bool = Depends(verify_api_key)):
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
            results_data = None
            
            if os.path.exists(clean_results_path):
                try:
                    # Verify the results file has valid content and required fields
                    with open(clean_results_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)
                    
                    # Check for required fields that indicate processing is truly complete
                    required_fields = ['combined_text', 'total_pages', 'processing_completed_at']
                    has_required_fields = all(field in results_data for field in required_fields)
                    has_substantial_content = len(results_data.get('combined_text', '')) > 10
                    
                    # Simple validation: just check that required fields exist and have content
                    # Don't validate field values - "Not available" is legitimate for documents without dates/parties
                    if has_required_fields and has_substantial_content:
                        results_ready = True
                        logger.info(f"✅ Job {job_id} - Results validated and ready for display")
                    else:
                        logger.info(f"⚠️ Job {job_id} - Results file exists but incomplete: fields={has_required_fields}, content={has_substantial_content}")
                        
                    if results_ready:
                        # PRODUCTION ENHANCEMENT: Include actual results content in API response
                        # This makes the API much more production-friendly for system integration
                        job_status.update({
                            'results': {
                                # PRIMARY RESULTS - What other systems actually need
                                'date': results_data.get('date', 'undated'),
                                'summary': results_data.get('summary', ''),
                                'extracted_info': results_data.get('extracted_info', {}),
                                
                                # BASIC METADATA - Essential processing details only
                                'total_pages': results_data.get('total_pages', 0),
                                'processing_completed_at': results_data.get('processing_completed_at', '')
                            }
                        })
                        
                        # Also try to load additional metrics if available
                        metrics_path = os.path.join(results_dir, "metrics.json")
                        if os.path.exists(metrics_path):
                            try:
                                with open(metrics_path, 'r', encoding='utf-8') as f:
                                    metrics = json.load(f)
                                job_status['results']['metrics'] = metrics
                            except Exception as e:
                                logger.warning(f"⚠️ Job {job_id} - Could not load metrics: {str(e)}")
                        
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
        
        # For production API, return only essential client data, not internal dev info
        if job_status.get('status') == 'COMPLETED' and 'results' in job_status:
            # Return clean production response with only essential data
            clean_response = {
                'status': job_status['status'],
                'progress': job_status.get('progress', 100),
                'message': job_status.get('message', 'Processing completed successfully'),
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
                'stage': job_status.get('stage', '')  # Keep stage for better UX during processing
            }
            
            # Add error info if failed
            if job_status.get('status') == 'FAILED' and job_status.get('error'):
                clean_response['error'] = job_status.get('error')
                
            return clean_response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")
