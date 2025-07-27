"""
Utility functions for tasks and API endpoints.
Contains commonly used functions like status updates and timestamp generation.
"""
import json
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

def get_timestamp():
    """
    Get current ISO format timestamp.
    
    Returns:
        str: Current timestamp in ISO format
    """
    return datetime.utcnow().isoformat()

def get_unix_timestamp():
    """
    Get current high-precision timestamp for performance tracking.
    
    Returns:
        float: Current high-precision timestamp
    """
    return time.perf_counter()

def calculate_duration(start_time, end_time):
    """
    Calculate duration between two timestamps.
    
    Args:
        start_time: Start timestamp (Unix time)
        end_time: End timestamp (Unix time)
        
    Returns:
        dict: Duration in various formats
    """
    duration_seconds = end_time - start_time
    
    return {
        'seconds': round(duration_seconds, 2),
        'formatted': format_duration(duration_seconds)
    }

def format_duration(seconds):
    """
    Format duration in human-readable format with better precision.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 0.001:  # Less than 1ms
        microseconds = int(seconds * 1000000)
        return f"{microseconds}μs"
    elif seconds < 1:  # Less than 1 second
        milliseconds = round(seconds * 1000, 1)
        return f"{milliseconds}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"

def update_job_timing(redis_client, job_id, stage, start_time=None, end_time=None):
    """
    Update job timing information.
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier
        stage: Processing stage name
        start_time: Start timestamp (Unix time)
        end_time: End timestamp (Unix time)
    """
    try:
        # Get current job data
        job_data = redis_client.get(f"job:{job_id}")
        if job_data:
            job_status = json.loads(job_data)
        else:
            job_status = {}
        
        # Initialize timing if not exists
        if 'timing' not in job_status:
            job_status['timing'] = {}
        
        # Update timing for the stage
        if start_time is not None:
            job_status['timing'][f'{stage}_start'] = start_time
        
        if end_time is not None:
            job_status['timing'][f'{stage}_end'] = end_time
            # Calculate duration if start time exists
            start_key = f'{stage}_start'
            if start_key in job_status['timing']:
                duration = calculate_duration(
                    job_status['timing'][start_key], 
                    end_time
                )
                job_status['timing'][f'{stage}_duration'] = duration
        
        # Store updated status
        redis_client.set(f"job:{job_id}", json.dumps(job_status))
        
    except Exception as e:
        logger.error(f"Error updating job timing: {str(e)}")

def update_chunk_progress(redis_client, job_id, total_chunks):
    """
    Update progress when a chunk completes processing.
    
    Progress calculation:
    - Chunk processing: 5% → 85% (80% total)
    - Each chunk completion: 80% / total_chunks
    - Merge task: 85% → 100% (handled separately)
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier
        total_chunks: Total number of chunks for this job
        
    Returns:
        int: Updated progress percentage
    """
    try:
        # Atomic increment of completed chunks counter
        completed_chunks = redis_client.incr(f"job:{job_id}:chunks_completed")
        
        # Set expiration for the chunks counter (7 days, same as job data)
        redis_client.expire(f"job:{job_id}:chunks_completed", 60 * 60 * 24 * 7)
        
        # Calculate progress: 5% base + (80% * completion_ratio)
        base_progress = 5
        chunk_progress_range = 80  # 80% allocated to chunk processing
        completion_ratio = min(completed_chunks / total_chunks, 1.0)  # Cap at 1.0
        
        current_progress = base_progress + int(chunk_progress_range * completion_ratio)
        
        # Cap progress at 85% (merge task handles 85% → 100%)
        current_progress = min(current_progress, 85)
        
        # Update job status with new progress
        update_job_status(redis_client, job_id, {
            'progress': current_progress,
            'chunks_completed': completed_chunks,
            'chunks_total': total_chunks,
            'message': f'Processing chunks... ({completed_chunks}/{total_chunks} chunks completed)',
            'updated_at': get_timestamp()
        })
        
        logger.info(f"📊 Job {job_id}: Chunk completed ({completed_chunks}/{total_chunks}), progress: {current_progress}%")
        
        return current_progress
        
    except Exception as e:
        logger.error(f"Error updating chunk progress for job {job_id}: {str(e)}")
        return 5  # Return base progress on error

def update_merge_progress(redis_client, job_id, stage, progress_percent):
    """
    Update progress during merge task phases.
    
    Merge task progress stages:
    - 85%: Merge started
    - 90%: Text combination complete
    - 95%: Summary generation complete  
    - 100%: Final results saved
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier
        stage: Merge stage description
        progress_percent: Progress percentage (85-100)
    """
    try:
        # Ensure progress is within merge range (85-100%)
        progress_percent = max(85, min(100, progress_percent))
        
        update_job_status(redis_client, job_id, {
            'progress': progress_percent,
            'message': f'Finalizing results... {stage}',
            'merge_stage': stage,
            'updated_at': get_timestamp()
        })
        
        logger.info(f"📊 Job {job_id}: Merge progress {progress_percent}% - {stage}")
        
    except Exception as e:
        logger.error(f"Error updating merge progress for job {job_id}: {str(e)}")

def update_job_status(redis_client, job_id, status_update):
    """
    Update job status in Redis.
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier
        status_update: Dict with status updates
    """
    try:
        # Get current status if it exists
        current_status = redis_client.get(f"job:{job_id}")
        
        if current_status:
            current_status = json.loads(current_status)
            # Update with new values
            current_status.update(status_update)
        else:
            current_status = status_update
        
        # Store updated status
        redis_client.set(f"job:{job_id}", json.dumps(current_status))
        
        # Set expiration (7 days)
        redis_client.expire(f"job:{job_id}", 60 * 60 * 24 * 7)
        
    except Exception as e:
        logger.error(f"Error updating job status in Redis: {str(e)}")