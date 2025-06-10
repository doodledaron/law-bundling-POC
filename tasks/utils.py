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
    Get current Unix timestamp for performance tracking.
    
    Returns:
        float: Current Unix timestamp
    """
    return time.time()

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
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
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