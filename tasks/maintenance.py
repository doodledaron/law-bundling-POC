"""
Maintenance tasks for the law document processing system.
Handles cleanup operations and system maintenance.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import redis
import time
from datetime import datetime, timedelta

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

@shared_task(name='tasks.maintenance.cleanup_expired_results')
def cleanup_expired_results():
    """
    Clean up expired result files and uploads.
    Removes files older than the specified expiration period.
    """
    try:
        # Expiration period (7 days)
        expiration_period = timedelta(days=7)
        now = datetime.now()
        
        # Count of files deleted
        results_deleted = 0
        uploads_deleted = 0
        
        # Clean up result files
        results_dir = 'results'
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                
                # Check file age
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_modified > expiration_period:
                    try:
                        # Extract job ID from filename (format: {job_id}_results.json)
                        job_id = filename.split('_')[0]
                        
                        # Check if job still exists in Redis
                        if not redis_client.exists(f"job:{job_id}"):
                            os.remove(file_path)
                            results_deleted += 1
                    except Exception as e:
                        logger.error(f"Error deleting result file {filename}: {str(e)}")
        
        # Clean up upload files
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                
                # Check file age
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_modified > expiration_period:
                    try:
                        # Extract job ID from filename (format: {job_id}.ext)
                        job_id = os.path.splitext(filename)[0]
                        
                        # Check if job still exists in Redis
                        if not redis_client.exists(f"job:{job_id}"):
                            os.remove(file_path)
                            uploads_deleted += 1
                    except Exception as e:
                        logger.error(f"Error deleting upload file {filename}: {str(e)}")
        
        logger.info(f"Cleanup complete: {results_deleted} result files and {uploads_deleted} upload files deleted")
        
        return {
            'results_deleted': results_deleted,
            'uploads_deleted': uploads_deleted,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error cleaning up expired results: {str(e)}")
        raise

@shared_task(name='tasks.maintenance.system_stats')
def system_stats():
    """
    Collect system statistics for monitoring.
    """
    try:
        # Get Redis stats
        redis_info = redis_client.info()
        redis_keys = redis_client.dbsize()
        
        # Get file system stats
        uploads_count = len(os.listdir('uploads')) if os.path.exists('uploads') else 0
        results_count = len(os.listdir('results')) if os.path.exists('results') else 0
        
        # Get job statistics
        pending_jobs = 0
        processing_jobs = 0
        completed_jobs = 0
        failed_jobs = 0
        
        # Iterate through job keys
        for key in redis_client.scan_iter("job:*"):
            try:
                job_data = redis_client.get(key)
                if job_data:
                    job = json.loads(job_data)
                    status = job.get('status', '').lower()
                    
                    if status == 'pending':
                        pending_jobs += 1
                    elif status == 'processing':
                        processing_jobs += 1
                    elif status == 'completed':
                        completed_jobs += 1
                    elif status == 'failed':
                        failed_jobs += 1
            except Exception as e:
                logger.error(f"Error processing Redis key {key}: {str(e)}")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'redis_used_memory': redis_info.get('used_memory_human', 'N/A'),
            'redis_keys': redis_keys,
            'files': {
                'uploads': uploads_count,
                'results': results_count
            },
            'jobs': {
                'pending': pending_jobs,
                'processing': processing_jobs,
                'completed': completed_jobs,
                'failed': failed_jobs,
                'total': pending_jobs + processing_jobs + completed_jobs + failed_jobs
            }
        }
        
        # Store stats in Redis with 24-hour expiration
        redis_client.setex(
            f"system:stats:{int(time.time())}",
            60 * 60 * 24,  # 24 hour expiration
            json.dumps(stats)
        )
        
        return stats
    
    except Exception as e:
        logger.error(f"Error collecting system stats: {str(e)}")
        raise