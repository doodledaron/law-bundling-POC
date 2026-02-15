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

import shutil

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

@shared_task(name='tasks.maintenance.cleanup_expired_results')
def cleanup_expired_results():
    """
    Clean up expired result files, uploads, and chunks.
    Removes files/directories older than the specified expiration period.
    """
    try:
        # Expiration period (7 days for results/uploads, 1 day for chunks)
        expiration_period = timedelta(days=7)
        chunk_expiration = timedelta(days=1)
        now = datetime.now()
        
        # Counters
        results_deleted = 0
        uploads_deleted = 0
        chunks_deleted = 0
        
        # --- Clean up result directories ---
        # Results are stored as directories: results/{job_id}/
        results_dir = 'results'
        if os.path.exists(results_dir):
            for entry in os.listdir(results_dir):
                entry_path = os.path.join(results_dir, entry)
                
                try:
                    # Check age (use modification time)
                    entry_modified = datetime.fromtimestamp(os.path.getmtime(entry_path))
                    if now - entry_modified > expiration_period:
                        # The directory name IS the job_id
                        job_id = entry
                        
                        # Only delete if Redis key has expired (job no longer active)
                        if not redis_client.exists(f"job:{job_id}"):
                            if os.path.isdir(entry_path):
                                shutil.rmtree(entry_path, ignore_errors=True)
                            else:
                                os.remove(entry_path)
                            results_deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting result entry {entry}: {str(e)}")
        
        # --- Clean up upload files ---
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                
                try:
                    file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if now - file_modified > expiration_period:
                        job_id = os.path.splitext(filename)[0]
                        
                        if not redis_client.exists(f"job:{job_id}"):
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path, ignore_errors=True)
                            else:
                                os.remove(file_path)
                            uploads_deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting upload file {filename}: {str(e)}")
        
        # --- Clean up chunk files (temporary, short expiry) ---
        chunks_dir = 'chunks'
        if os.path.exists(chunks_dir):
            for entry in os.listdir(chunks_dir):
                entry_path = os.path.join(chunks_dir, entry)
                
                try:
                    entry_modified = datetime.fromtimestamp(os.path.getmtime(entry_path))
                    if now - entry_modified > chunk_expiration:
                        if os.path.isdir(entry_path):
                            shutil.rmtree(entry_path, ignore_errors=True)
                        else:
                            os.remove(entry_path)
                        chunks_deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting chunk entry {entry}: {str(e)}")
        
        logger.info(
            f"Cleanup complete: {results_deleted} results, "
            f"{uploads_deleted} uploads, {chunks_deleted} chunks deleted"
        )
        
        return {
            'results_deleted': results_deleted,
            'uploads_deleted': uploads_deleted,
            'chunks_deleted': chunks_deleted,
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
                # Skip auxiliary keys (chunks_completed, etc.) - only process main job status keys
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if ':' in key_str and key_str.count(':') > 1:
                    # Skip keys like job:xxx:chunks_completed, job:xxx:total_chunks, etc.
                    continue
                
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