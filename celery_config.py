"""
Celery configuration for the law document processing system.
Sets up Celery app, task routing, and general configuration.
"""
from celery import Celery
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    'law_doc_processing',
    broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    include=[
        'tasks.document_tasks',
        'tasks.chunking_tasks',
        'tasks.maintenance',
        'tasks.ppstructure_tasks'

    ]
)

# Configure Celery settings
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Task execution settings
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_acks_late=True,  # Acknowledge tasks after execution
    
    # Result settings
    task_ignore_result=False,
    result_expires=60 * 60 * 24 * 7,  # 7 days
    
    # Task timeouts - OCR can take time for large docs
    task_time_limit=3600,  # Hard time limit - When reached, the task is forcefully terminated.
    task_soft_time_limit=3000,  # Soft time limit - When reached, a SoftTimeLimitExceeded exception is raised, allowing graceful handling.
    
    # Broker settings
    broker_transport_options={
        'visibility_timeout': 3600,
        'fanout_prefix': True,
        'fanout_patterns': True,
    },
    
    # Worker concurrency - REDUCED for memory-intensive OCR tasks
    # Set low concurrency for document workers to prevent memory exhaustion
    worker_concurrency=2,  # Max 2 concurrent workers per container (was os.cpu_count())
    
    # Worker memory management - CRITICAL for preventing SIGKILL
    worker_max_tasks_per_child=5,  # Restart worker after 5 tasks to prevent memory leaks
    worker_max_memory_per_child=2048000,  # Restart worker if memory exceeds 2GB (in KB)
    
    # Task pool settings for memory management
    worker_pool='threads',  # Use threads instead of processes for better memory sharing
    worker_pool_restarts=True,  # Allow pool restarts
    
    # Logging
    worker_redirect_stdouts=False # When False , worker's stdout/stderr are not redirected to the logging system (see print statements directly),In production, you might want to set this to True to capture all output in logs 
)

# Task routing - assign tasks to different queues
celery_app.conf.task_routes = {
    # Document processing tasks
    'tasks.process_document': {'queue': 'documents'},
    'tasks.process_large_document': {'queue': 'documents'},
    'tasks.process_small_document': {'queue': 'documents'},
    'tasks.process_document_chunk': {'queue': 'documents'},
    'tasks.finalize_document_processing': {'queue': 'documents'},
    
    # OCR tasks
    'tasks.ocr_tasks.*': {'queue': 'ocr'},
    
    # Chunking tasks
    'tasks.chunking.*': {'queue': 'documents'},
    
    # Extraction tasks
    'tasks.extraction.*': {'queue': 'extraction'},
    
    # Maintenance tasks
    'tasks.maintenance.*': {'queue': 'maintenance'}
}

# Retry settings
celery_app.conf.task_default_retry_delay = 60  # 1 minute delay
celery_app.conf.task_max_retries = 3  # Retry up to 3 times

# Optional scheduled tasks with Celery Beat
celery_app.conf.beat_schedule = {
    # Daily cleanup of expired results
    'cleanup-expired-results': {
        'task': 'tasks.maintenance.cleanup_expired_results',
        'schedule': 60 * 60 * 24,  # Daily
    },
    # Hourly system stats collection
    'collect-system-stats': {
        'task': 'tasks.maintenance.system_stats',
        'schedule': 60 * 60,  # Hourly
    }
}

logger.info("Celery configured successfully")