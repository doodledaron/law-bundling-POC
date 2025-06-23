"""
Celery configuration for the law document processing system.
Sets up Celery app, task routing, and general configuration with single worker internal parallelism.
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

# Simplified Celery settings for CUDA-safe processing
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Task execution settings - optimized for CUDA-safe processing
    worker_prefetch_multiplier=1,  # Process one task at a time for memory efficiency
    task_acks_late=True,  # Acknowledge tasks after execution
    
    # Result settings
    task_ignore_result=False,
    result_expires=60 * 60 * 24 * 7,  # 7 days
    
    # Task timeouts - reasonable for sequential processing
    task_time_limit=7200,  # Hard time limit: 2 hours for large documents
    task_soft_time_limit=6000,  # Soft time limit: 100 minutes with graceful handling
    
    # Simplified broker settings
    broker_transport_options={
        'visibility_timeout': 7200,  # Match task time limit
    },
    
    # Worker settings for CUDA-safe processing
    worker_concurrency=1,  # Single process for GPU stability and no duplicate logs
    
    # Worker memory management - for sequential chunk processing
    worker_max_tasks_per_child=3,  # Restart worker after 3 tasks to prevent memory leaks
    worker_max_memory_per_child=6144000,  # 6GB memory limit for single worker
    
    # Task pool settings for memory management
    worker_pool='threads',  # Use threads for better memory sharing
    worker_pool_restarts=True,  # Allow pool restarts
    
    # Logging settings
    worker_redirect_stdouts=False,  # Keep logs visible
    
    # Task result settings
    result_persistent=True,  # Persist results
    result_compression='gzip',  # Compress results to save memory
    
    # Disable advanced features that might cause issues
    task_always_eager=False,  # Never run tasks eagerly
)

# Simplified task routing - no priorities to avoid broker issues
celery_app.conf.task_routes = {
    # Document processing tasks
    'tasks.process_document': {'queue': 'documents'},
    
    # PPStructure tasks
    'tasks.process_document_with_ppstructure': {'queue': 'documents'},
    'tasks.warmup_ppstructure': {'queue': 'documents'},
    
    # Chunking tasks
    'tasks.create_document_chunks': {'queue': 'documents'},
    'tasks.update_chunk_status': {'queue': 'documents'},
    'tasks.chunking.*': {'queue': 'documents'},
    
    # Maintenance tasks
    'tasks.maintenance.*': {'queue': 'maintenance'},
    'tasks.cleanup_expired_results': {'queue': 'maintenance'},
    'tasks.system_stats': {'queue': 'maintenance'}
}

# Retry settings
celery_app.conf.task_default_retry_delay = 60  # 1 minute delay
celery_app.conf.task_max_retries = 2  # Reduced retries for faster failure detection

# Beat schedule for maintenance tasks
celery_app.conf.beat_schedule = {
    'cleanup-expired-results': {
        'task': 'tasks.maintenance.cleanup_expired_results',
        'schedule': 60 * 60 * 24,  # Daily
    },
    'collect-system-stats': {
        'task': 'tasks.maintenance.system_stats',
        'schedule': 60 * 60,  # Hourly
    }
}

logger.info("Simplified Celery configuration loaded for single worker internal parallelism")