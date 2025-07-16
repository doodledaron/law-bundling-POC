"""
Celery configuration for the law document processing system.
Streamlined configuration for direct PPStructure processing with built-in chunking.
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
    
    # Worker memory management - for heavy model processing (Option A: optimized for 3 concurrency)
    worker_max_tasks_per_child=6,  # Restart worker after 6 tasks to prevent memory leaks (optimized for fewer restarts)
    worker_max_memory_per_child=6144000,  # 6GB memory limit per worker (3 concurrent tasks per container)
    
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

# Task routing for high-throughput 6-container setup
celery_app.conf.task_routes = {
    # PPStructure tasks: All routed to shared chunk_queue for optimal load balancing
    # 6 containers with 3 concurrency each = 18 parallel workers available
    'tasks.process_document_with_ppstructure': {'queue': 'chunk_queue'},
    'tasks.merge_and_summarize_chunks': {'queue': 'chunk_queue'},
    'tasks.warmup_ppstructure': {'queue': 'chunk_queue'},
    
    # Maintenance tasks
    'tasks.maintenance.*': {'queue': 'maintenance'},
    'tasks.cleanup_expired_results': {'queue': 'maintenance'},
    'tasks.system_stats': {'queue': 'maintenance'}
}

# Retry settings optimized for high-throughput processing
celery_app.conf.task_default_retry_delay = 30  # Reduced delay for faster processing
celery_app.conf.task_max_retries = 2  # Keep retries low for faster failure detection

# High-throughput optimization settings
celery_app.conf.update(
    # Task execution settings
    task_acks_late=True,  # Acknowledge after completion for reliability
    worker_prefetch_multiplier=1,  # Prefetch 1 task per worker (3 concurrency * 1 = 3 tasks queued per container)
    
    # Result settings optimized for chunk processing
    result_expires=3600,  # 1 hour expiry for results
    result_persistent=True,  # Persist results for merge operations
    result_compression='gzip',  # Compress results to save Redis memory
    
    # Connection pool settings for 6 containers
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_pool_limit=25,  # Pool for 6 containers with 3 concurrency each
    
    # Performance optimizations
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

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

logger.info("Memory-optimized Celery configuration loaded with 6-container parallelism")
logger.info("Architecture: 6 containers * 3 concurrency = 18 parallel workers")
logger.info("Queue strategy: All tasks → chunk_queue for optimal load balancing")
logger.info("Chunking: Fixed 5-page chunks for predictable parallelism")
logger.info("Memory allocation: 8GB per container (48GB total)")
logger.info("Expected throughput: 3x-4x improvement with optimal CPU utilization")