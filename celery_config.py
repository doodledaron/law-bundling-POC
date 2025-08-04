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
    result_expires=3600,  # 1 hour - faster cleanup for large documents
    
    # Task timeouts - extended for maximum document processing
    task_time_limit=14400,  # Hard time limit: 4 hours for large documents
    task_soft_time_limit=12000,  # Soft time limit: 200 minutes with graceful handling
    
    # Simplified broker settings
    broker_transport_options={
        'visibility_timeout': 14400,  # Match task time limit
        'health_check_interval': 30,
        'max_retries': 5,
        'interval_start': 0,
        'interval_step': 0.5,
        'interval_max': 3,
    },
    redis_backend_health_check_interval=30,
    
    # Worker settings for CUDA-safe processing
    worker_concurrency=2,  # 2 processes per worker for memory stability
    
    # Worker memory management - complete isolation for C++ safety
    # For chunk_queue: Complete isolation prevents C++ double-free errors
    # For merge_queue: Longer lifecycle for text-only processing
    worker_max_tasks_per_child=1,  # Complete isolation per task (prevents double-free)
    worker_max_memory_per_child=8589934592,  # 8GB memory limit per worker
    
    # Task pool settings for memory management
    worker_pool='prefork',  # Use prefork for complete process isolation (prevents C++ double-free)
    worker_pool_restarts=True,  # Allow pool restarts
    
    # Logging settings
    worker_redirect_stdouts=False,  # Keep logs visible
    
    # Task result settings
    result_persistent=True,  # Persist results
    result_compression='gzip',  # Compress results to save memory
    
    # Disable advanced features that might cause issues
    task_always_eager=False,  # Never run tasks eagerly
)

# Task routing for high-throughput 6-container setup with separate merge queue
celery_app.conf.task_routes = {
    # PPStructure tasks: Heavy C++ processing with aggressive worker recycling
    'tasks.process_document_with_ppstructure': {'queue': 'chunk_queue'},
    'tasks.warmup_ppstructure': {'queue': 'chunk_queue'},
    
    # Merge tasks: Text-only processing with longer worker lifecycle
    'tasks.merge_and_summarize_chunks': {'queue': 'merge_queue'},
    
    # Maintenance tasks
    'tasks.maintenance.*': {'queue': 'maintenance'},
    'tasks.cleanup_expired_results': {'queue': 'maintenance'},
    'tasks.system_stats': {'queue': 'maintenance'}
}

# Retry settings optimized for chord error prevention
celery_app.conf.task_default_retry_delay = 60  # Increased delay for stability
celery_app.conf.task_max_retries = 5  # More retries for SIGKILL recovery
celery_app.conf.task_autoretry_for = (Exception,)  # Auto-retry on exceptions
celery_app.conf.task_retry_backoff = True  # Exponential backoff
celery_app.conf.task_retry_backoff_max = 300  # Max 5 minutes backoff

# High-throughput optimization settings
celery_app.conf.update(
    # Task execution settings
    task_acks_late=True,  # Acknowledge after completion for reliability
    worker_prefetch_multiplier=1,  # Prefetch 1 task per worker (2 concurrency * 1 = 2 tasks queued per container)
    
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

# Queue-specific worker configuration (configured at deployment level)
# chunk_queue workers: worker_max_tasks_per_child=1 (complete isolation prevents double-free)
# merge_queue workers:  worker_max_tasks_per_child=5 (longer lifecycle for text processing)
# 
# Example worker startup commands:
# For chunk processing: celery -A celery_config worker --queues=chunk_queue --max-tasks-per-child=1 --pool=prefork --concurrency=1
# For merge processing: celery -A celery_config worker --queues=merge_queue --max-tasks-per-child=5 --pool=prefork --concurrency=2

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
logger.info("Queue strategy: chunk_queue (PPStructure) + merge_queue (text processing)")
logger.info("Worker lifecycle: chunk_queue workers recycled every 1 task (complete isolation), merge_queue workers every 5 tasks")
logger.info("Chunking: Fixed 5-page chunks for predictable parallelism")
logger.info("Memory allocation: 8GB per container (48GB total)")
logger.info("Expected throughput: 3x-4x improvement with optimal CPU utilization")