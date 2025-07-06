"""
Task package for the law document processing system.
Contains modular tasks for document processing with PPStructure pipeline.
Enhanced with conditional imports based on worker type.
"""
import os

# Ensure required directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('chunks', exist_ok=True)

# Conditional imports based on worker type to avoid loading unnecessary modules
WORKER_TYPE = os.environ.get('WORKER_TYPE', 'all')
CELERY_WORKER_QUEUES = os.environ.get('CELERY_WORKER_QUEUES', 'all')

print(f"🔧 Tasks module loading for worker type: {WORKER_TYPE}, queues: {CELERY_WORKER_QUEUES}")

# Always import utilities (lightweight, no dependencies)
from tasks.utils import (
    get_timestamp,
    get_unix_timestamp,
    calculate_duration,
    format_duration,
    update_job_status,
    update_job_timing
)

# Base exports - available to all workers
__all__ = [
    'get_timestamp',
    'get_unix_timestamp', 
    'calculate_duration',
    'format_duration',
    'update_job_status',
    'update_job_timing'
]

# Load PPStructure tasks ONLY for PPStructure workers
if WORKER_TYPE == 'ppstructure':
    try:
        print("🔧 Loading PPStructure tasks for PPStructure worker...")
        from tasks.ppstructure_tasks import (
            process_document_with_ppstructure,
            warmup_ppstructure
        )
        __all__.extend(['process_document_with_ppstructure', 'warmup_ppstructure'])
        print("✅ PPStructure tasks loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load PPStructure tasks: {str(e)}")

# Load document_tasks ONLY for document coordination workers and API
if WORKER_TYPE in ['documents', 'all'] or WORKER_TYPE is None:
    try:
        print("📋 Loading document_tasks module for document coordination worker...")
        from tasks.document_tasks import (
            process_document
        )
        __all__.append('process_document')
        print("✅ document_tasks loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load document_tasks: {str(e)}")

# Load chunking tasks for API and maintenance workers (not specialized workers)
if WORKER_TYPE in ['all', 'maintenance'] or WORKER_TYPE is None:
    try:
        print("📋 Loading chunking_tasks module...")
        from tasks.chunking_tasks import (
            create_document_chunks,
            update_chunk_status
        )
        __all__.extend(['create_document_chunks', 'update_chunk_status'])
        print("✅ chunking_tasks loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load chunking_tasks: {str(e)}")

# Load maintenance tasks ONLY for maintenance workers
if WORKER_TYPE == 'maintenance':
    try:
        print("📋 Loading maintenance module...")
        from tasks.maintenance import (
            cleanup_expired_results, 
            system_stats
        )
        __all__.extend(['cleanup_expired_results', 'system_stats'])
        print("✅ maintenance loaded successfully")
    except Exception as e:
        print(f"⚠️  Failed to load maintenance: {str(e)}")

print(f"✅ Tasks module loaded with {len(__all__)} available functions for worker type: {WORKER_TYPE}")