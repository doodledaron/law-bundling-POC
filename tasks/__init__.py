"""
Task package for the law document processing system.
Contains modular tasks for document processing with PPStructure pipeline.
"""
import os

# Ensure required directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('chunks', exist_ok=True)

# Import all tasks directly - simplified approach for better reliability
from tasks.document_tasks import (
    process_document, 
    process_chunk
)

from tasks.chunking_tasks import (
    create_document_chunks,
    update_chunk_status
)

from tasks.maintenance import (
    cleanup_expired_results, 
    system_stats
)

from tasks.ppstructure_tasks import (
    process_document_with_ppstructure,
    warmup_ppstructure
)

from tasks.utils import (
    get_timestamp,
    get_unix_timestamp,
    calculate_duration,
    format_duration,
    update_job_status,
    update_job_timing
)

# Make tasks available for import
__all__ = [
    'process_document',
    'process_chunk', 
    'create_document_chunks',
    'update_chunk_status',
    'cleanup_expired_results',
    'system_stats',
    'process_document_with_ppstructure',
    'warmup_ppstructure'
]