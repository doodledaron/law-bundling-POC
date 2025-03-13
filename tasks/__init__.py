"""
Task package for the law document processing system.
Contains modular tasks for document processing, OCR, extraction, and maintenance.
"""
import os

# Ensure required directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('chunks', exist_ok=True)

# Import tasks for easy access
from tasks.document_tasks import (
    process_document, 
    process_large_document,
    process_small_document
)

from tasks.ocr_tasks import (
    process_pdf,
    process_image
)

from tasks.chunking_tasks import (
    create_document_chunks,
    get_next_chunk,
    update_chunk_status,
    merge_chunk_results
)

from tasks.extraction_tasks import (
    extract_document_info, 
    merge_extraction_results
)

from tasks.maintenance import (
    cleanup_expired_results, 
    system_stats
)

from tasks.utils import (
    get_timestamp,
    update_job_status,
    preprocess_image,
    clean_text,
    extract_nda_fields,
    generate_summary
)