"""
Task package for the law document processing system.
Contains modular tasks for document processing with PPStructure pipeline.
Streamlined for document-level chunking only.
"""
import os

# Ensure required directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('chunks', exist_ok=True)

print(f"🔧 Tasks module loading - streamlined for document processing")

# Always import utilities (lightweight, no dependencies)
from tasks.utils import (
    get_timestamp,
    get_unix_timestamp,
    calculate_duration,
    format_duration,
    update_job_status,
    update_job_timing,
    update_chunk_progress,
    update_merge_progress
)

# Base exports - always available
__all__ = [
    'get_timestamp',
    'get_unix_timestamp', 
    'calculate_duration',
    'format_duration',
    'update_job_status',
    'update_job_timing',
    'update_chunk_progress',
    'update_merge_progress'
]

# Import PPStructure tasks (main processor used by main.py)
try:
    print("🔧 Loading PPStructure tasks...")
    from tasks.ppstructure_tasks import (
        process_document_with_ppstructure,
        warmup_ppstructure
    )
    # Add to exports when successfully loaded
    __all__.extend(['process_document_with_ppstructure', 'warmup_ppstructure'])
    print("✅ PPStructure tasks loaded successfully")
    
except Exception as e:
    print(f"⚠️  Failed to load PPStructure tasks: {str(e)}")
    # PPStructure tasks not available, but utils still work

# Import Gemini-only processing tasks
try:
    print("🔧 Loading Gemini tasks...")
    from tasks.gemini_tasks import process_document_with_gemini_only
    __all__.append('process_document_with_gemini_only')
    print("✅ Gemini tasks loaded successfully")
except Exception as e:
    print(f"⚠️  Failed to load Gemini tasks: {str(e)}")

# Import merge/summarize tasks
try:
    print("🔧 Loading merge tasks...")
    from tasks.merge_tasks import merge_and_summarize_chunks
    __all__.append('merge_and_summarize_chunks')
    print("✅ Merge tasks loaded successfully")
except Exception as e:
    print(f"⚠️  Failed to load merge tasks: {str(e)}")

# Load maintenance tasks if needed
try:
    from tasks.maintenance import (
        cleanup_expired_results, 
        system_stats
    )
    __all__.extend(['cleanup_expired_results', 'system_stats'])
    print("✅ Maintenance tasks loaded")
except Exception as e:
    print(f"ℹ️  Maintenance tasks not available: {str(e)}")
    # Maintenance tasks not available, but core functionality still works

print(f"✅ Tasks module loaded with {len(__all__)} available functions")