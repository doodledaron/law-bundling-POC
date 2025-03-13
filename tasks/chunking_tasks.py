"""
Document chunking tasks for handling large documents.
Splits documents into manageable chunks for efficient processing.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import tempfile
import redis
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

# Import utilities
from tasks.utils import (
    update_job_status,
    get_timestamp
)

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Create chunks directory
os.makedirs('chunks', exist_ok=True)

@shared_task(name='tasks.chunking.create_document_chunks')
def create_document_chunks(job_id, file_path, file_name, overlap_pages=2):
    """
    Split a document into manageable chunks for processing.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original filename
        overlap_pages: Number of pages to overlap between chunks
        
    Returns:
        dict: Information about the chunks created
    """
    try:
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': 'Analyzing document and creating chunks',
            'progress': 10,
            'updated_at': get_timestamp()
        })
        
        # Create job-specific directory for chunks
        job_chunk_dir = os.path.join('chunks', job_id)
        os.makedirs(job_chunk_dir, exist_ok=True)
        
        # Get file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Select appropriate chunking method
        if file_ext == '.pdf':
            chunk_info = _chunk_pdf(job_id, file_path, job_chunk_dir, overlap_pages)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            chunk_info = _process_image_as_chunk(job_id, file_path, file_name, job_chunk_dir)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Update job status with chunking information
        update_job_status(redis_client, job_id, {
            'num_chunks': chunk_info['num_chunks'],
            'chunks_info': chunk_info['chunks_info'],
            'message': f'Created {chunk_info["num_chunks"]} chunks for processing',
            'progress': 20,
            'updated_at': get_timestamp()
        })
        
        return chunk_info
    
    except Exception as e:
        logger.error(f"Error creating document chunks for job {job_id}: {str(e)}")
        
        # Update job status on failure
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': f"Chunking error: {str(e)}",
            'message': 'Failed to create document chunks',
            'updated_at': get_timestamp()
        })
        
        # Re-raise to allow Celery to handle the error
        raise


def _chunk_pdf(job_id, file_path, job_chunk_dir, overlap_pages=2):
    """
    Split a PDF into chunks with overlap.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the PDF file
        job_chunk_dir: Directory to store chunks
        overlap_pages: Number of pages to overlap between chunks
        
    Returns:
        dict: Information about the chunks
    """
    try:
        # Open the PDF with PyMuPDF
        pdf_document = fitz.open(file_path)
        total_pages = pdf_document.page_count
        logger.info(f"PDF has {total_pages} pages")
        
        # Determine optimal chunk size based on document size
        if total_pages <= 50:
            pages_per_chunk = total_pages  # Process as single chunk
        elif total_pages <= 200:
            pages_per_chunk = 20
        elif total_pages <= 500:
            pages_per_chunk = 25
        else:
            pages_per_chunk = 50
        
        # Calculate number of chunks
        num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
        chunks = []
        chunks_info = []
        
        # Create chunks with overlap
        for i in range(num_chunks):
            # Start page includes overlap from previous chunk
            start_page = max(0, i * pages_per_chunk - overlap_pages)
            
            # End page includes overlap into next chunk
            end_page = min((i + 1) * pages_per_chunk + overlap_pages - 1, total_pages - 1)
            
            # Create a unique ID for this chunk
            chunk_id = f"chunk_{i:04d}"
            
            # Create a new PDF for this chunk
            chunk_path = os.path.join(job_chunk_dir, f"{chunk_id}.pdf")
            
            # Create new PDF
            chunk_pdf = fitz.open()
            
            # Copy pages from original to chunk
            for page_num in range(start_page, end_page + 1):
                chunk_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            
            # Save the chunk PDF
            chunk_pdf.save(chunk_path)
            chunk_pdf.close()
            
            # Create metadata for this chunk
            chunk_metadata = {
                "chunk_id": chunk_id,
                "job_id": job_id,
                "start_page": start_page,
                "end_page": end_page,
                "is_overlap_start": start_page > 0 and start_page != i * pages_per_chunk,
                "is_overlap_end": end_page < total_pages - 1 and end_page != (i + 1) * pages_per_chunk - 1,
                "original_start": i * pages_per_chunk,
                "original_end": min((i + 1) * pages_per_chunk - 1, total_pages - 1),
                "path": chunk_path
            }
            
            # Save metadata file
            metadata_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(chunk_metadata, f)
            
            chunks.append(chunk_metadata)
            chunks_info.append({
                "chunk_id": chunk_id,
                "pages": f"{start_page}-{end_page}",
                "total_pages": end_page - start_page + 1
            })
        
        # Close the original PDF
        pdf_document.close()
        
        # Create manifest
        manifest = {
            "job_id": job_id,
            "total_pages": total_pages,
            "num_chunks": num_chunks,
            "chunks": [c["chunk_id"] for c in chunks]
        }
        
        # Save manifest
        manifest_path = os.path.join(job_chunk_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        logger.info(f"Created {num_chunks} chunks for job {job_id}")
        
        return {
            "job_id": job_id,
            "num_chunks": num_chunks,
            "chunks": chunks,
            "chunks_info": chunks_info,
            "manifest_path": manifest_path
        }
    
    except Exception as e:
        logger.error(f"Error chunking PDF for job {job_id}: {str(e)}")
        raise


def _process_image_as_chunk(job_id, file_path, file_name, job_chunk_dir):
    """
    Treat an image as a single chunk (no splitting).
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the image file
        file_name: Original filename
        job_chunk_dir: Directory to store chunks
        
    Returns:
        dict: Information about the single chunk
    """
    try:
        # Create a unique ID for this chunk
        chunk_id = "chunk_0000"
        
        # Get file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Create path for chunk
        chunk_path = os.path.join(job_chunk_dir, f"{chunk_id}{file_ext}")
        
        # Copy original file to chunk location
        import shutil
        shutil.copy2(file_path, chunk_path)
        
        # Create metadata for this chunk
        chunk_metadata = {
            "chunk_id": chunk_id,
            "job_id": job_id,
            "path": chunk_path,
            "original_filename": file_name
        }
        
        # Save metadata file
        metadata_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(chunk_metadata, f)
        
        # Create manifest
        manifest = {
            "job_id": job_id,
            "total_pages": 1,
            "num_chunks": 1,
            "chunks": [chunk_id]
        }
        
        # Save manifest
        manifest_path = os.path.join(job_chunk_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        chunks_info = [{
            "chunk_id": chunk_id,
            "pages": "0-0",
            "total_pages": 1
        }]
        
        logger.info(f"Created single image chunk for job {job_id}")
        
        return {
            "job_id": job_id,
            "num_chunks": 1,
            "chunks": [chunk_metadata],
            "chunks_info": chunks_info,
            "manifest_path": manifest_path
        }
    
    except Exception as e:
        logger.error(f"Error processing image as chunk for job {job_id}: {str(e)}")
        raise


def get_next_chunk(job_id):
    """
    Get the next pending chunk for a job.
    
    Args:
        job_id: The job identifier
        
    Returns:
        dict: Next pending chunk metadata or None if no pending chunks
    """
    try:
        # Get job chunk directory
        job_chunk_dir = os.path.join('chunks', job_id)
        
        if not os.path.exists(job_chunk_dir):
            return None
        
        # Get the manifest
        manifest_path = os.path.join(job_chunk_dir, "manifest.json")
        
        if not os.path.exists(manifest_path):
            return None
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Look for the next pending chunk
        for chunk_id in manifest.get("chunks", []):
            # Get chunk metadata
            chunk_meta_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
            
            if os.path.exists(chunk_meta_path):
                with open(chunk_meta_path, 'r') as f:
                    chunk_metadata = json.load(f)
                
                # Check if chunk is pending (no status or status is 'pending')
                if "status" not in chunk_metadata or chunk_metadata["status"] == "pending":
                    return chunk_metadata
        
        # No pending chunks found
        return None
    
    except Exception as e:
        logger.error(f"Error getting next chunk for job {job_id}: {str(e)}")
        return None


def update_chunk_status(job_id, chunk_id, status, results=None):
    """
    Update the status of a chunk and optionally save processing results.
    
    Args:
        job_id: The job identifier
        chunk_id: The chunk identifier
        status: New status (e.g., 'processing', 'completed', 'failed')
        results: Optional results data to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get job chunk directory
        job_chunk_dir = os.path.join('chunks', job_id)
        
        # Get chunk metadata path
        chunk_meta_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
        
        if not os.path.exists(chunk_meta_path):
            logger.error(f"Chunk metadata not found for job {job_id}, chunk {chunk_id}")
            return False
        
        # Load existing metadata
        with open(chunk_meta_path, 'r') as f:
            chunk_metadata = json.load(f)
        
        # Update status
        chunk_metadata["status"] = status
        chunk_metadata["updated_at"] = get_timestamp()
        
        # Save results if provided
        if results:
            # Save results to a separate file
            results_path = os.path.join(job_chunk_dir, f"{chunk_id}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f)
            
            # Update metadata with results path
            chunk_metadata["results_path"] = results_path
        
        # Save updated metadata
        with open(chunk_meta_path, 'w') as f:
            json.dump(chunk_metadata, f)
        
        logger.info(f"Updated chunk {chunk_id} status to {status} for job {job_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating chunk status for job {job_id}, chunk {chunk_id}: {str(e)}")
        return False


@shared_task(name='tasks.chunking.merge_chunk_results')
def merge_chunk_results(job_id):
    """
    Merge results from all chunks of a job.
    
    Args:
        job_id: The job identifier
        
    Returns:
        dict: Merged results
    """
    try:
        # Import here to avoid circular imports
        from tasks.extraction_tasks import merge_extraction_results
        
        # Get job chunk directory
        job_chunk_dir = os.path.join('chunks', job_id)
        
        if not os.path.exists(job_chunk_dir):
            return {"status": "error", "message": "Job chunk directory not found"}
        
        # Get the manifest
        manifest_path = os.path.join(job_chunk_dir, "manifest.json")
        
        if not os.path.exists(manifest_path):
            return {"status": "error", "message": "Manifest not found"}
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Collect results from all chunks
        chunk_results = []
        
        for chunk_id in manifest.get("chunks", []):
            # Get chunk results
            results_path = os.path.join(job_chunk_dir, f"{chunk_id}_results.json")
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                chunk_results.append(results)
        
        # Merge results using extraction task
        merged_results = merge_extraction_results(job_id, chunk_results)
        
        return merged_results
        
    except Exception as e:
        logger.error(f"Error merging chunk results for job {job_id}: {str(e)}")
        return {"status": "error", "message": f"Error merging results: {str(e)}"}