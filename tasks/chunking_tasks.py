"""
Document chunking tasks for handling large documents.
Splits documents into manageable chunks for efficient processing.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import tempfile
import shutil
import redis
# Conditional imports for lite containers
try:
    import fitz  # PyMuPDF
    import cv2
    import numpy as np
    from PIL import Image
    HEAVY_DEPS_AVAILABLE = True
except ImportError as e:
    # In lite containers without heavy dependencies
    # Use print since logger is not yet defined
    print(f"INFO: Heavy dependencies not available in this container: {e}")
    HEAVY_DEPS_AVAILABLE = False

# Import utilities
from tasks.utils import (
    update_job_status,
    get_timestamp,
    format_duration
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
    # Check if heavy dependencies are available
    if not HEAVY_DEPS_AVAILABLE:
        raise RuntimeError("Document chunking requires heavy dependencies (PyMuPDF, OpenCV) - only available in document workers")
    
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
    Split a PDF into 20-page chunks with overlap.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the PDF file
        job_chunk_dir: Directory to store chunks
        overlap_pages: Number of pages to overlap between chunks
        
    Returns:
        dict: Information about the chunks
    """
    if not HEAVY_DEPS_AVAILABLE:
        raise RuntimeError("PDF chunking requires PyMuPDF - only available in document workers")
    
    try:
        # Open the PDF with PyMuPDF
        pdf_document = fitz.open(file_path)
        total_pages = pdf_document.page_count
        logger.info(f"PDF has {total_pages} pages")
        
        # Fixed 20-page chunks for robust processing
        pages_per_chunk = 20
        
        # Calculate number of chunks needed
        num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
        chunks = []
        chunks_info = []
        
        logger.info(f"Creating {num_chunks} chunks of maximum {pages_per_chunk} pages each")
        
        # Create chunks with strict page limits
        for i in range(num_chunks):
            # Calculate page ranges - ensure exactly 20 pages per chunk (except last)
            start_page = i * pages_per_chunk
            end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)
            
            # Ensure we don't exceed the page limit
            actual_pages = end_page - start_page + 1
            if actual_pages > pages_per_chunk:
                end_page = start_page + pages_per_chunk - 1
                actual_pages = pages_per_chunk
            
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
            
            # Create metadata for this chunk with continuous page numbering
            chunk_metadata = {
                "chunk_id": chunk_id,
                "job_id": job_id,
                "start_page": start_page + 1,  # Convert to 1-indexed for user display
                "end_page": end_page + 1,      # Convert to 1-indexed for user display
                "total_pages": actual_pages,
                "has_overlap": False,  # Remove overlap for now to prevent memory issues
                "path": chunk_path
            }
            
            # Save metadata file
            metadata_path = os.path.join(job_chunk_dir, f"{chunk_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(chunk_metadata, f)
            
            chunks.append(chunk_metadata)
            chunks_info.append({
                "chunk_id": chunk_id,
                "pages": f"{start_page + 1}-{end_page + 1}",  # Continuous page numbering (1-20, 21-40, etc.)
                "total_pages": actual_pages
            })
            
            logger.info(f"Chunk {chunk_id}: pages {start_page + 1}-{end_page + 1} ({actual_pages} pages)")
        
        # Close the original PDF
        pdf_document.close()
        
        # Create manifest
        manifest = {
            "job_id": job_id,
            "total_pages": total_pages,
            "num_chunks": num_chunks,
            "pages_per_chunk": pages_per_chunk,
            "overlap_pages": 0,  # Disabled overlap for memory stability
            "chunks": [c["chunk_id"] for c in chunks]
        }
        
        # Save manifest
        manifest_path = os.path.join(job_chunk_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        logger.info(f"Created {num_chunks} chunks for job {job_id} with maximum {pages_per_chunk} pages each")
        
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
    if not HEAVY_DEPS_AVAILABLE:
        raise RuntimeError("Image processing requires PIL/OpenCV - only available in document workers")
        
    try:
        # Create a unique ID for this chunk
        chunk_id = "chunk_0000"
        
        # Get file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Create path for chunk
        chunk_path = os.path.join(job_chunk_dir, f"{chunk_id}{file_ext}")
        
        # Copy original file to chunk location
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
        
        # Simple merge logic for PPStructure results
        merged_results = _merge_ppstructure_results(job_id, chunk_results)
        
        return merged_results
        
    except Exception as e:
        logger.error(f"Error merging chunk results for job {job_id}: {str(e)}")
        return {"status": "error", "message": f"Error merging results: {str(e)}"}


def _merge_ppstructure_results(job_id, chunk_results):
    """
    Merge PPStructure results from multiple chunks.
    
    Args:
        job_id: The job identifier
        chunk_results: List of results from individual chunks
        
    Returns:
        dict: Merged results
    """
    try:
        if not chunk_results:
            return {"status": "error", "message": "No chunk results to merge"}
        
        # Initialize merged data
        merged_text = []
        merged_entities = {}
        total_confidence_scores = []
        total_cost = 0.0
        total_pages = 0
        total_chunks_processed = 0
        
        # Performance tracking
        processing_times = []
        total_tokens = 0
        
        # Merge results from each chunk
        for i, result in enumerate(chunk_results):
            chunk_num = i + 1
            
            # Extract and mark text with chunk information
            if 'extracted_text' in result and result['extracted_text']:
                chunk_header = f"\n--- CHUNK {chunk_num} ---\n"
                merged_text.append(chunk_header + result['extracted_text'])
            
            # Collect confidence scores
            if 'average_confidence_formatted' in result:
                # Extract numeric value from formatted string like "85.5%"
                confidence_str = result['average_confidence_formatted'].replace('%', '')
                try:
                    confidence_val = float(confidence_str)
                    total_confidence_scores.append(confidence_val)
                except:
                    pass
            
            # Merge entities from each chunk
            if 'entities' in result and isinstance(result['entities'], dict):
                for key, value in result['entities'].items():
                    if key not in merged_entities:
                        merged_entities[key] = []
                    
                    # Handle different value types
                    if isinstance(value, list):
                        merged_entities[key].extend(value)
                    elif isinstance(value, dict):
                        # For dict values, convert to readable format
                        formatted_value = f"Chunk {chunk_num}: {str(value)}"
                        merged_entities[key].append(formatted_value)
                    else:
                        merged_entities[key].append(f"Chunk {chunk_num}: {str(value)}")
            
            # Aggregate performance data
            if 'performance' in result:
                perf = result['performance']
                if 'total_pages' in perf:
                    total_pages += perf['total_pages']
                if 'total_chunks_processed' in perf:
                    total_chunks_processed += perf['total_chunks_processed']
                if 'processing_time' in perf and 'seconds' in perf['processing_time']:
                    processing_times.append(perf['processing_time']['seconds'])
            
            # Aggregate cost information
            if 'cost_info' in result:
                cost_info = result['cost_info']
                if 'estimated_cost' in cost_info:
                    total_cost += cost_info.get('estimated_cost', 0.0)
                if 'token_usage' in cost_info and 'total_tokens' in cost_info['token_usage']:
                    total_tokens += cost_info['token_usage'].get('total_tokens', 0)
        
        # Calculate overall metrics
        average_confidence = sum(total_confidence_scores) / len(total_confidence_scores) if total_confidence_scores else 0
        total_processing_time = sum(processing_times) if processing_times else 0
        
        # Combine all text for final document
        full_text = '\n\n'.join(merged_text)
        
        # Generate a comprehensive summary for the entire document using all combined text
        summary = f"Multi-chunk document processed successfully. Contains {len(chunk_results)} chunks with {total_pages} total pages and {len(full_text)} characters of extracted text."
        
        # Generate final summary using ALL combined text from ALL chunks
        final_summary = summary  # Default fallback
        if full_text.strip():
            try:
                # Import text processor to generate final summary (dynamic import)
                from text_based_processor import TextBasedProcessor
                
                # Initialize text processor
                text_processor = TextBasedProcessor()
                
                # Generate comprehensive summary using all combined text
                logger.info(f"Generating final summary for job {job_id} using {len(full_text)} characters of combined text")
                
                summary_result = text_processor.summarize_document_text(full_text, f"Complete_Document_{job_id}")
                
                if summary_result and 'summary' in summary_result:
                    final_summary = summary_result['summary']
                    
                    # Also extract entities from the combined text
                    if 'analysis' in summary_result and summary_result['analysis']:
                        merged_entities.update(summary_result['analysis'])
                    
                    # Update cost information
                    if 'estimated_cost' in summary_result:
                        total_cost += summary_result.get('estimated_cost', 0.0)
                    if 'usage_info' in summary_result and 'total_tokens' in summary_result['usage_info']:
                        total_tokens += summary_result['usage_info'].get('total_tokens', 0)
                        
                logger.info(f"Successfully generated final summary for job {job_id}")
                
            except Exception as e:
                logger.error(f"Error generating final summary for job {job_id}: {str(e)}")
                final_summary = f"Document processed with {len(chunk_results)} chunks. Summary generation failed: {str(e)}"
        
        # Create performance summary
        performance_data = {
            "total_chunks": len(chunk_results),
            "total_pages": total_pages,
            "total_chunks_processed": total_chunks_processed,
            "total_processing_time": {
                "seconds": total_processing_time,
                "formatted": format_duration(total_processing_time)
            },
            "average_processing_time_per_chunk": {
                "seconds": total_processing_time / len(chunk_results) if chunk_results else 0,
                "formatted": format_duration(total_processing_time / len(chunk_results)) if chunk_results else "0s"
            },
            "parallel_processing": True
        }
        
        return {
            "status": "completed",
            "extracted_text": full_text,
            "entities": merged_entities,
            "summary": final_summary,
            "average_confidence": average_confidence,
            "average_confidence_formatted": f"{average_confidence:.1f}%",
            "performance": performance_data,
            "cost_info": {
                "total_estimated_cost": total_cost,
                "total_tokens_used": total_tokens,
                "cost_per_chunk": total_cost / len(chunk_results) if chunk_results else 0
            },
            "processing_method": "chunked_parallel",
            "job_id": job_id
        }
        
    except Exception as e:
        logger.error(f"Error merging PPStructure results: {str(e)}")
        return {"status": "error", "message": f"Error merging PPStructure results: {str(e)}"}