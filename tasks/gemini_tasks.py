"""
Gemini-only document processing task.
Processes documents using only Gemini AI without PPStructure layout analysis.
Extracted from ppstructure_tasks.py for better modularity.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import shutil
import datetime
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed

from tasks.utils import get_unix_timestamp, calculate_duration, update_job_status, get_timestamp, update_chunk_progress
from tasks.helpers import update_active_processes_worker
from text_based_processor import TextBasedProcessor

# Import pdf2image conditionally
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = get_task_logger(__name__)

# Initialize TextBasedProcessor for Gemini integration
text_processor = TextBasedProcessor()


@shared_task(name='tasks.process_document_with_gemini_only')
def process_document_with_gemini_only(job_id, file_path, file_name, generate_summary=True, actual_start_page=1):
    """
    Process a document using only Gemini AI without PPStructure layout analysis.
    
    This method is optimized for smaller documents or when PPStructure processing is not needed.
    It converts documents to images and uses Gemini for text extraction and analysis.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the document file
        file_name: Original file name
        generate_summary: Whether to generate summary (False for chunks, True for whole documents)
        actual_start_page: The actual starting page number (for continuous numbering across chunks)
        
    Returns:
        dict: Processing results compatible with PPStructure task format
    """
    try:
        logger.info(f"🤖 Starting Gemini-only processing for job {job_id}")
        
        # Update active process count (increment on start)
        update_active_processes_worker(1)
        
        # Start overall timing
        overall_start_time = get_unix_timestamp()
        
        # Detect if this is a chunk based on filename
        is_chunk = "chunk_" in file_name
        chunk_id = None
        if is_chunk:
            # Extract chunk_id from filename
            chunk_parts = file_name.split("_")
            if len(chunk_parts) >= 2:
                chunk_id = f"{chunk_parts[0]}_{chunk_parts[1]}"  # e.g., "chunk_0001"
                logger.info(f"🔄 Processing Gemini-only chunk: {chunk_id}")
        
        # Create result directories
        result_dir = os.path.join("results", job_id)
        images_dir = os.path.join(result_dir, "images")
        
        # Create result directories
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Process based on file type
        file_ext = os.path.splitext(file_name)[1].lower()
        
        logger.info(f"🤖 [GEMINI-ONLY] Document analysis:")
        logger.info(f"   📝 File: {file_name}")
        logger.info(f"   📄 Type: {file_ext}")
        logger.info(f"   🆔 Job ID: {job_id}")
        logger.info(f"   📦 Is chunk: {is_chunk}")
        logger.info(f"   📄 Starting page: {actual_start_page}")
        
        if file_ext == '.pdf':
            # Convert PDF to images for text extraction
            logger.info(f"📄 [GEMINI-ONLY] Converting PDF to images...")
            images = convert_from_bytes(
                open(file_path, "rb").read(),
                dpi=100,
                fmt='jpeg'
            )
            
            total_pages = len(images)
            logger.info(f"📊 [GEMINI-ONLY] PDF converted to {total_pages} page images")
            
            # Save all page images with correct page numbering
            image_paths = []
            for i, image in enumerate(images):
                actual_page_num = actual_start_page + i
                image_path = os.path.join(images_dir, f"page_{actual_page_num}.jpg")
                image.save(image_path)
                image_paths.append(image_path)
                logger.info(f"   📄 Page {actual_page_num}: Saved as {os.path.basename(image_path)}")
        else:
            # Single image file
            logger.info(f"🖼️  [GEMINI-ONLY] Processing single image file")
            image_path = os.path.join(images_dir, f"page_{actual_start_page}.jpg")
            shutil.copy2(file_path, image_path)
            image_paths = [image_path]
            total_pages = 1
            logger.info(f"   📄 Single page: Saved as {os.path.basename(image_path)}")
        
        # Process each page with Gemini for text extraction in parallel batches of 5
        all_ocr_text = []
        
        logger.info(f"🤖 [GEMINI-ONLY] Processing {len(image_paths)} pages with Gemini AI in parallel batches of 5")
        
        def process_single_page(page_info):
            """Helper function to process a single page with Gemini"""
            i, img_path = page_info
            actual_page_num = actual_start_page + i
            
            try:
                # Read image and convert to bytes for Gemini
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                
                # Use TextBasedProcessor to extract text from image
                page_text = text_processor.extract_text_from_image(img_bytes, f"Page {actual_page_num}")
                
                if page_text and page_text.strip():
                    result_text = [f"--- PAGE {actual_page_num} ---", page_text.strip()]
                    logger.info(f"   ✅ Page {actual_page_num}: {len(page_text)} characters extracted")
                    return (i, actual_page_num, result_text, None)
                else:
                    logger.warning(f"   ⚠️ Page {actual_page_num}: No text extracted")
                    result_text = [f"--- PAGE {actual_page_num} ---", "[No text extracted from this page]"]
                    return (i, actual_page_num, result_text, None)
                
            except Exception as e:
                logger.error(f"   ❌ Page {actual_page_num}: Error - {str(e)}")
                result_text = [f"--- PAGE {actual_page_num} ---", f"[Error extracting text: {str(e)}]"]
                return (i, actual_page_num, result_text, str(e))
        
        # Process pages in batches of 5
        batch_size = 5
        page_results = [None] * len(image_paths)  # Pre-allocate to maintain order
        
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_pages = [(i, image_paths[i]) for i in range(batch_start, batch_end)]
            
            logger.info(f"🔄 Processing batch {batch_start//batch_size + 1}: pages {batch_start + 1}-{batch_end}")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_page = {executor.submit(process_single_page, page_info): page_info for page_info in batch_pages}
                
                for future in as_completed(future_to_page):
                    try:
                        i, actual_page_num, result_text, error = future.result()
                        page_results[i] = result_text
                        logger.info(f"📄 [PAGE-{i+1:02d}] Completed processing page {actual_page_num}")
                    except Exception as e:
                        page_info = future_to_page[future]
                        i, img_path = page_info
                        actual_page_num = actual_start_page + i
                        logger.error(f"   ❌ Page {actual_page_num}: Batch processing error - {str(e)}")
                        page_results[i] = [f"--- PAGE {actual_page_num} ---", f"[Batch processing error: {str(e)}]"]
        
        # Combine results in correct page order
        for result_text in page_results:
            if result_text:
                all_ocr_text.extend(result_text)
        
        # Combine all text
        combined_text = "\n".join(all_ocr_text)
        logger.info(f"📄 Combined text length: {len(combined_text)} characters from {len(image_paths)} pages")
        
        # Generate document summary using Gemini (only if requested)
        summary_result = {}
        if generate_summary and combined_text.strip():
            try:
                logger.info(f"🧠 Generating summary with Gemini for job {job_id}")
                summary_result = text_processor.summarize_document_text(combined_text, file_name)
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                summary_result = {
                    "summary": "Summary generation failed",
                    "analysis": {"error": str(e)},
                    "usage_info": {"total_tokens": 0},
                    "estimated_cost": 0.0
                }
        elif not generate_summary:
            # For chunks, don't generate summary but provide placeholder
            summary_result = {
                "summary": None,
                "analysis": {},
                "usage_info": {"total_tokens": 0},
                "estimated_cost": 0.0
            }
        
        # Calculate performance metrics
        overall_end_time = get_unix_timestamp()
        processing_duration = calculate_duration(overall_start_time, overall_end_time)
        
        # Save results (compatible with PPStructure format)
        results_path = os.path.join(result_dir, "results.json")
        
        # Create average confidence (Gemini doesn't provide confidence scores)
        average_confidence_formatted = "N/A (Gemini AI)"
        
        # Save combined text to a separate file
        combined_text_path = os.path.join(result_dir, "combined_text.txt")
        with open(combined_text_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Extract structured information from summary result
        extracted_info = {}
        if generate_summary and summary_result.get("extracted_info"):
            extracted_info = summary_result["extracted_info"]
        else:
            # Default structure for chunks
            extracted_info = {
                "key_dates": "Not available",
                "main_parties": "Not available", 
                "case_reference_numbers": "Not available",
                "full_analysis": "No summary generated for individual chunks"
            }
        
        # Prepare clean results (compatible with PPStructure format)
        clean_results = {
            "filename": file_name,
            "job_id": job_id,
            "processing_completed_at": datetime.datetime.now().isoformat(),
            "total_pages": len(image_paths),
            "summary": summary_result.get("summary", "Summary not available"),
            "date": summary_result.get("date", "undated"),
            "extracted_info": extracted_info,
            "combined_text_path": f"results/{job_id}/combined_text.txt",
            "combined_text": combined_text,
            "estimated_cost": summary_result.get("estimated_cost", 0.0),
            "token_usage": summary_result.get("token_usage", {}),
            "processing_time_seconds": processing_duration.get("seconds", 0),
            "average_confidence_formatted": average_confidence_formatted,
            "processing_method": "gemini_only_complete"
        }
        
        # Save clean results to results.json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        # Update chunk progress - get total chunks from Redis using fresh connection
        try:
            # Import redis to create a fresh connection for this task
            import redis
            task_redis_client = redis.Redis.from_url(
                os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
            )
            
            job_data = task_redis_client.get(f"job:{job_id}")
            if job_data:
                job_status = json.loads(job_data)
                total_chunks = job_status.get('num_chunks', 1)
                current_progress = update_chunk_progress(task_redis_client, job_id, total_chunks)
                logger.info(f"📊 Updated chunk progress for job {job_id}: {current_progress}%")
        except Exception as e:
            logger.warning(f"Could not update chunk progress for job {job_id}: {str(e)}")
        
        # Update active process count (decrement on completion)
        update_active_processes_worker(-1)
        
        logger.info(f"✅ Gemini-only processing completed for job {job_id}")
        logger.info(f"📊 Processed {len(image_paths)} pages in {processing_duration['formatted']}")
        
        # For complete documents (not chunks), update job status to COMPLETED
        if not is_chunk:
            # Initialize Redis client
            redis_client = redis.Redis.from_url(
                os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
            )
            
            # CRITICAL: Do NOT mark as COMPLETED here - only the merge task should do that
            # Individual chunk tasks should never mark jobs as COMPLETED to avoid race conditions
            update_job_status(redis_client, job_id, {
                'status': 'PROCESSING',  # Keep as PROCESSING until merge task completes
                'message': f'Document "{file_name}" chunk processed successfully with {len(image_paths)} pages using AI text analysis, finalizing results...',
                'progress': 90,  # High progress but not 100%
                'results_path': results_path,
                'combined_text_path': combined_text_path,
                'total_pages': len(image_paths),
                'processing_completed_at': datetime.datetime.now().isoformat(),
                'average_confidence_formatted': average_confidence_formatted,
                'estimated_cost': summary_result.get("estimated_cost", 0.0),
                'processing_time_seconds': processing_duration.get("seconds", 0),
                'processing_method': 'gemini_only_complete',
                'filename': file_name,
                'updated_at': get_timestamp()
            })
            logger.info(f"📊 Job {job_id} marked as COMPLETED in Redis")
        
        # Return results compatible with PPStructure task format
        return {
            "job_id": job_id,
            "status": "CHUNK_COMPLETED" if is_chunk else "COMPLETED",
            "chunk_id": chunk_id if is_chunk and chunk_id else f"chunk_{actual_start_page}",
            "filename": file_name,
            "results_path": results_path,
            "message": f"AI text analysis completed: {len(image_paths)} pages",
            "processing_method": "gemini_only",
            # Include the actual extracted data for use by the merge function
            "combined_text": combined_text,
            "extracted_text": combined_text,  # Alias for compatibility
            "average_confidence_formatted": average_confidence_formatted,
            "cost_info": {"estimated_cost": summary_result.get("estimated_cost", 0.0)},
            "processing_time": processing_duration,
            "total_pages": len(image_paths),
            "summary": summary_result.get("summary", None),
            "extracted_info": extracted_info
        }
        
    except Exception as e:
        logger.error(f"Error in Gemini-only processing: {str(e)}")
        
        # Update active process count (decrement on failure)
        update_active_processes_worker(-1)
        
        # Return chunk failure result instead of raising exception
        return {
            "job_id": job_id,
            "status": "CHUNK_FAILED",
            "chunk_id": file_name,
            "filename": file_name,
            "error": str(e),
            "message": f"AI text analysis failed: {str(e)}",
            "processing_method": "gemini_only",
            "combined_text": "",
            "extracted_text": "",
            "total_pages": 0,
            "processing_time": {"seconds": 0},
            "summary": None,
            "extracted_info": {}
        }
