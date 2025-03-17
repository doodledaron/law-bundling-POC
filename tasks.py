import logging
import numpy as np
import cv2
import pymupdf  # PyMuPDF
import re
import paddleocr
from transformers import pipeline
from celery import Celery, signals  # Added signals import
import sys
import os
from logging.handlers import RotatingFileHandler
import pickle
from redis import Redis
import uuid 
import gc  # Add garbage collection module

gc.enable()  # Ensure garbage collection is enabled
# Set threshold for garbage collection to be more aggressive
gc.set_threshold(700, 10, 5)  # Adjust these values based on your memory requirements

# Configure Celery
celery_app = Celery('tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,  # Add this line
    worker_max_tasks_per_child=10,     # Restart worker after 10 tasks
    worker_max_memory_per_child=1024 * 1024 * 1024,  # 1GB memory limit
    worker_concurrency=2,  # Limit to 2 concurrent tasks
)

# Create a custom handler that flushes after each log
class FlushingHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Set up logging with proper flushing
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, "document_processor.log")

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        FlushingFileHandler(log_file, mode='a'),
        FlushingHandler(sys.stdout)
    ]
)

# Create logger with explicit handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger to avoid duplicate logs

# Add explicit handlers to our logger
file_handler = FlushingFileHandler(log_file, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = FlushingHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info("Starting application initialization")

# Configurable thresholds for chunking
PAGE_THRESHOLD = 10  # Process PDFs with more than 10 pages in chunks
CHUNK_SIZE = 5       # Process 5 pages per chunk


def cleanup_memory(label=""):
    """
    Perform memory cleanup to reclaim resources
    """
    # Log memory before cleanup if psutil is available
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)
        logger.info(f"{label} Memory before cleanup: {memory_before:.2f} MB")
    except ImportError:
        memory_before = 0
        logger.info(f"{label} Memory cleanup started (psutil not available for measurements)")
    
    # Run multiple garbage collection cycles
    collected = 0
    for i in range(3):
        collected += gc.collect()
    
    logger.info(f"{label} Memory cleanup collected {collected} objects")
    
    # Try to release memory to the OS (Linux/WSL2)
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        if hasattr(libc, 'malloc_trim'):
            result = libc.malloc_trim(0)
            if result > 0:
                logger.info(f"{label} Successfully released memory to OS")
    except Exception:
        pass
    
    # Log memory after cleanup if psutil is available
    try:
        import psutil
        process = psutil.Process()
        memory_after = process.memory_info().rss / (1024 * 1024)
        logger.info(f"{label} Memory after cleanup: {memory_after:.2f} MB")
        if memory_before > 0:
            logger.info(f"{label} Memory reduced by: {memory_before - memory_after:.2f} MB")
    except ImportError:
        pass
    
    flush_logs()


def clean_variables(*variables):
    """
    Explicitly delete variables and run garbage collection.
    
    Args:
        *variables: Variables to delete
        
    Example:
        clean_variables(img_data, nparr, img, processed_img, result)
    """
    for var in variables:
        if var is not None:
            var_type = type(var).__name__
            try:
                # Try to get variable size (may not work for all types)
                import sys
                var_size = sys.getsizeof(var) / (1024 * 1024)
                logger.debug(f"Deleting {var_type} object (approx. {var_size:.2f} MB)")
            except:
                logger.debug(f"Deleting {var_type} object")
            
            # Delete the variable
            del var
    
    # Run quick garbage collection to reclaim memory
    gc.collect(0)  # Quick collection of youngest generation only


# Add signal handlers for better memory management
@signals.task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Run before each task to ensure clean memory environment"""
    logger.info(f"Preparing to run task {task.name}[{task_id}]")
    gc.collect()  # Light GC before task

@signals.task_postrun.connect
def task_postrun_handler(task_id, task, *args, retval=None, state=None, **kwargs):
    """Clean up memory after each task"""
    logger.info(f"Finished task {task.name}[{task_id}] with state {state}")
    cleanup_memory(f"task_postrun:{task.name}")

def flush_logs():
    """Helper function to ensure logs are flushed"""
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def initialize_models():
    """
    Initialize all models used in the document processing pipeline.
    This function loads models once at startup rather than repeatedly during processing.
    
    Returns:
        tuple: Containing initialized models (ocr, summarizer)
    """
    logger.info("Starting initialization of models")
    flush_logs()
    
    # Initialize models with error handling
    ocr_model = None
    summarizer_model = None
    
    # Initialize PaddleOCR
    try:
        logger.info("Initializing PaddleOCR model")
        ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
        logger.info("PaddleOCR model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR model: {str(e)}", exc_info=True)
        flush_logs()
        raise RuntimeError(f"Critical error: PaddleOCR initialization failed: {str(e)}")
    
    # Initialize T5 summarization model
    try:
        logger.info("Initializing T5 summarization model")
        model_name = "t5-small"  # Make this configurable if needed
        summarizer_model = pipeline("summarization", model=model_name)
        logger.info(f"T5 summarization model ({model_name}) initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize T5 model: {str(e)}", exc_info=True)
        flush_logs()
        # Continue even if summarizer fails as it's not critical for core functionality
        logger.warning("Will continue without summarization capability")
        summarizer_model = None
    
    flush_logs()
    logger.info("Model initialization complete")
    
    return ocr_model, summarizer_model


# Initialize models
try:
    logger.info("Initializing document processing models")
    ocr, summarizer = initialize_models()
    logger.info("Models initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize models: {str(e)}")
    sys.exit(1)  # Exit if critical models can't be loaded


"""
1. User uploads PDF → main.py
2. If pages > PAGE_THRESHOLD:
   → Split into chunks of CHUNK_SIZE pages
   → Create separate task for each chunk
   → Return job_id to user
3. For each chunk:
   → Process OCR independently
   → Save intermediate results to Redis
4. When all chunks complete:
   → Combine text from all chunks
   → Run single analysis on combined text
   → Store final results

Main components for chunking
- Coordinator analyzes the document and creates the processing plan
- Workers process chunks in parallel, each handling a manageable portion
- Aggregator combines results and performs unified analysis
"""

def should_chunk_pdf(pdf_document):
    """Determine if a PDF needs to be processed in chunks based on page count"""
    return len(pdf_document) > PAGE_THRESHOLD


def preprocess_image(img):
    """
    Preprocess the image for better OCR results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get rid of the noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    return denoised

def clean_text(text):
    """
    Clean the extracted text for better regex matching
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters that might interfere with regex
    text = text.replace('\n', ' ').replace('\r', '')
    return text

def extract_nda_fields(text):
    """
    Extract relevant fields from NDA text using flexible regex patterns
    """
    logger.info("Extracting NDA fields from text")
    try:
        patterns = {
            # Match company name
            'company': r'between:\s+(.*?)(?=\s*:?\s*\("Discloser")',
            # Match recipient name
            'recipient': r'and\.\s+(.*?)(?=\s*:\s*\("Recipient")',
            # Match company address
            'company_address': r'(?:business\s*at\s*)(.*?)(?:;)',
            # Match recipient address
            'recipient_address': r'(?:residing\s*at\s*)(.*?)(?:\.)',
            # Match both initial duration and survival period
            'duration': r'period\s+of\s+(.*?)\s+years.*?additional\s+(.*?)\s+years',
            # Match governing law
            'governing_law': r'governed by and construed in accordance with the laws of the\.?\s*([^.]+?)(?:\.|$)',
            # Match confidential information
            'confidential_info': r'information\s+relating\s+to\s+(.*?)(?=\s*\(the "Confidential Information"\))',
            # Match dates
            'dates': r'\b(?:February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
        }

        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            if field == 'dates':
                matches = re.findall(pattern, text, re.IGNORECASE)
                fields[field] = list(set(matches)) if matches else []  # Remove duplicates
                logger.debug(f"Found dates: {fields[field]}")
            elif field == 'duration':
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and match.groups():
                    initial_term = match.group(1).strip()
                    survival_period = match.group(2).strip()
                    fields[field] = f"{initial_term} years with {survival_period} years survival period"
                    logger.debug(f"Found duration: {fields[field]}")
                else:
                    fields[field] = "Not found"
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                fields[field] = match.group(1).strip() if match and match.groups() else "Not found"
                # Clean up extra spaces and punctuation
                if fields[field] != "Not found":
                    fields[field] = re.sub(r'\s+', ' ', fields[field]).strip(' .,;:')
                    logger.debug(f"Found {field}: {fields[field]}")

        # Post-processing for governing law to ensure complete phrase
        if fields['governing_law'] != "Not found":
            fields['governing_law'] = "laws of the " + fields['governing_law']

        logger.info(f"Field extraction complete. Found {sum(1 for v in fields.values() if v != 'Not found' and (not isinstance(v, list) or len(v) > 0))} fields")
        flush_logs()
        return fields

    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}", exc_info=True)
        flush_logs()
        return {
            'company': "Not found",
            'recipient': "Not found",
            'company_address': "Not found",
            'recipient_address': "Not found",
            'duration': "Not found",
            'governing_law': "Not found",
            'confidential_info': "Not found",
            'dates': []
        }

def generate_summary(text, summarizer):
    """
    Generate a summary using the pre-initialized model.
    
    Args:
        text (str): Text to summarize
        summarizer: Pre-initialized summarization model
        
    Returns:
        str: Generated summary or error message
    """
    if summarizer is None:
        logger.warning("Summary generation skipped - model not available")
        flush_logs()
        return "Summary generation not available"
    
    try:
        # Limit input text to prevent model overload
        text = ' '.join(text.split()[:512])
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False, truncation=True)
        logger.info("Summary generated successfully")
        flush_logs()
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        flush_logs()
        return "Failed to generate summary"

# Coordinator Component
@celery_app.task(name="process_document")
def process_document_task(file_content, file_content_type, file_name, job_id=None):
    """
    Process a document and extract information, using chunking for large PDFs
    """
    logger.info(f"Processing document: {file_name} ({file_content_type})")
    flush_logs()
    
    try:
        # Use the provided job_id or generate a new one
        if job_id is None:
            job_id = str(uuid.uuid4())
            logger.info(f"Generated new job ID: {job_id}")
        else:
            logger.info(f"Using provided job ID: {job_id}")
            
        redis_client = Redis(host='redis', port=6379, db=0)
        
        # For PDF files, check if chunking is needed
        if file_content_type == "application/pdf":
            pdf_document = pymupdf.open(stream=file_content, filetype="pdf")
            total_pages = len(pdf_document)
            logger.info(f"PDF has {total_pages} pages")
            flush_logs()
            
            if should_chunk_pdf(pdf_document):
                # Process large PDF in chunks
                logger.info(f"PDF exceeds threshold of {PAGE_THRESHOLD} pages, processing in chunks")
                
                # Calculate the number of chunks needed
                total_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
                
                # Mark the job as chunked
                redis_client.set(f"status:{job_id}", "CHUNKED")
                redis_client.set(f"total_chunks:{job_id}", str(total_chunks))
                redis_client.set(f"completed_chunks:{job_id}", "0")
                
                # Create a task for each chunk
                for chunk_idx in range(total_chunks):
                    start_page = chunk_idx * CHUNK_SIZE
                    end_page = min(start_page + CHUNK_SIZE, total_pages)
                    
                    # Initialize the chunk status
                    redis_client.set(f"chunk_status:{job_id}:{chunk_idx}", "PENDING")
                    
                    # Create a task for this chunk
                    process_pdf_chunk.delay(
                        file_content=file_content,
                        start_page=start_page,
                        end_page=end_page,
                        job_id=job_id,
                        chunk_idx=chunk_idx,
                        total_chunks=total_chunks
                    )
                    
                    logger.info(f"Created task for chunk {chunk_idx+1}/{total_chunks} (pages {start_page+1}-{end_page})")
                
                pdf_document.close()
                # Clean up file_content which can be very large
                clean_variables(file_content, pdf_document)
                # Force garbage collection after creating all chunk tasks
                cleanup_memory("after_creating_chunks")
                flush_logs()
                
                # Return the job ID for tracking
                return {
                    "job_id": job_id,
                    "status": "CHUNKED",
                    "total_chunks": total_chunks
                }
            
            # For smaller PDFs, close and continue with normal processing
            pdf_document.close()
        
        # For non-PDF files or small PDFs, process normally
        logger.info(f"Processing document without chunking: {file_name}")
        redis_client.set(f"status:{job_id}", "PROCESSING")
        flush_logs()
        
        # Start regular processing
        if file_content_type == "application/pdf":
            logger.info("Processing PDF file")
            
            # Use PyMuPDF to process PDF
            pdf_document = pymupdf.open(stream=file_content, filetype="pdf")
            logger.info(f"PDF loaded with {len(pdf_document)} pages")
            
            full_text = ""
            confidence_scores = []
            
            for page_num in range(len(pdf_document)):
                logger.info(f"Processing PDF page {page_num+1}/{len(pdf_document)}")
                
                page = pdf_document.load_page(page_num)
                
                # Render page to an image
                pix = page.get_pixmap(alpha=False, dpi=150)  # Lower DPI for memory savings
                img_data = pix.tobytes("png")
                
                # Convert to numpy array for OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.warning(f"Failed to convert page {page_num+1} to image")
                    flush_logs()
                    continue
                    
                # Preprocess image
                processed_img = preprocess_image(img)

                # Process the image using PaddleOCR
                result = ocr.ocr(processed_img, cls=True)
                logger.info(f"OCR completed for page {page_num+1}")

                # Concatenate the OCR results into a full text string
                page_text = ""
                for res in result:
                    for line in res:
                        page_text += line[1][0] + " "
                        confidence_scores.append(line[1][1])
                
                logger.info(f"Extracted {len(page_text.split())} words from page {page_num+1}")
                full_text += page_text
                
                # Clean up large objects after each page to prevent memory buildup
                clean_variables(pix, img_data, nparr, img, processed_img, result, page)
                
                flush_logs()
            
            # Close the document
            pdf_document.close()
            clean_variables(pdf_document)
            
            cleaned_text = clean_text(full_text)
            if confidence_scores:
                average_confidence = sum(confidence_scores) / len(confidence_scores)
                average_confidence_formatted = f"{average_confidence:.2%}"
                logger.info(f"Average OCR confidence: {average_confidence_formatted}")
            else:
                average_confidence_formatted = "N/A"
                logger.warning("No confidence scores available")
            
        else:
            logger.info(f"Processing image file: {file_content_type}")
            
            # Convert image to OpenCV format
            nparr = np.frombuffer(file_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                error_message = "Image decode error: Could not decode the image."
                logger.error(error_message)
                flush_logs()
                raise ValueError(error_message)

            # Process single image
            processed_img = preprocess_image(img)
            result = ocr.ocr(processed_img, cls=True)
            logger.info("OCR completed for image")

            # Extract text and confidence scores
            full_text = ""
            confidence_scores = []
            for res in result:
                for line in res:
                    full_text += line[1][0] + " "
                    confidence_scores.append(line[1][1])
            
            logger.info(f"Extracted {len(full_text.split())} words from image")
            cleaned_text = clean_text(full_text)
            
            if confidence_scores:
                average_confidence = sum(confidence_scores) / len(confidence_scores)
                average_confidence_formatted = f"{average_confidence:.2%}"
                logger.info(f"Average OCR confidence: {average_confidence_formatted}")
            else:
                average_confidence_formatted = "N/A"
                logger.warning("No confidence scores available")
            
            # Clean up large image data
            clean_variables(nparr, img, processed_img, result)
        
        # We don't need the original file content anymore, so clean it up
        clean_variables(file_content)
        flush_logs()

        # Extract NDA fields
        logger.info("Extracting NDA fields from text")
        fields = extract_nda_fields(cleaned_text)
        logger.info(f"Extracted fields: {', '.join(fields.keys())}")
        flush_logs()

        # Generate summary
        logger.info("Generating text summary")
        summary = generate_summary(cleaned_text, summarizer)
        logger.info("Text summary generated")
        flush_logs()

        # Prepare result
        result = {
            "full_text": cleaned_text,
            "average_confidence_formatted": average_confidence_formatted,
            "summary": summary,
            **fields
        }

        # Store the result in Redis
        redis_client.set(f"result:{job_id}", pickle.dumps(result))
        redis_client.set(f"status:{job_id}", "SUCCESS")
        
        logger.info(f"Document {file_name} processed successfully")
        
        # Final cleanup
        clean_variables(full_text, cleaned_text, confidence_scores, fields, summary, result)
        cleanup_memory("after_document_processing")
        
        flush_logs()
        
        return {
            "job_id": job_id,
            "status": "SUCCESS"
        }

    except Exception as e:
        logger.error(f"Error processing document {file_name}: {str(e)}", exc_info=True)
        flush_logs()
        
        try:
            # Mark the job as failed
            redis_client = Redis(host='redis', port=6379, db=0)
            redis_client.set(f"status:{job_id}", "FAILED")
            redis_client.set(f"error:{job_id}", str(e))
        except Exception as redis_error:
            logger.error(f"Error updating Redis status: {str(redis_error)}")
        
        # Always clean up on error
        try:
            clean_variables(file_content)
            cleanup_memory("error_cleanup")
        except:
            pass
            
        raise


# Worker Component
@celery_app.task(name="process_pdf_chunk")
def process_pdf_chunk(file_content, start_page, end_page, job_id, chunk_idx, total_chunks):
    """Process a specific chunk of pages from a PDF document"""
    logger.info(f"Processing chunk {chunk_idx+1}/{total_chunks} (pages {start_page+1}-{end_page})")
    flush_logs()
    
    try:
        # Load the PDF
        pdf_document = pymupdf.open(stream=file_content, filetype="pdf")
        
        # Process only the specified range of pages
        full_text = ""
        confidence_scores = []
        
        for page_num in range(start_page, end_page):
            logger.info(f"Processing page {page_num+1} in chunk {chunk_idx+1}")
            flush_logs()
            
            page = pdf_document.load_page(page_num)
            
            # Render page to an image
            pix = page.get_pixmap(alpha=False, dpi=150)  # Lower DPI for memory savings
            img_data = pix.tobytes("png")
            
            # Convert to numpy array for OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning(f"Failed to convert page {page_num+1} to image")
                flush_logs()
                continue
                
            # Preprocess image
            processed_img = preprocess_image(img)

            # Process the image using PaddleOCR
            result = ocr.ocr(processed_img, cls=True)
            
            # Extract text from the page
            page_text = ""
            for res in result:
                for line in res:
                    page_text += line[1][0] + " "
                    confidence_scores.append(line[1][1])
            
            full_text += page_text
            
            # Clean up memory after processing each page
            clean_variables(pix, img_data, nparr, img, processed_img, result, page)
            gc.collect()
        
        # Clean the extracted text
        cleaned_text = clean_text(full_text)
        average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Store the chunk result in Redis
        redis_client = Redis(host='redis', port=6379, db=0)
        chunk_result = {
            "chunk_idx": chunk_idx,
            "text": cleaned_text,
            "confidence": average_confidence
        }
        redis_client.set(f"chunk:{job_id}:{chunk_idx}", pickle.dumps(chunk_result))
        redis_client.set(f"chunk_status:{job_id}:{chunk_idx}", "COMPLETED")
        
        # Check if all chunks are completed
        completed_chunks = 0
        for i in range(total_chunks):
            status = redis_client.get(f"chunk_status:{job_id}:{i}")
            if status and status.decode('utf-8') == "COMPLETED":
                completed_chunks += 1
        
        # Update the completed chunks counter
        redis_client.set(f"completed_chunks:{job_id}", str(completed_chunks))
        
        logger.info(f"Chunk {chunk_idx+1}/{total_chunks} completed. Progress: {completed_chunks}/{total_chunks}")
        flush_logs()
        
        # If all chunks are completed, combine results
        if completed_chunks == total_chunks:
            combine_chunks.delay(job_id=job_id, total_chunks=total_chunks)
        
        # Clean up before returning
        pdf_document.close()
        clean_variables(pdf_document, file_content, full_text, cleaned_text, confidence_scores, chunk_result)
        cleanup_memory(f"after_chunk_{chunk_idx}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_idx+1}: {str(e)}", exc_info=True)
        flush_logs()
        
        # Mark this chunk as failed
        try:
            redis_client = Redis(host='redis', port=6379, db=0)
            redis_client.set(f"chunk_status:{job_id}:{chunk_idx}", "FAILED")
        except:
            pass
        
        # Always clean up on error
        try:
            pdf_document.close()
            clean_variables(pdf_document, file_content)
            cleanup_memory("chunk_error_cleanup")
        except:
            pass
            
        raise


# Aggregator Component
@celery_app.task(name="combine_chunks")
def combine_chunks(job_id, total_chunks):
    """Combine results from all PDF chunks into a final result"""
    logger.info(f"Combining results from {total_chunks} chunks for job {job_id}")
    flush_logs()
    
    try:
        redis_client = Redis(host='redis', port=6379, db=0)
        
        # Collect all chunk texts and confidence scores
        all_text = ""
        all_confidence_scores = []
        
        for i in range(total_chunks):
            chunk_data = redis_client.get(f"chunk:{job_id}:{i}")
            if not chunk_data:
                logger.error(f"Missing data for chunk {i+1}")
                continue
                
            chunk_result = pickle.loads(chunk_data)
            all_text += chunk_result["text"] + " "
            all_confidence_scores.append(chunk_result["confidence"])
            
            # Clean up chunk data from memory after adding to all_text
            clean_variables(chunk_result, chunk_data)
            if i % 5 == 0:  # Run garbage collection every 5 chunks to avoid too frequent calls
                gc.collect()
        
        # Clean up the combined text
        all_text = clean_text(all_text)
        
        # Calculate average confidence
        average_confidence = sum(all_confidence_scores) / len(all_confidence_scores) if all_confidence_scores else 0
        
        # Extract fields and generate summary
        fields = extract_nda_fields(all_text)
        summary = generate_summary(all_text, summarizer)
        result = {
            "full_text": all_text,
            "average_confidence_formatted": f"{average_confidence:.2%}",
            "summary": summary,
            **fields
        }
        
        # Store the final result
        redis_client.set(f"result:{job_id}", pickle.dumps(result))
        redis_client.set(f"status:{job_id}", "SUCCESS")
        
        logger.info(f"Successfully combined all chunks for job {job_id}")
        flush_logs()
        
        # Clean up chunk data
        for i in range(total_chunks):
            redis_client.delete(f"chunk:{job_id}:{i}")
            redis_client.delete(f"chunk_status:{job_id}:{i}")
        
        # Final cleanup
        clean_variables(all_text, all_confidence_scores, fields, summary, result)
        cleanup_memory("after_combining_chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"Error combining chunks: {str(e)}", exc_info=True)
        flush_logs()
        
        # Mark the job as failed
        try:
            redis_client = Redis(host='redis', port=6379, db=0)
            redis_client.set(f"status:{job_id}", "FAILED")
            redis_client.set(f"error:{job_id}", str(e))
        except:
            pass

        # Clean up memory even in case of error
        cleanup_memory("combine_chunks_error")
        raise