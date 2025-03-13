"""
Utility functions for tasks and API endpoints.
Contains commonly used functions like status updates and timestamp generation.
"""
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_timestamp():
    """
    Get current ISO format timestamp.
    
    Returns:
        str: Current timestamp in ISO format
    """
    return datetime.utcnow().isoformat()

def update_job_status(redis_client, job_id, status_update):
    """
    Update job status in Redis.
    
    Args:
        redis_client: Redis client instance
        job_id: Unique job identifier
        status_update: Dict with status updates
    """
    try:
        # Get current status if it exists
        current_status = redis_client.get(f"job:{job_id}")
        
        if current_status:
            current_status = json.loads(current_status)
            # Update with new values
            current_status.update(status_update)
        else:
            current_status = status_update
        
        # Store updated status
        redis_client.set(f"job:{job_id}", json.dumps(current_status))
        
        # Set expiration (7 days)
        redis_client.expire(f"job:{job_id}", 60 * 60 * 24 * 7)
        
    except Exception as e:
        logger.error(f"Error updating job status in Redis: {str(e)}")

def preprocess_image(img):
    """
    Preprocess an image for better OCR results.
    
    Args:
        img: OpenCV image
        
    Returns:
        processed_img: Preprocessed image
    """
    import cv2
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get rid of the noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    return denoised

def clean_text(text):
    """
    Clean extracted text for better regex matching.
    
    Args:
        text: Raw text from OCR
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with regex
    text = text.replace('\n', ' ').replace('\r', '')
    
    return text

def extract_nda_fields(text):
    """
    Extract relevant fields from NDA text using flexible regex patterns.
    
    Args:
        text: Document text
        
    Returns:
        dict: Extracted fields
    """
    import re
    
    try:
        patterns = {
            # Match company name - fixed to remove trailing colon
            'company': r'between:\s+(.*?)(?=\s*:?\s*\("Discloser")',

            # Match recipient name - improved to get just the name
            'recipient': r'and\.\s+(.*?)(?=\s*:\s*\("Recipient")',

            # Match company address - unchanged as it works correctly
            'company_address': r'(?:business\s*at\s*)(.*?)(?:;)',

            # Match recipient address - unchanged as it works correctly
            'recipient_address': r'(?:residing\s*at\s*)(.*?)(?:\.)',

            # Match both initial duration and survival period
            'duration': r'period\s+of\s+(.*?)\s+years.*?additional\s+(.*?)\s+years',

            # Match governing law - fixed to capture full state law reference
            'governing_law': r'governed by and construed in accordance with the laws of the\.?\s*([^.]+?)(?:\.|$)',

            # Match confidential information - improved to capture full scope
            'confidential_info': r'information\s+relating\s+to\s+(.*?)(?=\s*\(the "Confidential Information"\))',

            # Match dates - improved format handling
            'dates': r'\b(?:February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
        }

        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            if field == 'dates':
                matches = re.findall(pattern, text, re.IGNORECASE)
                fields[field] = list(set(matches)) if matches else []  # Remove duplicates
            elif field == 'duration':
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and match.groups():
                    initial_term = match.group(1).strip()
                    survival_period = match.group(2).strip()
                    fields[field] = f"{initial_term} years with {survival_period} years survival period"
                else:
                    fields[field] = "Not found"
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                fields[field] = match.group(1).strip() if match and match.groups() else "Not found"
                # Clean up extra spaces and punctuation
                if fields[field] != "Not found":
                    fields[field] = re.sub(r'\s+', ' ', fields[field]).strip(' .,;:')  # Added ':' to strip

        # Post-processing for governing law to ensure complete phrase
        if fields['governing_law'] != "Not found":
            fields['governing_law'] = "laws of the " + fields['governing_law']

        return fields

    except Exception as e:
        logger.error(f"Error in pattern matching: {str(e)}")
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

def generate_summary(text):
    """
    Generate a summary of the input text using the transformers library.
    
    Args:
        text: Document text
        
    Returns:
        str: Summary text
    """
    from transformers import pipeline
    
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # Check if text is too short for summarization
        if len(text.split()) < 30:
            return "Text too short for meaningful summarization."
            
        # Limit input text to 1024 tokens to prevent model overload
        text = ' '.join(text.split()[:1024])
        
        # Generate summary
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False, truncation=True)
        
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary."