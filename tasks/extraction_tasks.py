"""
Entity extraction tasks for legal documents.
Handles NER and information extraction from processed text.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import redis

# Import utilities
from tasks.utils import (
    update_job_status,
    get_timestamp,
    extract_nda_fields,
    generate_summary
)

logger = get_task_logger(__name__)

# Initialize Redis client
redis_client = redis.Redis.from_url(
    os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

@shared_task(name='tasks.extraction.extract_document_info')
def extract_document_info(job_id, text, ocr_results=None):
    """
    Extract document information using NER and summarization.
    
    Args:
        job_id: Unique job identifier
        text: Document text
        ocr_results: OCR processing results (optional)
        
    Returns:
        dict: Extraction results
    """
    try:
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': 'Extracting document information',
            'progress': 75,
            'updated_at': get_timestamp()
        })
        
        # Extract fields from text
        fields = extract_nda_fields(text)
        
        # Generate summary
        summary = generate_summary(text)
        
        # Combine results
        results = {
            'entities': fields,
            'summary': summary
        }
        
        # Add OCR results if provided
        if ocr_results:
            results.update(ocr_results)
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'progress': 90,
            'message': 'Information extraction complete',
            'updated_at': get_timestamp()
        })
        
        return results
    
    except Exception as e:
        logger.error(f"Error extracting document information: {str(e)}")
        raise

@shared_task(name='tasks.extraction.merge_extraction_results')
def merge_extraction_results(job_id, chunk_results):
    """
    Merge extraction results from multiple document chunks.
    
    Args:
        job_id: Unique job identifier
        chunk_results: List of results from chunk processing
        
    Returns:
        dict: Merged results
    """
    try:
        # Update job status
        update_job_status(redis_client, job_id, {
            'status': 'PROCESSING',
            'message': 'Merging extraction results',
            'progress': 85,
            'updated_at': get_timestamp()
        })
        
        # Initialize merged results
        merged_text = ""
        confidence_scores = []
        entities = {
            'company': "Not found",
            'recipient': "Not found",
            'company_address': "Not found",
            'recipient_address': "Not found",
            'duration': "Not found",
            'governing_law': "Not found",
            'confidential_info': "Not found",
            'dates': []
        }
        
        # Merge chunk data
        for result in chunk_results:
            # Merge text
            if 'cleaned_text' in result:
                merged_text += result['cleaned_text'] + " "
            
            # Merge confidence scores
            if 'average_confidence' in result:
                confidence_scores.append(result['average_confidence'])
            
            # Merge entities with precedence for non-empty values
            if 'entities' in result:
                for key, value in result['entities'].items():
                    # For dates, combine lists
                    if key == 'dates':
                        entities[key].extend(value)
                        # Remove duplicates
                        entities[key] = list(set(entities[key]))
                    # For other fields, take non-empty values with precedence
                    elif value != "Not found" and entities[key] == "Not found":
                        entities[key] = value
        
        # Clean merged text
        merged_text = ' '.join(merged_text.split())
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Generate summary for complete document
        summary = generate_summary(merged_text)
        
        # Compile final results
        final_results = {
            'job_id': job_id,
            'extracted_text': merged_text,
            'entities': entities,
            'summary': summary,
            'average_confidence': avg_confidence,
            'average_confidence_formatted': f"{avg_confidence:.2%}"
        }
        
        # Update job status
        update_job_status(redis_client, job_id, {
            'progress': 95,
            'message': 'Results merging complete',
            'updated_at': get_timestamp()
        })
        
        return final_results
    
    except Exception as e:
        logger.error(f"Error merging extraction results: {str(e)}")
        raise