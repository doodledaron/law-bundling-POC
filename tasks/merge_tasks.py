"""
Merge and summarize task for combining chunk processing results.
Extracted from ppstructure_tasks.py for better modularity.
"""
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import json
import datetime
import redis

from tasks.utils import get_unix_timestamp, calculate_duration, update_job_status, get_timestamp, update_merge_progress
from text_based_processor import TextBasedProcessor

logger = get_task_logger(__name__)

# Initialize TextBasedProcessor for Gemini integration
text_processor = TextBasedProcessor()


@shared_task(bind=True, name='tasks.merge_and_summarize_chunks', 
              autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 30})
def merge_and_summarize_chunks(self, chunk_results, job_id):
    """
    Merge chunk processing results and generate final summary.
    
    This task is called by Celery chord after all chunk processing tasks complete.
    It combines the text from all chunks and generates a final document summary.
    
    Args:
        chunk_results: List of results from individual chunk processing tasks (passed automatically by chord)
        job_id: Unique job identifier (passed as argument)
        
    Returns:
        dict: Final processing results with merged content and summary
    """
    try:
        logger.info(f"🔄 Starting merge and summarize for job {job_id}")
        
        # Initialize Redis client for status updates
        redis_client = redis.Redis.from_url(
            os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        # Update progress to 85% - merge task started
        update_merge_progress(redis_client, job_id, "Combining chunk results", 85)
        
        # Log the number of chunk results received
        logger.info(f"📦 Received {len(chunk_results)} chunk results for job {job_id}")
        
        # Validate chunk results
        if not chunk_results:
            raise ValueError("No chunk results provided for merging")
        
        # Separate successful and failed chunks, and track processing methods
        successful_chunks = []
        failed_chunks = []
        ppstructure_chunks = []
        gemini_chunks = []
        
        for i, chunk_result in enumerate(chunk_results):
            if chunk_result and isinstance(chunk_result, dict):
                if chunk_result.get('status') == 'CHUNK_FAILED':
                    failed_chunks.append(chunk_result)
                    logger.warning(f"   ❌ Chunk {i}: FAILED - {chunk_result.get('error', 'Unknown error')}")
                else:
                    successful_chunks.append((i, chunk_result))
                    processing_method = chunk_result.get('processing_method', 'unknown')
                    
                    # Track processing methods (more robust matching)
                    if processing_method == 'ppstructure' or 'ppstructure' in processing_method.lower():
                        ppstructure_chunks.append(chunk_result)
                    elif processing_method in ['gemini_only', 'gemini', 'gemini_only_bulk']:
                        gemini_chunks.append(chunk_result)
                    else:
                        # Log unknown processing methods for debugging
                        logger.warning(f"   ⚠️  Unknown processing method: '{processing_method}' for chunk {i}")
                    
                    logger.info(f"   ✅ Chunk {i}: SUCCESS - {chunk_result.get('filename', 'unknown')} ({processing_method})")
            else:
                failed_chunks.append({"error": "Invalid chunk result", "chunk_index": i})
                logger.warning(f"   ⚠️ Chunk {i}: invalid result format")
        
        # Check if we have any successful chunks
        if not successful_chunks:
            raise ValueError("All chunks failed - no successful processing to merge")
        
        if failed_chunks:
            logger.warning(f"⚠️ {len(failed_chunks)} chunks failed, proceeding with {len(successful_chunks)} successful chunks")
        
        # Log mixed processing statistics
        logger.info(f"📊 Mixed processing summary:")
        logger.info(f"   🔧 PPStructure chunks: {len(ppstructure_chunks)}")
        logger.info(f"   🤖 Gemini-only chunks: {len(gemini_chunks)}")
        logger.info(f"   ✅ Total successful: {len(successful_chunks)}")
        logger.info(f"   ❌ Total failed: {len(failed_chunks)}")
        
        # Extract original document filename from the first successful chunk
        original_filename = "unknown_document"
        for _, chunk_result in successful_chunks:
            chunk_filename = chunk_result.get('filename', '')
            if chunk_filename.startswith('chunk_'):
                # Extract original filename: "chunk_0000_original.pdf" -> "original.pdf"
                parts = chunk_filename.split('_', 2)  # Split on first 2 underscores
                if len(parts) >= 3:
                    original_filename = parts[2]  # Everything after "chunk_XXXX_"
                    break
            else:
                original_filename = chunk_filename
                break
        
        logger.info(f"📄 Original document: {original_filename}")
        
        # CRITICAL: Sort chunks by chunk ID to ensure correct page order
        chunk_data = []
        for original_index, chunk_result in successful_chunks:
            filename = chunk_result.get('filename', '')
            # Extract chunk ID from filename (e.g., "chunk_0000" from "chunk_0000_file.pdf")
            chunk_id = "9999"  # Default high value if parsing fails
            if filename.startswith('chunk_'):
                try:
                    chunk_parts = filename.split('_')
                    if len(chunk_parts) >= 2:
                        chunk_id = chunk_parts[1]  # e.g., "0000" from "chunk_0000"
                except Exception as e:
                    logger.warning(f"Failed to parse chunk ID from {filename}: {e}")
            
            chunk_data.append({
                'chunk_id': chunk_id,
                'original_index': original_index,
                'result': chunk_result,
                'filename': filename
            })
            logger.info(f"   📦 Chunk {original_index}: ID={chunk_id}, filename={filename}")
        
        # Sort chunks by chunk ID to ensure correct page order
        chunk_data.sort(key=lambda x: x['chunk_id'])
        chunk_order_info = [f"{c['chunk_id']}({c['original_index']})" for c in chunk_data]
        logger.info(f"📋 Sorted chunks by ID: {chunk_order_info}")
        
        # Extract and combine text from all chunks in correct order
        all_combined_text = []
        total_pages = 0
        all_extracted_info = []
        total_processing_time = 0
        total_cost = 0.0
        
        for chunk_info in chunk_data:
            chunk_result = chunk_info['result']
            chunk_id = chunk_info['chunk_id']
            original_index = chunk_info['original_index']
            
            # Extract combined text from chunk
            chunk_text = chunk_result.get('combined_text', '') or chunk_result.get('extracted_text', '')
            if chunk_text:
                all_combined_text.append(chunk_text)
                logger.info(f"   ✅ Chunk {chunk_id} (orig index {original_index}): {len(chunk_text)} characters extracted")
            else:
                logger.warning(f"   ⚠️ Chunk {chunk_id} (orig index {original_index}): no text content found")
            
            # Accumulate metadata
            total_pages += chunk_result.get('total_pages', 0)
            if chunk_result.get('processing_time', {}).get('seconds'):
                total_processing_time += chunk_result.get('processing_time', {}).get('seconds', 0)
            if chunk_result.get('cost_info', {}).get('estimated_cost'):
                total_cost += chunk_result.get('cost_info', {}).get('estimated_cost', 0.0)
            
            # Collect extracted info (though chunks don't generate summaries)
            chunk_info_data = chunk_result.get('extracted_info', {})
            if chunk_info_data:
                all_extracted_info.append(chunk_info_data)
        
        # Combine all text into one document body IN CORRECT PAGE ORDER
        combined_text = '\n\n'.join(all_combined_text)
        logger.info(f"📄 Combined text length: {len(combined_text)} characters from {len(all_combined_text)} chunks")
        logger.info(f"📋 Chunks merged in correct page order: {[c['chunk_id'] for c in chunk_data]}")
        
        if not combined_text.strip():
            raise ValueError("No text content could be extracted from any chunks")
        
        # Update progress to 90% - text combination complete
        update_merge_progress(redis_client, job_id, "Generating document summary", 90)
        
        # Generate final summary using text_processor
        logger.info(f"🧠 Generating final summary for job {job_id}")
        
        # Generate summary using TextBasedProcessor
        summary_result = text_processor.summarize_document_text(combined_text, original_filename)
        
        # Log completion
        final_summary = summary_result.get('summary', 'Summary not available')
        logger.info(f"🧠 Final summary generated for job {job_id}, total length: {len(final_summary)}")
        
        # Update progress to 95% - summary generation complete
        update_merge_progress(redis_client, job_id, "Saving final results", 95)
        
        # Prepare final results directories
        result_dir = os.path.join("results", job_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save combined text to file
        combined_text_path = os.path.join(result_dir, "combined_text.txt")
        with open(combined_text_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Calculate average confidence (if available from chunks)
        average_confidence = 0.0
        confidence_count = 0
        for chunk_info in chunk_data:
            chunk_result = chunk_info['result']
            chunk_conf = chunk_result.get('average_confidence_formatted', '')
            if chunk_conf and chunk_conf != 'N/A':
                try:
                    # Extract percentage value
                    conf_value = float(chunk_conf.strip('%')) / 100.0
                    average_confidence += conf_value
                    confidence_count += 1
                except ValueError:
                    pass
        
        if confidence_count > 0:
            average_confidence = average_confidence / confidence_count
            average_confidence_formatted = f"{average_confidence:.1%}"
        else:
            average_confidence_formatted = "N/A"
        
        # Prepare clean results with ORIGINAL document filename
        clean_results = {
            "filename": original_filename,  # Use original filename, not chunk filename
            "job_id": job_id,
            "processing_completed_at": datetime.datetime.now().isoformat(),
            "total_pages": total_pages,
            "summary": final_summary,
            "date": summary_result.get("date", "undated"),
            "extracted_info": summary_result.get("extracted_info", {}),
            "combined_text_path": f"results/{job_id}/combined_text.txt",
            "combined_text": combined_text,
            "estimated_cost": summary_result.get("estimated_cost", total_cost),
            "token_usage": summary_result.get("token_usage", {}),
            "processing_time_seconds": total_processing_time,
            "average_confidence_formatted": average_confidence_formatted,
            "processing_method": "mixed_processing_chunked",
            "num_chunks_processed": len(chunk_results),
            "num_chunks_successful": len(successful_chunks),
            "num_chunks_failed": len(failed_chunks),
            "num_ppstructure_chunks": len(ppstructure_chunks),
            "num_gemini_chunks": len(gemini_chunks),
            "mixed_processing_ratio": f"{len(ppstructure_chunks)}:{len(gemini_chunks)} (PPStructure:Gemini)",
            "chunk_order": [c['chunk_id'] for c in chunk_data]  # Record the order used
        }
        
        # Save results to files with atomic write to prevent race conditions
        results_path = os.path.join(result_dir, "results.json")
        temp_results_path = results_path + '.tmp'
        
        # Write to temporary file first, then rename (atomic operation)
        with open(temp_results_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        # Atomic rename - this prevents API from reading partial files
        os.rename(temp_results_path, results_path)
        
        # Save metrics with atomic write
        metrics_path = os.path.join(result_dir, "metrics.json")
        temp_metrics_path = metrics_path + '.tmp'
        performance_metrics = {
            "job_id": job_id,
            "filename": original_filename,  # Use original filename
            "processing_end_time": get_unix_timestamp(),
            "total_processing_time": {"seconds": total_processing_time},
            "performance": {
                "total_pages": total_pages,
                "processing_time": {"seconds": total_processing_time},
                "mixed_processing": True,
                "num_chunks": len(chunk_results),
                "num_chunks_successful": len(successful_chunks),
                "num_chunks_failed": len(failed_chunks),
                "num_ppstructure_chunks": len(ppstructure_chunks),
                "num_gemini_chunks": len(gemini_chunks),
                "mixed_processing_ratio": f"{len(ppstructure_chunks)}:{len(gemini_chunks)} (PPStructure:Gemini)",
                "chunk_order": [c['chunk_id'] for c in chunk_data]
            },
            "confidence_metrics": {
                "average_confidence": average_confidence,
                "average_confidence_formatted": average_confidence_formatted
            },
            "cost_info": {
                "estimated_cost": summary_result.get("estimated_cost", total_cost),
                "token_usage": summary_result.get("token_usage", {})
            }
        }
        
        # Atomic write for metrics
        with open(temp_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=2)
        
        # Atomic rename for metrics
        os.rename(temp_metrics_path, metrics_path)
        
        # Update progress to 100% - all processing complete
        update_merge_progress(redis_client, job_id, "Processing complete", 100)
        
        # CRITICAL: Only NOW mark the job as COMPLETED (all chunks processed and merged)
        update_job_status(redis_client, job_id, {
            'status': 'COMPLETED',
            'message': f'Document "{original_filename}" processed successfully with {total_pages} pages using hybrid processing ({len(ppstructure_chunks)} OCR + {len(gemini_chunks)} AI)',
            'progress': 100,
            'results_path': results_path,
            'combined_text_path': combined_text_path,
            'total_pages': total_pages,
            'processing_completed_at': datetime.datetime.now().isoformat(),
            'average_confidence_formatted': average_confidence_formatted,
            'estimated_cost': summary_result.get("estimated_cost", total_cost),
            'processing_time_seconds': total_processing_time,
            'processing_method': 'mixed_processing_chunked',
            'num_chunks_processed': len(chunk_results),
            'num_chunks_successful': len(successful_chunks),
            'num_chunks_failed': len(failed_chunks),
            'num_ppstructure_chunks': len(ppstructure_chunks),
            'num_gemini_chunks': len(gemini_chunks),
            'mixed_processing_ratio': f"{len(ppstructure_chunks)}:{len(gemini_chunks)} (PPStructure:Gemini)",
            'filename': original_filename,  # Store original filename
            'updated_at': get_timestamp()
        })
        
        logger.info(f"✅ Merge and summarize completed for job {job_id}")
        logger.info(f"📄 Final document: {original_filename} ({total_pages} pages)")
        logger.info(f"📊 Mixed results: {len(ppstructure_chunks)} PPStructure + {len(gemini_chunks)} Gemini chunks")
        
        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "filename": original_filename,  # Return original filename
            "results_path": results_path,
            "message": f'Document "{original_filename}" processed successfully with {total_pages} pages using hybrid processing ({len(ppstructure_chunks)} OCR + {len(gemini_chunks)} AI)',
            "total_pages": total_pages,
            "num_chunks_processed": len(chunk_results),
            "num_chunks_successful": len(successful_chunks),
            "num_chunks_failed": len(failed_chunks),
            "num_ppstructure_chunks": len(ppstructure_chunks),
            "num_gemini_chunks": len(gemini_chunks),
            "mixed_processing_ratio": f"{len(ppstructure_chunks)}:{len(gemini_chunks)} (PPStructure:Gemini)",
            "processing_method": "mixed_processing_chunked",
            "summary": final_summary,
            "estimated_cost": summary_result.get("estimated_cost", total_cost)
        }
        
    except Exception as e:
        logger.error(f"Error in merge and summarize: {str(e)}")
        
        # CRITICAL: Only mark as FAILED here in the merge task
        update_job_status(redis_client, job_id, {
            'status': 'FAILED',
            'error': str(e),
            'message': f'Document merge and summarize failed: {str(e)}',
            'progress': 0,
            'updated_at': get_timestamp()
        })
        
        raise 
