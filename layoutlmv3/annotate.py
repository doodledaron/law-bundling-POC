import json
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
try:
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR with parameters optimized for document OCR
    ocr = PaddleOCR(
        lang='en',
        use_angle_cls=True,
        det_model_dir=None,  # Use default detection model
        rec_model_dir=None,  # Use default recognition model
        cls_model_dir=None,  # Use default classification model
        det_limit_side_len=2560,  # Increase from default for higher quality
        det_db_thresh=0.3,  # Lower threshold to detect more text regions
        det_db_box_thresh=0.5,  # Lower box threshold
        rec_batch_num=6,  # Increase batch size for faster processing
        rec_char_dict_path=None,  # Use default dictionary
        use_space_char=True,  # Important for document text
        show_log=False
    )
    OCR_AVAILABLE = True
except ImportError:
    print("PaddleOCR not available. Please install it with 'pip install paddleocr'")
    OCR_AVAILABLE = False
    ocr = None

from pdf2image import convert_from_path
import difflib
from PIL import Image
from tqdm import tqdm

# Configure logging with both file and console handlers
def setup_logging(output_dir: str):
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'processing_{timestamp}.log')
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def pdf_to_ocr_words(pdf_path: str, dpi: int = 400) -> List[List[Dict[str, Any]]]:
    """
    Convert PDF pages to images and run OCR on each image with improved parameters.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for PDF to image conversion (increased from 300 to 400)
        
    Returns:
        List of pages, where each page contains a list of word dictionaries with 'text' and 'bbox'
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info(f"Starting OCR processing for {pdf_path}")
    
    # Check if OCR is available
    if not OCR_AVAILABLE:
        logger.error("PaddleOCR is not available. Please install it with 'pip install paddleocr'")
        return []
    
    try:
        # Use higher DPI for better quality images
        # Add thread_count for faster processing on multi-core systems
        pages = convert_from_path(
            pdf_path, 
            dpi=dpi,
            thread_count=4,
            use_pdftocairo=True,  # Often provides better quality than pdftoppm
            grayscale=False       # Keep color for better OCR in some cases
        )
        logger.info(f"Successfully converted PDF to {len(pages)} pages")
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
        return []
        
    all_pages_ocr = []
    total_words = 0
    
    for page_num, image in enumerate(tqdm(pages, desc="Processing pages")):
        page_start_time = time.time()
        try:
            image_np = np.array(image)
            result = ocr.ocr(image_np, cls=True)
            
            page_words = []

            # Handle PaddleOCR result format - more robust handling for different versions
            try:
                # For newer PaddleOCR versions (>=2.0) - result structure is [[page_result]]
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                    ocr_results = result[0]
                else:
                    # For older versions - result structure is [line_result]
                    ocr_results = result

                # Process OCR results regardless of format
                if ocr_results:
                    for line_result in ocr_results:
                        try:
                            # Extract bbox and text+confidence based on structure
                            if isinstance(line_result, list) and len(line_result) >= 2:
                                bbox = line_result[0]
                                text_conf = line_result[1]
                                
                                # Handle text & confidence extraction 
                                if isinstance(text_conf, tuple) and len(text_conf) == 2:
                                    text, conf = text_conf
                                else:
                                    # If not a tuple, assume it's just text
                                    text = str(text_conf)
                                    conf = 1.0
                                
                                # Normalize bbox to [x0, y0, x1, y1]
                                if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                                    x_coords = [p[0] for p in bbox if isinstance(p, (list, tuple))]
                                    y_coords = [p[1] for p in bbox if isinstance(p, (list, tuple))]
                                    
                                    if x_coords and y_coords:
                                        x0, y0 = min(x_coords), min(y_coords)
                                        x1, y1 = max(x_coords), max(y_coords)
                                        
                                        page_words.append({
                                            'text': text,
                                            'bbox': [x0, y0, x1, y1],
                                            'confidence': conf,
                                            'page': page_num
                                        })
                        except Exception as e:
                            logger.warning(f"Error processing OCR item on page {page_num}: {str(e)}")
                            continue
            except Exception as e:
                logger.warning(f"Error processing OCR result structure on page {page_num}: {str(e)}")
            
            total_words += len(page_words)
            all_pages_ocr.append(page_words)
            
            page_time = time.time() - page_start_time
            logger.debug(f"Page {page_num + 1}: Processed {len(page_words)} words in {page_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing page {page_num} of {pdf_path}: {str(e)}")
            all_pages_ocr.append([])
    
    total_time = time.time() - start_time
    logger.info(f"Completed OCR processing for {pdf_path}")
    logger.info(f"Total words extracted: {total_words}")
    logger.info(f"Average words per page: {total_words / len(pages):.1f}")
    logger.info(f"Total processing time: {total_time:.2f}s")
    
    return all_pages_ocr

def align_annotation(ocr_words: List[Dict], annotation_text: str, min_confidence: float = 0.5) -> List[Dict]:
    """
    Aligns annotation text with OCR words using improved matching algorithm.
    
    Args:
        ocr_words: List of OCR word dictionaries
        annotation_text: Text to align
        min_confidence: Minimum confidence threshold for matching
        
    Returns:
        List of matched word dictionaries with their bounding boxes
    """
    logger = logging.getLogger(__name__)
    
    if not ocr_words:
        return []
    
    # More aggressive text normalization
    def normalize_text(text):
        import re
        # Convert to lowercase
        text = text.lower()
        # Replace common OCR errors and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation that often causes mismatches
        text = re.sub(r'[.,;:()"\'-]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Normalize texts for matching
    normalized_annotation = normalize_text(annotation_text)
    
    # Create two versions of OCR text - one with spaces between words, one without
    ocr_text_with_spaces = " ".join(normalize_text(word['text']) for word in ocr_words)
    ocr_text_no_spaces = "".join(normalize_text(word['text']) for word in ocr_words)
    
    # Try multiple matching strategies
    
    # 1. Direct substring match (most reliable)
    if normalized_annotation in ocr_text_with_spaces:
        start_idx = ocr_text_with_spaces.index(normalized_annotation)
        end_idx = start_idx + len(normalized_annotation)
        match_type = "exact"
        match_text = ocr_text_with_spaces
        logger.debug(f"Found exact match for: {annotation_text[:50]}...")
    
    # 2. Try matching without spaces (helps with word boundary issues)
    elif normalized_annotation.replace(" ", "") in ocr_text_no_spaces:
        no_space_annotation = normalized_annotation.replace(" ", "")
        start_idx = ocr_text_no_spaces.index(no_space_annotation)
        end_idx = start_idx + len(no_space_annotation)
        match_type = "no_spaces"
        match_text = ocr_text_no_spaces
        logger.debug(f"Found no-spaces match for: {annotation_text[:50]}...")
    
    # 3. Use fuzzy matching with a lower threshold (60% instead of 70%)
    else:
        # Try different combinations of text normalization for fuzzy matching
        matcher = difflib.SequenceMatcher(None, ocr_text_with_spaces, normalized_annotation)
        match = matcher.find_longest_match(0, len(ocr_text_with_spaces), 0, len(normalized_annotation))
        
        # If match is too small, try alternative approaches
        if match.size < len(normalized_annotation) * 0.6:
            # Try word-by-word matching for very low confidence cases
            annotation_words = normalized_annotation.split()
            if len(annotation_words) > 3:  # Only try for longer annotations
                # Check if at least 60% of the words appear in the OCR text
                found_words = [word for word in annotation_words if word in ocr_text_with_spaces]
                if len(found_words) / len(annotation_words) >= 0.6:
                    logger.debug(f"Found partial word match ({len(found_words)}/{len(annotation_words)} words) for: {annotation_text[:50]}...")
                    
                    # Use all OCR words as a fallback
                    # This is not ideal but better than nothing
                    return ocr_words
                
            logger.warning(f"Low confidence match for: {annotation_text[:50]}...")
            return []
        else:
            # Try sliding window matching for longer texts (over 100 chars)
            if len(normalized_annotation) > 100:
                # Use smaller chunks of the text to match
                chunk_size = min(80, len(normalized_annotation) // 2)
                # Try start, middle and end chunks
                start_chunk = normalized_annotation[:chunk_size]
                end_chunk = normalized_annotation[-chunk_size:]
                mid_point = len(normalized_annotation) // 2
                mid_chunk = normalized_annotation[mid_point-chunk_size//2:mid_point+chunk_size//2]
                
                for chunk in [start_chunk, mid_chunk, end_chunk]:
                    matcher = difflib.SequenceMatcher(None, ocr_text_with_spaces, chunk)
                    match = matcher.find_longest_match(0, len(ocr_text_with_spaces), 0, len(chunk))
                    
                    if match.size > len(chunk) * 0.7:  # Higher threshold for chunks
                        # Found a good chunk match, now expand to include surrounding context
                        start_idx = max(0, match.a - chunk_size)
                        end_idx = min(len(ocr_text_with_spaces), match.a + match.size + chunk_size)
                        
                        # Use this expanded region for word mapping
                        match_type = "chunk"
                        match_text = ocr_text_with_spaces
                        logger.debug(f"Found chunk match for: {annotation_text[:50]}...")
                        break
                else:
                    # None of the chunks matched well
                    logger.warning(f"Low confidence match for: {annotation_text[:50]}...")
                    return []
            
        start_idx = match.a
        end_idx = start_idx + match.size
        match_type = "fuzzy"
        match_text = ocr_text_with_spaces
        logger.debug(f"Found fuzzy match ({match.size/len(normalized_annotation):.2%} confidence) for: {annotation_text[:50]}...")

    # Map character positions to word indices
    matched_words = []
    
    # Different mapping strategy based on match type
    if match_type == "no_spaces":
        # For no_spaces matches, we need to map back to the original words
        # This is approximate but better than nothing
        char_count = 0
        for word in ocr_words:
            word_text = normalize_text(word['text'])
            word_no_spaces = word_text.replace(" ", "")
            word_length = len(word_no_spaces)
            
            word_start = char_count
            word_end = char_count + word_length
            
            # Check if word overlaps with matched region
            if word_end > start_idx and word_start < end_idx:
                if word.get('confidence', 1.0) >= min_confidence:
                    matched_words.append(word)
                    
            char_count += word_length
    else:
        # For exact and fuzzy matches with spaces
        current_pos = 0
        for word in ocr_words:
            word_text = normalize_text(word['text'])
            word_start = current_pos
            word_end = current_pos + len(word_text)
            
            # Check if word overlaps with matched region
            if word_end > start_idx and word_start < end_idx:
                if word.get('confidence', 1.0) >= min_confidence:
                    matched_words.append(word)
                    
            current_pos = word_end + 1  # +1 for space
    
    logger.debug(f"Matched {len(matched_words)} words using {match_type} matching")
    return matched_words

def process_cuad_document(pdf_path: str, annotations: Dict, output_dir: str):
    """
    Process a single CUAD document and create LayoutLMv3 compatible annotations.
    
    Args:
        pdf_path: Path to the PDF file
        annotations: CUAD annotations for this document
        output_dir: Directory to save the processed annotations
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    logger.info(f"Processing document: {pdf_path}")
    
    # Get OCR results for all pages
    all_pages_ocr = pdf_to_ocr_words(pdf_path)
    if not all_pages_ocr:
        logger.error(f"Failed to process PDF: {pdf_path}")
        return
    
    # Analyze document structure to help with annotation alignment
    document_structure = analyze_document_structure(all_pages_ocr)
        
    # Create LayoutLMv3 compatible annotations
    layoutlm_annotations = []
    
    # NEW: Track failed annotations
    failed_annotations = []
    
    # In CUAD dataset, each document has 'paragraphs' containing 'qas'
    total_qa_pairs = 0
    successful_matches = 0
    
    # Process each paragraph and its QA pairs
    if 'paragraphs' in annotations:
        for paragraph in annotations['paragraphs']:
            if 'qas' in paragraph:
                for qa in paragraph['qas']:
                    if 'answers' in qa and qa['answers']:
                        total_qa_pairs += len(qa['answers'])
                        
                        for answer in qa['answers']:
                            answer_text = answer['text']
                            
                            # Try multi-page matching for longer annotations
                            # if len(answer_text) > 200:
                            #     # Combine text from multiple pages for potential matches across page boundaries
                            #     match_found = try_multi_page_match(all_pages_ocr, answer_text, qa, layoutlm_annotations)
                            #     if match_found:
                            #         successful_matches += 1
                            #         continue
                            
                            # Standard single-page matching approach
                            match_found = False
                            for page_num, page_words in enumerate(all_pages_ocr):
                                matched_words = align_annotation(page_words, answer_text)
                                
                                if matched_words:
                                    match_found = True
                                    successful_matches += 1
                                    # Create annotation entry
                                    annotation = {
                                        'id': f"{qa['id']}_{page_num}",
                                        'question': qa['question'],
                                        'answer_text': answer_text,
                                        'page_number': page_num,
                                        'words': [
                                            {
                                                'text': word['text'],
                                                'bbox': word['bbox'],
                                                'confidence': word.get('confidence', 1.0)
                                            }
                                            for word in matched_words
                                        ]
                                    }
                                    layoutlm_annotations.append(annotation)
                                    logger.debug(f"Successfully matched answer for question: {qa['question'][:50]}...")
                                    break  # Found the answer, move to next QA pair
                            
                            if not match_found:
                                # Try fallback strategies for unmatched text
                                fallback_success = try_fallback_matching(all_pages_ocr, answer_text, qa, layoutlm_annotations)
                                if fallback_success:
                                    successful_matches += 1
                                else:
                                    # NEW: Track failed annotation
                                    failed_annotations.append({
                                        'id': qa['id'],
                                        'question': qa['question'],
                                        'answer_text': answer_text,
                                        'answer_length': len(answer_text)
                                    })
                                    logger.warning(f"Could not find match for answer: {answer_text[:50]}...")
    else:
        logger.error(f"No paragraphs found in document: {annotations['title']}")
        return
    
    # Post-process annotations to add metadata and improve quality
    layoutlm_annotations = post_process_annotations(layoutlm_annotations, document_structure)
    
    # Save annotations
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_layoutlm.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'file_name': pdf_path,
            'annotations': layoutlm_annotations
        }, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - start_time
    success_rate = successful_matches / total_qa_pairs if total_qa_pairs > 0 else 0
    
    logger.info(f"Completed processing {pdf_path}")
    logger.info(f"Success rate: {success_rate:.2%} ({successful_matches}/{total_qa_pairs} QA pairs)")
    logger.info(f"Processing time: {total_time:.2f}s")
    
    # NEW: Log detailed information about failed annotations if any
    if failed_annotations:
        log_failed_annotations(failed_annotations, logger)

def log_failed_annotations(failed_annotations, logger):
    """
    Log detailed information about failed annotations to help with debugging.
    
    Args:
        failed_annotations: List of dictionaries with information about failed annotations
        logger: Logger instance to use for output
    """
    if not failed_annotations:
        return
    
    total_failures = len(failed_annotations)
    
    # Create a prominent warning header
    logger.error("\n" + "!" * 80)
    logger.error(f"⚠️  FAILED ANNOTATION WARNING: {total_failures} annotations could not be matched")
    logger.error("!" * 80 + "\n")
    
    # Group failures by possible causes
    by_length = {"short": [], "medium": [], "long": [], "very_long": []}
    for failed in failed_annotations:
        length = len(failed['answer_text'])
        if length < 50:
            by_length["short"].append(failed)
        elif length < 200:
            by_length["medium"].append(failed)
        elif length < 500:
            by_length["long"].append(failed)
        else:
            by_length["very_long"].append(failed)
    
    # Log statistics by length category
    logger.error("FAILURE ANALYSIS BY TEXT LENGTH:")
    logger.error(f"  Short texts (<50 chars): {len(by_length['short'])} failures")
    logger.error(f"  Medium texts (50-200 chars): {len(by_length['medium'])} failures")
    logger.error(f"  Long texts (200-500 chars): {len(by_length['long'])} failures")
    logger.error(f"  Very long texts (>500 chars): {len(by_length['very_long'])} failures\n")
    
    # Log detailed information for each failed annotation
    logger.error("DETAILED FAILURE REPORT:")
    for i, failed in enumerate(failed_annotations):
        logger.error(f"\nFAILED ANNOTATION #{i+1}")
        logger.error(f"Question: {failed['question']}")
        logger.error(f"Text Length: {failed['answer_length']} characters")
        
        # For shorter texts, show the full text
        if failed['answer_length'] < 100:
            logger.error(f"Answer Text: {failed['answer_text']}")
        else:
            # For longer texts, show the beginning and end
            logger.error(f"Answer Text (first 100 chars): {failed['answer_text'][:100]}...")
            logger.error(f"Answer Text (last 100 chars): ...{failed['answer_text'][-100:]}")
        
        # Provide hints based on text characteristics
        hints = []
        if failed['answer_length'] > 500:
            hints.append("Text is very long - may span multiple pages")
        if any(char in failed['answer_text'] for char in ['•', '©', '®', '™', '§']):
            hints.append("Contains special characters that may affect matching")
        if failed['answer_text'].count('\n') > 3:
            hints.append("Contains multiple line breaks")
        if sum(1 for c in failed['answer_text'] if c.isdigit()) > failed['answer_length'] * 0.2:
            hints.append("Contains many numbers - consider specialized matching")
            
        if hints:
            logger.error("Possible issues:")
            for hint in hints:
                logger.error(f"  - {hint}")
            
        logger.error("-" * 50)
    
    # Add summary and suggestions
    logger.error("\nTROUBLESHOOTING SUGGESTIONS:")
    logger.error("1. Review OCR quality on pages with many failures")
    logger.error("2. Consider adjusting matching thresholds for specific document types")
    logger.error("3. For texts with special formatting, consider custom preprocessing")
    logger.error("4. Check for multi-page annotations that might need special handling")
    logger.error("\n" + "!" * 80 + "\n")

# find specific pdf files instead of using --limit
def find_specific_pdf_files(pdf_dir: str, part_config=None) -> List[str]:
    """
    Find PDF files based on a configuration of parts and directories.
    
    Args:
        pdf_dir: Base directory containing PDF parts
        part_config: Dictionary mapping part names to number of directories to include
                     If None, will default to {"Part_I": 5}
                     Special value "all" means include all directories
    
    Returns:
        List of PDF file paths matching the criteria
    """
    if not part_config:
        part_config = {"Part_I": 10}  # Default configuration
    
    pdf_files = []
    
    for part_name, num_dirs in part_config.items():
        part_path = os.path.join(pdf_dir, part_name)
        
        # Check if the part directory exists
        if not os.path.exists(part_path):
            print(f"Warning: {part_path} does not exist")
            continue
        
        # Get all subdirectories in the part
        subdirs = [d for d in os.listdir(part_path) 
                  if os.path.isdir(os.path.join(part_path, d))]
        
        # Sort them alphabetically
        subdirs.sort()
        
        # Take all or specified number of directories
        if num_dirs == "all":
            target_dirs = subdirs
        else:
            target_dirs = subdirs[:num_dirs]
            
        print(f"Processing from {part_name}: {len(target_dirs)} directories - {target_dirs}")
        
        # Now find all PDFs in these directories
        for subdir in target_dirs:
            subdir_path = os.path.join(part_path, subdir)
            for file in os.listdir(subdir_path):
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(subdir_path, file)
                    pdf_files.append(pdf_path)
    
    return pdf_files

def find_pdf_files(pdf_dir: str) -> List[str]:
    """
    Recursively finds all PDF files within the given directory.
    """
    pdf_files = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    return pdf_files

def try_multi_page_match(all_pages_ocr, answer_text, qa, layoutlm_annotations):
    """
    Try to match text that might span across multiple pages.
    """
    logger = logging.getLogger(__name__)
    
    # For simplicity, try combining adjacent pages (pairs of pages)
    for i in range(len(all_pages_ocr) - 1):
        # Combine words from two consecutive pages
        combined_words = all_pages_ocr[i] + all_pages_ocr[i+1]
        
        # Update page number in the combined list to maintain correct reference
        for word in all_pages_ocr[i+1]:
            word['original_page'] = word['page']  # Save original page
            word['page'] = i+1  # Keep page number consistent with word's source
        
        matched_words = align_annotation(combined_words, answer_text)
        
        if matched_words:
            # Group matched words by their page
            words_by_page = {}
            for word in matched_words:
                page = word.get('original_page', word['page'])
                if page not in words_by_page:
                    words_by_page[page] = []
                words_by_page[page].append(word)
            
            # Create annotations for each page that contains matched words
            for page_num, words in words_by_page.items():
                annotation = {
                    'id': f"{qa['id']}_{page_num}_multi",
                    'question': qa['question'],
                    'answer_text': answer_text,
                    'page_number': page_num,
                    'words': [
                        {
                            'text': word['text'],
                            'bbox': word['bbox'],
                            'confidence': word.get('confidence', 1.0)
                        }
                        for word in words
                    ],
                    'multi_page': True,
                    'part_of': len(words_by_page)
                }
                layoutlm_annotations.append(annotation)
            
            logger.debug(f"Found multi-page match for: {answer_text[:50]}...")
            return True
    
    return False

def try_fallback_matching(all_pages_ocr, answer_text, qa, layoutlm_annotations):
    """
    Try alternative matching approaches for difficult cases.
    """
    logger = logging.getLogger(__name__)
    
    # Strategy 1: Try keyword-based matching for longer texts
    if len(answer_text) > 100:
        # Extract keywords (simple approach)
        import re
        keywords = re.findall(r'\b\w{4,}\b', answer_text.lower())
        keywords = [k for k in keywords if k not in ['that', 'this', 'which', 'from', 'with', 'have', 'shall']]
        
        if len(keywords) > 3:  # Need enough meaningful keywords
            # Find pages with the highest density of these keywords
            page_scores = []
            for page_num, page_words in enumerate(all_pages_ocr):
                page_text = " ".join(w['text'].lower() for w in page_words)
                score = sum(1 for k in keywords if k in page_text)
                if score > len(keywords) * 0.4:  # At least 40% of keywords found
                    page_scores.append((page_num, score, page_words))
            
            # Use the page with the highest score if any good matches found
            if page_scores:
                page_scores.sort(key=lambda x: x[1], reverse=True)
                best_page_num, score, page_words = page_scores[0]
                
                # Use a lower threshold for accepting this match
                matched_words = align_annotation(page_words, answer_text, min_confidence=0.4)
                
                if matched_words:
                    # Create annotation with lower confidence flag
                    annotation = {
                        'id': f"{qa['id']}_{best_page_num}_keyword",
                        'question': qa['question'],
                        'answer_text': answer_text,
                        'page_number': best_page_num,
                        'words': [
                            {
                                'text': word['text'],
                                'bbox': word['bbox'],
                                'confidence': word.get('confidence', 1.0)
                            }
                            for word in matched_words
                        ],
                        'match_confidence': 'keyword_based',
                        'keyword_match_score': score / len(keywords)
                    }
                    layoutlm_annotations.append(annotation)
                    logger.debug(f"Found keyword-based match for: {answer_text[:50]}...")
                    return True
    
    # Strategy 2: For short texts, try exact match with individual words
    if len(answer_text) < 50:
        for page_num, page_words in enumerate(all_pages_ocr):
            for i, word in enumerate(page_words):
                if answer_text.lower() in word['text'].lower():
                    # Create annotation with single word match
                    annotation = {
                        'id': f"{qa['id']}_{page_num}_exact",
                        'question': qa['question'],
                        'answer_text': answer_text,
                        'page_number': page_num,
                        'words': [
                            {
                                'text': word['text'],
                                'bbox': word['bbox'],
                                'confidence': word.get('confidence', 1.0)
                            }
                        ],
                        'match_confidence': 'exact_word'
                    }
                    layoutlm_annotations.append(annotation)
                    logger.debug(f"Found exact word match for: {answer_text}")
                    return True
    
    return False

def analyze_document_structure(all_pages_ocr):
    """
    Analyze document structure to identify sections, headers, and other structural elements
    that can help with annotation alignment.
    """
    document_structure = {
        'headers': [],
        'sections': {},
        'page_headers': {},
        'page_footers': {}
    }
    
    # Process each page to identify potential headers based on position, font size (via bbox height)
    for page_num, page_words in enumerate(all_pages_ocr):
        # Sort words by y-coordinate to process from top to bottom
        sorted_by_y = sorted(page_words, key=lambda w: w['bbox'][1])
        
        # Identify potential page headers (words at the top of the page)
        top_words = [w for w in sorted_by_y if w['bbox'][1] < 150]  # Top 150px
        if top_words:
            document_structure['page_headers'][page_num] = top_words
        
        # Identify potential page footers (words at the bottom of the page)
        # Assuming a page height of around 1000-1200px
        bottom_words = [w for w in sorted_by_y if w['bbox'][3] > 1000]
        if bottom_words:
            document_structure['page_footers'][page_num] = bottom_words
        
        # Identify potential section headers based on text properties
        import re
        for word in page_words:
            text = word['text'].strip()
            # Check for all caps section headers or numbered sections
            if (text.isupper() and len(text) > 3) or re.match(r'^[0-9]+\.\s+[A-Z]', text):
                document_structure['headers'].append({
                    'text': text,
                    'page': page_num,
                    'bbox': word['bbox']
                })
    
    return document_structure

def post_process_annotations(annotations, document_structure):
    """
    Post-process annotations to improve quality and consistency.
    """
    processed_annotations = []
    
    for annotation in annotations:
        # Skip annotations with no words matched
        if not annotation.get('words'):
            continue
        
        # Calculate confidence metrics
        words = annotation['words']
        avg_confidence = sum(w.get('confidence', 0) for w in words) / len(words) if words else 0
        
        # Flag low-confidence annotations
        if avg_confidence < 0.7:
            annotation['low_confidence'] = True
        
        # Add document structure context if available
        page_num = annotation.get('page_number')
        if page_num is not None:
            # Add nearby headers for context
            nearby_headers = [h for h in document_structure['headers'] 
                              if h['page'] == page_num]
            if nearby_headers:
                annotation['nearby_headers'] = nearby_headers
        
        processed_annotations.append(annotation)
    
    return processed_annotations

def main():
    """
    Main function to process CUAD dataset with improved matching
    """
    import argparse
    
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Process CUAD dataset for LayoutLMv3')
    parser.add_argument('--cuad-dir', type=str, default="CUAD_v1/", 
                        help='Directory containing CUAD dataset files')
    parser.add_argument('--output-dir', type=str, default="CUAD_v1/layoutlmv3", 
                        help='Directory to save output files')
    parser.add_argument('--dpi', type=int, default=400, 
                        help='DPI for PDF to image conversion (higher = better quality but slower)')
    parser.add_argument('--match-threshold', type=float, default=0.6, 
                        help='Minimum threshold for fuzzy matching (0.0-1.0)')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit processing to this many documents (for testing)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip processing documents that already have output files')
    parser.add_argument('--target-part', type=str, default="Part_I", 
                    help='Target part directory to process (e.g., Part_I)')
    parser.add_argument('--num-dirs', type=int, default=5, 
                    help='Number of directories to process within the target part')
    parser.add_argument('--part-config', type=str, default=None,
                    help='JSON configuration of parts and directory counts, e.g., \'{"Part_I":"all","Part_II":10}\'')
    args = parser.parse_args()
    
    # Configure paths from arguments
    cuad_dir = args.cuad_dir
    pdf_dir = os.path.join(cuad_dir, "full_contract_pdf")
    annotation_file = os.path.join(cuad_dir, "CUAD_v1.json")
    output_dir = args.output_dir
    
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    start_time = time.time()
    
    logger.info("Starting CUAD dataset processing with improved matching algorithms")
    logger.info(f"Input directory: {cuad_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"PDF DPI: {args.dpi}")
    logger.info(f"Match threshold: {args.match_threshold}")
    
    # Check dependencies
    if not OCR_AVAILABLE:
        logger.error("PaddleOCR is not available. Please install with 'pip install paddleocr'")
        logger.error("Exiting the program.")
        return
    
    # Check if CUAD directory exists
    if not os.path.exists(cuad_dir):
        logger.error(f"CUAD directory not found: {cuad_dir}")
        logger.error("Please download the CUAD dataset and extract it to this directory.")
        return
        
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        logger.error("Please download the CUAD PDFs and extract them to the full_contract_pdf directory.")
        return
    
    # Load CUAD annotations
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            cuad_annotations = json.load(f)
        logger.info(f"Successfully loaded annotations from {annotation_file}")
        logger.info(f"Found {len(cuad_annotations['data'])} documents in annotations")
    except Exception as e:
        logger.error(f"Failed to load annotations: {str(e)}")
        return
    
    # Find PDF SPECIFIC files to process: example: python annotate.py --part-config='{"Part_I":"all","Part_II":10,"Part_III":5}'
    # Parse part configuration if provided
    part_config = None
    if args.part_config:
        try:
            part_config = json.loads(args.part_config)
        except json.JSONDecodeError:
            logger.error(f"Invalid part configuration format: {args.part_config}")
            logger.error("Expected JSON format like: '{\"Part_I\":\"all\",\"Part_II\":10}'")
            return

    # Use part config if provided, otherwise use target_part and num_dirs
    if part_config:
        pdf_files = find_specific_pdf_files(pdf_dir, part_config=part_config)
    else:
        # Default to single part with specified number of directories
        pdf_files = find_specific_pdf_files(pdf_dir, part_config={args.target_part: args.num_dirs})
        
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return

    # Apply document limit if specified
    if args.limit and args.limit > 0:
        pdf_files = pdf_files[:args.limit]
        logger.info(f"Limiting processing to the first {args.limit} PDF files")
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each document with progress tracking
    successful_docs = 0
    total_docs = len(pdf_files)
    total_qa_pairs = 0
    total_successful_matches = 0
    skipped_docs = 0
    
    for pdf_idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing documents")):
        doc_title = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, f"{doc_title}_layoutlm.json")
        
        # Skip if file exists and skip-existing flag is set
        if args.skip_existing and os.path.exists(output_path):
            logger.info(f"Skipping {doc_title} - output file already exists")
            skipped_docs += 1
            continue
        
        # Find the corresponding annotation
        doc = next((d for d in cuad_annotations['data'] if d['title'] == doc_title), None)
        
        if doc:
            logger.info(f"Processing document {pdf_idx+1}/{total_docs}: {doc_title}")
            
            try:
                process_cuad_document(pdf_path, doc, output_dir)
                successful_docs += 1
                
                # Count QA pairs and successful matches for more detailed stats
                qa_count = sum(len(para.get('qas', [])) for para in doc.get('paragraphs', []))
                total_qa_pairs += qa_count
                
                # Try to read the output file to get the number of successful matches
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        successful_matches = len(result_data.get('annotations', []))
                        total_successful_matches += successful_matches
                        logger.info(f"Successfully matched {successful_matches}/{qa_count} annotations")
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Error processing document {doc_title}: {str(e)}")
        else:
            logger.error(f"Annotation not found for PDF: {pdf_path}")
    
    # Calculate overall statistics
    total_time = time.time() - start_time
    processed_docs = total_docs - skipped_docs
    doc_success_rate = successful_docs / processed_docs if processed_docs > 0 else 0
    annotation_success_rate = total_successful_matches / total_qa_pairs if total_qa_pairs > 0 else 0
    
    logger.info("========== CUAD Dataset Processing Summary ==========")
    logger.info(f"Documents processed: {successful_docs}/{processed_docs} ({doc_success_rate:.2%} success rate)")
    if skipped_docs > 0:
        logger.info(f"Documents skipped: {skipped_docs}")
    logger.info(f"Annotations matched: {total_successful_matches}/{total_qa_pairs} ({annotation_success_rate:.2%} success rate)")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Average time per document: {total_time/processed_docs:.2f}s" if processed_docs > 0 else "No documents processed")
    logger.info("====================================================")
    
    # Write summary to output directory
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "documents": {
                "total": total_docs,
                "processed": processed_docs,
                "successful": successful_docs,
                "skipped": skipped_docs,
                "success_rate": doc_success_rate
            },
            "annotations": {
                "total": total_qa_pairs,
                "matched": total_successful_matches,
                "success_rate": annotation_success_rate
            },
            "timing": {
                "total_seconds": total_time,
                "average_per_document": total_time/processed_docs if processed_docs > 0 else 0
            },
            "settings": {
                "dpi": args.dpi,
                "match_threshold": args.match_threshold,
                "limit": args.limit
            }
        }, f, indent=2)

    # Log an overall summary of failures if there are any
    if total_qa_pairs > total_successful_matches:
        total_failures = total_qa_pairs - total_successful_matches
        failure_rate = total_failures / total_qa_pairs
        
        logger.error("\n" + "=" * 60)
        logger.error(f"OVERALL ANNOTATION FAILURE SUMMARY")
        logger.error("=" * 60)
        logger.error(f"Total failed annotations: {total_failures} / {total_qa_pairs} ({failure_rate:.2%})")
        logger.error(f"These annotations could not be matched to the document text.")
        logger.error("Check individual document logs for detailed failure reports.")
        logger.error("=" * 60 + "\n")

if __name__ == "__main__":
    main()
