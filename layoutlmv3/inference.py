import os
import json
import argparse
from typing import Dict, List, Tuple
import torch
import numpy as np
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import sys

# Set matplotlib backend to avoid TkAgg errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import cv2
import textwrap
from tqdm import tqdm
from paddleocr import PaddleOCR


def parse_args():
    # Check if CUDA is actually available
    cuda_available = torch.cuda.is_available()
    if not cuda_available and torch.backends.cuda.is_built():
        print("WARNING: CUDA is not available even though PyTorch detects CUDA installation.")
        print("This might be due to a driver issue or insufficient GPU memory.")
    elif not torch.backends.cuda.is_built():
        print("WARNING: PyTorch was not compiled with CUDA enabled.")
        print("Using CPU for inference, which will be slower.")

    default_device = "cuda" if cuda_available else "cpu"
    
    parser = argparse.ArgumentParser(description="Inference with LayoutLMv3 on legal documents")
    parser.add_argument("--model_dir", type=str, default="layoutlmv3/model/best_model", 
                        help="Directory containing the fine-tuned model")
    parser.add_argument("--processor_dir", type=str, default="layoutlmv3/model", 
                        help="Directory containing the processor")
    parser.add_argument("--pdf_path", type=str, required=True, 
                        help="Path to the PDF document to analyze")
    parser.add_argument("--output_dir", type=str, default="layoutlmv3/results", 
                        help="Directory to save the results")
    parser.add_argument("--device", type=str, default=default_device, 
                        help="Device to use for inference (cuda/cpu)")
    parser.add_argument("--dpi", type=int, default=300, 
                        help="DPI for PDF to image conversion")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, 
                        help="Threshold for token classification confidence")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        try:
            # Try standard parsing
            args = parser.parse_args()
        except SystemExit:
            print("Error parsing arguments. Trying to fix path with spaces...")
            
            # Try reconstructing the command with quotes around the PDF path
            import re
            cmdline = " ".join(sys.argv[1:])
            # Find --pdf_path argument and its value
            pdf_match = re.search(r'--pdf_path\s+(.+?)(?=\s+--|$)', cmdline + " --")
            if pdf_match:
                pdf_path = pdf_match.group(1).strip()
                # Replace with quoted version if not already quoted
                if not (pdf_path.startswith('"') and pdf_path.endswith('"')) and ' ' in pdf_path:
                    quoted_path = f'"{pdf_path}"'
                    cmdline = cmdline.replace(f"--pdf_path {pdf_path}", f"--pdf_path {quoted_path}")
                    print(f"Fixed command: {sys.argv[0]} {cmdline}")
                    
                    # Reconstruct argv for argparse
                    import shlex
                    new_argv = [sys.argv[0]] + shlex.split(cmdline)
                    sys.argv = new_argv
                    args = parser.parse_args()
                else:
                    # Create a temporary file with no spaces
                    temp_dir = os.path.dirname(os.path.abspath(__file__))
                    orig_pdf = pdf_path
                    temp_pdf = os.path.join(temp_dir, f"temp_{os.path.basename(orig_pdf).replace(' ', '_')}")
                    import shutil
                    print(f"Creating temporary file without spaces: {temp_pdf}")
                    try:
                        shutil.copy2(orig_pdf, temp_pdf)
                        # Replace in argv
                        pdf_idx = sys.argv.index("--pdf_path") + 1
                        if pdf_idx < len(sys.argv):
                            sys.argv[pdf_idx] = temp_pdf
                        args = parser.parse_args()
                        args.original_pdf_path = orig_pdf  # Store original path
                        args.is_temp_file = True
                    except Exception as e:
                        print(f"Error creating temporary file: {e}")
                        raise
            else:
                # If all else fails, resort to a default set of arguments
                print("Could not fix command line arguments, using default values.")
                args = parser.parse_args([])
                if "--pdf_path" not in cmdline:
                    print("ERROR: --pdf_path argument is required")
                    sys.exit(1)
    else:
        args = parser.parse_args()
    
    return args


def get_predictions(
    model: torch.nn.Module,
    processor: LayoutLMv3Processor,
    image: Image.Image,
    device: torch.device
) -> Tuple[List[str], List[List[int]], List[int], List[float]]:
    """
    Get token classification predictions for an image.
    
    Args:
        model: The fine-tuned LayoutLMv3 model.
        processor: The LayoutLMv3 processor.
        image: The input image.
        device: The device to run inference on.
        
    Returns:
        Tuple of (words, bboxes, predicted_labels, confidence_scores)
    """
    # Get OCR results using PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    
    # Perform OCR
    results = ocr.ocr(np.array(image), cls=True)
    
    if not results[0]:
        print("No text detected in image")
        return [], [], [], []
    
    # Extract text and bounding boxes
    words = []
    bboxes = []
    
    for line in results[0]:
        # Each line contains: ([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence))
        box = line[0]
        text = line[1][0]
        
        # Convert PaddleOCR bounding box to LayoutLMv3 format (x1, y1, x2, y2)
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        
        # Normalize to 0-1000 range
        width, height = image.size
        bbox_normalized = [
            int(1000 * x1 / width),
            int(1000 * y1 / height),
            int(1000 * x2 / width),
            int(1000 * y2 / height)
        ]
        
        words.append(text)
        bboxes.append(bbox_normalized)
    
    # Encode inputs
    encoding = processor(
        images=image,
        text=words,
        boxes=bboxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )
    
    # Move inputs to device
    for k, v in encoding.items():
        encoding[k] = v.to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Post-process predictions
    predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
    scores = torch.softmax(outputs.logits, dim=-1).max(-1).values.squeeze().cpu().numpy()
    
    # Filter out special tokens and padding
    mask = encoding["attention_mask"].squeeze().cpu().numpy() == 1
    word_ids = encoding.word_ids()
    
    predicted_labels = []
    confidence_scores = []
    
    for idx, m in enumerate(mask):
        if m and word_ids[idx] is not None:
            predicted_labels.append(predictions[idx])
            confidence_scores.append(scores[idx])
    
    # Group by words (since tokenization may split words)
    word_predictions = []
    word_scores = []
    
    current_word_id = -1
    for idx, word_id in enumerate(word_ids):
        if word_id is None or not mask[idx]:
            continue
        
        if word_id != current_word_id:
            word_predictions.append([])
            word_scores.append([])
            current_word_id = word_id
        
        word_predictions[-1].append(predictions[idx])
        word_scores[-1].append(scores[idx])
    
    # Take the most common prediction for each word
    final_predictions = []
    final_scores = []
    
    for word_preds, word_score in zip(word_predictions, word_scores):
        if not word_preds:
            continue
        
        # Most common prediction
        unique_preds, counts = np.unique(word_preds, return_counts=True)
        most_common_idx = np.argmax(counts)
        pred = unique_preds[most_common_idx]
        
        # Average confidence for the most common prediction
        pred_indices = [i for i, p in enumerate(word_preds) if p == pred]
        avg_score = np.mean([word_score[i] for i in pred_indices])
        
        final_predictions.append(pred)
        final_scores.append(avg_score)
    
    return words, bboxes, final_predictions, final_scores


def visualize_results(
    image: Image.Image,
    words: List[str],
    bboxes: List[List[int]],
    predictions: List[int],
    scores: List[float],
    label_map: Dict[str, int],
    confidence_threshold: float = 0.5,
    output_path: str = None
) -> Image.Image:
    """
    Visualize the token classification results on the image.
    
    Args:
        image: The input image.
        words: List of words.
        bboxes: List of bounding boxes.
        predictions: List of predicted label IDs.
        scores: List of confidence scores.
        label_map: Mapping from label ID to label name.
        confidence_threshold: Threshold for token classification confidence.
        output_path: Path to save the annotated image.
        
    Returns:
        The annotated image.
    """
    # If there are no words, return the original image
    if not words:
        print("No words to visualize")
        if output_path:
            # Save the original image
            image.save(output_path)
        return image
    
    # Create a copy of the image for drawing
    width, height = image.size
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    
    # Draw bounding boxes and labels directly on the image
    try:
        # Create a reverse label map (ID -> name)
        id_to_label = {v: k for k, v in label_map.items()}
        
        # Define a color map for different labels
        # Generate distinct colors using HSV color space
        label_colors = {}
        for label_id in set(predictions):
            if label_id == 0:  # Skip "O" label
                continue
            
            # Generate a vibrant color based on label_id
            hue = (label_id * 0.15) % 1.0  # Spread colors across hue spectrum
            rgb = hsv_to_rgb((hue, 0.8, 0.9))  # Convert HSV to RGB
            color = tuple(int(c * 255) for c in rgb)
            label_colors[label_id] = color
        
        # Draw each predicted entity
        for word, bbox, pred, score in zip(words, bboxes, predictions, scores):
            if pred == 0 or score < confidence_threshold:  # Skip "O" label or low confidence
                continue
            
            # Get the label name and color
            label_id = pred
            if label_id in id_to_label:
                label_name = id_to_label[label_id]
                color = label_colors.get(label_id, (255, 0, 0))  # Default to red
            else:
                continue  # Skip unknown labels
            
            # Normalize bounding box coordinates to image size
            x0, y0, x1, y1 = bbox
            x0 = int(x0 * width / 1000)
            y0 = int(y0 * height / 1000)
            x1 = int(x1 * width / 1000)
            y1 = int(y1 * height / 1000)
            
            # Draw rectangle
            draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)
            
            # Draw label text
            font_size = 12
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                # Use default font if arial is not available
                font = ImageFont.load_default()
            
            label_text = f"{label_name} ({score:.2f})"
            text_width, text_height = draw.textsize(label_text, font=font) if hasattr(draw, 'textsize') else (font_size * len(label_text) * 0.6, font_size)
            
            # Draw text background
            draw.rectangle([(x0, y0 - text_height - 2), (x0 + text_width, y0)], fill=color)
            
            # Draw text
            draw.text((x0, y0 - text_height - 2), label_text, fill="white", font=font)
        
        # Save the annotated image
        if output_path:
            # Save the original annotated image
            output_image.save(output_path)
            annotated_path = output_path.replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')
            output_image.save(annotated_path)
            print(f"Saved annotated image to {annotated_path}")
        
        return output_image
    
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        # If visualization fails, save the original image
        if output_path:
            image.save(output_path)
        return image
    
    # Create matplotlib figure for visualization - this is the backup method
    try:
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(np.array(image))
        
        # Create a reverse label map (ID -> name)
        id_to_label = {v: k for k, v in label_map.items()}
        
        # Generate distinct colors using HSV color space
        label_colors = {}
        for label_id in set(predictions):
            if label_id == 0:  # Skip "O" label
                continue
            
            # Generate a vibrant color based on label_id
            hue = (label_id * 0.15) % 1.0  # Spread colors across hue spectrum
            label_colors[label_id] = hsv_to_rgb((hue, 0.8, 0.9))
        
        # Draw each predicted entity
        for word, bbox, pred, score in zip(words, bboxes, predictions, scores):
            if pred == 0 or score < confidence_threshold:  # Skip "O" label or low confidence
                continue
            
            # Get the label name and color
            label_id = pred
            if label_id in id_to_label:
                label_name = id_to_label[label_id]
                color = label_colors.get(label_id, (1, 0, 0))  # Default to red
            else:
                continue  # Skip unknown labels
            
            # Normalize bounding box coordinates to image size
            x0, y0, x1, y1 = bbox
            x0 = x0 * width / 1000
            y0 = y0 * height / 1000
            x1 = x1 * width / 1000
            y1 = y1 * height / 1000
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, 
                linewidth=2, edgecolor=color, facecolor="none", alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label text
            label_text = f"{label_name} ({score:.2f})"
            ax.text(
                x0, y0 - 5, label_text, 
                color=color, fontsize=8, 
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
            )
        
        # Remove axes
        ax.axis("off")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            annotated_path = output_path.replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')
            plt.savefig(annotated_path, bbox_inches="tight", dpi=300)
            print(f"Saved annotated image to {annotated_path}")
        
        # Convert to PIL image
        fig.canvas.draw()
        try:
            # First try the standard method
            result_image = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
        except AttributeError:
            # If tostring_rgb is not available, use tostring_argb and convert format
            buf = fig.canvas.tostring_argb()
            width, height = fig.canvas.get_width_height()
            # ARGB to RGBA conversion
            buf = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            # Reorder from ARGB to RGBA
            buf = np.roll(buf, 1, axis=2)
            result_image = Image.fromarray(buf)
        
        plt.close(fig)
        
        return result_image
    except Exception as e:
        print(f"Error in matplotlib visualization: {str(e)}")
        if output_path:
            image.save(output_path)
        return image


def extract_clauses(
    words: List[str],
    bboxes: List[List[int]],
    predictions: List[int],
    scores: List[float],
    label_map: Dict[str, int],
    confidence_threshold: float = 0.5
) -> Dict[str, List[str]]:
    """
    Extract clauses from the predicted labels.
    
    Args:
        words: List of words.
        bboxes: List of bounding boxes.
        predictions: List of predicted label IDs.
        scores: List of confidence scores.
        label_map: Mapping from label ID to label name.
        confidence_threshold: Threshold for token classification confidence.
        
    Returns:
        Dictionary of extracted clauses by category.
    """
    # Print debug information
    print(f"Extracting clauses from {len(words)} words with threshold {confidence_threshold}")
    
    # Reverse label map (ID -> name)
    id_to_label = {v: k for k, v in label_map.items() if k != "O"}
    print(f"Available label IDs: {list(id_to_label.keys())}")
    
    # Check predicted labels
    pred_ids = [p for p, s in zip(predictions, scores) if s >= confidence_threshold]
    unique_preds = set(pred_ids)
    print(f"Predictions: {len(pred_ids)} words with predictions above threshold")
    print(f"Unique prediction IDs: {unique_preds}")
    
    # Group words by label
    clauses_by_category = {}
    
    current_label = None
    current_clause = []
    current_bboxes = []
    
    for word, bbox, pred, score in zip(words, bboxes, predictions, scores):
        if pred == 0 or score < confidence_threshold:  # Skip "O" label or low confidence
            if current_label and current_clause:
                # Save previous clause
                if current_label not in clauses_by_category:
                    clauses_by_category[current_label] = []
                clauses_by_category[current_label].append({
                    "text": " ".join(current_clause),
                    "confidence": sum([s for w, s in zip(current_clause, current_bboxes)]) / len(current_clause)
                })
                current_clause = []
                current_bboxes = []
                current_label = None
            continue
        
        # Get label name
        label_id = pred
        if label_id in id_to_label:
            label_name = id_to_label[label_id]
        else:
            continue  # Skip unknown labels
        
        # If new label or far apart from previous word, start new clause
        if label_name != current_label:
            if current_label and current_clause:
                # Save previous clause
                if current_label not in clauses_by_category:
                    clauses_by_category[current_label] = []
                clauses_by_category[current_label].append({
                    "text": " ".join(current_clause),
                    "confidence": sum([s for w, s in zip(current_clause, current_bboxes)]) / len(current_clause)
                })
                current_clause = []
                current_bboxes = []
            
            current_label = label_name
        
        # Add word to current clause
        current_clause.append(word)
        current_bboxes.append(score)
    
    # Add last clause if any
    if current_label and current_clause:
        if current_label not in clauses_by_category:
            clauses_by_category[current_label] = []
        clauses_by_category[current_label].append({
            "text": " ".join(current_clause),
            "confidence": sum([s for w, s in zip(current_clause, current_bboxes)]) / len(current_clause)
        })
    
    # Print results
    print(f"Extracted {sum(len(clauses) for clauses in clauses_by_category.values())} clauses across {len(clauses_by_category)} categories")
    for category, clauses in clauses_by_category.items():
        print(f"  - {category}: {len(clauses)} clauses")
    
    return clauses_by_category


def process_document(args):
    """
    Process a legal document using the fine-tuned LayoutLMv3 model.
    
    Args:
        args: Command line arguments.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor
    print(f"Loading model from {args.model_dir}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir)
    
    # Load processor or create a new one with OCR disabled
    try:
        processor = LayoutLMv3Processor.from_pretrained(args.processor_dir)
        # Check if processor has OCR disabled
        if getattr(processor.image_processor, "apply_ocr", False):
            print("Warning: Processor has OCR enabled. Creating a new processor with OCR disabled.")
            processor = LayoutLMv3Processor.from_pretrained(
                args.processor_dir, 
                apply_ocr=False
            )
    except Exception as e:
        print(f"Could not load processor from {args.processor_dir}: {str(e)}")
        print("Creating a new processor.")
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", 
            apply_ocr=False
        )
    
    # Load label map
    try:
        with open(os.path.join(args.processor_dir, "label_map.json"), "r") as f:
            label_map = json.load(f)
            print(f"Loaded label map with {len(label_map)} categories: {list(label_map.keys())}")
    except Exception as e:
        print(f"Error loading label map: {str(e)}")
        # Create a default label map
        print("Creating default label map")
        label_map = {"O": 0, "B-CLAUSE": 1, "I-CLAUSE": 2}
    
    # Set device
    device = torch.device(args.device)
    model.to(device)
    print(f"Model loaded and moved to {args.device}")
    
    # Handle PDF path (determine the actual filename to use for output)
    pdf_path = args.pdf_path
    original_pdf_path = getattr(args, 'original_pdf_path', pdf_path)
    display_pdf_name = os.path.splitext(os.path.basename(original_pdf_path))[0]
    print(f"Processing PDF: {original_pdf_path} (Using internal path: {pdf_path})")
    
    # Convert PDF to images
    print(f"Converting PDF to images using DPI={args.dpi}")
    
    try:
        pages = convert_from_path(pdf_path, dpi=args.dpi)
        print(f"Successfully converted PDF to {len(pages)} pages")
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {str(e)}")
        return {"clauses": {}, "error": f"Error converting PDF: {str(e)}"}
    
    # Create result directory for this document (using the original filename)
    doc_result_dir = os.path.join(args.output_dir, display_pdf_name)
    # Make sure the directory name is valid
    doc_result_dir = "".join(c for c in doc_result_dir if c.isalnum() or c in "._- ")
    os.makedirs(doc_result_dir, exist_ok=True)
    
    # Process each page
    all_extracted_clauses = {}
    
    for page_idx, page_image in enumerate(tqdm(pages, desc="Processing pages")):
        print(f"Processing page {page_idx + 1}/{len(pages)}")
        
        # Get predictions
        words, bboxes, predictions, scores = get_predictions(model, processor, page_image, device)
        print(f"Page {page_idx + 1}: Got {len(words)} words from OCR")
        
        if not words:
            print(f"Warning: No words detected on page {page_idx + 1}")
            continue
        
        # Extract clauses
        page_clauses = extract_clauses(words, bboxes, predictions, scores, label_map, args.confidence_threshold)
        
        # Merge with all clauses
        for label, clauses in page_clauses.items():
            if label not in all_extracted_clauses:
                all_extracted_clauses[label] = []
            
            # Add page number to each clause
            for clause in clauses:
                if isinstance(clause, dict):
                    clause["page"] = page_idx + 1
                    all_extracted_clauses[label].append(clause)
                else:
                    # Convert string clauses to dict format
                    all_extracted_clauses[label].append({
                        "text": clause,
                        "page": page_idx + 1,
                        "confidence": 0.7  # Default confidence
                    })
        
        # Visualize results
        output_image_path = os.path.join(doc_result_dir, f"page_{page_idx + 1}.jpg")
        
        try:
            # Try to visualize with annotations
            print(f"Visualizing page {page_idx + 1}...")
            visualize_results(
                page_image, words, bboxes, predictions, scores, 
                label_map, args.confidence_threshold, output_image_path
            )
        except Exception as e:
            print(f"Warning: Could not visualize results: {str(e)}")
            print("Saving original image without annotations")
            # Save original image as fallback
            page_image.save(output_image_path)
    
    # Add DEBUG category if no clauses were found
    if not all_extracted_clauses:
        print("Warning: No clauses were extracted from the document")
        all_extracted_clauses["DEBUG"] = [{
            "text": "No clauses detected in document. Try lowering the confidence threshold or check document quality.",
            "page": 1,
            "confidence": 1.0
        }]
    
    # Save extracted clauses to JSON
    clauses_output_path = os.path.join(doc_result_dir, "extracted_clauses.json")
    try:
        with open(clauses_output_path, "w", encoding="utf-8") as f:
            json.dump(all_extracted_clauses, f, indent=2, ensure_ascii=False)
        print(f"Saved extracted clauses to {clauses_output_path}")
    except Exception as e:
        print(f"Error saving clauses to JSON: {str(e)}")
    
    # Create a summary document
    summary_path = os.path.join(doc_result_dir, "summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"SUMMARY OF LEGAL CLAUSES IN DOCUMENT: {display_pdf_name}\n")
            f.write("=" * 80 + "\n\n")
            
            if not all_extracted_clauses or "DEBUG" in all_extracted_clauses:
                f.write("No legal clauses were detected in this document.\n")
                f.write("This could be due to:\n")
                f.write("1. The document doesn't contain the types of clauses the model was trained to detect\n")
                f.write("2. The confidence threshold is too high\n")
                f.write("3. The document quality is too low for accurate OCR\n\n")
            else:
                for label, clauses in sorted(all_extracted_clauses.items()):
                    f.write(f"## {label.upper()} ##\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, clause in enumerate(clauses):
                        if isinstance(clause, dict):
                            text = clause.get("text", "")
                            page = clause.get("page", "?")
                            confidence = clause.get("confidence", 0.0)
                            f.write(f"{i+1}. [Page {page}, Confidence: {confidence:.2f}] {text}\n\n")
                        else:
                            f.write(f"{i+1}. {clause}\n\n")
                    
                    f.write("\n")
    except Exception as e:
        print(f"Error creating summary document: {str(e)}")
    
    # Clean up temporary file if created
    if hasattr(args, 'is_temp_file') and args.is_temp_file and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            print(f"Removed temporary file {pdf_path}")
        except Exception as e:
            print(f"Warning: Failed to remove temporary file {pdf_path}: {str(e)}")
    
    print(f"Document processing complete. Results saved to {doc_result_dir}")
    return {"clauses": all_extracted_clauses, "output_dir": doc_result_dir}


if __name__ == "__main__":
    args = parse_args()
    process_document(args) 