import json
from pathlib import Path
import re
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OCR_JSON = "TC_label-studio_input_file.json"     # OCR file with predictions and bounding boxes
CONLL    = "project-8-at-2025-04-28-18-49-85c9b7a4.conll" # Single-document CONLL file
OUT_JSONL= "layoutlm_dataset.json"
LABEL_MAP_FILE = "label_map.json"
# Image directory where page images are stored
IMAGE_DIR = "image"
# ────────────────────────────────────────────────────────────────────────────────

def get_image_dimensions(image_path):
    """Get dimensions of an image file."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error getting dimensions for {image_path}: {e}")
        # Default dimensions if image can't be loaded
        return 2480, 3508

def extract_tokens_and_bboxes(page_data, page_num):
    """Extract tokens and bounding boxes from Label Studio OCR data."""
    tokens = []
    bboxes = []
    
    # Determine the image file path based on page number
    image_filename = f"Sample_NDA_page_{page_num}.png"
    image_path = Path(IMAGE_DIR) / image_filename
    
    # Get actual image dimensions
    page_width, page_height = get_image_dimensions(image_path)
    print(f"Page {page_num} dimensions: {page_width}x{page_height}")
    
    # Get all textarea regions sorted by y (vertical position) and then x (horizontal position)
    text_regions = [
        r for r in page_data["predictions"][0]["result"] 
        if r["type"] == "textarea" and r["from_name"] == "transcription"
    ]
    
    # Sort by y-coordinate (top to bottom) then x-coordinate (left to right)
    text_regions.sort(key=lambda r: (r["value"]["y"], r["value"]["x"]))
    
    # Process each region to get words and corresponding bounding boxes
    for region in text_regions:
        text = region["value"]["text"][0]
        words = text.split()
        
        if not words:  # Skip empty regions
            continue
            
        # Get original bounding box coordinates in pixels
        x = region["value"]["x"] * page_width / 100  # Convert from percentage to pixels
        y = region["value"]["y"] * page_height / 100
        width = region["value"]["width"] * page_width / 100
        height = region["value"]["height"] * page_height / 100
        
        # Calculate width per word (approximation)
        # This divides the region width proportionally among words based on character count
        total_chars = sum(len(word) for word in words)
        widths = [len(word) / total_chars * width for word in words]
        
        # Assign bounding boxes to each word with proportional widths
        current_x = x
        for i, word in enumerate(words):
            word_width = widths[i]
            
            # Create box for this word [x0, y0, x1, y1] in pixel coordinates
            word_box = [
                current_x,
                y, 
                current_x + word_width,
                y + height
            ]
            
            # Normalize to 0-1000 range
            normalized_box = [
                int(1000 * word_box[0] / page_width),
                int(1000 * word_box[1] / page_height),
                int(1000 * word_box[2] / page_width),
                int(1000 * word_box[3] / page_height)
            ]
            
            tokens.append(word)
            bboxes.append(normalized_box)
            
            # Update x position for next word
            current_x += word_width
    
    return tokens, bboxes

# Load OCR pages
ocr_pages = json.loads(Path(OCR_JSON).read_text(encoding="utf-8"))

# Extract tokens and bounding boxes from OCR data
pages_ocr_tokens = []
pages_ocr_bboxes = []
for i, page in enumerate(ocr_pages, start=1):
    tokens, bboxes = extract_tokens_and_bboxes(page, i)
    pages_ocr_tokens.append(tokens)
    pages_ocr_bboxes.append(bboxes)

# Read all tokens and labels from the CONLL file
all_conll_tokens = []
all_conll_tags = []

for raw in Path(CONLL).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    
    # If empty line or DOCSTART marker, skip
    if not line or line.startswith("-DOCSTART-"):
        continue
    
    # Parse token line - now properly handling labels with spaces
    parts = line.split()
    if len(parts) >= 4:  # Token, _, _, Label (Label may have spaces)
        token = parts[0]
        
        # Extract the full label including any spaces
        # The label starts at position 3 and goes to the end
        tag_parts = parts[3:]
        
        # Check if this is a B- or I- prefix that needs to be preserved
        prefix = ""
        if tag_parts[0].startswith("B-") or tag_parts[0].startswith("I-"):
            prefix = tag_parts[0][:2]  # Get the B- or I- prefix
            tag_parts[0] = tag_parts[0][2:]  # Remove prefix from first part
        
        # Join all parts of the tag and add prefix back if it existed
        tag = " ".join(tag_parts)
        if prefix:
            tag = prefix + tag
        
        all_conll_tokens.append(token)
        all_conll_tags.append(tag)

# Count total tokens in OCR data for reference
total_ocr_tokens = sum(len(tokens) for tokens in pages_ocr_tokens)
print(f"Total OCR tokens across all pages: {total_ocr_tokens}")
print(f"Total CONLL tokens: {len(all_conll_tokens)}")

# Split CONLL tokens into pages
# Strategy: We'll use the known transition point, looking for the token "o" that starts page 2
page_breakpoint_token = "o"  # First token on page 2
page_breakpoint_idx = -1

# Find the transition between page 1 and page 2
# We'll use a heuristic - look for the token that starts page 2
for i, token in enumerate(all_conll_tokens):
    if token == page_breakpoint_token and i > len(pages_ocr_tokens[0]) // 2:  # Only look in second half
        print(f"Found potential page break at index {i}, token: '{token}'")
        # Check surrounding tokens to confirm this is the right spot
        context = all_conll_tokens[i:i+5]
        if " ".join(context[:3]) == "o lawfully known":
            page_breakpoint_idx = i
            print(f"Confirmed page break at token {i}: {' '.join(context)}")
            break

if page_breakpoint_idx == -1:
    print("Warning: Could not find page transition. Treating all tokens as one page.")
    pages_conll_tokens = [all_conll_tokens]
    pages_conll_tags = [all_conll_tags]
else:
    # Split tokens and tags at the breakpoint
    pages_conll_tokens = [
        all_conll_tokens[:page_breakpoint_idx],
        all_conll_tokens[page_breakpoint_idx:]
    ]
    pages_conll_tags = [
        all_conll_tags[:page_breakpoint_idx],
        all_conll_tags[page_breakpoint_idx:]
    ]

# Create label map (string labels to integer IDs)
# First collect all unique labels
all_labels = set()
for tags in pages_conll_tags:
    all_labels.update(tags)

# Sort labels so "O" is first, followed by all B- labels, then all I- labels
sorted_labels = sorted(all_labels, key=lambda x: (
    0 if x == "O" else (1 if x.startswith("B-") else 2),
    x
))

# Create label_id map
label_to_id = {label: i for i, label in enumerate(sorted_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Save label map
with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

print(f"Created label map with {len(label_to_id)} labels at {LABEL_MAP_FILE}")

# Build LayoutLM dataset
layoutlm_dataset = []

for i, (ocr_tokens, ocr_bboxes, conll_tokens, conll_tags) in enumerate(
    zip(pages_ocr_tokens, pages_ocr_bboxes, pages_conll_tokens, pages_conll_tags), 
    start=1
):
    print(f"Page {i}: OCR tokens={len(ocr_tokens)}, CONLL tokens={len(conll_tokens)}")
    
    # Check token count match
    if len(ocr_tokens) != len(conll_tokens):
        print(f"Warning: Token count mismatch on page {i}")
        # Use the shorter length to avoid index errors
        min_length = min(len(ocr_tokens), len(conll_tokens))
        ocr_tokens = ocr_tokens[:min_length]
        ocr_bboxes = ocr_bboxes[:min_length]
        conll_tokens = conll_tokens[:min_length]
        conll_tags = conll_tags[:min_length]
    
    # Check token alignment
    token_mismatches = []
    for j, (ocr_token, conll_token) in enumerate(zip(ocr_tokens, conll_tokens)):
        if ocr_token.lower() != conll_token.lower():
            token_mismatches.append((j, ocr_token, conll_token))
    
    if token_mismatches:
        print(f"Token mismatches on page {i} ({len(token_mismatches)} total):")
        for j, ocr_token, conll_token in token_mismatches[:10]:  # Show first 10
            print(f"  Position {j+1}: OCR='{ocr_token}' vs CONLL='{conll_token}'")
        if len(token_mismatches) > 10:
            print(f"  ... and {len(token_mismatches) - 10} more")
    
    # Build example in correct LayoutLM format
    # Convert string tags to integer IDs using label map
    ner_tags = [label_to_id[tag] for tag in conll_tags]
    
    # Verify lengths match
    assert len(ocr_tokens) == len(ocr_bboxes) == len(ner_tags), \
        f"Mismatch in lengths: words={len(ocr_tokens)}, boxes={len(ocr_bboxes)}, ner_tags={len(ner_tags)}"
    
    layoutlm_dataset.append({
        "id": f"doc_{i}",
        "words": ocr_tokens,
        "boxes": ocr_bboxes,
        "ner_tags": ner_tags
    })

# Write dataset
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    json.dump(layoutlm_dataset, f, ensure_ascii=False, indent=2)

# Final verification of dataset
for i, example in enumerate(layoutlm_dataset):
    assert len(example["words"]) == len(example["boxes"]) == len(example["ner_tags"]), \
        f"Example {i} has mismatched lengths: words={len(example['words'])}, boxes={len(example['boxes'])}, ner_tags={len(example['ner_tags'])}"
    
    # Check that all boxes are properly normalized to 0-1000
    for j, box in enumerate(example["boxes"]):
        for k, coord in enumerate(box):
            assert 0 <= coord <= 1000, f"Example {i}, box {j}, coordinate {k} is out of bounds: {coord}"

print(f"✓ All verification checks passed!")
print(f"✔ Created LayoutLM dataset with {len(layoutlm_dataset)} documents at {OUT_JSONL} in the correct format for fine-tuning")
