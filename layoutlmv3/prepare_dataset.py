import os
import json
import shutil
import random
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from pdf2image import convert_from_path

# Define paths
ANNOTATIONS_DIR = "CUAD_v1/layoutlmv3"
PDF_DIR = "CUAD_v1/full_contract_pdf"
OUTPUT_DIR = "CUAD_v1/layoutlmv3_dataset_part1"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Define the question/label categories in CUAD
QUESTION_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date", 
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions", 
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary"
]

# Map categories to integer labels
LABEL_MAP = {cat: idx+1 for idx, cat in enumerate(QUESTION_CATEGORIES)}
LABEL_MAP["O"] = 0  # Outside/background label

def extract_category_from_id(annotation_id):
    """Extract the question category from the annotation ID."""
    parts = annotation_id.split('_')
    # The category is usually the second-to-last part
    if len(parts) >= 2:
        return parts[-2]
    return None

def convert_bbox_to_normalized(bbox, page_width, page_height):
    """Convert absolute bbox coordinates to normalized coordinates (0-1000)."""
    x0, y0, x1, y1 = bbox
    
    # Normalize to 0-1000 range
    x0_norm = int(1000 * x0 / page_width)
    y0_norm = int(1000 * y0 / page_height)
    x1_norm = int(1000 * x1 / page_width)
    y1_norm = int(1000 * y1 / page_height)
    
    # Ensure values are within range
    x0_norm = max(0, min(1000, x0_norm))
    y0_norm = max(0, min(1000, y0_norm))
    x1_norm = max(0, min(1000, x1_norm))
    y1_norm = max(0, min(1000, y1_norm))
    
    return [x0_norm, y0_norm, x1_norm, y1_norm]

def process_annotation_file(json_path, split_dir, split):
    """Process a single annotation file and convert it to LayoutLMv3 format."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pdf_path = data['file_name']
        # <<< ADD THIS LINE TO ENSURE CORRECT PATH SEPARATORS >>>
        pdf_path = pdf_path.replace("\\", "/") 
        
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Extract annotations
        annotations = data['annotations']
        if not annotations:
            print(f"No annotations found in {json_path}")
            return
        
        # Group annotations by page number
        annotations_by_page = {}
        for annotation in annotations:
            page_num = annotation.get('page_number', 0)
            if page_num not in annotations_by_page:
                annotations_by_page[page_num] = []
            annotations_by_page[page_num].append(annotation)
        
        # Convert PDF pages to images if needed
        try:
            pages = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {str(e)}")
            return
        
        # Process each page
        for page_num, page_img in enumerate(pages):
            if page_num not in annotations_by_page:
                continue  # Skip pages without annotations
            
            page_annotations = annotations_by_page[page_num]
            
            # Save the page image
            img_filename = f"{file_name}_page_{page_num}.jpg"
            img_path = os.path.join(split_dir, img_filename)
            page_img.save(img_path, "JPEG")
            
            # Get image dimensions
            width, height = page_img.size
            
            # Prepare the annotations in LayoutLMv3 format
            layoutlm_annotations = []
            
            for annotation in page_annotations:
                category = extract_category_from_id(annotation.get('id', ''))
                if not category or category not in LABEL_MAP:
                    continue
                
                label_id = LABEL_MAP[category]
                
                # Process each word in the annotation
                for word in annotation.get('words', []):
                    bbox = word.get('bbox', [0, 0, 0, 0])
                    normalized_bbox = convert_bbox_to_normalized(bbox, width, height)
                    
                    layoutlm_annotations.append({
                        'text': word.get('text', ''),
                        'bbox': normalized_bbox,
                        'label': label_id,
                        'category': category
                    })
            
            # Save the annotations to a JSON file
            annotation_filename = f"{file_name}_page_{page_num}.json"
            annotation_path = os.path.join(split_dir, annotation_filename)
            
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(layoutlm_annotations, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")

def prepare_dataset():
    """Prepare the dataset by splitting files into train/val/test sets and converting to LayoutLMv3 format."""
    print("Preparing dataset for LayoutLMv3 fine-tuning...")
    
    # Get all JSON annotation files
    json_files = [os.path.join(ANNOTATIONS_DIR, f) for f in os.listdir(ANNOTATIONS_DIR) 
                  if f.endswith('_layoutlm.json')]
    
    if not json_files:
        print(f"No annotation files found in {ANNOTATIONS_DIR}")
        return
    
    # Shuffle and split the files (70% train, 15% val, 15% test)
    random.shuffle(json_files)
    num_files = len(json_files)
    train_size = int(0.7 * num_files)
    val_size = int(0.15 * num_files)
    
    train_files = json_files[:train_size]
    val_files = json_files[train_size:train_size + val_size]
    test_files = json_files[train_size + val_size:]
    
    print(f"Splitting dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    
    # Process each split
    print("Processing training set...")
    for json_file in tqdm(train_files):
        process_annotation_file(json_file, TRAIN_DIR, 'train')
    
    print("Processing validation set...")
    for json_file in tqdm(val_files):
        process_annotation_file(json_file, VAL_DIR, 'val')
    
    print("Processing test set...")
    for json_file in tqdm(test_files):
        process_annotation_file(json_file, TEST_DIR, 'test')
    
    # Create a label mapping file
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    print(f"Dataset preparation complete. Output saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_dataset()