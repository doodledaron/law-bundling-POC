import json
import os
import argparse
from pathlib import Path
import re

def parse_conll(conll_file):
    """Parse CONLL file into documents with tokens and labels."""
    documents = []
    current_doc = []
    
    with open(conll_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # If empty line, skip
            if not line:
                continue
                
            # If document start marker
            if line.startswith('-DOCSTART-'):
                if current_doc:
                    documents.append(current_doc)
                    current_doc = []
                continue
                
            # Parse token line
            parts = line.split()
            if len(parts) >= 3:  # Token, _, _, Label
                token = parts[0]
                label = parts[-1]  # Last part is the label
                current_doc.append((token, label))
    
    # Add the last document
    if current_doc:
        documents.append(current_doc)
        
    return documents

def merge_conll_with_layoutlm(conll_file, layoutlm_file, output_file):
    """
    Merge CONLL annotations with LayoutLM data to create a dataset
    suitable for LayoutLMv3 fine-tuning.
    """
    # Parse CONLL file for token labels
    documents = parse_conll(conll_file)
    
    # Load LayoutLM data with bounding boxes
    with open(layoutlm_file, 'r', encoding='utf-8') as f:
        layoutlm_data = json.load(f)
    
    if len(documents) != len(layoutlm_data):
        raise ValueError(f"Number of documents in CONLL ({len(documents)}) doesn't match LayoutLM data ({len(layoutlm_data)})")
    
    # Merge token labels with bounding boxes
    layoutlm_dataset = []
    
    for doc_idx, (doc_tokens, layoutlm_doc) in enumerate(zip(documents, layoutlm_data)):
        layoutlm_tokens = layoutlm_doc["tokens"]
        
        if len(doc_tokens) != len(layoutlm_tokens):
            print(f"Warning: Document {doc_idx+1} has {len(doc_tokens)} tokens in CONLL but {len(layoutlm_tokens)} in LayoutLM data")
            # We'll try to align as many as possible
            
        # Create document with labeled tokens and bounding boxes
        labeled_tokens = []
        for i in range(min(len(doc_tokens), len(layoutlm_tokens))):
            conll_token, label = doc_tokens[i]
            layoutlm_token = layoutlm_tokens[i]
            
            if conll_token != layoutlm_token["text"]:
                print(f"Warning: Token mismatch in document {doc_idx+1}, position {i+1}:")
                print(f"  CONLL: '{conll_token}'")
                print(f"  LayoutLM: '{layoutlm_token['text']}'")
            
            labeled_tokens.append({
                "text": layoutlm_token["text"],
                "bbox": layoutlm_token["bbox"],
                "label": label
            })
        
        layoutlm_dataset.append({
            "id": f"doc_{doc_idx}",
            "image": layoutlm_doc["image"],
            "tokens": labeled_tokens
        })
    
    # Save the merged dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(layoutlm_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✔ Created LayoutLMv3 dataset with {len(layoutlm_dataset)} documents at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CONLL annotations to LayoutLMv3 dataset format")
    parser.add_argument("--conll", default="project-8-at-2025-04-28-18-28-c7133fe2.conll", help="Path to CONLL file with annotations")
    parser.add_argument("--layoutlm", default="layoutlm_data.json", help="Path to LayoutLM data with bounding boxes")
    parser.add_argument("--output", default="layoutlmv3_dataset.json", help="Output path for LayoutLMv3 dataset")
    
    args = parser.parse_args()
    
    merge_conll_with_layoutlm(args.conll, args.layoutlm, args.output) 