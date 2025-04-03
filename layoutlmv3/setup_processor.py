#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, AutoTokenizer

def setup_processor(output_dir, model_name="microsoft/layoutlmv3-base", apply_ocr=False):
    """
    Set up and save a LayoutLMv3Processor with consistent settings.
    
    Args:
        output_dir: Directory to save the processor.
        model_name: Base model name.
        apply_ocr: Whether to apply OCR during processing.
    """
    print(f"Setting up LayoutLMv3Processor with {model_name}")
    print(f"OCR setting: {'enabled' if apply_ocr else 'disabled'}")
    
    # Create tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=apply_ocr)
    
    # Create processor
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processor
    processor.save_pretrained(output_dir)
    print(f"Processor saved to {output_dir}")
    
    # Create a settings file to record OCR setting
    with open(os.path.join(output_dir, "processor_settings.txt"), "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"apply_ocr: {apply_ocr}\n")
    
    return processor

def main():
    parser = argparse.ArgumentParser(description="Set up LayoutLMv3Processor")
    parser.add_argument("--output_dir", type=str, default="layoutlmv3/processor",
                        help="Directory to save the processor")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base",
                        help="Base model name")
    parser.add_argument("--apply_ocr", action="store_true",
                        help="Whether to apply OCR during processing")
    parser.add_argument("--disable_ocr", action="store_true", default=True,
                        help="Disable OCR (overrides apply_ocr)")
    
    args = parser.parse_args()
    
    # If --disable_ocr is set, override apply_ocr
    if args.disable_ocr:
        args.apply_ocr = False
    
    setup_processor(args.output_dir, args.model_name, args.apply_ocr)

if __name__ == "__main__":
    main() 