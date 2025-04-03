#!/usr/bin/env python
"""
Script to run LayoutLMv3 inference on a PDF document.
This wrapper handles spaces in filenames and other common issues.
"""

import os
import sys
import argparse
import shlex
import subprocess
import shutil
import tempfile

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run LayoutLMv3 inference on a PDF document")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document to analyze")
    parser.add_argument("--model_dir", type=str, default="layoutlmv3/model/best_model",
                      help="Directory containing the fine-tuned model")
    parser.add_argument("--processor_dir", type=str, default="layoutlmv3/model",
                      help="Directory containing the processor")
    parser.add_argument("--output_dir", type=str, default="layoutlmv3/results",
                      help="Directory to save the results")
    parser.add_argument("--confidence", type=float, default=0.3,
                      help="Confidence threshold for predictions (default: 0.3)")
    parser.add_argument("--dpi", type=int, default=300,
                      help="DPI for PDF to image conversion (default: 300)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create a temporary copy of the PDF if the filename contains spaces
    pdf_path = args.pdf_path
    temp_file = None
    
    if ' ' in pdf_path:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_name = f"layoutlmv3_temp_{os.path.basename(pdf_path).replace(' ', '_')}"
        temp_file = os.path.join(temp_dir, temp_name)
        
        print(f"Creating temporary file without spaces: {temp_file}")
        try:
            shutil.copy2(pdf_path, temp_file)
            pdf_path = temp_file
        except Exception as e:
            print(f"Error creating temporary file: {e}")
            return 1
    
    # Construct the command for the inference script
    inference_script = os.path.join("layoutlmv3", "inference.py")
    cmd = [
        sys.executable,
        inference_script,
        "--pdf_path", pdf_path,
        "--model_dir", args.model_dir,
        "--processor_dir", args.processor_dir,
        "--output_dir", args.output_dir,
        "--confidence_threshold", str(args.confidence),
        "--dpi", str(args.dpi)
    ]
    
    # Run the inference script
    print(f"Running LayoutLMv3 inference on {args.pdf_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Inference completed with exit code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        return e.returncode
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")
            except Exception as e:
                print(f"Error removing temporary file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 