# LayoutLMv3 for Legal Document Analysis

## Installation

You can install the required dependencies using the provided requirements.txt file:

```bash
pip install -r layoutlmv3/requirements.txt
```

Or manually install the packages:

```bash
pip install torch transformers pdf2image pillow matplotlib opencv-python tqdm scikit-learn pandas paddlepaddle paddleocr
```

## Usage

### 1. Prepare Dataset

```bash
python layoutlmv3/prepare_dataset.py
```

### 2. Train Model

```bash
python layoutlmv3/train.py --num_epochs 5 --batch_size 2 --learning_rate 5e-5
```

Parameters:

- `--dataset_dir`: Dataset directory (default: `CUAD_v1/layoutlmv3_dataset`)
- `--output_dir`: Model save directory (default: `layoutlmv3/model`)
- `--model_name`: Base model (default: `microsoft/layoutlmv3-base`)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_epochs`: Training epochs (default: 10)
- `--device`: Training device (default: `cuda` if available)
- `--num_workers`: Data loader workers (default: 0)

### 3. Run Inference

```bash
python layoutlmv3/inference.py --pdf_path path/to/your/document.pdf
```

Parameters:

- `--model_dir`: Model directory (default: `layoutlmv3/model/best_model`)
- `--processor_dir`: Processor directory (default: `layoutlmv3/model`)
- `--pdf_path`: PDF to analyze (required)
- `--output_dir`: Results directory (default: `layoutlmv3/results`)
- `--confidence_threshold`: Detection threshold (default: 0.5)
- `--num_workers`: Data loader workers (default: 0)

### 4. Troubleshooting

If you encounter issues, run the diagnostic script to check your environment:

```bash
python layoutlmv3/debug_environment.py
```

This script will check:

- Required package installations
- CUDA availability and GPU information
- Dataset and model file status
- Disk space and permissions
- PDF processing libraries

The script will provide a summary of any issues detected and suggestions for fixing them.

#### Common Errors

1. **OCR-related errors**: If you see errors like `ValueError: You cannot provide bounding boxes if you initialized the image processor with apply_ocr set to True` or similar, the issue is related to OCR settings. The fix:

   ```bash
   # First, make sure the processor is initialized with OCR disabled
   python layoutlmv3/setup_processor.py --output_dir layoutlmv3/processor --disable_ocr

   # Then use this processor in your training
   python layoutlmv3/train.py --processor_dir layoutlmv3/processor
   ```

2. **CUDA errors**: If you see errors about CUDA not being available, follow the Memory Issues guidance below.

### 6. Memory Issues

LayoutLMv3 requires substantial GPU memory due to its multimodal nature (processing text, layout, and image information). If you encounter out of memory errors:

1. **Reduce batch size**: Use `--batch_size 1` to minimize memory usage

   ```bash
   python layoutlmv3/train.py --batch_size 1
   ```

2. **Use CPU instead of GPU**: Set `--device cpu` if you don't have enough GPU memory

   ```bash
   python layoutlmv3/train.py --device cpu
   ```

3. **Reduce number of workers**: Set `--num_workers 0` to eliminate parallel processing overhead

   ```bash
   python layoutlmv3/train.py --num_workers 0
   ```

4. **Common errors**:
   - `TypeError: LayoutLMv3Processor.__call__() missing 1 required positional argument: 'images'` - This indicates a parameter name mismatch in the processor. Make sure the processor is called with `images` instead of `image`.
   - `CUDA out of memory` - Reduce batch size and use fewer workers
   - `DataLoader worker error` - Reduce num_workers to 0

The pipeline scripts have been updated with defaults optimized for lower memory usage.

## Overview

LayoutLMv3 is a multimodal model that combines text, layout, and visual information to understand document structure. In this project, we fine-tune LayoutLMv3 to extract and classify legal clauses in contracts.

The pipeline consists of three main steps:

1. **Preprocessing**: Convert CUAD annotations to LayoutLMv3 format
2. **Training**: Fine-tune LayoutLMv3 for token classification
3. **Inference**: Apply the model to new legal documents

## Dataset Preparation

The dataset preparation script converts the CUAD annotations (which we've already enhanced with spatial coordinates) into the format required by LayoutLMv3.

```bash
python layoutlmv3/prepare_dataset.py
```

This script will:

1. Split the data into train, validation, and test sets
2. Convert PDF pages to images
3. Create annotation files in LayoutLMv3 format
4. Save the prepared dataset to `CUAD_v1/layoutlmv3_dataset`

## Training

To fine-tune LayoutLMv3 on the prepared dataset:

```bash
python layoutlmv3/train.py --num_epochs 5 --batch_size 2 --learning_rate 5e-5
```

Training parameters can be customized:

- `--dataset_dir`: Directory containing the prepared dataset (default: `CUAD_v1/layoutlmv3_dataset`)
- `--output_dir`: Directory to save the fine-tuned model (default: `layoutlmv3/model`)
- `--model_name`: Pretrained model name or path (default: `microsoft/layoutlmv3-base`)
- `--batch_size`: Batch size for training and evaluation (default: 4)
- `--learning_rate`: Learning rate for optimizer (default: 5e-5)
- `--num_epochs`: Number of training epochs (default: 10)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--warmup_ratio`: Ratio of warmup steps (default: 0.1)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--device`: Device to use for training (default: `cuda` if available, otherwise `cpu`)

## Inference

To apply the fine-tuned model to new legal documents:

```bash
python layoutlmv3/inference.py --pdf_path path/to/your/document.pdf
python .\layoutlmv3\inference.py --pdf_path CUAD_v1/full_contract_pdf_Part_I/Co_Branding/2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.pdf
```

Inference parameters:

- `--model_dir`: Directory containing the fine-tuned model (default: `layoutlmv3/model/best_model`)
- `--processor_dir`: Directory containing the processor (default: `layoutlmv3/model`)
- `--pdf_path`: Path to the PDF document to analyze (required)
- `--output_dir`: Directory to save the results (default: `layoutlmv3/results`)
- `--device`: Device to use for inference (default: `cuda` if available, otherwise `cpu`)
- `--dpi`: DPI for PDF to image conversion (default: 300)
- `--confidence_threshold`: Threshold for token classification confidence (default: 0.5)

The inference script will:

1. Convert the PDF to images
2. Apply the fine-tuned model to each page
3. Visualize the results with bounding boxes
4. Extract and categorize legal clauses
5. Generate a summary document with all identified clauses

## Folder Structure

```
layoutlmv3/
├── annotate.py              # Convert CUAD annotations to include spatial coordinates
├── prepare_dataset.py       # Prepare the dataset for LayoutLMv3
├── dataset.py               # Custom dataset class for LayoutLMv3
├── train.py                 # Fine-tune LayoutLMv3
├── inference.py             # Apply the model to new documents
├── run_pipeline.sh          # Bash script for running the entire pipeline
├── run_pipeline.ps1         # PowerShell script for running the entire pipeline
├── model/                   # Directory for the fine-tuned model
├── results/                 # Directory for inference results
└── README.md                # This file

CUAD_v1/
├── layoutlmv3/              # Directory with enhanced annotations
├── layoutlmv3_dataset/      # Directory with prepared dataset
│   ├── train/               # Training data
│   ├── val/                 # Validation data
│   └── test/                # Test data
└── full_contract_pdf/       # Original PDF documents
```

## Acknowledgements

This project uses the following open-source tools and datasets:

- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) from Microsoft
- [CUAD dataset](https://www.atticusprojectai.org/cuad) for legal document annotations
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for optical character recognition
- [Transformers](https://huggingface.co/transformers/) library from Hugging Face
