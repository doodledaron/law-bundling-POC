import os
import json
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


@dataclass
class DocumentExample:
    """Class for keeping track of an example for document understanding tasks."""
    image_path: str
    words: List[str]
    bboxes: List[List[int]]
    labels: List[int]
    original_annotation: Dict = None


class CUADDataset(Dataset):
    """Custom dataset for CUAD document understanding."""
    
    def __init__(
        self,
        data_dir: str,
        processor: LayoutLMv3Processor,
        max_seq_length: int = 512,
        is_training: bool = True
    ):
        """
        Initialize a CUAD dataset.
        
        Args:
            data_dir: Directory containing images and annotations.
            processor: LayoutLMv3Processor for tokenization and feature extraction.
            max_seq_length: Maximum sequence length.
            is_training: Whether the dataset is used for training.
        """
        self.data_dir = data_dir
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        
        self.examples = self._get_examples()
        
    def _get_examples(self) -> List[DocumentExample]:
        """Load examples from the data directory."""
        examples = []
        
        # Find all JSON annotation files
        for file_name in os.listdir(self.data_dir):
            if not file_name.endswith('.json') or file_name == 'label_map.json':
                continue
            
            # Get corresponding image file
            json_path = os.path.join(self.data_dir, file_name)
            image_file = file_name.replace('.json', '.jpg')
            image_path = os.path.join(self.data_dir, image_file)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found for {json_path}")
                continue
            
            # Load annotations
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            if not annotations:
                continue
            
            # Extract word-level information
            words = []
            bboxes = []
            labels = []
            
            for annotation in annotations:
                words.append(annotation['text'])
                bboxes.append(annotation['bbox'])
                labels.append(annotation['label'])
            
            examples.append(
                DocumentExample(
                    image_path=image_path,
                    words=words,
                    bboxes=bboxes,
                    labels=labels,
                    original_annotation=annotations
                )
            )
        
        print(f"Loaded {len(examples)} examples from {self.data_dir}")
        return examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example.
            
        Returns:
            Dict containing input features for the model.
        """
        example = self.examples[idx]
        
        # Load and preprocess the image
        image = Image.open(example.image_path).convert("RGB")
        
        # Encode the inputs using the processor
        # We assume OCR is disabled in the processor
        encoding = self.processor(
            images=image,
            text=example.words,
            boxes=example.bboxes,
            word_labels=example.labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                encoding[k] = v.squeeze(0)
        
        return encoding


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    processor: LayoutLMv3Processor,
    batch_size: int,
    max_seq_length: int = 512,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_dir: Directory containing training data.
        val_dir: Directory containing validation data.
        processor: LayoutLMv3Processor for tokenization and feature extraction.
        batch_size: Batch size for data loaders.
        max_seq_length: Maximum sequence length.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = CUADDataset(
        data_dir=train_dir,
        processor=processor,
        max_seq_length=max_seq_length,
        is_training=True
    )
    
    val_dataset = CUADDataset(
        data_dir=val_dir,
        processor=processor,
        max_seq_length=max_seq_length,
        is_training=False
    )
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader 