import os
import json
import argparse
import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    AdamW,
    get_scheduler
)
from sklearn.metrics import precision_recall_fscore_support

from dataset import create_data_loaders


def parse_args():
    # Check if CUDA is actually available
    cuda_available = torch.cuda.is_available()
    if not cuda_available and torch.backends.cuda.is_built():
        print("WARNING: CUDA is not available even though PyTorch detects CUDA installation.")
        print("This might be due to a driver issue or insufficient GPU memory.")
    elif not torch.backends.cuda.is_built():
        print("WARNING: PyTorch was not compiled with CUDA enabled.")
        print("Using CPU for training, which will be much slower.")

    default_device = "cuda" if cuda_available else "cpu"
    
    # Create a timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f"layoutlmv3/model/train-{timestamp}"
    
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on CUAD dataset")
    parser.add_argument("--dataset_dir", type=str, default="CUAD_v1/layoutlmv3_dataset", 
                        help="Directory containing the prepared dataset")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base", 
                        help="Pretrained model name or path")
    parser.add_argument("--processor_dir", type=str, default=None,
                        help="Directory containing a pre-configured processor (optional)")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                        help="Ratio of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay for optimizer")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="Log metrics every X steps")
    parser.add_argument("--eval_steps", type=int, default=100, 
                        help="Evaluate every X steps")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=default_device, 
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of worker processes for data loading")
    
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_labels: int
) -> Dict[str, float]:
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The model to evaluate.
        eval_dataloader: DataLoader for evaluation data.
        device: Device to use for evaluation.
        num_labels: Number of label classes.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=2)
        
        # Remove padding and special tokens
        for i, label in enumerate(labels):
            mask = (label != -100)
            true_label = label[mask]
            pred = preds[i][mask]
            
            all_labels.extend(true_label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    
    # Calculate metrics (ignore label 0 which is "O")
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", labels=list(range(1, num_labels))
    )
    
    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return results


def train(args):
    """
    Fine-tune LayoutLMv3 on the CUAD dataset.
    
    Args:
        args: Command line arguments.
    """
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create log file
    log_file_path = os.path.join(args.output_dir, "training_log.txt")
    log_file = open(log_file_path, "w")
    
    def log_message(message):
        """Write message to both console and log file"""
        print(message)
        log_file.write(f"{message}\n")
        log_file.flush()
    
    # Log training parameters
    log_message(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Parameters: {vars(args)}")
    
    # Validate device setting
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        log_message("WARNING: CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Load label map
    with open(os.path.join(args.dataset_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
    num_labels = max(int(v) for v in label_map.values()) + 1
    
    # Initialize processor and model
    log_message(f"Loading model {args.model_name}...")
    
    if args.processor_dir:
        log_message(f"Loading processor from {args.processor_dir}...")
        processor = LayoutLMv3Processor.from_pretrained(args.processor_dir)
    else:
        log_message("Creating new processor with OCR disabled...")
        processor = LayoutLMv3Processor.from_pretrained(
            args.model_name,
            apply_ocr=False  # Disable OCR since we have our own annotations
        )
    
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Save processor and label map
    log_message(f"Saving processor to {args.output_dir}...")
    processor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    
    # Set device
    device = torch.device(args.device)
    log_message(f"Using device: {device}")
    model.to(device)
    
    # Create data loaders
    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir = os.path.join(args.dataset_dir, "val")
    
    log_message(f"Creating data loaders (batch_size={args.batch_size}, num_workers={args.num_workers})...")
    train_dataloader, eval_dataloader = create_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        processor=processor,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = args.num_epochs * len(train_dataloader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    global_step = 0
    best_f1 = 0.0
    
    log_message(f"Starting training with {num_training_steps} steps...")
    
    # Create metrics history for plotting later
    metrics_history = {
        "steps": [],
        "loss": [],
        "eval_f1": [],
        "eval_precision": [],
        "eval_recall": []
    }
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        log_message(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for batch in tqdm(train_dataloader, desc=f"Training (Epoch {epoch + 1})"):
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Log training progress
            if global_step % args.save_steps == 0:
                current_loss = epoch_loss / (batch_idx + 1 if epoch == 0 else len(train_dataloader) * epoch + batch_idx + 1)
                log_message(f"Step {global_step}/{num_training_steps} - Loss: {current_loss:.4f}")
                metrics_history["steps"].append(global_step)
                metrics_history["loss"].append(current_loss)
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                log_message(f"\nEvaluating at step {global_step}...")
                results = evaluate(model, eval_dataloader, device, num_labels)
                log_message(f"Evaluation results: {results}")
                
                # Track metrics
                metrics_history["eval_f1"].append(results["f1"])
                metrics_history["eval_precision"].append(results["precision"])
                metrics_history["eval_recall"].append(results["recall"])
                
                # Save if best model
                if results["f1"] > best_f1:
                    best_f1 = results["f1"]
                    log_message(f"New best F1: {best_f1:.4f}, saving model...")
                    model.save_pretrained(os.path.join(args.output_dir, "best_model"))
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        log_message(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate at the end of each epoch
        log_message(f"Evaluating after epoch {epoch + 1}...")
        results = evaluate(model, eval_dataloader, device, num_labels)
        log_message(f"Evaluation results: {results}")
        
        # Save if best model
        if results["f1"] > best_f1:
            best_f1 = results["f1"]
            log_message(f"New best F1: {best_f1:.4f}, saving model...")
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
    
    # Save final model
    log_message("Training complete. Saving final model...")
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Save metrics history
    with open(os.path.join(args.output_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    log_message(f"Best F1: {best_f1:.4f}")
    log_message(f"Model saved to {args.output_dir}")
    log_message(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Close log file
    log_file.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)