import os
import json
import argparse
import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
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
    parser.add_argument("--checkpoint_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=default_device, 
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of worker processes for data loading")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from this checkpoint directory")
    
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    batch_idx: int,
    metrics_history: Dict,
    best_f1: float,
    args,
    output_dir: str
) -> None:
    """
    Save training checkpoint for resumable training.
    
    This function creates a checkpoint that contains everything needed to resume training
    from exactly where it left off. It saves:
    
    1. Model weights and configuration
    2. Optimizer state (learning rates, momentum, etc.)
    3. Scheduler state (for learning rate scheduling)
    4. Current epoch, step, and batch position
    5. Training metrics and best F1 score so far
    6. All training arguments for consistency
    
    The checkpoint is saved in a subdirectory of the output directory, and a pointer
    to the latest checkpoint is updated.
    
    Args:
        model: The model being trained
        optimizer: The optimizer (contains learning rates, momentum buffers, etc.)
        scheduler: The learning rate scheduler
        epoch: Current epoch number (0-indexed)
        global_step: Global training step (total batches processed)
        batch_idx: Current position within the epoch (to resume mid-epoch)
        metrics_history: Dictionary of recorded training metrics
        best_f1: Current best F1 score achieved during training
        args: Command line arguments used for training
        output_dir: Directory to save the checkpoint
    """
    # Create a directory for this specific checkpoint
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights and configuration
    model.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler states, along with training metadata
    torch.save({
        'optimizer': optimizer.state_dict(),  # Learning rates, momentum buffers, etc.
        'scheduler': scheduler.state_dict() if scheduler else None,  # LR scheduler state
        'epoch': epoch,  # Current epoch
        'global_step': global_step,  # Total steps processed so far
        'batch_idx': batch_idx,  # Current position in epoch (for mid-epoch resuming)
        'metrics_history': metrics_history,  # Training metrics (loss, F1, etc.)
        'best_f1': best_f1,  # Best F1 score so far
        'args': vars(args)  # All training arguments for consistency
    }, os.path.join(checkpoint_dir, "training_state.pt"))
    
    # Save a pointer to the latest checkpoint for easy resuming
    with open(os.path.join(output_dir, "latest_checkpoint.txt"), "w") as f:
        f.write(f"checkpoint-{global_step}")
    
    print(f"Checkpoint saved at step {global_step}")


def load_checkpoint(
    checkpoint_dir: str,
    device: torch.device
) -> Tuple[nn.Module, torch.optim.Optimizer, object, int, int, int, Dict, float, Dict]:
    """
    Load training checkpoint for resumable training.
    
    This function restores the complete training state from a checkpoint, including:
    
    1. Model weights and configuration
    2. Optimizer state (learning rates, momentum buffers)
    3. Scheduler state (learning rate schedule)
    4. Training progress (epoch, step, batch position)
    5. Training metrics and best F1 score
    6. Original training arguments
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        device: Device to load model and tensors onto (cuda/cpu)
        
    Returns:
        Tuple containing:
        - model: The restored model with weights
        - optimizer: Optimizer with restored state
        - scheduler: LR scheduler with restored state (or None)
        - epoch: Current epoch to resume from
        - global_step: Global step count
        - batch_idx: Batch index within the epoch
        - metrics_history: Dictionary of training metrics
        - best_f1: Best F1 score achieved so far
        - args_dict: Original training arguments as dictionary
    """
    print(f"Loading checkpoint from {checkpoint_dir}...")
    
    # Load model weights and configuration
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    
    # Load optimizer, scheduler, and other training state
    training_state = torch.load(
        os.path.join(checkpoint_dir, "training_state.pt"),
        map_location=device
    )
    
    # Extract individual components from saved state
    optimizer_state = training_state['optimizer']  # Optimizer state dict
    scheduler_state = training_state['scheduler']  # Scheduler state dict
    epoch = training_state['epoch']                # Current epoch
    global_step = training_state['global_step']    # Current global step
    batch_idx = training_state['batch_idx']        # Current batch within epoch
    metrics_history = training_state['metrics_history']  # Training metrics
    best_f1 = training_state['best_f1']            # Best F1 score so far
    args_dict = training_state['args']             # Original training arguments
    
    # Recreate optimizer with same parameters
    optimizer = AdamW(
        model.parameters(),
        lr=args_dict['learning_rate'],
        weight_decay=args_dict['weight_decay']
    )
    # Restore optimizer state (learning rates, momentum buffers, etc.)
    optimizer.load_state_dict(optimizer_state)
    
    # Recreate and restore scheduler if it was saved
    scheduler = None
    if scheduler_state:
        # Estimate training steps based on current progress
        num_training_steps = args_dict['num_epochs'] * (global_step // epoch if epoch > 0 else 100)
        num_warmup_steps = int(args_dict['warmup_ratio'] * num_training_steps)
        
        # Recreate scheduler with same parameters
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        # Restore scheduler state
        scheduler.load_state_dict(scheduler_state)
    
    print(f"Checkpoint loaded: Epoch {epoch+1}, Step {global_step}")
    
    return (model, optimizer, scheduler, epoch, global_step, batch_idx, 
            metrics_history, best_f1, args_dict)

def save_training_visualizations(metrics_history, log_path, output_dir, start_time):
    """
    Generate and save visualizations of training metrics.
    
    Args:
        metrics_history: Dictionary containing training metrics
        log_path: Path to the training log file
        output_dir: Directory to save the visualizations
        start_time: Training start time for labeling
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract information from log file
    training_params = {}
    steps_per_epoch = None
    num_epochs = None
    
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if "Parameters:" in line:
                    try:
                        params_str = line.split("Parameters:")[1].strip()
                        params = eval(params_str)
                        num_epochs = params.get('num_epochs', 25)
                        batch_size = params.get('batch_size', 2)
                        training_params = params
                    except:
                        pass
                
                # Look for total steps information
                if "Starting" in line and "training with" in line and "steps" in line:
                    try:
                        total_steps = int(line.split("with ")[1].split(" ")[0])
                        if num_epochs:
                            steps_per_epoch = total_steps / num_epochs
                    except:
                        pass
    
    # Extract training completion time
    completion_time = None
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "Training completed at:" in line:
                completion_time = line.strip().split("Training completed at:")[1].strip()
                break
    
    # If we couldn't determine steps_per_epoch, estimate from steps
    if not steps_per_epoch and 'steps' in metrics_history and metrics_history['steps']:
        steps_per_epoch = max(metrics_history['steps']) / num_epochs if num_epochs else 80
    
    # Convert steps to epochs for plotting
    if 'steps' in metrics_history and metrics_history['steps']:
        steps_as_epochs = [step / steps_per_epoch for step in metrics_history['steps']]
    else:
        steps_as_epochs = []
    
    # Extract epoch average losses
    epochs = []
    epoch_loss = []
    with open(log_path, "r") as f:
        for line in f:
            if "Epoch " in line and "average loss:" in line:
                try:
                    # The line format: "Epoch X average loss: Y.YYYY"
                    epoch_part = line.split("Epoch ")[1].split(" average")[0]
                    epoch_num = int(epoch_part)
                    loss_part = line.split("average loss: ")[1]
                    avg_loss = float(loss_part)
                    epochs.append(epoch_num)
                    epoch_loss.append(avg_loss)
                except Exception as e:
                    print(f"Error parsing epoch loss: {line.strip()} - {str(e)}")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'Training Metrics for {os.path.basename(output_dir)}', fontsize=16)
    
    # 1. Training Loss vs Epochs
    if metrics_history.get('steps') and metrics_history.get('loss'):
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(steps_as_epochs, metrics_history['loss'], 'b-', label='Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss vs Epochs')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Evaluation Metrics vs Epochs
    if metrics_history.get('steps') and metrics_history.get('eval_f1') and len(metrics_history['steps']) == len(metrics_history['eval_f1']):
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(steps_as_epochs, metrics_history['eval_precision'], 'r-', label='Precision')
        ax2.plot(steps_as_epochs, metrics_history['eval_recall'], 'g-', label='Recall')
        ax2.plot(steps_as_epochs, metrics_history['eval_f1'], 'b-', label='F1 Score')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Score')
        ax2.set_title('Evaluation Metrics vs Epochs')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Epoch Average Loss
    if epochs and epoch_loss:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(epochs, epoch_loss, 'r-o', label='Epoch Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Average Loss')
        ax3.set_title('Average Loss per Epoch')
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only show integer ticks for epochs
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start from 0
        ax3.set_ylim(bottom=0)
    
    # 4. Training Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')  # No axes for this text display
    
    # Extract best F1 score
    best_f1 = 0
    if metrics_history.get('eval_f1'):
        best_f1 = max(metrics_history['eval_f1'])
    else:
        with open(log_path, "r") as f:
            for line in f:
                if "Best F1:" in line:
                    try:
                        best_f1 = float(line.split("Best F1:")[1].strip())
                    except:
                        pass
    
    # Create training summary
    summary_text = f"Training Summary:\n\n"
    summary_text += f"Training started at: {start_time}\n"
    if completion_time:
        summary_text += f"Training completed at: {completion_time}\n"
    
    # Add training parameters
    if training_params:
        summary_text += f"Batch Size: {training_params.get('batch_size', 'N/A')}\n"
        summary_text += f"Learning Rate: {training_params.get('learning_rate', 'N/A')}\n"
        summary_text += f"Epochs: {training_params.get('num_epochs', 'N/A')}\n"
        summary_text += f"Device: {training_params.get('device', 'N/A')}\n"
    
    summary_text += f"\nBest F1 Score: {best_f1:.4f}\n"
    
    if epochs:
        summary_text += f"Completed Epochs: {max(epochs)}\n"
    
    if metrics_history.get('steps'):
        summary_text += f"Total Steps: {max(metrics_history['steps'])}\n"
        summary_text += f"Steps per Epoch: {steps_per_epoch:.0f}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top', wrap=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
    
    # Save the figure
    plt.savefig(os.path.join(vis_dir, "training_metrics.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(vis_dir, "training_metrics.pdf"), format="pdf", bbox_inches="tight")
    
    # Close the figure to free memory
    plt.close(fig)
    
    print(f"Training visualizations saved to {vis_dir}")

def train(args):
    """
    Fine-tune LayoutLMv3 on the CUAD dataset.
    
    Args:
        args: Command line arguments.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Record training start time
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Create log file (append mode when resuming)
    log_file_path = os.path.join(args.output_dir, "training_log.txt")
    log_file_mode = "a" if args.resume_from else "w"
    log_file = open(log_file_path, log_file_mode)
    
    def log_message(message):
        """Write message to both console and log file"""
        print(message)
        log_file.write(f"{message}\n")
        log_file.flush()
    
    # Set device
    device = torch.device(args.device)
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        log_message("WARNING: CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
        device = torch.device("cpu")
    
    # Resume from checkpoint or start new training
    if args.resume_from:
        # Find latest checkpoint if directory is provided
        if os.path.isdir(args.resume_from) and os.path.exists(os.path.join(args.resume_from, "latest_checkpoint.txt")):
            with open(os.path.join(args.resume_from, "latest_checkpoint.txt"), "r") as f:
                latest_checkpoint = f.read().strip()
            checkpoint_dir = os.path.join(args.resume_from, latest_checkpoint)
        else:
            checkpoint_dir = args.resume_from
        
        log_message(f"Resuming training from checkpoint: {checkpoint_dir}")
        
        # Load checkpoint
        (model, optimizer, scheduler, start_epoch, global_step, resume_batch_idx,
         metrics_history, best_f1, checkpoint_args) = load_checkpoint(checkpoint_dir, device)
        
        # Update args with original values for consistency
        for key, value in checkpoint_args.items():
            if key not in ['output_dir', 'resume_from']:
                setattr(args, key, value)
        
        # Load label map
        with open(os.path.join(args.dataset_dir, "label_map.json"), "r") as f:
            label_map = json.load(f)
        num_labels = max(int(v) for v in label_map.values()) + 1
        
        # Load processor
        processor_dir = os.path.join(args.resume_from, "processor") if os.path.exists(os.path.join(args.resume_from, "processor")) else args.model_name
        processor = LayoutLMv3Processor.from_pretrained(processor_dir)
        
        log_message(f"Resumed training state: Epoch {start_epoch + 1}, Step {global_step}")
        log_message(f"Best F1 so far: {best_f1}")
    else:
        log_message(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"Parameters: {vars(args)}")
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
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
        processor_dir = os.path.join(args.output_dir, "processor")
        log_message(f"Saving processor to {processor_dir}...")
        processor.save_pretrained(processor_dir)
        with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)
        
        model.to(device)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Initialize training state
        start_epoch = 0
        global_step = 0
        resume_batch_idx = 0
        best_f1 = 0.0
        metrics_history = {
            "steps": [],
            "loss": [],
            "eval_f1": [],
            "eval_precision": [],
            "eval_recall": []
        }
    
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
    
    # Create or update scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    if not args.resume_from or scheduler is None:
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    log_message(f"Starting/Resuming training with {num_training_steps} total steps...")
    
    #--------------------------------------------------------------------
    # HOW TO USE CHECKPOINTING:
    #
    # 1. For initial training:
    #    python train.py --output_dir my_training_run
    #
    # 2. If training is interrupted, resume with:
    #    python train.py --resume_from my_training_run
    #
    # The script will automatically find the latest checkpoint and continue
    # training from exactly where it left off.
    #--------------------------------------------------------------------
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        log_message(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Convert train_dataloader to list for easy skipping when resuming
        train_batches = list(train_dataloader)
        
        # Skip batches if resuming
        start_batch_idx = resume_batch_idx if epoch == start_epoch and resume_batch_idx > 0 else 0
        # Reset resume_batch_idx after first epoch
        resume_batch_idx = 0
        
        # Use enumerate and start from resume point if needed
        for batch_idx, batch in enumerate(tqdm(train_batches[start_batch_idx:], 
                                              desc=f"Training (Epoch {epoch + 1})",
                                              initial=start_batch_idx,
                                              total=len(train_batches))):
            # Adjust batch_idx to account for skipped batches
            batch_idx = batch_idx + start_batch_idx
            
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
            
            # Save checkpoint
            if global_step % args.checkpoint_steps == 0:
                log_message(f"Saving checkpoint at step {global_step}...")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    batch_idx=batch_idx + 1,  # +1 to start from next batch when resuming
                    metrics_history=metrics_history,
                    best_f1=best_f1,
                    args=args,
                    output_dir=args.output_dir
                )
        
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
        
        # Save checkpoint at end of each epoch
        log_message(f"Saving checkpoint at end of epoch {epoch + 1}...")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,  # +1 because we'll start the next epoch
            global_step=global_step,
            batch_idx=0,  # 0 because we'll start from the beginning of next epoch
            metrics_history=metrics_history,
            best_f1=best_f1,
            args=args,
            output_dir=args.output_dir
        )
    
    # Save final model
    log_message("Training complete. Saving final model...")
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Save metrics history
    with open(os.path.join(args.output_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    log_message(f"Best F1: {best_f1:.4f}")
    log_message(f"Model saved to {args.output_dir}")
    log_message(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save visualizations after training completes
    log_message("Generating training visualizations...")
    save_training_visualizations(
        metrics_history=metrics_history,
        log_path=log_file_path,
        output_dir=args.output_dir,
        start_time=start_time
    )
    # Close log file
    log_file.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)