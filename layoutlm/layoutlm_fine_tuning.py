"""
# LayoutLM Fine-Tuning for NER

This script provides a step-by-step process for fine-tuning LayoutLM on a token classification (NER) task
using the dataset prepared by merge-ocr-conll.py.
"""

# %% [markdown]
# ## 1. Install Required Libraries
# 
# Run this cell to install the required libraries if you haven't already.

# %%
# !pip install transformers datasets evaluate seqeval

# %% [markdown]
# ## 2. Import Libraries

# %%
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D
from transformers import (
    LayoutLMForTokenClassification, 
    LayoutLMTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 3. Load Dataset and Label Map

# %%
# Load the prepared dataset
with open('layoutlm_dataset.json', 'r', encoding='utf-8') as f:
    dataset_json = json.load(f)

# Load the label map
with open('label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)

label_to_id = label_map['label_to_id']
id_to_label = label_map['id_to_label']
num_labels = len(label_to_id)

print(f"Loaded dataset with {len(dataset_json)} examples")
print(f"Number of labels: {num_labels}")
print("Labels:")
for label, id in sorted(label_to_id.items(), key=lambda x: x[1]):
    print(f"  {id}: {label}")

# %% [markdown]
# ## 4. Create Hugging Face Dataset

# %%
# Define the features for our dataset
features = Features({
    'id': Value('string'),
    'words': Sequence(Value('string')),
    'boxes': Sequence(Sequence(Value('int64'), length=4)),
    'ner_tags': Sequence(ClassLabel(num_classes=num_labels, names=list(label_to_id.keys())))
})

# Create a dataset from the JSON
dataset = Dataset.from_list(dataset_json, features=features)

# Split dataset into train and eval sets (80/20 split)
# In a real scenario, you might want to use a more sophisticated split
# or have separate validation and test sets
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Train set: {len(dataset['train'])} examples")
print(f"Test set: {len(dataset['test'])} examples")

# %% [markdown]
# ## 5. Initialize TokenizerFast and Model

# %%
# Initialize the tokenizer and model
tokenizer = LayoutLMTokenizerFast.from_pretrained('microsoft/layoutlm-base-uncased')
model = LayoutLMForTokenClassification.from_pretrained(
    'microsoft/layoutlm-base-uncased',
    num_labels=num_labels
)

# %% [markdown]
# ## 6. Prepare the Tokenizer and Data Processing Function

# %%
# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"],
        boxes=examples["boxes"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_token_type_ids=True,
        return_attention_mask=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens have a word id that is None
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to -100 (ignored in loss)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization to the datasets
tokenized_train = dataset["train"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

tokenized_test = dataset["test"].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["test"].column_names
)

print(f"Keys in tokenized dataset: {list(tokenized_train.features.keys())}")
print(f"Example input_ids shape: {tokenized_train[0]['input_ids'][:10]}")

# %% [markdown]
# ## 7. Define Metrics and Evaluation Function

# %%
# Load the seqeval metric
seqeval = evaluate.load("seqeval")

# Define the compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[str(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[str(l)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# %% [markdown]
# ## 8. Define Training Arguments and Initialize Trainer

# %%
# Define training arguments
training_args = TrainingArguments(
    output_dir="./layoutlm-ner-results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize the data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 9. Train the Model

# %%
# Train the model
trainer.train()

# %% [markdown]
# ## 10. Evaluate the Model

# %%
# Evaluate the model
evaluation_results = trainer.evaluate()
print(f"Evaluation results: {evaluation_results}")

# %% [markdown]
# ## 11. Save the Fine-tuned Model

# %%
# Save the model
model_save_path = "./layoutlm-ner-finetuned"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# %% [markdown]
# ## 12. Test the Model on a Sample

# %%
def predict_sample(sample_idx=0, dataset_split="test"):
    """Test the model on a sample from the dataset."""
    # Get a sample from the test set
    sample = dataset[dataset_split][sample_idx]
    
    # Tokenize the sample
    tokens = tokenizer(
        sample["words"],
        boxes=sample["boxes"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )
    
    # Move to the correct device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    # Map predictions back to labels
    word_ids = tokens.word_ids()[0]
    predicted_labels = []
    true_labels = []
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            # Only consider the first token of each word
            if idx == 0 or word_ids[idx - 1] != word_idx:
                predicted_label = id_to_label[str(predictions[idx])]
                true_label = id_to_label[str(sample["ner_tags"][word_idx])]
                
                predicted_labels.append(predicted_label)
                true_labels.append(true_label)
    
    # Print results
    print("Sample words:", sample["words"])
    print("\nPredictions:")
    for word, pred_label, true_label in zip(sample["words"], predicted_labels, true_labels):
        if pred_label != true_label:
            print(f"{word:20} | Predicted: {pred_label:20} | True: {true_label:20} | INCORRECT")
        else:
            print(f"{word:20} | Predicted: {pred_label:20} | True: {true_label:20}")
    
    # Calculate accuracy
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    total = len(predicted_labels)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nAccuracy: {accuracy:.2f} ({correct}/{total})")

# Try predicting a sample
predict_sample(sample_idx=0)

# %% [markdown]
# ## 13. (Optional) Make Predictions on New Documents
# 
# This section demonstrates how to use the fine-tuned model on new documents.

# %%
def predict_on_new_document(words, boxes):
    """Make predictions on a new document."""
    # Ensure boxes are normalized to 0-1000
    normalized_boxes = [[min(max(0, coord), 1000) for coord in box] for box in boxes]
    
    # Tokenize
    tokens = tokenizer(
        words,
        boxes=normalized_boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )
    
    # Move to the correct device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    # Map predictions back to labels
    word_ids = tokens.word_ids()[0]
    results = []
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            # Only consider the first token of each word
            if idx == 0 or word_ids[idx - 1] != word_idx:
                predicted_label = id_to_label[str(predictions[idx])]
                results.append({
                    "word": words[word_idx],
                    "box": boxes[word_idx],
                    "label": predicted_label
                })
    
    return results

# Example usage:
# new_words = ["This", "is", "a", "sample", "document"]
# new_boxes = [[100, 100, 150, 120], [160, 100, 180, 120], [190, 100, 210, 120], 
#              [220, 100, 300, 120], [310, 100, 400, 120]]
# results = predict_on_new_document(new_words, new_boxes)
# for item in results:
#     print(f"{item['word']:20} | {item['label']}") 