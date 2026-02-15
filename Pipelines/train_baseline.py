# train_baseline.py - Train baseline model on known classes only
# This model will be used for OOD detection

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 70)
print("BASELINE MODEL TRAINING")
print("=" * 70)

ensure_output_dirs()

# =========================================================================
# CONFIGURATION
# =========================================================================

BASELINE_MODEL_DIR = os.path.join(OUTPUT_DIR, 'model_baseline')
os.makedirs(BASELINE_MODEL_DIR, exist_ok=True)

print(f"\n Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Baseline labels: {BASELINE_LABELS}")
print(f"   Output: {BASELINE_MODEL_DIR}")

# =========================================================================
# LOAD DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING DATA")
print('=' * 70)

train_df = pd.read_csv(TRAIN_CSV)
print(f" Loaded training data: {len(train_df)} samples")
print(f"   Labels: {sorted(train_df['label'].unique())}")

# Filter to baseline labels only
train_df = train_df[train_df['label'].isin(BASELINE_LABELS)].copy()
print(f" Filtered to baseline labels: {len(train_df)} samples")

# Create label mapping (original label -> 0-indexed)
unique_labels = sorted(train_df['label'].unique())
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print(f"\n Label mapping:")
for label, idx in label_to_idx.items():
    count = len(train_df[train_df['label'] == label])
    print(f"   {label} → {idx} ({count} samples)")

# Map labels to indices
train_df['label_idx'] = train_df['label'].map(label_to_idx)

# =========================================================================
# LOAD VALIDATION DATA (if available)
# =========================================================================

val_df = None
if os.path.exists(VAL_CSV):
    val_df = pd.read_csv(VAL_CSV)
    val_df = val_df[val_df['label'].isin(BASELINE_LABELS)].copy()
    val_df['label_idx'] = val_df['label'].map(label_to_idx)
    print(f" Loaded validation data: {len(val_df)} samples")


# =========================================================================
# DATASET CLASS
# =========================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =========================================================================
# INITIALIZE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: INITIALIZING MODEL")
print('=' * 70)

print(f" Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

num_labels = len(unique_labels)
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Model initialized")
print(f"   Classes: {num_labels}")
print(f"   Device: {device}")

# =========================================================================
# CREATE DATASETS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: CREATING DATASETS")
print('=' * 70)

train_dataset = TextDataset(
    train_df['content'].tolist(),
    train_df['label_idx'].tolist(),
    tokenizer,
    MAX_LENGTH
)
print(f" Training dataset: {len(train_dataset)} samples")

eval_dataset = None
if val_df is not None:
    eval_dataset = TextDataset(
        val_df['content'].tolist(),
        val_df['label_idx'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    print(f" Validation dataset: {len(eval_dataset)} samples")

# =========================================================================
# TRAINING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: TRAINING")
print('=' * 70)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


training_args = TrainingArguments(
    output_dir=BASELINE_MODEL_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    logging_dir=os.path.join(BASELINE_MODEL_DIR, 'logs'),
    logging_steps=100,
    eval_strategy='epoch' if eval_dataset else 'no',
    save_strategy='epoch',
    load_best_model_at_end=True if eval_dataset else False,
    save_total_limit=2,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print(f"\n Starting training...")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")

trainer.train()

# =========================================================================
# SAVE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: SAVING MODEL")
print('=' * 70)

trainer.save_model(BASELINE_MODEL_DIR)
tokenizer.save_pretrained(BASELINE_MODEL_DIR)

# Save label mappings (convert numpy types to native Python types)
label_mappings = {
    'label_to_idx': {str(k): int(v) for k, v in label_to_idx.items()},
    'idx_to_label': {str(k): int(v) for k, v in idx_to_label.items()},
    'baseline_labels': [int(l) for l in BASELINE_LABELS]
}

with open(os.path.join(BASELINE_MODEL_DIR, 'label_mappings.json'), 'w') as f:
    json.dump(label_mappings, f, indent=2)

print(f" Model saved to: {BASELINE_MODEL_DIR}")
print(f" Label mappings saved")

# =========================================================================
# FINAL EVALUATION
# =========================================================================

if eval_dataset:
    print(f"\n{'=' * 70}")
    print("STEP 6: FINAL EVALUATION")
    print('=' * 70)

    eval_results = trainer.evaluate()
    print(f"\n Validation Results:")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall: {eval_results['eval_recall']:.4f}")
    print(f"   F1: {eval_results['eval_f1']:.4f}")

print(f"\n{'=' * 70}")
print(" BASELINE MODEL TRAINING COMPLETED!")
print('=' * 70)
print(f"\n Output: {BASELINE_MODEL_DIR}")
print(f"Next: Run OOD detection on test data")