# train_cl_t1_bengali.py - Training script for Test_1

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 70)
print("ROMANIAN CONTINUAL LEARNING - TRAINING TEST_1")
print("=" * 70)

ensure_output_dirs()

# =========================================================================
# LOAD DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING TRAINING DATA")
print('=' * 70)

if not os.path.exists(T1_PROCESSED_CSV):
    print(f"ERROR: {T1_PROCESSED_CSV} not found!")
    print(f"   Please run pipeline_t1_bengali.py first")
    exit(1)

train_df = pd.read_csv(T1_PROCESSED_CSV)
print(f" Loaded {len(train_df)} samples")

all_labels = sorted(train_df['label'].unique())
all_labels = [int(l) for l in all_labels]

known_labels = [l for l in all_labels if l < 10]
discovered_labels = [l for l in all_labels if l >= 10]

print(f"\n Label distribution:")
for label in all_labels:
    count = len(train_df[train_df['label'] == label])
    label_type = "baseline" if label in known_labels else "discovered"
    print(f"  Label {label:2d} ({label_type:10s}): {count:5d} samples")

print(f"\n Summary:")
print(f"  Total samples: {len(train_df)}")
print(f"  Total classes: {len(all_labels)}")
print(f"  Baseline classes: {known_labels}")
print(f"  Discovered classes: {discovered_labels}")

# =========================================================================
# PREPARE DATASET
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: PREPARING DATASET")
print('=' * 70)

label2id = {int(label): idx for idx, label in enumerate(all_labels)}
id2label = {idx: int(label) for label, idx in label2id.items()}

print(f"\n  Label mapping:")
for orig_label, model_id in sorted(label2id.items()):
    print(f"  {orig_label:2d} → model ID {model_id}")

train_df['label_id'] = train_df['label'].map(label2id)

if train_df['label_id'].isna().any():
    print(f"\n ERROR: Some labels couldn't be mapped!")
    exit(1)

print(f"\n All labels mapped successfully")

# =========================================================================
# TOKENIZATION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: TOKENIZING DATA")
print('=' * 70)

print(f"\n Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(
        examples['content'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )

train_ds = Dataset.from_pandas(train_df[['content', 'label_id']])
train_ds = train_ds.rename_column('label_id', 'labels')

print(f"\n Tokenizing {len(train_ds)} samples...")
train_ds = train_ds.map(preprocess, batched=True)
train_ds = train_ds.remove_columns(['content'])

print(f" Dataset prepared")

# =========================================================================
# INITIALIZE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: INITIALIZING MODEL")
print('=' * 70)

print(f"\nLoading model: {MODEL_NAME}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label={str(k): str(v) for k, v in id2label.items()},
    label2id={str(k): v for k, v in label2id.items()},
    ignore_mismatched_sizes=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Model initialized")
print(f"   Classes: {len(label2id)}")
print(f"   Device: {device}")
print(f"   Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# =========================================================================
# TRAINING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: TRAINING")
print('=' * 70)

training_args = TrainingArguments(
    output_dir=MODEL_T1_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=f'{MODEL_T1_DIR}/logs',
    logging_steps=100,
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to='none',
    seed=42,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4 if torch.cuda.is_available() else 0,
    eval_strategy='no'
)

print(f"\n  Training parameters:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Total steps: ~{len(train_ds) // BATCH_SIZE * NUM_EPOCHS}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(f"\n Starting training...")

try:
    train_result = trainer.train()
    print(f"\n Training completed!")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    print(f"   Total steps: {train_result.global_step}")
except Exception as e:
    print(f"\n ERROR during training: {e}")
    raise

# =========================================================================
# SAVE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: SAVING MODEL")
print('=' * 70)

trainer.save_model(MODEL_T1_DIR)
tokenizer.save_pretrained(MODEL_T1_DIR)

label_mappings = {
    'label2id': {int(k): int(v) for k, v in label2id.items()},
    'id2label': {int(k): int(v) for k, v in id2label.items()},
    'all_labels': [int(l) for l in all_labels],
    'known_labels': [int(l) for l in known_labels],
    'discovered_labels': [int(l) for l in discovered_labels],
    'num_classes': len(all_labels),
    'model_name': MODEL_NAME,
    'max_length': MAX_LENGTH
}

with open(f'{MODEL_T1_DIR}/label_mappings.json', 'w', encoding='utf-8') as f:
    json.dump(label_mappings, f, ensure_ascii=False, indent=2)

training_config = {
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'max_length': MAX_LENGTH,
    'total_samples': len(train_ds),
    'total_steps': train_result.global_step if 'train_result' in locals() else 'unknown'
}

with open(f'{MODEL_T1_DIR}/training_config.json', 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

print(f"\n Model saved to: {MODEL_T1_DIR}")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("TRAINING T1 COMPLETED!")
print('=' * 70)

print(f"\n MODEL INFO:")
print(f"  Total classes: {len(all_labels)}")
print(f"  Baseline: {known_labels}")
print(f"  Discovered: {discovered_labels}")
print(f"  Training samples: {len(train_ds)}")

print(f"\n NEXT: Run evaluate_t1_bengali.py")
print("=" * 70)