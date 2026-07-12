# train_cl_t3_bengali.py - Incremental Training for Test_3 (Bengali)

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
print("ROMANIAN CONTINUAL LEARNING - TRAINING TEST_3 (INCREMENTAL)")
print("=" * 70)

ensure_output_dirs()

# =========================================================================
# LOAD DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING TRAINING DATA")
print('=' * 70)

if not os.path.exists(T3_PROCESSED_CSV):
    print(f"❌ ERROR: {T3_PROCESSED_CSV} not found!")
    print(f"   Please run pipeline_t3_bengali.py first")
    exit(1)

train_df = pd.read_csv(T3_PROCESSED_CSV)
print(f"✅ Loaded {len(train_df)} samples")

all_labels = sorted(train_df['label'].unique())
all_labels = [int(l) for l in all_labels]

print(f"\n📊 Label distribution:")
for label in all_labels:
    count = len(train_df[train_df['label'] == label])
    if label in BASELINE_LABELS:
        label_type = "baseline"
    elif label >= PSEUDO_LABEL_START_T1 and label < PSEUDO_LABEL_START_T2:
        label_type = "test_1"
    elif label >= PSEUDO_LABEL_START_T2 and label < PSEUDO_LABEL_START_T3:
        label_type = "test_2"
    elif label >= PSEUDO_LABEL_START_T3:
        label_type = "test_3"
    else:
        label_type = "unknown"
    print(f"  Label {label:2d} ({label_type:8s}): {count:5d} samples")

print(f"\n📈 Summary:")
print(f"  Total samples: {len(train_df)}")
print(f"  Total classes: {len(all_labels)}")

# =========================================================================
# LOAD BASE MODEL (T2)
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: LOADING BASE MODEL (T2)")
print('=' * 70)

if not os.path.exists(MODEL_T2_DIR):
    print(f"❌ ERROR: {MODEL_T2_DIR} not found!")
    print(f"   Please train test_2 model first")
    exit(1)

with open(f'{MODEL_T2_DIR}/label_mappings.json', 'r', encoding='utf-8') as f:
    t2_mappings = json.load(f)

t2_num_classes = t2_mappings['num_classes']
print(f"✅ T2 model has {t2_num_classes} classes")

# =========================================================================
# PREPARE DATASET
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: PREPARING DATASET")
print('=' * 70)

label2id = {int(label): idx for idx, label in enumerate(all_labels)}
id2label = {idx: int(label) for label, idx in label2id.items()}

print(f"\n🏷️  Label mapping:")
for orig_label, model_id in sorted(label2id.items()):
    print(f"  {orig_label:2d} → model ID {model_id}")

train_df['label_id'] = train_df['label'].map(label2id)
print(f"\n✅ Labels mapped successfully")

# =========================================================================
# TOKENIZATION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: TOKENIZING DATA")
print('=' * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_T2_DIR)


def preprocess(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=MAX_LENGTH)


train_ds = Dataset.from_pandas(train_df[['content', 'label_id']])
train_ds = train_ds.rename_column('label_id', 'labels')
train_ds = train_ds.map(preprocess, batched=True)
train_ds = train_ds.remove_columns(['content'])

print(f"✅ Tokenized {len(train_ds)} samples")

# =========================================================================
# LOAD AND EXPAND MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: LOADING AND EXPANDING MODEL")
print('=' * 70)


# Helper function to get num_labels from model weights
def get_num_labels_from_weights(model_path):
    """Read num_labels directly from saved model weights (most reliable)."""
    import glob
    num_labels = None

    # Try safetensors first
    safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    if safetensor_files:
        try:
            from safetensors import safe_open
            with safe_open(safetensor_files[0], framework="pt") as f:
                for key in f.keys():
                    if 'classifier' in key.lower() and ('weight' in key.lower() or 'bias' in key.lower()):
                        tensor = f.get_tensor(key)
                        num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                        break
        except Exception as e:
            print(f"   Warning: Could not read safetensors: {e}")

    # Try pytorch_model.bin
    if not num_labels:
        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(bin_path):
            try:
                state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
                for key, tensor in state_dict.items():
                    if 'classifier' in key.lower() and ('weight' in key.lower() or 'bias' in key.lower()):
                        num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                        break
            except Exception as e:
                print(f"   Warning: Could not read pytorch_model.bin: {e}")

    return num_labels


# Get actual num_labels from weights
actual_num_labels = get_num_labels_from_weights(MODEL_T2_DIR)
print(f"  T2 model actual num_labels (from weights): {actual_num_labels}")

# Load config and override num_labels
from transformers import AutoConfig

config_t2 = AutoConfig.from_pretrained(MODEL_T2_DIR)
if actual_num_labels:
    config_t2.num_labels = actual_num_labels
    config_t2.id2label = {i: str(i) for i in range(actual_num_labels)}
    config_t2.label2id = {str(i): i for i in range(actual_num_labels)}

model_t2 = AutoModelForSequenceClassification.from_pretrained(MODEL_T2_DIR, config=config_t2)

old_classifier = model_t2.classifier
old_num_labels = old_classifier.out_features
new_num_labels = len(label2id)

print(f"\n📊 Model expansion:")
print(f"  Old classes (T2): {old_num_labels}")
print(f"  New classes (T3): {new_num_labels}")
print(f"  Added: {new_num_labels - old_num_labels} classes")

if new_num_labels > old_num_labels:
    new_classifier = torch.nn.Linear(old_classifier.in_features, new_num_labels, bias=True)

    with torch.no_grad():
        new_classifier.weight[:old_num_labels] = old_classifier.weight
        new_classifier.bias[:old_num_labels] = old_classifier.bias
        torch.nn.init.normal_(new_classifier.weight[old_num_labels:], mean=0, std=0.02)
        torch.nn.init.zeros_(new_classifier.bias[old_num_labels:])

    model_t2.classifier = new_classifier
    model_t2.num_labels = new_num_labels
    print(f"✅ Classifier expanded")

model = model_t2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# =========================================================================
# TRAINING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: TRAINING")
print('=' * 70)

training_args = TrainingArguments(
    output_dir=MODEL_T3_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=f'{MODEL_T3_DIR}/logs',
    logging_steps=100,
    save_strategy='epoch',
    save_total_limit=2,
    report_to='none',
    seed=42,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4 if torch.cuda.is_available() else 0,
    eval_strategy='no'
)


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

print(f"\n🚀 Starting incremental training...")
print(f"   Base: T2 ({old_num_labels} classes) → T3 ({new_num_labels} classes)")

try:
    train_result = trainer.train()
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {train_result.training_loss:.4f}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    raise

# =========================================================================
# SAVE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 7: SAVING MODEL")
print('=' * 70)

# Ensure output directory exists
os.makedirs(MODEL_T3_DIR, exist_ok=True)

model.config.num_labels = new_num_labels
model.config.id2label = {int(k): str(v) for k, v in id2label.items()}
model.config.label2id = {str(k): int(v) for k, v in label2id.items()}
trainer.save_model(MODEL_T3_DIR)
tokenizer.save_pretrained(MODEL_T3_DIR)

# Separate labels by type
baseline_labels = [l for l in all_labels if l in BASELINE_LABELS]
t1_pseudo_labels = [l for l in all_labels if l >= PSEUDO_LABEL_START_T1 and l < PSEUDO_LABEL_START_T2]
t2_pseudo_labels = [l for l in all_labels if l >= PSEUDO_LABEL_START_T2 and l < PSEUDO_LABEL_START_T3]
t3_pseudo_labels = [l for l in all_labels if l >= PSEUDO_LABEL_START_T3]

label_mappings = {
    'label2id': {int(k): int(v) for k, v in label2id.items()},
    'id2label': {int(k): int(v) for k, v in id2label.items()},
    'all_labels': [int(l) for l in all_labels],
    'baseline_labels': baseline_labels,
    't1_pseudo_labels': t1_pseudo_labels,
    't2_pseudo_labels': t2_pseudo_labels,
    't3_pseudo_labels': t3_pseudo_labels,
    'num_classes': len(all_labels),
    'model_name': MODEL_NAME,
    'max_length': MAX_LENGTH,
    'base_model': MODEL_T2_DIR,
    'incremental_step': 3
}

label_mappings_path = os.path.join(MODEL_T3_DIR, 'label_mappings.json')
with open(label_mappings_path, 'w', encoding='utf-8') as f:
    json.dump(label_mappings, f, ensure_ascii=False, indent=2)

print(f"✅ Model saved to: {MODEL_T3_DIR}")
print(f"✅ Label mappings saved to: {label_mappings_path}")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("✅ TRAINING T3 COMPLETED!")
print('=' * 70)

print(f"\n📊 MODEL EVOLUTION:")
print(f"  T1: baseline + test_1 pseudo")
print(f"  T2: {old_num_labels} classes")
print(f"  T3: {new_num_labels} classes (+{new_num_labels - old_num_labels})")

print(f"\n📊 CLASS BREAKDOWN:")
print(f"  Baseline [0-3]: {baseline_labels}")
print(f"  Test_1 pseudo: {t1_pseudo_labels}")
print(f"  Test_2 pseudo: {t2_pseudo_labels}")
print(f"  Test_3 pseudo: {t3_pseudo_labels}")

print(f"\n🚀 NEXT: Run evaluate_t3_bengali.py")
print("=" * 70)