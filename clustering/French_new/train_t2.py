# train_cl_t2_bengali.py - Incremental Training for Test_2 (Bengali)

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
print("Russian CONTINUAL LEARNING - TRAINING TEST_2 (INCREMENTAL)")
print("=" * 70)

ensure_output_dirs()

# =========================================================================
# LOAD DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING TRAINING DATA")
print('=' * 70)

if not os.path.exists(T2_PROCESSED_CSV):
    print(f"❌ ERROR: {T2_PROCESSED_CSV} not found!")
    print(f"   Please run pipeline_t2_bengali.py first")
    exit(1)

train_df = pd.read_csv(T2_PROCESSED_CSV)
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
    elif label >= PSEUDO_LABEL_START_T2:
        label_type = "test_2"
    else:
        label_type = "unknown"
    print(f"  Label {label:2d} ({label_type:8s}): {count:5d} samples")

print(f"\n📈 Summary:")
print(f"  Total samples: {len(train_df)}")
print(f"  Total classes: {len(all_labels)}")

# =========================================================================
# LOAD BASE MODEL (T1)
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: LOADING BASE MODEL (T1)")
print('=' * 70)

if not os.path.exists(MODEL_T1_DIR):
    print(f"❌ ERROR: {MODEL_T1_DIR} not found!")
    print(f"   Please train test_1 model first")
    exit(1)

with open(f'{MODEL_T1_DIR}/label_mappings.json', 'r', encoding='utf-8') as f:
    t1_mappings = json.load(f)

t1_num_classes = t1_mappings['num_classes']
print(f"✅ T1 model has {t1_num_classes} classes")

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_T1_DIR)


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

# Load config to get the correct number of labels
from transformers import AutoConfig

config_t1 = AutoConfig.from_pretrained(MODEL_T1_DIR)
print(f"  T1 config num_labels: {config_t1.num_labels}")

model_t1 = AutoModelForSequenceClassification.from_pretrained(MODEL_T1_DIR, config=config_t1)

# old_classifier = model_t1.classifier
# old_num_labels = old_classifier.out_features
# new_num_labels = len(label2id)
#
# print(f"\n📊 Model expansion:")
# print(f"  Old classes (T1): {old_num_labels}")
# print(f"  New classes (T2): {new_num_labels}")
# print(f"  Added: {new_num_labels - old_num_labels} classes")
#
# if new_num_labels > old_num_labels:
#     new_classifier = torch.nn.Linear(old_classifier.in_features, new_num_labels, bias=True)
#
#     with torch.no_grad():
#         new_classifier.weight[:old_num_labels] = old_classifier.weight
#         new_classifier.bias[:old_num_labels] = old_classifier.bias
#         torch.nn.init.normal_(new_classifier.weight[old_num_labels:], mean=0, std=0.02)
#         torch.nn.init.zeros_(new_classifier.bias[old_num_labels:])
#
#     model_t1.classifier = new_classifier
#     model_t1.num_labels = new_num_labels
#     print(f"✅ Classifier expanded")

# Detect classifier type (BERT vs RoBERTa/CamemBERT)
classifier = model_t1.classifier
is_roberta_style = hasattr(classifier, 'out_proj')

if is_roberta_style:
    # RoBERTa/CamemBERT: classifier has dense + out_proj layers
    old_num_labels = classifier.out_proj.out_features
    in_features = classifier.out_proj.in_features
    print(f"  Classifier type: RoBERTa-style (CamembertClassificationHead)")
else:
    # BERT: classifier is nn.Linear directly
    old_num_labels = classifier.out_features
    in_features = classifier.in_features
    print(f"  Classifier type: BERT-style (nn.Linear)")

new_num_labels = len(label2id)

print(f"\n📊 Model expansion:")
print(f"  Old classes (T1): {old_num_labels}")
print(f"  New classes (T2): {new_num_labels}")
print(f"  Added: {new_num_labels - old_num_labels} classes")

if new_num_labels > old_num_labels:
    if is_roberta_style:
        # RoBERTa/CamemBERT: replace out_proj layer only
        old_out_proj = classifier.out_proj
        new_out_proj = torch.nn.Linear(in_features, new_num_labels, bias=True)

        with torch.no_grad():
            new_out_proj.weight[:old_num_labels] = old_out_proj.weight
            new_out_proj.bias[:old_num_labels] = old_out_proj.bias
            torch.nn.init.normal_(new_out_proj.weight[old_num_labels:], mean=0, std=0.02)
            torch.nn.init.zeros_(new_out_proj.bias[old_num_labels:])

        model_t1.classifier.out_proj = new_out_proj
    else:
        # BERT: replace entire classifier
        new_classifier = torch.nn.Linear(in_features, new_num_labels, bias=True)

        with torch.no_grad():
            new_classifier.weight[:old_num_labels] = classifier.weight
            new_classifier.bias[:old_num_labels] = classifier.bias
            torch.nn.init.normal_(new_classifier.weight[old_num_labels:], mean=0, std=0.02)
            torch.nn.init.zeros_(new_classifier.bias[old_num_labels:])

        model_t1.classifier = new_classifier

    model_t1.num_labels = new_num_labels
    print(f"✅ Classifier expanded")

model = model_t1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# =========================================================================
# TRAINING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: TRAINING")
print('=' * 70)

training_args = TrainingArguments(
    output_dir=MODEL_T2_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=f'{MODEL_T2_DIR}/logs',
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
print(f"   Base: T1 ({old_num_labels} classes) → T2 ({new_num_labels} classes)")

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

model.config.num_labels = new_num_labels
model.config.id2label = {int(k): str(v) for k, v in id2label.items()}
model.config.label2id = {str(k): int(v) for k, v in label2id.items()}
trainer.save_model(MODEL_T2_DIR)
tokenizer.save_pretrained(MODEL_T2_DIR)

# Separate labels by type
baseline_labels = [l for l in all_labels if l in BASELINE_LABELS]
t1_pseudo_labels = [l for l in all_labels if l >= PSEUDO_LABEL_START_T1 and l < PSEUDO_LABEL_START_T2]
t2_pseudo_labels = [l for l in all_labels if l >= PSEUDO_LABEL_START_T2]

label_mappings = {
    'label2id': {int(k): int(v) for k, v in label2id.items()},
    'id2label': {int(k): int(v) for k, v in id2label.items()},
    'all_labels': [int(l) for l in all_labels],
    'baseline_labels': baseline_labels,
    't1_pseudo_labels': t1_pseudo_labels,
    't2_pseudo_labels': t2_pseudo_labels,
    'num_classes': len(all_labels),
    'model_name': MODEL_NAME,
    'max_length': MAX_LENGTH,
    'base_model': MODEL_T1_DIR,
    'incremental_step': 2
}

with open(f'{MODEL_T2_DIR}/label_mappings.json', 'w', encoding='utf-8') as f:
    json.dump(label_mappings, f, ensure_ascii=False, indent=2)

print(f"✅ Model saved to: {MODEL_T2_DIR}")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("✅ TRAINING T2 COMPLETED!")
print('=' * 70)

print(f"\n📊 MODEL EVOLUTION:")
print(f"  T1: {old_num_labels} classes")
print(f"  T2: {new_num_labels} classes (+{new_num_labels - old_num_labels})")

print(f"\n📊 CLASS BREAKDOWN:")
print(f"  Baseline [0-3]: {baseline_labels}")
print(f"  Test_1 pseudo: {t1_pseudo_labels}")
print(f"  Test_2 pseudo: {t2_pseudo_labels}")

print(f"\n🚀 NEXT: Run evaluate_t2_russian.py")
print("=" * 70)