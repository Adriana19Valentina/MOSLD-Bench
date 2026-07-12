# evaluate_t2_bengali.py - Evaluation for Test_2 (Bengali)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 70)
print("ROMANIAN TEST_2 EVALUATION - ALL # evaluate_t2_bengali.py - Evaluation for Test_2 (Bengali)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 70)
print("ROMANIAN TEST_2 EVALUATION - ALL CLASSES [0-7]")
print("=" * 70)

# =========================================================================
# LOAD MAPPINGS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING MAPPINGS")
print('=' * 70)

# Model mappings
with open(f'{MODEL_T2_DIR}/label_mappings.json', 'r') as f:
    model_mappings = json.load(f)

num_classes = model_mappings['num_classes']
label2id = {int(k): int(v) for k, v in model_mappings['label2id'].items()}
id2label = {int(k): int(v) for k, v in model_mappings['id2label'].items()}

print(f"✅ Model has {num_classes} classes")
print(f"   Labels: {sorted(label2id.keys())}")

# T1 clustering (pseudo → GT for labels 4,5)
with open(T1_RESULTS_PKL, 'rb') as f:
    t1_results = pickle.load(f)

cluster_to_label_t1 = t1_results.get('cluster_to_label_eval') or t1_results['cluster_to_label_purity']
cluster_to_pseudo_t1 = t1_results['cluster_to_pseudo']

pseudo_to_gt_t1 = {}
for cluster_id, pseudo in cluster_to_pseudo_t1.items():
    if cluster_id in cluster_to_label_t1:
        pseudo_to_gt_t1[pseudo] = cluster_to_label_t1[cluster_id]

print(f"\n🗺️  T1 Pseudo → GT:")
for pseudo, gt in sorted(pseudo_to_gt_t1.items()):
    print(f"   Pseudo {pseudo} → GT {gt}")

# T2 clustering (pseudo → GT for labels 6,7)
with open(T2_RESULTS_PKL, 'rb') as f:
    t2_results = pickle.load(f)

cluster_to_label_t2 = t2_results.get('cluster_to_label_eval') or t2_results['cluster_to_label_purity']
cluster_to_pseudo_t2 = t2_results['cluster_to_pseudo']

pseudo_to_gt_t2 = {}
for cluster_id, pseudo in cluster_to_pseudo_t2.items():
    if cluster_id in cluster_to_label_t2:
        pseudo_to_gt_t2[pseudo] = cluster_to_label_t2[cluster_id]

print(f"\n🗺️  T2 Pseudo → GT:")
for pseudo, gt in sorted(pseudo_to_gt_t2.items()):
    print(f"   Pseudo {pseudo} → GT {gt}")

# Combined mapping
model_to_gt = {}
for label in BASELINE_LABELS:
    model_to_gt[label] = label
model_to_gt.update(pseudo_to_gt_t1)
model_to_gt.update(pseudo_to_gt_t2)

print(f"\n🗺️  Complete Model → GT:")
for model_label in sorted(model_to_gt.keys()):
    print(f"   Model {model_label} → GT {model_to_gt[model_label]}")

# =========================================================================
# LOAD MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: LOADING MODEL")
print('=' * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_T2_DIR)
config = AutoConfig.from_pretrained(MODEL_T2_DIR)
config.num_labels = num_classes

model = AutoModelForSequenceClassification.from_pretrained(MODEL_T2_DIR, config=config)
model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")

# =========================================================================
# LOAD TEST DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: LOADING TEST DATA")
print('=' * 70)

test_df = pd.read_csv(TEST_2_CSV)
print(f"✅ Loaded {len(test_df)} samples")

# Subsets
test_all = test_df.copy()
test_known = test_df[test_df['label'].isin(KNOWN_LABELS_T2)].copy()  # [0-5]
test_unknown = test_df[test_df['label'].isin(TEST_2_NEW_LABELS)].copy()  # [6,7]

print(f"\n📊 Subsets:")
print(f"   ALL [0-7]: {len(test_all)} samples")
print(f"   KNOWN [0-5]: {len(test_known)} samples")
print(f"   UNKNOWN [6-7]: {len(test_unknown)} samples")

# =========================================================================
# PREDICTION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: MAKING PREDICTIONS")
print('=' * 70)

def predict_batch(texts, batch_size=32):
    all_predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Handle None/NaN values
        batch_texts = [str(t) if t is not None and str(t) != 'nan' else '' for t in batch_texts]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
    return np.array(all_predictions)

def evaluate_subset(df, subset_name):
    print(f"\n🔮 Evaluating {subset_name} ({len(df)} samples)...")

    texts = df['content'].tolist()
    predictions_ids = predict_batch(texts)

    # Map to model labels then to GT
    predictions_model = [id2label[int(p)] for p in predictions_ids]
    predictions_gt = [model_to_gt.get(p, p) for p in predictions_model]

    y_true = df['label'].values
    y_pred = np.array(predictions_gt)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"✅ {subset_name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    return {
        'predictions_model': predictions_model,
        'predictions_gt': predictions_gt,
        'y_true': y_true,
        'metrics': {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1), 'samples': len(df)}
    }

results_all = evaluate_subset(test_all, "ALL [0-7]")
results_known = evaluate_subset(test_known, "KNOWN [0-5]")
results_unknown = evaluate_subset(test_unknown, "UNKNOWN [6-7]")

# =========================================================================
# DETAILED ANALYSIS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: DETAILED ANALYSIS")
print('=' * 70)

y_true_all = np.array(results_all['y_true'])
y_pred_all = np.array(results_all['predictions_gt'])
pred_model_all = results_all['predictions_model']

print(f"\n📊 Per-Class Accuracy:")

for gt_label in sorted(set(y_true_all)):
    mask = y_true_all == gt_label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_true_all[mask], y_pred_all[mask])
        if 'label_name' in test_all.columns:
            name = test_all[test_all['label'] == gt_label]['label_name'].iloc[0]
        else:
            name = f"Label {gt_label}"
        print(f"  GT {gt_label} ({name:12s}): {class_acc:.4f} ({class_acc * 100:5.2f}%) - {mask.sum()} samples")

# Forgetting analysis
print(f"\n📊 FORGETTING ANALYSIS:")

# Baseline [0-3]
baseline_mask = np.isin(y_true_all, BASELINE_LABELS)
if baseline_mask.sum() > 0:
    baseline_acc = accuracy_score(y_true_all[baseline_mask], y_pred_all[baseline_mask])
    print(f"\n  Baseline [0-3]: {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")

# T1 classes [4-5]
t1_mask = np.isin(y_true_all, TEST_1_NEW_LABELS)
if t1_mask.sum() > 0:
    t1_acc = accuracy_score(y_true_all[t1_mask], y_pred_all[t1_mask])
    print(f"  Test_1 [4-5]:   {t1_acc:.4f} ({t1_acc * 100:.2f}%)")

# =========================================================================
# SAVE RESULTS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: SAVING RESULTS")
print('=' * 70)

output_df = test_all.copy()
output_df['predicted_model'] = results_all['predictions_model']
output_df['predicted_gt'] = results_all['predictions_gt']
output_df.to_csv(os.path.join(OUTPUT_DIR, 'test_2_predictions.csv'), index=False)

summary = {
    'subset_metrics': {
        'all': results_all['metrics'],
        'known': results_known['metrics'],
        'unknown': results_unknown['metrics']
    },
    'model_to_gt_mapping': {int(k): int(v) for k, v in model_to_gt.items()},
    'forgetting': {
        'baseline': float(baseline_acc) if baseline_mask.sum() > 0 else None,
        'test1': float(t1_acc) if t1_mask.sum() > 0 else None
    }
}

with open(T2_EVAL_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Results saved to: {T2_EVAL_JSON}")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("✅ EVALUATION T2 COMPLETED!")
print('=' * 70)

print(f"\n📊 SUMMARY:")
print(f"  ALL [0-7]:     {results_all['metrics']['accuracy'] * 100:.2f}%")
print(f"  KNOWN [0-5]:   {results_known['metrics']['accuracy'] * 100:.2f}%")
print(f"  UNKNOWN [6-7]: {results_unknown['metrics']['accuracy'] * 100:.2f}%")

print(f"\n🚀 NEXT: Run pipeline_t3_bengali.py")
print("=" * 70)CLASSES [0-7]")
print("=" * 70)

# =========================================================================
# LOAD MAPPINGS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING MAPPINGS")
print('=' * 70)

# Model mappings
with open(f'{MODEL_T2_DIR}/label_mappings.json', 'r') as f:
    model_mappings = json.load(f)

num_classes = model_mappings['num_classes']
label2id = {int(k): int(v) for k, v in model_mappings['label2id'].items()}
id2label = {int(k): int(v) for k, v in model_mappings['id2label'].items()}

print(f"✅ Model has {num_classes} classes")
print(f"   Labels: {sorted(label2id.keys())}")

# T1 clustering (pseudo → GT for labels 4,5)
with open(T1_RESULTS_PKL, 'rb') as f:
    t1_results = pickle.load(f)

cluster_to_label_t1 = t1_results.get('cluster_to_label_eval') or t1_results['cluster_to_label_purity']
cluster_to_pseudo_t1 = t1_results['cluster_to_pseudo']

pseudo_to_gt_t1 = {}
for cluster_id, pseudo in cluster_to_pseudo_t1.items():
    if cluster_id in cluster_to_label_t1:
        pseudo_to_gt_t1[pseudo] = cluster_to_label_t1[cluster_id]

print(f"\n🗺️  T1 Pseudo → GT:")
for pseudo, gt in sorted(pseudo_to_gt_t1.items()):
    print(f"   Pseudo {pseudo} → GT {gt}")

# T2 clustering (pseudo → GT for labels 6,7)
with open(T2_RESULTS_PKL, 'rb') as f:
    t2_results = pickle.load(f)

cluster_to_label_t2 = t2_results.get('cluster_to_label_eval') or t2_results['cluster_to_label_purity']
cluster_to_pseudo_t2 = t2_results['cluster_to_pseudo']

pseudo_to_gt_t2 = {}
for cluster_id, pseudo in cluster_to_pseudo_t2.items():
    if cluster_id in cluster_to_label_t2:
        pseudo_to_gt_t2[pseudo] = cluster_to_label_t2[cluster_id]

print(f"\n🗺️  T2 Pseudo → GT:")
for pseudo, gt in sorted(pseudo_to_gt_t2.items()):
    print(f"   Pseudo {pseudo} → GT {gt}")

# Combined mapping
model_to_gt = {}
for label in BASELINE_LABELS:
    model_to_gt[label] = label
model_to_gt.update(pseudo_to_gt_t1)
model_to_gt.update(pseudo_to_gt_t2)

print(f"\n🗺️  Complete Model → GT:")
for model_label in sorted(model_to_gt.keys()):
    print(f"   Model {model_label} → GT {model_to_gt[model_label]}")

# =========================================================================
# LOAD MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: LOADING MODEL")
print('=' * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_T2_DIR)
config = AutoConfig.from_pretrained(MODEL_T2_DIR)
config.num_labels = num_classes

model = AutoModelForSequenceClassification.from_pretrained(MODEL_T2_DIR, config=config)
model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")

# =========================================================================
# LOAD TEST DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: LOADING TEST DATA")
print('=' * 70)

test_df = pd.read_csv(TEST_2_CSV)
print(f"✅ Loaded {len(test_df)} samples")

# Subsets
test_all = test_df.copy()
test_known = test_df[test_df['label'].isin(KNOWN_LABELS_T2)].copy()  # [0-5]
test_unknown = test_df[test_df['label'].isin(TEST_2_NEW_LABELS)].copy()  # [6,7]

print(f"\n📊 Subsets:")
print(f"   ALL [0-7]: {len(test_all)} samples")
print(f"   KNOWN [0-5]: {len(test_known)} samples")
print(f"   UNKNOWN [6-7]: {len(test_unknown)} samples")

# =========================================================================
# PREDICTION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: MAKING PREDICTIONS")
print('=' * 70)


def predict_batch(texts, batch_size=32):
    all_predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
    return np.array(all_predictions)


def evaluate_subset(df, subset_name):
    print(f"\n🔮 Evaluating {subset_name} ({len(df)} samples)...")

    texts = df['content'].tolist()
    predictions_ids = predict_batch(texts)

    # Map to model labels then to GT
    predictions_model = [id2label[int(p)] for p in predictions_ids]
    predictions_gt = [model_to_gt.get(p, p) for p in predictions_model]

    y_true = df['label'].values
    y_pred = np.array(predictions_gt)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"✅ {subset_name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    return {
        'predictions_model': predictions_model,
        'predictions_gt': predictions_gt,
        'y_true': y_true,
        'metrics': {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1),
                    'samples': len(df)}
    }


results_all = evaluate_subset(test_all, "ALL [0-7]")
results_known = evaluate_subset(test_known, "KNOWN [0-5]")
results_unknown = evaluate_subset(test_unknown, "UNKNOWN [6-7]")

# =========================================================================
# DETAILED ANALYSIS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: DETAILED ANALYSIS")
print('=' * 70)

y_true_all = np.array(results_all['y_true'])
y_pred_all = np.array(results_all['predictions_gt'])
pred_model_all = results_all['predictions_model']

print(f"\n📊 Per-Class Accuracy:")

for gt_label in sorted(set(y_true_all)):
    mask = y_true_all == gt_label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_true_all[mask], y_pred_all[mask])
        if 'label_name' in test_all.columns:
            name = test_all[test_all['label'] == gt_label]['label_name'].iloc[0]
        else:
            name = f"Label {gt_label}"
        print(f"  GT {gt_label} ({name:12s}): {class_acc:.4f} ({class_acc * 100:5.2f}%) - {mask.sum()} samples")

# Forgetting analysis
print(f"\n📊 FORGETTING ANALYSIS:")

# Baseline [0-3]
baseline_mask = np.isin(y_true_all, BASELINE_LABELS)
if baseline_mask.sum() > 0:
    baseline_acc = accuracy_score(y_true_all[baseline_mask], y_pred_all[baseline_mask])
    print(f"\n  Baseline [0-3]: {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")

# T1 classes [4-5]
t1_mask = np.isin(y_true_all, TEST_1_NEW_LABELS)
if t1_mask.sum() > 0:
    t1_acc = accuracy_score(y_true_all[t1_mask], y_pred_all[t1_mask])
    print(f"  Test_1 [4-5]:   {t1_acc:.4f} ({t1_acc * 100:.2f}%)")

# =========================================================================
# SAVE RESULTS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: SAVING RESULTS")
print('=' * 70)

output_df = test_all.copy()
output_df['predicted_model'] = results_all['predictions_model']
output_df['predicted_gt'] = results_all['predictions_gt']
output_df.to_csv(os.path.join(OUTPUT_DIR, 'test_2_predictions.csv'), index=False)

summary = {
    'subset_metrics': {
        'all': results_all['metrics'],
        'known': results_known['metrics'],
        'unknown': results_unknown['metrics']
    },
    'model_to_gt_mapping': {int(k): int(v) for k, v in model_to_gt.items()},
    'forgetting': {
        'baseline': float(baseline_acc) if baseline_mask.sum() > 0 else None,
        'test1': float(t1_acc) if t1_mask.sum() > 0 else None
    }
}

with open(T2_EVAL_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Results saved to: {T2_EVAL_JSON}")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("✅ EVALUATION T2 COMPLETED!")
print('=' * 70)

print(f"\n📊 SUMMARY:")
print(f"  ALL [0-7]:     {results_all['metrics']['accuracy'] * 100:.2f}%")
print(f"  KNOWN [0-5]:   {results_known['metrics']['accuracy'] * 100:.2f}%")
print(f"  UNKNOWN [6-7]: {results_unknown['metrics']['accuracy'] * 100:.2f}%")

print(f"\n🚀 NEXT: Run pipeline_t3_bengali.py")
print("=" * 70)