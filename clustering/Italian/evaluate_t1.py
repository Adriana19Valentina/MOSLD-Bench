# evaluate_t1_bengali.py - Evaluation script for Test_1 (Bengali)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 70)
print("ROMANIAN TEST_1 EVALUATION - ALL CLASSES [0-5]")
print("=" * 70)

# =========================================================================
# LOAD MAPPINGS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: LOADING MAPPINGS")
print('=' * 70)

with open(f'{MODEL_T1_DIR}/label_mappings.json', 'r', encoding='utf-8') as f:
    model_mappings = json.load(f)

label2id = {int(k): int(v) for k, v in model_mappings['label2id'].items()}
id2label = {int(k): int(v) for k, v in model_mappings['id2label'].items()}
known_labels = model_mappings['known_labels']
discovered_labels = model_mappings['discovered_labels']

print(f"✅ Model has {model_mappings['num_classes']} classes")
print(f"   Baseline labels: {known_labels}")
print(f"   Discovered pseudo-labels: {discovered_labels}")

# Load clustering results
with open(T1_RESULTS_PKL, 'rb') as f:
    clustering_results = pickle.load(f)

cluster_to_label_eval = clustering_results.get('cluster_to_label_eval') or clustering_results['cluster_to_label_purity']
cluster_to_pseudo = clustering_results['cluster_to_pseudo']

# Create pseudo → GT mapping
pseudo_to_gt = {}
for cluster_id, pseudo_label in cluster_to_pseudo.items():
    if cluster_id in cluster_to_label_eval:
        gt_label = cluster_to_label_eval[cluster_id]
        pseudo_to_gt[pseudo_label] = gt_label

print(f"\n🗺️  Pseudo → GT mapping:")
for pseudo, gt in sorted(pseudo_to_gt.items()):
    print(f"   Pseudo {pseudo} → GT {gt}")

# Complete model → GT mapping
model_to_gt = {}
for label in BASELINE_LABELS:
    model_to_gt[label] = label
model_to_gt.update(pseudo_to_gt)

print(f"\n🗺️  Complete Model → GT mapping:")
for model_label in sorted(model_to_gt.keys()):
    print(f"   Model {model_label} → GT {model_to_gt[model_label]}")

# =========================================================================
# LOAD MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: LOADING MODEL")
print('=' * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_T1_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_T1_DIR)
model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")

# =========================================================================
# LOAD TEST DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: LOADING TEST DATA")
print('=' * 70)

test_df = pd.read_csv(TEST_1_CSV)
print(f"✅ Loaded {len(test_df)} samples")

print(f"\n📊 Test set distribution:")
for label in sorted(test_df['label'].unique()):
    count = len(test_df[test_df['label'] == label])
    label_type = "known" if label in BASELINE_LABELS else "unknown"
    if 'label_name' in test_df.columns:
        label_name = test_df[test_df['label'] == label]['label_name'].iloc[0]
        print(f"   Label {label} ({label_name:12s}) [{label_type:7s}]: {count:5d} samples")
    else:
        print(f"   Label {label} [{label_type:7s}]: {count:5d} samples")

# Create subsets
test_all = test_df.copy()
test_known = test_df[test_df['label'].isin(BASELINE_LABELS)].copy()
test_unknown = test_df[test_df['label'].isin(TEST_1_NEW_LABELS)].copy()

print(f"\n📊 Subset sizes:")
print(f"   All:     {len(test_all):5d} samples")
print(f"   Known:   {len(test_known):5d} samples ({len(test_known) / len(test_all) * 100:.1f}%)")
print(f"   Unknown: {len(test_unknown):5d} samples ({len(test_unknown) / len(test_all) * 100:.1f}%)")

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

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_predictions)


def map_predictions(predictions_ids):
    predictions_model = [id2label[int(p)] for p in predictions_ids]
    predictions_gt = [model_to_gt.get(p, p) for p in predictions_model]
    return predictions_model, predictions_gt


print(f"\n🔮 Predicting on {len(test_all)} samples...")

texts_all = test_all['content'].tolist()
pred_ids_all, _ = predict_batch(texts_all), None
pred_ids_all = predict_batch(texts_all)
pred_model_all, pred_gt_all = map_predictions(pred_ids_all)

test_all['predicted_model'] = pred_model_all
test_all['predicted_gt'] = pred_gt_all

print(f"✅ Predictions completed")

# =========================================================================
# CALCULATE METRICS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: CALCULATING METRICS")
print('=' * 70)


def calculate_metrics(y_true, y_pred, subset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'subset': subset_name,
        'samples': len(y_true),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


# ALL
y_true_all = test_all['label'].values
y_pred_all = np.array(pred_gt_all)
results_all = calculate_metrics(y_true_all, y_pred_all, 'all')

# KNOWN
y_true_known = test_known['label'].values
pred_known = predict_batch(test_known['content'].tolist())
_, pred_gt_known = map_predictions(pred_known)
y_pred_known = np.array(pred_gt_known)
results_known = calculate_metrics(y_true_known, y_pred_known, 'known')

# UNKNOWN
y_true_unknown = test_unknown['label'].values
pred_unknown = predict_batch(test_unknown['content'].tolist())
_, pred_gt_unknown = map_predictions(pred_unknown)
y_pred_unknown = np.array(pred_gt_unknown)
results_unknown = calculate_metrics(y_true_unknown, y_pred_unknown, 'unknown')

# =========================================================================
# DISPLAY RESULTS
# =========================================================================

print(f"\n{'=' * 70}")
print("📊 RESULTS SUMMARY")
print('=' * 70)

print(f"\n{'Subset':<12} {'Samples':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print(f"{'-' * 70}")

for r in [results_all, results_known, results_unknown]:
    print(
        f"{r['subset']:<12} {r['samples']:>8} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f}")

# Per-class accuracy
print(f"\n📊 Per-Class Accuracy:")

print(f"\n🔵 KNOWN CLASSES:")
for label in BASELINE_LABELS:
    mask = y_true_all == label
    if mask.sum() > 0:
        acc = accuracy_score(y_true_all[mask], y_pred_all[mask])
        print(f"  Label {label}: {acc:.4f} ({acc * 100:.2f}%) - {mask.sum()} samples")

print(f"\n🔴 UNKNOWN CLASSES:")
for label in TEST_1_NEW_LABELS:
    mask = y_true_all == label
    if mask.sum() > 0:
        acc = accuracy_score(y_true_all[mask], y_pred_all[mask])
        print(f"  Label {label}: {acc:.4f} ({acc * 100:.2f}%) - {mask.sum()} samples")

# =========================================================================
# SAVE RESULTS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: SAVING RESULTS")
print('=' * 70)

output_results = {
    'subset_metrics': {
        'all': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results_all.items()},
        'known': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results_known.items()},
        'unknown': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results_unknown.items()}
    },
    'pseudo_to_gt_mapping': {int(k): int(v) for k, v in pseudo_to_gt.items()},
    'model_to_gt_mapping': {int(k): int(v) for k, v in model_to_gt.items()}
}

with open(T1_EVAL_JSON, 'w', encoding='utf-8') as f:
    json.dump(output_results, f, ensure_ascii=False, indent=2)

print(f"✅ Results saved to: {T1_EVAL_JSON}")

# Save predictions
test_all.to_csv(os.path.join(OUTPUT_DIR, 'test_1_predictions.csv'), index=False)
print(f"✅ Predictions saved")

# =========================================================================
# FINAL SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("✅ EVALUATION T1 COMPLETED!")
print('=' * 70)

print(f"\n📊 QUICK SUMMARY:")
print(f"  ALL:     {results_all['accuracy'] * 100:.2f}%")
print(f"  KNOWN:   {results_known['accuracy'] * 100:.2f}%")
print(f"  UNKNOWN: {results_unknown['accuracy'] * 100:.2f}%")

forgetting_rate = (1 - results_known['accuracy']) * 100
print(f"\n📉 Forgetting rate (baseline): {forgetting_rate:.2f}%")

print(f"\n💡 INTERPRETATION:")
if results_unknown['accuracy'] > 0.85:
    print(f"  ✅ EXCELLENT discovery performance!")
elif results_unknown['accuracy'] > 0.75:
    print(f"  ✅ GOOD discovery performance")
else:
    print(f"  ⚠️  Discovery performance needs improvement")

print(f"\n🚀 NEXT: Run pipeline_t2_bengali.py")
print("=" * 70)