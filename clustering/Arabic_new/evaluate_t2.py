#!/usr/bin/env python3
"""
evaluate_t2.py - Evaluation with SEQUENCE embedding method only

Hungarian mapping: doar pentru clasele noi din T2
Metrici:
  - Known = doar baseline (primele 4 clase)
  - New = T1 + T2 (toate clasele adăugate)
"""

import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score
from config import *

# =========================================================================
# CONFIGURATION
# =========================================================================
MODEL_PATH = "./arabic_cl_outputs_1/model_t2"
RESULTS_PATH = "./arabic_cl_outputs_1/test_2_results.pkl"
EVAL_T1_PATH = "./arabic_cl_outputs_1/eval_t1_results.json"
TEST_DATA_PATH = TEST_2_CSV
OUTPUT_PATH = "./arabic_cl_outputs_1/eval_t2_results.json"
EMBEDDING_MODEL = MODEL_NAME

# =========================================================================
# LABEL DEFINITIONS
# =========================================================================
# Pentru Hungarian mapping - doar clasele noi din T2
HUNGARIAN_TARGET_LABELS = TEST_2_NEW_LABELS  # ex: [6, 7]

# Pentru calculul metricilor
KNOWN_LABELS = BASELINE_LABELS  #+  ex: [0, 1, 2, 3] - doar baseline
NEW_LABELS = TEST_1_NEW_LABELS + TEST_2_NEW_LABELS  # ex: [4, 5, 6, 7] - toate adăugate
# NEW_LABELS = TEST_2_NEW_LABELS

def load_classification_model(model_path):
    print(f"📥 Loading classification model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    id2label_raw = model.config.id2label
    id2label = {int(k): int(v) for k, v in id2label_raw.items()}
    print(f"   Model classes: {model.config.num_labels}")
    print(f"   id2label: {id2label}")
    return model, tokenizer, id2label


def load_embedding_model(model_name):
    print(f"\n📥 Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def get_embedding(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding[0]


def compute_semantic_similarity_sequence(embed_model, tokenizer, keywords_list, class_names_list):
    """Compute semantic similarity using SEQUENCE method only."""
    print(f"\n📊 Computing semantic similarity (SEQUENCE method)...")
    print(f"   Clusters: {len(keywords_list)}")
    print(f"   Target classes: {class_names_list}")

    n_clusters = len(keywords_list)
    n_classes = len(class_names_list)

    if n_clusters == 0:
        return np.array([])

    # Sequence embedding: concatenated keywords
    print(f"\n   Computing cluster embeddings (concatenated keywords)...")
    cluster_embeddings = []
    for i, keywords in enumerate(keywords_list):
        if keywords:
            seq = ' '.join(keywords[:10])
            seq_embed = get_embedding(embed_model, tokenizer, seq)
        else:
            seq_embed = np.zeros(768)
        cluster_embeddings.append(seq_embed)
        print(f"      Cluster {i}: '{' '.join(keywords[:5]) if keywords else 'NO KEYWORDS'}...'")

    # Class name embeddings
    print(f"\n   Computing class name embeddings...")
    class_embeddings = []
    for class_name in class_names_list:
        embed = get_embedding(embed_model, tokenizer, class_name)
        class_embeddings.append(embed)
        print(f"      '{class_name}'")

    # Compute similarity matrix
    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    sim_matrix = np.zeros((n_clusters, n_classes))
    for i in range(n_clusters):
        for j in range(n_classes):
            sim_matrix[i, j] = cosine_sim(cluster_embeddings[i], class_embeddings[j])

    # Print matrix
    print(f"\n{'=' * 60}")
    print(f"📊 SIMILARITY MATRIX (SEQUENCE):")
    print('=' * 60)
    header = "            " + "".join([f"{name:>12}" for name in class_names_list])
    print(header)
    for i in range(n_clusters):
        row = f"Cluster {i:2d} |"
        for j in range(n_classes):
            row += f"{sim_matrix[i, j]:>12.4f}"
        print(row)

    return sim_matrix


def hungarian_mapping(similarity_matrix, pseudo_labels, gt_labels, class_names):
    if similarity_matrix.size == 0:
        print("   ⚠️  Empty similarity matrix!")
        mapping = {p: g for p, g in zip(pseudo_labels, gt_labels)}
        return mapping, 0.0

    cost_matrix = -similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    print(f"\n🗺️  Hungarian Mapping Result:")

    total_sim = 0
    for i, j in zip(row_ind, col_ind):
        pseudo = pseudo_labels[i]
        gt = gt_labels[j]
        sim = similarity_matrix[i, j]
        total_sim += sim
        gt_name = class_names.get(gt, str(gt))
        mapping[pseudo] = gt
        print(f"   Pseudo {pseudo} → GT {gt} ({gt_name}), similarity={sim:.4f}")

    avg_sim = total_sim / len(row_ind) if len(row_ind) > 0 else 0
    print(f"   Average similarity: {avg_sim:.4f}")

    return mapping, avg_sim


def predict_batch(model, tokenizer, texts, batch_size=32):
    device = next(model.parameters()).device
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def evaluate():
    print("=" * 70)
    print("T2 EVALUATION (SEQUENCE METHOD)")
    print("=" * 70)

    # Step 1: Load models
    print(f"\n{'=' * 70}")
    print("STEP 1: LOADING MODELS")
    print("=" * 70)

    clf_model, clf_tokenizer, id2label = load_classification_model(MODEL_PATH)
    embed_model, embed_tokenizer = load_embedding_model(EMBEDDING_MODEL)

    # Load T1 mapping
    print(f"\n📥 Loading T1 mapping from: {EVAL_T1_PATH}")
    pseudo_to_gt_t1 = {}
    if os.path.exists(EVAL_T1_PATH):
        with open(EVAL_T1_PATH, 'r') as f:
            t1_results = json.load(f)
        pseudo_to_gt_t1 = {int(k): int(v) for k, v in t1_results.get('mapping', {}).items()}
        print(f"   T1 mapping (pseudo → GT): {pseudo_to_gt_t1}")
    else:
        print(f"   ⚠️  T1 results not found!")

    # Step 2: Load clustering results
    print(f"\n{'=' * 70}")
    print("STEP 2: LOADING CLUSTERING RESULTS")
    print("=" * 70)

    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)

    cluster_keywords = results.get('cluster_keywords', {})
    cluster_keywords_unique = results.get('cluster_unique_keywords', {})
    cluster_keywords = {int(k) if isinstance(k, str) else k: v for k, v in cluster_keywords.items()}
    cluster_keywords_unique = {int(k) if isinstance(k, str) else k: v for k, v in cluster_keywords_unique.items()}
    pseudo_label_mapping = results.get('cluster_to_pseudo', {})

    use_unique = len(cluster_keywords_unique) > 0 and all(len(v) > 0 for v in cluster_keywords_unique.values())
    keywords_to_use = cluster_keywords_unique if use_unique else cluster_keywords
    print(f"   Using: {'UNIQUE' if use_unique else 'regular'} keywords")

    keywords_list = []
    pseudo_labels_ordered = []

    if pseudo_label_mapping:
        for cluster_id in sorted(pseudo_label_mapping.keys()):
            pseudo = pseudo_label_mapping[cluster_id]
            pseudo_labels_ordered.append(pseudo)
            kw = keywords_to_use.get(cluster_id, [])
            keywords_list.append(kw)
            print(f"   Cluster {cluster_id} → Pseudo {pseudo}: {kw[:5] if kw else 'NO KEYWORDS'}...")

    # Step 3: Load test data
    print(f"\n{'=' * 70}")
    print("STEP 3: LOADING TEST DATA")
    print("=" * 70)

    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Loaded {len(test_df)} samples")

    # Hungarian mapping uses ONLY T2 new labels
    hungarian_target_labels = HUNGARIAN_TARGET_LABELS
    hungarian_class_names = [CLASS_NAMES[l] for l in hungarian_target_labels]
    print(f"\n   Hungarian target labels (T2 only): {hungarian_target_labels}")
    print(f"   Hungarian target class names: {hungarian_class_names}")

    # Metrics use different grouping
    print(f"\n   Metrics grouping:")
    print(f"      KNOWN labels (baseline): {KNOWN_LABELS}")
    print(f"      NEW labels (T1+T2): {NEW_LABELS}")

    # Step 4: Make predictions
    print(f"\n{'=' * 70}")
    print("STEP 4: MAKING PREDICTIONS")
    print("=" * 70)

    predictions = predict_batch(clf_model, clf_tokenizer, test_df['content'].tolist())
    print(f"   Unique predictions (model_ids): {sorted(set(predictions))}")

    # Step 5: Semantic Hungarian mapping (ONLY for T2 new classes)
    print(f"\n{'=' * 70}")
    print("STEP 5: SEMANTIC HUNGARIAN MAPPING (T2 classes only)")
    print("=" * 70)

    sim_matrix = compute_semantic_similarity_sequence(
        embed_model, embed_tokenizer,
        keywords_list,
        hungarian_class_names  # doar clasele T2!
    )

    mapping_t2, avg_sim = hungarian_mapping(
        sim_matrix,
        pseudo_labels_ordered,
        hungarian_target_labels,  # doar clasele T2!
        CLASS_NAMES
    )

    # Step 6: Evaluate
    print(f"\n{'=' * 70}")
    print("STEP 6: EVALUATION")
    print("=" * 70)

    gt_labels_all = test_df['label'].values

    # Build full_mapping: model_id → GT_label
    full_mapping = {}
    for model_id, pseudo in id2label.items():
        model_id = int(model_id)
        pseudo = int(pseudo)

        if pseudo in BASELINE_LABELS:
            # Baseline classes: direct mapping
            full_mapping[model_id] = pseudo
        elif pseudo in pseudo_to_gt_t1:
            # T1 classes: use T1 mapping
            full_mapping[model_id] = pseudo_to_gt_t1[pseudo]
        elif pseudo in mapping_t2:
            # T2 classes: use T2 Hungarian mapping
            full_mapping[model_id] = mapping_t2[pseudo]

    print(f"\n🗺️  FULL MAPPING (model_id → GT):")
    for mid, gt in sorted(full_mapping.items()):
        pseudo = id2label.get(mid, '?')
        gt_name = CLASS_NAMES.get(gt, str(gt))
        print(f"   model_id {mid} → pseudo {pseudo} → GT {gt} ({gt_name})")

    # Map predictions
    mapped_preds = np.array([full_mapping.get(p, -1) for p in predictions])
    valid_mask = mapped_preds >= 0

    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================

    # OVERALL
    overall_acc = accuracy_score(gt_labels_all[valid_mask], mapped_preds[valid_mask])
    overall_f1_macro = f1_score(gt_labels_all[valid_mask], mapped_preds[valid_mask], average='macro', zero_division=0)
    overall_f1_weighted = f1_score(gt_labels_all[valid_mask], mapped_preds[valid_mask], average='weighted', zero_division=0)

    # KNOWN (doar baseline - primele 4 clase)
    known_mask = np.isin(gt_labels_all, KNOWN_LABELS) & valid_mask
    print(f"Unique GT in known_mask: {np.unique(gt_labels_all[known_mask])}")
    print(f"Unique Pred in known_mask: {np.unique(mapped_preds[known_mask])}")
    print(f"Unique GT in known_mask: {np.unique(gt_labels_all[known_mask])}")
    print(f"Unique Pred in known_mask: {np.unique(mapped_preds[known_mask])}")
    print(f"Known Labels!!: {KNOWN_LABELS}")
    if known_mask.sum() > 0:
        known_acc = accuracy_score(gt_labels_all[known_mask], mapped_preds[known_mask])
        # known_f1_macro = f1_score(gt_labels_all[known_mask], mapped_preds[known_mask], average='macro', zero_division=0)
        known_f1_macro = f1_score(
            gt_labels_all[known_mask],
            mapped_preds[known_mask],
            labels=KNOWN_LABELS,
            average='macro',
            zero_division=0
        )
        known_f1_weighted = f1_score(gt_labels_all[known_mask], mapped_preds[known_mask], average='weighted', zero_division=0)
    else:
        known_acc = known_f1_macro = known_f1_weighted = 0.0

    # NEW (T1 + T2 - toate clasele adăugate)
    new_mask = np.isin(gt_labels_all, NEW_LABELS) & valid_mask
    if new_mask.sum() > 0:
        new_acc = accuracy_score(gt_labels_all[new_mask], mapped_preds[new_mask])
        # new_f1_macro = f1_score(gt_labels_all[new_mask], mapped_preds[new_mask], average='macro', zero_division=0)
        new_f1_macro = f1_score(
            gt_labels_all[new_mask],
            mapped_preds[new_mask],
            labels=NEW_LABELS,
            average='macro',
            zero_division=0
        )
        new_f1_weighted = f1_score(gt_labels_all[new_mask], mapped_preds[new_mask], average='weighted', zero_division=0)
    else:
        new_acc = new_f1_macro = new_f1_weighted = 0.0

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}

    print(f"\n📊 Per-class metrics (KNOWN - Baseline):")
    for label in KNOWN_LABELS:
        mask = (gt_labels_all == label) & valid_mask
        if mask.sum() > 0:
            acc = accuracy_score(gt_labels_all[mask], mapped_preds[mask])
            binary_gt = (gt_labels_all[valid_mask] == label).astype(int)
            binary_pred = (mapped_preds[valid_mask] == label).astype(int)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            per_class_acc[label] = acc
            per_class_f1[label] = f1
            print(f"   {label} ({CLASS_NAMES.get(label, '?')}): acc={acc:.4f}, f1={f1:.4f} (n={mask.sum()})")

    print(f"\n📊 Per-class metrics (NEW - T1+T2):")
    for label in NEW_LABELS:
        mask = (gt_labels_all == label) & valid_mask
        if mask.sum() > 0:
            acc = accuracy_score(gt_labels_all[mask], mapped_preds[mask])
            binary_gt = (gt_labels_all[valid_mask] == label).astype(int)
            binary_pred = (mapped_preds[valid_mask] == label).astype(int)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            per_class_acc[label] = acc
            per_class_f1[label] = f1
            print(f"   {label} ({CLASS_NAMES.get(label, '?')}): acc={acc:.4f}, f1={f1:.4f} (n={mask.sum()})")

    # =========================================================================
    # PRINT TABLE
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("📊 RESULTS SUMMARY - T2")
    print('=' * 70)

    print(f"\n   ┌─────────────────┬────────────┬────────────┬────────────┐")
    print(f"   │                 │  Accuracy  │  F1-Macro  │ F1-Weighted│")
    print(f"   ├─────────────────┼────────────┼────────────┼────────────┤")
    print(f"   │ OVERALL         │   {overall_acc:.4f}   │   {overall_f1_macro:.4f}   │   {overall_f1_weighted:.4f}   │")
    print(f"   │ KNOWN (base)    │   {known_acc:.4f}   │   {known_f1_macro:.4f}   │   {known_f1_weighted:.4f}   │")
    print(f"   │ NEW (T1+T2)     │   {new_acc:.4f}   │   {new_f1_macro:.4f}   │   {new_f1_weighted:.4f}   │")
    print(f"   └─────────────────┴────────────┴────────────┴────────────┘")

    print(f"\n   Samples: Overall={valid_mask.sum()}, Known={known_mask.sum()}, New={new_mask.sum()}")

    # Save results
    output = {
        'method': 'SEQUENCE',
        'step': 'T2',
        'overall_accuracy': overall_acc,
        'overall_f1_macro': overall_f1_macro,
        'overall_f1_weighted': overall_f1_weighted,
        'known_accuracy': known_acc,
        'known_f1_macro': known_f1_macro,
        'known_f1_weighted': known_f1_weighted,
        'new_accuracy': new_acc,
        'new_f1_macro': new_f1_macro,
        'new_f1_weighted': new_f1_weighted,
        'per_class_accuracy': {CLASS_NAMES.get(k, str(k)): v for k, v in per_class_acc.items()},
        'per_class_f1': {CLASS_NAMES.get(k, str(k)): v for k, v in per_class_f1.items()},
        'mapping_t2': {str(k): v for k, v in mapping_t2.items()},
        'mapping_t1': pseudo_to_gt_t1,
        'avg_similarity': avg_sim,
        'class_names': CLASS_NAMES
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("✅ T2 EVALUATION COMPLETED!")
    print('=' * 70)

    print(f"\n📊 SEMANTIC MAPPING (T2 only):")
    for pseudo, gt in mapping_t2.items():
        print(f"   Pseudo {pseudo} → GT {gt} ({CLASS_NAMES.get(gt, '?')})")

    return output


if __name__ == '__main__':
    evaluate()