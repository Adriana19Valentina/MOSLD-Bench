
"""
evaluate_t3.py - Evaluation with SEQUENCE embedding method only
Includes detailed metrics: accuracy and F1-macro for known/new classes
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
# MODEL_PATH = "./english_cl_outputs_1/model_t3"
# RESULTS_PATH = "./english_cl_outputs_1/test_3_results.pkl"
# EVAL_T1_PATH = "./english_cl_outputs_1/eval_t1_results.json"
# EVAL_T2_PATH = "./english_cl_outputs_1/eval_t2_results.json"
# TEST_DATA_PATH = TEST_3_CSV
# OUTPUT_PATH = "./english_cl_outputs_1/eval_t3_results.json"
EMBEDDING_MODEL = MODEL_NAME
MODEL_PATH = "./romanian_cl_outputs_1/model_t3"
RESULTS_PATH = "./romanian_cl_outputs_1/test_3_results.pkl"
EVAL_T1_PATH = "./romanian_cl_outputs_1/eval_t1_results.json"
EVAL_T2_PATH = "./romanian_cl_outputs_1/eval_t2_results.json"
TEST_DATA_PATH = TEST_3_CSV
OUTPUT_PATH = "./romanian_cl_outputs_1/eval_t3_results.json"

# All known labels at T3 (baseline + T1 + T2)
KNOWN_LABELS_T3 = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS


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
    print("T3 EVALUATION (SEQUENCE METHOD)")
    print("=" * 70)

    # Step 1: Load models
    print(f"\n{'=' * 70}")
    print("STEP 1: LOADING MODELS")
    print("=" * 70)

    clf_model, clf_tokenizer, id2label = load_classification_model(MODEL_PATH)
    embed_model, embed_tokenizer = load_embedding_model(EMBEDDING_MODEL)

    # Load T1 and T2 mappings
    print(f"\n📥 Loading previous mappings...")

    pseudo_to_gt_t1 = {}
    if os.path.exists(EVAL_T1_PATH):
        with open(EVAL_T1_PATH, 'r') as f:
            t1_results = json.load(f)
        pseudo_to_gt_t1 = {int(k): int(v) for k, v in t1_results.get('mapping', {}).items()}
        print(f"   T1 mapping: {pseudo_to_gt_t1}")

    pseudo_to_gt_t2 = {}
    if os.path.exists(EVAL_T2_PATH):
        with open(EVAL_T2_PATH, 'r') as f:
            t2_results = json.load(f)
        pseudo_to_gt_t2 = {int(k): int(v) for k, v in t2_results.get('mapping', {}).items()}
        print(f"   T2 mapping: {pseudo_to_gt_t2}")

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

    new_labels = TEST_3_NEW_LABELS
    new_class_names = [CLASS_NAMES[l] for l in new_labels]
    print(f"   T3 NEW labels: {new_labels}")
    print(f"   T3 NEW class names: {new_class_names}")

    # Step 4: Make predictions
    print(f"\n{'=' * 70}")
    print("STEP 4: MAKING PREDICTIONS")
    print("=" * 70)

    predictions = predict_batch(clf_model, clf_tokenizer, test_df['content'].tolist())
    print(f"   Unique predictions (model_ids): {sorted(set(predictions))}")

    # Step 5: Semantic Hungarian mapping (SEQUENCE only)
    print(f"\n{'=' * 70}")
    print("STEP 5: SEMANTIC HUNGARIAN MAPPING")
    print("=" * 70)

    sim_matrix = compute_semantic_similarity_sequence(
        embed_model, embed_tokenizer,
        keywords_list,
        new_class_names
    )

    mapping, avg_sim = hungarian_mapping(sim_matrix, pseudo_labels_ordered, new_labels, CLASS_NAMES)

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
            full_mapping[model_id] = pseudo
        elif pseudo in pseudo_to_gt_t1:
            full_mapping[model_id] = pseudo_to_gt_t1[pseudo]
        elif pseudo in pseudo_to_gt_t2:
            full_mapping[model_id] = pseudo_to_gt_t2[pseudo]
        elif pseudo in mapping:
            full_mapping[model_id] = mapping[pseudo]

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
    overall_f1_weighted = f1_score(gt_labels_all[valid_mask], mapped_preds[valid_mask], average='weighted',
                                   zero_division=0)

    # KNOWN (baseline + T1 + T2)
    known_mask = np.isin(gt_labels_all, KNOWN_LABELS_T3) & valid_mask
    if known_mask.sum() > 0:
        known_acc = accuracy_score(gt_labels_all[known_mask], mapped_preds[known_mask])
        known_f1_macro = f1_score(gt_labels_all[known_mask], mapped_preds[known_mask], average='macro', zero_division=0)
        known_f1_weighted = f1_score(gt_labels_all[known_mask], mapped_preds[known_mask], average='weighted',
                                     zero_division=0)
    else:
        known_acc = known_f1_macro = known_f1_weighted = 0.0

    # NEW (T3 classes)
    new_mask = np.isin(gt_labels_all, new_labels) & valid_mask
    if new_mask.sum() > 0:
        new_acc = accuracy_score(gt_labels_all[new_mask], mapped_preds[new_mask])
        new_f1_macro = f1_score(gt_labels_all[new_mask], mapped_preds[new_mask], average='macro', zero_division=0)
        new_f1_weighted = f1_score(gt_labels_all[new_mask], mapped_preds[new_mask], average='weighted', zero_division=0)
    else:
        new_acc = new_f1_macro = new_f1_weighted = 0.0

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}

    print(f"\n📊 Per-class metrics (KNOWN - Baseline + T1 + T2):")
    for label in KNOWN_LABELS_T3:
        mask = (gt_labels_all == label) & valid_mask
        if mask.sum() > 0:
            acc = accuracy_score(gt_labels_all[mask], mapped_preds[mask])
            binary_gt = (gt_labels_all[valid_mask] == label).astype(int)
            binary_pred = (mapped_preds[valid_mask] == label).astype(int)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            per_class_acc[label] = acc
            per_class_f1[label] = f1
            print(f"   {label} ({CLASS_NAMES.get(label, '?')}): acc={acc:.4f}, f1={f1:.4f} (n={mask.sum()})")

    print(f"\n📊 Per-class metrics (NEW - T3):")
    for label in new_labels:
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
    # FORGETTING ANALYSIS
    # =========================================================================
    print(f"\n📊 FORGETTING ANALYSIS (per-step):")

    baseline_mask = np.isin(gt_labels_all, BASELINE_LABELS) & valid_mask
    t1_mask = np.isin(gt_labels_all, TEST_1_NEW_LABELS) & valid_mask
    t2_mask = np.isin(gt_labels_all, TEST_2_NEW_LABELS) & valid_mask
    t3_mask = np.isin(gt_labels_all, TEST_3_NEW_LABELS) & valid_mask

    baseline_acc = accuracy_score(gt_labels_all[baseline_mask],
                                  mapped_preds[baseline_mask]) if baseline_mask.sum() > 0 else 0
    t1_acc = accuracy_score(gt_labels_all[t1_mask], mapped_preds[t1_mask]) if t1_mask.sum() > 0 else 0
    t2_acc = accuracy_score(gt_labels_all[t2_mask], mapped_preds[t2_mask]) if t2_mask.sum() > 0 else 0
    t3_acc = accuracy_score(gt_labels_all[t3_mask], mapped_preds[t3_mask]) if t3_mask.sum() > 0 else 0

    print(f"   BASELINE: acc={baseline_acc:.4f} (n={baseline_mask.sum()})")
    print(f"   T1:       acc={t1_acc:.4f} (n={t1_mask.sum()})")
    print(f"   T2:       acc={t2_acc:.4f} (n={t2_mask.sum()})")
    print(f"   T3:       acc={t3_acc:.4f} (n={t3_mask.sum()})")

    # =========================================================================
    # PRINT TABLE
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("📊 RESULTS SUMMARY - T3")
    print('=' * 70)

    print(f"\n   ┌─────────────────┬────────────┬────────────┬────────────┐")
    print(f"   │                 │  Accuracy  │  F1-Macro  │ F1-Weighted│")
    print(f"   ├─────────────────┼────────────┼────────────┼────────────┤")
    print(
        f"   │ OVERALL         │   {overall_acc:.4f}   │   {overall_f1_macro:.4f}   │   {overall_f1_weighted:.4f}   │")
    print(f"   │ KNOWN           │   {known_acc:.4f}   │   {known_f1_macro:.4f}   │   {known_f1_weighted:.4f}   │")
    print(f"   │ NEW (T3)        │   {new_acc:.4f}   │   {new_f1_macro:.4f}   │   {new_f1_weighted:.4f}   │")
    print(f"   └─────────────────┴────────────┴────────────┴────────────┘")

    print(f"\n   📊 FORGETTING TABLE:")
    print(f"   ┌─────────────────┬────────────┬────────────┐")
    print(f"   │ Step            │  Accuracy  │  Samples   │")
    print(f"   ├─────────────────┼────────────┼────────────┤")
    print(f"   │ BASELINE        │   {baseline_acc:.4f}   │   {baseline_mask.sum():>6}   │")
    print(f"   │ T1              │   {t1_acc:.4f}   │   {t1_mask.sum():>6}   │")
    print(f"   │ T2              │   {t2_acc:.4f}   │   {t2_mask.sum():>6}   │")
    print(f"   │ T3              │   {t3_acc:.4f}   │   {t3_mask.sum():>6}   │")
    print(f"   └─────────────────┴────────────┴────────────┘")

    # Save results
    output = {
        'method': 'SEQUENCE',
        'step': 'T3',
        'overall_accuracy': overall_acc,
        'overall_f1_macro': overall_f1_macro,
        'overall_f1_weighted': overall_f1_weighted,
        'known_accuracy': known_acc,
        'known_f1_macro': known_f1_macro,
        'known_f1_weighted': known_f1_weighted,
        'new_accuracy': new_acc,
        'new_f1_macro': new_f1_macro,
        'new_f1_weighted': new_f1_weighted,
        'forgetting': {
            'baseline_acc': baseline_acc,
            't1_acc': t1_acc,
            't2_acc': t2_acc,
            't3_acc': t3_acc
        },
        'per_class_accuracy': {CLASS_NAMES.get(k, str(k)): v for k, v in per_class_acc.items()},
        'per_class_f1': {CLASS_NAMES.get(k, str(k)): v for k, v in per_class_f1.items()},
        'mapping': {str(k): v for k, v in mapping.items()},
        'avg_similarity': avg_sim,
        'class_names': CLASS_NAMES
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("✅ T3 EVALUATION COMPLETED!")
    print('=' * 70)

    print(f"\n📊 SEMANTIC MAPPING (T3):")
    for pseudo, gt in mapping.items():
        print(f"   Pseudo {pseudo} → GT {gt} ({CLASS_NAMES.get(gt, '?')})")

    return output


if __name__ == '__main__':
    evaluate()