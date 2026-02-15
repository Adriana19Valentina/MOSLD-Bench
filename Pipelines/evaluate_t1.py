import os
import sys
import json
import torch
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import *


MODEL_PATH = "./english_cl_outputs_1/model_t1"
RESULTS_PATH = "./english_cl_outputs_1/test_1_results.pkl"
TEST_DATA_PATH = TEST_1_CSV
OUTPUT_PATH = "./english_cl_outputs_1/eval_t1_results.json"
EMBEDDING_MODEL = MODEL_NAME


def load_classification_model(model_path):
    """Load the trained classification model."""
    print(f" Loading classification model from: {model_path}")
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
    """Load model for computing embeddings."""
    print(f"\n Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def get_embedding(model, tokenizer, text):
    """Get embedding for a single text."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding[0]


def compute_semantic_similarity_sequence(embed_model, tokenizer, keywords_list, class_names_list):
    """
    Compute semantic similarity using SEQUENCE method only.
    Concatenates keywords into a sequence and computes embedding.
    """
    print(f"\n Computing semantic similarity (SEQUENCE method)...")
    print(f"   Clusters: {len(keywords_list)}")
    print(f"   Target classes: {class_names_list}")

    n_clusters = len(keywords_list)
    n_classes = len(class_names_list)

    # Sequence embedding: concatenated keywords
    print(f"\n   Computing cluster embeddings (concatenated keywords)...")
    cluster_embeddings = []
    for i, keywords in enumerate(keywords_list):
        seq = ' '.join(keywords[:10])
        seq_embed = get_embedding(embed_model, tokenizer, seq)
        cluster_embeddings.append(seq_embed)
        print(f"      Cluster {i}: '{seq[:60]}...'")

    # Class name embeddings
    print(f"\n   Computing class name embeddings...")
    class_embeddings = []
    for class_name in class_names_list:
        embed = get_embedding(embed_model, tokenizer, class_name)
        class_embeddings.append(embed)
        print(f"      '{class_name}'")

    # Compute similarity matrix
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_matrix = np.zeros((n_clusters, n_classes))
    for i in range(n_clusters):
        for j in range(n_classes):
            sim_matrix[i, j] = cosine_sim(cluster_embeddings[i], class_embeddings[j])

    # Print matrix
    print(f"\n{'=' * 60}")
    print(f" SIMILARITY MATRIX (SEQUENCE):")
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
    """Use Hungarian algorithm to find optimal mapping."""
    if similarity_matrix.size == 0:
        print("   Empty similarity matrix!")
        mapping = {p: g for p, g in zip(pseudo_labels, gt_labels)}
        return mapping, 0.0

    cost_matrix = -similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    print(f"\nHungarian Mapping Result:")

    total_sim = 0
    for i, j in zip(row_ind, col_ind):
        pseudo = pseudo_labels[i]
        gt = gt_labels[j]
        sim = similarity_matrix[i, j]
        total_sim += sim

        gt_name = class_names.get(gt, str(gt))
        mapping[pseudo] = gt
        print(f"   Pseudo {pseudo} → GT {gt} ({gt_name}), similarity={sim:.4f}")

    avg_sim = total_sim / len(row_ind) if len(row_ind) > 0 else 0.0
    print(f"   Average similarity: {avg_sim:.4f}")

    return mapping, avg_sim


def predict_batch(model, tokenizer, texts, batch_size=32):
    """Make predictions for a batch of texts."""
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


def evaluate_with_mapping(predictions, gt_labels, mapping, id2label, baseline_labels, new_labels, class_names):
    """
    Evaluate predictions with detailed metrics for known and new classes.

    Returns dict with:
    - overall_acc, overall_f1
    - known_acc, known_f1
    - new_acc, new_f1
    - per_class metrics
    """
    print(f"\n{'=' * 60}")
    print(f" EVALUATION")
    print('=' * 60)

    # Build full mapping: model_id → GT_label
    full_mapping = {}
    for model_id, pseudo in id2label.items():
        model_id = int(model_id)
        pseudo = int(pseudo)

        if pseudo in baseline_labels:
            full_mapping[model_id] = pseudo
        elif pseudo in mapping:
            full_mapping[model_id] = mapping[pseudo]

    print(f"\n FULL MAPPING (model_id → GT):")
    for mid, gt in sorted(full_mapping.items()):
        pseudo = id2label.get(mid, '?')
        gt_name = class_names.get(gt, str(gt))
        print(f"   model_id {mid} → pseudo {pseudo} → GT {gt} ({gt_name})")

    # Map predictions
    mapped_preds = []
    for p in predictions:
        if p in full_mapping:
            mapped_preds.append(full_mapping[p])
        else:
            mapped_preds.append(-1)

    mapped_preds = np.array(mapped_preds)
    valid_mask = mapped_preds >= 0


    overall_acc = accuracy_score(gt_labels[valid_mask], mapped_preds[valid_mask])
    overall_f1 = f1_score(gt_labels[valid_mask], mapped_preds[valid_mask], average='macro', zero_division=0)
    overall_f1_weighted = f1_score(gt_labels[valid_mask], mapped_preds[valid_mask], average='weighted', zero_division=0)

    known_mask = np.isin(gt_labels, baseline_labels) & valid_mask
    if known_mask.sum() > 0:
        known_acc = accuracy_score(gt_labels[known_mask], mapped_preds[known_mask])
        # known_f1 = f1_score(gt_labels[known_mask], mapped_preds[known_mask], average='macro', zero_division=0)
        known_f1= f1_score(
            gt_labels[known_mask],
            mapped_preds[known_mask],
            labels=BASELINE_LABELS,
            average='macro',
            zero_division=0
        )
        known_f1_weighted = f1_score(gt_labels[known_mask], mapped_preds[known_mask], average='weighted',
                                     zero_division=0)
    else:
        known_acc = known_f1 = known_f1_weighted = 0.0


    new_mask = np.isin(gt_labels, new_labels) & valid_mask
    if new_mask.sum() > 0:
        new_acc = accuracy_score(gt_labels[new_mask], mapped_preds[new_mask])
        # new_f1 = f1_score(gt_labels[new_mask], mapped_preds[new_mask], average='macro', zero_division=0)
        new_f1 = f1_score(
            gt_labels[new_mask],
            mapped_preds[new_mask],
            labels=TEST_1_NEW_LABELS,
            average='macro',
            zero_division=0
        )
        new_f1_weighted = f1_score(gt_labels[new_mask], mapped_preds[new_mask], average='weighted', zero_division=0)
    else:
        new_acc = new_f1 = new_f1_weighted = 0.0

    # =========================================================================
    # PER-CLASS METRICS
    # =========================================================================
    per_class_acc = {}
    per_class_f1 = {}

    # Known classes
    print(f"\n Per-class metrics (KNOWN):")
    for label in baseline_labels:
        mask = (gt_labels == label) & valid_mask
        if mask.sum() > 0:
            acc = accuracy_score(gt_labels[mask], mapped_preds[mask])
            # F1 for single class
            binary_gt = (gt_labels[valid_mask] == label).astype(int)
            binary_pred = (mapped_preds[valid_mask] == label).astype(int)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            per_class_acc[label] = acc
            per_class_f1[label] = f1
            print(f"   {label} ({class_names.get(label, '?')}): acc={acc:.4f}, f1={f1:.4f} (n={mask.sum()})")

    # New classes
    print(f"\n Per-class metrics (NEW):")
    for label in new_labels:
        mask = (gt_labels == label) & valid_mask
        if mask.sum() > 0:
            acc = accuracy_score(gt_labels[mask], mapped_preds[mask])
            binary_gt = (gt_labels[valid_mask] == label).astype(int)
            binary_pred = (mapped_preds[valid_mask] == label).astype(int)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            per_class_acc[label] = acc
            per_class_f1[label] = f1
            print(f"   {label} ({class_names.get(label, '?')}): acc={acc:.4f}, f1={f1:.4f} (n={mask.sum()})")

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f" RESULTS SUMMARY")
    print('=' * 60)

    print(f"\n   OVERALL:")
    print(f"      Accuracy:    {overall_acc:.4f}")
    print(f"      F1 (macro):  {overall_f1:.4f}")
    print(f"      F1 (weighted): {overall_f1_weighted:.4f}")

    print(f"\n   KNOWN CLASSES ({len(baseline_labels)} classes, {known_mask.sum()} samples):")
    print(f"      Accuracy:    {known_acc:.4f}")
    print(f"      F1 (macro):  {known_f1:.4f}")
    print(f"      F1 (weighted): {known_f1_weighted:.4f}")

    print(f"\n   NEW CLASSES ({len(new_labels)} classes, {new_mask.sum()} samples):")
    print(f"      Accuracy:    {new_acc:.4f}")
    print(f"      F1 (macro):  {new_f1:.4f}")
    print(f"      F1 (weighted): {new_f1_weighted:.4f}")

    return {
        'overall_acc': overall_acc,
        'overall_f1_macro': overall_f1,
        'overall_f1_weighted': overall_f1_weighted,
        'known_acc': known_acc,
        'known_f1_macro': known_f1,
        'known_f1_weighted': known_f1_weighted,
        'new_acc': new_acc,
        'new_f1_macro': new_f1,
        'new_f1_weighted': new_f1_weighted,
        'per_class_acc': per_class_acc,
        'per_class_f1': per_class_f1,
        'mapping': mapping,
        'full_mapping': full_mapping
    }


def evaluate():
    """Main evaluation function."""
    print("=" * 70)
    print("EVALUATION - T1 (SEQUENCE METHOD)")
    print("=" * 70)

    # Step 1: Load classification model
    print(f"\n{'=' * 70}")
    print("STEP 1: LOADING CLASSIFICATION MODEL")
    print("=" * 70)
    clf_model, clf_tokenizer, id2label = load_classification_model(MODEL_PATH)

    # Step 2: Load embedding model
    print(f"\n{'=' * 70}")
    print("STEP 2: LOADING EMBEDDING MODEL")
    print("=" * 70)
    embed_model, embed_tokenizer = load_embedding_model(EMBEDDING_MODEL)

    # Step 3: Load clustering results
    print(f"\n{'=' * 70}")
    print("STEP 3: LOADING CLUSTERING RESULTS")
    print("=" * 70)
    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)

    cluster_keywords = results.get('cluster_keywords', {})
    cluster_keywords_unique = results.get('cluster_keywords_unique', {})
    cluster_keywords = {int(k) if isinstance(k, str) else k: v for k, v in cluster_keywords.items()}
    cluster_keywords_unique = {int(k) if isinstance(k, str) else k: v for k, v in cluster_keywords_unique.items()}
    pseudo_label_mapping = results.get('pseudo_label_mapping', {})

    # Use unique keywords if available
    use_unique = len(cluster_keywords_unique) > 0 and all(len(v) > 0 for v in cluster_keywords_unique.values())
    keywords_to_use = cluster_keywords_unique if use_unique else cluster_keywords
    print(f"   Using: {'UNIQUE' if use_unique else 'regular'} keywords")

    # Prepare keywords list
    keywords_list = []
    pseudo_labels_ordered = []

    if pseudo_label_mapping:
        for cluster_id in sorted(pseudo_label_mapping.keys()):
            pseudo = pseudo_label_mapping[cluster_id]
            pseudo_labels_ordered.append(pseudo)
            kw = keywords_to_use.get(cluster_id, [])
            keywords_list.append(kw)
            print(f"   Cluster {cluster_id} → Pseudo {pseudo}: {kw[:5]}...")
    else:
        for key in sorted(keywords_to_use.keys()):
            kw = keywords_to_use[key]
            keywords_list.append(kw)
            pseudo = 109 + len(pseudo_labels_ordered)
            pseudo_labels_ordered.append(pseudo)

    # Step 4: Load test data
    print(f"\n{'=' * 70}")
    print("STEP 4: LOADING TEST DATA")
    print("=" * 70)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Loaded {len(test_df)} samples")

    new_labels = TEST_1_NEW_LABELS
    new_class_names = [CLASS_NAMES[l] for l in new_labels]
    print(f"   NEW labels: {new_labels}")
    print(f"   NEW class names: {new_class_names}")

    # Step 5: Make predictions
    print(f"\n{'=' * 70}")
    print("STEP 5: MAKING PREDICTIONS")
    print("=" * 70)
    predictions = predict_batch(clf_model, clf_tokenizer, test_df['content'].tolist())
    print(f"   Unique predictions (model_ids): {sorted(set(predictions))}")

    # Step 6: Semantic Hungarian mapping (SEQUENCE only)
    print(f"\n{'=' * 70}")
    print("STEP 6: SEMANTIC HUNGARIAN MAPPING")
    print("=" * 70)

    sim_matrix = compute_semantic_similarity_sequence(
        embed_model, embed_tokenizer,
        keywords_list,
        new_class_names
    )

    mapping, avg_sim = hungarian_mapping(
        sim_matrix, pseudo_labels_ordered, new_labels, CLASS_NAMES
    )

    # Step 7: Evaluate
    print(f"\n{'=' * 70}")
    print("STEP 7: EVALUATION")
    print("=" * 70)

    gt_labels = test_df['label'].values
    results = evaluate_with_mapping(
        predictions, gt_labels, mapping, id2label,
        BASELINE_LABELS, new_labels, CLASS_NAMES
    )

    # Save results
    output = {
        'method': 'SEQUENCE',
        'overall_accuracy': results['overall_acc'],
        'overall_f1_macro': results['overall_f1_macro'],
        'overall_f1_weighted': results['overall_f1_weighted'],
        'known_accuracy': results['known_acc'],
        'known_f1_macro': results['known_f1_macro'],
        'known_f1_weighted': results['known_f1_weighted'],
        'new_accuracy': results['new_acc'],
        'new_f1_macro': results['new_f1_macro'],
        'new_f1_weighted': results['new_f1_weighted'],
        'per_class_accuracy': {CLASS_NAMES.get(k, str(k)): v for k, v in results['per_class_acc'].items()},
        'per_class_f1': {CLASS_NAMES.get(k, str(k)): v for k, v in results['per_class_f1'].items()},
        'mapping': {str(k): v for k, v in results['mapping'].items()},
        'avg_similarity': avg_sim,
        'class_names': CLASS_NAMES
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETED!")
    print('=' * 70)

    print(f"\n FINAL METRICS:")
    print(f"   ┌─────────────────┬────────────┬────────────┬────────────┐")
    print(f"   │                 │  Accuracy  │  F1-Macro  │ F1-Weighted│")
    print(f"   ├─────────────────┼────────────┼────────────┼────────────┤")
    print(
        f"   │ OVERALL         │   {results['overall_acc']:.4f}   │   {results['overall_f1_macro']:.4f}   │   {results['overall_f1_weighted']:.4f}   │")
    print(
        f"   │ KNOWN           │   {results['known_acc']:.4f}   │   {results['known_f1_macro']:.4f}   │   {results['known_f1_weighted']:.4f}   │")
    print(
        f"   │ NEW             │   {results['new_acc']:.4f}   │   {results['new_f1_macro']:.4f}   │   {results['new_f1_weighted']:.4f}   │")
    print(f"   └─────────────────┴────────────┴────────────┴────────────┘")

    return results


if __name__ == '__main__':
    results = evaluate()