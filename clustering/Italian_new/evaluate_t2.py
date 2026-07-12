#!/usr/bin/env python3
# evaluate_t2_semantic.py - Evaluation with proper CLASS_NAMES for semantic mapping
#
# T2 evaluates: baseline (0-3) + T1 discovered (4-5) + T2 new (6-7: culture, climate)

import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from config import *
# =========================================================================
# CONFIGURATION
# =========================================================================

MODEL_PATH = "./italian_cl_outputs_1/model_t2"
RESULTS_PATH = "./italian_cl_outputs_1/test_2_results.pkl"
TEST_DATA_PATH = TEST_2_CSV
OUTPUT_PATH = "./italian_cl_outputs_1/eval_t2_results.json"

EMBEDDING_MODEL = MODEL_NAME


# All known labels at T2 (baseline + T1)
KNOWN_LABELS_T2 = BASELINE_LABELS + TEST_1_NEW_LABELS



# =========================================================================


def load_classification_model(model_path):
    print(f"📥 Loading classification model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # Convert id2label to int
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


def compute_semantic_similarity(embed_model, tokenizer, keywords_list, class_names_list):
    print(f"\n📊 Computing semantic similarity...")
    print(f"   Clusters: {len(keywords_list)}")
    print(f"   Target classes: {class_names_list}")

    n_clusters = len(keywords_list)
    n_classes = len(class_names_list)

    if n_clusters == 0:
        return np.array([]), np.array([])

    # Method 1: Mean of keyword embeddings
    cluster_embeddings_mean = []
    for i, keywords in enumerate(keywords_list):
        if keywords:
            keyword_embeds = [get_embedding(embed_model, tokenizer, kw) for kw in keywords[:10]]
            mean_embed = np.mean(keyword_embeds, axis=0)
        else:
            mean_embed = np.zeros(768)
        cluster_embeddings_mean.append(mean_embed)
        print(f"   Cluster {i}: {keywords[:5] if keywords else 'NO KEYWORDS'}...")

    # Method 2: Sequence
    cluster_embeddings_seq = []
    for keywords in keywords_list:
        if keywords:
            seq = ' '.join(keywords[:10])
            seq_embed = get_embedding(embed_model, tokenizer, seq)
        else:
            seq_embed = np.zeros(768)
        cluster_embeddings_seq.append(seq_embed)

    # Class name embeddings
    class_embeddings = [get_embedding(embed_model, tokenizer, name) for name in class_names_list]

    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    sim_matrix_mean = np.zeros((n_clusters, n_classes))
    sim_matrix_seq = np.zeros((n_clusters, n_classes))

    for i in range(n_clusters):
        for j in range(n_classes):
            sim_matrix_mean[i, j] = cosine_sim(cluster_embeddings_mean[i], class_embeddings[j])
            sim_matrix_seq[i, j] = cosine_sim(cluster_embeddings_seq[i], class_embeddings[j])

    # Print matrices
    print(f"\n{'=' * 60}")
    print(f"📊 SIMILARITY MATRIX (MEAN method):")
    print('=' * 60)
    header = "            " + "".join([f"{name:>12}" for name in class_names_list])
    print(header)
    for i in range(n_clusters):
        row = f"Cluster {i:2d} |"
        for j in range(n_classes):
            row += f"{sim_matrix_mean[i, j]:>12.4f}"
        print(row)

    print(f"\n{'=' * 60}")
    print(f"📊 SIMILARITY MATRIX (SEQUENCE method):")
    print('=' * 60)
    print(header)
    for i in range(n_clusters):
        row = f"Cluster {i:2d} |"
        for j in range(n_classes):
            row += f"{sim_matrix_seq[i, j]:>12.4f}"
        print(row)

    return sim_matrix_mean, sim_matrix_seq


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


def compute_metrics(y_true, y_pred, labels, class_names):
    """Compute comprehensive metrics."""
    metrics = {}

    # Overall metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    # Per-class metrics
    metrics['per_class'] = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            class_name = class_names.get(label, str(label))
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)

            metrics['per_class'][class_name] = {
                'accuracy': float(accuracy_score(y_true[mask], y_pred[mask])),
                'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                'support': int(mask.sum())
            }

    return metrics


def evaluate():
    print("=" * 70)
    print("T2 EVALUATION WITH SEMANTIC CLASS NAMES")
    print("(culture, climate)")
    print("=" * 70)

    # Load models
    print(f"\n{'=' * 70}")
    print("STEP 1: LOADING MODELS")
    print("=" * 70)

    clf_model, clf_tokenizer, id2label = load_classification_model(MODEL_PATH)
    embed_model, embed_tokenizer = load_embedding_model(EMBEDDING_MODEL)

    # Load clustering results
    print(f"\n{'=' * 70}")
    print("STEP 2: LOADING CLUSTERING RESULTS")
    print("=" * 70)

    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)

    print(f"   Results keys: {list(results.keys())}")

    cluster_keywords = results.get('cluster_keywords', {})
    cluster_keywords_unique = results.get('cluster_keywords_unique', {})
    pseudo_label_mapping = results.get('cluster_to_pseudo', {})

    print(f"   pseudo_label_mapping: {pseudo_label_mapping}")

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
            print(f"   Cluster {cluster_id} → Pseudo {pseudo}: {kw[:5] if kw else 'NO KEYWORDS'}...")
    else:
        # Fallback
        for key in sorted(keywords_to_use.keys()):
            kw = keywords_to_use[key]
            keywords_list.append(kw)
            pseudo = 209 + len(pseudo_labels_ordered)  # T2 starts at 209
            pseudo_labels_ordered.append(pseudo)

    # Load test data
    print(f"\n{'=' * 70}")
    print("STEP 3: LOADING TEST DATA")
    print("=" * 70)

    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Loaded {len(test_df)} samples")
    print(f"   Labels in test: {sorted(test_df['label'].unique())}")

    # T2 NEW class names
    new_labels = TEST_2_NEW_LABELS
    new_class_names = [CLASS_NAMES[l] for l in new_labels]
    print(f"   T2 NEW labels: {new_labels}")
    print(f"   T2 NEW class names: {new_class_names}")

    # Make predictions
    print(f"\n{'=' * 70}")
    print("STEP 4: MAKING PREDICTIONS")
    print("=" * 70)

    print(f"   Predicting on {len(test_df)} samples...")
    predictions = predict_batch(clf_model, clf_tokenizer, test_df['content'].tolist())
    # predictions_mapped = np.array([id2label[p] for p in predictions])

    predictions_mapped = predictions
    print(f"   Unique predictions: {sorted(set(predictions_mapped))}")

    # Semantic similarity for T2 NEW
    print(f"\n{'=' * 70}")
    print("STEP 5: SEMANTIC HUNGARIAN MAPPING")
    print("=" * 70)

    pseudo_labels = pseudo_labels_ordered
    print(f"   Pseudo-labels to map: {pseudo_labels}")
    print(f"   Target GT classes: {new_labels}")
    print(f"   Target class NAMES: {new_class_names}")

    sim_mean, sim_seq = compute_semantic_similarity(
        embed_model, embed_tokenizer,
        keywords_list,
        new_class_names
    )

    print(f"\n--- Method 1: MEAN ---")
    mapping_mean, avg_sim_mean = hungarian_mapping(sim_mean, pseudo_labels, new_labels, CLASS_NAMES)

    print(f"\n--- Method 2: SEQUENCE ---")
    mapping_seq, avg_sim_seq = hungarian_mapping(sim_seq, pseudo_labels, new_labels, CLASS_NAMES)

    # Compare mappings
    print(f"\n{'=' * 60}")
    print("📊 MAPPING COMPARISON:")
    print('=' * 60)
    mappings_match = mapping_mean == mapping_seq
    print(f"   Mappings identical: {'✅ YES' if mappings_match else '❌ NO'}")
    print(f"   Avg similarity (MEAN):     {avg_sim_mean:.4f}")
    print(f"   Avg similarity (SEQUENCE): {avg_sim_seq:.4f}")
    if not mappings_match:
        print(f"   MEAN mapping: {mapping_mean}")
        print(f"   SEQ mapping:  {mapping_seq}")

    # Evaluate with both mappings
    print(f"\n{'=' * 70}")
    print("STEP 6: EVALUATION")
    print("=" * 70)

    gt_labels_all = test_df['label'].values
    all_labels_in_test = sorted(test_df['label'].unique())

    def evaluate_with_mapping(predictions, gt_labels, mapping, method_name):
        # Build full mapping
        # full_mapping = {l: l for l in KNOWN_LABELS_T2}
        # full_mapping.update(mapping)
        full_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

        # Map predictions

        mapped_preds = []
        for p in predictions:
            if p in full_mapping:
                mapped_preds.append(full_mapping[p])
            else:
                mapped_preds.append(-1)
        mapped_preds = np.array(mapped_preds)

        # Filter valid
        valid_mask = mapped_preds >= 0
        y_true = gt_labels[valid_mask]
        y_pred = mapped_preds[valid_mask]

        # Overall metrics
        overall_metrics = compute_metrics(y_true, y_pred, all_labels_in_test, CLASS_NAMES)

        # Known vs New split
        known_mask = np.isin(gt_labels, KNOWN_LABELS_T2) & valid_mask
        new_mask = np.isin(gt_labels, new_labels) & valid_mask

        # Known classes metrics
        known_metrics = {}
        if known_mask.sum() > 0:
            known_metrics = compute_metrics(gt_labels[known_mask], mapped_preds[known_mask], KNOWN_LABELS_T2,
                                            CLASS_NAMES)

        # New classes metrics
        new_metrics = {}
        if new_mask.sum() > 0:
            new_metrics = compute_metrics(gt_labels[new_mask], mapped_preds[new_mask], new_labels, CLASS_NAMES)

        return {
            'overall': overall_metrics,
            'known_metrics': known_metrics,
            'new_metrics': new_metrics,
            'mapping': mapping
        }

    results_mean = evaluate_with_mapping(predictions_mapped, gt_labels_all, mapping_mean, "MEAN")
    results_seq = evaluate_with_mapping(predictions_mapped, gt_labels_all, mapping_seq, "SEQUENCE")

    # Print results
    for method_name, res in [("MEAN", results_mean), ("SEQUENCE", results_seq)]:
        print(f"\n{'=' * 60}")
        print(f"📊 METHOD: {method_name}")
        print('=' * 60)
        print(f"   OVERALL:")
        print(f"      Accuracy:           {res['overall']['accuracy']:.4f}")
        print(f"      F1 (weighted):      {res['overall']['f1_weighted']:.4f}")
        print(f"      F1 (macro):         {res['overall']['f1_macro']:.4f}")
        print(f"      Precision (weighted): {res['overall']['precision_weighted']:.4f}")
        print(f"      Recall (weighted):    {res['overall']['recall_weighted']:.4f}")

        if res['known_metrics']:
            print(f"\n   KNOWN CLASSES ({[CLASS_NAMES[l] for l in KNOWN_LABELS_T2]}):")
            print(f"      Accuracy:           {res['known_metrics']['accuracy']:.4f}")
            print(f"      F1 (weighted):      {res['known_metrics']['f1_weighted']:.4f}")
            print(f"      F1 (macro):         {res['known_metrics']['f1_macro']:.4f}")
            print(f"      Precision (weighted): {res['known_metrics']['precision_weighted']:.4f}")
            print(f"      Recall (weighted):    {res['known_metrics']['recall_weighted']:.4f}")

        if res['new_metrics']:
            print(f"\n   NEW CLASSES ({new_class_names}):")
            print(f"      Accuracy:           {res['new_metrics']['accuracy']:.4f}")
            print(f"      F1 (weighted):      {res['new_metrics']['f1_weighted']:.4f}")
            print(f"      F1 (macro):         {res['new_metrics']['f1_macro']:.4f}")
            print(f"      Precision (weighted): {res['new_metrics']['precision_weighted']:.4f}")
            print(f"      Recall (weighted):    {res['new_metrics']['recall_weighted']:.4f}")

            print(f"\n      Per-class breakdown:")
            for class_name, metrics in res['new_metrics'].get('per_class', {}).items():
                print(f"         {class_name}:")
                print(f"            Accuracy:  {metrics['accuracy']:.4f}")
                print(f"            Precision: {metrics['precision']:.4f}")
                print(f"            Recall:    {metrics['recall']:.4f}")
                print(f"            F1:        {metrics['f1']:.4f}")

    # Select best method based on new accuracy
    best_method = "SEQUENCE" if results_seq['new_metrics'].get('accuracy', 0) >= results_mean['new_metrics'].get(
        'accuracy', 0) else "MEAN"
    best_results = results_seq if best_method == "SEQUENCE" else results_mean
    best_mapping = mapping_seq if best_method == "SEQUENCE" else mapping_mean

    print(f"\n{'=' * 60}")
    print(f"🏆 BEST METHOD: {best_method}")
    print('=' * 60)

    # Save comprehensive results
    output = {
        'step': 'T2',
        'new_classes': new_class_names,
        'known_classes': [CLASS_NAMES[l] for l in KNOWN_LABELS_T2],
        'best_method': best_method,
        'mapping': {str(k): v for k, v in best_mapping.items()},
        'mapping_readable': {str(k): CLASS_NAMES[v] for k, v in best_mapping.items()},

        'overall_metrics': {
            'accuracy': best_results['overall']['accuracy'],
            'f1_weighted': best_results['overall']['f1_weighted'],
            'f1_macro': best_results['overall']['f1_macro'],
            'precision_weighted': best_results['overall']['precision_weighted'],
            'precision_macro': best_results['overall']['precision_macro'],
            'recall_weighted': best_results['overall']['recall_weighted'],
            'recall_macro': best_results['overall']['recall_macro'],
        },

        'known_metrics': {
            'accuracy': best_results['known_metrics'].get('accuracy', 0),
            'f1_weighted': best_results['known_metrics'].get('f1_weighted', 0),
            'f1_macro': best_results['known_metrics'].get('f1_macro', 0),
            'precision_weighted': best_results['known_metrics'].get('precision_weighted', 0),
            'precision_macro': best_results['known_metrics'].get('precision_macro', 0),
            'recall_weighted': best_results['known_metrics'].get('recall_weighted', 0),
            'recall_macro': best_results['known_metrics'].get('recall_macro', 0),
        } if best_results['known_metrics'] else {},

        'new_metrics': {
            'accuracy': best_results['new_metrics'].get('accuracy', 0),
            'f1_weighted': best_results['new_metrics'].get('f1_weighted', 0),
            'f1_macro': best_results['new_metrics'].get('f1_macro', 0),
            'precision_weighted': best_results['new_metrics'].get('precision_weighted', 0),
            'precision_macro': best_results['new_metrics'].get('precision_macro', 0),
            'recall_weighted': best_results['new_metrics'].get('recall_weighted', 0),
            'recall_macro': best_results['new_metrics'].get('recall_macro', 0),
        } if best_results['new_metrics'] else {},

        'per_class_metrics': best_results['overall']['per_class'],
        'known_per_class': best_results['known_metrics'].get('per_class', {}) if best_results['known_metrics'] else {},
        'new_per_class': best_results['new_metrics'].get('per_class', {}) if best_results['new_metrics'] else {},

        'comparison': {
            'mean': {
                'new_accuracy': results_mean['new_metrics'].get('accuracy', 0) if results_mean['new_metrics'] else 0,
                'avg_similarity': avg_sim_mean,
                'mapping': {str(k): v for k, v in mapping_mean.items()}
            },
            'sequence': {
                'new_accuracy': results_seq['new_metrics'].get('accuracy', 0) if results_seq['new_metrics'] else 0,
                'avg_similarity': avg_sim_seq,
                'mapping': {str(k): v for k, v in mapping_seq.items()}
            }
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("✅ T2 EVALUATION COMPLETED!")
    print('=' * 70)
    print(f"\n📊 SEMANTIC MAPPING:")
    for pseudo, gt in best_mapping.items():
        print(f"   Pseudo {pseudo} → GT {gt} ({CLASS_NAMES[gt]})")

    print(f"\n📊 FINAL METRICS:")
    print(f"   OVERALL:")
    print(f"      Accuracy:      {best_results['overall']['accuracy']:.4f}")
    print(f"      F1 (weighted): {best_results['overall']['f1_weighted']:.4f}")
    print(f"      Precision:     {best_results['overall']['precision_weighted']:.4f}")
    print(f"      Recall:        {best_results['overall']['recall_weighted']:.4f}")

    if best_results['known_metrics']:
        print(f"\n   KNOWN:")
        print(f"      Accuracy:      {best_results['known_metrics']['accuracy']:.4f}")
        print(f"      F1 (weighted): {best_results['known_metrics']['f1_weighted']:.4f}")
        print(f"      Precision:     {best_results['known_metrics']['precision_weighted']:.4f}")
        print(f"      Recall:        {best_results['known_metrics']['recall_weighted']:.4f}")

    if best_results['new_metrics']:
        print(f"\n   NEW (T2):")
        print(f"      Accuracy:      {best_results['new_metrics']['accuracy']:.4f}")
        print(f"      F1 (weighted): {best_results['new_metrics']['f1_weighted']:.4f}")
        print(f"      Precision:     {best_results['new_metrics']['precision_weighted']:.4f}")
        print(f"      Recall:        {best_results['new_metrics']['recall_weighted']:.4f}")

    return output


if __name__ == '__main__':
    evaluate()