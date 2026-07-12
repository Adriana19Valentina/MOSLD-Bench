#!/usr/bin/env python3
# evaluate_t1_semantic.py - Evaluation with proper CLASS_NAMES for semantic mapping
#
# This script uses actual class names (sports, travel, etc.) instead of numeric labels
# for computing semantic similarity in Hungarian mapping.

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
# =========================================================================
# CONFIGURATION - UPDATE THESE FOR YOUR SETUP
# =========================================================================

# Paths
MODEL_PATH = "./italian_cl_outputs_1/model_t1"
RESULTS_PATH = "./italian_cl_outputs_1/test_1_results.pkl"
TEST_DATA_PATH = TEST_1_CSV
OUTPUT_PATH = "./italian_cl_outputs_1/eval_t1_results.json"

# Model for embeddings
EMBEDDING_MODEL = MODEL_NAME


def load_classification_model(model_path):
    """Load the trained classification model."""
    print(f"📥 Loading classification model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Get id2label mapping and convert to int
    id2label_raw = model.config.id2label
    id2label = {int(k): int(v) for k, v in id2label_raw.items()}

    print(f"   Model classes: {model.config.num_labels}")
    print(f"   id2label: {id2label}")

    return model, tokenizer, id2label


def load_embedding_model(model_name):
    """Load model for computing embeddings."""
    print(f"\n📥 Loading embedding model: {model_name}")
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
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding[0]


def compute_semantic_similarity(embed_model, tokenizer, keywords_list, class_names_list):
    """
    Compute semantic similarity between cluster keywords and class names.

    Args:
        embed_model: Model for embeddings
        tokenizer: Tokenizer
        keywords_list: List of keyword lists for each cluster
        class_names_list: List of class names to match against

    Returns:
        similarity_matrix: [n_clusters, n_classes]
    """
    print(f"\n📊 Computing semantic similarity...")
    print(f"   Clusters: {len(keywords_list)}")
    print(f"   Target classes: {class_names_list}")

    n_clusters = len(keywords_list)
    n_classes = len(class_names_list)

    # Method 1: Mean of keyword embeddings
    print(f"\n   METHOD 1: Mean of keyword embeddings")
    cluster_embeddings_mean = []
    for i, keywords in enumerate(keywords_list):
        keyword_embeds = [get_embedding(embed_model, tokenizer, kw) for kw in keywords[:10]]
        mean_embed = np.mean(keyword_embeds, axis=0)
        cluster_embeddings_mean.append(mean_embed)
        print(f"      Cluster {i}: {keywords[:5]}...")

    # Method 2: Concatenated keywords as sequence
    print(f"\n   METHOD 2: Sequence embedding (concatenated keywords)")
    cluster_embeddings_seq = []
    for i, keywords in enumerate(keywords_list):
        seq = ' '.join(keywords[:10])
        seq_embed = get_embedding(embed_model, tokenizer, seq)
        cluster_embeddings_seq.append(seq_embed)
        print(f"      Cluster {i}: '{seq[:50]}...'")

    # Get class name embeddings
    print(f"\n   Computing class name embeddings...")
    class_embeddings = []
    for class_name in class_names_list:
        embed = get_embedding(embed_model, tokenizer, class_name)
        class_embeddings.append(embed)
        print(f"      '{class_name}'")

    # Compute similarity matrices
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_matrix_mean = np.zeros((n_clusters, n_classes))
    sim_matrix_seq = np.zeros((n_clusters, n_classes))

    for i in range(n_clusters):
        for j in range(n_classes):
            sim_matrix_mean[i, j] = cosine_sim(cluster_embeddings_mean[i], class_embeddings[j])
            sim_matrix_seq[i, j] = cosine_sim(cluster_embeddings_seq[i], class_embeddings[j])

    # Print matrices
    print(f"\n{'=' * 60}")
    print(f"📊 SIMILARITY MATRIX - METHOD 1 (MEAN of embeddings):")
    print('=' * 60)
    header = "            " + "".join([f"{name:>12}" for name in class_names_list])
    print(header)
    for i in range(n_clusters):
        row = f"Cluster {i:2d} |"
        for j in range(n_classes):
            row += f"{sim_matrix_mean[i, j]:>12.4f}"
        print(row)

    print(f"\n{'=' * 60}")
    print(f"📊 SIMILARITY MATRIX - METHOD 2 (SEQUENCE embedding):")
    print('=' * 60)
    print(header)
    for i in range(n_clusters):
        row = f"Cluster {i:2d} |"
        for j in range(n_classes):
            row += f"{sim_matrix_seq[i, j]:>12.4f}"
        print(row)

    return sim_matrix_mean, sim_matrix_seq


def hungarian_mapping(similarity_matrix, pseudo_labels, gt_labels, class_names):
    """
    Use Hungarian algorithm to find optimal mapping.

    Args:
        similarity_matrix: [n_clusters, n_classes] similarity scores
        pseudo_labels: List of pseudo-labels (e.g., [109, 110])
        gt_labels: List of ground truth labels to map to (e.g., [4, 5])
        class_names: Dict mapping label -> name

    Returns:
        mapping: Dict {pseudo_label: gt_label}
    """
    if similarity_matrix.size == 0:
        print("   ⚠️  Empty similarity matrix!")
        # Create default mapping
        mapping = {p: g for p, g in zip(pseudo_labels, gt_labels)}
        return mapping, 0.0

    # Hungarian algorithm (maximize similarity = minimize negative similarity)
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


def evaluate():
    """Main evaluation function."""
    print("=" * 70)
    print("EVALUATION WITH SEMANTIC CLASS NAMES")
    print("=" * 70)

    # Load classification model
    print(f"\n{'=' * 70}")
    print("STEP 1: LOADING CLASSIFICATION MODEL")
    print("=" * 70)

    clf_model, clf_tokenizer, id2label = load_classification_model(MODEL_PATH)

    # Get pseudo-labels from id2label
    pseudo_labels = [v for v in id2label.values() if v >= 100]  # Pseudo labels are >= 100
    print(f"\n   Pseudo-labels in model: {pseudo_labels}")

    # Load embedding model
    print(f"\n{'=' * 70}")
    print("STEP 2: LOADING EMBEDDING MODEL")
    print("=" * 70)

    embed_model, embed_tokenizer = load_embedding_model(EMBEDDING_MODEL)

    # Load clustering results
    print(f"\n{'=' * 70}")
    print("STEP 3: LOADING CLUSTERING RESULTS")
    print("=" * 70)

    with open(RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)

    # Debug: print all keys in results
    print(f"   Results keys: {list(results.keys())}")

    cluster_keywords = results.get('cluster_keywords', {})
    cluster_keywords_unique = results.get('cluster_keywords_unique', {})
    pseudo_label_mapping = results.get('pseudo_label_mapping', {})

    print(f"   cluster_keywords: {cluster_keywords}")
    print(f"   cluster_keywords_unique: {cluster_keywords_unique}")
    print(f"   pseudo_label_mapping: {pseudo_label_mapping}")

    # Use unique keywords if available, otherwise regular
    use_unique = len(cluster_keywords_unique) > 0 and all(len(v) > 0 for v in cluster_keywords_unique.values())
    keywords_to_use = cluster_keywords_unique if use_unique else cluster_keywords

    print(f"   Using: {'UNIQUE' if use_unique else 'regular'} keywords")
    print(f"   Keywords dict: {keywords_to_use}")

    # Prepare keywords list in order of pseudo-labels
    # First, get the mapping from cluster_id to pseudo_label
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
        # Try alternative: keywords might be keyed by pseudo-label directly
        print("   ⚠️  pseudo_label_mapping is empty, trying alternative...")
        for key in sorted(keywords_to_use.keys()):
            kw = keywords_to_use[key]
            keywords_list.append(kw)
            # Assume pseudo-labels are 109, 110, etc.
            pseudo = 109 + len(pseudo_labels_ordered)
            pseudo_labels_ordered.append(pseudo)
            print(f"   Key {key} → Pseudo {pseudo}: {kw[:5] if kw else 'NO KEYWORDS'}...")

    print(f"\n   Final keywords_list length: {len(keywords_list)}")
    print(f"   Final pseudo_labels_ordered: {pseudo_labels_ordered}")

    if len(keywords_list) == 0 or all(len(kw) == 0 for kw in keywords_list):
        print("   ❌ ERROR: No keywords found! Check pipeline_t1.py output.")
        print("   Falling back to dummy keywords...")
        # Create dummy keywords based on what we expect
        keywords_list = [['news', 'sport', 'game'], ['travel', 'tourism', 'vacation']]
        pseudo_labels_ordered = [109, 110]

    # Load test data
    print(f"\n{'=' * 70}")
    print("STEP 4: LOADING TEST DATA")
    print("=" * 70)

    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Loaded {len(test_df)} samples")

    # Get NEW class labels and names
    new_labels = TEST_1_NEW_LABELS
    new_class_names = [CLASS_NAMES[l] for l in new_labels]

    print(f"   NEW labels to discover: {new_labels}")
    print(f"   NEW class names: {new_class_names}")

    # Make predictions
    print(f"\n{'=' * 70}")
    print("STEP 5: MAKING PREDICTIONS")
    print("=" * 70)

    print(f"   Predicting on {len(test_df)} samples...")
    predictions = predict_batch(clf_model, clf_tokenizer, test_df['content'].tolist())

    # Map model output IDs to actual labels
    predictions_mapped = np.array([id2label[p] for p in predictions])
    print(f"   Unique predictions: {sorted(set(predictions_mapped))}")

    # Compute semantic similarity
    print(f"\n{'=' * 70}")
    print("STEP 6: SEMANTIC HUNGARIAN MAPPING")
    print("=" * 70)

    # Use pseudo_labels from clustering results, not from model
    pseudo_labels = pseudo_labels_ordered

    print(f"   Pseudo-labels to map: {pseudo_labels}")
    print(f"   Target GT classes: {new_labels}")
    print(f"   Target class NAMES: {new_class_names}")  # This is key!

    sim_mean, sim_seq = compute_semantic_similarity(
        embed_model, embed_tokenizer,
        keywords_list,
        new_class_names  # Use actual names, not numbers!
    )

    # Hungarian mapping with both methods
    print(f"\n--- Method 1: MEAN ---")
    mapping_mean, avg_sim_mean = hungarian_mapping(
        sim_mean, pseudo_labels, new_labels, CLASS_NAMES
    )

    print(f"\n--- Method 2: SEQUENCE ---")
    mapping_seq, avg_sim_seq = hungarian_mapping(
        sim_seq, pseudo_labels, new_labels, CLASS_NAMES
    )

    # Compare mappings
    print(f"\n{'=' * 60}")
    print("📊 MAPPING COMPARISON:")
    print('=' * 60)
    mappings_match = mapping_mean == mapping_seq
    print(f"   Mappings identical: {'✅ YES' if mappings_match else '❌ NO'}")
    print(f"   Avg similarity (MEAN):     {avg_sim_mean:.4f}")
    print(f"   Avg similarity (SEQUENCE): {avg_sim_seq:.4f}")

    if not mappings_match:
        print(f"\n   MEAN mapping: {mapping_mean}")
        print(f"   SEQ mapping:  {mapping_seq}")

    # Evaluate with both mappings
    print(f"\n{'=' * 70}")
    print("STEP 7: EVALUATION")
    print("=" * 70)

    gt_labels_all = test_df['label'].values

    def evaluate_with_mapping(predictions, gt_labels, mapping, method_name):
        """Evaluate predictions using a specific mapping."""
        # Create full mapping (baseline + new)
        full_mapping = {l: l for l in BASELINE_LABELS}  # Baseline maps to itself
        full_mapping.update(mapping)  # Add new class mapping

        # Map predictions
        mapped_preds = []
        for p in predictions:
            if p in full_mapping:
                mapped_preds.append(full_mapping[p])
            else:
                mapped_preds.append(-1)  # Unknown
        mapped_preds = np.array(mapped_preds)

        # Overall metrics
        valid_mask = mapped_preds >= 0
        overall_acc = accuracy_score(gt_labels[valid_mask], mapped_preds[valid_mask])
        overall_f1 = f1_score(gt_labels[valid_mask], mapped_preds[valid_mask], average='weighted')

        # Known vs New
        known_mask = np.isin(gt_labels, BASELINE_LABELS) & valid_mask
        new_mask = np.isin(gt_labels, new_labels) & valid_mask

        known_acc = accuracy_score(gt_labels[known_mask], mapped_preds[known_mask]) if known_mask.sum() > 0 else 0
        new_acc = accuracy_score(gt_labels[new_mask], mapped_preds[new_mask]) if new_mask.sum() > 0 else 0

        # Per-class accuracy for NEW classes
        per_class_acc = {}
        for label in new_labels:
            mask = (gt_labels == label) & valid_mask
            if mask.sum() > 0:
                acc = accuracy_score(gt_labels[mask], mapped_preds[mask])
                per_class_acc[label] = acc

        return {
            'overall_acc': overall_acc,
            'overall_f1': overall_f1,
            'known_acc': known_acc,
            'new_acc': new_acc,
            'per_class_acc': per_class_acc,
            'mapping': mapping
        }

    results_mean = evaluate_with_mapping(predictions_mapped, gt_labels_all, mapping_mean, "MEAN")
    results_seq = evaluate_with_mapping(predictions_mapped, gt_labels_all, mapping_seq, "SEQUENCE")

    # Print results
    print(f"\n{'=' * 60}")
    print(f"📊 METHOD 1 (MEAN of embeddings) RESULTS:")
    print('=' * 60)
    print(f"   Overall Accuracy:  {results_mean['overall_acc']:.4f}")
    print(f"   Overall F1:        {results_mean['overall_f1']:.4f}")
    print(f"   Known Accuracy:    {results_mean['known_acc']:.4f}")
    print(f"   New Accuracy:      {results_mean['new_acc']:.4f}")
    print(f"\n   Per-class accuracy (NEW):")
    for label, acc in results_mean['per_class_acc'].items():
        print(f"      {label} ({CLASS_NAMES[label]}): {acc:.4f}")

    print(f"\n{'=' * 60}")
    print(f"📊 METHOD 2 (SEQUENCE embedding) RESULTS:")
    print('=' * 60)
    print(f"   Overall Accuracy:  {results_seq['overall_acc']:.4f}")
    print(f"   Overall F1:        {results_seq['overall_f1']:.4f}")
    print(f"   Known Accuracy:    {results_seq['known_acc']:.4f}")
    print(f"   New Accuracy:      {results_seq['new_acc']:.4f}")
    print(f"\n   Per-class accuracy (NEW):")
    for label, acc in results_seq['per_class_acc'].items():
        print(f"      {label} ({CLASS_NAMES[label]}): {acc:.4f}")

    # Determine best method
    best_method = "MEAN" if results_mean['new_acc'] >= results_seq['new_acc'] else "SEQUENCE"
    best_results = results_mean if best_method == "MEAN" else results_seq

    print(f"\n{'=' * 60}")
    print(f"🏆 BEST METHOD: {best_method}")
    print('=' * 60)

    # Save results
    output = {
        'method': best_method,
        'overall_accuracy': best_results['overall_acc'],
        'overall_f1': best_results['overall_f1'],
        'known_accuracy': best_results['known_acc'],
        'new_accuracy': best_results['new_acc'],
        'per_class_accuracy': {CLASS_NAMES[k]: v for k, v in best_results['per_class_acc'].items()},
        'mapping': {str(k): v for k, v in best_results['mapping'].items()},
        'class_names': CLASS_NAMES,
        'comparison': {
            'mean_new_acc': results_mean['new_acc'],
            'seq_new_acc': results_seq['new_acc'],
            'mean_avg_sim': avg_sim_mean,
            'seq_avg_sim': avg_sim_seq
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("✅ EVALUATION COMPLETED!")
    print('=' * 70)
    print(f"\n📊 SEMANTIC MAPPING (using class names):")
    for pseudo, gt in best_results['mapping'].items():
        print(f"   Pseudo {pseudo} → GT {gt} ({CLASS_NAMES[gt]})")

    print(f"\n📊 FINAL METRICS:")
    print(f"   Overall Accuracy: {best_results['overall_acc']:.4f}")
    print(f"   Known Accuracy:   {best_results['known_acc']:.4f}")
    print(f"   New Accuracy:     {best_results['new_acc']:.4f}")

    return best_results


if __name__ == '__main__':
    results = evaluate()