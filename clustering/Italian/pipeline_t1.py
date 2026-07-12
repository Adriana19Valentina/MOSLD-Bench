# pipeline_t1.py - Clustering pipeline for Test_1 with OOD Detection
# First detects unknown samples, then clusters them

import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import warnings
import sys
import os
import string
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from ood_detection import detect_unknown_samples, filter_unknown_samples

warnings.filterwarnings('ignore')

print("=" * 70)
print(f"{LANGUAGE} TEST_1 PIPELINE - OOD DETECTION + CLUSTERING")
print("=" * 70)

ensure_output_dirs()

# =========================================================================
# CONFIGURATION
# =========================================================================

BASELINE_MODEL_DIR = os.path.join(OUTPUT_DIR, 'model_baseline')
OOD_THRESHOLD = None  # Will be computed automatically, or set manually (e.g., 0.5)

# =========================================================================
# STEP 1: CHECK BASELINE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: CHECKING BASELINE MODEL")
print('=' * 70)

if not os.path.exists(BASELINE_MODEL_DIR):
    print(f"❌ Baseline model not found at: {BASELINE_MODEL_DIR}")
    print(f"   Please run train_baseline.py first!")
    sys.exit(1)

print(f"✅ Baseline model found: {BASELINE_MODEL_DIR}")

# =========================================================================
# STEP 2: OOD DETECTION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: OOD DETECTION - DETECTING UNKNOWN SAMPLES")
print('=' * 70)

ood_results = detect_unknown_samples(
    model_path=BASELINE_MODEL_DIR,
    test_csv=TEST_1_CSV,
    known_labels=BASELINE_LABELS,
    train_csv=TRAIN_CSV,  # Needed for Mahalanobis distance
    threshold=OOD_THRESHOLD,
    threshold_method=OOD_THRESHOLD_METHOD,
    batch_size=BATCH_SIZE,
    use_entropy_filter=OOD_USE_ENTROPY_FILTER
)

# Get detected unknown samples
unknown_mask = ood_results['unknown_mask']
unknown_indices = ood_results['unknown_indices']
msp_scores = ood_results['msp_scores']
threshold = ood_results['threshold']

print(f"\n📊 OOD Detection Summary:")
print(f"   Total samples: {len(unknown_mask)}")
print(f"   Detected UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")
print(f"   Detected KNOWN: {np.sum(~unknown_mask)} ({100 * np.mean(~unknown_mask):.1f}%)")
print(f"   Threshold used: {threshold:.4f}")

# =========================================================================
# STEP 3: LOAD DATA FOR CLUSTERING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: LOADING DATA FOR CLUSTERING")
print('=' * 70)

# Load full test data
test_df = pd.read_csv(TEST_1_CSV)

# Get only detected unknown samples for clustering
unknown_df = test_df[unknown_mask].copy()
unknown_df['msp_score'] = msp_scores[unknown_mask]
unknown_df['original_index'] = unknown_indices

print(f"✅ Unknown samples for clustering: {len(unknown_df)}")

# Also load known data for replay
train_df = pd.read_csv(TRAIN_CSV)
known_df = train_df[train_df['label'].isin(BASELINE_LABELS)][['content', 'label']].copy()
print(f"✅ Known samples (baseline): {len(known_df)}")

# Extract texts and labels for clustering
texts_unknown = unknown_df['content'].tolist()
labels_GT_unknown = unknown_df['label'].tolist()  # Ground truth (for evaluation only)

print(f"\n📊 Ground truth distribution in detected unknown:")
gt_counts = Counter(labels_GT_unknown)
for label, count in sorted(gt_counts.items()):
    print(f"   Label {label}: {count} samples ({100 * count / len(labels_GT_unknown):.1f}%)")

# =========================================================================
# STEP 4: GENERATE EMBEDDINGS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: GENERATING EMBEDDINGS FOR CLUSTERING")
print('=' * 70)

print(f"🤖 Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")


def get_embeddings(texts, batch_size=32):
    """Generate BERT embeddings for texts"""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Convert to strings and handle None/NaN
        batch_texts = [str(t) if t is not None and str(t) != 'nan' else '' for t in batch_texts]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.append(batch_embeddings)

        if (i // batch_size) % 10 == 0:
            print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.vstack(embeddings)


print(f"\n📊 Generating embeddings for {len(texts_unknown)} unknown samples...")
embeddings_unknown = get_embeddings(texts_unknown)
print(f"✅ Embeddings shape: {embeddings_unknown.shape}")

# =========================================================================
# STEP 5: DETERMINE OPTIMAL K
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: DETERMINING OPTIMAL K")
print('=' * 70)

k_range = range(K_MIN, K_MAX + 1)

metrics = {
    'silhouette': [],
    'davies_bouldin': [],
    'calinski_harabasz': [],
    'inertia': []
}

print(f"Testing K from {K_MIN} to {K_MAX}...")

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_unknown)

    sil = silhouette_score(embeddings_unknown, cluster_labels)
    dbi = davies_bouldin_score(embeddings_unknown, cluster_labels)
    chi = calinski_harabasz_score(embeddings_unknown, cluster_labels)

    metrics['silhouette'].append(sil)
    metrics['davies_bouldin'].append(dbi)
    metrics['calinski_harabasz'].append(chi)
    metrics['inertia'].append(kmeans.inertia_)

    print(f"  K={k}: Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}, Calinski={chi:.1f}")


# =========================================================================
# IMPROVED K SELECTION: Gap Statistic + Combined scoring
# =========================================================================

def normalize(arr):
    """Normalize array to [0, 1]"""
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def find_elbow(values, k_range):
    """Find elbow point using maximum curvature"""
    values = np.array(values)
    k_list = list(k_range)

    if len(values) < 3:
        return k_list[0]

    n = len(values)
    line_start = np.array([0, values[0]])
    line_end = np.array([n - 1, values[-1]])

    max_dist = 0
    elbow_idx = 0

    for i in range(1, n - 1):
        point = np.array([i, values[i]])
        dist = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return k_list[elbow_idx]


def compute_gap_statistic(data, k_range, n_refs=10):
    """
    Compute Gap Statistic for K selection.
    Gap(k) = E[log(W_k_ref)] - log(W_k)
    Higher gap = better K
    """
    gaps = []

    for k in k_range:
        # Compute W_k for real data
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data)
        w_k = kmeans.inertia_

        # Compute W_k for reference (uniform) data
        ref_inertias = []
        for _ in range(n_refs):
            # Generate uniform reference data
            ref_data = np.random.uniform(
                low=data.min(axis=0),
                high=data.max(axis=0),
                size=data.shape
            )
            ref_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=None)
            ref_kmeans.fit(ref_data)
            ref_inertias.append(ref_kmeans.inertia_)

        # Gap = E[log(W_ref)] - log(W)
        gap = np.mean(np.log(ref_inertias)) - np.log(w_k + 1e-10)
        gaps.append(gap)

    return np.array(gaps)


# Compute Gap Statistic
print(f"\n📊 Computing Gap Statistic...")
gap_scores = compute_gap_statistic(embeddings_unknown, k_range, n_refs=5)

# Normalize metrics
sil_norm = normalize(metrics['silhouette'])  # Higher is better
dbi_norm = 1 - normalize(metrics['davies_bouldin'])  # Lower is better, so invert
chi_norm = normalize(metrics['calinski_harabasz'])  # Higher is better
gap_norm = normalize(gap_scores)  # Higher is better

# Elbow on inertia
k_elbow = find_elbow(metrics['inertia'], k_range)
k_list = list(k_range)

# Combined score with weights - NO PENALTY for small K
# Gap statistic is the key: it detects real structure vs random
combined_scores = (
        0.10 * sil_norm +  # Cluster quality (low weight - biased to small K)
        0.15 * dbi_norm +  # Cluster separation
        0.35 * chi_norm +  # Variance ratio - prefers more clusters when real
        0.40 * gap_norm  # Gap statistic - KEY metric, compares to uniform
)

# Small bonus for K at elbow point
for i, k in enumerate(k_list):
    if k == k_elbow:
        combined_scores[i] += 0.05

print(f"\n📊 Combined Scoring (Gap + CHI dominant):")
print(f"   Elbow point: K={k_elbow}")
print(f"   {'K':<4} {'Score':<8} {'Sil':<6} {'DBI':<6} {'CHI':<6} {'Gap':<6}")
print(f"   {'-' * 45}")
for i, k in enumerate(k_list):
    star = " ⭐" if combined_scores[i] == max(combined_scores) else ""
    elbow_str = " (elbow)" if k == k_elbow else ""
    print(
        f"   {k:<4} {combined_scores[i]:<8.3f} {sil_norm[i]:<6.2f} {dbi_norm[i]:<6.2f} {chi_norm[i]:<6.2f} {gap_norm[i]:<6.2f}{star}{elbow_str}")

# Select K with highest combined score
best_k_idx = np.argmax(combined_scores)
K_FINAL = k_list[best_k_idx]

print(f"\n🎯 Selected K={K_FINAL} (combined score: {combined_scores[best_k_idx]:.3f})")

# Ground truth K for comparison
gt_k = len(set(labels_GT_unknown))
print(f"📊 Ground truth K: {gt_k}")

# =========================================================================
# STEP 6: FINAL CLUSTERING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: FINAL CLUSTERING")
print('=' * 70)

kmeans_final = KMeans(n_clusters=K_FINAL, init='k-means++', n_init=10, random_state=42)
cluster_labels_final = kmeans_final.fit_predict(embeddings_unknown)
centroids = kmeans_final.cluster_centers_

print(f"✅ Clustering completed with K={K_FINAL}")

# =========================================================================
# STEP 7: CLUSTER ANALYSIS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 7: CLUSTER ANALYSIS")
print('=' * 70)

cluster_info = {}

for cluster_id in range(K_FINAL):
    cluster_mask = cluster_labels_final == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_texts = [texts_unknown[i] for i in cluster_indices]
    cluster_labels_gt = [labels_GT_unknown[i] for i in cluster_indices]

    label_counts = Counter(cluster_labels_gt)
    dominant_label = max(label_counts, key=label_counts.get)
    purity = label_counts[dominant_label] / len(cluster_labels_gt)

    cluster_info[cluster_id] = {
        'size': len(cluster_indices),
        'indices': cluster_indices,
        'texts': cluster_texts,
        'labels_gt': cluster_labels_gt,
        'distribution': dict(label_counts),
        'dominant_label': dominant_label,
        'purity': purity
    }

    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {len(cluster_indices)}")
    print(f"  Distribution: {dict(label_counts)}")
    print(f"  Dominant label: {dominant_label}")
    print(f"  Purity: {purity:.2%}")

overall_purity = np.mean([info['purity'] for info in cluster_info.values()])
print(f"\n📊 Overall purity: {overall_purity:.2%}")

# Compute ARI
ari = adjusted_rand_score(labels_GT_unknown, cluster_labels_final)
print(f"📊 Adjusted Rand Index: {ari:.4f}")

# =========================================================================
# STEP 8: EXTRACT KEYWORDS FOR HUNGARIAN MAPPING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 8: EXTRACTING KEYWORDS FROM CLUSTERS")
print('=' * 70)


def clean_text(text):
    """Remove punctuation and normalize text"""
    if not text:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


def extract_top_keywords(texts, n_top=10, min_word_length=3):
    """Extract top keywords using TF-IDF"""
    if not texts:
        return []

    cleaned_texts = [clean_text(t) for t in texts if t]
    cleaned_texts = [t for t in cleaned_texts if t]

    if not cleaned_texts:
        return []

    try:
        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words=list(STOP_WORDS),
            token_pattern=rf"(?u)\b[^\W\d_]{{{min_word_length},}}\b",
            lowercase=True
        )
        tfidf_matrix = tfidf.fit_transform(cleaned_texts)
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_scores.argsort()[::-1][:n_top]
        return [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
    except:
        return []


# Extract keywords for each cluster
cluster_keywords = {}
cluster_all_keywords = {}

for cluster_id in range(K_FINAL):
    cluster_texts = cluster_info[cluster_id]['texts']
    all_keywords = extract_top_keywords(cluster_texts, n_top=30)
    cluster_all_keywords[cluster_id] = all_keywords
    cluster_keywords[cluster_id] = all_keywords[:10]
    print(f"  Cluster {cluster_id} top keywords: {cluster_keywords[cluster_id][:5]}...")

# Find unique keywords per cluster
print(f"\n📊 Finding UNIQUE keywords per cluster...")
all_kw_to_clusters = {}
for cluster_id in range(K_FINAL):
    for kw in cluster_all_keywords[cluster_id]:
        if kw not in all_kw_to_clusters:
            all_kw_to_clusters[kw] = []
        all_kw_to_clusters[kw].append(cluster_id)

cluster_unique_keywords = {}
for cluster_id in range(K_FINAL):
    unique_kws = [kw for kw in cluster_all_keywords[cluster_id]
                  if len(all_kw_to_clusters[kw]) == 1]
    cluster_unique_keywords[cluster_id] = unique_kws[:10]
    print(f"  Cluster {cluster_id} UNIQUE: {cluster_unique_keywords[cluster_id]}")

# Use unique keywords if available
for cluster_id in range(K_FINAL):
    if cluster_unique_keywords[cluster_id]:
        cluster_keywords[cluster_id] = cluster_unique_keywords[cluster_id]

# =========================================================================
# STEP 9: HUNGARIAN ALGORITHM MAPPING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 9: HUNGARIAN ALGORITHM MAPPING")
print('=' * 70)


def get_text_embedding(text, tokenizer, model, device):
    """Get embedding for a text"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]


# Purity-based mapping for comparison
cluster_to_label_purity = {cid: info['dominant_label'] for cid, info in cluster_info.items()}

# Get discovered class names
discovered_labels = TEST_1_NEW_LABELS
discovered_class_names = []
if 'label_name' in unknown_df.columns:
    for label_id in discovered_labels:
        matches = unknown_df[unknown_df['label'] == label_id]['label_name']
        if len(matches) > 0:
            discovered_class_names.append(matches.iloc[0])
        else:
            discovered_class_names.append(str(label_id))
else:
    discovered_class_names = [str(l) for l in discovered_labels]

print(f"  Discovered labels: {discovered_labels}")
print(f"  Class names: {discovered_class_names}")

# Compute keyword embeddings
keyword_embeddings = []
for cluster_id in range(K_FINAL):
    keywords = cluster_keywords[cluster_id]
    if keywords:
        keywords_text = ' '.join(keywords)
        emb = get_text_embedding(keywords_text, tokenizer, model, device)
        keyword_embeddings.append(emb)
    else:
        keyword_embeddings.append(centroids[cluster_id])
keyword_embeddings = np.array(keyword_embeddings)

# Compute class embeddings
class_embeddings = []
for class_name in discovered_class_names:
    emb = get_text_embedding(class_name, tokenizer, model, device)
    class_embeddings.append(emb)
class_embeddings = np.array(class_embeddings)

# Hungarian algorithm
if discovered_class_names and K_FINAL <= len(discovered_class_names):
    keyword_embeddings_norm = keyword_embeddings / np.linalg.norm(keyword_embeddings, axis=1, keepdims=True)
    class_embeddings_norm = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
    similarity_matrix = keyword_embeddings_norm @ class_embeddings_norm.T

    print(f"\n📊 Similarity Matrix:")
    print(f"    {''.join([f'{name:>12}' for name in discovered_class_names])}")
    for cluster_id in range(K_FINAL):
        row = ''.join([f'{similarity_matrix[cluster_id, i]:12.4f}' for i in range(len(discovered_class_names))])
        print(f"C{cluster_id}: {row}")

    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    cluster_to_label_auto = {}
    total_similarity = 0.0
    print(f"\n🗺️  Automatic Mapping (Hungarian):")
    for cluster_id, class_idx in zip(row_ind, col_ind):
        sim = similarity_matrix[cluster_id, class_idx]
        total_similarity += sim
        label_val = discovered_labels[class_idx]
        class_name = discovered_class_names[class_idx]
        cluster_to_label_auto[cluster_id] = label_val
        print(f"  Cluster {cluster_id} → Label {label_val} ({class_name}), sim={sim:.4f}")

    avg_similarity = total_similarity / len(row_ind)
    print(f"\n📊 Average similarity: {avg_similarity:.4f}")

    cluster_to_label_eval = cluster_to_label_auto
else:
    print(f"⚠️  Using purity-based mapping (K > num_classes)")
    cluster_to_label_eval = cluster_to_label_purity

# =========================================================================
# STEP 10: ASSIGN PSEUDO-LABELS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 10: ASSIGNING PSEUDO-LABELS")
print('=' * 70)

cluster_to_pseudo = {
    cluster_id: PSEUDO_LABEL_START_T1 + cluster_id
    for cluster_id in range(K_FINAL)
}

print(f"\n🏷️  Pseudo-label assignment:")
for cluster_id, pseudo_label in cluster_to_pseudo.items():
    gt_mapping = cluster_to_label_eval.get(cluster_id, '?')
    print(f"  Cluster {cluster_id} → Pseudo {pseudo_label} (GT: {gt_mapping})")

# =========================================================================
# STEP 11: SAMPLE SELECTION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 11: SAMPLE SELECTION (ADAPTIVE)")
print('=' * 70)

selected_indices = []
selected_pseudo_labels = []

for cluster_id in range(K_FINAL):
    cluster_indices = cluster_info[cluster_id]['indices']
    cluster_embeddings = embeddings_unknown[cluster_indices]
    centroid = centroids[cluster_id]
    purity = cluster_info[cluster_id]['purity']

    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    adaptive_ratio = SAMPLE_SELECTION_RATIO * (0.8 + 0.4 * purity)
    adaptive_ratio = min(adaptive_ratio, 0.8)

    n_select = max(1, int(len(cluster_indices) * adaptive_ratio))
    nearest_indices = distances.argsort()[:n_select]
    selected_local = cluster_indices[nearest_indices]

    pseudo_label = cluster_to_pseudo[cluster_id]

    selected_indices.extend(selected_local)
    selected_pseudo_labels.extend([pseudo_label] * len(selected_local))

    print(f"  Cluster {cluster_id}: {len(cluster_indices)} → {n_select} selected (purity={purity:.2%})")

print(f"\n✅ Total selected: {len(selected_indices)} samples")

# =========================================================================
# STEP 12: CREATE COMBINED DATASET
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 12: CREATING COMBINED DATASET")
print('=' * 70)

# Selected new samples with pseudo-labels
selected_texts = [texts_unknown[i] for i in selected_indices]
new_data = pd.DataFrame({
    'content': selected_texts,
    'label': selected_pseudo_labels
})

# Combine with known data
all_data = pd.concat([known_df, new_data], ignore_index=True)
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Combined dataset: {len(all_data)} samples")
print(f"   Known (baseline): {len(known_df)}")
print(f"   New (discovered): {len(new_data)}")
print(f"   Labels: {sorted(all_data['label'].unique())}")

# =========================================================================
# STEP 13: SAVE RESULTS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 13: SAVING RESULTS")
print('=' * 70)

# Save processed data
all_data.to_csv(T1_PROCESSED_CSV, index=False)
print(f"✅ Saved to: {T1_PROCESSED_CSV}")

# Save clustering results
results = {
    'K_final': K_FINAL,
    'K_ground_truth': gt_k,
    'overall_purity': overall_purity,
    'ari': ari,
    'silhouette': metrics['silhouette'][best_k_idx],
    'cluster_info': {str(k): {
        'size': v['size'],
        'distribution': v['distribution'],
        'dominant_label': v['dominant_label'],
        'purity': v['purity']
    } for k, v in cluster_info.items()},
    'cluster_to_pseudo': cluster_to_pseudo,
    'cluster_to_label_eval': cluster_to_label_eval,
    'ood_detection': {
        'threshold': threshold,
        'total_samples': len(unknown_mask),
        'detected_unknown': int(np.sum(unknown_mask)),
        'detected_known': int(np.sum(~unknown_mask))
    }
}

with open(T1_RESULTS_PKL, 'wb') as f:
    pickle.dump(results, f)
print(f"✅ Saved to: {T1_RESULTS_PKL}")

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'=' * 70}")
print("✅ PIPELINE T1 COMPLETED!")
print('=' * 70)

print(f"\n📊 OOD DETECTION:")
print(f"   Threshold: {threshold:.4f}")
print(f"   Detected unknown: {np.sum(unknown_mask)}/{len(unknown_mask)}")

print(f"\n📊 CLUSTERING:")
print(f"   Predicted K: {K_FINAL}")
print(f"   Ground truth K: {gt_k}")
print(f"   Overall purity: {overall_purity:.2%}")
print(f"   ARI: {ari:.4f}")

print(f"\n🏷️  PSEUDO-LABEL MAPPING:")
for cluster_id, pseudo in cluster_to_pseudo.items():
    gt = cluster_to_label_eval.get(cluster_id, '?')
    print(f"   Cluster {cluster_id} → Pseudo {pseudo} (likely GT {gt})")

print(f"\n📁 OUTPUT FILES:")
print(f"   - {T1_PROCESSED_CSV}")
print(f"   - {T1_RESULTS_PKL}")
print(f"\n🚀 NEXT: Run train_t1.py")
print('=' * 70)