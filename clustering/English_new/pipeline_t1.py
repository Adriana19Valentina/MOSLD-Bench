import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import string
import re
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from ood_detection import detect_unknown_samples


print('=' * 70)
print(f'{LANGUAGE.upper()} TEST_1 PIPELINE - OOD DETECTION + CLUSTERING')
print('=' * 70)

# =========================================================================
# STEP 1: CHECK BASELINE MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: CHECKING BASELINE MODEL")
print('=' * 70)

if not os.path.exists(BASELINE_MODEL_DIR):
    raise FileNotFoundError(f"Baseline model not found at {BASELINE_MODEL_DIR}")
print(f" Baseline model found: {BASELINE_MODEL_DIR}")

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
    val_csv=VAL_CSV,
    target_tpr=OOD_TARGET_TPR,
)

unknown_mask = ood_results['unknown_mask']
threshold = ood_results['threshold']

print(f"\n OOD Detection Summary:")
print(f"   Detected UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")
print(f"   Detected KNOWN: {np.sum(~unknown_mask)} ({100 * np.mean(~unknown_mask):.1f}%)")

# =========================================================================
# STEP 3: LOAD DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: LOADING DATA")
print('=' * 70)

test_df = pd.read_csv(TEST_1_CSV)
unknown_df = test_df[unknown_mask].copy()
known_df = pd.read_csv(TRAIN_CSV)

texts_unknown = unknown_df['content'].tolist()
labels_GT_unknown = unknown_df['label'].tolist()

print(f" Unknown samples for clustering: {len(texts_unknown)}")
print(f" Known data for replay: {len(known_df)}")

# =========================================================================
# STEP 4: GENERATE EMBEDDINGS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 4: GENERATING EMBEDDINGS")
print('=' * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


def get_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(t) if t is not None and str(t) != 'nan' else '' for t in batch_texts]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


print(f" Generating embeddings for {len(texts_unknown)} samples...")
embeddings_unknown = get_embeddings(texts_unknown)
print(f" Embeddings shape: {embeddings_unknown.shape}")

# =========================================================================
# STEP 5: DETERMINE OPTIMAL K
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: DETERMINING OPTIMAL K")
print('=' * 70)



n_samples = len(embeddings_unknown)
K_MIN_LOCAL = 2  # At least 2 clusters
K_MAX_LOCAL = 8

print(f" Unknown samples to cluster: {n_samples}")
print(f" Testing K from {K_MIN_LOCAL} to {K_MAX_LOCAL}...")

# Check if K is forced in config (for experiments where we want to test specific K)
FORCE_K = globals().get('FORCE_K_T1', None)
if FORCE_K is not None:
    print(f"  K forced to {FORCE_K} by config")
    K_FINAL = FORCE_K
    k_range = range(FORCE_K, FORCE_K + 1)
else:
    k_range = range(K_MIN_LOCAL, K_MAX_LOCAL + 1)

metrics = {'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': [], 'inertia': []}

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


# K Selection with multiple metrics
def normalize(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def find_elbow(values, k_range):
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


sil_norm = normalize(metrics['silhouette'])
dbi_norm = 1 - normalize(metrics['davies_bouldin'])
chi_norm = normalize(metrics['calinski_harabasz'])

k_elbow = find_elbow(metrics['inertia'], k_range)
k_list = list(k_range)

combined_scores = (
        0.20 * sil_norm +
        0.20 * dbi_norm +
        0.60 * chi_norm
)

for i, k in enumerate(k_list):
    if k == k_elbow:
        combined_scores[i] += 0.05

print(f"\n Combined Scoring:")
print(f"   Elbow point: K={k_elbow}")
for i, k in enumerate(k_list):
    star = " ⭐" if combined_scores[i] == max(combined_scores) else ""
    print(f"   K={k}: {combined_scores[i]:.3f}{star}")

best_k_idx = np.argmax(combined_scores)
K_FINAL = k_list[best_k_idx]

print(f"\n Selected K={K_FINAL}")

# Ground truth K for reference (NOT used in selection)
gt_k = len(set(labels_GT_unknown))
print(f"Ground truth K (for reference): {gt_k}")

# =========================================================================
# STEP 6: FINAL CLUSTERING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: FINAL CLUSTERING")
print('=' * 70)

kmeans_final = KMeans(n_clusters=K_FINAL, init='k-means++', n_init=20, random_state=42)
cluster_labels_final = kmeans_final.fit_predict(embeddings_unknown)
centroids = kmeans_final.cluster_centers_

print(f" Final clustering with K={K_FINAL}")

# =========================================================================
# STEP 7: CLUSTER ANALYSIS (for reference only)
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 7: CLUSTER ANALYSIS")
print('=' * 70)

cluster_info = {}
for cluster_id in range(K_FINAL):
    cluster_indices = np.where(cluster_labels_final == cluster_id)[0]
    cluster_texts = [texts_unknown[i] for i in cluster_indices]
    cluster_labels_gt = [labels_GT_unknown[i] for i in cluster_indices]

    label_counts = Counter(cluster_labels_gt)
    dominant_label = label_counts.most_common(1)[0][0] if label_counts else -1
    purity = label_counts.most_common(1)[0][1] / len(cluster_indices) if cluster_indices.size > 0 else 0

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
    print(f"  GT Distribution: {dict(label_counts)}")
    print(f"  Purity: {purity:.2%}")

overall_purity = np.mean([info['purity'] for info in cluster_info.values()])
ari = adjusted_rand_score(labels_GT_unknown, cluster_labels_final)

print(f"\n Overall purity: {overall_purity:.2%}")
print(f" Adjusted Rand Index: {ari:.4f}")

# =========================================================================
# STEP 8: EXTRACT KEYWORDS WITH TF-IDF (for interpretability)
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 8: EXTRACTING KEYWORDS FROM CLUSTERS (TF-IDF)")
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

import jieba

def extract_top_keywords(texts, n_top=10, min_word_length=3):
    """Extract top keywords using TF-IDF"""
    if not texts:
        return []

    cleaned_texts = [clean_text(t) for t in texts if t]
    cleaned_texts = [t for t in cleaned_texts if t]

    if not cleaned_texts:
        return []
    try:
        # Use STOP_WORDS from config if available, otherwise use default
        # stop_words = list(STOP_WORDS) if 'STOP_WORDS' in dir() else list(ENGLISH_STOP_WORDS)
        tfidf = TfidfVectorizer(
            stop_words=list(STOP_WORDS_EN),
            lowercase=True,
            max_features=100,
        )

        sw = set(tfidf.get_stop_words() or [])
        tfidf_matrix = tfidf.fit_transform(cleaned_texts)
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_scores.argsort()[::-1][:n_top]
        return [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
    except Exception as e:
        print(f"  TF-IDF failed: {e}")
        return []


# Extract keywords for each cluster
cluster_keywords = {}
cluster_all_keywords = {}

for cluster_id in range(K_FINAL):
    cluster_texts = cluster_info[cluster_id]['texts']
    # Extract more keywords initially to find unique ones
    all_keywords = extract_top_keywords(cluster_texts, n_top=50)
    cluster_all_keywords[cluster_id] = all_keywords
    cluster_keywords[cluster_id] = all_keywords[:10]
    print(f"  Cluster {cluster_id} top keywords: {cluster_keywords[cluster_id][:5]}...")

# Find UNIQUE keywords per cluster (keywords that appear only in one cluster)
print(f"\n Finding UNIQUE keywords per cluster...")
all_kw_to_clusters = {}
for cluster_id in range(K_FINAL):
    for kw in cluster_all_keywords[cluster_id]:
        if kw not in all_kw_to_clusters:
            all_kw_to_clusters[kw] = []
        all_kw_to_clusters[kw].append(cluster_id)

cluster_unique_keywords = {}
MIN_UNIQUE_KEYWORDS = 5  # Minimum number of unique keywords we want

for cluster_id in range(K_FINAL):
    # Get keywords that appear ONLY in this cluster
    unique_kws = [kw for kw in cluster_all_keywords[cluster_id]
                  if len(all_kw_to_clusters[kw]) == 1]

    # If not enough unique keywords, add "mostly unique" ones (appear in max 2 clusters)
    if len(unique_kws) < MIN_UNIQUE_KEYWORDS:
        mostly_unique = [kw for kw in cluster_all_keywords[cluster_id]
                         if len(all_kw_to_clusters[kw]) <= 2 and kw not in unique_kws]
        unique_kws.extend(mostly_unique)

    # Still not enough? Add top keywords as fallback
    if len(unique_kws) < MIN_UNIQUE_KEYWORDS:
        remaining = [kw for kw in cluster_all_keywords[cluster_id] if kw not in unique_kws]
        unique_kws.extend(remaining[:MIN_UNIQUE_KEYWORDS - len(unique_kws)])

    cluster_unique_keywords[cluster_id] = unique_kws[:15]  # Keep top 15
    print(
        f"  Cluster {cluster_id} UNIQUE ({len([k for k in unique_kws[:10] if len(all_kw_to_clusters.get(k, [])) == 1])}/10 strictly unique): {cluster_unique_keywords[cluster_id][:7]}...")

# Store both top and unique keywords
for cluster_id in range(K_FINAL):
    cluster_info[cluster_id]['keywords'] = cluster_keywords[cluster_id]
    cluster_info[cluster_id]['unique_keywords'] = cluster_unique_keywords[cluster_id]

# =========================================================================
# STEP 9: ASSIGN PSEUDO-LABELS (NO GT MAPPING!)
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 9: ASSIGNING PSEUDO-LABELS")
print('=' * 70)

# Simply assign sequential pseudo-labels to clusters
# NO mapping to ground truth - that happens in evaluation!
cluster_to_pseudo = {
    cluster_id: PSEUDO_LABEL_START_T1 + cluster_id
    for cluster_id in range(K_FINAL)
}

print(f"\n  Pseudo-label assignment (no GT mapping):")
for cluster_id, pseudo_label in cluster_to_pseudo.items():
    size = cluster_info[cluster_id]['size']
    print(f"  Cluster {cluster_id} → Pseudo {pseudo_label} ({size} samples)")

# =========================================================================
# STEP 10: SAMPLE SELECTION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 10: SAMPLE SELECTION")
print('=' * 70)

selected_indices = []
selected_pseudo_labels = []

for cluster_id in range(K_FINAL):
    cluster_indices = cluster_info[cluster_id]['indices']
    cluster_embeddings = embeddings_unknown[cluster_indices]
    centroid = centroids[cluster_id]
    purity = cluster_info[cluster_id]['purity']

    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    # Adaptive selection ratio based on cluster quality
    adaptive_ratio = SAMPLE_SELECTION_RATIO * (0.8 + 0.4 * purity)
    adaptive_ratio = min(adaptive_ratio, 0.8)

    n_select = max(1, int(len(cluster_indices) * adaptive_ratio))
    nearest_indices = distances.argsort()[:n_select]
    selected_local = cluster_indices[nearest_indices]

    pseudo_label = cluster_to_pseudo[cluster_id]

    selected_indices.extend(selected_local)
    selected_pseudo_labels.extend([pseudo_label] * len(selected_local))

    print(f"  Cluster {cluster_id}: {len(cluster_indices)} → {n_select} selected")

print(f"\nTotal selected: {len(selected_indices)} samples")

# =========================================================================
# STEP 11: CREATE COMBINED DATASET
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 11: CREATING COMBINED DATASET")
print('=' * 70)

selected_texts = [texts_unknown[i] for i in selected_indices]
new_data = pd.DataFrame({
    'content': selected_texts,
    'label': selected_pseudo_labels
})

all_data = pd.concat([known_df[['content', 'label']], new_data], ignore_index=True)
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Combined dataset: {len(all_data)} samples")
print(f"   Known (baseline): {len(known_df)}")
print(f"   New (discovered): {len(new_data)}")
print(f"   Labels: {sorted(all_data['label'].unique())}")

# =========================================================================
# STEP 12: SAVE RESULTS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 12: SAVING RESULTS")
print('=' * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_data.to_csv(T1_PROCESSED_CSV, index=False)
print(f"Saved to: {T1_PROCESSED_CSV}")

# Save clustering results (NO GT mapping!)
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
        'purity': v['purity'],
        'keywords': v.get('keywords', []),
        'unique_keywords': v.get('unique_keywords', [])
    } for k, v in cluster_info.items()},
    'cluster_to_pseudo': cluster_to_pseudo,
    'cluster_keywords': {str(k): v for k, v in cluster_keywords.items()},
    'cluster_unique_keywords': {str(k): v for k, v in cluster_unique_keywords.items()},
    # Store GT info for evaluation reference
    'cluster_gt_distribution': {str(k): v['distribution'] for k, v in cluster_info.items()},
    'ood_detection': {
        'threshold': float(threshold),
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

print(f"\n OOD DETECTION:")
print(f"   Threshold: {threshold:.4f}")
print(f"   Detected unknown: {np.sum(unknown_mask)}/{len(unknown_mask)}")

print(f"\n CLUSTERING:")
print(f"   K selected: {K_FINAL}")
print(f"   K ground truth: {gt_k}")
print(f"   Overall purity: {overall_purity:.2%}")
print(f"   ARI: {ari:.4f}")

print(f"\n PSEUDO-LABELS ASSIGNED:")
for cluster_id, pseudo in cluster_to_pseudo.items():
    kws = cluster_keywords.get(cluster_id, [])[:3]
    kw_str = ", ".join(kws) if kws else "no keywords"
    print(f"   Cluster {cluster_id} → Pseudo {pseudo} (keywords: {kw_str})")

print(f"\n OUTPUT FILES:")
print(f"   - {T1_PROCESSED_CSV}")
print(f"   - {T1_RESULTS_PKL}")
print(f"\n NEXT: Run train_t1.py")
print('=' * 70)