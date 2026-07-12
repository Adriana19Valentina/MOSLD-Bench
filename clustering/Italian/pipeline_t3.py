# pipeline_t3.py - Clustering pipeline for Test_3 with OOD Detection
# Uses model_t2 for OOD detection, then clusters unknown samples

import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import warnings
import sys
import os
import string
import re
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from ood_detection import detect_unknown_samples

warnings.filterwarnings('ignore')

print("=" * 70)
print(f"{LANGUAGE} TEST_3 PIPELINE - OOD DETECTION + CLUSTERING")
print("=" * 70)

ensure_output_dirs()

# Use model_t2 for OOD detection
OOD_MODEL_DIR = MODEL_T2_DIR

# =========================================================================
# STEP 1: CHECK REQUIRED MODEL
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 1: CHECKING REQUIRED MODEL")
print('=' * 70)

if not os.path.exists(OOD_MODEL_DIR):
    print(f"❌ Model not found at: {OOD_MODEL_DIR}")
    sys.exit(1)

print(f"✅ OOD model found: {OOD_MODEL_DIR}")

# =========================================================================
# STEP 2: OOD DETECTION
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 2: OOD DETECTION")
print('=' * 70)

ood_results = detect_unknown_samples(
    model_path=OOD_MODEL_DIR,
    test_csv=TEST_3_CSV,
    known_labels=KNOWN_LABELS_T3,
    train_csv=TRAIN_CSV,  # Needed for Mahalanobis distance
    threshold=OOD_THRESHOLD,
    threshold_method=OOD_THRESHOLD_METHOD,
    batch_size=BATCH_SIZE,
    use_entropy_filter=OOD_USE_ENTROPY_FILTER
)

unknown_mask = ood_results['unknown_mask']
threshold = ood_results['threshold']

print(f"\n📊 Detected UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")

# =========================================================================
# STEP 3: LOAD DATA
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 3: LOADING DATA")
print('=' * 70)

test_df = pd.read_csv(TEST_3_CSV)
unknown_df = test_df[unknown_mask].copy()

texts_unknown = unknown_df['content'].tolist()
labels_GT_unknown = unknown_df['label'].tolist()

print(f"✅ Unknown samples: {len(unknown_df)}")

# Replay data
train_df = pd.read_csv(TRAIN_CSV)
known_df = train_df[train_df['label'].isin(BASELINE_LABELS)][['content', 'label']].copy()

t1_df = pd.read_csv(T1_PROCESSED_CSV)
t1_replay = t1_df[t1_df['label'] >= PSEUDO_LABEL_START_T1][['content', 'label']].copy()

t2_df = pd.read_csv(T2_PROCESSED_CSV)
t2_replay = t2_df[t2_df['label'] >= PSEUDO_LABEL_START_T2][['content', 'label']].copy()

known_df = pd.concat([known_df, t1_replay, t2_replay], ignore_index=True)
print(f"✅ Replay data: {len(known_df)}")

# =========================================================================
# STEP 4: EMBEDDINGS
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
        batch = texts[i:i + batch_size]
        # Convert to strings and handle None/NaN
        batch = [str(t) if t is not None and str(t) != 'nan' else '' for t in batch]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)


embeddings_unknown = get_embeddings(texts_unknown)
print(f"✅ Embeddings: {embeddings_unknown.shape}")

# =========================================================================
# STEP 5: CLUSTERING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 5: CLUSTERING")
print('=' * 70)

k_range = range(K_MIN, K_MAX + 1)
metrics = {'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': [], 'inertia': []}

print(f"Testing K from {K_MIN} to {K_MAX}...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings_unknown)
    metrics['silhouette'].append(silhouette_score(embeddings_unknown, labels))
    metrics['davies_bouldin'].append(davies_bouldin_score(embeddings_unknown, labels))
    metrics['calinski_harabasz'].append(calinski_harabasz_score(embeddings_unknown, labels))
    metrics['inertia'].append(kmeans.inertia_)
    print(
        f"  K={k}: Sil={metrics['silhouette'][-1]:.4f}, DBI={metrics['davies_bouldin'][-1]:.4f}, CHI={metrics['calinski_harabasz'][-1]:.1f}")


# Combined scoring for K selection
def normalize(arr):
    arr = np.array(arr)
    if arr.max() == arr.min(): return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def find_elbow(values, k_range):
    values = np.array(values)
    k_list = list(k_range)
    if len(values) < 3: return k_list[0]
    n = len(values)
    line_start, line_end = np.array([0, values[0]]), np.array([n - 1, values[-1]])
    max_dist, elbow_idx = 0, 0
    for i in range(1, n - 1):
        point = np.array([i, values[i]])
        dist = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        if dist > max_dist: max_dist, elbow_idx = dist, i
    return k_list[elbow_idx]


sil_norm = normalize(metrics['silhouette'])
dbi_norm = 1 - normalize(metrics['davies_bouldin'])
chi_norm = normalize(metrics['calinski_harabasz'])

k_elbow = find_elbow(metrics['inertia'], k_range)
k_list = list(k_range)
elbow_bonus = np.array([0.1 if k >= k_elbow else 0 for k in k_list])

combined_scores = 0.30 * sil_norm + 0.25 * dbi_norm + 0.35 * chi_norm + elbow_bonus

print(f"\n📊 Combined Scoring (Elbow={k_elbow}):")
for i, k in enumerate(k_list):
    star = "⭐" if combined_scores[i] == max(combined_scores) else ""
    print(f"   K={k}: Score={combined_scores[i]:.3f} {star}")

K_FINAL = k_list[np.argmax(combined_scores)]
print(f"\n🎯 Selected K={K_FINAL}")

kmeans_final = KMeans(n_clusters=K_FINAL, init='k-means++', n_init=10, random_state=42)
cluster_labels_final = kmeans_final.fit_predict(embeddings_unknown)
centroids = kmeans_final.cluster_centers_

# =========================================================================
# STEP 6: CLUSTER ANALYSIS
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 6: CLUSTER ANALYSIS")
print('=' * 70)

cluster_info = {}
for cid in range(K_FINAL):
    mask = cluster_labels_final == cid
    indices = np.where(mask)[0]
    texts = [texts_unknown[i] for i in indices]
    gt_labels = [labels_GT_unknown[i] for i in indices]

    counts = Counter(gt_labels)
    dominant = max(counts, key=counts.get)
    purity = counts[dominant] / len(gt_labels)

    cluster_info[cid] = {
        'size': len(indices), 'indices': indices, 'texts': texts,
        'labels_gt': gt_labels, 'distribution': dict(counts),
        'dominant_label': dominant, 'purity': purity
    }
    print(f"Cluster {cid}: size={len(indices)}, dominant={dominant}, purity={purity:.2%}")

overall_purity = np.mean([info['purity'] for info in cluster_info.values()])
ari = adjusted_rand_score(labels_GT_unknown, cluster_labels_final)
print(f"\n📊 Purity: {overall_purity:.2%}, ARI: {ari:.4f}")

# =========================================================================
# STEP 7: HUNGARIAN MAPPING
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 7: HUNGARIAN MAPPING")
print('=' * 70)


def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())


def extract_keywords(texts, n=10):
    cleaned = [clean_text(t) for t in texts if t]
    cleaned = [t for t in cleaned if t]
    if not cleaned: return []
    try:
        tfidf = TfidfVectorizer(max_features=100, stop_words=list(STOP_WORDS),
                                token_pattern=rf"(?u)\b[^\W\d_]{{{min_word_length},}}\b")
        matrix = tfidf.fit_transform(cleaned)
        features = tfidf.get_feature_names_out()
        scores = np.array(matrix.sum(axis=0)).flatten()
        return [features[i] for i in scores.argsort()[::-1][:n] if scores[i] > 0]
    except:
        return []


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]


# Keywords
cluster_kw = {cid: extract_keywords(info['texts'], 30)[:10] for cid, info in cluster_info.items()}
all_kw = {}
for cid, kws in cluster_kw.items():
    for kw in kws:
        all_kw.setdefault(kw, []).append(cid)
for cid in cluster_kw:
    unique = [kw for kw in cluster_kw[cid] if len(all_kw[kw]) == 1][:10]
    if unique: cluster_kw[cid] = unique
    print(f"  Cluster {cid}: {cluster_kw[cid][:5]}")

# Class names
discovered_labels = TEST_3_NEW_LABELS
discovered_names = []
if 'label_name' in unknown_df.columns:
    for lid in discovered_labels:
        m = unknown_df[unknown_df['label'] == lid]['label_name']
        discovered_names.append(m.iloc[0] if len(m) > 0 else str(lid))
else:
    discovered_names = [str(l) for l in discovered_labels]

# Embeddings & similarity
kw_emb = np.array([get_embedding(' '.join(cluster_kw[cid])) for cid in range(K_FINAL)])
cls_emb = np.array([get_embedding(n) for n in discovered_names])

kw_norm = kw_emb / np.linalg.norm(kw_emb, axis=1, keepdims=True)
cls_norm = cls_emb / np.linalg.norm(cls_emb, axis=1, keepdims=True)
sim = kw_norm @ cls_norm.T

print(f"\n📊 Similarity Matrix:")
for cid in range(K_FINAL):
    print(f"C{cid}: {[f'{sim[cid, i]:.3f}' for i in range(len(discovered_names))]}")

row_ind, col_ind = linear_sum_assignment(-sim)
cluster_to_label_eval = {cid: discovered_labels[cidx] for cid, cidx in zip(row_ind, col_ind)}

print(f"\n🗺️  Mapping:")
for cid, cidx in zip(row_ind, col_ind):
    print(f"  Cluster {cid} → {discovered_labels[cidx]} ({discovered_names[cidx]})")

# =========================================================================
# STEP 8: SAMPLE SELECTION & SAVE
# =========================================================================

print(f"\n{'=' * 70}")
print("STEP 8: SAMPLE SELECTION")
print('=' * 70)

cluster_to_pseudo = {cid: PSEUDO_LABEL_START_T3 + cid for cid in range(K_FINAL)}

selected_idx = []
selected_labels = []

for cid in range(K_FINAL):
    idx = cluster_info[cid]['indices']
    emb = embeddings_unknown[idx]
    dist = np.linalg.norm(emb - centroids[cid], axis=1)
    purity = cluster_info[cid]['purity']

    ratio = min(SAMPLE_SELECTION_RATIO * (0.8 + 0.4 * purity), 0.8)
    n = max(1, int(len(idx) * ratio))

    nearest = dist.argsort()[:n]
    selected_idx.extend(idx[nearest])
    selected_labels.extend([cluster_to_pseudo[cid]] * n)

    print(f"  Cluster {cid}: {len(idx)} → {n}")

# Combine
new_data = pd.DataFrame({'content': [texts_unknown[i] for i in selected_idx], 'label': selected_labels})
all_data = pd.concat([known_df, new_data], ignore_index=True).sample(frac=1, random_state=42)

print(f"\n✅ Combined: {len(all_data)} samples")

# Save
all_data.to_csv(T3_PROCESSED_CSV, index=False)
print(f"✅ Saved: {T3_PROCESSED_CSV}")

results = {
    'K_final': K_FINAL, 'overall_purity': overall_purity, 'ari': ari,
    'cluster_to_pseudo': cluster_to_pseudo, 'cluster_to_label_eval': cluster_to_label_eval,
    'ood_threshold': threshold
}
with open(T3_RESULTS_PKL, 'wb') as f:
    pickle.dump(results, f)

print(f"\n{'=' * 70}")
print("✅ PIPELINE T3 COMPLETED!")
print('=' * 70)