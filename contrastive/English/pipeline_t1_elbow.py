# pipeline_t1_elbow_contrastive.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle

print("=" * 70)
print("PIPELINE T1 - ELBOW METHOD + CONTRASTIVE LEARNING")
print("=" * 70)

with open('embeddings_test_1.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)

discovered_texts = embeddings_dict['content']
discovered_embeddings = np.array(embeddings_dict['embeddings'])
unknown_labels_GT = embeddings_dict['label']  # [1, 3, 11, ...] Ground truth

print(f"\n📊 Dataset info:")
print(f"  Samples: {len(discovered_texts)}")
print(f"  Embedding dim: {discovered_embeddings.shape[1]}")
print(f"  Ground truth classes: {set(unknown_labels_GT)}")  # {1, 3, 11}

# =========================================================================
# STEP 1: ELBOW METHOD - Test different K values
# =========================================================================

print("\n" + "=" * 70)
print("STEP 1: ELBOW METHOD - Finding optimal K")
print("=" * 70)

K_range = range(3, 7)
metrics = {
    'K': [],
    'inertia': [],
    'silhouette': [],
    'davies_bouldin': [],
    'calinski_harabasz': []
}

for K in K_range:
    print(f"\n🔍 Testing K = {K}...")

    # KMeans clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(discovered_embeddings)

    # Metrici
    inertia = kmeans.inertia_
    silhouette = silhouette_score(discovered_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(discovered_embeddings, cluster_labels)
    calinski = calinski_harabasz_score(discovered_embeddings, cluster_labels)

    metrics['K'].append(K)
    metrics['inertia'].append(inertia)
    metrics['silhouette'].append(silhouette)
    metrics['davies_bouldin'].append(davies_bouldin)
    metrics['calinski_harabasz'].append(calinski)

    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz: {calinski:.2f} (higher is better)")

    # Calculează purity per cluster
    from collections import Counter

    purities = []
    for cluster_id in range(K):
        mask = (cluster_labels == cluster_id)
        true_labels_in_cluster = [unknown_labels_GT[i] for i, m in enumerate(mask) if m]
        if len(true_labels_in_cluster) > 0:
            counts = Counter(true_labels_in_cluster)
            dominant_count = counts.most_common(1)[0][1]
            purity = dominant_count / len(true_labels_in_cluster)
            purities.append(purity)

    avg_purity = np.mean(purities) if purities else 0
    print(f"  Average Purity: {avg_purity:.4f}")

# =========================================================================
# STEP 2: VIZUALIZARE ELBOW PLOTS
# =========================================================================

print("\n" + "=" * 70)
print("STEP 2: VISUALIZING ELBOW PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Inertia (SSE)
axes[0, 0].plot(metrics['K'], metrics['inertia'], 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 0].set_ylabel('Inertia (SSE)', fontsize=12)
axes[0, 0].set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Silhouette Score
axes[0, 1].plot(metrics['K'], metrics['silhouette'], 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Davies-Bouldin Index
axes[1, 0].plot(metrics['K'], metrics['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Calinski-Harabasz Index
axes[1, 1].plot(metrics['K'], metrics['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1, 1].set_ylabel('Calinski-Harabasz Score', fontsize=12)
axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_method_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Elbow plots saved: elbow_method_analysis.png")

# =========================================================================
# STEP 3: DETERMINARE K OPTIM (automată + manuală)
# =========================================================================

print("\n" + "=" * 70)
print("STEP 3: DETERMINING OPTIMAL K")
print("=" * 70)

# Metodă 1: Elbow pe Inertia (knee detection)
from kneed import KneeLocator

try:
    knee_locator = KneeLocator(metrics['K'], metrics['inertia'],
                               curve='convex', direction='decreasing')
    K_elbow = knee_locator.knee
    print(f"\n📍 Elbow method (Inertia): K = {K_elbow}")
except:
    K_elbow = None
    print(f"\n⚠️  Elbow method: No clear elbow detected")

# Metodă 2: Best Silhouette
K_silhouette = metrics['K'][np.argmax(metrics['silhouette'])]
print(f"📍 Best Silhouette: K = {K_silhouette} (score: {max(metrics['silhouette']):.4f})")

# Metodă 3: Best Davies-Bouldin (minimize)
K_db = metrics['K'][np.argmin(metrics['davies_bouldin'])]
print(f"📍 Best Davies-Bouldin: K = {K_db} (score: {min(metrics['davies_bouldin']):.4f})")

# Metodă 4: Best Calinski-Harabasz
K_ch = metrics['K'][np.argmax(metrics['calinski_harabasz'])]
print(f"📍 Best Calinski-Harabasz: K = {K_ch} (score: {max(metrics['calinski_harabasz']):.2f})")

# Voting/Consensus
from collections import Counter

votes = [k for k in [K_elbow, K_silhouette, K_db, K_ch] if k is not None]
K_consensus = Counter(votes).most_common(1)[0][0]

print(f"\n🗳️  Consensus K (most voted): {K_consensus}")
print(f"   Votes: {Counter(votes)}")

# MANUAL OVERRIDE (opțional)
K_optimal = K_consensus  # Sau poți seta manual: K_optimal = 5
print(f"\n✅ OPTIMAL K SELECTED: {K_optimal}")

# =========================================================================
# STEP 4: CLUSTERING CU K OPTIM
# =========================================================================

print("\n" + "=" * 70)
print(f"STEP 4: CLUSTERING WITH K = {K_optimal}")
print("=" * 70)

kmeans_final = KMeans(n_clusters=K_optimal, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(discovered_embeddings)

print(f"\n📊 Clustering results:")
print(f"  Number of clusters: {K_optimal}")
print(f"  Silhouette: {silhouette_score(discovered_embeddings, cluster_labels):.4f}")

# Analiză purity per cluster
print(f"\n📋 Cluster analysis:")
cluster_info = {}
for cluster_id in range(K_optimal):
    mask = (cluster_labels == cluster_id)
    cluster_size = mask.sum()

    true_labels_in_cluster = [unknown_labels_GT[i] for i, m in enumerate(mask) if m]
    counts = Counter(true_labels_in_cluster)
    dominant_class = counts.most_common(1)[0][0]
    dominant_count = counts.most_common(1)[0][1]
    purity = dominant_count / cluster_size

    cluster_info[cluster_id] = {
        'size': cluster_size,
        'dominant_class': dominant_class,
        'purity': purity,
        'distribution': dict(counts)
    }

    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {cluster_size}")
    print(f"  Dominant class: {dominant_class} (GT label)")
    print(f"  Purity: {purity:.2%}")
    print(f"  Distribution: {dict(counts)}")

# =========================================================================
# STEP 5: MAPARE K CLUSTERE → 3 CLASE GROUND TRUTH
# =========================================================================

print("\n" + "=" * 70)
print(f"STEP 5: MAPPING {K_optimal} CLUSTERS → 3 GROUND TRUTH CLASSES")
print("=" * 70)

# Ground truth classes: {1, 3, 11}
# Trebuie să grupăm K_optimal clustere în 3 grupe

# Strategie: Group clusters bazat pe clasa dominantă
from collections import defaultdict

class_to_clusters = defaultdict(list)

for cluster_id, info in cluster_info.items():
    dominant_class = info['dominant_class']
    class_to_clusters[dominant_class].append(cluster_id)

print(f"\n📍 Mapping strategy:")
for gt_class, clusters in sorted(class_to_clusters.items()):
    print(f"  GT class {gt_class}: Clusters {clusters}")

# Creare mapare: cluster_id → pseudo_label
# Pseudo-labels: 13, 14, 15 (pentru cele 3 clase GT: 3, 11, 1)
GT_TO_PSEUDO = {
    3: 13,  # Athlete → pseudo 13
    11: 14,  # Album → pseudo 14
    1: 15  # Edu → pseudo 15
}

cluster_to_pseudo = {}
for cluster_id, info in cluster_info.items():
    dominant_class = info['dominant_class']
    pseudo_label = GT_TO_PSEUDO[dominant_class]
    cluster_to_pseudo[cluster_id] = pseudo_label

print(f"\n🗺️  Cluster → Pseudo-label mapping:")
for cluster_id in sorted(cluster_to_pseudo.keys()):
    pseudo = cluster_to_pseudo[cluster_id]
    gt_class = cluster_info[cluster_id]['dominant_class']
    purity = cluster_info[cluster_id]['purity']
    print(f"  Cluster {cluster_id} → Pseudo {pseudo} (GT class {gt_class}, purity {purity:.2%})")

# =========================================================================
# STEP 6: SAMPLE SELECTION (40% nearest to centroid per cluster)
# =========================================================================

print("\n" + "=" * 70)
print("STEP 6: SAMPLE SELECTION (40% per cluster)")
print("=" * 70)

selected_indices = []
for cluster_id in range(K_optimal):
    mask = (cluster_labels == cluster_id)
    cluster_indices = np.where(mask)[0]
    cluster_embeddings = discovered_embeddings[mask]
    centroid = kmeans_final.cluster_centers_[cluster_id]

    # Distanță la centroid
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    sorted_indices = cluster_indices[np.argsort(distances)]

    # Select top 40%
    n_select = max(1, int(0.4 * len(sorted_indices)))
    selected = sorted_indices[:n_select]
    selected_indices.extend(selected)

    print(f"  Cluster {cluster_id}: {len(cluster_indices)} → {n_select} selected")

selected_indices = np.array(selected_indices)
print(f"\n✅ Total selected samples: {len(selected_indices)}")

# =========================================================================
# STEP 7: KEYWORD EXTRACTION (per cluster)
# =========================================================================

print("\n" + "=" * 70)
print("STEP 7: KEYWORD EXTRACTION (TF-IDF per cluster)")
print("=" * 70)

from sklearn.feature_extraction.text import TfidfVectorizer

cluster_keywords = {}
for cluster_id in range(K_optimal):
    mask = (cluster_labels == cluster_id)
    cluster_texts = [discovered_texts[i] for i, m in enumerate(mask) if m]

    if len(cluster_texts) < 2:
        cluster_keywords[cluster_id] = ['default', 'keywords', 'text']
        continue

    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(cluster_texts)

    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    top_indices = scores.argsort()[-3:][::-1]
    keywords = [feature_names[i] for i in top_indices]

    cluster_keywords[cluster_id] = keywords

    pseudo_label = cluster_to_pseudo[cluster_id]
    print(f"  Cluster {cluster_id} (Pseudo {pseudo_label}): {keywords}")

# =========================================================================
# STEP 8: KEYWORD EMBEDDINGS (K embeddings pentru K clustere)
# =========================================================================

print("\n" + "=" * 70)
print("STEP 8: COMPUTING KEYWORD EMBEDDINGS")
print("=" * 70)

from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

keyword_embeddings = {}
for cluster_id, keywords in cluster_keywords.items():
    keyword_text = " ".join(keywords)

    inputs = tokenizer(keyword_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        keyword_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    keyword_embeddings[cluster_id] = keyword_embedding
    print(f"  Cluster {cluster_id}: '{keyword_text}' → embedding shape {keyword_embedding.shape}")

# Save keyword embeddings
with open('keyword_embeddings_t1_elbow_contrastive.pkl', 'wb') as f:
    pickle.dump(keyword_embeddings, f)
print(f"\n✅ Keyword embeddings saved: keyword_embeddings_t1_elbow_contrastive.pkl")

# =========================================================================
# STEP 9: CREATE TRAINING DATASET
# =========================================================================

print("\n" + "=" * 70)
print("STEP 9: CREATING TRAINING DATASET")
print("=" * 70)

# Selected samples cu pseudo-labels
selected_texts = [discovered_texts[i] for i in selected_indices]
selected_cluster_labels = [cluster_labels[i] for i in selected_indices]
selected_pseudo_labels = [cluster_to_pseudo[cl] for cl in selected_cluster_labels]

cl_train_df = pd.DataFrame({
    'content': selected_texts,
    'label': selected_pseudo_labels
})

# Combine cu baseline data (replay)
baseline_train_df = pd.read_csv('/home/alin/Desktop/ContinualLearning/datasets/English/train.csv')
combined_train_df = pd.concat([baseline_train_df, cl_train_df], ignore_index=True)

combined_train_df.to_csv('cl_train_t1_elbow_contrastive.csv', index=False)
print(f"\n✅ Training dataset saved: cl_train_t1_elbow_contrastive.csv")
print(f"  Baseline samples: {len(baseline_train_df)}")
print(f"  CL samples (discovered): {len(cl_train_df)}")
print(f"  Total: {len(combined_train_df)}")

# Distribution check
print(f"\n📊 Label distribution in CL samples:")
print(cl_train_df['label'].value_counts().sort_index())

# =========================================================================
# STEP 10: CREATE LABEL MAPPINGS FOR CONTRASTIVE TRAINER
# =========================================================================

print("\n" + "=" * 70)
print("STEP 10: CREATING LABEL MAPPINGS FOR CONTRASTIVE TRAINER")
print("=" * 70)

# Label2id pentru training
unique_labels = sorted(combined_train_df['label'].unique())
label2id_cl = {str(label): idx for idx, label in enumerate(unique_labels)}
id2label_cl = {idx: str(label) for label, idx in label2id_cl.items()}

print(f"\n🏷️  label2id mapping:")
for label, idx in sorted(label2id_cl.items(), key=lambda x: int(x[0])):
    print(f"  {label} → {idx}")

# Cluster to label index mapping (pentru contrastive trainer)
# Pseudo-labels 13, 14, 15 vor avea indices în label2id
pseudo_to_index = {
    int(label): label2id_cl[label]
    for label in ['13', '14', '15']
}

print(f"\n🔗 Pseudo-label to index:")
for pseudo, idx in sorted(pseudo_to_index.items()):
    print(f"  Pseudo {pseudo} → Index {idx}")

# CRITICAL: label_to_cluster pentru ContrastiveTrainer
# Mapare: label_index → cluster_id (pentru contrastive loss)
label_to_cluster = {}
for cluster_id, pseudo_label in cluster_to_pseudo.items():
    label_index = pseudo_to_index[pseudo_label]
    label_to_cluster[label_index] = cluster_id

print(f"\n🎯 label_to_cluster mapping (for ContrastiveTrainer):")
for label_idx, cluster_id in sorted(label_to_cluster.items()):
    pseudo = [p for p, idx in pseudo_to_index.items() if idx == label_idx][0]
    print(f"  Label index {label_idx} (pseudo {pseudo}) → Cluster {cluster_id}")

# Save mappings
mappings = {
    'label2id': label2id_cl,
    'id2label': id2label_cl,
    'label_to_cluster': label_to_cluster,
    'cluster_to_pseudo': cluster_to_pseudo,
    'pseudo_to_index': pseudo_to_index,
    'K_optimal': K_optimal
}

with open('mappings_t1_elbow_contrastive.pkl', 'wb') as f:
    pickle.dump(mappings, f)
print(f"\n✅ Mappings saved: mappings_t1_elbow_contrastive.pkl")

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nKey outputs:")
print(f"  1. elbow_method_analysis.png - Elbow plots")
print(f"  2. keyword_embeddings_t1_elbow_contrastive.pkl - {K_optimal} keyword embeddings")
print(f"  3. cl_train_t1_elbow_contrastive.csv - Training dataset")
print(f"  4. mappings_t1_elbow_contrastive.pkl - All mappings")
print(f"\nNext step: Run train_cl_t1_elbow_contrastive.py")