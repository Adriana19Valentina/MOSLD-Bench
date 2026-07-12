# pipeline_t1_contrastive.py - Pipeline CL cu keyword embeddings pentru contrastive

import pandas as pd
import numpy as np
import torch
import json
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from auto_mapping import automatic_cluster_mapping

print("=" * 70)
print("CONTINUAL LEARNING PIPELINE - TEST_1 (CONTRASTIVE + MAPARE AUTOMATA)")
print("=" * 70)


test_1_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_1.csv")
KNOWN_CLASSES = [0, 2, 7, 12]
unknown_df = test_1_df[~test_1_df['label'].isin(KNOWN_CLASSES)].copy()
unknown_texts = unknown_df['content'].tolist()
unknown_labels = unknown_df['label'].tolist()

print(f"\nClase unknown (noi): {sorted(set(unknown_labels))}")
print(f"Total unknown: {len(unknown_texts)} texte")


# Embeddings
def get_bert_embeddings(texts, model_path, batch_size=32):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    print(f"Procesare {len(texts)} texte...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model.bert(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)


unknown_embeddings = get_bert_embeddings(unknown_texts, "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final")

# Clustering
print("\nClustering k-means (k=4)")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(unknown_embeddings)
centroids = kmeans.cluster_centers_

sil = silhouette_score(unknown_embeddings, cluster_labels)
label_map = {1: 0, 3: 1, 11: 2}
true_cluster_ids = [label_map[l] for l in unknown_labels]
ari = adjusted_rand_score(true_cluster_ids, cluster_labels)

print(f"Silhouette: {sil:.3f}, ARI: {ari:.3f}")

# STEP 4: TF-IDF KEYWORDS
print("\n" + "-" * 70)
print("STEP 4: TF-IDF KEYWORDS")
print("-" * 70)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50,
                             stop_words='english', max_df=0.8, min_df=2)

cluster_keywords = {}

for cid in range(3):
    mask = (cluster_labels == cid)
    cluster_texts = [t for t, m in zip(unknown_texts, mask) if m]

    tfidf = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()
    avg_tfidf = np.asarray(tfidf.mean(axis=0)).flatten()

    top_idx = avg_tfidf.argsort()[-5:][::-1]
    keywords = [feature_names[i] for i in top_idx]
    cluster_keywords[cid] = keywords

    print(f"  Cluster {cid}: {keywords}")


# STEP 4.5: MAPARE AUTOMATa (pentru evaluare)

print("\n" + "-" * 70)
print("STEP 4.5: MAPARE AUTOMATĂ CLUSTER → GROUND TRUTH")
print("-" * 70)

ground_truth_classes = {
    1: 'EducationalInstitution',
    3: 'Athlete',
    11: 'Album'
}

auto_mapping, avg_similarity = automatic_cluster_mapping(
    cluster_keywords,
    ground_truth_classes
)

mapping_data = {
    'cluster_to_class': auto_mapping,
    'average_similarity': float(avg_similarity),
    'cluster_keywords': cluster_keywords,
    'ground_truth_classes': ground_truth_classes
}

with open('auto_mapping_t1_contrastive.json', 'w') as f:
    json.dump(mapping_data, f, indent=2)

print(f"\n✓ Mapare automata salvata: auto_mapping_t1_contrastive.json")
print(f"  Average similarity: {avg_similarity:.4f}")

# STEP 4.6: KEYWORD EMBEDDINGS (pentru contrastive loss)

print("\n" + "-" * 70)
print("STEP 4.6: KEYWORD EMBEDDINGS PENTRU CONTRASTIVE LOSS")
print("-" * 70)


def get_keyword_embeddings(keywords_dict, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    keyword_embeddings = {}

    for cluster_id, keywords in keywords_dict.items():
        # Concateneaza top 3 keywords
        keyword_text = " ".join(keywords[:3])

        print(f"  Cluster {cluster_id}: '{keyword_text}'")

        inputs = tokenizer(keyword_text, return_tensors='pt',
                           max_length=256, truncation=True)

        with torch.no_grad():
            outputs = model.bert(**inputs)
            keyword_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        keyword_embeddings[cluster_id] = keyword_emb.squeeze()

    return keyword_embeddings


keyword_embeddings = get_keyword_embeddings(cluster_keywords, "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final")

with open('keyword_embeddings_t1_contrastive.pkl', 'wb') as f:
    pickle.dump(keyword_embeddings, f)

print(f"\n✓ Keyword embeddings salvate: keyword_embeddings_t1_contrastive.pkl")
print(f"  Numar clustere: {len(keyword_embeddings)}")
print(f"  Dimensiune embedding: {list(keyword_embeddings.values())[0].shape}")


# STEP 5-7: Sample selection, Replay, Combine
print("\n" + "-" * 70)
print("STEP 5: SAMPLE SELECTION (TOP 40%)")
print("-" * 70)

PERCENTILE = 40
selected_texts = []
selected_pseudo_labels = []

for cid in range(3):
    mask = (cluster_labels == cid)
    cluster_emb = unknown_embeddings[mask]
    cluster_txts = [t for t, m in zip(unknown_texts, mask) if m]

    distances = np.linalg.norm(cluster_emb - centroids[cid], axis=1)
    threshold = np.percentile(distances, PERCENTILE)
    close_mask = (distances <= threshold)

    selected = [t for t, close in zip(cluster_txts, close_mask) if close]
    pseudo_label = 13 + cid

    selected_texts.extend(selected)
    selected_pseudo_labels.extend([pseudo_label] * len(selected))

    print(f"  Cluster {cid} → Pseudo-label {pseudo_label}: {len(selected)} selected")

print(f"\n✓ Total selected: {len(selected_texts)} texte")

# Replay
train_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/train.csv")
REPLAY_PER_CLASS = 1500

replay_texts = []
replay_labels = []

for label in [0, 2, 7, 12]:
    class_data = train_df[train_df['label'] == label]
    n = min(REPLAY_PER_CLASS, len(class_data))
    sampled = class_data.sample(n, random_state=42)

    replay_texts.extend(sampled['content'].tolist())
    replay_labels.extend([label] * n)

print(f"\nReplay: {len(replay_texts)} texte")

# Combine
all_texts = replay_texts + selected_texts
all_labels = replay_labels + selected_pseudo_labels

cl_df = pd.DataFrame({'content': all_texts, 'label': all_labels})
cl_df['y_soft'] = None
cl_df.to_csv('./cl_train_t1_contrastive.csv', index=False)

print(f"\n✓ Dataset salvat: ./cl_train_t1_contrastive.csv")
print(f"  Total: {len(cl_df)} examples")
print(f"  Ratio known:new = {len(replay_texts) / len(selected_texts):.2f}:1")

print("\n" + "=" * 70)
print("PIPELINE CONTRASTIVE COMPLET!")
print("=" * 70)
print(f"Mapare automata similarity: {avg_similarity:.4f}")
print("=" * 70)