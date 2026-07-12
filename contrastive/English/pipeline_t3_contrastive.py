# pipeline_t3_contrastive.py - Pipeline test_3 cu keyword embeddings

import pandas as pd
import numpy as np
import torch
import json
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 70)
print("CONTINUAL LEARNING PIPELINE - TEST_3 (CONTRASTIVE)")
print("=" * 70)

# ============================================================================
# STEP 1: EXTRAGE UNKNOWN DIN TEST_3
# ============================================================================
print("\n" + "-" * 70)
print("STEP 1: EXTRAGERE UNKNOWN DIN TEST_3")
print("-" * 70)

test_3_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_3.csv")
print(f"Test_3 total: {len(test_3_df)} texte")

# Clase cunoscute după test_2: 0,1,2,3,4,6,7,8,11,12
KNOWN_CLASSES = [0, 1, 2, 3, 4, 6, 7, 8, 11, 12]
unknown_df = test_3_df[~test_3_df['label'].isin(KNOWN_CLASSES)].copy()

unknown_texts = unknown_df['content'].tolist()
unknown_labels = unknown_df['label'].tolist()

print(f"\nClase known (după test_2): {KNOWN_CLASSES}")
print(f"Clase unknown (noi): {sorted(set(unknown_labels))}")
print(f"  5: MeanOfTransportation")
print(f"  9: WrittenWork")
print(f"  10: Plant")
print(f"  13: Animal")
print(f"Total unknown: {len(unknown_texts)} texte")
print(f"Distribuție unknown:")
for label in sorted(set(unknown_labels)):
    count = unknown_labels.count(label)
    print(f"  Clasa {label}: {count} texte")

# ============================================================================
# STEP 2: EMBEDDINGS
# ============================================================================
print("\n" + "-" * 70)
print("STEP 2: EXTRAGERE EMBEDDINGS BERT")
print("-" * 70)


def get_bert_embeddings(texts, model_path, batch_size=32):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    print(f"  Procesare {len(texts)} texte...")
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


unknown_embeddings = get_bert_embeddings(unknown_texts, "./ckpt_cl_t2_contrastive/final")

# ============================================================================
# STEP 3: CLUSTERING (k=4 pentru 4 clase)
# ============================================================================
print("\n" + "-" * 70)
print("STEP 3: K-MEANS CLUSTERING (k=4)")
print("-" * 70)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(unknown_embeddings)
centroids = kmeans.cluster_centers_

sil = silhouette_score(unknown_embeddings, cluster_labels)
label_map = {5: 0, 9: 1, 10: 2, 13: 3}
true_cluster_ids = [label_map[l] for l in unknown_labels]
ari = adjusted_rand_score(true_cluster_ids, cluster_labels)

print(f"Silhouette: {sil:.3f}, ARI: {ari:.3f}")

# Purity
class_names_display = {5: 'MeanOfTransportation', 9: 'WrittenWork', 10: 'Plant', 13: 'Animal'}
for cid in range(4):
    mask = (cluster_labels == cid)
    true_in_cluster = [l for l, m in zip(unknown_labels, mask) if m]
    if len(true_in_cluster) == 0:
        continue
    label_counts = pd.Series(true_in_cluster).value_counts()
    dominant = label_counts.idxmax()
    purity = label_counts.max() / len(true_in_cluster)
    print(f"  Cluster {cid}: Dominant={dominant} ({class_names_display[dominant]}), Purity={purity:.2%}")
    print(f"    Distribution: {label_counts.to_dict()}")

# ============================================================================
# STEP 4: TF-IDF KEYWORDS
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4: TF-IDF KEYWORDS")
print("-" * 70)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=50,
                             stop_words='english', max_df=0.8, min_df=2)

cluster_keywords = {}

for cid in range(4):
    mask = (cluster_labels == cid)
    cluster_texts = [t for t, m in zip(unknown_texts, mask) if m]

    if len(cluster_texts) < 3:
        print(f"  Cluster {cid}: prea puține texte")
        continue

    tfidf = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()
    avg_tfidf = np.asarray(tfidf.mean(axis=0)).flatten()

    top_idx = avg_tfidf.argsort()[-5:][::-1]
    keywords = [feature_names[i] for i in top_idx]
    cluster_keywords[cid] = keywords

    print(f"  Cluster {cid}: {keywords}")

# ============================================================================
# STEP 4.5: KEYWORD EMBEDDINGS
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4.5: KEYWORD EMBEDDINGS PENTRU CONTRASTIVE LOSS")
print("-" * 70)


def get_keyword_embeddings(keywords_dict, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    keyword_embeddings = {}

    for cluster_id, keywords in keywords_dict.items():
        keyword_text = " ".join(keywords[:3])
        print(f"  Cluster {cluster_id}: '{keyword_text}'")

        inputs = tokenizer(keyword_text, return_tensors='pt',
                           max_length=256, truncation=True)

        with torch.no_grad():
            outputs = model.bert(**inputs)
            keyword_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        keyword_embeddings[cluster_id] = keyword_emb.squeeze()

    return keyword_embeddings


keyword_embeddings = get_keyword_embeddings(cluster_keywords, "./ckpt_cl_t2_contrastive/final")

with open('keyword_embeddings_t3_contrastive.pkl', 'wb') as f:
    pickle.dump(keyword_embeddings, f)

print(f"\n✓ Keyword embeddings salvate: keyword_embeddings_t3_contrastive.pkl")

# ============================================================================
# STEP 5: SAMPLE SELECTION
# ============================================================================
print("\n" + "-" * 70)
print("STEP 5: SAMPLE SELECTION (TOP 40%)")
print("-" * 70)

PERCENTILE = 40
selected_texts = []
selected_pseudo_labels = []

for cid in range(4):
    mask = (cluster_labels == cid)
    cluster_emb = unknown_embeddings[mask]
    cluster_txts = [t for t, m in zip(unknown_texts, mask) if m]

    distances = np.linalg.norm(cluster_emb - centroids[cid], axis=1)
    threshold = np.percentile(distances, PERCENTILE)
    close_mask = (distances <= threshold)

    selected = [t for t, close in zip(cluster_txts, close_mask) if close]
    pseudo_label = 19 + cid  # 19, 20, 21, 22

    selected_texts.extend(selected)
    selected_pseudo_labels.extend([pseudo_label] * len(selected))

    print(f"  Cluster {cid} → Pseudo-label {pseudo_label}: {len(selected)} selected")

print(f"\n✓ Total selected: {len(selected_texts)} texte")

# ============================================================================
# STEP 6: REPLAY
# ============================================================================
print("\n" + "-" * 70)
print("STEP 6: REPLAY BUFFER")
print("-" * 70)

cl_t2_df = pd.read_csv("./cl_train_t2_contrastive.csv")

replay_texts = cl_t2_df['content'].tolist()
replay_labels = cl_t2_df['label'].tolist()

print(f"Replay din cl_train_t2: {len(replay_texts)} texte")

# ============================================================================
# STEP 7: COMBINE
# ============================================================================
print("\n" + "-" * 70)
print("STEP 7: COMBINE ȘI SALVARE")
print("-" * 70)

all_texts = replay_texts + selected_texts
all_labels = replay_labels + selected_pseudo_labels

cl_df = pd.DataFrame({'content': all_texts, 'label': all_labels})
cl_df['y_soft'] = None
cl_df.to_csv('./cl_train_t3_contrastive.csv', index=False)

print(f"\n✓ Dataset salvat: ./cl_train_t3_contrastive.csv")
print(f"  Total: {len(cl_df)} examples")
if len(selected_texts) > 0:
    print(f"  Ratio replay:new = {len(replay_texts) / len(selected_texts):.2f}:1")

print("\n" + "=" * 70)
print("PIPELINE T3 CONTRASTIVE COMPLET!")
print("=" * 70)
print("Următorul pas: python train_cl_t3_contrastive.py")
print("=" * 70)