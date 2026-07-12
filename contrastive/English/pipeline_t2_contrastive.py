# pipeline_t2_contrastive.py - Pipeline test_2 cu keyword embeddings

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
print("CONTINUAL LEARNING PIPELINE - TEST_2 (CONTRASTIVE)")
print("=" * 70)

# ============================================================================
# STEP 1: EXTRAGE UNKNOWN DIN TEST_2
# ============================================================================
print("\n" + "-" * 70)
print("STEP 1: EXTRAGERE UNKNOWN DIN TEST_2")
print("-" * 70)

test_2_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_2.csv")
print(f"Test_2 total: {len(test_2_df)} texte")

# Clase cunoscute după test_1: 0,1,2,3,7,11,12
KNOWN_CLASSES = [0, 1, 2, 3, 7, 11, 12]
unknown_df = test_2_df[~test_2_df['label'].isin(KNOWN_CLASSES)].copy()

unknown_texts = unknown_df['content'].tolist()
unknown_labels = unknown_df['label'].tolist()

print(f"\nClase known (după test_1): {KNOWN_CLASSES}")
print(f"Clase unknown (noi): {sorted(set(unknown_labels))}")
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


unknown_embeddings = get_bert_embeddings(unknown_texts, "./ckpt_cl_t1_contrastive/final")

# ============================================================================
# STEP 3: CLUSTERING
# ============================================================================
print("\n" + "-" * 70)
print("STEP 3: K-MEANS CLUSTERING (k=3)")
print("-" * 70)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(unknown_embeddings)
centroids = kmeans.cluster_centers_

sil = silhouette_score(unknown_embeddings, cluster_labels)
label_map = {4: 0, 6: 1, 8: 2}
true_cluster_ids = [label_map[l] for l in unknown_labels]
ari = adjusted_rand_score(true_cluster_ids, cluster_labels)

print(f"Silhouette: {sil:.3f}, ARI: {ari:.3f}")

# Purity
class_names_display = {4: 'OfficeHolder', 6: 'Building', 8: 'Village'}
for cid in range(3):
    mask = (cluster_labels == cid)
    true_in_cluster = [l for l, m in zip(unknown_labels, mask) if m]
    if len(true_in_cluster) == 0:
        continue
    label_counts = pd.Series(true_in_cluster).value_counts()
    dominant = label_counts.idxmax()
    purity = label_counts.max() / len(true_in_cluster)
    print(f"  Cluster {cid}: Dominant={dominant} ({class_names_display[dominant]}), Purity={purity:.2%}")

# ============================================================================
# STEP 4: TF-IDF KEYWORDS
# ============================================================================
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

# ============================================================================
# STEP 4.5: KEYWORD EMBEDDINGS (pentru contrastive loss)
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


keyword_embeddings = get_keyword_embeddings(cluster_keywords, "./ckpt_cl_t1_contrastive/final")

with open('keyword_embeddings_t2_contrastive.pkl', 'wb') as f:
    pickle.dump(keyword_embeddings, f)

print(f"\n✓ Keyword embeddings salvate: keyword_embeddings_t2_contrastive.pkl")

# ============================================================================
# STEP 5: SAMPLE SELECTION
# ============================================================================
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
    pseudo_label = 16 + cid  # 16, 17, 18

    selected_texts.extend(selected)
    selected_pseudo_labels.extend([pseudo_label] * len(selected))

    print(f"  Cluster {cid} → Pseudo-label {pseudo_label}: {len(selected)} selected")

print(f"\n✓ Total selected: {len(selected_texts)} texte")

# ============================================================================
# STEP 6: REPLAY - TOATE datele din cl_train_t1_contrastive.csv
# ============================================================================
print("\n" + "-" * 70)
print("STEP 6: REPLAY BUFFER")
print("-" * 70)

cl_t1_df = pd.read_csv("./cl_train_t1_contrastive.csv")

replay_texts = cl_t1_df['content'].tolist()
replay_labels = cl_t1_df['label'].tolist()

print(f"Replay din cl_train_t1: {len(replay_texts)} texte")

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
cl_df.to_csv('./cl_train_t2_contrastive.csv', index=False)

print(f"\n✓ Dataset salvat: ./cl_train_t2_contrastive.csv")
print(f"  Total: {len(cl_df)} examples")
print(f"  Ratio replay:new = {len(replay_texts) / len(selected_texts):.2f}:1")

print("\n" + "=" * 70)
print("PIPELINE T2 CONTRASTIVE COMPLET!")
print("=" * 70)
print("Următorul pas: python train_cl_t2_contrastive.py")
print("=" * 70)

""" python pipeline_t2_contrastive.py
======================================================================
CONTINUAL LEARNING PIPELINE - TEST_2 (CONTRASTIVE)
======================================================================

----------------------------------------------------------------------
STEP 1: EXTRAGERE UNKNOWN DIN TEST_2
----------------------------------------------------------------------
Test_2 total: 4500 texte

Clase known (după test_1): [0, 1, 2, 3, 7, 11, 12]
Clase unknown (noi): [4, 6, 8]
Total unknown: 1350 texte
Distribuție unknown:
  Clasa 4: 450 texte
  Clasa 6: 450 texte
  Clasa 8: 450 texte

----------------------------------------------------------------------
STEP 2: EXTRAGERE EMBEDDINGS BERT
----------------------------------------------------------------------
  Procesare 1350 texte...

----------------------------------------------------------------------
STEP 3: K-MEANS CLUSTERING (k=3)
----------------------------------------------------------------------
Silhouette: 0.745, ARI: 0.731
  Cluster 0: Dominant=6 (Building), Purity=98.14%
  Cluster 1: Dominant=8 (Village), Purity=76.95%
  Cluster 2: Dominant=4 (OfficeHolder), Purity=98.89%

----------------------------------------------------------------------
STEP 4: TF-IDF KEYWORDS
----------------------------------------------------------------------
  Cluster 0: ['building', 'church', 'historic', 'located', 'house']
  Cluster 1: ['village', 'district', 'county', 'population', 'census']
  Cluster 2: ['born', 'politician', 'member', 'minister', 'served']

----------------------------------------------------------------------
STEP 4.5: KEYWORD EMBEDDINGS PENTRU CONTRASTIVE LOSS
----------------------------------------------------------------------
  Cluster 0: 'building church historic'
  Cluster 1: 'village district county'
  Cluster 2: 'born politician member'

✓ Keyword embeddings salvate: keyword_embeddings_t2_contrastive.pkl

----------------------------------------------------------------------
STEP 5: SAMPLE SELECTION (TOP 40%)
----------------------------------------------------------------------
  Cluster 0 → Pseudo-label 16: 129 selected
  Cluster 1 → Pseudo-label 17: 231 selected
  Cluster 2 → Pseudo-label 18: 181 selected

✓ Total selected: 541 texte

----------------------------------------------------------------------
STEP 6: REPLAY BUFFER
----------------------------------------------------------------------
Replay din cl_train_t1: 6541 texte

----------------------------------------------------------------------
STEP 7: COMBINE ȘI SALVARE
----------------------------------------------------------------------

✓ Dataset salvat: ./cl_train_t2_contrastive.csv
  Total: 7082 examples
  Ratio replay:new = 12.09:1

======================================================================
PIPELINE T2 CONTRASTIVE COMPLET!
======================================================================
Următorul pas: python train_cl_t2_contrastive.py
======================================================================"""