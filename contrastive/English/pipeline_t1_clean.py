# pipeline_t1_clean.py - Continual learning test_1 (cu printuri)
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 70)
print("CONTINUAL LEARNING PIPELINE - TEST_1 (CLEAN)")
print("=" * 70)

# STEP 1: EXTRAGE UNKNOWN DIN TEST_1
print("\n" + "-" * 70)
print("STEP 1: EXTRAGERE UNKNOWN DIN TEST_1")
print("-" * 70)

test_1_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_1.csv")
print(f"Test_1 total: {len(test_1_df)} texte")
print(f"Distributie toate clasele:")
print(test_1_df['label'].value_counts().sort_index())

KNOWN_CLASSES = [0, 2, 7, 12]
unknown_df = test_1_df[~test_1_df['label'].isin(KNOWN_CLASSES)].copy()

unknown_texts = unknown_df['content'].tolist()
unknown_labels = unknown_df['label'].tolist()

print(f"\nClase known: {KNOWN_CLASSES}")
print(f"Clase unknown (noi): {sorted(set(unknown_labels))}")
print(f"Total unknown: {len(unknown_texts)} texte")
print(f"Distributie unknown:")
for label in sorted(set(unknown_labels)):
    count = unknown_labels.count(label)
    print(f"Clasa {label}: {count} texte")

# STEP 2: EMBEDDINGS BERT
print("\n" + "-" * 70)
print("STEP 2: EXTRAGERE EMBEDDINGS BERT")
print("-" * 70)

def get_bert_embeddings(texts, model_path, batch_size=32):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    print(f"  Model incarcat din: {model_path}")
    print(f"  Batch size: {batch_size}")
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

        if (i // batch_size) % 10 == 0:
            print(f"  Procesate: {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    return embeddings

unknown_embeddings = get_bert_embeddings(unknown_texts, "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final")

# STEP 3: K-MEANS CLUSTERING
print("\n" + "-" * 70)
print("STEP 3: K-MEANS CLUSTERING")
print("-" * 70)

print("Testing k values pentru elbow method...")
for k in range(2, 7):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_test = kmeans_test.fit_predict(unknown_embeddings)
    sil_test = silhouette_score(unknown_embeddings, labels_test)
    print(f"  k={k}: Silhouette={sil_test:.3f}")

print("\nClustering final cu k=3")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(unknown_embeddings)
centroids = kmeans.cluster_centers_

print(f"✓ Clustering complet")
print(f"  Centroids shape: {centroids.shape}")
print(f"  Inertia: {kmeans.inertia_:.2f}")

sil = silhouette_score(unknown_embeddings, cluster_labels)
label_map = {1: 0, 3: 1, 11: 2}
true_cluster_ids = [label_map[l] for l in unknown_labels]
ari = adjusted_rand_score(true_cluster_ids, cluster_labels)
nmi = normalized_mutual_info_score(true_cluster_ids, cluster_labels)

print(f"\nMETRICI CLUSTERING:")
print(f"  Silhouette score: {sil:.3f}")
print(f"  Adjusted Rand Index: {ari:.3f}")
print(f"  Normalized Mutual Info: {nmi:.3f}")

print(f"\nPURITY PER CLUSTER:")
class_names = {1: 'EducationalInstitution', 3: 'Athlete', 11: 'Album'}

for cid in range(3):
    mask = (cluster_labels == cid)
    true_in_cluster = [l for l, m in zip(unknown_labels, mask) if m]

    if len(true_in_cluster) == 0:
        continue

    label_counts = pd.Series(true_in_cluster).value_counts()
    dominant = label_counts.idxmax()
    purity = label_counts.max() / len(true_in_cluster)

    print(f"\n  Cluster {cid}:")
    print(f"    Size: {len(true_in_cluster)}")
    print(f"    Dominant: Clasa {dominant} ({class_names[dominant]})")
    print(f"    Purity: {purity:.2%}")
    print(f"    Distribution: {label_counts.to_dict()}")

# STEP 4: TF-IDF KEYWORDS
print("\n" + "-" * 70)
print("STEP 4: TF-IDF KEYWORDS")
print("-" * 70)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50,
    stop_words='english',
    max_df=0.8,
    min_df=2
)

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

# STEP 4.5: KEYWORD EMBEDDINGS pentru Contrastive Learning
print("\n" + "-" * 70)
print("STEP 4.5: KEYWORD EMBEDDINGS PENTRU CONTRASTIVE LEARNING")
print("-" * 70)


def get_keyword_embeddings(keywords_dict, model_path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    keyword_embeddings = {}

    for cluster_id, keywords in keywords_dict.items():
        # Concatenează top 3 keywords
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

import pickle

with open('keyword_embeddings_t1.pkl', 'wb') as f:
    pickle.dump(keyword_embeddings, f)

print("\n✓ Keyword embeddings salvate în: keyword_embeddings_t1.pkl")
print(f"  Număr clustere: {len(keyword_embeddings)}")
print(f"  Dimensiune embedding: {list(keyword_embeddings.values())[0].shape}")

# STEP 5: SAMPLE SELECTION
print("\n" + "-" * 70)
print("STEP 5: SAMPLE SELECTION (TOP 40% APROAPE DE CENTROID)")
print("-" * 70)

PERCENTILE = 40
selected_texts = []
selected_pseudo_labels = []
selected_true_labels = []

for cid in range(3):
    mask = (cluster_labels == cid)
    cluster_emb = unknown_embeddings[mask]
    cluster_txts = [t for t, m in zip(unknown_texts, mask) if m]
    cluster_true = [l for l, m in zip(unknown_labels, mask) if m]

    distances = np.linalg.norm(cluster_emb - centroids[cid], axis=1)
    threshold = np.percentile(distances, PERCENTILE)
    close_mask = (distances <= threshold)

    selected = [t for t, close in zip(cluster_txts, close_mask) if close]
    selected_true = [l for l, close in zip(cluster_true, close_mask) if close]

    pseudo_label = 13 + cid

    selected_texts.extend(selected)
    selected_pseudo_labels.extend([pseudo_label] * len(selected))
    selected_true_labels.extend(selected_true)

    if len(selected_true) > 0:
        dominant = pd.Series(selected_true).mode()[0]
        purity_selected = (pd.Series(selected_true) == dominant).mean()

        print(f"\n  Cluster {cid} → Pseudo-label {pseudo_label}:")
        print(f"    Total în cluster: {len(cluster_txts)}")
        print(f"    Selected (top {PERCENTILE}%): {len(selected)}")
        print(f"    Purity în selected: {purity_selected:.2%}")
        print(f"    Distribution: {pd.Series(selected_true).value_counts().to_dict()}")

print(f"\nTotal selected pentru CL: {len(selected_texts)} texte")


# STEP 6: REPLAY BUFFER (75% din train)
print("\n" + "-" * 70)
print("STEP 6: REPLAY BUFFER")
print("-" * 70)

train_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/train.csv")
REPLAY_PER_CLASS = 1500

print(f"Replay strategy: {REPLAY_PER_CLASS} samples/class (75% din available)")

replay_texts = []
replay_labels = []

for label in [0, 2, 7, 12]:
    class_data = train_df[train_df['label'] == label]
    n_available = len(class_data)
    n = min(REPLAY_PER_CLASS, n_available)
    sampled = class_data.sample(n, random_state=42)

    replay_texts.extend(sampled['content'].tolist())
    replay_labels.extend([label] * n)

    print(f"  Clasa {label}: {n}/{n_available} samples ({n / n_available * 100:.1f}%)")

print(f"\n✓ Total replay: {len(replay_texts)} texte")

# STEP 7: COMBINE ȘI SALVARE
print("\n" + "-" * 70)
print("STEP 7: COMBINE ȘI SALVARE")
print("-" * 70)

all_texts = replay_texts + selected_texts
all_labels = replay_labels + selected_pseudo_labels

print(f"Datе finale pentru CL:")
print(f"  Replay (known): {len(replay_texts)}")
print(f"  Discovered (new): {len(selected_texts)}")
print(f"  Total: {len(all_texts)}")
print(f"  Ratio known:new = {len(replay_texts) / len(selected_texts):.2f}:1")

cl_df = pd.DataFrame({
    'content': all_texts,
    'label': all_labels
})
cl_df['y_soft'] = None

cl_df.to_csv('./cl_train_t1_clean.csv', index=False)

print(f"\n✓ Dataset salvat în: ./cl_train_t1_clean.csv")
print(f"\nDistribuție finală:")
print(cl_df['label'].value_counts().sort_index())


print("\n" + "=" * 70)
print("PIPELINE COMPLET!")
print("=" * 70)
print("Următorul pas: python train_cl_t1_clean.py")
print("=" * 70)