# auto_mapping.py - Mapare automată cluster → ground truth class

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def get_text_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=256, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding.squeeze()


def compute_cluster_embeddings(cluster_keywords, model, tokenizer):
    cluster_embeddings = {}

    for cluster_id, keywords in cluster_keywords.items():
        # Concatenează top 3 keywords
        keyword_text = " ".join(keywords[:3])
        embedding = get_text_embedding(keyword_text, model, tokenizer)
        cluster_embeddings[cluster_id] = embedding

    return cluster_embeddings


def compute_class_embeddings(class_names, model, tokenizer):
    class_embeddings = {}

    for class_id, class_name in class_names.items():
        embedding = get_text_embedding(class_name, model, tokenizer)
        class_embeddings[class_id] = embedding

    return class_embeddings


def map_clusters_to_classes(cluster_embeddings, class_embeddings):
    """
    Mapare optimă cluster → class folosind Hungarian algorithm.
    Maximizează suma cosine similarities.
    """
    cluster_ids = sorted(cluster_embeddings.keys())
    class_ids = sorted(class_embeddings.keys())

    cluster_matrix = np.array([cluster_embeddings[cid] for cid in cluster_ids])
    class_matrix = np.array([class_embeddings[cid] for cid in class_ids])

    # Cosine similarity matrix: (n_clusters, n_classes)
    similarity_matrix = cosine_similarity(cluster_matrix, class_matrix)

    print("\nCosine Similarity Matrix:")
    print(f"{'':>12}", end="")
    for class_id in class_ids:
        print(f"Class_{class_id:>2}  ", end="")
    print()

    for i, cluster_id in enumerate(cluster_ids):
        print(f"Cluster {cluster_id:>4}", end="")
        for j in range(len(class_ids)):
            print(f"{similarity_matrix[i, j]:>10.4f}", end="")
        print()

    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    cluster_to_class = {}
    total_similarity = 0

    for cluster_idx, class_idx in zip(row_ind, col_ind):
        cluster_id = cluster_ids[cluster_idx]
        class_id = class_ids[class_idx]
        similarity = similarity_matrix[cluster_idx, class_idx]

        cluster_to_class[cluster_id] = class_id
        total_similarity += similarity
        print(f"\n✓ Cluster {cluster_id} → Class {class_id} (similarity: {similarity:.4f})")

    avg_similarity = total_similarity / len(row_ind)
    print(f"\nAverage mapping similarity: {avg_similarity:.4f}")

    return cluster_to_class, avg_similarity


def automatic_cluster_mapping(cluster_keywords, ground_truth_classes,
                              model_name="bert-base-multilingual-cased"):

    print("=" * 70)
    print("MAPARE AUTOMAT CLUSTERE → CLASE GROUND TRUTH")
    print("=" * 70)

    print(f"\nÎncărcare model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    print("\nCluster keywords:")
    for cid, kw in cluster_keywords.items():
        print(f"  Cluster {cid}: {kw[:3]}")

    print("\nGround truth classes:")
    for cid, name in ground_truth_classes.items():
        print(f"  Class {cid}: {name}")

    print("\nCalculare embeddings...")
    cluster_emb = compute_cluster_embeddings(cluster_keywords, model, tokenizer)
    class_emb = compute_class_embeddings(ground_truth_classes, model, tokenizer)

    mapping, avg_sim = map_clusters_to_classes(cluster_emb, class_emb)

    print("\n" + "=" * 70)
    print("MAPARE FINALA:")
    for cluster_id in sorted(mapping.keys()):
        class_id = mapping[cluster_id]
        class_name = ground_truth_classes[class_id]
        keywords = ", ".join(cluster_keywords[cluster_id][:3])
        print(f"  Cluster {cluster_id} ({keywords}) → Class {class_id} ({class_name})")
    print("=" * 70)

    return mapping, avg_sim