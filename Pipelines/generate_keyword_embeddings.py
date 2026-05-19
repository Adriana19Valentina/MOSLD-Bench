import argparse
import pickle
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def generate_keyword_embeddings(results_path, model_name, output_path, max_keywords=15):
    print(f"\n{'=' * 60}")
    print("GENERATING KEYWORD EMBEDDINGS")
    print('=' * 60)

    print(f"Loading results from: {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    cluster_keywords = results.get('cluster_keywords', {})
    cluster_unique = results.get('cluster_unique_keywords', {})

    if cluster_unique and all(len(v) > 0 for v in cluster_unique.values()):
        keywords_to_use = {int(k): v for k, v in cluster_unique.items()}
        print("  Using UNIQUE keywords")
    else:
        keywords_to_use = {int(k): v for k, v in cluster_keywords.items()}
        print("  Using regular keywords")

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Device: {device}")

    hidden_size = model.config.hidden_size
    keyword_embeddings = {}

    print(f"\nProcessing {len(keywords_to_use)} clusters...")

    for cluster_id in sorted(keywords_to_use.keys()):
        keywords = keywords_to_use[cluster_id][:max_keywords]

        if not keywords:
            print(f"  Cluster {cluster_id}: NO KEYWORDS - using zero vector")
            keyword_embeddings[cluster_id] = np.zeros(hidden_size)
            continue

        embeddings = []
        for kw in keywords:
            inputs = tokenizer(kw, return_tensors='pt', truncation=True, max_length=32)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(emb[0])

        keyword_embeddings[cluster_id] = np.mean(embeddings, axis=0)
        print(f"  Cluster {cluster_id}: {len(keywords)} keywords -> {keywords[:5]}")

    with open(output_path, 'wb') as f:
        pickle.dump(keyword_embeddings, f)

    print(f"\nSaved to: {output_path}")
    return keyword_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--max_keywords', type=int, default=15)

    args = parser.parse_args()
    generate_keyword_embeddings(args.results_path, args.model_name, args.output_path, args.max_keywords)
