# ood_detection.py - Out-of-Distribution Detection for Unknown Class Discovery
# Detects samples that don't belong to known classes using trained model

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class TextDataset(Dataset):
    """Simple dataset for text classification"""

    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ''
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def extract_embeddings(model, dataloader, device):
    """Extract CLS embeddings from model."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                cls_embedding = outputs.hidden_states[-1][:, 0, :]
            else:
                cls_embedding = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'last_hidden_state') else \
                outputs[0][:, 0, :]

            embeddings.extend(cls_embedding.cpu().numpy())

    return np.array(embeddings)


def compute_class_statistics(embeddings, labels, known_labels):
    """Compute mean and covariance for each known class."""
    class_means = {}
    class_embeddings = {}

    for label in known_labels:
        mask = np.array(labels) == label
        if np.sum(mask) > 0:
            class_emb = embeddings[mask]
            class_means[label] = np.mean(class_emb, axis=0)
            class_embeddings[label] = class_emb

    all_centered = []
    for label in known_labels:
        if label in class_embeddings:
            centered = class_embeddings[label] - class_means[label]
            all_centered.append(centered)

    all_centered = np.vstack(all_centered)

    try:
        cov_estimator = LedoitWolf()
        cov_estimator.fit(all_centered)
        precision_matrix = cov_estimator.precision_
    except Exception as e:
        print(f"   Warning: LedoitWolf failed ({e}), using pseudo-inverse")
        cov = np.cov(all_centered.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        precision_matrix = np.linalg.pinv(cov)

    return class_means, precision_matrix


def mahalanobis_distance(x, mean, precision):
    """Compute Mahalanobis distance."""
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, precision), diff))


def compute_mahalanobis_scores(embeddings, class_means, precision_matrix):
    """Compute minimum Mahalanobis distance to any known class."""
    scores = []
    nearest_class = []

    for emb in tqdm(embeddings, desc="Computing Mahalanobis distances"):
        min_dist = float('inf')
        best_class = None

        for label, mean in class_means.items():
            dist = mahalanobis_distance(emb, mean, precision_matrix)
            if dist < min_dist:
                min_dist = dist
                best_class = label

        scores.append(min_dist)
        nearest_class.append(best_class)

    return np.array(scores), np.array(nearest_class)


def compute_ood_scores(model, dataloader, device):
    """Compute OOD detection scores for all samples."""
    model.eval()

    msp_scores = []
    entropy_scores = []
    energy_scores = []
    predictions = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing OOD scores"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = F.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            msp_scores.extend(max_probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy_scores.extend(entropy.cpu().numpy())

            T = 1.0
            energy = -T * torch.logsumexp(logits / T, dim=-1)
            energy_scores.extend(energy.cpu().numpy())

            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return (
        np.array(msp_scores),
        np.array(entropy_scores),
        np.array(energy_scores),
        np.array(predictions),
        np.array(all_probs),
        np.array(all_logits)
    )


def find_threshold_without_gt(msp_scores, entropy_scores, energy_scores=None, mahalanobis_scores=None, method='energy'):
    """Find threshold WITHOUT using ground truth labels."""

    if method == 'mahalanobis' and mahalanobis_scores is not None:
        print(f"   Mahalanobis-based detection:")
        print(f"     Distance min={mahalanobis_scores.min():.2f}, max={mahalanobis_scores.max():.2f}")
        print(f"     Distance mean={mahalanobis_scores.mean():.2f}, std={mahalanobis_scores.std():.2f}")

        for p in [50, 60, 70, 80, 90]:
            t = np.percentile(mahalanobis_scores, p)
            n_unknown = np.sum(mahalanobis_scores > t)
            print(
                f"     {p}th percentile: {t:.2f} -> {n_unknown} unknown ({100 * n_unknown / len(mahalanobis_scores):.1f}%)")

        threshold = np.percentile(mahalanobis_scores, 70)
        print(f"     Selected: 70th percentile = {threshold:.2f}")
        return threshold, 'mahalanobis'

    elif method == 'energy' and energy_scores is not None:
        print(f"   Energy-based detection:")
        print(f"     Energy min={energy_scores.min():.4f}, max={energy_scores.max():.4f}")

        for p in [30, 40, 50, 60, 70]:
            t = np.percentile(energy_scores, p)
            n_unknown = np.sum(energy_scores > t)
            print(
                f"     {p}th percentile: {t:.4f} -> {n_unknown} unknown ({100 * n_unknown / len(energy_scores):.1f}%)")

        threshold = np.percentile(energy_scores, 30)
        print(f"     Selected: 30th percentile = {threshold:.4f}")
        return threshold, 'energy'

    else:
        threshold = np.percentile(msp_scores, 90)
        print(f"   Conservative threshold (90th percentile): {threshold:.4f}")
        return threshold, 'msp'


def detect_unknown_samples(
        model_path,
        test_csv,
        known_labels,
        train_csv=None,
        threshold=None,
        threshold_method='energy',
        batch_size=32,
        device=None,
        use_entropy_filter=True
):
    """Detect unknown samples in test set using trained model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print("OOD DETECTION - DETECTING UNKNOWN SAMPLES")
    print('=' * 70)

    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get num_labels from config.json (most reliable)
    import json
    num_labels = None

    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        if 'num_labels' in config_data:
            num_labels = config_data['num_labels']
        elif 'id2label' in config_data:
            num_labels = len(config_data['id2label'])
        elif 'label2id' in config_data:
            num_labels = len(config_data['label2id'])
        if num_labels:
            print(f"   Found {num_labels} classes in config.json")

    # Fallback to weights
    if not num_labels:
        safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
        if safetensor_files:
            try:
                from safetensors import safe_open
                with safe_open(safetensor_files[0], framework="pt") as f:
                    for key in f.keys():
                        if key.endswith('classifier.weight') or key.endswith('classifier.bias'):
                            tensor = f.get_tensor(key)
                            num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                            print(f"   Found {num_labels} classes from weights")
                            break
            except Exception as e:
                print(f"   Warning: Could not read safetensors: {e}")

    if not num_labels:
        raise ValueError(f"Could not determine num_labels for model at {model_path}")

    print(f"   Model has {num_labels} classes")

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    config.id2label = {i: str(i) for i in range(num_labels)}
    config.label2id = {str(i): i for i in range(num_labels)}

    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    print(f"📂 Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    texts = test_df['content'].tolist()
    labels = test_df['label'].tolist()

    print(f"   Total samples: {len(texts)}")
    print(f"   Known labels (for eval): {known_labels}")
    print(f"   Labels in test: {sorted(test_df['label'].unique())}")

    dataset = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n🔍 Computing OOD scores...")
    msp_scores, entropy_scores, energy_scores, predictions, all_probs, all_logits = compute_ood_scores(
        model, dataloader, device
    )

    print(f"\n📊 Score Distributions:")
    print(
        f"   MSP:     min={msp_scores.min():.4f}, max={msp_scores.max():.4f}, mean={msp_scores.mean():.4f}, std={msp_scores.std():.4f}")
    print(
        f"   Entropy: min={entropy_scores.min():.4f}, max={entropy_scores.max():.4f}, mean={entropy_scores.mean():.4f}, std={entropy_scores.std():.4f}")
    print(
        f"   Energy:  min={energy_scores.min():.4f}, max={energy_scores.max():.4f}, mean={energy_scores.mean():.4f}, std={energy_scores.std():.4f}")

    # Auto-switch to energy if MSP variance is too low
    if msp_scores.std() < 0.02 and threshold_method not in ['energy', 'mahalanobis']:
        print(f"\n   ⚠️  MSP variance too low ({msp_scores.std():.4f}) - switching to ENERGY!")
        threshold_method = 'energy'

    # Compute Mahalanobis if needed
    mahalanobis_scores = None
    if threshold_method == 'mahalanobis':
        if train_csv is None:
            print(f"\n   ⚠️  train_csv not provided, falling back to energy")
            threshold_method = 'energy'
        else:
            print(f"\n🔍 Computing Mahalanobis distances...")
            train_df = pd.read_csv(train_csv)
            train_texts = train_df['content'].tolist()
            train_labels = train_df['label'].tolist()

            train_dataset = TextDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            train_embeddings = extract_embeddings(model, train_dataloader, device)
            class_means, precision_matrix = compute_class_statistics(train_embeddings, train_labels, known_labels)

            test_embeddings = extract_embeddings(model, dataloader, device)
            mahalanobis_scores, _ = compute_mahalanobis_scores(test_embeddings, class_means, precision_matrix)

            print(
                f"\n📊 Mahalanobis: min={mahalanobis_scores.min():.2f}, max={mahalanobis_scores.max():.2f}, mean={mahalanobis_scores.mean():.2f}")

    # Find threshold
    if threshold is None:
        print(f"\n📊 Finding threshold using method: {threshold_method}")
        threshold, score_type = find_threshold_without_gt(
            msp_scores, entropy_scores, energy_scores, mahalanobis_scores, method=threshold_method
        )
    else:
        score_type = threshold_method

    # Detect unknown
    if score_type == 'mahalanobis':
        unknown_mask = mahalanobis_scores > threshold
    elif score_type == 'energy':
        unknown_mask = energy_scores > threshold
    else:
        unknown_mask = msp_scores < threshold

    unknown_indices = np.where(unknown_mask)[0]

    print(f"\n📊 Detection Results:")
    print(f"   Threshold: {threshold:.4f} ({score_type})")
    print(f"   Detected as UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")
    print(f"   Detected as KNOWN: {np.sum(~unknown_mask)} ({100 * np.mean(~unknown_mask):.1f}%)")

    # Evaluate against ground truth
    known_mask_gt = np.array([l in known_labels for l in labels])
    if np.sum(~known_mask_gt) > 0:
        tp = np.sum(unknown_mask & ~known_mask_gt)
        fp = np.sum(unknown_mask & known_mask_gt)
        fn = np.sum(~unknown_mask & ~known_mask_gt)
        tn = np.sum(~unknown_mask & known_mask_gt)

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        contamination = fp / (np.sum(unknown_mask) + 1e-10)

        print(f"\n📊 Ground Truth Evaluation:")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"   ⚠️  Contamination: {100 * contamination:.1f}%")

    return {
        'unknown_mask': unknown_mask,
        'unknown_indices': unknown_indices,
        'msp_scores': msp_scores,
        'entropy_scores': entropy_scores,
        'energy_scores': energy_scores,
        'mahalanobis_scores': mahalanobis_scores,
        'threshold': threshold,
        'score_type': score_type,
        'predictions': predictions,
        'all_probs': all_probs,
        'texts': texts,
        'labels': labels
    }


def filter_unknown_samples(test_csv, detection_results, output_csv=None):
    """Filter test CSV to only include detected unknown samples."""
    test_df = pd.read_csv(test_csv)
    unknown_mask = detection_results['unknown_mask']
    filtered_df = test_df[unknown_mask].copy()

    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(filtered_df)} unknown samples to: {output_csv}")

    return filtered_df


if __name__ == "__main__":
    print("OOD Detection Module - Clean Version")
    print("Methods: mahalanobis, energy, msp")