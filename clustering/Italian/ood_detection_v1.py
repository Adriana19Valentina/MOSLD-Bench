# ood_detection.py - Out-of-Distribution Detection for Unknown Class Discovery
# Detects samples that don't belong to known classes using trained model

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
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
        text = str(self.texts[idx])
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

            # Get CLS token embedding from last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                cls_embedding = outputs.hidden_states[-1][:, 0, :]
            else:
                # Fallback for models without hidden_states
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

    # Compute shared covariance (tied covariance - more robust)
    all_centered = []
    for label in known_labels:
        if label in class_embeddings:
            centered = class_embeddings[label] - class_means[label]
            all_centered.append(centered)

    all_centered = np.vstack(all_centered)

    # Use LedoitWolf for robust covariance estimation
    try:
        cov_estimator = LedoitWolf()
        cov_estimator.fit(all_centered)
        precision_matrix = cov_estimator.precision_
    except Exception as e:
        print(f"   Warning: LedoitWolf failed ({e}), using pseudo-inverse")
        cov = np.cov(all_centered.T)
        # Add small regularization
        cov += np.eye(cov.shape[0]) * 1e-6
        precision_matrix = np.linalg.pinv(cov)

    return class_means, precision_matrix


def mahalanobis_distance(x, mean, precision):
    """Compute Mahalanobis distance."""
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, precision), diff))


def compute_mahalanobis_scores(embeddings, class_means, precision_matrix):
    """Compute minimum Mahalanobis distance to any known class for each sample."""
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
    """
    Compute OOD detection scores for all samples.

    Returns:
        msp_scores: Maximum Softmax Probability (higher = more confident = likely known)
        entropy_scores: Entropy of predictions (higher = more uncertain = likely unknown)
        energy_scores: Energy score (lower = more confident = likely known)
        predictions: Predicted class for each sample
        all_probs: Full probability distributions
        all_logits: Raw logits (for energy calculation)
    """
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

            # Softmax probabilities
            probs = F.softmax(logits, dim=-1)

            # Maximum Softmax Probability (MSP)
            max_probs, preds = torch.max(probs, dim=-1)
            msp_scores.extend(max_probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy_scores.extend(entropy.cpu().numpy())

            # Energy score: -T * log(sum(exp(logits/T)))
            # Lower energy = more confident (known), Higher energy = less confident (unknown)
            T = 1.0  # Temperature
            energy = -T * torch.logsumexp(logits / T, dim=-1)
            energy_scores.extend(energy.cpu().numpy())

            # Store full probability distribution and logits
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


def find_threshold_without_gt(msp_scores, entropy_scores, energy_scores=None, mahalanobis_scores=None,
                              method='mahalanobis'):
    """
    Find threshold WITHOUT using ground truth labels.

    Methods:
    - 'mahalanobis': Use Mahalanobis distance (best for similar classes)
    - 'energy': Use energy score (better for overconfident models)
    - 'conservative': Use low percentile of MSP
    """

    if method == 'mahalanobis' and mahalanobis_scores is not None:
        # Mahalanobis: higher distance = more likely unknown
        print(f"   Mahalanobis-based detection:")
        print(f"     Distance min={mahalanobis_scores.min():.2f}, max={mahalanobis_scores.max():.2f}")
        print(f"     Distance mean={mahalanobis_scores.mean():.2f}, std={mahalanobis_scores.std():.2f}")

        for p in [50, 60, 70, 80, 90]:
            t = np.percentile(mahalanobis_scores, p)
            n_unknown = np.sum(mahalanobis_scores > t)
            print(
                f"     {p}th percentile: {t:.2f} -> {n_unknown} unknown ({100 * n_unknown / len(mahalanobis_scores):.1f}%)")

        # Use 70th percentile - samples with high distance are unknown
        threshold = np.percentile(mahalanobis_scores, 70)
        print(f"     Selected: 70th percentile = {threshold:.2f}")

        return threshold, 'mahalanobis'

    elif method == 'energy' and energy_scores is not None:
        # Energy-based: higher energy = more uncertain = unknown
        print(f"   Energy-based detection:")
        print(f"     Energy min={energy_scores.min():.4f}, max={energy_scores.max():.4f}")

        for p in [30, 40, 50, 60, 70]:
            t = np.percentile(energy_scores, p)
            n_unknown = np.sum(energy_scores > t)
            print(
                f"     {p}th percentile: {t:.4f} -> {n_unknown} unknown ({100 * n_unknown / len(energy_scores):.1f}%)")

        # Use 30th percentile for better recall (70% of samples as unknown)
        threshold_energy = np.percentile(energy_scores, 30)
        print(f"     Selected: 30th percentile = {threshold_energy:.4f}")

        return threshold_energy, 'energy'

    elif method == 'conservative':
        threshold = np.percentile(msp_scores, 90)
        print(f"   Conservative threshold (90th percentile): {threshold:.4f}")
        return threshold, 'msp'

    else:
        # Default: percentile-based MSP
        threshold = np.percentile(msp_scores, 85)
        return threshold, 'msp'


def detect_unknown_samples(
        model_path,
        test_csv,
        known_labels,
        train_csv=None,
        threshold=None,
        threshold_method='mahalanobis',
        batch_size=32,
        device=None,
        use_entropy_filter=True
):
    """
    Detect unknown samples in test set using trained model.

    Args:
        model_path: Path to trained model directory
        test_csv: Path to test CSV file
        known_labels: List of known class labels (for evaluation only!)
        train_csv: Path to training CSV (needed for Mahalanobis)
        threshold: Manual threshold (if None, computed automatically)
        threshold_method: 'mahalanobis', 'energy', 'conservative'
        batch_size: Batch size for inference
        device: torch device
        use_entropy_filter: Also filter by entropy (for non-mahalanobis methods)

    Returns:
        results: Dictionary containing detection results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print("OOD DETECTION - DETECTING UNKNOWN SAMPLES")
    print('=' * 70)

    # Load tokenizer
    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get num_labels - PRIORITIZE config.json (most reliable for transformers models)
    import json
    num_labels = None

    # Source 1: config.json (saved by transformers - most reliable!)
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        # Try different keys
        if 'num_labels' in config_data:
            num_labels = config_data['num_labels']
        elif 'id2label' in config_data:
            num_labels = len(config_data['id2label'])
        elif 'label2id' in config_data:
            num_labels = len(config_data['label2id'])

        if num_labels:
            print(f"   Found {num_labels} classes in config.json")

    # Source 2: Fallback to reading from weights (only if config failed)
    if not num_labels:
        safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
        if safetensor_files:
            try:
                from safetensors import safe_open
                with safe_open(safetensor_files[0], framework="pt") as f:
                    for key in f.keys():
                        # Be more specific - look for the final output layer
                        # Typical patterns: classifier.weight, classifier.bias, out_proj.weight
                        if key.endswith('classifier.weight') or key.endswith('classifier.bias'):
                            tensor = f.get_tensor(key)
                            num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                            print(f"   Found {num_labels} classes from {key} (safetensors)")
                            break
                        elif key.endswith('out_proj.weight') or key.endswith('out_proj.bias'):
                            tensor = f.get_tensor(key)
                            num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                            print(f"   Found {num_labels} classes from {key} (safetensors)")
                            break
            except Exception as e:
                print(f"   Warning: Could not read safetensors: {e}")

    if not num_labels:
        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(bin_path):
            try:
                state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
                for key, tensor in state_dict.items():
                    if key.endswith('classifier.weight') or key.endswith('classifier.bias'):
                        num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                        print(f"   Found {num_labels} classes from {key} (pytorch_model.bin)")
                        break
                    elif key.endswith('out_proj.weight') or key.endswith('out_proj.bias'):
                        num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                        print(f"   Found {num_labels} classes from {key} (pytorch_model.bin)")
                        break
            except Exception as e:
                print(f"   Warning: Could not read pytorch_model.bin: {e}")

    if not num_labels:
        raise ValueError(f"Could not determine num_labels for model at {model_path}")

    print(f"   Model has {num_labels} classes")

    # Load model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    config.id2label = {i: str(i) for i in range(num_labels)}
    config.label2id = {str(i): i for i in range(num_labels)}

    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    # Load test data
    print(f"📂 Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    texts = test_df['content'].tolist()
    labels = test_df['label'].tolist()

    print(f"   Total samples: {len(texts)}")
    print(f"   Known labels (for eval): {known_labels}")
    print(f"   Labels in test: {sorted(test_df['label'].unique())}")

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute standard OOD scores
    print(f"\n🔍 Computing OOD scores...")
    msp_scores, entropy_scores, energy_scores, predictions, all_probs, all_logits = compute_ood_scores(
        model, dataloader, device
    )

    # Print score distributions
    print(f"\n📊 Score Distributions:")
    print(
        f"   MSP:     min={msp_scores.min():.4f}, max={msp_scores.max():.4f}, mean={msp_scores.mean():.4f}, std={msp_scores.std():.4f}")
    print(
        f"   Entropy: min={entropy_scores.min():.4f}, max={entropy_scores.max():.4f}, mean={entropy_scores.mean():.4f}, std={entropy_scores.std():.4f}")
    print(
        f"   Energy:  min={energy_scores.min():.4f}, max={energy_scores.max():.4f}, mean={energy_scores.mean():.4f}, std={energy_scores.std():.4f}")

    # Auto-detect: if MSP has very low variance, switch to energy or mahalanobis
    if msp_scores.std() < 0.02 and threshold_method not in ['energy', 'mahalanobis']:
        print(f"\n   ⚠️  MSP variance too low ({msp_scores.std():.4f}) - auto-switching to MAHALANOBIS method!")
        threshold_method = 'mahalanobis'

    # Compute Mahalanobis scores if needed
    mahalanobis_scores = None
    if threshold_method == 'mahalanobis':
        if train_csv is None:
            print(f"\n   ⚠️  train_csv not provided, falling back to energy method")
            threshold_method = 'energy'
        else:
            print(f"\n🔍 Computing Mahalanobis distances...")

            # Load training data
            train_df = pd.read_csv(train_csv)
            train_texts = train_df['content'].tolist()
            train_labels = train_df['label'].tolist()

            print(f"   Training samples: {len(train_texts)}")

            # Extract embeddings for training data
            train_dataset = TextDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            print(f"   Extracting training embeddings...")
            train_embeddings = extract_embeddings(model, train_dataloader, device)

            # Compute class statistics
            print(f"   Computing class statistics...")
            class_means, precision_matrix = compute_class_statistics(
                train_embeddings, train_labels, known_labels
            )
            print(f"   Classes with statistics: {list(class_means.keys())}")

            # Extract embeddings for test data
            print(f"   Extracting test embeddings...")
            test_embeddings = extract_embeddings(model, dataloader, device)

            # Compute Mahalanobis distances
            mahalanobis_scores, nearest_classes = compute_mahalanobis_scores(
                test_embeddings, class_means, precision_matrix
            )

            print(f"\n📊 Mahalanobis Score Distribution:")
            print(f"   Min: {mahalanobis_scores.min():.2f}")
            print(f"   Max: {mahalanobis_scores.max():.2f}")
            print(f"   Mean: {mahalanobis_scores.mean():.2f}")
            print(f"   Std: {mahalanobis_scores.std():.2f}")

    # Determine threshold
    if threshold is None:
        print(f"\n📊 Finding threshold using method: {threshold_method}")
        threshold, score_type = find_threshold_without_gt(
            msp_scores, entropy_scores, energy_scores, mahalanobis_scores, method=threshold_method
        )
    else:
        print(f"\n📊 Using provided threshold: {threshold:.4f}")
        score_type = threshold_method

    # Detect unknown samples based on score type
    if score_type == 'mahalanobis':
        # Higher Mahalanobis distance = unknown
        unknown_mask = mahalanobis_scores > threshold
        print(f"\n   Using MAHALANOBIS distances for detection")
        print(f"   Samples with distance > {threshold:.2f} = UNKNOWN")
    elif score_type == 'energy':
        # Higher energy = unknown
        unknown_mask = energy_scores > threshold
        print(f"\n   Using ENERGY scores for detection")
        print(f"   Samples with energy > {threshold:.4f} = UNKNOWN")
    else:
        # Lower MSP = unknown
        unknown_mask = msp_scores < threshold
        print(f"\n   Using MSP scores for detection")

    unknown_indices = np.where(unknown_mask)[0]

    print(f"\n📊 Detection Results:")
    print(f"   Threshold: {threshold:.4f} ({score_type})")
    print(f"   Detected as UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")
    print(f"   Detected as KNOWN: {np.sum(~unknown_mask)} ({100 * np.mean(~unknown_mask):.1f}%)")

    # Evaluate against ground truth
    known_mask_gt = np.array([l in known_labels for l in labels])
    if np.sum(~known_mask_gt) > 0:
        true_positives = np.sum(unknown_mask & ~known_mask_gt)
        false_positives = np.sum(unknown_mask & known_mask_gt)
        false_negatives = np.sum(~unknown_mask & ~known_mask_gt)
        true_negatives = np.sum(~unknown_mask & known_mask_gt)

        print(f"\n📊 Ground Truth Comparison (for evaluation):")
        print(f"   True Positives (unknown→unknown): {true_positives}")
        print(f"   False Positives (known→unknown): {false_positives}")
        print(f"   False Negatives (unknown→known): {false_negatives}")
        print(f"   True Negatives (known→known): {true_negatives}")

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        print(f"\n   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")

        # Show contamination rate
        contamination = false_positives / (np.sum(unknown_mask) + 1e-10)
        print(f"\n   ⚠️  Contamination (known samples in 'unknown' pool): {100 * contamination:.1f}%")

        # Show per-class detection rates for unknown classes
        print(f"\n📊 Per-class detection rates (unknown classes):")
        for label in sorted(test_df['label'].unique()):
            if label not in known_labels:
                mask_label = np.array(labels) == label
                detected = np.sum(unknown_mask & mask_label)
                total = np.sum(mask_label)
                print(f"   Class {label}: {detected}/{total} detected ({100 * detected / total:.1f}%)")

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
    """
    Filter test CSV to only include detected unknown samples.
    """
    test_df = pd.read_csv(test_csv)

    unknown_mask = detection_results['unknown_mask']
    filtered_df = test_df[unknown_mask].copy()

    filtered_df['msp_score'] = detection_results['msp_scores'][unknown_mask]
    filtered_df['entropy_score'] = detection_results['entropy_scores'][unknown_mask]

    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(filtered_df)} unknown samples to: {output_csv}")

    return filtered_df


if __name__ == "__main__":
    print("OOD Detection Module v3 - with Mahalanobis Distance")
    print("=" * 50)
    print("\nMethods available:")
    print("  - mahalanobis: Best for semantically similar classes")
    print("  - energy: Good for overconfident models")
    print("  - conservative: Simple MSP-based")


def compute_ood_scores(model, dataloader, device):
    """
    Compute OOD detection scores for all samples.

    Returns:
        msp_scores: Maximum Softmax Probability (higher = more confident = likely known)
        entropy_scores: Entropy of predictions (higher = more uncertain = likely unknown)
        energy_scores: Energy score (lower = more confident = likely known)
        predictions: Predicted class for each sample
        all_probs: Full probability distributions
        all_logits: Raw logits (for energy calculation)
    """
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

            # Softmax probabilities
            probs = F.softmax(logits, dim=-1)

            # Maximum Softmax Probability (MSP)
            max_probs, preds = torch.max(probs, dim=-1)
            msp_scores.extend(max_probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy_scores.extend(entropy.cpu().numpy())

            # Energy score: -T * log(sum(exp(logits/T)))
            # Lower energy = more confident (known), Higher energy = less confident (unknown)
            T = 1.0  # Temperature
            energy = -T * torch.logsumexp(logits / T, dim=-1)
            energy_scores.extend(energy.cpu().numpy())

            # Store full probability distribution and logits
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


def find_threshold_without_gt(msp_scores, entropy_scores, energy_scores=None, method='energy'):
    """
    Find threshold WITHOUT using ground truth labels.

    Methods:
    - 'conservative': Use low percentile of MSP (catches more unknowns, higher recall)
    - 'entropy_based': Combine MSP and entropy
    - 'statistical': Use mean - k*std of MSP distribution
    - 'energy': Use energy score (better for overconfident models)
    """

    if method == 'energy' and energy_scores is not None:
        # Energy-based: higher energy = more uncertain = unknown
        # Use percentile to find threshold
        # Higher percentile = more samples marked as unknown = better recall

        # Try different percentiles and report
        print(f"   Energy-based detection:")
        print(f"     Energy min={energy_scores.min():.4f}, max={energy_scores.max():.4f}")

        for p in [30, 40, 50, 60, 70]:
            t = np.percentile(energy_scores, p)
            n_unknown = np.sum(energy_scores > t)
            print(
                f"     {p}th percentile: {t:.4f} -> {n_unknown} unknown ({100 * n_unknown / len(energy_scores):.1f}%)")

        # Use 30th percentile for better recall (70% of samples as unknown)
        threshold_energy = np.percentile(energy_scores, 30)
        print(f"     Selected: 30th percentile = {threshold_energy:.4f}")

        return threshold_energy, 'energy'

    elif method == 'conservative':
        # Assume most samples should be classified confidently
        # Use a high threshold to only keep very confident predictions as "known"
        threshold = np.percentile(msp_scores, 90)  # Top 10% confidence = known
        print(f"   Conservative threshold (90th percentile): {threshold:.4f}")
        return threshold, 'msp'

    elif method == 'entropy_based':
        # Combine MSP and entropy
        msp_norm = (msp_scores - msp_scores.min()) / (msp_scores.max() - msp_scores.min() + 1e-10)
        ent_norm = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min() + 1e-10)
        combined = msp_norm - 0.5 * ent_norm
        threshold = np.percentile(combined, 85)
        threshold = np.percentile(msp_scores, 85)
        print(f"   Entropy-based threshold (85th percentile): {threshold:.4f}")
        return threshold, 'msp'

    elif method == 'statistical':
        mean_msp = np.mean(msp_scores)
        std_msp = np.std(msp_scores)
        threshold = mean_msp - 2 * std_msp
        threshold = max(threshold, 0.5)
        print(f"   Statistical threshold (mean - 2*std): {threshold:.4f}")
        print(f"   MSP mean: {mean_msp:.4f}, std: {std_msp:.4f}")
        return threshold, 'msp'

    else:
        # Default: percentile-based MSP
        threshold = np.percentile(msp_scores, 85)
        return threshold, 'msp'


def detect_unknown_samples(
        model_path,
        test_csv,
        known_labels,
        threshold=None,
        threshold_method='conservative',
        batch_size=32,
        device=None,
        use_entropy_filter=True
):
    """
    Detect unknown samples in test set using trained model.

    Args:
        model_path: Path to trained model directory
        test_csv: Path to test CSV file
        known_labels: List of known class labels (for evaluation only!)
        threshold: MSP threshold (if None, will be computed automatically)
        threshold_method: Method for automatic threshold ('conservative', 'entropy_based', 'statistical')
        batch_size: Batch size for inference
        device: torch device
        use_entropy_filter: Also filter by entropy (double check)

    Returns:
        results: Dictionary containing detection results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print("OOD DETECTION - DETECTING UNKNOWN SAMPLES")
    print('=' * 70)

    # Load model and tokenizer
    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get num_labels from model weights (most reliable)
    import json
    num_labels = None

    # Source 1: Read directly from model weights
    safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    if safetensor_files:
        try:
            from safetensors import safe_open
            with safe_open(safetensor_files[0], framework="pt") as f:
                for key in f.keys():
                    if 'classifier' in key.lower() and ('weight' in key.lower() or 'bias' in key.lower()):
                        tensor = f.get_tensor(key)
                        num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                        break
        except Exception as e:
            print(f"   Warning: Could not read safetensors: {e}")

    if not num_labels:
        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(bin_path):
            try:
                state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
                for key, tensor in state_dict.items():
                    if 'classifier' in key.lower() and ('weight' in key.lower() or 'bias' in key.lower()):
                        num_labels = tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[0]
                        break
            except Exception as e:
                print(f"   Warning: Could not read pytorch_model.bin: {e}")

    # Source 2: config.json as fallback
    if not num_labels:
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            num_labels = config_data.get('num_labels')
            if not num_labels and 'id2label' in config_data:
                num_labels = len(config_data['id2label'])

    if not num_labels:
        raise ValueError(f"Could not determine num_labels for model at {model_path}")

    print(f"   Model has {num_labels} classes")

    # Load model with correct config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    config.id2label = {i: str(i) for i in range(num_labels)}
    config.label2id = {str(i): i for i in range(num_labels)}

    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    # Load test data
    print(f"📂 Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    texts = test_df['content'].tolist()
    labels = test_df['label'].tolist()

    print(f"   Total samples: {len(texts)}")
    print(f"   Known labels (for eval): {known_labels}")
    print(f"   Labels in test: {sorted(test_df['label'].unique())}")

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute OOD scores
    print(f"\n🔍 Computing OOD scores...")
    msp_scores, entropy_scores, energy_scores, predictions, all_probs, all_logits = compute_ood_scores(
        model, dataloader, device
    )

    # Print score distributions
    print(f"\n📊 Score Distributions:")
    print(
        f"   MSP:     min={msp_scores.min():.4f}, max={msp_scores.max():.4f}, mean={msp_scores.mean():.4f}, std={msp_scores.std():.4f}")
    print(
        f"   Entropy: min={entropy_scores.min():.4f}, max={entropy_scores.max():.4f}, mean={entropy_scores.mean():.4f}, std={entropy_scores.std():.4f}")
    print(
        f"   Energy:  min={energy_scores.min():.4f}, max={energy_scores.max():.4f}, mean={energy_scores.mean():.4f}, std={energy_scores.std():.4f}")

    # Auto-detect: if MSP has very low variance, switch to energy
    if msp_scores.std() < 0.02 and threshold_method != 'energy':
        print(f"\n   ⚠️  MSP variance too low ({msp_scores.std():.4f}) - auto-switching to ENERGY method!")
        threshold_method = 'energy'

    # Determine threshold (WITHOUT using ground truth!)
    if threshold is None:
        print(f"\n📊 Finding threshold using method: {threshold_method}")
        threshold, score_type = find_threshold_without_gt(msp_scores, entropy_scores, energy_scores,
                                                          method=threshold_method)
    else:
        print(f"\n📊 Using provided threshold: {threshold:.4f}")
        score_type = 'msp'

    # Detect unknown samples based on score type
    if score_type == 'energy':
        # For energy: higher = more uncertain = unknown
        unknown_mask = energy_scores > threshold
        print(f"\n   Using ENERGY scores for detection")
        print(f"   Energy threshold: {threshold:.4f}")
        print(f"   Samples with energy > threshold = UNKNOWN")
    else:
        # For MSP: lower = more uncertain = unknown
        unknown_mask_msp = msp_scores < threshold

        # Optional: Also filter by entropy (high entropy = uncertain = unknown)
        if use_entropy_filter:
            entropy_threshold = np.percentile(entropy_scores, 50)
            unknown_mask_entropy = entropy_scores > entropy_threshold

            # Combine with energy if available
            if energy_scores is not None:
                energy_threshold = np.percentile(energy_scores, 50)
                unknown_mask_energy = energy_scores > energy_threshold
                # Unknown if: low MSP OR high entropy OR high energy
                unknown_mask = unknown_mask_msp | unknown_mask_entropy | unknown_mask_energy
                print(f"\n   Combined detection (MSP + Entropy + Energy):")
                print(f"   MSP threshold: {threshold:.4f}")
                print(f"   Entropy threshold: {entropy_threshold:.4f}")
                print(f"   Energy threshold: {energy_threshold:.4f}")
                print(f"   Unknown by MSP: {np.sum(unknown_mask_msp)}")
                print(f"   Unknown by entropy: {np.sum(unknown_mask_entropy)}")
                print(f"   Unknown by energy: {np.sum(unknown_mask_energy)}")
            else:
                unknown_mask = unknown_mask_msp | unknown_mask_entropy
                print(f"   Entropy threshold (median): {entropy_threshold:.4f}")
                print(f"   Unknown by MSP alone: {np.sum(unknown_mask_msp)}")
                print(f"   Unknown by entropy alone: {np.sum(unknown_mask_entropy)}")

            print(f"   Final unknown (combined): {np.sum(unknown_mask)}")
        else:
            unknown_mask = unknown_mask_msp

    unknown_indices = np.where(unknown_mask)[0]

    print(f"\n📊 Detection Results:")
    print(f"   Threshold: {threshold:.4f} ({score_type})")
    print(f"   Detected as UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")
    print(f"   Detected as KNOWN: {np.sum(~unknown_mask)} ({100 * np.mean(~unknown_mask):.1f}%)")

    # Evaluate against ground truth (for reporting only, not used in threshold!)
    known_mask_gt = np.array([l in known_labels for l in labels])
    if np.sum(~known_mask_gt) > 0:
        true_positives = np.sum(unknown_mask & ~known_mask_gt)
        false_positives = np.sum(unknown_mask & known_mask_gt)
        false_negatives = np.sum(~unknown_mask & ~known_mask_gt)
        true_negatives = np.sum(~unknown_mask & known_mask_gt)

        print(f"\n📊 Ground Truth Comparison (for evaluation):")
        print(f"   True Positives (unknown→unknown): {true_positives}")
        print(f"   False Positives (known→unknown): {false_positives}")
        print(f"   False Negatives (unknown→known): {false_negatives}")
        print(f"   True Negatives (known→known): {true_negatives}")

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        print(f"\n   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")

        # Show contamination rate
        contamination = false_positives / (np.sum(unknown_mask) + 1e-10)
        print(f"\n   ⚠️  Contamination (known samples in 'unknown' pool): {100 * contamination:.1f}%")

    # MSP distribution stats
    print(f"\n📊 MSP Score Distribution:")
    print(f"   Min: {msp_scores.min():.4f}")
    print(f"   Max: {msp_scores.max():.4f}")
    print(f"   Mean: {msp_scores.mean():.4f}")
    print(f"   Std: {msp_scores.std():.4f}")
    print(f"   Median: {np.median(msp_scores):.4f}")

    # Percentiles
    print(f"\n   Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"     {p}th: {np.percentile(msp_scores, p):.4f}")

    return {
        'unknown_mask': unknown_mask,
        'unknown_indices': unknown_indices,
        'msp_scores': msp_scores,
        'entropy_scores': entropy_scores,
        'energy_scores': energy_scores,
        'threshold': threshold,
        'score_type': score_type if 'score_type' in dir() else 'msp',
        'predictions': predictions,
        'all_probs': all_probs,
        'texts': texts,
        'labels': labels
    }


def filter_unknown_samples(test_csv, detection_results, output_csv=None):
    """
    Filter test CSV to only include detected unknown samples.
    """
    test_df = pd.read_csv(test_csv)

    unknown_mask = detection_results['unknown_mask']
    filtered_df = test_df[unknown_mask].copy()

    filtered_df['msp_score'] = detection_results['msp_scores'][unknown_mask]
    filtered_df['entropy_score'] = detection_results['entropy_scores'][unknown_mask]

    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(filtered_df)} unknown samples to: {output_csv}")

    return filtered_df


if __name__ == "__main__":
    print("OOD Detection Module v2")
    print("=" * 50)
    print("\nImprovements:")
    print("  - Threshold computed WITHOUT ground truth")
    print("  - Multiple threshold methods: conservative, entropy_based, statistical")
    print("  - Optional entropy-based double filtering")
    print("  - Contamination rate reporting")