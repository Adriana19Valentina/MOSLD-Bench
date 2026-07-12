# ood_detection.py - Out-of-Distribution Detection for Unknown Class Discovery
# Detects samples that don't belong to known classes using trained model

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm
import os
import sys

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


def compute_ood_scores(model, dataloader, device):
    """
    Compute OOD detection scores for all samples.

    Returns:
        msp_scores: Maximum Softmax Probability (higher = more confident = likely known)
        entropy_scores: Entropy of predictions (higher = more uncertain = likely unknown)
        predictions: Predicted class for each sample
        all_probs: Full probability distributions
    """
    model.eval()

    msp_scores = []
    entropy_scores = []
    predictions = []
    all_probs = []

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

            # Store full probability distribution
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(msp_scores),
        np.array(entropy_scores),
        np.array(predictions),
        np.array(all_probs)
    )


def find_optimal_threshold(known_scores, unknown_scores, method='f1'):
    """
    Find optimal threshold to separate known from unknown samples.

    Args:
        known_scores: OOD scores for known class samples (should be high for MSP, low for entropy)
        unknown_scores: OOD scores for unknown class samples
        method: 'f1' for best F1, 'percentile' for percentile-based

    Returns:
        optimal_threshold: Best threshold value
        metrics: Dictionary with performance metrics
    """
    # Create labels: 0 = known, 1 = unknown
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])

    if method == 'f1':
        # For MSP: unknown has LOWER scores, so we need to flip
        # precision_recall_curve expects higher scores for positive class
        precision, recall, thresholds = precision_recall_curve(y_true, -scores)

        # Compute F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = -thresholds[best_idx] if best_idx < len(thresholds) else -thresholds[-1]

        metrics = {
            'precision': precision[best_idx],
            'recall': recall[best_idx],
            'f1': f1_scores[best_idx],
            'auc_roc': roc_auc_score(y_true, -scores)
        }
    else:
        # Percentile-based: use 5th percentile of known scores
        optimal_threshold = np.percentile(known_scores, 5)

        # Compute metrics at this threshold
        pred_unknown = scores < optimal_threshold
        tp = np.sum(pred_unknown & (y_true == 1))
        fp = np.sum(pred_unknown & (y_true == 0))
        fn = np.sum(~pred_unknown & (y_true == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': roc_auc_score(y_true, -scores)
        }

    return optimal_threshold, metrics


def detect_unknown_samples(
        model_path,
        test_csv,
        known_labels,
        threshold=None,
        threshold_percentile=95,
        batch_size=32,
        device=None
):
    """
    Detect unknown samples in test set using trained model.

    Args:
        model_path: Path to trained model directory
        test_csv: Path to test CSV file
        known_labels: List of known class labels
        threshold: MSP threshold (if None, will be computed automatically)
        threshold_percentile: Percentile for automatic threshold (default 95)
        batch_size: Batch size for inference
        device: torch device

    Returns:
        results: Dictionary containing:
            - unknown_mask: Boolean array, True for detected unknown samples
            - unknown_indices: Indices of unknown samples
            - msp_scores: MSP scores for all samples
            - entropy_scores: Entropy scores for all samples
            - threshold: Used threshold
            - predictions: Model predictions for all samples
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print("OOD DETECTION - DETECTING UNKNOWN SAMPLES")
    print('=' * 70)

    # Load model and tokenizer
    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get num_labels - MUST match the actual saved weights
    import json
    import glob
    num_labels = None

    # Source 1: Read directly from model weights (most reliable!)
    # Try safetensors first
    safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    if safetensor_files:
        try:
            from safetensors import safe_open
            with safe_open(safetensor_files[0], framework="pt") as f:
                for key in f.keys():
                    if 'classifier' in key.lower() and ('weight' in key.lower() or 'bias' in key.lower()):
                        tensor = f.get_tensor(key)
                        # For weight: shape is [num_labels, hidden_size]
                        # For bias: shape is [num_labels]
                        if len(tensor.shape) == 1:
                            num_labels = tensor.shape[0]
                        else:
                            num_labels = tensor.shape[0]
                        print(f"   Found {num_labels} classes from classifier weights (safetensors)")
                        break
        except Exception as e:
            print(f"   Could not read safetensors: {e}")

    # Try pytorch_model.bin
    if not num_labels:
        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(bin_path):
            try:
                state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
                for key, tensor in state_dict.items():
                    if 'classifier' in key.lower() and ('weight' in key.lower() or 'bias' in key.lower()):
                        if len(tensor.shape) == 1:
                            num_labels = tensor.shape[0]
                        else:
                            num_labels = tensor.shape[0]
                        print(f"   Found {num_labels} classes from classifier weights (pytorch_model.bin)")
                        break
            except Exception as e:
                print(f"   Could not read pytorch_model.bin: {e}")

    # Source 2: config.json as fallback
    if not num_labels:
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            num_labels = config_data.get('num_labels')
            if not num_labels and 'id2label' in config_data:
                num_labels = len(config_data['id2label'])
            if not num_labels and 'label2id' in config_data:
                num_labels = len(config_data['label2id'])
            if num_labels:
                print(f"   Found {num_labels} classes in config.json")

    if not num_labels:
        raise ValueError(f"Could not determine num_labels for model at {model_path}")

    # Load model with correct num_labels from weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    # Also update id2label and label2id to match
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
    print(f"   Known labels: {known_labels}")
    print(f"   Labels in test: {sorted(test_df['label'].unique())}")

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute OOD scores
    print(f"\n🔍 Computing OOD scores...")
    msp_scores, entropy_scores, predictions, all_probs = compute_ood_scores(
        model, dataloader, device
    )

    # Determine threshold
    if threshold is None:
        # Split by ground truth known/unknown for threshold tuning
        known_mask_gt = np.array([l in known_labels for l in labels])

        if np.sum(~known_mask_gt) > 0:  # If we have unknown samples (for validation)
            print(f"\n📊 Finding optimal threshold using ground truth...")
            known_msp = msp_scores[known_mask_gt]
            unknown_msp = msp_scores[~known_mask_gt]

            threshold, metrics = find_optimal_threshold(known_msp, unknown_msp, method='f1')

            print(f"   Optimal threshold: {threshold:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1: {metrics['f1']:.4f}")
            print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        else:
            # Use percentile-based threshold on known samples
            print(f"\n📊 Using percentile-based threshold ({threshold_percentile}th percentile)...")
            threshold = np.percentile(msp_scores, 100 - threshold_percentile)
            print(f"   Threshold: {threshold:.4f}")

    # Detect unknown samples (MSP below threshold = unknown)
    unknown_mask = msp_scores < threshold
    unknown_indices = np.where(unknown_mask)[0]

    print(f"\n📊 Detection Results:")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Detected as UNKNOWN: {np.sum(unknown_mask)} ({100 * np.mean(unknown_mask):.1f}%)")
    print(f"   Detected as KNOWN: {np.sum(~unknown_mask)} ({100 * np.mean(~unknown_mask):.1f}%)")

    # If we have ground truth, show accuracy
    known_mask_gt = np.array([l in known_labels for l in labels])
    if np.sum(~known_mask_gt) > 0:
        # True unknown detected as unknown
        true_positives = np.sum(unknown_mask & ~known_mask_gt)
        # Known detected as unknown (false positives)
        false_positives = np.sum(unknown_mask & known_mask_gt)
        # True unknown detected as known (false negatives)
        false_negatives = np.sum(~unknown_mask & ~known_mask_gt)
        # Known detected as known (true negatives)
        true_negatives = np.sum(~unknown_mask & known_mask_gt)

        print(f"\n📊 Ground Truth Comparison:")
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

    # MSP distribution stats
    print(f"\n📊 MSP Score Distribution:")
    print(f"   Min: {msp_scores.min():.4f}")
    print(f"   Max: {msp_scores.max():.4f}")
    print(f"   Mean: {msp_scores.mean():.4f}")
    print(f"   Std: {msp_scores.std():.4f}")

    if np.sum(known_mask_gt) > 0 and np.sum(~known_mask_gt) > 0:
        print(
            f"\n   Known samples MSP: mean={msp_scores[known_mask_gt].mean():.4f}, std={msp_scores[known_mask_gt].std():.4f}")
        print(
            f"   Unknown samples MSP: mean={msp_scores[~known_mask_gt].mean():.4f}, std={msp_scores[~known_mask_gt].std():.4f}")

    return {
        'unknown_mask': unknown_mask,
        'unknown_indices': unknown_indices,
        'msp_scores': msp_scores,
        'entropy_scores': entropy_scores,
        'threshold': threshold,
        'predictions': predictions,
        'all_probs': all_probs,
        'texts': texts,
        'labels': labels
    }


def filter_unknown_samples(test_csv, detection_results, output_csv=None):
    """
    Filter test CSV to only include detected unknown samples.

    Args:
        test_csv: Path to original test CSV
        detection_results: Results from detect_unknown_samples()
        output_csv: Path to save filtered CSV (optional)

    Returns:
        filtered_df: DataFrame with only unknown samples
    """
    test_df = pd.read_csv(test_csv)

    unknown_mask = detection_results['unknown_mask']
    filtered_df = test_df[unknown_mask].copy()

    # Add OOD scores
    filtered_df['msp_score'] = detection_results['msp_scores'][unknown_mask]
    filtered_df['entropy_score'] = detection_results['entropy_scores'][unknown_mask]

    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(filtered_df)} unknown samples to: {output_csv}")

    return filtered_df


if __name__ == "__main__":
    # Example usage
    print("OOD Detection Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from ood_detection import detect_unknown_samples, filter_unknown_samples")
    print("\n  # Detect unknown samples")
    print("  results = detect_unknown_samples(")
    print("      model_path='./model_baseline',")
    print("      test_csv='./test_1.csv',")
    print("      known_labels=[1, 2, 3, 4]")
    print("  )")
    print("\n  # Filter to only unknown samples")
    print("  unknown_df = filter_unknown_samples('./test_1.csv', results)")