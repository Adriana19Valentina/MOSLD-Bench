import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class TextDataset(Dataset):

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


def compute_energy_scores(model, dataloader, device):
    """
    Compute Energy-based OOD scores for all samples.
    Energy: -logsumexp(logits)
    Known samples have LOW energy, unknown samples have HIGH energy.

    Returns:
        energy_scores: Energy scores for each sample
        predictions: predicted class indices
    """
    model.eval()

    energy_scores = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Energy scores"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Energy: -logsumexp(logits)
            energy = -torch.logsumexp(logits, dim=-1)
            energy_scores.extend(energy.cpu().numpy())

            # Predictions (for known samples)
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())

    return np.array(energy_scores), np.array(predictions)


def calibrate_threshold_on_validation(
        model,
        tokenizer,
        val_csv,
        device,
        target_tpr=0.95,
        batch_size=32
):
    """
    Calibrate Energy threshold on validation set.
    Energy: known = LOW score, unknown = HIGH score
    We want target_tpr% of known samples to be BELOW threshold.

    Returns:
        float: calibrated threshold
    """
    print(f"\n{'=' * 60}")
    print("📊 CALIBRATING ENERGY THRESHOLD ON VALIDATION SET")
    print('=' * 60)
    print(f"   Validation file: {val_csv}")
    print(f"   Target TPR: {target_tpr * 100:.0f}%")

    val_df = pd.read_csv(val_csv)
    texts = val_df['content'].tolist()
    texts = [str(t) if t is not None and str(t) != 'nan' else '' for t in texts]

    print(f"   Validation samples: {len(texts)}")

    if 'label' in val_df.columns:
        print(f"   Validation labels: {sorted(val_df['label'].unique())}")

    dataset = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    energy_scores, _ = compute_energy_scores(model, dataloader, device)

    # Calibrate threshold: target_tpr% of known should be below threshold
    threshold = np.percentile(energy_scores, target_tpr * 100)

    print(f"\n📊 Validation Energy Statistics:")
    print(f"   Min: {energy_scores.min():.4f}")
    print(f"   Max: {energy_scores.max():.4f}")
    print(f"   Mean: {energy_scores.mean():.4f}")
    print(f"   Std: {energy_scores.std():.4f}")
    print(f"   ✅ Threshold ({target_tpr * 100:.0f}th percentile): {threshold:.4f}")

    return threshold, energy_scores


def find_threshold_fallback(energy_scores):
    """
    Fallback: Find threshold WITHOUT validation set.
    Uses mean + k*std heuristic.
    """
    print(f"\n⚠️  FALLBACK: Finding threshold without validation set")
    print(f"   Using mean + 0.5*std heuristic (less accurate!)")

    threshold = energy_scores.mean() + 0.5 * energy_scores.std()
    print(f"   Energy threshold: {threshold:.4f}")

    return threshold


def evaluate_detection(unknown_mask, labels, known_labels):
    """Evaluate OOD detection against ground truth."""
    known_mask_gt = np.array([l in known_labels for l in labels])

    tp = np.sum(unknown_mask & ~known_mask_gt)  # Correctly detected unknown
    fp = np.sum(unknown_mask & known_mask_gt)   # Known incorrectly marked as unknown
    fn = np.sum(~unknown_mask & ~known_mask_gt) # Unknown missed
    tn = np.sum(~unknown_mask & known_mask_gt)  # Correctly detected known

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    contamination = fp / (np.sum(unknown_mask) + 1e-10)

    n_unknown_detected = np.sum(unknown_mask)
    n_known_detected = np.sum(~unknown_mask)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'contamination': contamination,
        'n_unknown': n_unknown_detected,
        'n_known': n_known_detected,
        'pct_unknown': 100 * n_unknown_detected / len(labels),
        'pct_known': 100 * n_known_detected / len(labels)
    }


def detect_unknown_samples(
        model_path,
        test_csv,
        known_labels,
        val_csv=None,
        target_tpr=0.95,
        batch_size=32,
        device=None
):
    """
    Detect unknown samples using Energy-based OOD detection.

    Args:
        model_path: path to trained model
        test_csv: path to test CSV
        known_labels: list of known class labels
        val_csv: path to validation CSV (for threshold calibration, RECOMMENDED)
        target_tpr: target True Positive Rate for known classes (default 0.95)
        batch_size: batch size for inference
        device: cuda/cpu

    Returns:
        dict with detection results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print("OOD DETECTION - ENERGY METHOD")
    print('=' * 70)


    tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    if not num_labels:
        safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
        if safetensor_files:
            try:
                from safetensors import safe_open
                with safe_open(safetensor_files[0], framework="pt") as f:
                    for key in f.keys():
                        if key.endswith('classifier.weight') or key.endswith('classifier.bias'):
                            tensor = f.get_tensor(key)
                            num_labels = tensor.shape[0]
                            print(f"   Found {num_labels} classes from weights")
                            break
            except Exception as e:
                print(f"   Warning: Could not read safetensors: {e}")

    if not num_labels:
        raise ValueError(f"Could not determine num_labels for model at {model_path}")

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    # =========================================================================
    # LOAD TEST DATA
    # =========================================================================
    print(f"\n📂 Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    texts = test_df['content'].tolist()
    texts = [str(t) if t is not None and str(t) != 'nan' else '' for t in texts]
    labels = test_df['label'].tolist()

    print(f"   Total samples: {len(texts)}")
    print(f"   Known labels: {known_labels}")
    print(f"   Labels in test: {sorted(test_df['label'].unique())}")

    # Ground truth distribution
    known_mask_gt = np.array([l in known_labels for l in labels])
    gt_unknown = np.sum(~known_mask_gt)
    gt_known = np.sum(known_mask_gt)
    print(f"\n📊 Ground Truth Distribution:")
    print(f"   GT Unknown: {gt_unknown:,} ({100 * gt_unknown / len(labels):.1f}%)")
    print(f"   GT Known: {gt_known:,} ({100 * gt_known / len(labels):.1f}%)")

    dataset = TextDataset(texts, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # =========================================================================
    # COMPUTE ENERGY SCORES ON TEST SET
    # =========================================================================
    print(f"\n🔍 Computing Energy scores on test set...")
    energy_scores, predictions = compute_energy_scores(model, dataloader, device)

    print(f"\n📊 Test Set Energy Distribution:")
    print(f"   Min: {energy_scores.min():.4f}")
    print(f"   Max: {energy_scores.max():.4f}")
    print(f"   Mean: {energy_scores.mean():.4f}")
    print(f"   Std: {energy_scores.std():.4f}")

    # =========================================================================
    # FIND THRESHOLD
    # =========================================================================
    if val_csv is not None and os.path.exists(val_csv):
        threshold, val_energy_scores = calibrate_threshold_on_validation(
            model, tokenizer, val_csv, device,
            target_tpr=target_tpr,
            batch_size=batch_size
        )
    else:
        threshold = find_threshold_fallback(energy_scores)
        val_energy_scores = None

    # =========================================================================
    # DETECT UNKNOWN SAMPLES
    # =========================================================================
    # Energy: known = LOW, unknown = HIGH
    # Sample is unknown if energy > threshold
    unknown_mask = energy_scores > threshold

    results = evaluate_detection(unknown_mask, labels, known_labels)

    print(f"\n{'=' * 60}")
    print("📊 DETECTION RESULTS (ENERGY)")
    print('=' * 60)
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Detected UNKNOWN: {results['n_unknown']:,} ({results['pct_unknown']:.1f}%)")
    print(f"   Detected KNOWN: {results['n_known']:,} ({results['pct_known']:.1f}%)")
    print(f"   ─────────────────────────────")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1: {results['f1']:.4f}")
    print(f"   Contamination: {results['contamination'] * 100:.1f}%")

    return {
        'unknown_mask': unknown_mask,
        'unknown_indices': np.where(unknown_mask)[0],
        'energy_scores': energy_scores,
        'threshold': threshold,
        'predictions': predictions,
        'texts': texts,
        'labels': labels,
        'metrics': results,
        'val_energy_scores': val_energy_scores
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
    print("=" * 70)
    print("OOD Detection Module - Energy Method")
    print("=" * 70)
    print("\nUsage:")
    print("  from ood_detection_energy import detect_unknown_samples")
    print("")
    print("  results = detect_unknown_samples(")
    print("      model_path='./model',")
    print("      test_csv='./test.csv',")
    print("      known_labels=[0, 1, 2, 3],")
    print("      val_csv='./val.csv',")
    print("      target_tpr=0.95")
    print("  )")
    print("")
    print("  # Access results")
    print("  print(f\"Detected {results['metrics']['n_unknown']} unknown samples\")")
    print("  print(f\"F1 Score: {results['metrics']['f1']:.4f}\")")