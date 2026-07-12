#!/usr/bin/env python3
# diagnose_ood.py - Investigate why OOD detection is not working well
#
# This script analyzes:
# 1. Energy score distribution for known vs new classes
# 2. Whether the baseline model can distinguish classes
# 3. Embedding space visualization

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

# Configuration - adjust these paths
MODEL_PATH = "./russian_cl_outputs/model_baseline"
DATA_PATH = "/home/alin/Desktop/ContinualLearning/datasets/Russian/Russian_balanced_v2/test_1.csv"
MODEL_NAME = "DeepPavlov/rubert-base-cased"

# Class names for Russian dataset
CLASS_NAMES = {
    0: 'conflicts',
    1: 'economy',
    2: 'politics',
    3: 'science',
    4: 'health',
    5: 'society',
    6: 'sports',
    7: 'culture',
    8: 'climate',
    9: 'travel'
}

BASELINE_LABELS = [0, 1, 2, 3]
NEW_LABELS_T1 = [4, 5]


def load_model_and_tokenizer(model_path):
    """Load the baseline model."""
    print(f"📥 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"   Classes: {model.config.num_labels}")
    return model, tokenizer


def compute_energy_scores(model, tokenizer, texts, batch_size=32):
    """Compute energy scores for texts."""
    device = next(model.parameters()).device
    all_energies = []
    all_logits = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Energy score: -logsumexp(logits)
            energy = -torch.logsumexp(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            max_probs = probs.max(dim=1).values

            all_energies.extend(energy.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.extend(max_probs.cpu().numpy())

    return np.array(all_energies), np.vstack(all_logits), np.array(all_probs)


def get_embeddings(model_name, tokenizer, texts, batch_size=32):
    """Get embeddings from base model."""
    print(f"📥 Loading embedding model: {model_name}")
    embed_model = AutoModel.from_pretrained(model_name)
    embed_model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = embed_model.to(device)

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = embed_model(**inputs)
            # Use CLS token
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def analyze_ood_detection():
    """Main analysis function."""
    print("=" * 70)
    print("OOD DETECTION DIAGNOSTIC")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   Total samples: {len(df)}")

    # Show distribution
    print(f"\n📊 Label distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        name = CLASS_NAMES.get(label, f"class_{label}")
        role = "BASELINE" if label in BASELINE_LABELS else "NEW"
        print(f"   {label} ({name:10s}): {count:5d} [{role}]")

    # Separate known and new
    known_df = df[df['label'].isin(BASELINE_LABELS)]
    new_df = df[df['label'].isin(NEW_LABELS_T1)]

    print(f"\n   Known samples: {len(known_df)}")
    print(f"   New samples: {len(new_df)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Sample for faster analysis
    sample_size = min(2000, len(df))
    known_sample = known_df.sample(n=min(1000, len(known_df)), random_state=42)
    new_sample = new_df.sample(n=min(1000, len(new_df)), random_state=42)

    print(f"\n📊 Analyzing {len(known_sample)} known + {len(new_sample)} new samples...")

    # Compute energy scores
    print("\n" + "=" * 70)
    print("ENERGY SCORE ANALYSIS")
    print("=" * 70)

    print("\n🔄 Computing energy scores for KNOWN samples...")
    known_energies, known_logits, known_probs = compute_energy_scores(
        model, tokenizer, known_sample['content'].tolist()
    )

    print("🔄 Computing energy scores for NEW samples...")
    new_energies, new_logits, new_probs = compute_energy_scores(
        model, tokenizer, new_sample['content'].tolist()
    )

    # Statistics
    print(f"\n📊 Energy Score Statistics:")
    print(f"   KNOWN: mean={known_energies.mean():.4f}, std={known_energies.std():.4f}")
    print(f"          min={known_energies.min():.4f}, max={known_energies.max():.4f}")
    print(f"   NEW:   mean={new_energies.mean():.4f}, std={new_energies.std():.4f}")
    print(f"          min={new_energies.min():.4f}, max={new_energies.max():.4f}")

    # Overlap analysis
    print(f"\n📊 Distribution Overlap:")
    known_p25, known_p75 = np.percentile(known_energies, [25, 75])
    new_p25, new_p75 = np.percentile(new_energies, [25, 75])
    print(f"   KNOWN IQR: [{known_p25:.4f}, {known_p75:.4f}]")
    print(f"   NEW IQR:   [{new_p25:.4f}, {new_p75:.4f}]")

    # Check separability
    all_energies = np.concatenate([known_energies, new_energies])
    all_labels = np.array([0] * len(known_energies) + [1] * len(new_energies))

    # Try different thresholds
    print(f"\n📊 Threshold Analysis (higher energy = more likely OOD):")
    print(f"   {'Threshold':>10} | {'Known→Known':>12} | {'New→OOD':>12} | {'Accuracy':>10}")
    print("   " + "-" * 55)

    best_acc = 0
    best_threshold = None

    for percentile in [50, 60, 70, 80, 90, 95]:
        threshold = np.percentile(known_energies, percentile)
        known_correct = (known_energies < threshold).sum() / len(known_energies)
        new_correct = (new_energies >= threshold).sum() / len(new_energies)
        accuracy = (known_correct * len(known_energies) + new_correct * len(new_energies)) / (
                    len(known_energies) + len(new_energies))

        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold

        print(f"   {threshold:>10.4f} | {known_correct:>11.1%} | {new_correct:>11.1%} | {accuracy:>9.1%}")

    print(f"\n   Best threshold: {best_threshold:.4f} (accuracy: {best_acc:.1%})")

    # Max probability analysis
    print(f"\n📊 Max Probability (Confidence) Statistics:")
    print(f"   KNOWN: mean={known_probs.mean():.4f}, std={known_probs.std():.4f}")
    print(f"   NEW:   mean={new_probs.mean():.4f}, std={new_probs.std():.4f}")

    # Prediction analysis for NEW samples
    print("\n" + "=" * 70)
    print("PREDICTION ANALYSIS FOR NEW CLASSES")
    print("=" * 70)

    new_preds = np.argmax(new_logits, axis=1)
    pred_counts = Counter(new_preds)

    print(f"\n📊 Where do NEW samples get predicted?")
    for pred_class, count in sorted(pred_counts.items()):
        pct = 100 * count / len(new_preds)
        name = CLASS_NAMES.get(pred_class, f"class_{pred_class}")
        print(f"   Predicted as {pred_class} ({name:10s}): {count:4d} ({pct:5.1f}%)")

    # Per new class analysis
    print(f"\n📊 Breakdown by actual NEW class:")
    for new_label in NEW_LABELS_T1:
        mask = new_sample['label'].values == new_label
        if mask.sum() > 0:
            class_preds = new_preds[mask]
            class_energies = new_energies[mask]

            name = CLASS_NAMES.get(new_label, f"class_{new_label}")
            print(f"\n   {new_label} ({name}):")
            print(f"      Energy: mean={class_energies.mean():.4f}, std={class_energies.std():.4f}")

            pred_dist = Counter(class_preds)
            for pred, cnt in sorted(pred_dist.items(), key=lambda x: -x[1])[:3]:
                pred_name = CLASS_NAMES.get(pred, f"class_{pred}")
                print(f"      → Predicted as {pred} ({pred_name}): {cnt} ({100 * cnt / len(class_preds):.1f}%)")

    # Semantic similarity between classes
    print("\n" + "=" * 70)
    print("SEMANTIC ANALYSIS - WHY CLASSES MIGHT BE CONFUSED")
    print("=" * 70)

    print(f"\n📊 Class descriptions:")
    print(f"   BASELINE:")
    for l in BASELINE_LABELS:
        print(f"      {l}: {CLASS_NAMES[l]}")
    print(f"   NEW (T1):")
    for l in NEW_LABELS_T1:
        print(f"      {l}: {CLASS_NAMES[l]}")

    print(f"\n⚠️  Potential confusion pairs:")
    print(f"   - 'health' (4) might overlap with 'science' (3)")
    print(f"   - 'society' (5) might overlap with 'politics' (2) or 'economy' (1)")

    # Embedding visualization
    print("\n" + "=" * 70)
    print("EMBEDDING SPACE VISUALIZATION")
    print("=" * 70)

    # Get embeddings
    combined_sample = pd.concat([known_sample, new_sample])
    print(f"\n🔄 Computing embeddings for {len(combined_sample)} samples...")

    embeddings = get_embeddings(
        MODEL_NAME, tokenizer,
        combined_sample['content'].tolist()
    )

    print("🔄 Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))

    labels = combined_sample['label'].values
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for label in sorted(combined_sample['label'].unique()):
        mask = labels == label
        name = CLASS_NAMES.get(label, f"class_{label}")
        role = "★ " if label in NEW_LABELS_T1 else ""
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[label]],
            label=f"{role}{label}: {name}",
            alpha=0.6,
            s=20
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE of Russian News Classes\n(★ = NEW classes to discover)')
    plt.tight_layout()

    output_path = './russian_cl_outputs/tsne_diagnosis.png'
    os.makedirs('./russian_cl_outputs', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    plt.close()

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    energy_gap = new_energies.mean() - known_energies.mean()

    print(f"\n📊 Key Findings:")
    print(f"   Energy gap (NEW - KNOWN): {energy_gap:.4f}")

    if abs(energy_gap) < 0.5:
        print(f"   ⚠️  PROBLEM: Energy gap is too small - classes are not separable!")
        print(f"\n💡 Recommendations:")
        print(f"   1. The baseline model treats NEW classes similarly to KNOWN classes")
        print(f"   2. This suggests high semantic overlap between classes")
        print(f"   3. Options:")
        print(f"      a) Use a different OOD method (e.g., Mahalanobis distance)")
        print(f"      b) Train baseline model longer for better discrimination")
        print(f"      c) Use class names that are more semantically distinct")
        print(f"      d) Accept that this dataset is inherently difficult for OOD")
    else:
        print(f"   ✅ Energy gap is reasonable")

    print(f"\n📊 Best achievable OOD accuracy: {best_acc:.1%}")
    if best_acc < 0.7:
        print(f"   ⚠️  This is below 70% - OOD detection will struggle")

    return {
        'energy_gap': energy_gap,
        'best_accuracy': best_acc,
        'best_threshold': best_threshold,
        'known_energy_mean': known_energies.mean(),
        'new_energy_mean': new_energies.mean()
    }


if __name__ == '__main__':
    results = analyze_ood_detection()