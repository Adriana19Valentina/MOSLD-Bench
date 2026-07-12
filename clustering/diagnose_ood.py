# diagnose_ood.py - Analyze why Tech class is not detected as OOD

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
import sys
import os

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
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def extract_embeddings(model, dataloader, device):
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
                cls_embedding = outputs[0][:, 0, :]
            embeddings.extend(cls_embedding.cpu().numpy())
    return np.array(embeddings)


print("=" * 70)
print("DIAGNOSTIC: Analyzing Mahalanobis scores per class")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(BASELINE_MODEL_DIR)
model = model.to(device)
model.eval()

# Load train data for class statistics
print("\n📂 Loading training data...")
train_df = pd.read_csv(TRAIN_CSV)
train_texts = train_df['content'].tolist()
train_labels = train_df['label'].tolist()

train_dataset = TextDataset(train_texts, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_embeddings = extract_embeddings(model, train_loader, device)

# Compute class statistics
print("\n📊 Computing class statistics...")
class_means = {}
class_embeddings = {}
for label in BASELINE_LABELS:
    mask = np.array(train_labels) == label
    if np.sum(mask) > 0:
        class_emb = train_embeddings[mask]
        class_means[label] = np.mean(class_emb, axis=0)
        class_embeddings[label] = class_emb

all_centered = []
for label in BASELINE_LABELS:
    if label in class_embeddings:
        centered = class_embeddings[label] - class_means[label]
        all_centered.append(centered)
all_centered = np.vstack(all_centered)

cov_estimator = LedoitWolf()
cov_estimator.fit(all_centered)
precision_matrix = cov_estimator.precision_

# Load test data
print("\n📂 Loading test data...")
test_df = pd.read_csv(TEST_1_CSV)
test_texts = test_df['content'].tolist()
test_labels = test_df['label'].tolist()

test_dataset = TextDataset(test_texts, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_embeddings = extract_embeddings(model, test_loader, device)

# Compute Mahalanobis distances
print("\n📊 Computing Mahalanobis distances...")
mahalanobis_scores = []
nearest_classes = []

for emb in tqdm(test_embeddings, desc="Computing distances"):
    min_dist = float('inf')
    best_class = None
    for label, mean in class_means.items():
        diff = emb - mean
        dist = np.sqrt(np.dot(np.dot(diff, precision_matrix), diff))
        if dist < min_dist:
            min_dist = dist
            best_class = label
    mahalanobis_scores.append(min_dist)
    nearest_classes.append(best_class)

mahalanobis_scores = np.array(mahalanobis_scores)
nearest_classes = np.array(nearest_classes)
test_labels = np.array(test_labels)

# Analyze per class
print("\n" + "=" * 70)
print("MAHALANOBIS DISTANCE DISTRIBUTION PER CLASS")
print("=" * 70)

class_names = {1: 'Culture', 2: 'Finance', 3: 'Politics', 4: 'Science', 5: 'Sport', 6: 'Tech'}

for label in sorted(set(test_labels)):
    mask = test_labels == label
    scores = mahalanobis_scores[mask]
    nearest = nearest_classes[mask]

    name = class_names.get(label, f'Class {label}')
    is_known = label in BASELINE_LABELS
    status = "KNOWN" if is_known else "NEW (should be OOD)"

    print(f"\n📊 {name} (label={label}) - {status}")
    print(f"   Samples: {len(scores)}")
    print(f"   Mahalanobis: min={scores.min():.2f}, max={scores.max():.2f}")
    print(f"   Mahalanobis: mean={scores.mean():.2f}, std={scores.std():.2f}")
    print(
        f"   Percentiles: 25th={np.percentile(scores, 25):.2f}, 50th={np.percentile(scores, 50):.2f}, 75th={np.percentile(scores, 75):.2f}")

    # Where are they being mapped?
    nearest_counts = pd.Series(nearest).value_counts()
    print(f"   Nearest known class distribution:")
    for cls, count in nearest_counts.items():
        cls_name = class_names.get(cls, f'Class {cls}')
        print(f"      → {cls_name}: {count} ({100 * count / len(nearest):.1f}%)")

# Threshold analysis
print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

for percentile in [30, 40, 50, 60, 70]:
    threshold = np.percentile(mahalanobis_scores, percentile)
    print(f"\n📊 Threshold at {percentile}th percentile: {threshold:.2f}")

    for label in sorted(set(test_labels)):
        mask = test_labels == label
        scores = mahalanobis_scores[mask]
        detected_as_ood = np.sum(scores > threshold)
        total = len(scores)
        name = class_names.get(label, f'Class {label}')
        print(f"   {name}: {detected_as_ood}/{total} detected as OOD ({100 * detected_as_ood / total:.1f}%)")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Check if Tech overlaps with Science
tech_scores = mahalanobis_scores[test_labels == 6]
science_train_scores = []
for emb in train_embeddings[np.array(train_labels) == 4]:
    min_dist = float('inf')
    for label, mean in class_means.items():
        diff = emb - mean
        dist = np.sqrt(np.dot(np.dot(diff, precision_matrix), diff))
        if dist < min_dist:
            min_dist = dist
    science_train_scores.append(min_dist)
science_train_scores = np.array(science_train_scores)

print(f"\n📊 Science (train) Mahalanobis: mean={science_train_scores.mean():.2f}, std={science_train_scores.std():.2f}")
print(f"📊 Tech (test) Mahalanobis: mean={tech_scores.mean():.2f}, std={tech_scores.std():.2f}")

overlap = np.mean(tech_scores < np.percentile(science_train_scores, 95))
print(f"\n⚠️  {100 * overlap:.1f}% of Tech samples fall within Science's 95th percentile range")
print("   This explains why Tech is hard to detect as OOD!")

if overlap > 0.5:
    print("\n💡 SUGGESTION: Mahalanobis alone won't work well for Tech vs Science")
    print("   Consider using a hybrid approach or class-specific thresholds")