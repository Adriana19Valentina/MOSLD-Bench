# evaluate_cl_contrastive_manual.py - Evaluare cu MAPARE MANUALĂ

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

print("=" * 70)
print("EVALUARE CONTRASTIVE CU MAPARE MANUALĂ")
print("=" * 70)

# ============================================================================
# MAPARE MANUALĂ (ground truth)
# ============================================================================

# Bazat pe purity din clustering:
# Cluster 0: Athlete (99.34% purity) → Class 3
# Cluster 1: Album (100% purity) → Class 11
# Cluster 2: Edu (99.78% purity) → Class 1

PSEUDO_TO_REAL_MANUAL = {
    13: 3,  # Cluster 0 → Athlete
    14: 11,  # Cluster 1 → Album
    15: 1  # Cluster 2 → Edu
}

print("\n🔧 Mapare MANUALĂ (bazată pe purity clustering):")
print(f"  13 → 3 (Athlete)")
print(f"  14 → 11 (Album)")
print(f"  15 → 1 (EducationalInstitution)")

# Setează global pentru evaluate_accuracy
PSEUDO_TO_REAL = PSEUDO_TO_REAL_MANUAL


# ============================================================================
# FUNCȚIE EVALUARE
# ============================================================================

def evaluate_accuracy(model_path, texts, labels, label2id, remap=True):
    """Evaluează accuracy pe un set de date."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    predictions = []
    confidences = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors='pt')

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        max_conf, preds = probs.max(dim=-1)

        predictions.extend(preds.cpu().numpy())
        confidences.extend(max_conf.cpu().numpy())

    id2label = {v: int(k) for k, v in label2id.items()}
    pred_labels = [id2label.get(p, -1) for p in predictions]

    if remap:
        pred_labels = [PSEUDO_TO_REAL.get(p, p) for p in pred_labels]

    correct = sum([p == t for p, t in zip(pred_labels, labels)])
    accuracy = correct / len(labels) if len(labels) > 0 else 0.0

    return accuracy, np.array(confidences), pred_labels


# ============================================================================
# ÎNCARCĂ DATE
# ============================================================================

val_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/val.csv")
test_1_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_1.csv")

label2id_baseline = {"0": 0, "2": 1, "7": 2, "12": 3}
label2id_cl = {"0": 0, "2": 1, "7": 2, "12": 3, "13": 4, "14": 5, "15": 6}

# ============================================================================
# 1. CATASTROPHIC FORGETTING CHECK
# ============================================================================
print("\n" + "=" * 70)
print("1. CATASTROPHIC FORGETTING CHECK")
print("=" * 70)

acc_baseline, _, _ = evaluate_accuracy(
    "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final",
    val_df['content'].tolist(),
    val_df['label'].tolist(),
    label2id_baseline,
    remap=False
)

acc_cl, _, _ = evaluate_accuracy(
    "./ckpt_cl_t1_contrastive/final",
    val_df['content'].tolist(),
    val_df['label'].tolist(),
    label2id_cl,
    remap=True
)

drop = acc_baseline - acc_cl

print(f"\nValidation (clase 0,2,7,12):")
print(f"  Model baseline:      {acc_baseline:.4f} ({acc_baseline * 100:.2f}%)")
print(f"  Model Contrastive:   {acc_cl:.4f} ({acc_cl * 100:.2f}%)")
print(f"  Drop:                {drop:.4f} ({drop * 100:.2f}%)")

if drop < 0.03:
    print("  ✓ Forgetting minim (<3%)")
elif drop < 0.05:
    print("  ⚠ Forgetting moderat (3-5%)")
else:
    print("  ✗ Catastrophic forgetting (>5%)")

# ============================================================================
# 2. ACCURACY PE CLASE NOI (1, 3, 11)
# ============================================================================
print("\n" + "=" * 70)
print("2. PERFORMANȚĂ PE CLASE NOI DIN TEST_1")
print("=" * 70)

unknown_t1 = test_1_df[test_1_df['label'].isin([1, 3, 11])]

acc_new, _, preds_new = evaluate_accuracy(
    "./ckpt_cl_t1_contrastive/final",
    unknown_t1['content'].tolist(),
    unknown_t1['label'].tolist(),
    label2id_cl,
    remap=True
)

print(f"\nClase noi (1, 3, 11):")
print(f"  Overall accuracy: {acc_new:.4f} ({acc_new * 100:.2f}%)")

# Per-class accuracy
print(f"\n  Per-class:")
class_names = {1: 'EducationalInstitution', 3: 'Athlete', 11: 'Album'}

for cls in [1, 3, 11]:
    cls_df = unknown_t1[unknown_t1['label'] == cls]
    acc_cls, _, _ = evaluate_accuracy(
        "./ckpt_cl_t1_contrastive/final",
        cls_df['content'].tolist(),
        cls_df['label'].tolist(),
        label2id_cl,
        remap=True
    )
    print(f"    Clasa {cls:2d} ({class_names[cls]:25s}): {acc_cls:.4f} ({acc_cls * 100:.2f}%)")

# ============================================================================
# 3. OVERALL ACCURACY PE TEST_1 COMPLET
# ============================================================================
print("\n" + "=" * 70)
print("3. OVERALL ACCURACY PE TEST_1")
print("=" * 70)

acc_all, _, preds_all = evaluate_accuracy(
    "./ckpt_cl_t1_contrastive/final",
    test_1_df['content'].tolist(),
    test_1_df['label'].tolist(),
    label2id_cl,
    remap=True
)

print(f"Toate clasele (0,1,2,3,7,11,12): {acc_all:.4f} ({acc_all * 100:.2f}%)")

# ============================================================================
# 4. CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 70)
print("4. CLASSIFICATION REPORT PE TEST_1")
print("=" * 70)

report = classification_report(
    test_1_df['label'].tolist(),
    preds_all,
    labels=[0, 1, 2, 3, 7, 11, 12],
    target_names=['Company(0)', 'Edu(1)', 'Artist(2)', 'Athlete(3)',
                  'Nature(7)', 'Album(11)', 'Film(12)'],
    digits=4
)

print(report)

# ============================================================================
# REZUMAT
# ============================================================================
print("\n" + "=" * 70)
print("REZUMAT - CONTRASTIVE CU MAPARE MANUALĂ")
print("=" * 70)

print(f"\n📊 METRICI:")
print(f"  Baseline accuracy (validation): {acc_baseline * 100:.2f}%")
print(f"  Contrastive accuracy (validation): {acc_cl * 100:.2f}%")
print(f"  Forgetting: {drop * 100:.2f}%")
print(f"  Accuracy pe clase noi: {acc_new * 100:.2f}%")
print(f"  Overall test_1: {acc_all * 100:.2f}%")

print(f"\n🔧 MAPARE MANUALĂ:")
print(f"  Bazată pe: Purity clustering (99-100%)")
print(f"  Mapping: {PSEUDO_TO_REAL_MANUAL}")

print(f"\n🔥 CONTRASTIVE LEARNING:")
print(f"  Combined Loss: CE + 0.5 * Contrastive")
print(f"  Temperature: 0.5")

if drop < 0.05 and acc_new > 0.75:
    print("\n🎉 CONTINUAL LEARNING CONTRASTIVE REUȘIT!")
    print("  - Forgetting minim")
    print("  - Accuracy bună pe clase noi")
elif drop > 0.05:
    print("\n⚠️  Forgetting prea mare")
elif acc_new < 0.75:
    print("\n⚠️  Accuracy scăzută pe clase noi")

print("=" * 70)