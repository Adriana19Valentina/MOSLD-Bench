# evaluate_cl_contrastive.py - COMPLET cu mapare automată

import pandas as pd
import torch
import numpy as np
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

print("=" * 70)
print("EVALUARE CONTRASTIVE CU MAPARE AUTOMATĂ")
print("=" * 70)

# ============================================================================
# ÎNCARCĂ MAPARE AUTOMATĂ
# ============================================================================

with open('auto_mapping_t1_contrastive.json', 'r') as f:
    mapping_data = json.load(f)

auto_cluster_to_class = {int(k): int(v) for k, v in mapping_data['cluster_to_class'].items()}
avg_similarity = mapping_data['average_similarity']

print(f"\nMapare automată: {auto_cluster_to_class}")
print(f"Similarity: {avg_similarity:.4f}")

# Convertește la pseudo→real
PSEUDO_TO_REAL = {}
for cluster_id, class_id in auto_cluster_to_class.items():
    pseudo_label = 13 + cluster_id
    PSEUDO_TO_REAL[pseudo_label] = class_id

print(f"Pseudo→Real: {PSEUDO_TO_REAL}")


# ============================================================================
# FUNCȚIE EVALUARE
# ============================================================================

def evaluate_accuracy(model_path, texts, labels, label2id, remap=True):
    """
    Evaluează accuracy pe un set de date.

    Args:
        model_path: Path la model
        texts: Lista de texte
        labels: Ground truth labels
        label2id: Mapping label string → index
        remap: Dacă True, mapează pseudo-labels la clase reale

    Returns:
        accuracy, confidences, predictions
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model.eval()

    predictions = []
    confidences = []

    # Procesare în batch-uri
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

    # Convertește prediction indices → labels
    id2label = {v: int(k) for k, v in label2id.items()}
    pred_labels = [id2label.get(p, -1) for p in predictions]

    # Remap pseudo-labels la clase reale
    if remap:
        pred_labels = [PSEUDO_TO_REAL.get(p, p) for p in pred_labels]

    # Calculează accuracy
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
    "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/checkpoint-3000",
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
per_class_acc = {}

for cls in [1, 3, 11]:
    cls_df = unknown_t1[unknown_t1['label'] == cls]
    acc_cls, _, _ = evaluate_accuracy(
        "./ckpt_cl_t1_contrastive/final",
        cls_df['content'].tolist(),
        cls_df['label'].tolist(),
        label2id_cl,
        remap=True
    )
    per_class_acc[cls] = acc_cls
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
# 4. ANALIZA MAPĂRII AUTOMATE
# ============================================================================
print("\n" + "=" * 70)
print("4. ANALIZA CALITĂȚII MAPĂRII AUTOMATE")
print("=" * 70)

print(f"\nSimilaritate medie mapare: {avg_similarity:.4f}")

if avg_similarity >= 0.7:
    print("  ✓ Mapare de înaltă calitate")
elif avg_similarity >= 0.5:
    print("  ⚠ Mapare de calitate medie")
else:
    print("  ✗ Mapare de calitate scăzută")

# Verifică dacă maparea automată e corectă comparativ cu ground truth
print("\nVerificare mapare:")
expected_mapping = {
    0: 3,  # Athlete (keywords: football, played)
    1: 11,  # Album (keywords: band, released) - WRONG! Should be 1 (Edu)
    2: 1  # EducationalInstitution (keywords: school) - WRONG! Should be 11 (Album)
}

# Ground truth corect bazat pe purity
correct_mapping = {
    0: 3,  # Cluster 0 → Athlete (corect)
    1: 11,  # Cluster 1 → Album (corect)
    2: 1  # Cluster 2 → Edu (corect)
}

correct_mappings = 0
for cluster_id, predicted_class in auto_cluster_to_class.items():
    expected_class = correct_mapping.get(cluster_id, -1)
    is_correct = (predicted_class == expected_class)
    correct_mappings += int(is_correct)

    status = "✓" if is_correct else "✗"
    print(f"  {status} Cluster {cluster_id}: Predicted={predicted_class}, Expected={expected_class}")

mapping_accuracy = correct_mappings / len(auto_cluster_to_class) if len(auto_cluster_to_class) > 0 else 0
print(f"\nMapping accuracy: {mapping_accuracy:.2%} ({correct_mappings}/{len(auto_cluster_to_class)})")

# ============================================================================
# 5. CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 70)
print("5. CLASSIFICATION REPORT PE TEST_1")
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
print("REZUMAT - CONTRASTIVE CU MAPARE AUTOMATĂ")
print("=" * 70)

print(f"\n📊 METRICI:")
print(f"  Baseline accuracy (validation): {acc_baseline * 100:.2f}%")
print(f"  Contrastive accuracy (validation): {acc_cl * 100:.2f}%")
print(f"  Forgetting: {drop * 100:.2f}%")
print(f"  Accuracy pe clase noi: {acc_new * 100:.2f}%")
print(f"  Overall test_1: {acc_all * 100:.2f}%")

print(f"\n🔍 MAPARE AUTOMATĂ:")
print(f"  Similarity medie: {avg_similarity:.4f}")
print(f"  Mapping accuracy: {mapping_accuracy:.2%}")
print(f"  Cluster→Class: {auto_cluster_to_class}")

print(f"\n🔥 CONTRASTIVE LEARNING:")
print(f"  Combined Loss: CE + {0.5} * Contrastive")
print(f"  Temperature: {0.5}")

if drop < 0.05 and acc_new > 0.75:
    print("\n🎉 CONTINUAL LEARNING CONTRASTIVE REUȘIT!")
elif drop > 0.05:
    print("\n⚠️  Forgetting prea mare")
elif acc_new < 0.75:
    print("\n⚠️  Accuracy scăzută - verifică maparea sau hyperparametrii")

print("=" * 70)
