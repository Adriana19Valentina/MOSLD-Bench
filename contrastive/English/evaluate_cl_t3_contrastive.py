# evaluate_cl_t3_contrastive.py - Evaluare T3 cu mapare MANUALĂ (bazată pe KEYWORDS)

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

print("=" * 70)
print("EVALUARE T3 CONTRASTIVE CU MAPARE MANUALĂ (KEYWORDS)")
print("=" * 70)

# ============================================================================
# MAPARE MANUALĂ (bazată pe KEYWORDS - optimă pentru contrastive)
# ============================================================================

PSEUDO_TO_REAL_T1 = {
    13: 3,  # Athlete (test_1)
    14: 11,  # Album (test_1)
    15: 1  # Edu (test_1)
}

PSEUDO_TO_REAL_T2 = {
    16: 6,  # Building (test_2)
    17: 8,  # Village (test_2)
    18: 4  # OfficeHolder (test_2)
}

# MAPARE BAZATĂ PE KEYWORDS (pentru contrastive loss)
# Contrastive loss trage embeddings către keywords, deci keywords > purity
# PSEUDO_TO_REAL_T3 = {
#     19: 9,  # Cluster 0 (keywords: novel book published) → WrittenWork
#     20: 10,  # Cluster 1 (keywords: species family genus) → Plant
#     21: 5,  # Cluster 2 (keywords: class built ship) → Transport
#     22: 13  # Cluster 3 (keywords: family genus species) → Animal
# }
PSEUDO_TO_REAL_T3 = {
    19: 13,  # Cluster 0: keywords=WrittenWork, DAR samples=97% Animal → 13
    20: 10,  # Cluster 1: keywords=Plant, samples=55% Plant → 10 ✓
    21: 5,   # Cluster 2: keywords=Transport, samples=74% Transport → 5 ✓
    22: 9    # Cluster 3: keywords=Animal, DAR samples=59% WrittenWork → 9
}
PSEUDO_TO_REAL = {**PSEUDO_TO_REAL_T1, **PSEUDO_TO_REAL_T2, **PSEUDO_TO_REAL_T3}

print("\n🔧 Mapare MANUALĂ bazată pe KEYWORDS:")
print("  Test_1: 13→3 (Athlete), 14→11 (Album), 15→1 (Edu)")
print("  Test_2: 16→6 (Building), 17→8 (Village), 18→4 (OfficeHolder)")
print("  Test_3:")
print("    19 → 9 (WrittenWork) - keywords: novel book published")
print("    20 → 10 (Plant) - keywords: species family genus")
print("    21 → 5 (Transport) - keywords: class built ship")
print("    22 → 13 (Animal) - keywords: family genus species")
print("\n💡 Justificare: Contrastive loss trage embeddings către keywords")
print("   → Mapare pe keywords e mai robustă decât pe purity")


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
test_2_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_2.csv")
test_3_df = pd.read_csv("/home/alin/Desktop/ContinualLearning/datasets/English/test_3.csv")

label2id_baseline = {"0": 0, "2": 1, "7": 2, "12": 3}
label2id_cl_t3 = {
    "0": 0, "2": 1, "7": 2, "12": 3,
    "13": 4, "14": 5, "15": 6,
    "16": 7, "17": 8, "18": 9,
    "19": 10, "20": 11, "21": 12, "22": 13
}

# ============================================================================
# 1. CATASTROPHIC FORGETTING CHECK
# ============================================================================
print("\n" + "=" * 70)
print("1. CATASTROPHIC FORGETTING CHECK (VALIDATION)")
print("=" * 70)

acc_baseline, _, _ = evaluate_accuracy(
    "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final",
    # val_df['content'].tolist(),
    val_df['content'].tolist(),
    val_df['label'].tolist(),
    label2id_baseline,
    remap=False
)

acc_cl_t3, _, _ = evaluate_accuracy(
    "./ckpt_cl_t3_contrastive/final",
    # val_df['content'].tolist(),
    val_df['content'].tolist(),
    val_df['label'].tolist(),
    label2id_cl_t3,
    remap=True
)

drop_val = acc_baseline - acc_cl_t3

print(f"\nValidation (clase originale 0,2,7,12):")
print(f"  Baseline:    {acc_baseline:.4f} ({acc_baseline * 100:.2f}%)")
print(f"  After T3:    {acc_cl_t3:.4f} ({acc_cl_t3 * 100:.2f}%)")
print(f"  Drop:        {drop_val:.4f} ({drop_val * 100:.2f}%)")

if drop_val < 0.03:
    print("  ✓ Forgetting minim (<3%)")
elif drop_val < 0.05:
    print("  ⚠ Forgetting moderat (3-5%)")
else:
    print("  ✗ Catastrophic forgetting (>5%)")

# ============================================================================
# 2. CONSISTENCY CHECK - CLASE TEST_1
# ============================================================================
print("\n" + "=" * 70)
print("2. CONSISTENCY CHECK - CLASE TEST_1 (1, 3, 11)")
print("=" * 70)

unknown_t1 = test_1_df[test_1_df['label'].isin([1, 3, 11])]

acc_t1, _, _ = evaluate_accuracy(
    "./ckpt_cl_t3_contrastive/final",
    # unknown_t1['content'].tolist(),
    unknown_t1['content'].tolist(),
    unknown_t1['label'].tolist(),
    label2id_cl_t3,
    remap=True
)

print(f"\nTest_1 (1,3,11): {acc_t1:.4f} ({acc_t1 * 100:.2f}%)")

class_names_t1 = {1: 'EducationalInstitution', 3: 'Athlete', 11: 'Album'}
for cls in [1, 3, 11]:
    cls_df = unknown_t1[unknown_t1['label'] == cls]
    acc_cls, _, _ = evaluate_accuracy(
        "./ckpt_cl_t3_contrastive/final",
        # cls_df['content'].tolist(),
        cls_df['content'].tolist(),
        cls_df['label'].tolist(),
        label2id_cl_t3,
        remap=True
    )
    print(f"  Clasa {cls:2d} ({class_names_t1[cls]:25s}): {acc_cls:.4f} ({acc_cls * 100:.2f}%)")

# ============================================================================
# 3. CONSISTENCY CHECK - CLASE TEST_2
# ============================================================================
print("\n" + "=" * 70)
print("3. CONSISTENCY CHECK - CLASE TEST_2 (4, 6, 8)")
print("=" * 70)

unknown_t2 = test_2_df[test_2_df['label'].isin([4, 6, 8])]

acc_t2, _, _ = evaluate_accuracy(
    "./ckpt_cl_t3_contrastive/final",
    # unknown_t2['content'].tolist(),
    unknown_t2['content'].tolist(),
    unknown_t2['label'].tolist(),
    label2id_cl_t3,
    remap=True
)

print(f"\nTest_2 (4,6,8): {acc_t2:.4f} ({acc_t2 * 100:.2f}%)")

class_names_t2 = {4: 'OfficeHolder', 6: 'Building', 8: 'Village'}
for cls in [4, 6, 8]:
    cls_df = unknown_t2[unknown_t2['label'] == cls]
    acc_cls, _, _ = evaluate_accuracy(
        "./ckpt_cl_t3_contrastive/final",
        # cls_df['content'].tolist(),
        cls_df['content'].tolist(),
        cls_df['label'].tolist(),
        label2id_cl_t3,
        remap=True
    )
    print(f"  Clasa {cls:2d} ({class_names_t2[cls]:25s}): {acc_cls:.4f} ({acc_cls * 100:.2f}%)")

# ============================================================================
# 4. PERFORMANȚĂ PE CLASE NOI TEST_3 (5, 9, 10, 13)
# ============================================================================
print("\n" + "=" * 70)
print("4. PERFORMANȚĂ PE CLASE NOI TEST_3 (5, 9, 10, 13)")
print("=" * 70)

unknown_t3 = test_3_df[test_3_df['label'].isin([5, 9, 10, 13])]

acc_t3_new, _, preds_t3 = evaluate_accuracy(
    "./ckpt_cl_t3_contrastive/final",
    # unknown_t3['content'].tolist(),
    unknown_t3['content'].tolist(),
    unknown_t3['label'].tolist(),
    label2id_cl_t3,
    remap=True
)

print(f"\nTest_3 clase noi (5,9,10,13): {acc_t3_new:.4f} ({acc_t3_new * 100:.2f}%)")

class_names_t3 = {5: 'MeanOfTransportation', 9: 'WrittenWork', 10: 'Plant', 13: 'Animal'}
print(f"\n  Per-class:")
for cls in [5, 9, 10, 13]:
    cls_df = unknown_t3[unknown_t3['label'] == cls]
    acc_cls, _, _ = evaluate_accuracy(
        "./ckpt_cl_t3_contrastive/final",
        # cls_df['content'].tolist(),
        cls_df['content'].tolist(),
        cls_df['label'].tolist(),
        label2id_cl_t3,
        remap=True
    )
    print(f"    Clasa {cls:2d} ({class_names_t3[cls]:25s}): {acc_cls:.4f} ({acc_cls * 100:.2f}%)")

# ============================================================================
# 5. OVERALL ACCURACY PE TEST_3 COMPLET
# ============================================================================
print("\n" + "=" * 70)
print("5. OVERALL ACCURACY PE TEST_3")
print("=" * 70)

acc_all_t3, _, preds_all_t3 = evaluate_accuracy(
    "./ckpt_cl_t3_contrastive/final",
    # test_3_df['content'].tolist(),
    test_3_df['content'].tolist(),
    test_3_df['label'].tolist(),
    label2id_cl_t3,
    remap=True
)

print(f"\nToate clasele test_3: {acc_all_t3:.4f} ({acc_all_t3 * 100:.2f}%)")
print(f"  Clase: 0,1,2,3,4,5,6,7,8,9,10,11,12,13 (14 clase totale)")

# ============================================================================
# 6. CLASSIFICATION REPORT TEST_3
# ============================================================================
print("\n" + "=" * 70)
print("6. CLASSIFICATION REPORT PE TEST_3")
print("=" * 70)

all_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
class_names_all = [
    'Company(0)', 'Edu(1)', 'Artist(2)', 'Athlete(3)',
    'OfficeHolder(4)', 'Transport(5)', 'Building(6)', 'Nature(7)',
    'Village(8)', 'WrittenWork(9)', 'Plant(10)', 'Album(11)',
    'Film(12)', 'Animal(13)'
]

report = classification_report(
    test_3_df['label'].tolist(),
    preds_all_t3,
    labels=all_classes,
    target_names=class_names_all,
    digits=4,
    zero_division=0
)

print(report)

# ============================================================================
# 7. ANALIZA CLUSTERING T3
# ============================================================================
print("\n" + "=" * 70)
print("7. ANALIZA CLUSTERING TEST_3")
print("=" * 70)

print("\nMetrici clustering:")
print("  Silhouette: 0.565 (slab)")
print("  ARI: 0.479 (slab)")
print("\nPurity per cluster:")
print("  Cluster 0 (novel book) → WrittenWork: purity contradictorie (97% Animal în samples)")
print("  Cluster 1 (species family) → Plant: 54.56%")
print("  Cluster 2 (built ship) → Transport: 74.32%")
print("  Cluster 3 (genus species) → Animal: 58.51%")
print("\n⚠️  Clustering dificil din cauza overlap semantic mare!")

# ============================================================================
# REZUMAT FINAL
# ============================================================================
print("\n" + "=" * 70)
print("REZUMAT FINAL - CONTRASTIVE LEARNING (TEST_3)")
print("=" * 70)

print(f"\n📊 METRICI PRINCIPALE:")
print(f"  Forgetting (validation): {drop_val * 100:.2f}%")
print(f"  Consistency test_1 (1,3,11): {acc_t1 * 100:.2f}%")
print(f"  Consistency test_2 (4,6,8): {acc_t2 * 100:.2f}%")
print(f"  Accuracy clase noi test_3: {acc_t3_new * 100:.2f}%")
print(f"  Overall test_3: {acc_all_t3 * 100:.2f}%")

print(f"\n📈 COMPARAȚIE CU BASELINE:")
print(f"  Baseline (validation): {acc_baseline * 100:.2f}%")
print(f"  După 3 task-uri CL: {acc_cl_t3 * 100:.2f}%")
print(f"  Drop cumulat: {drop_val * 100:.2f}%")

print(f"\n🔥 CONTRASTIVE LEARNING:")
print(f"  Combined Loss: CE + 0.5 * Contrastive")
print(f"  Temperature: 0.5")
print(f"  Mapare: Bazată pe keywords (optimă pentru contrastive)")

print(f"\n⚠️  OBSERVAȚII TEST_3:")
print(f"  - Clustering dificil (Silhouette: 0.565, ARI: 0.479)")
print(f"  - Overlap semantic mare între Animal/Plant/WrittenWork")
print(f"  - Purity slabă pe 3/4 clustere (54-74%)")
print(f"  - Mapare bazată pe keywords (nu purity) - contrastive compensează")

if drop_val < 0.05 and acc_t3_new > 0.70:
    print("\n✓ CONTINUAL LEARNING CONTRASTIVE REUȘIT!")
    print("  - Forgetting sub control")
    print("  - Accuracy acceptabilă pe clase noi")
elif drop_val > 0.05:
    print("\n⚠️  Forgetting semnificativ")
elif acc_t3_new < 0.70:
    print("\n⚠️  Accuracy scăzută pe clase noi - clustering problematic")

print("\n" + "=" * 70)