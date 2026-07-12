# evaluate_cl_t2_contrastive.py - Evaluare T2 cu mapare MANUALĂ

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

print("=" * 70)
print("EVALUARE T2 CONTRASTIVE CU MAPARE MANUALĂ")
print("=" * 70)

# ============================================================================
# MAPARE MANUALĂ (bazată pe purity clustering T2)
# ============================================================================

PSEUDO_TO_REAL_T1 = {
    13: 3,  # Athlete
    14: 11,  # Album
    15: 1  # Edu
}

PSEUDO_TO_REAL_T2 = {
    16: 6,   # Cluster 0 → Building
    17: 8,   # Cluster 1 → Village
    18: 4    # Cluster 2 → OfficeHolder
}


PSEUDO_TO_REAL = {**PSEUDO_TO_REAL_T1, **PSEUDO_TO_REAL_T2}

print("\n🔧 Mapare MANUALĂ (bazată pe purity):")
print("  Test_1: 13→3 (Athlete), 14→11 (Album), 15→1 (Edu)")
print("  Test_2: 16→6 (Building), 17→8 (Village), 18→4 (OfficeHolder)")
# ============================================================================
# FUNCȚIE EVALUARE
# ============================================================================

def evaluate_accuracy(model_path, texts, labels, label2id, remap=True):
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

label2id_baseline = {"0": 0, "2": 1, "7": 2, "12": 3}
label2id_cl_t1 = {"0": 0, "2": 1, "7": 2, "12": 3, "13": 4, "14": 5, "15": 6}
label2id_cl_t2 = {
    "0": 0, "2": 1, "7": 2, "12": 3,
    "13": 4, "14": 5, "15": 6,
    "16": 7, "17": 8, "18": 9
}

# ============================================================================
# EVALUARE
# ============================================================================

# 1. Forgetting
print("\n" + "=" * 70)
print("1. CATASTROPHIC FORGETTING CHECK")
print("=" * 70)

acc_baseline, _, _ = evaluate_accuracy("/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final", val_df['content'].tolist(),
                                       val_df['label'].tolist(), label2id_baseline, False)
acc_cl_t2, _, _ = evaluate_accuracy("./ckpt_cl_t2_contrastive/final", val_df['content'].tolist(),
                                    val_df['label'].tolist(), label2id_cl_t2, True)

print(f"Validation: {acc_baseline:.4f} → {acc_cl_t2:.4f} (drop: {(acc_baseline - acc_cl_t2) * 100:.2f}%)")

# 2. Consistency test_1
print("\n2. CONSISTENCY CHECK - CLASE TEST_1")
unknown_t1 = test_1_df[test_1_df['label'].isin([1, 3, 11])]
acc_t1, _, _ = evaluate_accuracy("./ckpt_cl_t2_contrastive/final", unknown_t1['content'].tolist(),
                                 unknown_t1['label'].tolist(), label2id_cl_t2, True)
print(f"Test_1 (1,3,11): {acc_t1:.4f} ({acc_t1 * 100:.2f}%)")

# 3. Accuracy test_2
print("\n3. PERFORMANȚĂ TEST_2 (4,6,8)")
unknown_t2 = test_2_df[test_2_df['label'].isin([4, 6, 8])]
acc_t2_new, _, _ = evaluate_accuracy("./ckpt_cl_t2_contrastive/final", unknown_t2['content'].tolist(),
                                     unknown_t2['label'].tolist(), label2id_cl_t2, True)
print(f"Test_2 (4,6,8): {acc_t2_new:.4f} ({acc_t2_new * 100:.2f}%)")

# Per-class
for cls in [4, 6, 8]:
    cls_df = unknown_t2[unknown_t2['label'] == cls]
    acc_cls, _, _ = evaluate_accuracy("./ckpt_cl_t2_contrastive/final", cls_df['content'].tolist(),
                                      cls_df['label'].tolist(), label2id_cl_t2, True)
    names = {4: 'OfficeHolder', 6: 'Building', 8: 'Village'}
    print(f"  Clasa {cls} ({names[cls]}): {acc_cls:.4f}")

# 4. Overall
acc_all, _, _ = evaluate_accuracy("./ckpt_cl_t2_contrastive/final", test_2_df['content'].tolist(),
                                  test_2_df['label'].tolist(), label2id_cl_t2, True)
print(f"\nOverall test_2: {acc_all:.4f} ({acc_all * 100:.2f}%)")

print("\n" + "=" * 70)
print("EVALUARE T2 CONTRASTIVE COMPLETĂ!")
print("=" * 70)