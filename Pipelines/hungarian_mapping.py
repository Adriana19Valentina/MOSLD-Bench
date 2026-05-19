import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter


def create_contingency_matrix(cluster_labels, gt_labels):
    cluster_ids = sorted(set(cluster_labels))
    gt_class_ids = sorted(set(gt_labels))

    matrix = np.zeros((len(cluster_ids), len(gt_class_ids)), dtype=int)

    for i, c in enumerate(cluster_ids):
        for j, g in enumerate(gt_class_ids):
            matrix[i, j] = np.sum((cluster_labels == c) & (gt_labels == g))

    return matrix, cluster_ids, gt_class_ids


def hungarian_mapping(cluster_labels, gt_labels):
    contingency, cluster_ids, gt_class_ids = create_contingency_matrix(
        cluster_labels, gt_labels
    )

    n_clusters = len(cluster_ids)
    n_classes = len(gt_class_ids)

    print(f"\nHungarian Mapping:")
    print(f"   Clusters: {n_clusters}, GT Classes: {n_classes}")

    cost_matrix = -contingency

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        cluster_id = cluster_ids[r]
        gt_class_id = gt_class_ids[c]
        count = contingency[r, c]
        mapping[cluster_id] = gt_class_id
        print(f"   Cluster {cluster_id} -> GT {gt_class_id} ({count} samples)")

    assigned_clusters = set(cluster_ids[r] for r in row_ind)
    assigned_classes = set(gt_class_ids[c] for c in col_ind)

    unassigned_clusters = set(cluster_ids) - assigned_clusters
    unassigned_classes = set(gt_class_ids) - assigned_classes

    if unassigned_clusters:
        print(f"\n   Unassigned clusters: {unassigned_clusters}")
        for uc in unassigned_clusters:
            uc_idx = cluster_ids.index(uc)
            total = contingency[uc_idx].sum()
            print(f"      Cluster {uc}: {total} samples (no GT match)")

    if unassigned_classes:
        print(f"\n   Unassigned GT classes: {unassigned_classes}")

    return mapping, unassigned_clusters, unassigned_classes, contingency


def apply_mapping(predictions, mapping, unassigned_label=-1):
    mapped = np.array([
        mapping.get(p, unassigned_label) for p in predictions
    ])
    return mapped


def evaluate_with_hungarian(predictions, gt_labels, known_labels=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report

    mapping, unassigned_clusters, unassigned_classes, contingency = hungarian_mapping(
        predictions, gt_labels
    )

    mapped_preds = apply_mapping(predictions, mapping, unassigned_label=-1)

    assigned_mask = mapped_preds != -1
    n_assigned = np.sum(assigned_mask)
    n_unassigned = np.sum(~assigned_mask)

    print(f"\nSample Assignment:")
    print(f"   Assigned: {n_assigned} ({100 * n_assigned / len(predictions):.1f}%)")
    print(f"   Unassigned: {n_unassigned} ({100 * n_unassigned / len(predictions):.1f}%)")

    if n_assigned > 0:
        y_true_assigned = gt_labels[assigned_mask]
        y_pred_assigned = mapped_preds[assigned_mask]

        acc = accuracy_score(y_true_assigned, y_pred_assigned)
        prec = precision_score(y_true_assigned, y_pred_assigned, average='weighted', zero_division=0)
        rec = recall_score(y_true_assigned, y_pred_assigned, average='weighted', zero_division=0)
        f1 = f1_score(y_true_assigned, y_pred_assigned, average='weighted', zero_division=0)

        print(f"\nMetrics (assigned samples only):")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
    else:
        acc = prec = rec = f1 = 0.0

    print(f"\nPer-Class Accuracy:")
    unique_gt = sorted(set(gt_labels))
    per_class_acc = {}

    for gt_class in unique_gt:
        mask = gt_labels == gt_class
        if np.sum(mask) > 0:
            correct = np.sum((mapped_preds == gt_class) & mask)
            total = np.sum(mask)
            class_acc = correct / total
            per_class_acc[gt_class] = class_acc

            status = "KNOWN" if known_labels and gt_class in known_labels else "NEW"
            print(f"   Class {gt_class} [{status}]: {class_acc:.4f} ({correct}/{total})")

    return {
        'mapping': mapping,
        'unassigned_clusters': unassigned_clusters,
        'unassigned_classes': unassigned_classes,
        'contingency_matrix': contingency,
        'mapped_predictions': mapped_preds,
        'n_assigned': n_assigned,
        'n_unassigned': n_unassigned,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'per_class_accuracy': per_class_acc
    }


def print_contingency_matrix(contingency, cluster_ids, gt_class_ids):
    print("\nContingency Matrix (Cluster x GT Class):")

    header = "        " + "".join([f"GT_{g:>6}" for g in gt_class_ids])
    print(header)
    print("        " + "-" * (7 * len(gt_class_ids)))

    for i, c in enumerate(cluster_ids):
        row = f"Clust_{c:>2}|"
        for j in range(len(gt_class_ids)):
            row += f"{contingency[i, j]:>7}"
        print(row)


if __name__ == "__main__":
    np.random.seed(42)

    gt = np.array([0] * 100 + [1] * 100 + [2] * 100)

    clusters = np.array(
        [0] * 80 + [3] * 20 +
        [1] * 90 + [0] * 10 +
        [2] * 85 + [1] * 15
    )

    print("=" * 60)
    print("HUNGARIAN MAPPING TEST")
    print("=" * 60)

    results = evaluate_with_hungarian(clusters, gt, known_labels=[0])

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
