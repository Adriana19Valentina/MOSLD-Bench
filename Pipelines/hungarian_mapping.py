import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter


def create_contingency_matrix(cluster_labels, gt_labels):
    """
    Create contingency matrix between clusters and GT classes.

    Args:
        cluster_labels: Array of cluster assignments
        gt_labels: Array of ground truth labels

    Returns:
        matrix: Shape (n_clusters, n_gt_classes)
        cluster_ids: List of unique cluster IDs
        gt_class_ids: List of unique GT class IDs
    """
    cluster_ids = sorted(set(cluster_labels))
    gt_class_ids = sorted(set(gt_labels))

    matrix = np.zeros((len(cluster_ids), len(gt_class_ids)), dtype=int)

    for i, c in enumerate(cluster_ids):
        for j, g in enumerate(gt_class_ids):
            matrix[i, j] = np.sum((cluster_labels == c) & (gt_labels == g))

    return matrix, cluster_ids, gt_class_ids


def hungarian_mapping(cluster_labels, gt_labels):
    """
    Map clusters to GT classes using Hungarian Algorithm (optimal 1-to-1 assignment).

    If n_clusters > n_gt_classes: extra clusters remain unassigned
    If n_clusters < n_gt_classes: some GT classes won't have a cluster

    Args:
        cluster_labels: Array of cluster assignments (pseudo-labels)
        gt_labels: Array of ground truth labels

    Returns:
        mapping: Dict {cluster_id: gt_class_id} for assigned clusters
        unassigned_clusters: Set of cluster IDs that couldn't be mapped
        unassigned_classes: Set of GT class IDs that have no cluster
        contingency: The contingency matrix used
    """
    # Create contingency matrix
    contingency, cluster_ids, gt_class_ids = create_contingency_matrix(
        cluster_labels, gt_labels
    )

    n_clusters = len(cluster_ids)
    n_classes = len(gt_class_ids)

    print(f"\n📊 Hungarian Mapping:")
    print(f"   Clusters: {n_clusters}, GT Classes: {n_classes}")

    # Hungarian algorithm needs cost matrix (we want to maximize matches, so negate)
    # If matrix is not square, linear_sum_assignment handles it
    cost_matrix = -contingency  # Negative because we maximize

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        cluster_id = cluster_ids[r]
        gt_class_id = gt_class_ids[c]
        count = contingency[r, c]
        mapping[cluster_id] = gt_class_id
        print(f"   Cluster {cluster_id} → GT {gt_class_id} ({count} samples)")

    # Find unassigned
    assigned_clusters = set(cluster_ids[r] for r in row_ind)
    assigned_classes = set(gt_class_ids[c] for c in col_ind)

    unassigned_clusters = set(cluster_ids) - assigned_clusters
    unassigned_classes = set(gt_class_ids) - assigned_classes

    if unassigned_clusters:
        print(f"\n   ⚠️  Unassigned clusters: {unassigned_clusters}")
        for uc in unassigned_clusters:
            uc_idx = cluster_ids.index(uc)
            total = contingency[uc_idx].sum()
            print(f"      Cluster {uc}: {total} samples (no GT match)")

    if unassigned_classes:
        print(f"\n   ⚠️  Unassigned GT classes: {unassigned_classes}")

    return mapping, unassigned_clusters, unassigned_classes, contingency


def apply_mapping(predictions, mapping, unassigned_label=-1):
    """
    Apply cluster-to-GT mapping to predictions.

    Args:
        predictions: Array of cluster/pseudo-label predictions
        mapping: Dict {cluster_id: gt_class_id}
        unassigned_label: Label to use for unassigned clusters (default: -1)

    Returns:
        mapped_predictions: Array with GT class labels (-1 for unassigned)
    """
    mapped = np.array([
        mapping.get(p, unassigned_label) for p in predictions
    ])
    return mapped


def evaluate_with_hungarian(predictions, gt_labels, known_labels=None):
    """
    Full evaluation pipeline with Hungarian mapping.

    Args:
        predictions: Model predictions (pseudo-labels or cluster IDs)
        gt_labels: Ground truth labels
        known_labels: Labels that were known during training (for separate eval)

    Returns:
        results: Dict with metrics and mapping info
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report

    # Get Hungarian mapping
    mapping, unassigned_clusters, unassigned_classes, contingency = hungarian_mapping(
        predictions, gt_labels
    )

    # Apply mapping
    mapped_preds = apply_mapping(predictions, mapping, unassigned_label=-1)

    # Separate assigned vs unassigned samples
    assigned_mask = mapped_preds != -1
    n_assigned = np.sum(assigned_mask)
    n_unassigned = np.sum(~assigned_mask)

    print(f"\n📊 Sample Assignment:")
    print(f"   Assigned: {n_assigned} ({100 * n_assigned / len(predictions):.1f}%)")
    print(f"   Unassigned: {n_unassigned} ({100 * n_unassigned / len(predictions):.1f}%)")

    # Metrics on assigned samples only
    if n_assigned > 0:
        y_true_assigned = gt_labels[assigned_mask]
        y_pred_assigned = mapped_preds[assigned_mask]

        acc = accuracy_score(y_true_assigned, y_pred_assigned)
        prec = precision_score(y_true_assigned, y_pred_assigned, average='weighted', zero_division=0)
        rec = recall_score(y_true_assigned, y_pred_assigned, average='weighted', zero_division=0)
        f1 = f1_score(y_true_assigned, y_pred_assigned, average='weighted', zero_division=0)

        print(f"\n📊 Metrics (assigned samples only):")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
    else:
        acc = prec = rec = f1 = 0.0

    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    unique_gt = sorted(set(gt_labels))
    per_class_acc = {}

    for gt_class in unique_gt:
        mask = gt_labels == gt_class
        if np.sum(mask) > 0:
            # For this class, what fraction was correctly predicted?
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
    """Pretty print the contingency matrix."""
    print("\n📊 Contingency Matrix (Cluster × GT Class):")

    # Header
    header = "        " + "".join([f"GT_{g:>6}" for g in gt_class_ids])
    print(header)
    print("        " + "-" * (7 * len(gt_class_ids)))

    # Rows
    for i, c in enumerate(cluster_ids):
        row = f"Clust_{c:>2}|"
        for j in range(len(gt_class_ids)):
            row += f"{contingency[i, j]:>7}"
        print(row)


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)

    # Simulate: 3 GT classes, 4 clusters (one extra)
    gt = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # Clusters mostly align but not perfectly
    clusters = np.array(
        [0] * 80 + [3] * 20 +  # GT 0 -> mostly cluster 0, some to extra cluster 3
        [1] * 90 + [0] * 10 +  # GT 1 -> mostly cluster 1
        [2] * 85 + [1] * 15  # GT 2 -> mostly cluster 2
    )

    print("=" * 60)
    print("HUNGARIAN MAPPING TEST")
    print("=" * 60)

    results = evaluate_with_hungarian(clusters, gt, known_labels=[0])

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)