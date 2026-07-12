"""
plda_mosld.py
=============
Integrare PLDA cu MOSLD-Bench pe cele 3 taskuri (T1, T2, T3).

Diferente fata de PLDA original:
- Backbone: bert-base-uncased (CLS token) in loc de ViT/DeiT
- OOD detection: energy-based din pipeline-urile voastre (deja rulat)
- Labels: pseudo-labels din K-means clustering (nu ground truth)
- Stream: batch per task (T1->T2->T3) in loc de imagine cu imagine

Presupuneri:
- pipeline_t1.py, pipeline_t2.py, pipeline_t3.py au fost deja rulate
- T1_PROCESSED_CSV, T2_PROCESSED_CSV, T3_PROCESSED_CSV exista
- T1_RESULTS_PKL, T2_RESULTS_PKL, T3_RESULTS_PKL exista

Usage:
    python plda_mosld.py
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment

# =========================================================================
# CONFIG (importat din config.py al tau)
# =========================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import (
        MODEL_NAME, MAX_LENGTH,
        BASELINE_LABELS, TEST_1_NEW_LABELS, TEST_2_NEW_LABELS, TEST_3_NEW_LABELS,
        CLASS_NAMES,
        TRAIN_CSV, VAL_CSV, TEST_1_CSV, TEST_2_CSV, TEST_3_CSV,
        T1_PROCESSED_CSV, T1_RESULTS_PKL,
        T2_PROCESSED_CSV, T2_RESULTS_PKL,
        T3_PROCESSED_CSV, T3_RESULTS_PKL,
        PSEUDO_LABEL_START_T1, PSEUDO_LABEL_START_T2, PSEUDO_LABEL_START_T3,
        OUTPUT_DIR,
    )
    print(f"Config loaded: {MODEL_NAME}, {len(BASELINE_LABELS)} baseline classes")
except ImportError:
    # Fallback pentru testare fara config.py
    print("WARNING: config.py not found, using defaults for English dataset")
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    BASELINE_LABELS = [0, 1, 2, 3]
    TEST_1_NEW_LABELS = [4, 5, 6]
    TEST_2_NEW_LABELS = [7, 8, 9]
    TEST_3_NEW_LABELS = [10, 11, 12, 13]
    CLASS_NAMES = {i: f"Class_{i}" for i in range(14)}
    DATASET_DIR = './dataset'
    TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
    VAL_CSV = os.path.join(DATASET_DIR, 'val.csv')
    TEST_1_CSV = os.path.join(DATASET_DIR, 'test_1.csv')
    TEST_2_CSV = os.path.join(DATASET_DIR, 'test_2.csv')
    TEST_3_CSV = os.path.join(DATASET_DIR, 'test_3.csv')
    OUTPUT_DIR = './plda_outputs'
    T1_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_1_processed.csv')
    T2_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_2_processed.csv')
    T3_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_3_processed.csv')
    T1_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_1_results.pkl')
    T2_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_2_results.pkl')
    T3_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_3_results.pkl')
    PSEUDO_LABEL_START_T1 = 109
    PSEUDO_LABEL_START_T2 = 119
    PSEUDO_LABEL_START_T3 = 129

# Numar total de clase (baseline + T1 + T2 + T3)
ALL_GT_LABELS = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS + TEST_3_NEW_LABELS
NUM_CLASSES_TOTAL = len(ALL_GT_LABELS)

# Feature size: bert-base = 768, bert-large = 1024
FEATURE_SIZE = 768
SHRINKAGE = 1e-4
BATCH_SIZE_ENCODE = 32

# Output pentru PLDA
PLDA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'plda_results')
os.makedirs(PLDA_OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# =========================================================================
# PLDA MODEL (din PLDA_Model.py original)
# =========================================================================

class PLDA(torch.nn.Module):
    def __init__(self, input_shape, num_classes, test_batch_size=1024, shrinkage_param=1e-4):
        super(PLDA, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param

        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)
        self.muK_all = torch.zeros((input_shape)).to(self.device)
        self.cK_all = torch.tensor([0]).to(self.device)
        self.Sigma_all = torch.ones((input_shape, input_shape)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    def fit_base(self, X, y):
        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()

        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        self.muK_all = X.mean(0)
        self.cK_all = X.shape[0]

        print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().detach().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

        cov_estimator_all = OAS(assume_centered=True)
        cov_estimator_all.fit((X - self.muK_all).cpu().detach().numpy())
        self.Sigma_all = torch.from_numpy(cov_estimator_all.covariance_).float().to(self.device)

    def fit_open_world(self, x, y):
        x = x.to(self.device)
        y = y.long().to(self.device)

        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1
            self.muK_all[:] += ((x - self.muK_all[:]) / (self.cK_all + 1))[0]
            self.cK_all += 1

    def predict(self, X, return_probas=False):
        X = X.to(self.device)

        with torch.no_grad():
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            if self.prev_num_updates != self.num_updates:
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma +
                    self.shrinkage_param * torch.eye(self.input_shape).to(self.device)
                )
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()


# =========================================================================
# ENCODER TEXT
# =========================================================================

print(f"\nLoading encoder: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()
print("Encoder loaded.")


def get_features(texts, batch_size=BATCH_SIZE_ENCODE):
    """
    Encode texts -> embeddings normalizate (CLS token).
    Identic cu get_embeddings din pipeline-urile voastre,
    dar cu L2 normalizare + scaling 0.1 ca in PLDA original.
    """
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [str(t) if t is not None and str(t) != 'nan' else '' for t in batch]

        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            output = encoder(**inputs)
            emb = output.last_hidden_state[:, 0, :]  # CLS token
            emb = F.normalize(emb, p=2, dim=1)       # L2 normalize
            emb = emb * 0.1                           # scale (ca in PLDA original)

        all_embeddings.append(emb.cpu())

        if (i // batch_size) % 10 == 0:
            print(f'\r  Encoding: {min(i + batch_size, len(texts))}/{len(texts)}', end='')

    print()
    return torch.cat(all_embeddings, dim=0)


# =========================================================================
# EVALUARE
# =========================================================================

def get_embedding(model, tokenizer, text):
    """Embedding pentru un singur text (CLS token)."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', padding=True,
                       truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding[0]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def semantic_hungarian_mapping(encoder, tokenizer, keywords_dict,
                                pseudo_labels_ordered, new_gt_labels_this_task):
    """
    Mapeaza pseudo-labels la GT labels folosind similaritate semantica +
    Hungarian algorithm.

    IMPORTANT: Hungarian mapping se face DOAR pe clasele noi din etapa curenta
    (ex: T2 → doar TEST_2_NEW_LABELS), identic cu evaluate_t1/t2/t3.py.

    Params:
        encoder: model pentru embeddings
        tokenizer: tokenizer
        keywords_dict: {cluster_id: [keywords]} din results pkl
        pseudo_labels_ordered: lista ordonata de pseudo-labels din etapa curenta
        new_gt_labels_this_task: GT labels noi DOAR din aceasta etapa

    Returns:
        mapping: {pseudo_label -> gt_label}
        avg_sim: similaritate medie
    """
    n_clusters = len(pseudo_labels_ordered)
    new_class_names = [CLASS_NAMES.get(l, str(l)) for l in new_gt_labels_this_task]

    if n_clusters == 0:
        return {}, 0.0

    # Embeddings clustere (keywords concatenate)
    cluster_embeddings = []
    for i in range(n_clusters):
        kws = keywords_dict.get(i, [])
        seq = ' '.join(kws[:10]) if kws else f'cluster {i}'
        emb = get_embedding(encoder, tokenizer, seq)
        cluster_embeddings.append(emb)
        print(f"    Cluster {i} (pseudo={pseudo_labels_ordered[i]}): "
              f"'{seq[:50]}...'")

    # Embeddings class names
    class_embeddings = []
    for name in new_class_names:
        emb = get_embedding(encoder, tokenizer, name)
        class_embeddings.append(emb)

    # Matrice de similaritate
    n_gt = len(new_gt_labels_this_task)
    sim_matrix = np.zeros((n_clusters, n_gt))
    for i in range(n_clusters):
        for j in range(n_gt):
            sim_matrix[i, j] = cosine_sim(cluster_embeddings[i], class_embeddings[j])

    # Afisare matrice
    print(f"\n  Similarity matrix (clusters x new classes THIS TASK):")
    header = "            " + "".join([f"{CLASS_NAMES.get(g, g):>14}" for g in new_gt_labels_this_task])
    print(header)
    for i in range(n_clusters):
        row = f"  Pseudo {pseudo_labels_ordered[i]:3d} |"
        for j in range(n_gt):
            row += f"{sim_matrix[i, j]:>14.4f}"
        print(row)

    # Hungarian algorithm
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    total_sim = 0.0
    print(f"\n  Hungarian mapping (THIS TASK ONLY):")
    for i, j in zip(row_ind, col_ind):
        pseudo = pseudo_labels_ordered[i]
        gt = new_gt_labels_this_task[j]
        sim = sim_matrix[i, j]
        total_sim += sim
        mapping[pseudo] = gt
        print(f"    Pseudo {pseudo} → GT {gt} "
              f"({CLASS_NAMES.get(gt, gt)}), sim={sim:.4f}")

    avg_sim = total_sim / len(row_ind) if row_ind.size > 0 else 0.0
    print(f"  Average similarity: {avg_sim:.4f}")
    return mapping, avg_sim


def evaluate_on_test(classifier, test_csv, pseudo_to_gt_map,
                     task_name, all_known_pseudo_labels,
                     baseline_labels, new_labels_this_task):
    """
    Evalueaza PLDA pe un fisier de test.
    Metrici identice cu evaluate_t1/t2/t3.py:
    - accuracy si F1 macro/weighted overall / known / new
    - per-class accuracy si F1
    - ARI pe clasele noi

    Params:
        classifier: instanta PLDA
        test_csv: calea catre fisierul de test
        pseudo_to_gt_map: dictionar complet {pseudo_label -> gt_label}
        task_name: string pentru afisare
        all_known_pseudo_labels: set cu toate pseudo-labelurile cunoscute
        baseline_labels: lista GT labels baseline
        new_labels_this_task: GT labels NOI introduse in aceasta etapa

    Returns:
        dict cu metrici
    """
    from sklearn.metrics import accuracy_score, f1_score

    print(f"\n{'=' * 70}")
    print(f"EVALUARE {task_name}")
    print('=' * 70)

    test_df = pd.read_csv(test_csv)
    texts = test_df['content'].tolist()
    gt_labels_np = test_df['label'].values

    print(f"  Test samples: {len(texts)}")
    print(f"  GT label distribution: {dict(pd.Series(gt_labels_np).value_counts().sort_index())}")

    # Genereaza embeddings si obtine scoruri PLDA
    X_test = get_features(texts)
    scores = classifier.predict(X_test, return_probas=False)  # (N, PLDA_NUM_CLASSES)

    # Mascam scorurile la clasele cunoscute
    # (baseline indices + pseudo-label indices)
    known_indices = list(baseline_labels) + list(all_known_pseudo_labels)
    masked_scores = torch.full_like(scores, float('-inf'))
    for idx in known_indices:
        if idx < scores.shape[1]:
            masked_scores[:, idx] = scores[:, idx]

    _, pred_indices = masked_scores.max(dim=1)
    pred_indices_np = pred_indices.numpy()

    # Mapeaza predictiile pseudo -> GT
    mapped_preds = np.array([
        pseudo_to_gt_map.get(p, p) for p in pred_indices_np
    ])

    valid_mask = mapped_preds >= 0

    # ── Overall ──────────────────────────────────────────────────────────
    overall_acc = accuracy_score(gt_labels_np[valid_mask], mapped_preds[valid_mask])
    overall_f1_macro = f1_score(gt_labels_np[valid_mask], mapped_preds[valid_mask],
                                average='macro', zero_division=0)
    overall_f1_weighted = f1_score(gt_labels_np[valid_mask], mapped_preds[valid_mask],
                                   average='weighted', zero_division=0)

    # ── Known (baseline) ─────────────────────────────────────────────────
    known_mask = np.isin(gt_labels_np, baseline_labels) & valid_mask
    if known_mask.sum() > 0:
        known_acc = accuracy_score(gt_labels_np[known_mask], mapped_preds[known_mask])
        known_f1_macro = f1_score(gt_labels_np[known_mask], mapped_preds[known_mask],
                                  labels=baseline_labels, average='macro', zero_division=0)
        known_f1_weighted = f1_score(gt_labels_np[known_mask], mapped_preds[known_mask],
                                     average='weighted', zero_division=0)
    else:
        known_acc = known_f1_macro = known_f1_weighted = 0.0

    # ── New classes ───────────────────────────────────────────────────────
    # NEW = toate clasele adaugate PANA ACUM (cumulative), nu doar din aceasta etapa
    # identic cu evaluate_t2.py: NEW_LABELS = TEST_1_NEW_LABELS + TEST_2_NEW_LABELS
    # new_labels_this_task contine labels cumulative (ex: T1+T2 pentru TEST_2)
    all_new_labels = new_labels_this_task  # deja cumulative, pasate din exterior
    new_mask = np.isin(gt_labels_np, all_new_labels) & valid_mask
    if new_mask.sum() > 0:
        new_acc = accuracy_score(gt_labels_np[new_mask], mapped_preds[new_mask])
        new_f1_macro = f1_score(gt_labels_np[new_mask], mapped_preds[new_mask],
                                labels=list(all_new_labels), average='macro', zero_division=0)
        new_f1_weighted = f1_score(gt_labels_np[new_mask], mapped_preds[new_mask],
                                   average='weighted', zero_division=0)
    else:
        new_acc = new_f1_macro = new_f1_weighted = 0.0

    # ── Per-class ─────────────────────────────────────────────────────────
    per_class_acc = {}
    per_class_f1 = {}
    all_labels_in_test = np.unique(gt_labels_np)
    for label in all_labels_in_test:
        mask = (gt_labels_np == label) & valid_mask
        if mask.sum() > 0:
            acc = accuracy_score(gt_labels_np[mask], mapped_preds[mask])
            binary_gt = (gt_labels_np[valid_mask] == label).astype(int)
            binary_pred = (mapped_preds[valid_mask] == label).astype(int)
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            per_class_acc[int(label)] = float(acc)
            per_class_f1[int(label)] = float(f1)

    # ── ARI pe clasele noi ───────────────────────────────────────────────
    if new_mask.sum() > 0:
        ari = adjusted_rand_score(
            gt_labels_np[new_mask],
            pred_indices_np[new_mask]  # pred inainte de mapping, pt ARI
        )
    else:
        ari = 0.0

    # ── Forgetting analysis (per step, doar la TEST_3) ────────────────────
    forgetting = {}
    if task_name == 'TEST_3':
        step_groups = {
            'baseline': baseline_labels,
            'T1': TEST_1_NEW_LABELS,
            'T2': TEST_2_NEW_LABELS,
            'T3': TEST_3_NEW_LABELS,
        }
        print(f"\n  Forgetting Analysis (per step):")
        print(f"  {'Step':<12} {'Accuracy':>10} {'Samples':>10}")
        print(f"  {'─' * 35}")
        for step_name, step_labels in step_groups.items():
            step_mask = np.isin(gt_labels_np, step_labels) & valid_mask
            if step_mask.sum() > 0:
                step_acc = float(accuracy_score(gt_labels_np[step_mask], mapped_preds[step_mask]))
            else:
                step_acc = 0.0
            forgetting[step_name] = {'acc': step_acc, 'n': int(step_mask.sum())}
            print(f"  {step_name:<12} {step_acc:>10.4f} {step_mask.sum():>10}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n  {'':17} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}")
    print(f"  {'─' * 52}")
    print(f"  {'OVERALL':17} {overall_acc:>10.4f} {overall_f1_macro:>10.4f} {overall_f1_weighted:>12.4f}")
    print(f"  {'KNOWN':17} {known_acc:>10.4f} {known_f1_macro:>10.4f} {known_f1_weighted:>12.4f}")
    print(f"  {'NEW':17} {new_acc:>10.4f} {new_f1_macro:>10.4f} {new_f1_weighted:>12.4f}")
    print(f"  {'ARI (new only)':17} {ari:>10.4f}")

    print(f"\n  Per-class (known):")
    for label in baseline_labels:
        if label in per_class_acc:
            print(f"    {label} ({CLASS_NAMES.get(label, '?'):25}): "
                  f"acc={per_class_acc[label]:.4f}, f1={per_class_f1[label]:.4f}")

    print(f"\n  Per-class (new):")
    for label in all_new_labels:
        if label in per_class_acc:
            print(f"    {label} ({CLASS_NAMES.get(label, '?'):25}): "
                  f"acc={per_class_acc[label]:.4f}, f1={per_class_f1[label]:.4f}")

    return {
        'task': task_name,
        'total_samples': int(len(texts)),
        'overall_acc': float(overall_acc),
        'overall_f1_macro': float(overall_f1_macro),
        'overall_f1_weighted': float(overall_f1_weighted),
        'known_acc': float(known_acc),
        'known_f1_macro': float(known_f1_macro),
        'known_f1_weighted': float(known_f1_weighted),
        'new_acc': float(new_acc),
        'new_f1_macro': float(new_f1_macro),
        'new_f1_weighted': float(new_f1_weighted),
        'ari': float(ari),
        'forgetting': forgetting,  # populated only for TEST_3, {} otherwise
        'per_class_acc': {CLASS_NAMES.get(k, str(k)): v for k, v in per_class_acc.items()},
        'per_class_f1': {CLASS_NAMES.get(k, str(k)): v for k, v in per_class_f1.items()},
    }


# =========================================================================
# STEP 1: FIT BASE
# =========================================================================

print('\n' + '=' * 70)
print('STEP 1: FIT BASE (pre-deployment)')
print('=' * 70)

start_time = time.time()

train_df = pd.read_csv(TRAIN_CSV)
print(f"Training samples: {len(train_df)}")
print(f"Baseline classes: {BASELINE_LABELS}")
print(f"Class names: {[CLASS_NAMES.get(l, l) for l in BASELINE_LABELS]}")

# PLDA opereaza in spatiul tuturor pseudo-labelurilor posibile
# num_classes trebuie sa fie >= max(pseudo_label) + 1
max_pseudo = max(PSEUDO_LABEL_START_T3 + 10,
                 PSEUDO_LABEL_START_T2 + 10,
                 PSEUDO_LABEL_START_T1 + 10)
PLDA_NUM_CLASSES = max_pseudo + 1

print(f"\nPLDA spatiu clase: {PLDA_NUM_CLASSES} (include pseudo-labels)")
classifier = PLDA(FEATURE_SIZE, PLDA_NUM_CLASSES, shrinkage_param=SHRINKAGE)

base_texts = train_df['content'].tolist()
base_labels = torch.tensor(train_df['label'].tolist(), dtype=torch.long)

print(f"\nGenerating base embeddings for {len(base_texts)} samples...")
X_base = get_features(base_texts)
classifier.fit_base(X_base, base_labels)

# Pseudo-to-GT mapping initial: baseline labels se mapeaza la ele insele
pseudo_to_gt_map = {l: l for l in BASELINE_LABELS}

# ID classes curente (GT labels bine invatate)
id_classes = BASELINE_LABELS.copy()
# Pseudo-labels curente (pt evaluare restrictata)
all_known_pseudo_labels = set()

print(f"\nBase fitting done in {time.time() - start_time:.1f}s")


# =========================================================================
# STEP 2: OPEN WORLD T1
# Preia pseudo-labels din pipeline_t1.py (deja rulat)
# =========================================================================

print('\n' + '=' * 70)
print('STEP 2: OPEN WORLD T1')
print('=' * 70)

if not os.path.exists(T1_RESULTS_PKL):
    raise FileNotFoundError(
        f"T1 results not found: {T1_RESULTS_PKL}\n"
        "Please run pipeline_t1.py first."
    )

with open(T1_RESULTS_PKL, 'rb') as f:
    t1_results = pickle.load(f)

t1_cluster_to_pseudo = t1_results['cluster_to_pseudo']
t1_k = t1_results['K_final']
t1_ari_clustering = t1_results.get('ari', None)

print(f"T1 clustering: K={t1_k}, ARI={t1_ari_clustering:.4f}" if t1_ari_clustering else f"T1 clustering: K={t1_k}")
print(f"T1 pseudo-labels: {list(t1_cluster_to_pseudo.values())}")

# Incarca datele procesate T1 (baseline + nou)
t1_df = pd.read_csv(T1_PROCESSED_CSV)
# Doar exemplele noi (cu pseudo-labels, nu baseline)
new_t1_df = t1_df[~t1_df['label'].isin(BASELINE_LABELS)].copy()
new_t1_texts = new_t1_df['content'].tolist()
new_t1_pseudo_labels = new_t1_df['label'].tolist()

print(f"\nT1 new samples for fit_open_world: {len(new_t1_texts)}")
print(f"Pseudo-label distribution: {dict(pd.Series(new_t1_pseudo_labels).value_counts())}")

# Genereaza embeddings pentru T1
print("\nGenerating T1 embeddings...")
X_t1 = get_features(new_t1_texts)

# fit_open_world: exemplu cu exemplu (ca in PLDA original)
print("Running fit_open_world for T1...")
for i in range(len(new_t1_texts)):
    x = X_t1[i:i+1]  # (1, 768)
    y = torch.tensor([new_t1_pseudo_labels[i]], dtype=torch.long)
    classifier.fit_open_world(x, y)
    if i % 100 == 0:
        print(f'\r  fit_open_world: {i}/{len(new_t1_texts)}', end='')
print(f'\r  fit_open_world: {len(new_t1_texts)}/{len(new_t1_texts)} done')

# Actualizeaza pseudo-labels cunoscute
t1_pseudo_set = set(t1_cluster_to_pseudo.values())
all_known_pseudo_labels.update(t1_pseudo_set)

# Mapping pseudo -> GT pentru T1: semantic Hungarian (identic cu evaluate_t1.py)
t1_cluster_keywords = {int(k): v for k, v in t1_results.get('cluster_keywords', {}).items()}
t1_cluster_unique_kw = {int(k): v for k, v in t1_results.get('cluster_unique_keywords', {}).items()}
use_unique_t1 = (len(t1_cluster_unique_kw) > 0 and
                 all(len(v) > 0 for v in t1_cluster_unique_kw.values()))
t1_keywords_to_use = t1_cluster_unique_kw if use_unique_t1 else t1_cluster_keywords
t1_pseudo_ordered = [t1_cluster_to_pseudo[k] for k in sorted(t1_cluster_to_pseudo.keys())]

print(f"\nSemantic Hungarian mapping T1 ({'unique' if use_unique_t1 else 'regular'} keywords)...")
t1_mapping, t1_avg_sim = semantic_hungarian_mapping(
    encoder, tokenizer,
    t1_keywords_to_use,
    t1_pseudo_ordered,
    TEST_1_NEW_LABELS
)
pseudo_to_gt_map.update(t1_mapping)

# Evalueaza pe TEST_1
results_t1 = evaluate_on_test(
    classifier, TEST_1_CSV,
    pseudo_to_gt_map=pseudo_to_gt_map,
    task_name='TEST_1',
    all_known_pseudo_labels=all_known_pseudo_labels,
    baseline_labels=BASELINE_LABELS,
    new_labels_this_task=TEST_1_NEW_LABELS  # cumulative: doar T1
)

# Actualizeaza id_classes cu clasele T1 (bine invatate dupa toate exemplele)
id_classes = BASELINE_LABELS + TEST_1_NEW_LABELS


# =========================================================================
# STEP 3: OPEN WORLD T2
# =========================================================================

print('\n' + '=' * 70)
print('STEP 3: OPEN WORLD T2')
print('=' * 70)

if not os.path.exists(T2_RESULTS_PKL):
    raise FileNotFoundError(
        f"T2 results not found: {T2_RESULTS_PKL}\n"
        "Please run pipeline_t2.py first."
    )

with open(T2_RESULTS_PKL, 'rb') as f:
    t2_results = pickle.load(f)

t2_cluster_to_pseudo = t2_results['cluster_to_pseudo']
t2_k = t2_results['K_final']
t2_ari_clustering = t2_results.get('ari', None)

print(f"T2 clustering: K={t2_k}, ARI={t2_ari_clustering:.4f}" if t2_ari_clustering else f"T2 clustering: K={t2_k}")
print(f"T2 pseudo-labels: {list(t2_cluster_to_pseudo.values())}")

# Incarca datele procesate T2
t2_df = pd.read_csv(T2_PROCESSED_CSV)
known_labels_t2 = BASELINE_LABELS + list(all_known_pseudo_labels)
new_t2_df = t2_df[~t2_df['label'].isin(known_labels_t2)].copy()
new_t2_texts = new_t2_df['content'].tolist()
new_t2_pseudo_labels = new_t2_df['label'].tolist()

print(f"\nT2 new samples for fit_open_world: {len(new_t2_texts)}")
print(f"Pseudo-label distribution: {dict(pd.Series(new_t2_pseudo_labels).value_counts())}")

# Genereaza embeddings pentru T2
print("\nGenerating T2 embeddings...")
X_t2 = get_features(new_t2_texts)

# fit_open_world pentru T2
print("Running fit_open_world for T2...")
for i in range(len(new_t2_texts)):
    x = X_t2[i:i+1]
    y = torch.tensor([new_t2_pseudo_labels[i]], dtype=torch.long)
    classifier.fit_open_world(x, y)
    if i % 100 == 0:
        print(f'\r  fit_open_world: {i}/{len(new_t2_texts)}', end='')
print(f'\r  fit_open_world: {len(new_t2_texts)}/{len(new_t2_texts)} done')

# Actualizeaza pseudo-labels cunoscute
t2_pseudo_set = set(t2_cluster_to_pseudo.values())
all_known_pseudo_labels.update(t2_pseudo_set)

# Mapping pseudo -> GT pentru T2: semantic Hungarian
t2_cluster_keywords = {int(k): v for k, v in t2_results.get('cluster_keywords', {}).items()}
t2_cluster_unique_kw = {int(k): v for k, v in t2_results.get('cluster_unique_keywords', {}).items()}
use_unique_t2 = (len(t2_cluster_unique_kw) > 0 and
                 all(len(v) > 0 for v in t2_cluster_unique_kw.values()))
t2_keywords_to_use = t2_cluster_unique_kw if use_unique_t2 else t2_cluster_keywords
t2_pseudo_ordered = [t2_cluster_to_pseudo[k] for k in sorted(t2_cluster_to_pseudo.keys())]

print(f"\nSemantic Hungarian mapping T2 ({'unique' if use_unique_t2 else 'regular'} keywords)...")
t2_mapping, t2_avg_sim = semantic_hungarian_mapping(
    encoder, tokenizer,
    t2_keywords_to_use,
    t2_pseudo_ordered,
    TEST_2_NEW_LABELS
)
pseudo_to_gt_map.update(t2_mapping)

# Evalueaza pe TEST_2
results_t2 = evaluate_on_test(
    classifier, TEST_2_CSV,
    pseudo_to_gt_map=pseudo_to_gt_map,
    task_name='TEST_2',
    all_known_pseudo_labels=all_known_pseudo_labels,
    baseline_labels=BASELINE_LABELS,
    new_labels_this_task=TEST_1_NEW_LABELS + TEST_2_NEW_LABELS  # cumulative: T1+T2
)

# Actualizeaza id_classes
id_classes = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS


# =========================================================================
# STEP 4: OPEN WORLD T3
# =========================================================================

print('\n' + '=' * 70)
print('STEP 4: OPEN WORLD T3')
print('=' * 70)

if not os.path.exists(T3_RESULTS_PKL):
    raise FileNotFoundError(
        f"T3 results not found: {T3_RESULTS_PKL}\n"
        "Please run pipeline_t3.py first."
    )

with open(T3_RESULTS_PKL, 'rb') as f:
    t3_results = pickle.load(f)

t3_cluster_to_pseudo = t3_results['cluster_to_pseudo']
t3_k = t3_results['K_final']
t3_ari_clustering = t3_results.get('ari', None)

print(f"T3 clustering: K={t3_k}, ARI={t3_ari_clustering:.4f}" if t3_ari_clustering else f"T3 clustering: K={t3_k}")
print(f"T3 pseudo-labels: {list(t3_cluster_to_pseudo.values())}")

# Incarca datele procesate T3
t3_df = pd.read_csv(T3_PROCESSED_CSV)
known_labels_t3 = BASELINE_LABELS + list(all_known_pseudo_labels)
new_t3_df = t3_df[~t3_df['label'].isin(known_labels_t3)].copy()
new_t3_texts = new_t3_df['content'].tolist()
new_t3_pseudo_labels = new_t3_df['label'].tolist()

print(f"\nT3 new samples for fit_open_world: {len(new_t3_texts)}")
print(f"Pseudo-label distribution: {dict(pd.Series(new_t3_pseudo_labels).value_counts())}")

# Genereaza embeddings pentru T3
print("\nGenerating T3 embeddings...")
X_t3 = get_features(new_t3_texts)

# fit_open_world pentru T3
print("Running fit_open_world for T3...")
for i in range(len(new_t3_texts)):
    x = X_t3[i:i+1]
    y = torch.tensor([new_t3_pseudo_labels[i]], dtype=torch.long)
    classifier.fit_open_world(x, y)
    if i % 100 == 0:
        print(f'\r  fit_open_world: {i}/{len(new_t3_texts)}', end='')
print(f'\r  fit_open_world: {len(new_t3_texts)}/{len(new_t3_texts)} done')

# Actualizeaza pseudo-labels cunoscute
t3_pseudo_set = set(t3_cluster_to_pseudo.values())
all_known_pseudo_labels.update(t3_pseudo_set)

# Mapping pseudo -> GT pentru T3: semantic Hungarian
t3_cluster_keywords = {int(k): v for k, v in t3_results.get('cluster_keywords', {}).items()}
t3_cluster_unique_kw = {int(k): v for k, v in t3_results.get('cluster_unique_keywords', {}).items()}
use_unique_t3 = (len(t3_cluster_unique_kw) > 0 and
                 all(len(v) > 0 for v in t3_cluster_unique_kw.values()))
t3_keywords_to_use = t3_cluster_unique_kw if use_unique_t3 else t3_cluster_keywords
t3_pseudo_ordered = [t3_cluster_to_pseudo[k] for k in sorted(t3_cluster_to_pseudo.keys())]

print(f"\nSemantic Hungarian mapping T3 ({'unique' if use_unique_t3 else 'regular'} keywords)...")
t3_mapping, t3_avg_sim = semantic_hungarian_mapping(
    encoder, tokenizer,
    t3_keywords_to_use,
    t3_pseudo_ordered,
    TEST_3_NEW_LABELS
)
pseudo_to_gt_map.update(t3_mapping)

# Evalueaza pe TEST_3
results_t3 = evaluate_on_test(
    classifier, TEST_3_CSV,
    pseudo_to_gt_map=pseudo_to_gt_map,
    task_name='TEST_3',
    all_known_pseudo_labels=all_known_pseudo_labels,
    baseline_labels=BASELINE_LABELS,
    new_labels_this_task=TEST_1_NEW_LABELS + TEST_2_NEW_LABELS + TEST_3_NEW_LABELS  # cumulative: T1+T2+T3
)


# =========================================================================
# STEP 5: SUMAR FINAL
# =========================================================================

total_time = time.time() - start_time

print('\n' + '=' * 70)
print('SUMAR FINAL PLDA + MOSLD-Bench')
print('=' * 70)

print(f"\n  {'Task':<10} {'Acc':>8} {'F1-Mac':>8} {'F1-W':>8} {'KnownAcc':>10} {'NewAcc':>9} {'ARI':>8}")
print(f"  {'─' * 65}")
for res in [results_t1, results_t2, results_t3]:
    print(f"  {res['task']:<10} "
          f"{res['overall_acc']:>8.4f} "
          f"{res['overall_f1_macro']:>8.4f} "
          f"{res['overall_f1_weighted']:>8.4f} "
          f"{res['known_acc']:>10.4f} "
          f"{res['new_acc']:>9.4f} "
          f"{res['ari']:>8.4f}")

# Forgetting table (din TEST_3)
if results_t3.get('forgetting'):
    print(f"\n  FORGETTING ANALYSIS (TEST_3):")
    print(f"  {'Step':<12} {'Accuracy':>10} {'Samples':>10}")
    print(f"  {'─' * 35}")
    for step_name, step_data in results_t3['forgetting'].items():
        print(f"  {step_name:<12} {step_data['acc']:>10.4f} {step_data['n']:>10}")

print(f"\nTotal time: {total_time:.1f}s")

# Salveaza rezultatele
final_results = {
    'model': MODEL_NAME,
    'feature_size': FEATURE_SIZE,
    'shrinkage': SHRINKAGE,
    'baseline_labels': BASELINE_LABELS,
    'total_time_seconds': float(total_time),
    'pseudo_to_gt_map': {str(k): v for k, v in pseudo_to_gt_map.items()},
    'semantic_mapping': {
        'T1': {'mapping': {str(k): v for k, v in t1_mapping.items()}, 'avg_sim': float(t1_avg_sim)},
        'T2': {'mapping': {str(k): v for k, v in t2_mapping.items()}, 'avg_sim': float(t2_avg_sim)},
        'T3': {'mapping': {str(k): v for k, v in t3_mapping.items()}, 'avg_sim': float(t3_avg_sim)},
    },
    'clustering_ari': {
        'T1': t1_results.get('ari'),
        'T2': t2_results.get('ari'),
        'T3': t3_results.get('ari'),
    },
    'results': {
        'TEST_1': results_t1,
        'TEST_2': results_t2,
        'TEST_3': results_t3,
    }
}

output_path = os.path.join(PLDA_OUTPUT_DIR, 'plda_mosld_results.json')
with open(output_path, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\nRezultate salvate in: {output_path}")
print('=' * 70)