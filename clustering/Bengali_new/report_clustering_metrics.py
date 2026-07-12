#!/usr/bin/env python3
"""
report_clustering_metrics.py — precision / recall / F1 per clasa, DIN pasul de
clusterizare, folosind pkl-urile salvate de pipeline (`test_1/2/3_results.pkl`).

Ideea: la fiecare pas, esantioanele detectate ca OOD sunt clusterizate (K-means).
Fiecare cluster e mapat la clasa lui reala MAJORITARA, apoi tratam maparea ca pe o
predictie si calculam precision/recall/F1 per clasa REALA, doar pe esantioanele
care au intrat in clusterizare. Asta arata cat de bine "nimereste" clusterizarea
fiecare clasa noua.

Sursa datelor: `cluster_gt_distribution` din pkl = tabel de contingenta
  {cluster_id: {true_label: count}} — contine si clasele noi descoperite,
  si eventualele clase cunoscute care s-au scurs in poolul OOD.

Note:
  - Maparea e majority-vote (fiecare cluster -> clasa lui dominanta). Daca doua
    clustere pica pe aceeasi clasa, ambele prezic acea clasa (clustere sparte).
  - O clasa reala fara niciun cluster majoritar pe ea are recall calculabil dar
    precision/predicted = 0 (nu a fost "gasita") — exact semnalul util cand
    K_found < K_true.
  - Agregatele ARI / silhouette / purity / NMI raman raportate ca context.

Usage:
  python report_clustering_metrics.py                       # ./bengali_cl_outputs_1
  python report_clustering_metrics.py -d ../Arabic_new/arabic_cl_outputs_1
  python report_clustering_metrics.py --names 4:state,5:entertainment,...
  python report_clustering_metrics.py -o clustering_prf.json
"""

import os
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)


def load_contingency(d):
    """-> dict {cluster_id(int): {true_label(int): count(int)}}"""
    cgd = d.get("cluster_gt_distribution") or {}
    out = {}
    for c, gts in cgd.items():
        out[int(c)] = {int(g): int(n) for g, n in gts.items()}
    return out


def majority_map(cont):
    """cluster -> clasa reala majoritara"""
    return {c: max(gts.items(), key=lambda kv: kv[1])[0] for c, gts in cont.items() if gts}


def arrays_from_map(cont, cmap):
    """Reconstruieste (y_true, y_pred) unde y_pred = clasa majoritara a clusterului."""
    y_true, y_pred = [], []
    for c, gts in cont.items():
        pred = cmap.get(c)
        for g, n in gts.items():
            y_true.extend([g] * n)
            y_pred.extend([pred] * n)
    return np.array(y_true), np.array(y_pred)


def per_class_prf(y_true, y_pred):
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return labels, p, r, f, s


def cluster_metrics(cont):
    y_true, y_pred = arrays_from_map(cont, majority_map(cont))
    if len(y_true) == 0:
        return None
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "V_measure": float(homogeneity_completeness_v_measure(y_true, y_pred)[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default="./bengali_cl_outputs_1")
    ap.add_argument("--names", default=None,
                    help='mapare label->nume, ex "4:state,5:entertainment"')
    ap.add_argument("-o", "--out", default=None, help="salveaza si JSON")
    args = ap.parse_args()

    names = {}
    if args.names:
        for tok in args.names.split(","):
            k, v = tok.split(":")
            names[int(k)] = v

    def label_name(lbl):
        return f"{lbl} {names.get(lbl, '')}".strip()

    report = {}
    print(f"\nFolder: {os.path.abspath(args.dir)}")

    for step in (1, 2, 3):
        pkl = os.path.join(args.dir, f"test_{step}_results.pkl")
        print(f"\n{'=' * 72}\nT{step}", end="  ")
        if not os.path.exists(pkl):
            print(f"(lipseste {os.path.basename(pkl)})")
            continue
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        cont = load_contingency(d)
        if not cont:
            print("(fara cluster_gt_distribution)")
            continue

        print(f"K_found={d.get('K_final')}  K_true={d.get('K_ground_truth')}  "
              f"ARI={d.get('ari'):.4f}  Silhou={d.get('silhouette'):.4f}  "
              f"Purity={d.get('overall_purity'):.4f}")
        cmap = majority_map(cont)
        # cate clustere au fost mapate pe fiecare clasa (clustere sparte / lipsa)
        clusters_per_class = defaultdict(int)
        for c, g in cmap.items():
            clusters_per_class[g] += 1

        y_true, y_pred = arrays_from_map(cont, cmap)
        labels, p, r, f, s = per_class_prf(y_true, y_pred)

        print(f"\n{'Clasa (label)':<22}{'Prec':>8}{'Recall':>8}{'F1':>8}"
              f"{'Support':>9}{'#clust':>7}")
        print("-" * 62)
        rows = {}
        for lbl, pp, rr, ff, ss in zip(labels, p, r, f, s):
            print(f"{label_name(lbl):<22}{pp:>8.4f}{rr:>8.4f}{ff:>8.4f}"
                  f"{int(ss):>9}{clusters_per_class.get(lbl, 0):>7}")
            rows[lbl] = {"precision": float(pp), "recall": float(rr),
                         "f1": float(ff), "support": int(ss),
                         "clusters_mapped": clusters_per_class.get(lbl, 0)}

        # macro pe clasele efectiv prezente
        print("-" * 62)
        print(f"{'macro avg':<22}{np.mean(p):>8.4f}{np.mean(r):>8.4f}{np.mean(f):>8.4f}")
        cm = cluster_metrics(cont)
        print(f"cluster-level: ARI={cm['ARI']:.4f}  NMI={cm['NMI']:.4f}  "
              f"V-measure={cm['V_measure']:.4f}")

        report[f"T{step}"] = {
            "K_found": d.get("K_final"), "K_true": d.get("K_ground_truth"),
            "ari_stored": d.get("ari"), "silhouette": d.get("silhouette"),
            "purity": d.get("overall_purity"),
            "cluster_level": cm, "per_class": rows,
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSalvat: {args.out}")


if __name__ == "__main__":
    main()
