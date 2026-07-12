#!/usr/bin/env python3
"""
prepare_bengali_cl_splits.py - v3 cu deduplicare și split-uri disjuncte
"""

import os
import argparse
import json
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURAȚIE BENGALI
# ============================================================================

ORIGINAL_LABELS = {
    0: 'politics',
    1: 'sports',
    2: 'technology',
    3: 'lifestyle',
    4: 'state',
    5: 'national',
    6: 'entertainment',
    7: 'editorial',
    8: 'kolkata',
    9: 'international'
}

# Optiunea A: grupul geografic confuzabil (national/international/state/kolkata)
# ramane in baseline (supervizat), iar pasii de descoperire au doar teme distincte.
NEW_ORDER = [
    5,  # 0 - national (Baseline)
    9,  # 1 - international (Baseline)
    4,  # 2 - state (Baseline)
    8,  # 3 - kolkata (Baseline)
    0,  # 4 - politics (T1)
    6,  # 5 - entertainment (T1)
    1,  # 6 - sports (T2)
    2,  # 7 - technology (T2)
    3,  # 8 - lifestyle (T3)
    7,  # 9 - editorial (T3)
]

CLASS_NAMES = [
    'national',       # 0 - Baseline
    'international',  # 1 - Baseline
    'state',          # 2 - Baseline
    'kolkata',        # 3 - Baseline
    'politics',       # 4 - T1
    'entertainment',  # 5 - T1
    'sports',         # 6 - T2
    'technology',     # 7 - T2
    'lifestyle',      # 8 - T3
    'editorial',      # 9 - T3
]

BENGALI_NAMES = {
    'politics': 'রাজনীতি',
    'technology': 'প্রযুক্তি',
    'international': 'আন্তর্জাতিক',
    'national': 'জাতীয়',
    'sports': 'ক্রীড়া',
    'lifestyle': 'জীবনধারা',
    'entertainment': 'বিনোদন',
    'editorial': 'সম্পাদকীয়',
    'kolkata': 'কলকাতা',
    'state': 'রাজ্য',
}

SPLIT_CONFIG = {
    "baseline": [0, 1, 2, 3],
    "T1": [4, 5],
    "T2": [6, 7],
    "T3": [8, 9],
}


def create_label_mapping():
    """old_label -> new_label"""
    mapping = {}
    for new_label, old_label in enumerate(NEW_ORDER):
        mapping[old_label] = new_label
    return mapping


def create_splits_no_overlap(df, output_dir,
                              baseline_train_ratio=0.40,
                              baseline_val_ratio=0.10,
                              random_state=42):
    """
    Creează split-uri FĂRĂ suprapuneri și cu deduplicare.
    """

    np.random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🔧 BENGALI - CREARE SPLITS (v3 - DEDUPLICARE + DISJUNCT)")
    print(f"{'=' * 70}")

    # Detect content column
    if 'content' in df.columns:
        content_col = 'content'
    elif 'text' in df.columns:
        content_col = 'text'
    else:
        raise ValueError("No content column found!")

    # ========================================================================
    # DEDUPLICARE GLOBALĂ
    # ========================================================================

    print(f"\n📊 DEDUPLICARE:")
    before = len(df)
    df = df.drop_duplicates(subset=[content_col], keep='first')
    after = len(df)
    print(f"   Înainte: {before:,} rânduri")
    print(f"   După:    {after:,} rânduri")
    print(f"   Șterse:  {before - after:,} duplicate")

    # Label mapping
    label_mapping = create_label_mapping()
    df['new_label'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['new_label'])
    df['new_label'] = df['new_label'].astype(int)

    # Shuffle global
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    baseline_labels = SPLIT_CONFIG["baseline"]

    # ========================================================================
    # PASUL 1: Împarte fiecare clasă
    # ========================================================================

    train_dfs = []
    val_dfs = []
    test_pool_dfs = []

    print(f"\n📊 Împărțire per clasă:")
    print(f"   Baseline (0-3): train={baseline_train_ratio}, val={baseline_val_ratio}")
    print(f"   T1/T2/T3: doar test")
    print(f"\n{'Label':<6} {'Class':<15} {'Total':>7} {'Train':>7} {'Val':>7} {'TestPool':>8}")
    print("-" * 65)

    for label in range(10):
        class_df = df[df['new_label'] == label].copy()
        class_df = class_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        n = len(class_df)

        if label in baseline_labels:
            train_end = int(n * baseline_train_ratio)
            val_end = int(n * (baseline_train_ratio + baseline_val_ratio))

            train_part = class_df.iloc[:train_end]
            val_part = class_df.iloc[train_end:val_end]
            test_part = class_df.iloc[val_end:]
        else:
            train_part = pd.DataFrame(columns=class_df.columns)
            val_part = pd.DataFrame(columns=class_df.columns)
            test_part = class_df

        train_dfs.append(train_part)
        val_dfs.append(val_part)
        test_pool_dfs.append(test_part)

        marker = "B" if label in baseline_labels else ("1" if label < 6 else ("2" if label < 8 else "3"))
        print(f"{label:<6} {CLASS_NAMES[label]:<15} {n:>7,} {len(train_part):>7,} {len(val_part):>7,} {len(test_part):>8,} [{marker}]")

    # ========================================================================
    # PASUL 2: train.csv și val.csv
    # ========================================================================

    train_baseline = pd.concat([train_dfs[l] for l in baseline_labels], ignore_index=True)
    val_baseline = pd.concat([val_dfs[l] for l in baseline_labels], ignore_index=True)

    train_baseline = train_baseline.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_baseline = val_baseline.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_baseline[[content_col, 'new_label']].rename(
        columns={content_col: 'content', 'new_label': 'label'}
    ).to_csv(os.path.join(output_dir, "train.csv"), index=False)

    val_baseline[[content_col, 'new_label']].rename(
        columns={content_col: 'content', 'new_label': 'label'}
    ).to_csv(os.path.join(output_dir, "val.csv"), index=False)

    print(f"\n✅ train.csv: {len(train_baseline):,}")
    print(f"✅ val.csv:   {len(val_baseline):,}")

    # ========================================================================
    # PASUL 3: Împarte test_pool în 3 părți DISJUNCTE
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(f"📊 TEST POOL → 3 PĂRȚI DISJUNCTE")
    print(f"{'=' * 70}")

    test_1_dfs = []
    test_2_dfs = []
    test_3_dfs = []

    print(f"\n{'Label':<6} {'Class':<15} {'Pool':>7} {'T1':>7} {'T2':>7} {'T3':>7}")
    print("-" * 55)

    for label in range(10):
        pool = test_pool_dfs[label].copy()
        pool = pool.sample(frac=1, random_state=random_state).reset_index(drop=True)

        n = len(pool)
        split1 = int(n / 3)
        split2 = int(2 * n / 3)

        part_1 = pool.iloc[:split1]
        part_2 = pool.iloc[split1:split2]
        part_3 = pool.iloc[split2:]

        test_1_dfs.append(part_1)
        test_2_dfs.append(part_2)
        test_3_dfs.append(part_3)

        print(f"{label:<6} {CLASS_NAMES[label]:<15} {n:>7,} {len(part_1):>7,} {len(part_2):>7,} {len(part_3):>7,}")

    # ========================================================================
    # PASUL 4: Construiește test_1, test_2, test_3
    # ========================================================================

    # test_1: baseline + T1 (labels 0-5)
    test_1_labels = baseline_labels + SPLIT_CONFIG["T1"]
    test_1 = pd.concat([test_1_dfs[l] for l in test_1_labels], ignore_index=True)
    test_1 = test_1.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_1[[content_col, 'new_label']].rename(
        columns={content_col: 'content', 'new_label': 'label'}
    ).to_csv(os.path.join(output_dir, "test_1.csv"), index=False)

    # test_2: baseline + T1 + T2 (labels 0-7)
    test_2_labels = baseline_labels + SPLIT_CONFIG["T1"] + SPLIT_CONFIG["T2"]
    test_2 = pd.concat([test_2_dfs[l] for l in test_2_labels], ignore_index=True)
    test_2 = test_2.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_2[[content_col, 'new_label']].rename(
        columns={content_col: 'content', 'new_label': 'label'}
    ).to_csv(os.path.join(output_dir, "test_2.csv"), index=False)

    # test_3: toate (labels 0-9)
    test_3_labels = list(range(10))
    test_3 = pd.concat([test_3_dfs[l] for l in test_3_labels], ignore_index=True)
    test_3 = test_3.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_3[[content_col, 'new_label']].rename(
        columns={content_col: 'content', 'new_label': 'label'}
    ).to_csv(os.path.join(output_dir, "test_3.csv"), index=False)

    print(f"\n✅ test_1.csv: {len(test_1):,} (known: 0-3, OOD: 4-5)")
    print(f"✅ test_2.csv: {len(test_2):,} (known: 0-5, OOD: 6-7)")
    print(f"✅ test_3.csv: {len(test_3):,} (known: 0-7, OOD: 8-9)")

    # ========================================================================
    # VERIFICARE SUPRAPUNERI
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(f"🔍 VERIFICARE SUPRAPUNERI")
    print(f"{'=' * 70}")

    t1 = pd.read_csv(os.path.join(output_dir, "test_1.csv"))
    t2 = pd.read_csv(os.path.join(output_dir, "test_2.csv"))
    t3 = pd.read_csv(os.path.join(output_dir, "test_3.csv"))
    train = pd.read_csv(os.path.join(output_dir, "train.csv"))
    val = pd.read_csv(os.path.join(output_dir, "val.csv"))

    t1_c = set(t1['content'].tolist())
    t2_c = set(t2['content'].tolist())
    t3_c = set(t3['content'].tolist())
    train_c = set(train['content'].tolist())
    val_c = set(val['content'].tolist())

    checks = [
        ("test_1 ∩ test_2", t1_c & t2_c),
        ("test_1 ∩ test_3", t1_c & t3_c),
        ("test_2 ∩ test_3", t2_c & t3_c),
        ("train ∩ test_1", train_c & t1_c),
        ("train ∩ test_2", train_c & t2_c),
        ("train ∩ test_3", train_c & t3_c),
        ("train ∩ val", train_c & val_c),
        ("val ∩ test_1", val_c & t1_c),
        ("val ∩ test_2", val_c & t2_c),
        ("val ∩ test_3", val_c & t3_c),
    ]

    all_ok = True
    for name, overlap in checks:
        status = "✅" if len(overlap) == 0 else "❌"
        if len(overlap) > 0:
            all_ok = False
        print(f"   {name}: {len(overlap)} {status}")

    # ========================================================================
    # METADATA
    # ========================================================================

    metadata = {
        "dataset": "Bengali L3Cube-IndicNews",
        "version": "v3 - deduplicated, disjoint splits",
        "class_names": CLASS_NAMES,
        "bengali_names": BENGALI_NAMES,
        "split_config": SPLIT_CONFIG,
        "label_mapping_old_to_new": {str(k): v for k, v in create_label_mapping().items()},
        "baseline_train_ratio": baseline_train_ratio,
        "random_state": random_state,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ========================================================================
    # SUMAR FINAL
    # ========================================================================

    print(f"\n{'=' * 85}")
    print(f"📊 SUMAR FINAL")
    print(f"{'=' * 85}")

    print(f"\n{'Class':<15} {'Label':>5} │ {'train':>8} {'val':>8} │ {'test_1':>8} {'test_2':>8} {'test_3':>8}")
    print(f"{'-' * 15} {'-' * 5} │ {'-' * 8} {'-' * 8} │ {'-' * 8} {'-' * 8} {'-' * 8}")

    for idx, cls in enumerate(CLASS_NAMES):
        train_cnt = len(train[train['label'] == idx])
        val_cnt = len(val[val['label'] == idx])
        test1_cnt = len(t1[t1['label'] == idx])
        test2_cnt = len(t2[t2['label'] == idx])
        test3_cnt = len(t3[t3['label'] == idx])

        marker = "B" if idx < 4 else ("1" if idx < 6 else ("2" if idx < 8 else "3"))

        print(f"{cls:<15} {idx:>4}{marker} │ {train_cnt:>8,} {val_cnt:>8,} │ {test1_cnt:>8,} {test2_cnt:>8,} {test3_cnt:>8,}")

    print(f"{'-' * 15} {'-' * 5} │ {'-' * 8} {'-' * 8} │ {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"{'TOTAL':<15} {'':>5} │ {len(train):>8,} {len(val):>8,} │ {len(t1):>8,} {len(t2):>8,} {len(t3):>8,}")

    if all_ok:
        print(f"\n✅ SUCCES! Toate split-urile sunt DISJUNCTE.")
    else:
        print(f"\n❌ ATENȚIE: Există suprapuneri!")

    print(f"\n✅ DONE! Salvat în: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="./bengali_splits")
    parser.add_argument("--train-ratio", "-t", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"📂 Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Loaded {len(df):,} samples")

    create_splits_no_overlap(df, args.output,
                              baseline_train_ratio=args.train_ratio,
                              random_state=args.seed)


if __name__ == "__main__":
    main()