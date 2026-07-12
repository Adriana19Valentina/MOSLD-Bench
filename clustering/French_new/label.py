#!/usr/bin/env python3
"""
Script pentru verificarea numărului de labels diferite în fișierele CSV.
"""

import pandas as pd
from pathlib import Path

files = ["/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/train.csv",
         "/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/val.csv",
         "/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/test_1.csv",
         "/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/test_2.csv",
         "/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2/test_3.csv"]

all_labels = set()

print("=" * 60)
print("Verificare labels în fișierele CSV")
print("=" * 60)

for file in files:
    if Path(file).exists():
        df = pd.read_csv(file)

        # Încearcă să găsească coloana de labels (label, labels, class, category, etc.)
        label_col = None
        for col in ["label", "labels", "class", "category", "target"]:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            # Dacă nu găsește, presupune că ultima coloană este label-ul
            label_col = df.columns[-1]

        unique_labels = df[label_col].unique()
        num_labels = len(unique_labels)

        print(f"\n📄 {file}:")
        print(f"   Coloană folosită: '{label_col}'")
        print(f"   Număr rânduri: {len(df)}")
        print(f"   Labels unice: {num_labels}")
        print(f"   Labels: {sorted(unique_labels)}")

        all_labels.update(unique_labels)
    else:
        print(f"\n❌ {file}: FIȘIER NEGĂSIT")

print("\n" + "=" * 60)
print(f"📊 SUMAR TOTAL:")
print(f"   Total labels unice (toate fișierele): {len(all_labels)}")
print(f"   Lista completă: {sorted(all_labels)}")
print("=" * 60)