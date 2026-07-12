import json, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np


class NewsDataset(Dataset):
    def __init__(self, csv_path, model_name, label2id, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.max_len = max_len

        print(f"[NewsDataset] Loaded {len(self.df)} samples from {csv_path}")
        if 'y_soft' in self.df.columns:
            soft_count = self.df['y_soft'].notna().sum()
            hard_count = self.df['y_soft'].isna().sum()
            print(f"[NewsDataset] Hard: {hard_count}, Soft: {soft_count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row["content"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if "y_soft" in self.df.columns and pd.notna(row.get("y_soft", np.nan)):
            dist = torch.zeros(self.num_labels, dtype=torch.float)
            try:
                d = json.loads(row["y_soft"])
                for cls, w in d.items():
                    cls_str = str(cls)
                    if cls_str in self.label2id:
                        dist[self.label2id[cls_str]] = float(w)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Error parsing y_soft at index {i}: {e}")

            if dist.sum() == 0 and pd.notna(row.get("label", np.nan)):
                label_str = str(int(row["label"]))
                if label_str in self.label2id:
                    dist[self.label2id[label_str]] = 1.0

            item["labels"] = dist
        else:
            dist = torch.zeros(self.num_labels, dtype=torch.float)
            label_str = str(int(row["label"]))
            if label_str in self.label2id:
                dist[self.label2id[label_str]] = 1.0
            item["labels"] = dist

        return item


class SoftDataCollator:
    def __call__(self, features):
        batch = {}
        for k in ["input_ids", "attention_mask", "labels"]:
            batch[k] = torch.stack([f[k] for f in features])
        return batch