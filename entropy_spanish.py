import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

# --- SETUP DEVICE ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS (Metal GPU) backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA backend")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5

# --- LOAD DATA ---
def load_data(path):
    return pd.read_csv(path)

# Correct PathF
base_path = "/home/alin/Desktop/ContinualLearning/datasets/Spanish"

train_df = load_data(os.path.join(base_path, "train.csv"))
val_df = load_data(os.path.join(base_path, "val.csv"))
test1_df = load_data(os.path.join(base_path, "test_1.csv"))
test2_df = load_data(os.path.join(base_path, "test_2.csv"))
test3_df = load_data(os.path.join(base_path, "test_3.csv"))

# --- LABEL PROCESSING (FIXED) ---
# 1. Gather all unique labels
raw_labels = sorted(
    list(set(train_df["label"].unique()) | 
         set(test1_df["label"].unique()) | 
         set(test2_df["label"].unique()) | 
         set(test3_df["label"].unique()))
)

# 2. Convert Numpy types to Python native types (Fixes JSON serialization error)
all_labels = []
for l in raw_labels:
    if hasattr(l, "item"):
        all_labels.append(l.item())  # Converts numpy.int64 -> int
    else:
        all_labels.append(l)

# 3. Create Mappings
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for i, l in enumerate(all_labels)}

# Known labels (Converted to native types if necessary)
known_labels_raw = set(train_df["label"].unique())
known_ids_set = set()
for l in known_labels_raw:
    val = l.item() if hasattr(l, "item") else l
    if val in label2id:
        known_ids_set.add(label2id[val])

print(f"Number of labels in total: {len(all_labels)}")
print(f"Number of KNOWN labels: {len(known_ids_set)}")

# --- DATASET CLASS ---
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.texts = df["content"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Ensure label matches the key type in label2id
        raw_label = self.labels[idx]
        label_key = raw_label.item() if hasattr(raw_label, "item") else raw_label
        
        label_id = self.label2id[label_key]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label_id, dtype=torch.long)
        return item

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = NewsDataset(train_df, tokenizer, label2id)
val_dataset = NewsDataset(val_df, tokenizer, label2id)
test1_dataset = NewsDataset(test1_df, tokenizer, label2id)
test2_dataset = NewsDataset(test2_df, tokenizer, label2id)
test3_dataset = NewsDataset(test3_df, tokenizer, label2id)

# --- MODEL INITIALIZATION ---
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(all_labels),
    label2id=label2id,
    id2label=id2label,
    problem_type="single_label_classification"
).to(device)

# --- METRICS FUNCTION ---
def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    preds = probs.argmax(-1)

    # Binary: 1 = Known, 0 = Unknown
    y_true_is_known = np.array([1 if lbl in known_ids_set else 0 for lbl in labels])
    y_score_confidence = probs.max(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(y_true_is_known, y_score_confidence)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_seen_vs_unknown": auc
    }

# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- TRAIN ---
trainer.train()

print("\n--- Validation Results ---")
print(trainer.evaluate(val_dataset))

# --- TEST FUNCTION ---
def test_model(dataset, name, temp_scaling=2.5):
    print(f"\n{'='*20}\nTesting on {name}\n{'='*20}")
    
    # 1. Standard Metrics
    metrics = trainer.evaluate(dataset)
    print(f"--- Standard Metrics ---")
    print(f"Accuracy:  {metrics['eval_accuracy']:.4f}")
    print(f"F1 Score:  {metrics['eval_f1']:.4f}")
    print(f"Precision: {metrics['eval_precision']:.4f}")
    print(f"Recall:    {metrics['eval_recall']:.4f}")
    print(f"AUC (K vs U): {metrics['eval_auc_seen_vs_unknown']:.4f}")

    # 2. Entropy Analysis
    raw_pred = trainer.predict(dataset)
    logits = torch.tensor(raw_pred.predictions)
    labels = raw_pred.label_ids

    # Temperature Scaling
    probs = torch.softmax(logits / temp_scaling, dim=1).numpy()
    entropy_scores = entropy(probs, axis=1)

    # 1 = Unknown, 0 = Known (for this specific entropy analysis where High Entropy = Unknown)
    is_unknown_truth = np.array([1 if lbl not in known_ids_set else 0 for lbl in labels])
    
    total = len(labels)
    n_unknown = np.sum(is_unknown_truth)
    n_known = total - n_unknown
    print(f"\n--- Open Set Counts ---")
    print(f"Total: {total} | Known: {n_known} | Unknown: {n_unknown}")

    if n_unknown > 0 and n_known > 0:
        ent_known = entropy_scores[is_unknown_truth == 0]
        ent_unknown = entropy_scores[is_unknown_truth == 1]
        
        print(f"Avg Entropy (Known):   {np.mean(ent_known):.4f}")
        print(f"Avg Entropy (Unknown): {np.mean(ent_unknown):.4f}")
        
        auc_ent = roc_auc_score(is_unknown_truth, entropy_scores)
        print(f"Entropy AUC: {auc_ent:.4f}")

        # Plots
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(is_unknown_truth, entropy_scores)
        plt.plot(fpr, tpr, label=f"Entropy (AUC={auc_ent:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC: Unknown Detection ({name})")
        plt.legend()
        plt.savefig(os.path.join(base_path, f"{name}_roc.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(ent_known, bins=30, alpha=0.5, label='Known', density=True)
        plt.hist(ent_unknown, bins=30, alpha=0.5, label='Unknown', density=True)
        plt.title(f"Entropy Distribution (T={temp_scaling})")
        plt.legend()
        plt.savefig(os.path.join(base_path, f"{name}_hist.png"))
        plt.close()

    # Confusion Matrix
    preds = probs.argmax(-1)
    if len(all_labels) < 50:
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=(10, 10))
        disp.plot(cmap="Blues", xticks_rotation=90, ax=plt.gca())
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(os.path.join(base_path, f"{name}_conf_mat.png"))
        plt.close()

test_model(test1_dataset, "Test_1", temp_scaling=2.5)
test_model(test2_dataset, "Test_2", temp_scaling=2.5)
test_model(test3_dataset, "Test_3", temp_scaling=2.5)