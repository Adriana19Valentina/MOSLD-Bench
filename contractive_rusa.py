import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True) 

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS (Metal GPU) backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA backend")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5
LAMBDA_CONTRACTIVE = 0.1  # Weight for the contractive penalty (hyperparameter)

def load_data(path):
    return pd.read_csv(path)

base_path = "/home/alin/Desktop/ContinualLearning/datasets/Russian/Russian_balanced"
train_df = load_data(os.path.join(base_path, "train.csv"))
val_df = load_data(os.path.join(base_path, "val.csv"))
test1_df = load_data(os.path.join(base_path, "test_1.csv"))
test2_df = load_data(os.path.join(base_path, "test_2.csv"))
test3_df = load_data(os.path.join(base_path, "test_3.csv"))


known_labels = set(train_df["label"].unique())
all_labels = set(train_df["label"].unique()) | set(test1_df["label"].unique()) | \
             set(test2_df["label"].unique()) | set(test3_df["label"].unique())

raw_labels_list = sorted(list(all_labels))

labels_list = []
for l in raw_labels_list:
    if isinstance(l, (np.integer, np.floating)):
        labels_list.append(int(l))  
    else:
        labels_list.append(l)

label2id = {l: i for i, l in enumerate(labels_list)}
id2label = {i: l for i, l in enumerate(labels_list)}
print(f"Known (Train) Labels: {len(known_labels)}")
print(f"Total Labels (Open Set): {len(labels_list)}")


model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification",
).to(device)

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
        label_name = self.labels[idx]
        label_id = self.label2id[label_name]
        
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


model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_list),
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification"
).to(device)


class ContractiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")
        labels = inputs.get("labels")

        embedding_layer = model.bert.embeddings
        inputs_embeds = embedding_layer(
            input_ids=input_ids, 
            token_type_ids=token_type_ids
        )

        if model.training:
            noise = torch.randn_like(inputs_embeds) * 0.05
            inputs_embeds = inputs_embeds + noise.to(inputs_embeds.device)
            
            inputs_embeds.retain_grad()

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        ce_loss = outputs.loss
        
        if model.training and torch.is_grad_enabled() and inputs_embeds.grad_fn is not None:
            # penalty =the norm of the Jacobian of the output 
            # J_DCN = L + lambda * || dy/dx ||
        
            logits = outputs.logits
            batch_size = logits.shape[0]
            
            if labels is not None:
                relevant_logits = logits[torch.arange(batch_size), labels]
            else:
                relevant_logits = torch.max(logits, dim=1)[0]

            grads = torch.autograd.grad(
                relevant_logits.sum(), 
                inputs_embeds, 
                create_graph=True, 
                retain_graph=True, 
                only_inputs=True
            )[0]
            
            penalty = grads.view(batch_size, -1).norm(2, dim=1).mean()
            
            total_loss = ce_loss + (LAMBDA_CONTRACTIVE * penalty)
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss

def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    preds = probs.argmax(-1)

    known_ids = {label2id[l] for l in known_labels}
    
    y_true_bin = np.array([1 if lbl in known_ids else 0 for lbl in labels])
    
    # Confidence score (max probability) is the standard baseline for OOD detection
    y_score_bin = probs.max(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    try:
        # AUC for detecting Known vs Unknown
        auc = roc_auc_score(y_true_bin, y_score_bin)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "auc_known_vs_new": auc
    }

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_contractive"),
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
    save_total_limit=2,
)

trainer = ContractiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("Validation Results:")
metrics_val = trainer.evaluate(val_dataset)
print(metrics_val)

# TEST + PLOTS
def test_model(dataset, name):
    print(f"\n--- Testing {name} ---")
    results = trainer.evaluate(dataset)
    print(f"Metrics: {results}")

    # Predictions
    raw_pred = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(raw_pred.predictions), dim=1).numpy()
    preds = probs.argmax(-1)
    labels_ids = raw_pred.label_ids

    known_ids = {label2id[l] for l in known_labels}

    # Binary: 1 = Known Class, 0 = New Class
    y_true_is_known = np.array([1 if lbl in known_ids else 0 for lbl in labels_ids])
    y_score_confidence = probs.max(axis=1)

    # Unseen stats
    unseen_count = np.sum(y_true_is_known == 0)
    total = len(labels_ids)
    print(f"Unseen/New classes in set: {unseen_count}/{total} ({unseen_count/total:.2%})")

    # ---- ROC Curve (Known vs New) ----
    if len(np.unique(y_true_is_known)) > 1:
        fpr, tpr, _ = roc_curve(y_true_is_known, y_score_confidence)
        auc = roc_auc_score(y_true_is_known, y_score_confidence)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], '--', color="gray")
        plt.title(f"ROC: Detecting Known vs New ({name})")
        plt.xlabel("False Positive Rate (New classified as Known)")
        plt.ylabel("True Positive Rate (Known classified as Known)")
        plt.legend()
        save_path = os.path.join(base_path, "figs", f"{name}_roc.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        print("ROC curve skipped (Dataset contains only Known or only New classes).")

    # ---- Confusion Matrix ----
    if len(labels_list) < 50:
        cm = confusion_matrix(labels_ids, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(cmap="Blues", ax=ax, xticks_rotation=90)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "figs", f"{name}_conf_mat.png"))

test_model(test1_dataset, "Test_1")
test_model(test2_dataset, "Test_2")
test_model(test3_dataset, "Test_3")