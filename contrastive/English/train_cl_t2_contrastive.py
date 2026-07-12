# train_cl_t2_contrastive.py - Training test_2 contrastive

from transformers import AutoModelForSequenceClassification, TrainingArguments
from data_loader import NewsDataset
from contrastive_trainer import ContrastiveTrainer
import pandas as pd

print("="*70)
print("CONTINUAL LEARNING T2 - TRAINING CONTRASTIVE")
print("="*70)

# Label mapping (10 clase)
label2id_cl_t2 = {
    "0": 0, "2": 1, "7": 2, "12": 3,      # original
    "13": 4, "14": 5, "15": 6,             # test_1
    "16": 7, "17": 8, "18": 9              # test_2
}
id2label_cl_t2 = {v: k for k, v in label2id_cl_t2.items()}
num_labels = 10

# Mapare label → cluster (pentru discovered classes test_2)
LABEL_TO_CLUSTER = {
    7: 0,  # pseudo 16 → cluster 0
    8: 1,  # pseudo 17 → cluster 1
    9: 2   # pseudo 18 → cluster 2
}

print(f"Clase totale: {num_labels}")
print(f"Label to cluster (T2): {LABEL_TO_CLUSTER}")

# Dataset
cl_df = pd.read_csv('./cl_train_t2_contrastive.csv')
print(f"\nDataset: {len(cl_df)} examples")

# Model (pornind de la T1)
model_cl_t2 = AutoModelForSequenceClassification.from_pretrained(
    "./ckpt_cl_t1_contrastive/final",
    num_labels=num_labels,
    id2label=id2label_cl_t2,
    label2id=label2id_cl_t2,
    ignore_mismatched_sizes=True
)

print(f"✓ Model: 7 clase → {num_labels} clase")

# Dataset
cl_train_ds = NewsDataset(
    './cl_train_t2_contrastive.csv',
    "bert-base-multilingual-cased",
    label2id_cl_t2,
    max_len=256
)

# Training arguments
args = TrainingArguments(
    output_dir="./ckpt_cl_t2_contrastive",
    per_device_train_batch_size=16,
    learning_rate=5e-6,
    num_train_epochs=5,
    warmup_ratio=0.2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=50,
    seed=42,
)

# Contrastive Trainer
trainer = ContrastiveTrainer(
    model=model_cl_t2,
    args=args,
    train_dataset=cl_train_ds,
    keyword_embeddings_path='keyword_embeddings_t2_contrastive.pkl',
    label_to_cluster_map=LABEL_TO_CLUSTER,
    contrastive_weight=0.5,
    temperature=0.5
)

print("\n" + "="*70)
print("START TRAINING T2 CONTRASTIVE")
print("="*70)

trainer.train()

trainer.save_model("./ckpt_cl_t2_contrastive/final")
print("\n✓ Model salvat: ./ckpt_cl_t2_contrastive/final")

train_results = trainer.state.log_history
final_loss = [x['loss'] for x in train_results if 'loss' in x][-1]
print(f"✓ Final training loss: {final_loss:.4f}")

print("\n" + "="*70)
print("TRAINING T2 CONTRASTIVE COMPLET!")
print("="*70)