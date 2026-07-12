
from transformers import AutoModelForSequenceClassification, TrainingArguments
from data_loader import NewsDataset
from contrastive_trainer import ContrastiveTrainer
import pandas as pd

print("="*70)
print("CONTINUAL LEARNING T1 - TRAINING CONTRASTIVE (CU MAPARE AUTOMATĂ)")
print("="*70)

label2id_cl = {
    "0": 0, "2": 1, "7": 2, "12": 3,
    "13": 4, "14": 5, "15": 6
}
id2label_cl = {v: k for k, v in label2id_cl.items()}
num_labels = 7

LABEL_TO_CLUSTER = {
    4: 0,  # pseudo 13 → cluster 0
    5: 1,  # pseudo 14 → cluster 1
    6: 2   # pseudo 15 → cluster 2
}

print(f"Clase totale: {num_labels}")
print(f"Label to cluster: {LABEL_TO_CLUSTER}")

# Dataset
cl_df = pd.read_csv('./cl_train_t1_contrastive.csv')
print(f"\nDataset: {len(cl_df)} examples")

model_cl = AutoModelForSequenceClassification.from_pretrained(
    "/home/alin/Desktop/ContinualLearning/clustering/ckpt_baseline/final",
    num_labels=num_labels,
    id2label=id2label_cl,
    label2id=label2id_cl,
    ignore_mismatched_sizes=True
)

print(f"✓ Model: 4 clase → {num_labels} clase")

cl_train_ds = NewsDataset(
    './cl_train_t1_contrastive.csv',
    "bert-base-multilingual-cased",
    label2id_cl,
    max_len=256
)

# Training arguments
args = TrainingArguments(
    output_dir="./ckpt_cl_t1_contrastive",
    per_device_train_batch_size=16,
    learning_rate=5e-6,
    num_train_epochs=5,
    warmup_ratio=0.2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=50,
    seed=42,
)

print("\nHyperparametri:")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Epochs: {args.num_train_epochs}")
print(f"  Contrastive weight: 0.5")
print(f"  Temperature: 0.5")

trainer = ContrastiveTrainer(
    model=model_cl,
    args=args,
    train_dataset=cl_train_ds,
    keyword_embeddings_path='keyword_embeddings_t1_contrastive.pkl',
    label_to_cluster_map=LABEL_TO_CLUSTER,
    contrastive_weight=0.5,
    temperature=0.5
)

print("\n" + "="*70)
print("START TRAINING CONTRASTIVE")
print("="*70)

trainer.train()

trainer.save_model("./ckpt_cl_t1_contrastive/final")
print("\n✓ Model salvat: ./ckpt_cl_t1_contrastive/final")

train_results = trainer.state.log_history
final_loss = [x['loss'] for x in train_results if 'loss' in x][-1]
print(f"✓ Final training loss: {final_loss:.4f}")

print("\n" + "="*70)
print("TRAINING CONTRASTIVE COMPLET!")
print("="*70)
