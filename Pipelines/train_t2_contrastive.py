import os
import sys
import json
import torch
import pickle
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

from config import *
from contrastive_trainer import ContrastiveTrainer
from generate_keyword_embeddings import generate_keyword_embeddings

MODEL_PREV_PATH = MODEL_T1_DIR
PROCESSED_DATA_PATH = os.path.join(EXISTING_RESULTS_DIR, 'test_2_processed.csv')
CLUSTERING_RESULTS_PATH = os.path.join(EXISTING_RESULTS_DIR, 'test_2_results.pkl')
KEYWORD_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'keyword_embeddings_t2.pkl')
MODEL_SAVE_PATH = MODEL_T2_DIR


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }


def train():
    print("=" * 70)
    print("T2 TRAINING WITH CONTRASTIVE LOSS")
    print("=" * 70)
    print(f"Loss = CE + {CONTRASTIVE_WEIGHT} x Contrastive")

    print(f"\n{'=' * 60}")
    print("STEP 1: LOADING T1 MODEL")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PREV_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PREV_PATH)

    print(f"  T1 model: {MODEL_PREV_PATH}")
    print(f"  Classes: {model.config.num_labels}")

    old_num_labels = model.config.num_labels
    old_id2label = {int(k): int(v) for k, v in model.config.id2label.items()}
    old_label2id = {int(v): int(k) for k, v in model.config.id2label.items()}

    print(f"\n{'=' * 60}")
    print("STEP 2: LOADING CLUSTERING RESULTS")
    print("=" * 60)

    with open(CLUSTERING_RESULTS_PATH, 'rb') as f:
        clustering_results = pickle.load(f)

    cluster_to_pseudo = clustering_results.get('cluster_to_pseudo', {})
    print(f"  Clusters: {clustering_results.get('K_final')}")
    print(f"  Cluster -> Pseudo: {cluster_to_pseudo}")

    t2_pseudo_labels = sorted(cluster_to_pseudo.values())
    print(f"  T2 new pseudo-labels: {t2_pseudo_labels}")

    print(f"\n{'=' * 60}")
    print("STEP 3: LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"  Samples: {len(df)}")

    print(f"\n{'=' * 60}")
    print("STEP 4: EXPANDING MODEL")
    print("=" * 60)

    new_num_labels = old_num_labels + len(t2_pseudo_labels)
    print(f"  {old_num_labels} -> {new_num_labels} classes")

    old_classifier = model.classifier
    in_features = old_classifier.in_features
    new_classifier = torch.nn.Linear(in_features, new_num_labels)

    with torch.no_grad():
        new_classifier.weight[:old_num_labels] = old_classifier.weight
        new_classifier.bias[:old_num_labels] = old_classifier.bias
        torch.nn.init.xavier_uniform_(new_classifier.weight[old_num_labels:])
        new_classifier.bias[old_num_labels:].zero_()

    model.classifier = new_classifier
    model.config.num_labels = new_num_labels

    new_id2label = {int(k): int(v) for k, v in old_id2label.items()}
    new_label2id = {int(k): int(v) for k, v in old_label2id.items()}

    for i, pseudo in enumerate(t2_pseudo_labels):
        model_id = int(old_num_labels + i)
        new_id2label[model_id] = int(pseudo)
        new_label2id[int(pseudo)] = model_id

    model.config.id2label = new_id2label
    model.config.label2id = new_label2id
    print(f"  id2label: {new_id2label}")

    print(f"\n{'=' * 60}")
    print("STEP 5: PREPARING DATASET")
    print("=" * 60)

    dataset = Dataset.from_pandas(df[['content', 'label']])

    def tokenize_fn(examples):
        return tokenizer(examples['content'], padding='max_length',
                         truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['content'])
    tokenized = tokenized.rename_column('label', 'labels')

    def remap(example):
        if example['labels'] in new_label2id:
            example['labels'] = new_label2id[example['labels']]
        return example

    tokenized = tokenized.map(remap)
    tokenized.set_format('torch')
    print(f"   {len(tokenized)} samples prepared")

    print(f"\n{'=' * 60}")
    print("STEP 6: KEYWORD EMBEDDINGS")
    print("=" * 60)

    if not os.path.exists(KEYWORD_EMBEDDINGS_PATH):
        generate_keyword_embeddings(
            results_path=CLUSTERING_RESULTS_PATH,
            model_name=MODEL_NAME,
            output_path=KEYWORD_EMBEDDINGS_PATH
        )
    else:
        print(f"   Already exists: {KEYWORD_EMBEDDINGS_PATH}")

    label_to_cluster = {v: int(k) for k, v in cluster_to_pseudo.items()}
    label_to_cluster_model = {}
    for pseudo, cluster in label_to_cluster.items():
        if pseudo in new_label2id:
            label_to_cluster_model[new_label2id[pseudo]] = cluster

    print(f"  Label (model_id) -> Cluster: {label_to_cluster_model}")

    print(f"\n{'=' * 60}")
    print("STEP 7: TRAINING")
    print("=" * 60)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=torch.cuda.is_available(),
        report_to='none',
        seed=SEED
    )

    split = tokenized.train_test_split(test_size=0.1, seed=SEED)

    trainer = ContrastiveTrainer(
        keyword_embeddings_path=KEYWORD_EMBEDDINGS_PATH,
        label_to_cluster_map=label_to_cluster_model,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        temperature=TEMPERATURE,
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")
    print(f"\n Starting training...")

    train_result = trainer.train()

    print(f"\n{'=' * 60}")
    print("STEP 8: SAVING MODEL")
    print("=" * 60)

    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    info = {
        'step': 'T2',
        'contrastive_weight': CONTRASTIVE_WEIGHT,
        'temperature': TEMPERATURE,
        'label_to_cluster': label_to_cluster_model,
        'id2label': new_id2label,
        'label2id': new_label2id,
        'loss_summary': trainer.get_loss_summary(),
        'train_loss': train_result.training_loss
    }

    with open(os.path.join(MODEL_SAVE_PATH, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2, default=str)

    print(f"  Saved to: {MODEL_SAVE_PATH}")

    print(f"\n{'=' * 60}")
    print("T2 CONTRASTIVE TRAINING COMPLETED!")
    print("=" * 60)

    loss = trainer.get_loss_summary()
    print(f"  Avg CE: {loss['avg_ce']:.4f}")
    print(f"  Avg Contrastive: {loss['avg_contrastive']:.4f}")


if __name__ == '__main__':
    train()
