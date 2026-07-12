import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Determine device (0 for CUDA, -1 for CPU/MPS)
if torch.cuda.is_available():
    device_id = 0
    print("✅ Using device: CUDA")
elif torch.backends.mps.is_available():
    device_id = "mps"
    print("✅ Using device: MPS")
else:
    device_id = -1
    print("✅ Using device: CPU")

MODEL_NAME = "facebook/bart-large-mnli"

# Define Output Directory for Images
OUTPUT_DIR = "english_bart_mnli_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# DATASET CONFIGURATION (ENGLISH)
# =========================================================================
DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/English/compute/english_splits'

TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_1_CSV = os.path.join(DATASET_DIR, 'test_1.csv')
TEST_2_CSV = os.path.join(DATASET_DIR, 'test_2.csv')
TEST_3_CSV = os.path.join(DATASET_DIR, 'test_3.csv')

CLASS_NAMES = {
    0: 'Company',
    1: 'Athlete',
    2: 'MeanOfTransportation',
    3: 'NaturalPlace',
    4: 'Animal',
    5: 'Album',
    6: 'Building',
    7: 'Plant',
    8: 'Film',
    9: 'WrittenWork',
    10: 'Artist',
    11: 'Village',
    12: 'EducationalInstitution',
    13: 'OfficeHolder'
}

BASELINE_LABELS = [0, 1, 2, 3]
TEST_1_NEW_LABELS = [4, 5, 6]
TEST_2_NEW_LABELS = [7, 8, 9]
TEST_3_NEW_LABELS = [10, 11, 12, 13]

TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": TEST_1_CSV,
        "known_ids": BASELINE_LABELS + TEST_1_NEW_LABELS 
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": TEST_2_CSV,
        "known_ids": BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": TEST_3_CSV,
        "known_ids": BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS + TEST_3_NEW_LABELS
    }
]

print(f"Loading {MODEL_NAME} Zero-Shot Pipeline...")
# The pipeline handles tokenization, model loading, and NLI logic automatically
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=device_id
)

# --- BATCH PREDICTION FUNCTION USING PIPELINE ---
def predict_bart_mnli_batch(texts, candidate_labels_text, batch_size=16):
    all_pred_indices = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying Batches"):
        batch_texts = texts[i : i + batch_size]
        
        # Run the zero-shot pipeline (multi_label=False ensures softmax across all labels)
        results = classifier(batch_texts, candidate_labels_text, multi_label=False)
        
        # If batch_size=1, the pipeline returns a single dict instead of a list
        if isinstance(results, dict):
            results = [results]
            
        for res in results:
            # The top prediction is always the first item in the 'labels' list
            top_label_str = res['labels'][0]
            
            # Map the predicted string back to its index in our current candidate list
            idx = candidate_labels_text.index(top_label_str)
            all_pred_indices.append(idx)
            
    return all_pred_indices


def evaluate_step(step_config, previous_known_ids):
    name = step_config["name"]
    path = step_config["path"]
    known_ids = step_config["known_ids"]
    step_id = step_config["id"]
    
    old_classes = [cls for cls in known_ids if cls in previous_known_ids]
    new_classes = [cls for cls in known_ids if cls not in previous_known_ids]
    
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"Known (Base/Old) Classes : {old_classes}")
    print(f"Unknown (New) Classes    : {new_classes}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None

    # Get string names for the labels active in this step
    candidate_names = [CLASS_NAMES[i] for i in known_ids]
    true_labels = df['label'].tolist()
    texts = df['content'].tolist()
    
    # Get predictions via BART-MNLI
    relative_indices = predict_bart_mnli_batch(texts, candidate_names, batch_size=16)
    
    # Convert relative indices back to global class IDs
    predicted_ids = [known_ids[idx] for idx in relative_indices]

    # --- Metrics Calculation ---
    overall_acc = accuracy_score(true_labels, predicted_ids)
    overall_f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)
    
    old_acc, old_f1, new_acc, new_f1 = None, None, None, None

    print(f"\n📊 Results for {name}:")
    print(f"  ➜ Overall Acc : {overall_acc:.4f} | Overall F1 : {overall_f1:.4f}")

    if old_classes:
        old_indices = [i for i, label in enumerate(true_labels) if label in old_classes]
        true_old = [true_labels[i] for i in old_indices]
        pred_old = [predicted_ids[i] for i in old_indices]
        if true_old:
            old_acc = accuracy_score(true_old, pred_old)
            old_f1 = f1_score(true_old, pred_old, average='weighted', labels=old_classes, zero_division=0)
            print(f"  ➜ Known Acc   : {old_acc:.4f} | Known F1   : {old_f1:.4f}")

    if new_classes:
        new_indices = [i for i, label in enumerate(true_labels) if label in new_classes]
        true_new = [true_labels[i] for i in new_indices]
        pred_new = [predicted_ids[i] for i in new_indices]
        if true_new:
            new_acc = accuracy_score(true_new, pred_new)
            new_f1 = f1_score(true_new, pred_new, average='weighted', labels=new_classes, zero_division=0)
            print(f"  ➜ Unknown Acc : {new_acc:.4f} | Unknown F1 : {new_f1:.4f}")

    # --- Plotting Confusion Matrix ---
    fixed_labels = [CLASS_NAMES[i] for i in known_ids]
    
    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', # Changed to Blues to differentiate from DAMO results
                xticklabels=fixed_labels,
                yticklabels=fixed_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (BART-MNLI) - {name}\nOverall F1: {overall_f1:.2f}')
    
    save_path = os.path.join(OUTPUT_DIR, f"bart_mnli_cm_step_{step_id}.png")
    plt.savefig(save_path)
    plt.close()

    return {
        "Step": name,
        "Total Samples": len(texts),
        "Overall Acc": round(overall_acc, 4),
        "Overall F1": round(overall_f1, 4),
        "Known Acc": round(old_acc, 4) if old_acc is not None else "N/A",
        "Known F1": round(old_f1, 4) if old_f1 is not None else "N/A",
        "Unknown Acc": round(new_acc, 4) if new_acc is not None else "N/A",
        "Unknown F1": round(new_f1, 4) if new_f1 is not None else "N/A"
    }


if __name__ == "__main__":
    all_results = []
    
    previous_known_ids = BASELINE_LABELS.copy()
    
    for step in TEST_SCHEDULE:
        current_classes = step["known_ids"]
        new_classes_in_step = [cls for cls in current_classes if cls not in previous_known_ids]
        
        step_metrics = evaluate_step(step, previous_known_ids)
        if step_metrics:
            all_results.append(step_metrics)
            
        previous_known_ids.extend(new_classes_in_step)

    print("\n\n" + "="*80)
    print("📊 FINAL ZERO-SHOT EXPERIMENT SUMMARY (ENGLISH BART-MNLI)")
    print("="*80)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        
        csv_save_path = os.path.join(OUTPUT_DIR, "english_bart_mnli_summary.csv")
        results_df.to_csv(csv_save_path, index=False)
        print("\n" + "="*80)
        print(f"💾 Saved full summary table to: {csv_save_path}")