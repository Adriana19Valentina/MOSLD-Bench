import pandas as pd
import torch
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cpu')
print(f"✅ Using device: {device}")

MODEL_NAME = "DAMO-NLP-SG/zero-shot-classify-SSTuning-XLM-R"

# Define Output Directory for Images
OUTPUT_DIR = "spanish_cl_zero_shot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# DATASET CONFIGURATION (SPANISH)
# =========================================================================
DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/Spanish'

TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_1_CSV = os.path.join(DATASET_DIR, 'test_1.csv')
TEST_2_CSV = os.path.join(DATASET_DIR, 'test_2.csv')
TEST_3_CSV = os.path.join(DATASET_DIR, 'test_3.csv')

CLASS_NAMES = {
    0: 'cultura',
    1: 'economía',
    2: 'deporte',
    3: 'internacional',   # Baseline
    4: 'política',        # T1 New
    5: 'ciencia',         # T1 New
    6: 'opinión',         # T2 New
    7: 'impactante',      # T2 New
    8: 'tecnología',      # T3 New
    9: 'salud'            # T3 New
}

# Classes in train.csv (baseline)
BASELINE_LABELS = [0, 1, 2, 3]

TEST_1_NEW_LABELS = [4, 5]
TEST_2_NEW_LABELS = [6, 7]
TEST_3_NEW_LABELS = [8, 9]

# Dynamically building the schedule
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": TEST_1_CSV,
        # Baseline + New in Test 1 -> [0, 1, 2, 3, 4, 5]
        "known_ids": BASELINE_LABELS + TEST_1_NEW_LABELS 
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": TEST_2_CSV,
        # Baseline + Test 1 + New in Test 2 -> [0, 1, 2, 3, 4, 5, 6, 7]
        "known_ids": BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": TEST_3_CSV,
        # All classes up to Test 3 -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        "known_ids": BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS + TEST_3_NEW_LABELS
    }
]

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

list_ABC = [x for x in string.ascii_uppercase]

# --- BATCH PREDICTION FUNCTION ---
def predict_sstuning_batch(texts, candidate_labels_text, batch_size=16):
    formatted_labels = [x + '.' if x[-1] != '.' else x for x in candidate_labels_text]
    padded_labels = formatted_labels + [tokenizer.pad_token] * (20 - len(formatted_labels))
    s_option = ' '.join([f'({list_ABC[i]}) {padded_labels[i]}' for i in range(len(padded_labels))])
    
    full_inputs = [f'{s_option} {tokenizer.sep_token} {t}' for t in texts]
    all_pred_indices = []
    
    for i in tqdm(range(0, len(full_inputs), batch_size), desc="Classifying Batches"):
        batch = full_inputs[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        valid_logits = logits[:, 0:len(formatted_labels)]
        batch_preds = torch.argmax(valid_logits, dim=-1).tolist()
        
        if isinstance(batch_preds, int):
            batch_preds = [batch_preds]
            
        all_pred_indices.extend(batch_preds)
        
    return all_pred_indices


def evaluate_step(step_config, previous_known_ids):
    name = step_config["name"]
    path = step_config["path"]
    known_ids = step_config["known_ids"]
    step_id = step_config["id"]
    
    # Identify Known (Old) vs Unknown (New) classes for this specific step
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

    candidate_names = [CLASS_NAMES[i] for i in known_ids]
    true_labels = df['label'].tolist()
    texts = df['content'].tolist()
    
    # Get batched predictions
    relative_indices = predict_sstuning_batch(texts, candidate_names, batch_size=16)
    predicted_ids = [known_ids[idx] for idx in relative_indices]

    # --- Metrics Calculation ---
    overall_acc = accuracy_score(true_labels, predicted_ids)
    overall_f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)
    
    old_acc, old_f1, new_acc, new_f1 = None, None, None, None

    print(f"\n📊 Results for {name}:")
    print(f"  ➜ Overall Acc : {overall_acc:.4f} | Overall F1 : {overall_f1:.4f}")

    # Known Classes Metrics
    if old_classes:
        old_indices = [i for i, label in enumerate(true_labels) if label in old_classes]
        true_old = [true_labels[i] for i in old_indices]
        pred_old = [predicted_ids[i] for i in old_indices]
        if true_old:
            old_acc = accuracy_score(true_old, pred_old)
            old_f1 = f1_score(true_old, pred_old, average='weighted', labels=old_classes, zero_division=0)
            print(f"  ➜ Known Acc   : {old_acc:.4f} | Known F1   : {old_f1:.4f}")
        else:
            print(f"  ➜ Known Acc   : N/A (No old samples in test set)")

    # Unknown Classes Metrics
    if new_classes:
        new_indices = [i for i, label in enumerate(true_labels) if label in new_classes]
        true_new = [true_labels[i] for i in new_indices]
        pred_new = [predicted_ids[i] for i in new_indices]
        if true_new:
            new_acc = accuracy_score(true_new, pred_new)
            new_f1 = f1_score(true_new, pred_new, average='weighted', labels=new_classes, zero_division=0)
            print(f"  ➜ Unknown Acc : {new_acc:.4f} | Unknown F1 : {new_f1:.4f}")
        else:
            print(f"  ➜ Unknown Acc : N/A (No new samples in test set)")

    # --- Plotting Confusion Matrix ---
    # Arabic text formatting removed; using standard labels for Spanish
    fixed_labels = [CLASS_NAMES[i] for i in known_ids]
    
    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=fixed_labels,
                yticklabels=fixed_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (SSTuning) - {name}\nOverall F1: {overall_f1:.2f}')
    
    # Save the plot automatically
    save_path = os.path.join(OUTPUT_DIR, f"damo_cm_step_{step_id}.png")
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
    
    # START with your Baseline classes (0-3) as the initial "Known" classes.
    previous_known_ids = BASELINE_LABELS.copy()
    
    for step in TEST_SCHEDULE:
        current_classes = step["known_ids"]
        # Find which classes are brand new in this specific step
        new_classes_in_step = [cls for cls in current_classes if cls not in previous_known_ids]
        
        # Run the evaluation
        step_metrics = evaluate_step(step, previous_known_ids)
        if step_metrics:
            all_results.append(step_metrics)
            
        # Update the known classes memory for the NEXT loop iteration
        previous_known_ids.extend(new_classes_in_step)

    # Print a final beautiful summary table
    print("\n\n" + "="*80)
    print("📊 FINAL ZERO-SHOT EXPERIMENT SUMMARY (SPANISH)")
    print("="*80)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        
        csv_save_path = os.path.join(OUTPUT_DIR, "spanish_continual_learning_summary.csv")
        results_df.to_csv(csv_save_path, index=False)
        print("\n" + "="*80)
        print(f"💾 Saved full summary table to: {csv_save_path}")