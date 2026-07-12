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

# --- Arabic Text Rendering ---
# Ensure these are installed: pip install arabic-reshaper python-bidi
import arabic_reshaper
from bidi.algorithm import get_display

# ------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------
warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/alin/Desktop/ContinualLearning/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cpu')
print(f"✅ Using device: {device}")

MODEL_NAME = "DAMO-NLP-SG/zero-shot-classify-SSTuning-XLM-R"
# Define Directories
DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/Bengali/bengali_splits'
EXPERIMENT_DIR = '/home/alin/Desktop/ContinualLearning/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Select Device (Prioritize CUDA)
device = torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cpu')
print(f"✅ Using device: {device}")

# Model Name
MODEL_NAME = "DAMO-NLP-SG/zero-shot-classify-SSTuning-XLM-R"

# Bengali Class Mappings
CLASS_NAMES = {
    0: 'রাজনীতি',    # Politics
    1: 'প্রযুক্তি',   # Technology
    2: 'আন্তর্জাতিক', # International
    3: 'জাতীয়',      # National
    4: 'ক্রীড়া',      # Sports
    5: 'জীবনধারা',    # Lifestyle
    6: 'বিনোদন',      # Entertainment
    7: 'সম্পাদকীয়',   # Editorial
    8: 'কলকাতা',     # Kolkata
    9: 'রাজ্য',       # State
}

# Define Continual Learning Schedule
# Test 1: Baseline (0-3) + Test 1 New (4-5)
# Test 2: Previous + Test 2 New (6-7)
# Test 3: Previous + Test 3 New (8-9)
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": os.path.join(DATASET_DIR, "test_1.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5],
        "save_name": "bengali_1.png"
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": os.path.join(DATASET_DIR, "test_2.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "save_name": "bengali_2.png"
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": os.path.join(DATASET_DIR, "test_3.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_name": "bengali_3.png"
    }
]
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

list_ABC = [x for x in string.ascii_uppercase]

# ------------------------------
# 2. BATCH PREDICTION LOGIC
# ------------------------------
def predict_sstuning_batch(texts, candidate_labels_text, batch_size=16):
    formatted_labels = [x + '.' if x[-1] != '.' else x for x in candidate_labels_text]
    padded_labels = formatted_labels + [tokenizer.pad_token] * (20 - len(formatted_labels))
    s_option = ' '.join([f'({list_ABC[i]}) {padded_labels[i]}' for i in range(len(padded_labels))])
    
    full_inputs = [f'{s_option} {tokenizer.sep_token} {t}' for t in texts]
    all_pred_indices = []
    
    for i in tqdm(range(0, len(full_inputs), batch_size), desc="Batch Inference"):
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

# ------------------------------
# 3. EVALUATION FUNCTION
# ------------------------------
def evaluate_step(step_config, previous_known_ids):
    name = step_config["name"]
    path = step_config["path"]
    known_ids = step_config["known_ids"]
    step_id = step_config["id"]
    
    # 1. Identify Known (Old) vs Unknown (New) classes
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
    
    print(f"Classifying {len(texts)} samples...")
    relative_indices = predict_sstuning_batch(texts, candidate_names, batch_size=16)
    
    # Map back to global IDs
    predicted_ids = [known_ids[idx] for idx in relative_indices]

    # 2. Metrics Calculation
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

    # 3. Plotting Confusion Matrix
    fixed_labels = [get_display(arabic_reshaper.reshape(CLASS_NAMES[i])) for i in known_ids]
    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=fixed_labels,
                yticklabels=fixed_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (SSTuning) - {name}\nOverall F1: {overall_f1:.2f}')
    
    save_path = os.path.join(OUTPUT_DIR, f"damo_cm_step_{step_id}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"\n📷 Saved confusion matrix to: {save_path}")

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

# ------------------------------
# 4. MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    all_results = []
    
    # Initialize previous_known_ids with Test Set 1's classes
    # so they are treated as the "Known Base" from the very start.
    base_classes = TEST_SCHEDULE[0]["known_ids"].copy()
    previous_known_ids = base_classes  
    
    print("\n🚀 Starting Continual Learning Evaluation with DAMO SSTuning...")
    
    for step in TEST_SCHEDULE:
        current_classes = step["known_ids"]
        new_classes_in_step = [cls for cls in current_classes if cls not in previous_known_ids]
        
        # Run Evaluation
        step_metrics = evaluate_step(step, previous_known_ids)
        
        if step_metrics:
            all_results.append(step_metrics)
            
        # Add the newly discovered classes to our known memory for the NEXT step
        previous_known_ids.extend(new_classes_in_step)
        
    # Generate Final Summary
    print("\n\n" + "="*95)
    print("📊 FINAL EXPERIMENT SUMMARY")
    print("="*95)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        
        csv_save_path = os.path.join(OUTPUT_DIR, "damo_continual_learning_summary.csv")
        results_df.to_csv(csv_save_path, index=False)
        print("\n" + "="*95)
        print(f"💾 Saved full summary table to: {csv_save_path}")
    else:
        print("⚠️ No results were generated.")