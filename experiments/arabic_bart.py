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

# --- Install these if you haven't already for Arabic rendering ---
# pip install arabic-reshaper python-bidi
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

CLASS_NAMES = {
    0: 'رياضة',      # Sports
    1: 'سياسة',      # Politics
    2: 'اقتصاد',     # Economy
    3: 'منوعات',     # Misc
    4: 'فن',         # Art
    5: 'تكنولوجيا',  # Tech
    6: 'طب',         # Medicine
    7: 'ثقافة',      # Culture
    8: 'دين',        # Religion
    9: 'مجتمع',      # Society
}

TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/Arabic/test_1.csv",
        "known_ids": [0, 1, 2, 3, 4, 5] 
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/Arabic/test_2.csv",
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7]
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/Arabic/test_3.csv",
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
        
        # If batch size is 1, tolist() might not return a list, so we handle it
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
    
    # Identify Old vs New classes
    old_classes = previous_known_ids
    new_classes = [cls for cls in known_ids if cls not in previous_known_ids]
    
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print(f"Old Classes: {old_classes}")
    print(f"New Classes: {new_classes}")
    print(f"{'='*50}")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None

    candidate_names = [CLASS_NAMES[i] for i in known_ids]
    true_labels = df['label'].tolist()
    texts = df['content'].tolist()
    
    print(f"Classifying {len(texts)} samples...")
    
    # Run predictions
    relative_indices = predict_sstuning_batch(texts, candidate_names, batch_size=16)
    
    # Map back to global IDs (This defines predicted_ids!)
    predicted_ids = [known_ids[idx] for idx in relative_indices]

    # --- Metrics Calculation ---
    overall_acc = accuracy_score(true_labels, predicted_ids)
    overall_f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)
    
    old_acc, old_f1, new_acc, new_f1 = None, None, None, None

    # Old Classes Metrics
    if old_classes:
        old_indices = [i for i, label in enumerate(true_labels) if label in old_classes]
        true_old = [true_labels[i] for i in old_indices]
        pred_old = [predicted_ids[i] for i in old_indices]
        if true_old:
            old_acc = accuracy_score(true_old, pred_old)
            old_f1 = f1_score(true_old, pred_old, average='weighted', labels=old_classes, zero_division=0)

    # New Classes Metrics
    if new_classes:
        new_indices = [i for i, label in enumerate(true_labels) if label in new_classes]
        true_new = [true_labels[i] for i in new_indices]
        pred_new = [predicted_ids[i] for i in new_indices]
        if true_new:
            new_acc = accuracy_score(true_new, pred_new)
            new_f1 = f1_score(true_new, pred_new, average='weighted', labels=new_classes, zero_division=0)

    # --- Plotting Confusion Matrix ---
    # Fix Arabic text for plotting
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
    print(f"📷 Saved confusion matrix to: {save_path}")

    return {
        "Step": name,
        "Total Samples": len(texts),
        "Overall Acc": round(overall_acc, 4),
        "Overall F1": round(overall_f1, 4),
        "Old (Known) Acc": round(old_acc, 4) if old_acc is not None else "N/A",
        "Old (Known) F1": round(old_f1, 4) if old_f1 is not None else "N/A",
        "New (Unknown) Acc": round(new_acc, 4) if new_acc is not None else "N/A",
        "New (Unknown) F1": round(new_f1, 4) if new_f1 is not None else "N/A"
    }

# ------------------------------
# 4. MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    all_results = []
    previous_known_ids = []  
    
    print("\n🚀 Starting Continual Learning Evaluation with DAMO SSTuning...")
    
    for step in TEST_SCHEDULE:
        step_metrics = evaluate_step(step, previous_known_ids)
        
        if step_metrics:
            all_results.append(step_metrics)
            
        previous_known_ids = step["known_ids"].copy()
        
    print("\n\n" + "="*90)
    print("📊 FINAL EXPERIMENT SUMMARY")
    print("="*90)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        
        csv_save_path = os.path.join(OUTPUT_DIR, "damo_continual_learning_summary.csv")
        results_df.to_csv(csv_save_path, index=False)
        print("\n" + "="*90)
        print(f"💾 Saved full summary to: {csv_save_path}")
    else:
        print("⚠️ No results were generated.")