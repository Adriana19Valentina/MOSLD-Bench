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

# ------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------
DATASET_DIR = '/home/alin/Desktop/MOSLD-Bench/French'
EXPERIMENT_DIR = '/home/alin/Desktop/ContinualLearning/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Select Device (Prioritize CUDA)
if torch.backends.mps.is_available():
    device_str = "mps"
    device_idx = "mps"
elif torch.cuda.is_available():
    device_str = "cuda"
    device_idx = 0
else:
    device_str = "cpu"
    device_idx = -1

print(f"✅ Using device: {device_str}")

# Model Name
MODEL_NAME = "facebook/bart-large-mnli"

# French Class Mappings
CLASS_NAMES = {
    0: 'politique',      # Politics
    1: 'voyage',         # Travel
    2: 'international',  # International
    3: 'science',        # Science
    4: 'culture',        # Culture
    5: 'sport',          # Sport
    6: 'médias',         # Media
    7: 'économie',       # Economy
    8: 'technologie',    # Technology
    9: 'mode de vie'     # Lifestyle
}

LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

# Define Continual Learning Schedule
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": os.path.join(DATASET_DIR, "test_1.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5],
        "save_name": "french_1.png"
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": os.path.join(DATASET_DIR, "test_2.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "save_name": "french_2.png"
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": os.path.join(DATASET_DIR, "test_3.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_name": "french_3.png"
    }
]

# ------------------------------
# 2. LOAD PIPELINE
# ------------------------------
print(f"Loading {MODEL_NAME}...")
classifier = pipeline("zero-shot-classification", 
                      model=MODEL_NAME, 
                      device=device_idx)

# ------------------------------
# 3. EVALUATION FUNCTION
# ------------------------------
def evaluate_step(step_config):
    name = step_config["name"]
    path = step_config["path"]
    known_ids = step_config["known_ids"]
    save_name = step_config["save_name"]
    
    print(f"\n{'='*40}")
    print(f"Processing: {name}")
    print(f"Active Classes: {known_ids}")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f" File not found: {path}")
        return None

    text_col = next((c for c in ['text', 'content', 'sentence', 'body'] if c in df.columns), None)
    if text_col is None:
        print(f" Error: Text column not found. Columns: {df.columns.tolist()}")
        return None

    candidate_labels = [CLASS_NAMES[i] for i in known_ids]
    true_labels = df['label'].tolist()
    texts = df[text_col].tolist()
    predicted_ids = []

    print(f"Classifying {len(texts)} samples (Batch mode)...")
    
    batch_size = 16 
    for i in tqdm(range(0, len(texts), batch_size)):
        # Bulletproof fix: Manually chop texts to roughly the first 300 words 
        # to ensure they never exceed the 514 token limit when combined with the hypothesis.
        batch_texts = [" ".join(str(t).split()[:300]) for t in texts[i : i + batch_size]]
        
        # Explicitly enforce max_length alongside truncation
        results = classifier(batch_texts, 
                             candidate_labels, 
                             hypothesis_template="Ce texte parle de {}.", 
                             truncation=True,
                             max_length=512) # Added max_length here
        if isinstance(results, dict): results = [results]
            
        for res in results:
            top_label = res['labels'][0]
            predicted_ids.append(LABEL_TO_ID[top_label])

    acc = accuracy_score(true_labels, predicted_ids)
    f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)
    
    print(f" Completed {name} - Acc: {acc:.4f}, F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_ids, 
                                labels=known_ids, 
                                target_names=candidate_labels, 
                                zero_division=0))

    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', # Changed to Reds for visual distinction
                xticklabels=candidate_labels,
                yticklabels=candidate_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    
    plt.title(f'XLM-R Zero-Shot (French) - {name}\nAcc: {acc:.2f} | F1: {f1:.2f}')
    
    save_path = os.path.join(EXPERIMENT_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📷 Saved confusion matrix to: {save_path}")

    return {
        "Step": name,
        "Accuracy": acc,
        "F1 (Weighted)": f1,
        "Samples": len(texts)
    }

# ------------------------------
# 4. MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    all_results = []
    
    for step in TEST_SCHEDULE:
        result = evaluate_step(step)
        if result:
            all_results.append(result)

    print("\n\n" + "="*50)
    print("FINAL SUMMARY (XLM-R FRENCH)")
    print("="*50)
    
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No results generated.")
    print("="*50)