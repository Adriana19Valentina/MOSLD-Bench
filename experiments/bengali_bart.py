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

DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/Bengali/bengali_splits'
EXPERIMENT_DIR = '/home/alin/Desktop/ContinualLearning/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_idx = "mps"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_idx = 0
else:
    device = torch.device("cpu")
    device_idx = -1

print(f"✅ Using device: {device}")

# Change the model to BART-large-MNLI
MODEL_NAME = "facebook/bart-large-mnli"

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

LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

# Define Continual Learning Schedule
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": os.path.join(DATASET_DIR, "test_1.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5],
        "save_name": "bengali_bart_1.png" # Updated save name
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": os.path.join(DATASET_DIR, "test_2.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "save_name": "bengali_bart_2.png" # Updated save name
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": os.path.join(DATASET_DIR, "test_3.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_name": "bengali_bart_3.png" # Updated save name
    }
]

print(f"Loading {MODEL_NAME}...")
classifier = pipeline("zero-shot-classification", 
                      model=MODEL_NAME, 
                      device=device_idx)

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
        batch_texts = texts[i : i + batch_size]
        
        # Updated hypothesis template to Bengali
        results = classifier(batch_texts, 
                             candidate_labels, 
                             hypothesis_template="এই লেখাটি {} সম্পর্কে।", # "This text is about {}." in Bengali
                             truncation=True) # Ensure truncation is True to avoid size errors
        
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', # Use a different color scheme (e.g., Oranges)
                xticklabels=candidate_labels,
                yticklabels=candidate_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    
    # Updated title to Bengali
    plt.title(f'BART Zero-Shot (Bengali) - {name}\nAcc: {acc:.2f} | F1: {f1:.2f}')
    
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

if __name__ == "__main__":
    all_results = []
    
    for step in TEST_SCHEDULE:
        result = evaluate_step(step)
        if result:
            all_results.append(result)

    print("\n\n" + "="*50)
    print("FINAL SUMMARY (BART-LARGE-MNLI BENGALI)")
    print("="*50)
    
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No results generated.")
    print("="*50)