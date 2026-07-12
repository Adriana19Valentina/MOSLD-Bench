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

DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/English'
EXPERIMENT_DIR = '/home/alin/Desktop/ContinualLearning/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

device = 0 if torch.cuda.is_available() else -1
device_str = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
    device_str = "mps"

print(f"✅ Using device: {device_str} ({device})")

MODEL_NAME = "facebook/bart-large-mnli"

CLASS_NAMES = {
    0: 'Sports',
    1: 'Politics',
    2: 'Economy',
    3: 'Misc',
    4: 'Art',
    5: 'Tech',
    6: 'Medicine',
    7: 'Culture',
    8: 'Religion',
    9: 'Society',
}

LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

# Updated paths to reflect English datasets
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/English/test_1.csv",
        "known_ids": [0, 1, 2, 3, 4, 5],
        "save_name": "english_1.png"
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/English/test_2.csv",
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "save_name": "english_2.png"
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/English/test_3.csv",
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_name": "english_3.png"
    }
]


print(f"Loading {MODEL_NAME}...")
classifier = pipeline("zero-shot-classification", 
                      model=MODEL_NAME, 
                      device=device)


def evaluate_step(step_config):
    name = step_config["name"]
    path = step_config["path"]
    known_ids = step_config["known_ids"]
    save_name = step_config["save_name"]
    
    print(f"\n{'='*40}")
    print(f"Processing: {name}")
    print(f"Active Classes: {known_ids}")
    
    # Load Data
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None

    candidate_labels = [CLASS_NAMES[i] for i in known_ids]
    true_labels = df['label'].tolist()
    
    if 'content' not in df.columns:
        print(f"❌ Error: Column 'content' not found in {path}.")
        return None
        
    texts = df['content'].tolist()
    predicted_ids = []

    print(f"Classifying {len(texts)} samples...")
    
    batch_size = 16 
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        results = classifier(batch_texts, candidate_labels)
        
        if isinstance(results, dict): results = [results]
            
        for res in results:
            top_predicted_label = res['labels'][0]
            pred_id = LABEL_TO_ID[top_predicted_label]
            predicted_ids.append(pred_id)

    acc = accuracy_score(true_labels, predicted_ids)
    f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)
    
    print(f"✅ Completed {name} - Acc: {acc:.4f}, F1: {f1:.4f}")

    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CLASS_NAMES[i] for i in known_ids],
                yticklabels=[CLASS_NAMES[i] for i in known_ids])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (BART) - {name}\nAcc: {acc:.2f} | F1: {f1:.2f}')
    
    save_path = os.path.join(EXPERIMENT_DIR, save_name)
    plt.savefig(save_path)
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

    # Print Final Summary
    print("\n\n" + "="*50)
    print("FINAL SUMMARY (BART Zero-Shot - Chinese)")
    print("="*50)
    
    # Create DataFrame for clean printing
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No results generated.")
    print("="*50)