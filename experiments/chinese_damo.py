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

# ------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------

# Filter warnings
warnings.filterwarnings("ignore")

# Define Directories
DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/Chinese'
EXPERIMENT_DIR = '/home/alin/Desktop/ContinualLearning/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Select Device (Prioritize CUDA)
device = torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cpu')
print(f"✅ Using device: {device}")

# Model Name
MODEL_NAME = "facebook/bart-large-mnli"

# Chinese Class Mappings
CLASS_NAMES = {
    0: '体育',      # Sports
    1: '娱乐',      # Entertainment
    2: '家居',      # Home
    3: '房产',      # Real Estate
    4: '教育',      # Education
    5: '时尚',      # Fashion
    6: '时政',      # Politics
    7: '游戏',      # Games
    8: '社会',      # Society
    9: '科技',      # Technology
    10: '股票',     # Stock
    11: '财经',     # Finance
    12: '彩票',     # Lottery
    13: '星座'      # Horoscope
}

# Define Continual Learning Schedule
# Test 1: Baseline (0-3) + Test 1 New (4-6) -> Total 7 classes
# Test 2: Previous + Test 2 New (7-9) -> Total 10 classes
# Test 3: Previous + Test 3 New (10-13) -> Total 14 classes
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": os.path.join(DATASET_DIR, "test_1.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6],
        "save_name": "chinese_1.png"
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": os.path.join(DATASET_DIR, "test_2.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_name": "chinese_2.png"
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": os.path.join(DATASET_DIR, "test_3.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "save_name": "chinese_3.png"
    }
]

# ------------------------------
# 2. LOAD MODEL
# ------------------------------
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Helper for labeling (A, B, C...)
list_ABC = [x for x in string.ascii_uppercase]

# ------------------------------
# 3. PREDICTION LOGIC (SSTuning)
# ------------------------------
def predict_sstuning(text, candidate_labels_text):
    """
    Formats the input specifically for the DAMO SSTuning model:
    Format: "(A) Label1. (B) Label2. ... [SEP] Text"
    """
    # 1. Format labels
    formatted_labels = [x + '.' if x[-1] != '.' else x for x in candidate_labels_text]
    
    # 2. Pad to 20 labels (Requirement of this specific model architecture)
    padded_labels = formatted_labels + [tokenizer.pad_token] * (20 - len(formatted_labels))
    
    # 3. Construct the options string
    s_option = ' '.join([f'({list_ABC[i]}) {padded_labels[i]}' for i in range(len(padded_labels))])
    
    # 4. Construct final input
    full_input = f'{s_option} {tokenizer.sep_token} {text}'

    # 5. Tokenize
    inputs = tokenizer(
        [full_input], 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 6. Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # 7. Get Prediction
    valid_logits = logits[:, 0:len(formatted_labels)]
    prediction_idx = torch.argmax(valid_logits, dim=-1).item()
    
    return prediction_idx

# ------------------------------
# 4. EVALUATION LOOP
# ------------------------------
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

    # Check for text column (could be 'text', 'content', 'sentence' etc)
    text_col = None
    for col in ['text', 'content', 'sentence', 'body']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print(f"❌ Error: Could not find text column. Columns are: {df.columns.tolist()}")
        return None

    candidate_names = [CLASS_NAMES[i] for i in known_ids]
    true_labels = df['label'].tolist()
    texts = df[text_col].tolist()
    predicted_ids = []

    print(f"Classifying {len(texts)} samples...")
    
    for text in tqdm(texts):
        # Predict
        pred_relative_idx = predict_sstuning(text, candidate_names)
        
        # Map back to global ID
        pred_global_id = known_ids[pred_relative_idx]
        predicted_ids.append(pred_global_id)

    # --- Metrics ---
    acc = accuracy_score(true_labels, predicted_ids)
    f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)

    print(f"✅ Completed {name} - Acc: {acc:.4f}, F1: {f1:.4f}")
    
    # --- Classification Report ---
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_ids, 
                                labels=known_ids,
                                target_names=[CLASS_NAMES[i] for i in known_ids],
                                zero_division=0))

    # --- Save Confusion Matrix ---
    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    plt.figure(figsize=(12, 10))
    
    # Note: If squares appear, it is a system font issue with Chinese.
    # We specify a font family that often supports Chinese if available, but it depends on OS.
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif'] 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[CLASS_NAMES[i] for i in known_ids],
                yticklabels=[CLASS_NAMES[i] for i in known_ids])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Confusion Matrix (SSTuning) - {name}\nAcc: {acc:.2f} | F1: {f1:.2f}')
    
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
# 5. MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    all_results = []
    
    for step in TEST_SCHEDULE:
        result = evaluate_step(step)
        if result:
            all_results.append(result)

    # Print Final Summary Table
    print("\n\n" + "="*50)
    print("FINAL SUMMARY (DAMO-SSTuning Chinese)")
    print("="*50)
    
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No results generated.")
    print("="*50)