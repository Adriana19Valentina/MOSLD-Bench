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
DATASET_DIR = '/home/alin/Desktop/MOSLD-Bench/Japanese'
EXPERIMENT_DIR = '/home/alin/Desktop/ContinualLearning/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Select Device (Prioritize CUDA)
device = torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cpu')
print(f"✅ Using device: {device}")

# Model Name
MODEL_NAME = "facebook/bart-large-mnli"

# Japanese Class Mappings
CLASS_NAMES = {
    0: '医療',                # Medical
    1: 'テクノロジー',         # Technology
    2: 'エンターテインメント',  # Entertainment
    3: 'スポーツ',             # Sports
    4: '企業',                 # Enterprise/Business
    5: '家族',                 # Family
    6: '環境',                 # Environment
    7: '政治情勢',             # Political Situation
    8: '福祉',                 # Welfare
    9: '事件'                  # Incident/Case
}

# Define Continual Learning Schedule
# Test 1: Baseline (0-3) + Test 1 New (4-5) -> Total 6 classes
# Test 2: Previous + Test 2 New (6-7) -> Total 8 classes
# Test 3: Previous + Test 3 New (8-9) -> Total 10 classes
TEST_SCHEDULE = [
    {
        "id": "1",
        "name": "Test Set 1",
        "path": os.path.join(DATASET_DIR, "test_1.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5],
        "save_name": "japanese_1.png"
    },
    {
        "id": "2",
        "name": "Test Set 2",
        "path": os.path.join(DATASET_DIR, "test_2.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "save_name": "japanese_2.png"
    },
    {
        "id": "3",
        "name": "Test Set 3",
        "path": os.path.join(DATASET_DIR, "test_3.csv"),
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_name": "japanese_3.png"
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
    
    # Attempt to use fonts that support Japanese characters
    plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    
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
    print("FINAL SUMMARY (DAMO-SSTuning Japanese)")
    print("="*50)
    
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No results generated.")
    print("="*50)