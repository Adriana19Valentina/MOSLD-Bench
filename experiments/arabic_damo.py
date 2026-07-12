import pandas as pd
import torch
import string
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm


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
        "name": "Test Set 1",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/Arabic/test_1.csv",
        "known_ids": [0, 1, 2, 3, 4, 5] 
    },
    {
        "name": "Test Set 2",
        "path": "/home/alin/Desktop/ContinualLearning/datasets/Arabic/test_2.csv",
        "known_ids": [0, 1, 2, 3, 4, 5, 6, 7]
    },
    {
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

def predict_sstuning(text, candidate_labels_text):
    
    formatted_labels = [x + '.' if x[-1] != '.' else x for x in candidate_labels_text]
    
    padded_labels = formatted_labels + [tokenizer.pad_token] * (20 - len(formatted_labels))
    
    s_option = ' '.join([f'({list_ABC[i]}) {padded_labels[i]}' for i in range(len(padded_labels))])
    
    full_input = f'{s_option} {tokenizer.sep_token} {text}'

    inputs = tokenizer(
        [full_input], 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    valid_logits = logits[:, 0:len(formatted_labels)]
    prediction_idx = torch.argmax(valid_logits, dim=-1).item()
    
    return prediction_idx


def evaluate_step(step_config):
    name = step_config["name"]
    path = step_config["path"]
    known_ids = step_config["known_ids"]
    
    print(f"\n{'='*40}")
    print(f"Processing: {name}")
    print(f"Active Classes: {known_ids}")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return

    candidate_names = [CLASS_NAMES[i] for i in known_ids]
    
    true_labels = df['label'].tolist()
    
    texts = df['content'].tolist()
    
    predicted_ids = []

    print(f"Classifying {len(texts)} samples...")
    
    for text in tqdm(texts):
        pred_relative_idx = predict_sstuning(text, candidate_names)
        
        pred_global_id = known_ids[pred_relative_idx]
        predicted_ids.append(pred_global_id)

    acc = accuracy_score(true_labels, predicted_ids)
    
    f1 = f1_score(true_labels, predicted_ids, average='weighted', labels=known_ids, zero_division=0)

    print(f"\n Accuracy on {name}: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_ids, 
                                labels=known_ids,
                                target_names=[CLASS_NAMES[i] for i in known_ids],
                                zero_division=0))

    # مصفوفة الارتباك (Confusion Matrix)
    cm = confusion_matrix(true_labels, predicted_ids, labels=known_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[CLASS_NAMES[i] for i in known_ids],
                yticklabels=[CLASS_NAMES[i] for i in known_ids])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (SSTuning) - {name}\nF1: {f1:.2f}')
    plt.show()


if __name__ == "__main__":
    for step in TEST_SCHEDULE:
        evaluate_step(step)