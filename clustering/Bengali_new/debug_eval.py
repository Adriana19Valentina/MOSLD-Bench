import torch
import numpy as np
import pandas as pd
import json
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========== CONFIGURARE - MODIFICĂ AICI ===========
MODEL_PATH = './english_cl_outputs_1/model_t1'
TEST_CSV = '/home/alin/Desktop/ContinualLearning/datasets/English/compute/english_splits/test_1.csv'
PROCESSED_CSV = './english_cl_outputs_1/test_1_processed.csv'
NEW_LABELS = [4, 5, 6]  # Animal, Album, Building
CLASS_NAMES = {
    0: 'Company', 1: 'Athlete', 2: 'MeanOfTransportation', 3: 'NaturalPlace',
    4: 'Animal', 5: 'Album', 6: 'Building'
}
# ===================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("DEBUG EVALUATION")
print("=" * 70)

# 1. Model config
print("\n📋 STEP 1: MODEL CONFIG")
print("-" * 50)
with open(f'{MODEL_PATH}/config.json', 'r') as f:
    config = json.load(f)
print(f"num_labels: {config.get('num_labels', 'MISSING!')}")
id2label = config.get('id2label', {})
print(f"id2label: {id2label}")

# 2. Training data
print("\n📋 STEP 2: TRAINING DATA")
print("-" * 50)
train_df = pd.read_csv(PROCESSED_CSV)
print("Label distribution:")
print(train_df['label'].value_counts().sort_index())

# 3. Test data
print("\n📋 STEP 3: TEST DATA")
print("-" * 50)
test_df = pd.read_csv(TEST_CSV)
print("Label distribution:")
print(test_df['label'].value_counts().sort_index())

# 4. Model predictions
print("\n📋 STEP 4: MODEL PREDICTIONS FOR NEW CLASSES")
print("-" * 50)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()

for gt_label in NEW_LABELS:
    samples = test_df[test_df['label'] == gt_label].head(30)

    if len(samples) == 0:
        print(f"\n⚠️  GT {gt_label} ({CLASS_NAMES.get(gt_label, '?')}): NO SAMPLES!")
        continue

    predictions = []
    for _, row in samples.iterrows():
        text = str(row['content'])[:512] if row['content'] else ''
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred_id)

    pred_counter = Counter(predictions)
    print(f"\n🔍 GT {gt_label} ({CLASS_NAMES.get(gt_label, '?')}):")
    print(f"   Samples tested: {len(predictions)}")
    print(f"   Model predictions (model_id): {dict(pred_counter)}")

    # Convertește la pseudo-labels
    pseudo_preds = [int(id2label.get(str(p), id2label.get(p, p))) for p in predictions]
    pseudo_counter = Counter(pseudo_preds)
    print(f"   As pseudo-labels: {dict(pseudo_counter)}")

# 5. Verifică full_mapping
print("\n📋 STEP 5: MAPPING ANALYSIS")
print("-" * 50)
print("În evaluate, full_mapping ar trebui să fie:")
print("  model_id → GT_label")
print()
print("Dacă folosești {i: i for i in range(N)}:")
print("  model_id 0 → GT 0 ✓")
print("  model_id 1 → GT 1 ✓")
print("  ...")
print("  model_id 4 → GT 4 (Animal) ← GREȘIT dacă model_id 4 = pseudo 109!")
print()
print("Corect ar fi să folosești Hungarian mapping pentru clasele noi:")
print("  model_id 4 → pseudo 109 → GT ? (din Hungarian)")
print("  model_id 5 → pseudo 110 → GT ? (din Hungarian)")