import pandas as pd
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

print("=" * 70)
print("GENERARE EMBEDDINGS PENTRU TEST_1")
print("=" * 70)


MODEL_NAME = "bert-base-multilingual-cased"
BATCH_SIZE = 32
MAX_LENGTH = 128

# Paths
TEST_1_CSV = '/home/alin/Desktop/ContinualLearning/datasets/English/test_1.csv'
OUTPUT_PKL = 'embeddings_test_1.pkl'

print(f"\n📂 Loading data from: {TEST_1_CSV}")

# =========================================================================
# STEP 1: LOAD DATA
# =========================================================================

try:
    test_1_df = pd.read_csv(TEST_1_CSV)
    print(f"✅ Loaded {len(test_1_df)} samples")
    print(f"   Columns: {test_1_df.columns.tolist()}")
    print(f"   Classes: {sorted(test_1_df['label'].unique())}")
except FileNotFoundError:
    print(f"❌ ERROR: {TEST_1_CSV} not found!")
    print(f"\nPlease create {TEST_1_CSV} with columns: ['content', 'label']")
    print(f"Expected labels: 1 (Edu), 3 (Athlete), 11 (Album)")
    exit(1)

# Verificare clase
expected_classes = {1, 3, 11}
actual_classes = set(test_1_df['label'].unique())
if actual_classes != expected_classes:
    print(f"⚠️  WARNING: Expected classes {expected_classes}, got {actual_classes}")

# =========================================================================
# STEP 2: INITIALIZE MODEL
# =========================================================================

print(f"\n🤖 Loading BERT model: {MODEL_NAME}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print(f"✅ Model loaded successfully")

# =========================================================================
# STEP 3: GENERATE EMBEDDINGS
# =========================================================================

print(f"\n🔄 Generating embeddings...")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Max length: {MAX_LENGTH}")

texts = test_1_df['content'].tolist()
labels = test_1_df['label'].tolist()

all_embeddings = []

# Process în batches
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
    batch_texts = texts[i:i + BATCH_SIZE]

    # Tokenize
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract [CLS] token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    all_embeddings.append(cls_embeddings)

# Concatenate all batches
embeddings = np.vstack(all_embeddings)
print(f"\n✅ Generated embeddings shape: {embeddings.shape}")

# =========================================================================
# STEP 4: SAVE TO PICKLE
# =========================================================================

embeddings_dict = {
    'content': texts,
    'embeddings': embeddings,
    'label': labels  # Ground truth pentru evaluare
}

with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f"\n✅ Embeddings saved to: {OUTPUT_PKL}")
print(f"   Contents:")
print(f"     - texts: {len(texts)} samples")
print(f"     - embeddings: {embeddings.shape}")
print(f"     - labels: {len(labels)} ground truth labels")

print("\n" + "=" * 70)
print("✅ EMBEDDING GENERATION COMPLETED!")
print("=" * 70)
print(f"\nYou can now run: python pipeline_t1_elbow.py")