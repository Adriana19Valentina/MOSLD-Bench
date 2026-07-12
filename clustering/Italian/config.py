# config.py - Central Configuration for Continual Learning Pipeline
# =====================================================================
# MODIFY THIS FILE FOR YOUR LANGUAGE/DATASET
# =====================================================================

import os

# =========================================================================
# LANGUAGE & MODEL SETTINGS - CHANGE THESE FOR YOUR LANGUAGE
# =========================================================================

LANGUAGE = 'Italian'  # Just for display purposes

# BERT Model for your language
# Examples:
#   Romanian: 'dumitrescustefan/bert-base-romanian-cased-v1'
#   Bengali:  'sagorsarker/bangla-bert-base' or 'csebuetnlp/banglabert'
#   Arabic:   'aubmindlab/bert-base-arabertv2'
#   Russian:  'DeepPavlov/rubert-base-cased'
#   Italian:  'dbmdz/bert-base-italian-cased'
#   Turkish:  'dbmdz/bert-base-turkish-cased'
#   Chinese:  'bert-base-chinese'
#   Japanese: 'cl-tohoku/bert-base-japanese'

MODEL_NAME = 'xlm-roberta-base'

# =========================================================================
# DATASET PATHS - CHANGE THESE FOR YOUR DATASET LOCATION
# =========================================================================

DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/Italian/Italian'

TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_1_CSV = os.path.join(DATASET_DIR, 'test_1.csv')
TEST_2_CSV = os.path.join(DATASET_DIR, 'test_2.csv')
TEST_3_CSV = os.path.join(DATASET_DIR, 'test_3.csv')
VAL_CSV = os.path.join(DATASET_DIR, 'val.csv')

# =========================================================================
# CLASS CONFIGURATION - CHANGE THESE BASED ON YOUR DATASET
# =========================================================================

# Classes in train.csv (baseline)
BASELINE_LABELS = [1, 7, 3, 2]

# New classes added at each test step
TEST_1_NEW_LABELS = [6, 4]  # New in test_1.csv
TEST_2_NEW_LABELS = [0, 5]  # New in test_2.csv
TEST_3_NEW_LABELS = [8, 9]  # New in test_3.csv

# Known classes at each step (cumulative)
KNOWN_LABELS_T1 = BASELINE_LABELS  # [1,2,3,4]
KNOWN_LABELS_T2 = BASELINE_LABELS + TEST_1_NEW_LABELS  # [1,2,3,4,5,6]
KNOWN_LABELS_T3 = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS  # [1,2,3,4,5,6,7,8]


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def expand_classifier(model, new_num_labels):
    """
    Expand classifier head for both BERT and RoBERTa/CamemBERT architectures.
    Preserves weights for existing classes and initializes new class weights.

    Args:
        model: HuggingFace model with classifier head
        new_num_labels: Total number of labels after expansion

    Returns:
        model: Model with expanded classifier
        old_num_labels: Number of labels before expansion
    """
    import torch

    # Detect classifier architecture
    if hasattr(model.classifier, 'out_proj'):
        # RoBERTa/CamemBERT/Umberto style
        old_classifier = model.classifier.out_proj
        is_roberta = True
    elif hasattr(model.classifier, 'out_features'):
        # BERT style - classifier is nn.Linear
        old_classifier = model.classifier
        is_roberta = False
    else:
        raise ValueError(f"Unknown classifier architecture: {type(model.classifier)}")

    old_num_labels = old_classifier.out_features

    if new_num_labels > old_num_labels:
        new_classifier = torch.nn.Linear(old_classifier.in_features, new_num_labels)

        with torch.no_grad():
            # Copy existing weights
            new_classifier.weight[:old_num_labels] = old_classifier.weight
            new_classifier.bias[:old_num_labels] = old_classifier.bias
            # Initialize new weights
            torch.nn.init.normal_(new_classifier.weight[old_num_labels:], std=0.02)
            torch.nn.init.zeros_(new_classifier.bias[old_num_labels:])

        if is_roberta:
            model.classifier.out_proj = new_classifier
        else:
            model.classifier = new_classifier

        model.num_labels = new_num_labels
        model.config.num_labels = new_num_labels

        print(f"✅ Classifier expanded: {old_num_labels} → {new_num_labels} classes")
    else:
        print(f"ℹ️  No expansion needed: {old_num_labels} >= {new_num_labels}")

    return model, old_num_labels
# =========================================================================
# OUTPUT DIRECTORY
# =========================================================================
OOD_THRESHOLD_METHOD = 'mahalanobis'

# Manual threshold (set to None to use automatic)
OOD_THRESHOLD = None

# Use entropy as additional filter (for non-mahalanobis methods)
OOD_USE_ENTROPY_FILTER = True

OUTPUT_DIR = './italian_cl_outputs'

MODEL_T1_DIR = os.path.join(OUTPUT_DIR, 'model_t1')
MODEL_T2_DIR = os.path.join(OUTPUT_DIR, 'model_t2')
MODEL_T3_DIR = os.path.join(OUTPUT_DIR, 'model_t3')

# Processed data files
T1_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_1_processed.csv')
T2_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_2_processed.csv')
T3_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_3_processed.csv')

# Clustering results
T1_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_1_results.pkl')
T2_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_2_results.pkl')
T3_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_3_results.pkl')

# Evaluation results
T1_EVAL_JSON = os.path.join(OUTPUT_DIR, 'eval_test_1.json')
T2_EVAL_JSON = os.path.join(OUTPUT_DIR, 'eval_test_2.json')
T3_EVAL_JSON = os.path.join(OUTPUT_DIR, 'eval_test_3.json')

# =========================================================================
# PSEUDO-LABEL CONFIGURATION
# =========================================================================

# Pseudo-labels start from these values to avoid conflict with GT labels
PSEUDO_LABEL_START_T1 = 14  # T1 discovered: 13, 14, ...
PSEUDO_LABEL_START_T2 = 17  # T2 discovered: 16, 17, ...
PSEUDO_LABEL_START_T3 = 21  # T3 discovered: 19, 20, ...

# =========================================================================
# CLUSTERING CONFIGURATION
# =========================================================================

K_MIN = 2  # Minimum K for K-means
K_MAX = 8  # Maximum K for K-means
SAMPLE_SELECTION_RATIO = 0.4  # Base ratio for sample selection
PURITY_THRESHOLD = 0.70  # Minimum acceptable purity

# Force K to specific value (set to None to use automatic selection)
# If you know the number of new classes, set these values
FORCE_K_T1 = None  # Set to len(TEST_1_NEW_LABELS) e.g., 3 to force K=3
FORCE_K_T2 = None  # Set to len(TEST_2_NEW_LABELS)
FORCE_K_T3 = None  # Set to len(TEST_3_NEW_LABELS)

# =========================================================================
# TRAINING CONFIGURATION
# =========================================================================

NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# =========================================================================
# STOP WORDS (Multilingual: Romanian + English + common)
STOP_WORDS = {
    # articoli determinativi / indeterminativi
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una", "un'",

    # preposizioni semplici
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",

    # preposizioni articolate
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "col", "coi", "con", "sul", "sullo", "sulla", "sui", "sugli", "sulle",

    # congiunzioni / locuzioni
    "e", "ed", "o", "od", "oppure", "ovvero",
    "ma", "però", "tuttavia", "anzi", "invece",
    "perché", "poiché", "siccome", "dato", "dato che", "visto", "visto che",
    "se", "anche se", "qualora", "mentre", "quando", "finché", "affinché",
    "cioè", "ossia", "quindi", "dunque", "allora", "pertanto",

    # pronomi personali (forme toniche)
    "io", "tu", "lui", "lei", "noi", "voi", "loro", "esso", "essa", "essi", "esse",

    # pronomi clitici / combinazioni comuni
    "mi", "ti", "si", "ci", "vi",
    "lo", "la", "li", "le", "gli", "ne",
    "melo", "mela", "meli", "mele",
    "telo", "tela", "teli", "tele",
    "selo", "sela", "seli", "sele",
    "celo", "cela", "celi"

}



# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_T1_DIR, exist_ok=True)
    os.makedirs(MODEL_T2_DIR, exist_ok=True)
    os.makedirs(MODEL_T3_DIR, exist_ok=True)


def get_pseudo_label_range(step):
    """Get the pseudo-label range for a given step"""
    if step == 1:
        return PSEUDO_LABEL_START_T1, PSEUDO_LABEL_START_T2
    elif step == 2:
        return PSEUDO_LABEL_START_T2, PSEUDO_LABEL_START_T3
    elif step == 3:
        return PSEUDO_LABEL_START_T3, PSEUDO_LABEL_START_T3 + 10
    else:
        raise ValueError(f"Invalid step: {step}")


def print_config():
    """Print current configuration"""
    print(f"\n{'=' * 60}")
    print(f"CONFIGURATION - {LANGUAGE}")
    print('=' * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Baseline classes: {BASELINE_LABELS}")
    print(f"Test_1 new classes: {TEST_1_NEW_LABELS}")
    print(f"Test_2 new classes: {TEST_2_NEW_LABELS}")
    print(f"Test_3 new classes: {TEST_3_NEW_LABELS}")
    print('=' * 60)


if __name__ == '__main__':
    print_config()