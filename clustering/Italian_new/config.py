# config.py - Central Configuration for Continual Learning Pipeline
# =====================================================================
# RUSSIAN LANGUAGE CONFIGURATION
# =====================================================================

import os
# =========================================================================
# LANGUAGE & MODEL SETTINGS
# =========================================================================
from spacy.lang.it.stop_words import STOP_WORDS as IT_STOP_WORDS

STOP_WORDS_IT = set(IT_STOP_WORDS)

LANGUAGE = 'Italian'

# BERT Model for Russian
MODEL_NAME = 'Musixmatch/umberto-commoncrawl-cased-v1'

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
OOD_THRESHOLD_METHOD = 'energy'

# Classes in train.csv (baseline)
BASELINE_LABELS = [1, 2, 3, 7]

# New classes added at each test step
TEST_1_NEW_LABELS = [4, 6]  # New in test_1.csv
TEST_2_NEW_LABELS = [0, 5]  # New in test_2.csv
TEST_3_NEW_LABELS = [8, 9]  # New in test_3.csv

CLASS_NAMES = {
    0: 'ambiente',
    1: 'artiespettacolo',
    2: 'economiaefinanza',   # Baseline
    3: 'politica',           # Baseline
    4: 'salute',                 # T1 - persoană
    5: 'scienzaetecnologia',               # T1 - structură
    6: 'societa',                 # T1 - biologie
    7: 'sport',                  # T2 - muzică
    8: 'lifestyle',                # T2 - locație
    9: 'viaggiaeturismo',                  # T2 - biologie
     # T3
}
# "grouped_educatie.csv": 7,
# "grouped_natura.csv": 8,
# "grouped_sanatate.csv": 9,
# "grouped_vacante.csv": 10
# # Known classes at each step (cumulative)
KNOWN_LABELS_T1 = BASELINE_LABELS  # [1,2,3,4]
KNOWN_LABELS_T2 = BASELINE_LABELS + TEST_1_NEW_LABELS  # [1,2,3,4,5,6]
KNOWN_LABELS_T3 = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS  # [1,2,3,4,5,6,7,8]

# =========================================================================
# OUTPUT DIRECTORY
# =========================================================================

OUTPUT_DIR = './italian_cl_outputs_1'

# Model directories
BASELINE_MODEL_DIR = os.path.join(OUTPUT_DIR, 'model_baseline')
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

_ALL_GT_LABELS = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS + TEST_3_NEW_LABELS
PSEUDO_LABEL_START_T1 = max(_ALL_GT_LABELS) + 100  # e.g., 110
PSEUDO_LABEL_START_T2 = PSEUDO_LABEL_START_T1 + 10  # 120
PSEUDO_LABEL_START_T3 = PSEUDO_LABEL_START_T2 + 10  # 130

# =========================================================================
# CLUSTERING CONFIGURATION
# =========================================================================

K_MIN = 2
K_MAX = 8
SAMPLE_SELECTION_RATIO = 0.4
PURITY_THRESHOLD = 0.70

FORCE_K_T1 = None
FORCE_K_T2 = None
FORCE_K_T3 = None

# =========================================================================
# OOD DETECTION CONFIGURATION
# =========================================================================

OOD_THRESHOLD = None
OOD_USE_ENTROPY_FILTER = True

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
# STOP WORDS (Russian + English)
# =========================================================================
# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BASELINE_MODEL_DIR, exist_ok=True)
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