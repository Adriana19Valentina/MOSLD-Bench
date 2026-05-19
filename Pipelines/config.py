import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOP_WORDS_EN = set(ENGLISH_STOP_WORDS)

LANGUAGE = 'English'

MODEL_NAME = 'bert-base-uncased'


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXISTING_RESULTS_DIR = os.path.join(BASE_DIR, 'english_cl_outputs_1')
MODEL_BASELINE_DIR = os.path.join(EXISTING_RESULTS_DIR, 'model_baseline')
OUTPUT_DIR = os.path.join(BASE_DIR, 'english_cl_contrastive')
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
SEED = 42
CONTRASTIVE_WEIGHT = 0.3
TEMPERATURE = 0.07
DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/English/compute/english_splits'

TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_1_CSV = os.path.join(DATASET_DIR, 'test_1.csv')
TEST_2_CSV = os.path.join(DATASET_DIR, 'test_2.csv')
TEST_3_CSV = os.path.join(DATASET_DIR, 'test_3.csv')
VAL_CSV = os.path.join(DATASET_DIR, 'val.csv')


OOD_THRESHOLD_METHOD = 'energy'

BASELINE_LABELS = [0, 1, 2, 3]

TEST_1_NEW_LABELS = [4, 5, 6]
TEST_2_NEW_LABELS = [7, 8, 9]
TEST_3_NEW_LABELS = [10, 11, 12, 13]
OOD_TARGET_TPR = 0.85
OOD_PREFERRED_METHOD = None
CLASS_NAMES = {
    0: 'Company',
    1: 'Athlete',
    2: 'MeanOfTransportation',
    3: 'NaturalPlace',
    4: 'Animal',
    5: 'Album',
    6: 'Building',
    7: 'Plant',
    8: 'Film',
    9: 'WrittenWork',
    10: 'Artist',
    11: 'Village',
    12: 'EducationalInstitution',
    13: 'OfficeHolder'
}

KNOWN_LABELS_T1 = BASELINE_LABELS
KNOWN_LABELS_T2 = BASELINE_LABELS + TEST_1_NEW_LABELS
KNOWN_LABELS_T3 = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS


OUTPUT_DIR = './english_cl_outputs_1'

BASELINE_MODEL_DIR = os.path.join(OUTPUT_DIR, 'model_baseline')
MODEL_T1_DIR = os.path.join(OUTPUT_DIR, 'model_t1')
MODEL_T2_DIR = os.path.join(OUTPUT_DIR, 'model_t2')
MODEL_T3_DIR = os.path.join(OUTPUT_DIR, 'model_t3')

T1_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_1_processed.csv')
T2_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_2_processed.csv')
T3_PROCESSED_CSV = os.path.join(OUTPUT_DIR, 'test_3_processed.csv')

T1_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_1_results.pkl')
T2_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_2_results.pkl')
T3_RESULTS_PKL = os.path.join(OUTPUT_DIR, 'test_3_results.pkl')

T1_EVAL_JSON = os.path.join(OUTPUT_DIR, 'eval_test_1.json')
T2_EVAL_JSON = os.path.join(OUTPUT_DIR, 'eval_test_2.json')
T3_EVAL_JSON = os.path.join(OUTPUT_DIR, 'eval_test_3.json')



_ALL_GT_LABELS = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS + TEST_3_NEW_LABELS
PSEUDO_LABEL_START_T1 = 9 + 100
PSEUDO_LABEL_START_T2 = PSEUDO_LABEL_START_T1 + 10
PSEUDO_LABEL_START_T3 = PSEUDO_LABEL_START_T2 + 10


K_MIN = 2
K_MAX = 8
SAMPLE_SELECTION_RATIO = 0.4
PURITY_THRESHOLD = 0.70

FORCE_K_T1 = None
FORCE_K_T2 = None
FORCE_K_T3 = None

OOD_THRESHOLD = None
OOD_USE_ENTROPY_FILTER = True

NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BASELINE_MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_T1_DIR, exist_ok=True)
    os.makedirs(MODEL_T2_DIR, exist_ok=True)
    os.makedirs(MODEL_T3_DIR, exist_ok=True)


def get_pseudo_label_range(step):
    if step == 1:
        return PSEUDO_LABEL_START_T1, PSEUDO_LABEL_START_T2
    elif step == 2:
        return PSEUDO_LABEL_START_T2, PSEUDO_LABEL_START_T3
    elif step == 3:
        return PSEUDO_LABEL_START_T3, PSEUDO_LABEL_START_T3 + 10
    else:
        raise ValueError(f"Invalid step: {step}")


def print_config():
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
