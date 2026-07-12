# config.py - Central Configuration for Continual Learning Pipeline
# =====================================================================
# RUSSIAN LANGUAGE CONFIGURATION
# =====================================================================

import os
# config.py
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP_WORDS
STOP_WORDS_FR = set(FR_STOP_WORDS)



LANGUAGE = 'French'

MODEL_NAME = 'camembert-base'
OOD_TARGET_TPR = 0.85  # 95% din known să fie clasificate corect
OOD_PREFERRED_METHOD = None  # None = auto-select, 'energy', sau 'msp'
# =========================================================================
# DATASET PATHS - CHANGE THESE FOR YOUR DATASET LOCATION
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXISTING_RESULTS_DIR = os.path.join(BASE_DIR, 'french_cl_outputs_1')
MODEL_BASELINE_DIR = os.path.join(EXISTING_RESULTS_DIR, 'model_baseline')
OUTPUT_DIR = os.path.join(BASE_DIR, 'french_cl_outputs_1')
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
SEED = 42
CONTRASTIVE_WEIGHT = 0.3
TEMPERATURE = 0.07
DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/French/french_anonymized_2'

TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_1_CSV = os.path.join(DATASET_DIR, 'test_1.csv')
TEST_2_CSV = os.path.join(DATASET_DIR, 'test_2.csv')
TEST_3_CSV = os.path.join(DATASET_DIR, 'test_3.csv')
VAL_CSV = os.path.join(DATASET_DIR, 'val.csv')

# =========================================================================
# CLASS CONFIGURATION - CHANGE THESE BASED ON YOUR DATASET
# =========================================================================
OOD_THRESHOLD_METHOD = 'energy'
CLASS_NAMES = {
    0: 'politique',
    1: 'voyage',
    2: 'international',
    3: 'science',
    4: 'culture',
    5: 'sport',
    6: 'médias',
    7: 'économie',
    8: 'technologie',
    9: 'mode de vie'
}

# Classes in train.csv (baseline)
BASELINE_LABELS = [0, 1, 2, 3]

TEST_1_NEW_LABELS = [4, 5]  # New in test_1.csv
TEST_2_NEW_LABELS = [6, 7]  # New in test_2.csv
TEST_3_NEW_LABELS = [8, 9]  # New in test_3.csv

CLASS_NAMES = {
    0: 'politique',
    1: 'voyage',
    2: 'international',
    3: 'science',
    4: 'culture',
    5: 'sport',
    6: 'médias',
    7: 'économie',
    8: 'technologie',
    9: 'mode de vie'
}

# Known classes at each step (cumulative)
KNOWN_LABELS_T1 = BASELINE_LABELS  # [1,2,3,4]
KNOWN_LABELS_T2 = BASELINE_LABELS + TEST_1_NEW_LABELS  # [1,2,3,4,5,6]
KNOWN_LABELS_T3 = BASELINE_LABELS + TEST_1_NEW_LABELS + TEST_2_NEW_LABELS  # [1,2,3,4,5,6,7,8]

# =========================================================================
# OUTPUT DIRECTORY
# =========================================================================

OUTPUT_DIR = './french_cl_outputs_1'

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

# STOP_WORDS = {
#     # 结构助词 / 语气词 / 高频虚词
#     '的', '了', '着', '过', '得', '地', '之', '其',
#     '啊', '呢', '吧', '吗', '呀', '哦', '啦', '么',
#
#     # 代词（人称 / 指示）
#     '我', '你', '他', '她', '它',
#     '我们', '你们', '他们', '她们', '它们',
#     '自己', '本人', '大家', '人们',
#     '这', '那', '这里', '那里', '这儿', '那儿',
#     '这些', '那些', '此', '彼',
#
#     # 系词 / 判断 / 存在
#     '是', '不是', '有', '没有', '无', '不存在',
#
#     # 介词 / 介词性结构（常见搭配）
#     '在', '从', '到', '向', '往', '对', '对于', '关于',
#     '把', '被', '给', '跟', '和', '与', '同', '比',
#     '按', '按照', '根据', '通过', '以', '用', '为', '为了', '由于',
#
#     # 连词（转折/因果/条件/递进/并列）
#     '和', '与', '及', '以及', '并', '并且', '而', '而且',
#     '或', '或者', '还是',
#     '但', '但是', '不过', '然而', '可是',
#     '因为', '由于', '所以', '因此', '因而',
#     '如果', '若', '假如', '即使', '即便',
#     '当', '当时', '当…时', '一旦', '然后', '于是',
#     '此外', '另外', '同时',
#
#     # 情态/助动/高频动词（按需保留或删除）
#     '会', '能', '可以', '可能', '应该', '必须', '需要', '想', '要',
#     '进行', '进行中', '成为', '具有', '包括', '属于',
#
#     # 副词（程度/范围/频率/时间等）
#     '很', '非常', '十分', '更', '最',
#     '也', '都', '就', '还', '又', '再', '只', '才',
#     '已经', '仍', '仍然', '正在', '刚', '刚刚',
#     '目前', '现在', '当前',
#     '今天', '昨日', '昨天', '明日', '明天',
#     '以前', '之后', '后来', '一直', '从来', '经常', '常常', '有时', '有时候', '偶尔',
#
#     # 疑问词
#     '什么', '谁', '哪', '哪里', '哪儿', '哪个', '哪些',
#     '什么时候', '何时', '为何', '为什么',
#     '怎么', '怎么样',
#     '多少', '几', '是否',
#
#     # 否定
#     '不', '没', '没有', '无', '无法', '未', '并非',
#
#     # 数量词/泛化词（可选）
#     '一些', '某', '某些', '每', '各', '任何', '所有', '全部', '一切',
#     '大多数', '大部分', '很多', '许多', '少数', '少量', '若干', '几个',
#
#     # 其他功能性常用词
#     '比如', '例如', '例如说', '就是', '即', '即是',
#     '等等', '等', '等于',
# }




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