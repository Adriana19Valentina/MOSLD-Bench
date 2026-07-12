# config.py - Central Configuration for Continual Learning Pipeline
# =====================================================================
# RUSSIAN LANGUAGE CONFIGURATION
# =====================================================================

import os
from spacy.lang.ro.stop_words import STOP_WORDS as RO_STOP_WORDS
# =========================================================================
# LANGUAGE & MODEL SETTINGS
# =========================================================================
STOP_WORDS_RO = set(RO_STOP_WORDS)
LANGUAGE = 'Romanian'

# BERT Model for Russian
MODEL_NAME = 'readerbench/RoBERT-base'

# =========================================================================
# DATASET PATHS - CHANGE THESE FOR YOUR DATASET LOCATION
# =========================================================================

DATASET_DIR = '/home/alin/Desktop/ContinualLearning/datasets/Romanian/Romanian/romanian_splits'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXISTING_RESULTS_DIR = os.path.join(BASE_DIR, 'romanian_cl_outputs_1')
MODEL_BASELINE_DIR = os.path.join(EXISTING_RESULTS_DIR, 'model_baseline')
OUTPUT_DIR = os.path.join(BASE_DIR, 'english_cl_contrastive')
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
SEED = 42
CONTRASTIVE_WEIGHT = 0.3
TEMPERATURE = 0.07
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
BASELINE_LABELS = [0, 1, 2, 3]

# New classes added at each test step
TEST_1_NEW_LABELS = [4, 5]  # New in test_1.csv
TEST_2_NEW_LABELS = [6, 7]  # New in test_2.csv
TEST_3_NEW_LABELS = [8, 9]  # New in test_3.csv

CLASS_NAMES = {
    0: 'sanatate',
    1: 'educatie',
    2: 'natura',
    3: 'politica',  # T1 - easy
    4: 'financiar',  # T1 - easy
    5: 'sport',  # T2 - medium
    6: 'vacante',  # T2 - medium
    7: 'tehnologie',  # T3 - hard
    8: 'stiinta', # T3 - hard,
    9:'cultura'
}
OOD_TARGET_TPR = 0.85  # 95% din known să fie clasificate corect
OOD_PREFERRED_METHOD = None  # None = auto-select, 'energy', sau 'msp'
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

OUTPUT_DIR = './romanian_cl_outputs_1'

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

STOP_WORDS = {
    # Romanian stop words - basic
    'de', 'la', 'a', 'în', 'in', 'si', 'și', 'cu', 'pe', 'din', 'ce', 'care', 'nu', 'o', 'un', 'una',
    'este', 'sunt', 'am', 'ai', 'are', 'avea', 'fost', 'fi', 'fiind', 'pentru', 'sau', 'dar',
    'mai', 'se', 'să', 'sa', 'ca', 'le', 'lui', 'ei', 'lor', 'al', 'ale', 'acestei', 'acest', 'această',
    'aceasta', 'acesta', 'acestea', 'aceștia', 'aceste', 'aceşti', 'acei', 'acea', 'acela',
    'precum', 'prin', 'despre', 'spre', 'după', 'dupa', 'între', 'intre', 'sub', 'fără', 'fara', 'peste', 'până',
    'pana',
    'când', 'cand', 'cum', 'unde', 'cât', 'cat', 'câți', 'cati', 'câte', 'cate', 'câtă', 'cata', 'cine', 'tot', 'toți',
    'toti', 'toate',
    'foarte', 'doar', 'chiar', 'încă', 'inca', 'deja', 'apoi', 'iar', 'nici', 'ori', 'fie', 'decât', 'decat',
    'așa', 'asa', 'astfel', 'deci', 'însă', 'insa', 'totuși', 'totusi', 'deși', 'desi', 'dacă', 'daca', 'atunci',
    'acum', 'aici',
    'cel', 'cea', 'cei', 'cele', 'alt', 'alta', 'alți', 'alti', 'alte', 'altul', 'alta',
    'ne', 'vă', 'va', 'te', 'mă', 'ma', 'îl', 'il', 'îi', 'ii', 'le', 'li', 'ni', 'vi',
    'eu', 'tu', 'el', 'ea', 'noi', 'voi', 'ei', 'ele', 'meu', 'mea', 'mei', 'mele',
    'tău', 'tau', 'ta', 'tăi', 'tai', 'tale', 'său', 'sau', 'sa', 'săi', 'sai', 'sale',
    'nostru', 'noastră', 'noastra', 'noștri', 'nostri', 'noastre',
    'vostru', 'voastră', 'voastra', 'voștri', 'vostri', 'voastre',
    'sine', 'său', 'sa', 'săi', 'sale',

    # Romanian stop words - verbs (common forms)
    'poate', 'pot', 'putem', 'puteți', 'puteti', 'putea', 'putând', 'putand',
    'face', 'fac', 'facem', 'faceți', 'faceti', 'făcut', 'facut', 'făcea', 'facea',
    'avea', 'avem', 'aveți', 'aveti', 'avut', 'având', 'avand',
    'spune', 'spus', 'spunem', 'spuneți', 'spuneti', 'spunea', 'spunând', 'spunand',
    'zice', 'zis', 'zicem', 'ziceți', 'ziceti', 'zicea', 'zicând', 'zicand',
    'vrea', 'vreau', 'vrei', 'vrem', 'vreți', 'vreti', 'vrut', 'vrând', 'vrand',
    'ști', 'stiu', 'știu', 'știi', 'stii', 'știe', 'stie', 'știm', 'stim', 'știți', 'stiti', 'știut', 'stiut',
    'trebui', 'trebuie', 'trebuia', 'trebuit',
    'veni', 'vine', 'vin', 'venim', 'veniți', 'veniti', 'venit', 'venea', 'venind',
    'lua', 'iau', 'iei', 'ia', 'luăm', 'luam', 'luați', 'luati', 'luat', 'luând', 'luand',
    'da', 'dau', 'dai', 'dăm', 'dam', 'dați', 'dati', 'dat', 'dând', 'dand', 'dea',
    'vedea', 'văd', 'vad', 'vezi', 'vede', 'vedem', 'vedeți', 'vedeti', 'văzut', 'vazut',
    'crede', 'cred', 'crezi', 'credem', 'credeți', 'credeti', 'crezut', 'crezând', 'crezand',
    'pune', 'pun', 'pui', 'punem', 'puneți', 'puneti', 'pus', 'punând', 'punand',
    'rămâne', 'ramane', 'rămân', 'raman', 'rămâi', 'ramai', 'rămânem', 'ramanem', 'rămas', 'ramas',
    'ajunge', 'ajung', 'ajungi', 'ajungem', 'ajungeți', 'ajungeti', 'ajuns',
    'merge', 'merg', 'mergi', 'mergem', 'mergeți', 'mergeti', 'mers', 'mergând', 'mergand',
    'afla', 'aflu', 'afli', 'aflăm', 'aflam', 'aflați', 'aflati', 'aflat',
    'părea', 'parea', 'par', 'pari', 'pare', 'părem', 'parem', 'părut', 'parut',
    'exista', 'există', 'existat', 'existând', 'existand',

    # Romanian stop words - adjectives/adverbs (common)
    'mare', 'mari', 'mic', 'mică', 'mica', 'mici',
    'bun', 'bună', 'buna', 'buni', 'bune',
    'nou', 'nouă', 'noua', 'noi', 'noul', 'noii',
    'primul', 'prima', 'primii', 'primele', 'prim',
    'ultim', 'ultima', 'ultimul', 'ultimii', 'ultimele',
    'multe', 'mult', 'mulți', 'multi', 'multă', 'multa',
    'puțin', 'putin', 'puțini', 'putini', 'puține', 'putine', 'puțină', 'putina',
    'parte', 'părți', 'parti',
    'fel', 'mod', 'chip',
    'loc', 'locul', 'locuri',
    'timp', 'timpul', 'timpuri',
    'ani', 'an', 'anul', 'anului',
    'zi', 'zile', 'zilele', 'ziua',
    'lună', 'luna', 'luni', 'lunii',

    # Romanian stop words - pronouns/articles
    'unul', 'una', 'unii', 'unele', 'unui', 'unei',
    'niciun', 'nicio', 'niciunul', 'niciuna',
    'fiecare', 'orice', 'oricare', 'oricum', 'oricât', 'oricat',
    'cineva', 'ceva', 'altcineva', 'altceva',
    'nimeni', 'nimic',

    # Romanian stop words - prepositions/conjunctions
    'contra', 'versus', 'conform', 'potrivit', 'privind',
    'înainte', 'inainte', 'înapoi', 'inapoi',
    'afară', 'afara', 'înăuntru', 'inăuntru', 'inauntru',
    'sus', 'jos', 'stânga', 'stanga', 'dreapta',
    'asemenea', 'asemeni',

    # Common Romanian words that appear frequently but aren't content
    'ine', 'ție', 'tie',  # fragments
    'fost', 'era', 'erau', 'erat',
    'fiind', 'fiindcă', 'fiindca',
    'încât', 'incat', 'întrucât', 'intrucat',
    'deoarece', 'fiindcă', 'fiindca',
    'totuși', 'totusi', 'oricum',
    'însă', 'insa', 'deci', 'așadar', 'asadar',

    # Numbers as words
    'unu', 'doi', 'două', 'doua', 'trei', 'patru', 'cinci', 'șase', 'sase',
    'șapte', 'sapte', 'opt', 'nouă', 'noua', 'zece',
    'sută', 'suta', 'mie', 'milion', 'milioane', 'miliard', 'miliarde',

    # Currency/measurement words
    'lei', 'euro', 'dolari', 'dolar', 'procent', 'procente',

    # English stop words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if', 'because',
    'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',

    # Common words to exclude
    'com', 'www', 'http', 'https', 'html', 'php', 'asp', 'jpg', 'png', 'gif',
    'nbsp', 'quot', 'amp', 'lt', 'gt',
}





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