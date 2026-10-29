# MOSLD-Bench

A multilingual benchmark for **open-set continual learning** in text classification.
A model is trained on a small set of known classes (KKC) and then receives three
successive test streams, each one introducing new, unlabeled classes (UUC). At each
step the pipeline must:

1. detect samples from unknown classes (energy-based OOD detection),
2. group them into new classes (k-means clustering, TF-IDF keywords per cluster),
3. retrain the classifier on the known and newly discovered classes, with replay
   from earlier steps,
4. be evaluated against the ground truth (clusters are mapped to real classes with
   the Hungarian algorithm).

## Repository structure

```
MOSLD/
├── dataset/
│   ├── Arabic/  Bengali/  Chinese/  English/  French/  hungarian/
│   └── Italian/  Japanese/  Romanian/  Russian/  Spanish/  Turkish/
│       ├── train.csv      # known classes only (labels 0-3)
│       ├── val.csv        # known classes only
│       ├── test_1.csv     # known classes + 1st batch of new classes
│       ├── test_2.csv     # + 2nd batch of new classes
│       └── test_3.csv     # + 3rd batch of new classes
└── Pipelines/         # experiment code
```

### Data format

Every CSV file has two columns:

| column    | description                          |
|-----------|--------------------------------------|
| `content` | the text (some languages anonymize named entities as `$NE$`) |
| `label`   | integer class id                     |

Each test set adds new classes cumulatively. The number of new classes depends on
the language:

| Language                     | train / val | test_1 | test_2 | test_3 |
|------------------------------|-------------|--------|--------|--------|
| English, Chinese             | 0-3         | 0-6    | 0-9    | 0-13   |
| Turkish                      | 0-3         | 0-6    | 0-9    | 0-12   |
| All other languages          | 0-3         | 0-5    | 0-7    | 0-9    |

### Category labels

Category labels per language and evaluation split. Each test set is cumulative:
Test 1 contains the known classes (KKCs) plus the Test 1 unknown classes (UUCs),
Test 2 contains all Test 1 classes plus the Test 2 UUCs, and so on. The plus sign
marks the new UUCs at each stage.

| Language | Baseline (Train & Val) | Test 1 | Test 2 | Test 3 |
|----------|------------------------|--------|--------|--------|
| Arabic | sports, politics, economy, miscellaneous | + art, technology | + medicine, culture | + religion, society |
| Bengali | politics, technology, international, national | + sports, lifestyle | + entertainment, editorial | + Kolkata, state / region |
| Chinese | sports, entertainment, home, real estate | + education, fashion, politics & current affairs | + games, society, technology | + stocks, finance, lottery, horoscope |
| English | company, athlete, mean of transportation, natural place | + animal, album, building | + plant, film, written work | + artist, village, educational institution, office holder |
| French | politics, travel, international, science | + culture, sports | + media, economy | + technology, lifestyle |
| Hungarian | technology, sports, automotive, economy | + home, entertainment | + international, culture | + health, lifestyle |
| Italian | sports, politics, arts & entertainment, economy & finance | + society, health | + science & technology, environment | + travel & tourism, lifestyle |
| Japanese | medical, technology, entertainment, sports | + business / company, family | + environment, political situation | + welfare, incidents / events |
| Romanian | health, education, nature, politics | + finance, sports | + vacation, technology | + science, culture |
| Russian | conflicts, economy, politics, science | + sports, travel | + culture, climate | + health, society |
| Spanish | culture, economy, sports, international | + politics, science | + opinion, impactful / shocking | + technology, health |
| Turkish | sports, general, current affairs, world | + economy, magazine / entertainment, planet / environment | + politics, Turkey-specific, health | + culture & arts, technology, life / lifestyle |

Check the label ranges of a language before you edit the config:

```bash
python -c "import pandas as pd; [print(s, sorted(pd.read_csv(f'dataset/Romanian/{s}.csv').label.unique())) for s in ['train','test_1','test_2','test_3']]"
```

### `Pipelines/`

| File | Role |
|------|------|
| `config.py` | All settings: language, model, data paths, class splits, hyperparameters |
| `ood_detection.py` | Energy-based OOD detection used to flag unknown samples |
| `train_baseline.py` | Trains the baseline classifier on `train.csv` (known classes) |
| `pipeline_t1.py` / `t2` / `t3` | OOD detection, k-means clustering and TF-IDF keywords on `test_N.csv`; writes `test_N_processed.csv` + `test_N_results.pkl` |
| `train_t1.py` / `t2` / `t3` | Retrains the classifier on known + pseudo-labeled classes (T2/T3 continue from the previous model) |
| `evaluate_t1.py` / `t2` / `t3` | Evaluates step N against ground truth (Hungarian mapping) → `eval_test_N.json` |
| `hungarian_mapping.py` | Helper for cluster → class assignment and metrics |
| `run_full_pipeline.py` | Runs every step above in order |
| `contrastive_trainer.py` | `Trainer` subclass with a cross-entropy + contrastive loss |
| `generate_keyword_embeddings.py` | Embeds the cluster keywords for the contrastive loss |
| `train_t*_contrastive.py`, `run_contrastive.py` | Contrastive variant of the retraining step |

## Language-specific BERT models

Each language uses its own BERT model as the starting point for fine-tuning.
Preliminary experiments with multilingual BERT (mBERT) gave significantly worse
results in several languages, so we selected language-specific models. The same
models are used for both the naive approach (which ignores new classes) and the
proposed OSLD approaches, so the comparison between methods is fair.

All models use the base architecture, adapted for uncased text processing.

| Language  | Model        |
|-----------|--------------|
| Arabic    | AraBERT      |
| Bengali   | BanglaBERT   |
| Chinese   | Chinese BERT |
| English   | BERT         |
| French    | French BERT  |
| Hungarian | huBERT       |
| Italian   | Italian BERT |
| Japanese  | TohokuBERT   |
| Romanian  | RoBERT       |
| Russian   | ruBERT       |
| Spanish   | Spanish BERT |
| Turkish   | BERTurk      |

Set the matching Hugging Face model id as `MODEL_NAME` in `Pipelines/config.py`.

## LLM prompts

The LLM-based approach uses Qwen3.5-9B with two prompts: (a) open-set discovery
and generation of class labels, and (b) canonicalization/merging of the resulting
label set. Text files containing the training and test samples were attached to
both prompts.

### (a) Open-vocabulary classification and label discovery

```text
You are given a list of known categories. The text may or may not belong to one of
them. The known list is INCOMPLETE on purpose; many texts belong to categories that
are NOT on the list.

Decision rule:
1. Read the text and decide what it is fundamentally about (a person, an
   organization, a place, a film, a book, a piece of music, a plant, an animal,
   a vehicle, a building, etc.).
2. If that concept is one of the known categories, output that category name
   EXACTLY as written below.
3. Otherwise, INVENT a new category name that names the concept (one or two words,
   TitleCase, e.g. "Animal", "Film", "WrittenWork", "Building"). Do not force a
   wrong fit just to use a known category.

Output format: ONLY the category name. No explanation, no quotes, no punctuation,
no extra words.

Known categories:
{class_list}

Text:
{content}
```

### (b) Per-epoch label vocabulary unification

```text
You are organising a list of text-classification category labels. The list
contains duplicates, near-duplicates, sub-types of the same broader concept, and
trivially-different surface forms. Most cosmetic differences (case, whitespace,
hyphenation, British vs American spelling, plural vs singular) have already been
collapsed by a deterministic pass before you got this list --- the labels you see
now are mostly genuine concepts, but many of them still overlap and should be
merged.

Your job: produce the smallest reasonable canonical list and a mapping that sends
every input label to a canonical label.

GRANULARITY REFERENCE --- the originally-known categories that established the
benchmark's granularity. Your canonical labels should sit at roughly this level of
detail. Do NOT propose canonicals more fine-grained than the reference; if a label
is a sub-type of a reference category, merge it into a broader concept at the
reference level.

PROTECTED labels (must appear in the canonical list, byte-for-byte unchanged):
{protected}

Rules:
1. Every label in PROTECTED must appear in canonical, exactly as written.
2. Be aggressive about merging: synonyms ("Vehicle"/"MeanOfTransportation"),
   sub-types of one broad concept ("SpaceMission"/"Spacecraft" -> one canonical),
   closely related concepts the reference treats as one bucket.
3. Do NOT split protected labels by language or style.
4. Only keep separate canonicals for genuinely distinct, non-overlapping concepts.
5. Prefer fewer, broader canonicals. If in doubt, merge.

Output: Respond ONLY with JSON in this exact schema, no prose, no fences:
{"canonical": ["..."],
 "mapping": {"input_label": "canonical_label"}}

Every input label must appear as a key in mapping, and every value of mapping must
appear in canonical.
```

## Installation

Python 3.10+ is recommended. A GPU is strongly recommended.

```bash
pip install torch transformers datasets scikit-learn scipy pandas numpy
```

## Configuration

Edit `Pipelines/config.py` before each run:

```python
LANGUAGE   = 'Romanian'
MODEL_NAME = '...'                            # language-specific BERT, see "Language-specific BERT models"
DATASET_DIR = '../dataset/Romanian'           # relative to Pipelines/
OUTPUT_DIR  = './romanian_cl_outputs'

BASELINE_LABELS    = [0, 1, 2, 3]
TEST_1_NEW_LABELS  = [4, 5]                   # must match the label table above
TEST_2_NEW_LABELS  = [6, 7]
TEST_3_NEW_LABELS  = [8, 9]
CLASS_NAMES = {0: '...', 1: '...', ...}
```

Things to watch in `config.py`:

- **Paths are relative to `Pipelines/`.** The scripts run with `Pipelines/` as their
  working directory, so the default `DATASET_DIR = './english_splits'` will not find
  the data. Point it to the language folder, e.g. `'../dataset/Romanian'`.
- `OUTPUT_DIR` and `MAX_LENGTH` are each defined twice; **the later definitions win**
  (`OUTPUT_DIR = './english_cl_outputs_1'`, `MAX_LENGTH = 128`).
- `EXISTING_RESULTS_DIR` (used by the contrastive scripts) must point to the
  `OUTPUT_DIR` of a finished standard run.
- TF-IDF keyword extraction uses English stop words (`STOP_WORDS_EN`). Replace
  them for other languages if you need better keywords.
- Clustering: `K_MIN` / `K_MAX` bound the k search; `FORCE_K_T1..T3` fix k
  manually; `OOD_THRESHOLD` / `OOD_TARGET_TPR` control the OOD threshold.

Check the active configuration with:

```bash
cd Pipelines
python config.py
```

## Running

All commands run from `Pipelines/`:

```bash
cd Pipelines
```

### Full pipeline

```bash
python run_full_pipeline.py            # all steps
python run_full_pipeline.py --list     # list the steps with their numbers
python run_full_pipeline.py --start 5 --end 7    # a range of steps
python run_full_pipeline.py --step train_t2      # a single step, matched by name
```

Steps, in order:

| # | Step | Reads | Writes (in `OUTPUT_DIR`) |
|---|------|-------|--------------------------|
| 1 | `train_baseline.py` | `train.csv`, `val.csv` | `model_baseline/` |
| 2 | `pipeline_t1.py` | `test_1.csv`, `train.csv` | `test_1_processed.csv`, `test_1_results.pkl` |
| 3 | `train_t1.py` | `test_1_processed.csv` | `model_t1/` |
| 4 | `evaluate_t1.py` | `model_t1/`, `test_1.csv` | `eval_test_1.json` |
| 5 | `pipeline_t2.py` | `test_2.csv` + replay from T1 | `test_2_processed.csv`, `test_2_results.pkl` |
| 6 | `train_t2.py` | `model_t1/` | `model_t2/` |
| 7 | `evaluate_t2.py` | `model_t2/`, `test_2.csv` | `eval_test_2.json` |
| 8 | `pipeline_t3.py` | `test_3.csv` + replay from T2 | `test_3_processed.csv`, `test_3_results.pkl` |
| 9 | `train_t3.py` | `model_t2/` | `model_t3/` |
| 10 | `evaluate_t3.py` | `model_t3/`, `test_3.csv` | `eval_test_3.json` |
| 11 | `generate_final_summary.py` | | final results JSON |

Each step depends on the previous ones, so you can resume a stopped run with
`--start N`. You can also run any script on its own, for example `python pipeline_t2.py`.

### Contrastive variant

This variant needs a finished standard run (at least steps 1, 2, 5 and 8), because it
reuses the baseline model and the processed/clustering files from `EXISTING_RESULTS_DIR`.

```bash
python run_contrastive.py              # T1 → T2 → T3 with CE + contrastive loss
python run_contrastive.py --step 2     # a single step
```

Results go to the contrastive `OUTPUT_DIR` (`english_cl_contrastive` by default).
`CONTRASTIVE_WEIGHT` and `TEMPERATURE` in `config.py` control the loss.

## Known issues

- `generate_final_summary.py` (step 11) is not in the repository, so the last step
  of `run_full_pipeline.py` reports a failure. Steps 1-10 are not affected; run
  `python run_full_pipeline.py --end 10` to skip it.
- The `--t1` / `--t2` / `--t3` shortcuts in `run_full_pipeline.py` use outdated step
  ranges (1-3 / 4-6 / 7-9). Use `--start` / `--end` with the table above instead.
- The execution log is always written to `russian_cl_outputs/pipeline_execution_log.json`,
  whatever `OUTPUT_DIR` is set to.

## License

This repository (code and data) is released under the CC BY-NC-SA 4.0 license.


