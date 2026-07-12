# Bengali Continual Learning — rulare pe alt PC

Pipeline de continual learning cu descoperire de clase noi (OOD + clustering) pe
setul Bengali (L3Cube IndicNews, 10 clase). Distribuția curentă este **Opțiunea B**.

## Distribuția claselor (Opțiunea B)

Dimensiuni **4 / 2 / 2 / 2**. Doar 2 clase geografice (national/international) stau
în baseline supervizat, alături de 2 teme clar separabile (sports/technology).
Celelalte 2 geografice (state, kolkata) sunt descoperite în pași DIFERIȚI (T1, T2),
fiecare împerecheat cu o temă distinctă — ca să nu se contopească la clustering și
să nu supraîncarce baseline-ul cu 4 clase confuzabile (problema din Opțiunea A).

| Etapă | Label-uri | Clase |
|-------|-----------|-------|
| Baseline | 0–3 | national, international, sports, technology |
| T1 (nou) | 4–5 | state, entertainment |
| T2 (nou) | 6–7 | kolkata, politics |
| T3 (nou) | 8–9 | lifestyle, editorial |

> Notă: modelul din `config.py` este încă `bert-base-uncased`, care nu a fost
> pre-antrenat pe bengali (sparge textul în caractere). Pentru rezultate reale,
> schimbă `MODEL_NAME` cu un model multilingv/bengali (ex.
> `bert-base-multilingual-cased` sau `l3cube-pune/bengali-bert`).

Datele split sunt deja incluse în `datasets/Bengali/bengali_splits/`
(`train/val/test_1/test_2/test_3.csv`). Nu e nevoie să le regenerezi.

## Setup

```bash
# din radacina repo-ului
python -m venv venv && source venv/bin/activate      # optional
pip install -r clustering/Bengali_new/requirements.txt
```

CUDA: `torch==2.5.1` a fost instalat cu cu124. Dacă placa/driverul diferă,
instalează varianta de torch potrivită de pe pytorch.org, apoi restul din requirements.

## Rulare (din folderul Bengali_new)

```bash
cd clustering/Bengali_new

# tot lantul, in ordine:
python train_baseline.py       # 1. antreneaza baseline pe clasele 0-3
python pipeline_t1.py          # 2. OOD + clustering clase noi (T1)
python train_t1.py             # 3. antrenare incrementala
python evaluate_t1.py          # 4. evaluare T1
python pipeline_t2.py          # 5-7. idem T2
python train_t2.py
python evaluate_t2.py
python pipeline_t3.py          # 8-10. idem T3
python train_t3.py
python evaluate_t3.py
```

Sau folosind orchestratorul:

```bash
python run_full_pipeline.py            # tot lantul
python run_full_pipeline.py --t1       # doar T1 (pasii 1-3 din script)
```

## Rezultate

Modelele și rezultatele apar în `clustering/Bengali_new/bengali_cl_outputs_1/`
(ignorat de git): `model_baseline/`, `model_t1..3/`, `eval_t*_results.json`,
`test_*_results.pkl`.

## Regenerarea split-urilor (optional)

Distribuția e definită în `datasets/Bengali/prepare_bengali_cl_splits.py`
(`NEW_ORDER` + `CLASS_NAMES`). Necesită CSV-ul sursă `bengali_combined.csv`
(NU e în git — 235MB). Cu seed=42 rezultatul e determinist:

```bash
python datasets/Bengali/prepare_bengali_cl_splits.py \
  --input datasets/Bengali/bengali_combined.csv \
  --output datasets/Bengali/bengali_splits --train-ratio 0.40 --seed 42
```
