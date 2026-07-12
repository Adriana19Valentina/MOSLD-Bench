#!/bin/bash
# english_contrastive.sh - Pipeline complet Continual Learning cu CONTRASTIVE LOSS
# Cu mapare automată și logging complet

# Culori pentru output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
NC = '\033[0m'  # No Color

# Timestamp
TIMESTAMP =$(date + "%Y%m%d_%H%M%S")
LOG_DIR = "logs_contrastive_${TIMESTAMP}"
mkdir - p ${LOG_DIR}

MAIN_LOG = "${LOG_DIR}/continual_learning_contrastive_full.log"

# Funcție pentru logging
log_step()
{
    echo - e
"${BLUE}=====================================================================${NC}"
echo - e
"${BLUE}$1${NC}"
echo - e
"${BLUE}=====================================================================${NC}"
echo
""
echo
"=====================================================================" >> ${MAIN_LOG}
echo
"$1" >> ${MAIN_LOG}
echo
"=====================================================================" >> ${MAIN_LOG}
echo
"" >> ${MAIN_LOG}
}

log_success()
{
    echo - e
"${GREEN}✓ $1${NC}"
echo
"✓ $1" >> ${MAIN_LOG}
}

log_error()
{
    echo - e
"${RED}✗ $1${NC}"
echo
"✗ $1" >> ${MAIN_LOG}
}

log_info()
{
    echo - e
"${YELLOW}→ $1${NC}"
echo
"→ $1" >> ${MAIN_LOG}
}

log_contrastive()
{
    echo - e
"${MAGENTA}🔥 $1${NC}"
echo
"🔥 $1" >> ${MAIN_LOG}
}

# Start
echo - e
"${MAGENTA}=====================================================================${NC}"
echo - e
"${MAGENTA}CONTINUAL LEARNING PIPELINE - CONTRASTIVE LOSS${NC}"
echo - e
"${MAGENTA}=====================================================================${NC}"
echo
""
echo
"Timestamp: ${TIMESTAMP}"
echo
"Log directory: ${LOG_DIR}"
echo
"Main log: ${MAIN_LOG}"
echo
""

{
    echo
"======================================================================"
echo
"CONTINUAL LEARNING PIPELINE - CONTRASTIVE LOSS"
echo
"======================================================================"
echo
""
echo
"Timestamp: ${TIMESTAMP}"
echo
"Started at: $(date)"
echo
""
} >> ${MAIN_LOG}

# ============================================================================
# BASELINE TRAINING (opțional - doar dacă nu există)
# ============================================================================

if [ ! -d "./ckpt_baseline/final"]; then
log_step
"ETAPA 0: BASELINE TRAINING (4 clase: 0,2,7,12)"
log_info
"Training model baseline pe clase cunoscute..."

python
train_baseline.py
2 > & 1 | tee ${LOG_DIR} / 00
_baseline_training.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Baseline training complet"
else
log_error
"Baseline training eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}
else
log_info
"Baseline model deja existent, se sare peste training"
fi

# ============================================================================
# TASK 1: TEST_1 (clase noi: 1, 3, 11) - CONTRASTIVE
# ============================================================================

log_step
"TASK 1: CONTINUAL LEARNING PE TEST_1 CU CONTRASTIVE LOSS"
log_contrastive
"Folosim Combined Loss: Cross-Entropy + Contrastive"

# T1.1: Pipeline (clustering + sample selection + mapare automată + keyword embeddings)
log_info
"T1.1: Pipeline - clustering, mapare automată și keyword embeddings..."
python
pipeline_t1_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 01
_t1_pipeline_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Pipeline T1 contrastive complet"

# Extrage metrici
echo
"" >> ${MAIN_LOG}
echo
"METRICI CLUSTERING T1:" >> ${MAIN_LOG}
grep - A
5
"METRICI CLUSTERING" ${LOG_DIR} / 01
_t1_pipeline_contrastive.log >> ${MAIN_LOG}
echo
"" >> ${MAIN_LOG}
echo
"MAPARE AUTOMATĂ T1:" >> ${MAIN_LOG}
grep
"Average mapping similarity" ${LOG_DIR} / 01
_t1_pipeline_contrastive.log >> ${MAIN_LOG}
grep
"✓ Cluster.*→ Class" ${LOG_DIR} / 01
_t1_pipeline_contrastive.log >> ${MAIN_LOG}
echo
"" >> ${MAIN_LOG}
echo
"KEYWORD EMBEDDINGS T1:" >> ${MAIN_LOG}
grep - A
5
"KEYWORD EMBEDDINGS" ${LOG_DIR} / 01
_t1_pipeline_contrastive.log | tail - 4 >> ${MAIN_LOG}
else
log_error
"Pipeline T1 contrastive eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# T1.2: Training cu contrastive loss
log_info
"T1.2: Training cu Combined Loss (CE + Contrastive)..."
log_contrastive
"Contrastive weight: 0.5, Temperature: 0.5"

python
train_cl_t1_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 02
_t1_training_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Training T1 contrastive complet"

# Extrage loss components
echo
"" >> ${MAIN_LOG}
echo
"TRAINING LOSSES T1 (CONTRASTIVE):" >> ${MAIN_LOG}
grep
"CE:.*Contrastive:.*Total:" ${LOG_DIR} / 02
_t1_training_contrastive.log | head - 5 >> ${MAIN_LOG}
echo
"..." >> ${MAIN_LOG}
grep
"CE:.*Contrastive:.*Total:" ${LOG_DIR} / 02
_t1_training_contrastive.log | tail - 3 >> ${MAIN_LOG}
grep
"Final training loss" ${LOG_DIR} / 02
_t1_training_contrastive.log >> ${MAIN_LOG}
else
log_error
"Training T1 contrastive eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# T1.3: Evaluare
log_info
"T1.3: Evaluare model T1 contrastive..."
python
evaluate_cl_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 03
_t1_evaluation_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Evaluare T1 contrastive completă"

# Extrage rezultate
echo
"" >> ${MAIN_LOG}
echo
"REZULTATE TEST_1 (CONTRASTIVE):" >> ${MAIN_LOG}
grep - A
15
"REZUMAT" ${LOG_DIR} / 03
_t1_evaluation_contrastive.log >> ${MAIN_LOG}
else
log_error
"Evaluare T1 contrastive eșuată"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# ============================================================================
# TASK 2: TEST_2 (clase noi: 4, 6, 8) - CONTRASTIVE
# ============================================================================

log_step
"TASK 2: CONTINUAL LEARNING PE TEST_2 CU CONTRASTIVE LOSS"

# T2.0: Evaluare pre-CL (AUROC)
log_info
"T2.0: Evaluare pre-CL - AUROC detection..."
python
evaluate_pre_t2.py
2 > & 1 | tee ${LOG_DIR} / 04
_t2_pre_evaluation.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Evaluare pre-CL T2 completă"
grep
"AUROC" ${LOG_DIR} / 04
_t2_pre_evaluation.log >> ${MAIN_LOG}
fi
echo
"" >> ${MAIN_LOG}

# T2.1: Pipeline contrastive
log_info
"T2.1: Pipeline T2 contrastive..."
python
pipeline_t2_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 05
_t2_pipeline_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Pipeline T2 contrastive complet"
echo
"" >> ${MAIN_LOG}
echo
"METRICI CLUSTERING T2:" >> ${MAIN_LOG}
grep - A
5
"METRICI CLUSTERING" ${LOG_DIR} / 05
_t2_pipeline_contrastive.log >> ${MAIN_LOG}
echo
"MAPARE AUTOMATĂ T2:" >> ${MAIN_LOG}
grep
"Average mapping similarity" ${LOG_DIR} / 05
_t2_pipeline_contrastive.log >> ${MAIN_LOG}
else
log_error
"Pipeline T2 contrastive eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# T2.2: Training contrastive
log_info
"T2.2: Training T2 cu contrastive loss..."
python
train_cl_t2_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 06
_t2_training_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Training T2 contrastive complet"
echo
"" >> ${MAIN_LOG}
echo
"TRAINING LOSSES T2 (CONTRASTIVE):" >> ${MAIN_LOG}
grep
"CE:.*Contrastive:.*Total:" ${LOG_DIR} / 06
_t2_training_contrastive.log | head - 3 >> ${MAIN_LOG}
grep
"Final training loss" ${LOG_DIR} / 06
_t2_training_contrastive.log >> ${MAIN_LOG}
else
log_error
"Training T2 contrastive eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# T2.3: Evaluare
log_info
"T2.3: Evaluare T2 contrastive..."
python
evaluate_cl_t2_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 07
_t2_evaluation_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Evaluare T2 contrastive completă"
echo
"" >> ${MAIN_LOG}
echo
"REZULTATE TEST_2 (CONTRASTIVE):" >> ${MAIN_LOG}
grep - A
20
"REZUMAT FINAL" ${LOG_DIR} / 07
_t2_evaluation_contrastive.log >> ${MAIN_LOG}
else
log_error
"Evaluare T2 contrastive eșuată"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# ============================================================================
# TASK 3: TEST_3 (clase noi: 5, 9, 10, 13) - CONTRASTIVE
# ============================================================================

log_step
"TASK 3: CONTINUAL LEARNING PE TEST_3 CU CONTRASTIVE LOSS"

# T3.0: Evaluare pre-CL (AUROC)
log_info
"T3.0: Evaluare pre-CL - AUROC detection..."
python
evaluate_pre_t3.py
2 > & 1 | tee ${LOG_DIR} / 0
8
_t3_pre_evaluation.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Evaluare pre-CL T3 completă"
grep
"AUROC" ${LOG_DIR} / 0
8
_t3_pre_evaluation.log >> ${MAIN_LOG}
fi
echo
"" >> ${MAIN_LOG}

# T3.1: Pipeline contrastive
log_info
"T3.1: Pipeline T3 contrastive..."
python
pipeline_t3_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 0
9
_t3_pipeline_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Pipeline T3 contrastive complet"
echo
"" >> ${MAIN_LOG}
echo
"METRICI CLUSTERING T3:" >> ${MAIN_LOG}
grep - A
5
"METRICI CLUSTERING" ${LOG_DIR} / 0
9
_t3_pipeline_contrastive.log >> ${MAIN_LOG}
else
log_error
"Pipeline T3 contrastive eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# T3.2: Training contrastive
log_info
"T3.2: Training T3 cu contrastive loss..."
python
train_cl_t3_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 10
_t3_training_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Training T3 contrastive complet"
echo
"" >> ${MAIN_LOG}
echo
"TRAINING LOSSES T3 (CONTRASTIVE):" >> ${MAIN_LOG}
grep
"Final training loss" ${LOG_DIR} / 10
_t3_training_contrastive.log >> ${MAIN_LOG}
else
log_error
"Training T3 contrastive eșuat"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# T3.3: Evaluare
log_info
"T3.3: Evaluare T3 contrastive..."
python
evaluate_cl_t3_contrastive.py
2 > & 1 | tee ${LOG_DIR} / 11
_t3_evaluation_contrastive.log

if [ ${PIPESTATUS[0]} -eq 0]; then
log_success
"Evaluare T3 contrastive completă"
echo
"" >> ${MAIN_LOG}
echo
"REZULTATE TEST_3 (CONTRASTIVE):" >> ${MAIN_LOG}
grep - A
25
"REZUMAT FINAL" ${LOG_DIR} / 11
_t3_evaluation_contrastive.log >> ${MAIN_LOG}
else
log_error
"Evaluare T3 contrastive eșuată"
exit
1
fi
echo
"" >> ${MAIN_LOG}

# ============================================================================
# GENERARE RAPORT FINAL CONTRASTIVE
# ============================================================================

log_step
"GENERARE RAPORT FINAL CONTRASTIVE"

FINAL_REPORT = "${LOG_DIR}/FINAL_REPORT_CONTRASTIVE.txt"

{
    echo
"======================================================================"
echo
"RAPORT FINAL - CONTINUAL LEARNING CU CONTRASTIVE LOSS"
echo
"======================================================================"
echo
""
echo
"Timestamp: ${TIMESTAMP}"
echo
"Completed at: $(date)"
echo
""
echo
"======================================================================"
echo
"METODOLOGIE"
echo
"======================================================================"
echo
""
echo
"Loss Function: Combined Loss = Cross-Entropy + λ * Contrastive Loss"
echo
"  - Cross-Entropy: Standard classification loss"
echo
"  - Contrastive Loss: Pull embeddings towards keyword centroids"
echo
"  - Lambda (contrastive_weight): 0.5"
echo
"  - Temperature: 0.5"
echo
""
echo
"Mapare Automată: Cosine similarity între cluster keywords și class names"
echo
""

echo
"======================================================================"
echo
"REZUMAT CLUSTERING + MAPARE AUTOMATĂ"
echo
"======================================================================"
echo
""

echo
"TEST_1 (clase 1, 3, 11):"
grep - A
3
"METRICI CLUSTERING" ${LOG_DIR} / 01
_t1_pipeline_contrastive.log | head - 4
grep
"Average mapping similarity" ${LOG_DIR} / 01
_t1_pipeline_contrastive.log
echo
""

echo
"TEST_2 (clase 4, 6, 8):"
grep - A
3
"METRICI CLUSTERING" ${LOG_DIR} / 05
_t2_pipeline_contrastive.log | head - 4
grep
"Average mapping similarity" ${LOG_DIR} / 05
_t2_pipeline_contrastive.log
echo
""

echo
"TEST_3 (clase 5, 9, 10, 13):"
grep - A
3
"METRICI CLUSTERING" ${LOG_DIR} / 0
9
_t3_pipeline_contrastive.log | head - 4
grep
"Average mapping similarity" ${LOG_DIR} / 0
9
_t3_pipeline_contrastive.log
echo
""

echo
"======================================================================"
echo
"REZULTATE FINALE (CONTRASTIVE)"
echo
"======================================================================"
echo
""

echo
"--- TEST_1 ---"
grep - A
10
"REZUMAT" ${LOG_DIR} / 03
_t1_evaluation_contrastive.log | grep - E
"(Forgetting|Accuracy|Overall|Mapping)"
echo
""

echo
"--- TEST_2 ---"
grep - A
15
"REZUMAT FINAL" ${LOG_DIR} / 07
_t2_evaluation_contrastive.log | grep - E
"(Forgetting|Accuracy|Overall)"
echo
""

echo
"--- TEST_3 (FINAL) ---"
grep - A
20
"REZUMAT FINAL" ${LOG_DIR} / 11
_t3_evaluation_contrastive.log | grep - E
"(Forgetting|Accuracy|Overall|Classes)"
echo
""

echo
"======================================================================"
echo
"TRAINING LOSSES (CONTRASTIVE)"
echo
"======================================================================"
echo
""
echo
"Test_1 sample losses (first 5 steps):"
grep
"CE:.*Contrastive:.*Total:" ${LOG_DIR} / 02
_t1_training_contrastive.log | head - 5
echo
""
echo
"Test_1 final loss:"
grep
"Final training loss" ${LOG_DIR} / 02
_t1_training_contrastive.log
echo
""
echo
"Test_2 final loss:"
grep
"Final training loss" ${LOG_DIR} / 06
_t2_training_contrastive.log
echo
""
echo
"Test_3 final loss:"
grep
"Final training loss" ${LOG_DIR} / 10
_t3_training_contrastive.log
echo
""

echo
"======================================================================"
echo
"OOD DETECTION (AUROC)"
echo
"======================================================================"
echo
""
echo
"Test_2 pre-CL:"
grep
"AUROC (known vs unknown)" ${LOG_DIR} / 04
_t2_pre_evaluation.log
echo
""
echo
"Test_3 pre-CL:"
grep
"AUROC (known vs unknown)" ${LOG_DIR} / 0
8
_t3_pre_evaluation.log
echo
""

echo
"======================================================================"
echo
"COMPARAȚIE CU SOFTMAX SIMPLU"
echo
"======================================================================"
echo
""
echo
"Pentru comparație, rulați și ./english.sh (softmax simplu)"
echo
"Apoi comparați:"
echo
"  - Forgetting rate"
echo
"  - Accuracy pe clase noi"
echo
"  - Overall accuracy"
echo
"  - Training stability (convergență)"
echo
""

echo
"======================================================================"
echo
"FIȘIERE GENERATE"
echo
"======================================================================"
echo
""
echo
"Modele contrastive:"
ls - lh
ckpt_ * _contrastive / final / pytorch_model.bin
2 > / dev / null | | echo
"  (verificați ckpt_*_contrastive/)"
echo
""
echo
"Datasets CL:"
ls - lh
cl_train_ * _contrastive.csv
echo
""
echo
"Mapări automate:"
ls - lh
auto_mapping_ * _contrastive.json
echo
""
echo
"Keyword embeddings:"
ls - lh
keyword_embeddings_ * _contrastive.pkl
echo
""
echo
"Grafice ROC:"
ls - lh
roc_curve_ *.png
2 > / dev / null | | echo
"  (nu s-au generat)"
echo
""

echo
"======================================================================"
echo
"LOGS INDIVIDUALE"
echo
"======================================================================"
echo
""
ls - lh ${LOG_DIR} / *.log
echo
""

echo
"======================================================================"
echo
"PIPELINE CONTRASTIVE COMPLET!"
echo
"======================================================================"

} > ${FINAL_REPORT}

log_success
"Raport final generat: ${FINAL_REPORT}"

# Afișează raport final
echo
""
echo - e
"${MAGENTA}=====================================================================${NC}"
echo - e
"${MAGENTA}RAPORT FINAL CONTRASTIVE${NC}"
echo - e
"${MAGENTA}=====================================================================${NC}"
cat ${FINAL_REPORT}

# Copiază raport în main log
cat ${FINAL_REPORT} >> ${MAIN_LOG}

# Summary
echo
""
log_step
"PIPELINE CONTRASTIVE COMPLET FINALIZAT!"
echo
""
log_success
"Toate fișierele de log în: ${LOG_DIR}/"
log_success
"Log principal: ${MAIN_LOG}"
log_success
"Raport final: ${FINAL_REPORT}"
echo
""
echo - e
"${YELLOW}Verificați rezultatele în:${NC}"
echo - e
"  ${BLUE}cat ${FINAL_REPORT}${NC}"
echo
""
echo - e
"${MAGENTA}🔥 Contrastive Learning: Keywords pull embeddings towards semantic centroids${NC}"
echo
""

exit
0