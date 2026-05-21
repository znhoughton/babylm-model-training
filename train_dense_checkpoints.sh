#!/usr/bin/env bash

set -euo pipefail

############################################
# USER SETTINGS
############################################

export HF_HOME=/workspace/hf_cache

DATASET="znhoughton/babylm-150m-v3"
TOKENIZER_NAME="opt-babylm-100m-bpe"
BLOCK_SIZE=1024
VOCAB_SIZE=8192
SEED=964

# Local scratch dir with enough space for all checkpoints (~286GB for 125m)
SCRATCH_DIR="/dpluth-data/tmp_model"

############################################
# STEP 0: Reuse existing tokenizer
############################################

TOKENIZER_PATH="models/${TOKENIZER_NAME}"

if [ -d "${TOKENIZER_PATH}" ]; then
    echo "=== Tokenizer already exists at ${TOKENIZER_PATH}, skipping ==="
else
    echo "=== Training tokenizer ==="
    python tokenizer_and_config.py \
        --base_model facebook/opt-125m \
        --model_name ${TOKENIZER_NAME} \
        --train_file ${DATASET} \
        --from_iterator \
        --bpe \
        --vocab ${VOCAB_SIZE} \
        --hidden_size 768 \
        --attention_heads 12 \
        --layers 12 \
        --intermediate_size 3072 \
        --max_len ${BLOCK_SIZE}
fi

############################################
# FUNCTION: train one OPT model
# Single GPU, 1 epoch, save every step.
# No uploads during training — all checkpoints
# are pushed to Hub after training completes.
############################################

train_opt_dense () {
    MODEL_SIZE=$1
    BASE_MODEL=$2
    HIDDEN=$3
    HEADS=$4
    LAYERS=$5
    FFN=$6
    BATCH=$7
    LR=$8
    WARMUP_STEPS=$9

    TOKENS_PER_STEP=$((BLOCK_SIZE * BATCH))
    TOTAL_STEPS=$((150000000 / TOKENS_PER_STEP))

    MODEL_NAME="opt-babylm-${MODEL_SIZE}-1ep-dense"
    HUB_MODEL_ID="znhoughton/${MODEL_NAME}-seed${SEED}"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="${SCRATCH_DIR}/${MODEL_NAME}_${SEED}"

    echo "============================================================"
    echo "=== Training ${MODEL_NAME} ==="
    echo "=== Tokens/step:  ${TOKENS_PER_STEP} ==="
    echo "=== Approx total steps: ~${TOTAL_STEPS} ==="
    echo "=== Warmup steps: ${WARMUP_STEPS} ==="
    echo "=== Saving every step to ${RUN_DIR} ==="
    echo "=== Approx local storage: ~$((TOTAL_STEPS * 250 / 1024)) GB ==="
    echo "=== Hub push deferred until after training ==="
    echo "============================================================"

    # Build config
    python tokenizer_and_config.py \
        --base_model ${BASE_MODEL} \
        --model_name ${MODEL_NAME} \
        --train_file ${DATASET} \
        --from_iterator \
        --bpe \
        --vocab ${VOCAB_SIZE} \
        --hidden_size ${HIDDEN} \
        --attention_heads ${HEADS} \
        --layers ${LAYERS} \
        --intermediate_size ${FFN} \
        --max_len ${BLOCK_SIZE}

    # Train: single GPU, 1 epoch, save every step, no hub push during training
    python train_autoreg.py \
        --model_type opt \
        --config_name ${MODEL_PATH} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${DATASET} \
        --do_train \
        --bf16 \
        --gradient_checkpointing \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps 1 \
        --optim adamw_torch_fused \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --save_steps 1 \
        --save_only_model \
        --logging_steps 1 \
        --report_to tensorboard \
        --num_train_epochs 1 \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --torch_compile \
        --overwrite_output_dir

    echo "=== Training complete. Pushing checkpoints to Hub... ==="

    # Push all checkpoint-N folders to Hub as separate commits,
    # then verify count before deleting local files.
    python push_checkpoints_to_hub.py \
        --run_dir "${RUN_DIR}" \
        --hub_model_id "${HUB_MODEL_ID}" \
        --verify_before_delete

    echo "=== Upload verified. Deleting local scratch directory ${RUN_DIR} ==="
    rm -rf "${RUN_DIR}"
}

############################################
# OPT-125M - single GPU
#
# tokens/step = 1024 x 128 = 131,072
# total steps ~ 150M / 131,072 ~ 1,144
# warmup      = 114 steps (~10%)
# storage     ~ 1,144 x 250MB ~ 286GB
############################################

train_opt_dense \
    125m \
    facebook/opt-125m \
    768 \
    12 \
    12 \
    3072 \
    128 \
    3e-4 \
    114