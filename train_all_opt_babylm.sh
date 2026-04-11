#!/usr/bin/env bash
set -euo pipefail
############################################
# USER SETTINGS
############################################
DATASET="znhoughton/babylm-150m-v3"
TOKENIZER_NAME="opt-babylm-100m-bpe"
BLOCK_SIZE=1024
VOCAB_SIZE=8192
# TARGET: 10M tokens per checkpoint
TOKENS_PER_CHECKPOINT=10000000
SAVE_TOTAL_LIMIT=1
WARMUP_STEPS=4000
SEED=964
############################################
# STEP 0: Train tokenizer ONCE
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
############################################
train_opt () {
    MODEL_SIZE=$1
    BASE_MODEL=$2
    HIDDEN=$3
    HEADS=$4
    LAYERS=$5
    FFN=$6
    BATCH=$7
    GRAD_ACCUM=$8
    LR=$9
    
    # Calculate tokens per step and save_steps
    TOKENS_PER_STEP=$((BLOCK_SIZE * BATCH * GRAD_ACCUM * 2))
    SAVE_STEPS=$((TOKENS_PER_CHECKPOINT / TOKENS_PER_STEP))
    
    MODEL_NAME="opt-babylm-${MODEL_SIZE}-20eps"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="runs/${MODEL_NAME}_${SEED}-20eps"
    
    echo "============================================================"
    echo "=== Training ${MODEL_NAME} ==="
    echo "=== Tokens/step: ${TOKENS_PER_STEP} ==="
    echo "=== Save every ${SAVE_STEPS} steps (${TOKENS_PER_CHECKPOINT} tokens) ==="
    echo "============================================================"
    
    # Build config (cheap, safe to re-run)
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
    
    # Train
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_autoreg.py \
        --model_type opt \
        --config_name ${MODEL_PATH} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${DATASET} \
        --do_train \
        --bf16 \
        --gradient_checkpointing \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --save_only_model \
        --logging_steps 10 \
        --report_to tensorboard \
        --num_train_epochs 20 \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --push_to_hub \
        --hub_model_id znhoughton/${MODEL_NAME}-seed${SEED} \
        --hub_strategy checkpoint \
        --ddp_find_unused_parameters False
    
    echo "=== Finished training ${MODEL_NAME} ==="
    # IMPORTANT: free disk before next model
    echo "=== Deleting local run directory ${RUN_DIR} ==="
    rm -rf "${RUN_DIR}"
}


############################################
# OPT-125M - OPTIMIZED FOR 2x H100
# Target: Max batch size while maintaining speed
# tokens/step = 1024 × 512 × 1 × 2 = 1,048,576
# save_steps = 10M / 1,048,576 ≈ 10 steps
############################################
train_opt \
  125m \
  facebook/opt-125m \
  768 \
  12 \
  12 \
  3072 \
  400 \
  1 \
  3e-4

############################################
# OPT-350M - OPTIMIZED FOR 2x H100
# tokens/step = 1024 × 384 × 2 × 2 = 1,572,864
# save_steps = 10M / 1,572,864 ≈ 6 steps
############################################
train_opt \
  350m \
  facebook/opt-350m \
  1024 \
  16 \
  24 \
  4096 \
  300 \
  2 \
  1e-4

############################################
# OPT-1.3B - OPTIMIZED FOR 2x H100
# tokens/step = 1024 × 256 × 4 × 2 = 2,097,152
# save_steps = 10M / 2,097,152 ≈ 5 steps
############################################
train_opt \
  1.3b \
  facebook/opt-1.3b \
  2048 \
  32 \
  24 \
  8192 \
  100 \
  4 \
  1e-4