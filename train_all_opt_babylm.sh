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
# TARGET: 20M tokens per checkpoint
TOKENS_PER_CHECKPOINT=20000000
SAVE_TOTAL_LIMIT=1
SEED=964

# How many GPUs to shard across. The per-device batch is derived from this so
# the global batch never changes; see train_opt(). Default 2, matching the
# original April runs.
NUM_GPUS="${NUM_GPUS-2}"

############################################
# MODEL SELECTION
#
#   ./train_all_opt_babylm.sh              # all three, as before
#   ./train_all_opt_babylm.sh 350m         # just the 350M
#   ./train_all_opt_babylm.sh 125m 1.3b    # any subset
#
# NAME_SUFFIX is appended to the model and Hub names so a retrain lands in a
# new repo instead of overwriting weights that existing analyses (and the
# arXiv preprints) point at. Set NAME_SUFFIX="" to overwrite in place.
############################################
NAME_SUFFIX="${NAME_SUFFIX--prenorm}"

if [ $# -eq 0 ]; then
  TO_TRAIN=(125m 350m 1.3b)
else
  TO_TRAIN=("$@")
fi

should_train () {
  local want=$1 m
  for m in "${TO_TRAIN[@]}"; do
    [ "${m}" = "${want}" ] && return 0
  done
  return 1
}

echo "=== Will train: ${TO_TRAIN[*]} ==="
echo "=== Name suffix: '${NAME_SUFFIX}' ==="

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
# Args: MODEL_SIZE BASE_MODEL HIDDEN HEADS
#       LAYERS FFN BATCH GRAD_ACCUM LR WARMUP
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
    WARMUP_STEPS=${10}

    # The BATCH values below are calibrated for 2 GPUs. Scale the per-device
    # batch inversely with the GPU count so the GLOBAL batch stays fixed: that
    # keeps total steps, the warmup fraction, and the checkpoint cadence
    # identical no matter how many GPUs we rent. Renting more GPUs should buy
    # wall-clock time, never a different training schedule.
    if [ $(( BATCH * 2 % NUM_GPUS )) -ne 0 ]; then
        echo "ERROR: BATCH*2 (${BATCH}*2) is not divisible by NUM_GPUS (${NUM_GPUS});" >&2
        echo "       the global batch would change. Pick a compatible GPU count." >&2
        exit 1
    fi
    PER_DEVICE_BATCH=$(( BATCH * 2 / NUM_GPUS ))

    TOKENS_PER_STEP=$((BLOCK_SIZE * PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS))
    SAVE_STEPS=$((TOKENS_PER_CHECKPOINT / TOKENS_PER_STEP))

    MODEL_NAME="opt-babylm-${MODEL_SIZE}-20eps${NAME_SUFFIX}"
    MODEL_PATH="models/${MODEL_NAME}"
    # Every checkpoint is kept now (see --save_total_limit below), so this has to
    # live somewhere with room for the whole run: ~150 checkpoints x ~1.4GB.
    # A container overlay is typically 40GB, which is nowhere near enough.
    RUN_DIR="${RUN_ROOT-/workspace/runs}/${MODEL_NAME}_${SEED}-20eps"

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

    # Architecture guard. AutoConfig.from_pretrained inherits anything we do
    # not override from the base checkpoint, and facebook/opt-350m is the one
    # OPT size that ships post-LN with a 512-dim embedding projection. Catching
    # that here costs a second; catching it after the run costs the run.
    python check_config.py "${MODEL_PATH}" --hidden ${HIDDEN} --layers ${LAYERS} --heads ${HEADS} --ffn ${FFN}

    # Train
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) \
    torchrun --nproc_per_node=${NUM_GPUS} train_autoreg.py \
        --model_type opt \
        --config_name ${MODEL_PATH} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${DATASET} \
        --do_train \
        --bf16 \
        --gradient_checkpointing \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${PER_DEVICE_BATCH} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --optim adamw_torch_fused \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --save_only_model \
        --logging_steps 10 \
        --report_to tensorboard \
        --num_train_epochs 20 \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --push_to_hub \
        --hub_model_id znhoughton/${MODEL_NAME}-seed${SEED} \
        --hub_strategy checkpoint \
        --torch_compile \
        --ddp_find_unused_parameters False \
        --overwrite_output_dir

    echo "=== Finished training ${MODEL_NAME} ==="

    # Trainer silently skips a Hub push when the previous one is still uploading,
    # so the local checkpoints are the authoritative copy. Keep them until the
    # Hub has been verified; DELETE_RUN_DIR=1 opts back into the old behaviour.
    if [ "${DELETE_RUN_DIR-0}" = "1" ]; then
        echo "=== Deleting local run directory ${RUN_DIR} ==="
        rm -rf "${RUN_DIR}"
    else
        echo "=== Keeping ${RUN_DIR} ($(du -sh "${RUN_DIR}" 2>/dev/null | cut -f1)) ==="
        echo "=== Verify the Hub before deleting it ==="
    fi
}


############################################
# OPT-125M - 2x A100 80GB
# tokens/step = 1024 × 400 × 1 × 2 = 819,200
# total steps ≈ 3,660; warmup = 366 (10%)
# save_steps = 20M / 819,200 ≈ 24 steps
############################################
if should_train 125m; then
train_opt \
  125m \
  facebook/opt-125m \
  768 \
  12 \
  12 \
  3072 \
  400 \
  1 \
  3e-4 \
  366
fi

############################################
# OPT-350M
# global tokens/step = 1024 × 200 × 1 × 2 = 409,600 (fixed; NUM_GPUS only
#   changes how that batch is split across devices)
# total steps ≈ 7,320; warmup = 732 (10%)
# save_steps = 20M / 409,600 ≈ 48 steps
#
# LR: the series follows half of OPT Table 1 (125M 6e-4→3e-4,
# 1.3B 2e-4→1e-4). The 350M was at 1e-4, which is the 1.3B's rate rather than
# half of OPT's 3e-4 for this size, and broke that rule. Corrected to 1.5e-4.
############################################
if should_train 350m; then
train_opt \
  350m \
  facebook/opt-350m \
  1024 \
  16 \
  24 \
  4096 \
  200 \
  1 \
  1.5e-4 \
  732
fi

############################################
# OPT-1.3B - 2x A100 80GB
# tokens/step = 1024 × 100 × 1 × 2 = 204,800
# total steps ≈ 14,648; warmup = 1465 (10%)
# save_steps = 20M / 204,800 ≈ 97 steps
############################################
if should_train 1.3b; then
train_opt \
  1.3b \
  facebook/opt-1.3b \
  2048 \
  32 \
  24 \
  8192 \
  100 \
  1 \
  1e-4 \
  1465
fi
