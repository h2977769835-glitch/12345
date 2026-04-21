#!/bin/bash

set -ex

# Minimal SFT launch script for recommendation_gr.
# Required env vars:
#   HF_MODEL_PATH
#   TORCH_DIST_CKPT
#   REC_SAVE_PATH
#   REC_TRAIN_JSON
# Optional env var:
#   REC_MAX_SEQ_LENGTH

export PYTHONBUFFERED=1

if [ -z "${HF_MODEL_PATH}" ] || [ -z "${TORCH_DIST_CKPT}" ] || [ -z "${REC_SAVE_PATH}" ] || [ -z "${REC_TRAIN_JSON}" ] || [ -z "${REC_ITEM2TOKEN_PATH}" ]; then
  echo "Missing required env vars. Need: HF_MODEL_PATH, TORCH_DIST_CKPT, REC_SAVE_PATH, REC_TRAIN_JSON, REC_ITEM2TOKEN_PATH"
  exit 1
fi

export REC_MAX_SEQ_LENGTH=${REC_MAX_SEQ_LENGTH:-100}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

source "${ROOT_DIR}/scripts/models/qwen2.5-0.5B.sh"

ray stop --force || true
pkill -9 sglang || true
sleep 2

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS_PER_NODE:-1} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT:-8265}

CKPT_ARGS=(
  --hf-checkpoint "${HF_MODEL_PATH}"
  --ref-load "${TORCH_DIST_CKPT}"
  --save "${REC_SAVE_PATH}"
  --save-interval 200
)

REC_ARGS=(
  --rollout-function-path examples.recommendation_gr.rec_sft_rollout.generate_rollout
  --data-source-path examples.recommendation_gr.rec_data_source.RecommendationDataSource
  --prompt-data "${REC_TRAIN_JSON}"
  --rollout-shuffle
  --rollout-batch-size 16
  --global-batch-size 16
  --n-samples-per-prompt 1
  --num-epoch 1
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
  --input-key prompt
  --label-key label
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style cosine
  --min-lr 1e-6
  --lr-warmup-fraction 0.1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROOT_DIR}:$PYTHONPATH\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT:-8265}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${ROOT_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${NUM_GPUS_PER_NODE:-1} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${REC_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${MISC_ARGS[@]}
