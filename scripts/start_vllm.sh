#!/bin/bash
# Start vLLM server in background, output only the PID.
#
# Environment:
#   VLLM_MODEL            - Model to serve (required)
#   VLLM_API_KEY          - API key (optional)
#   VLLM_GPU_MEM_UTIL     - GPU memory utilization (default: 0.85)
#   VLLM_TENSOR_PARALLEL  - Tensor parallel size (default: 1)
#   VLLM_MAX_MODEL_LEN    - Max model length (default: 32768)
#   VLLM_ENABLE_EP        - Enable expert parallel for MoE (optional)
#   VLLM_LOG_FILE         - Log file path (default: /tmp/vllm.log)

set -e

TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EP:-}"
LOG_FILE="${VLLM_LOG_FILE:-/tmp/vllm.log}"

VLLM_CMD="vllm serve $VLLM_MODEL \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL"

if [[ -n "${VLLM_API_KEY:-}" ]]; then
    VLLM_CMD="$VLLM_CMD --api-key $VLLM_API_KEY"
fi

if [[ -n "$ENABLE_EXPERT_PARALLEL" ]]; then
    VLLM_CMD="$VLLM_CMD --enable-expert-parallel"
fi

$VLLM_CMD > "$LOG_FILE" 2>&1 &

echo $!
