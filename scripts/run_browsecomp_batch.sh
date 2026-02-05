#!/usr/bin/env bash
# Run DrBench tasks with BrowseComp offline web search.
#
# Environment variables:
#   VLLM_MODEL           - (required) Model to serve, e.g. Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
#   VLLM_GPU_MEM_UTIL    - GPU memory utilization (default: 0.85)
#   VLLM_API_URL         - vLLM API endpoint (default: http://127.0.0.1:8000)
#   VLLM_TENSOR_PARALLEL - Tensor parallel size (default: 1)
#   VLLM_MAX_MODEL_LEN   - Max model length (default: 32768)
#   VLLM_LOG_DIR         - Directory for vLLM logs (default: $REPO_DIR/logs)
#
#   TASKS                - Space-separated task IDs (default: DR0001-DR0015)
#
#   BROWSECOMP_INDEX     - Path glob for index shards (default: /home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl)
#   BROWSECOMP_MODEL     - Embedding model for queries (default: Qwen/Qwen3-Embedding-4B)
#   BROWSECOMP_DATASET   - HuggingFace corpus dataset (default: Tevatron/browsecomp-plus-corpus)
#   BROWSECOMP_TOP_K     - Results per query (default: 5)
#
# CLI args passed to run_tasks.py:
#   --model              - LLM model name (from VLLM_MODEL)
#   --llm-provider       - vllm
#   --embedding-provider - huggingface
#   --embedding-model    - Embedding model (from BROWSECOMP_MODEL)
#   --run-dir            - Output directory
#   --max-iterations     - 20
#   --concurrent-actions - 3
#   --browsecomp         - Enable BrowseComp offline search
#   --browsecomp-index   - Index path glob
#   --browsecomp-model   - Embedding model
#   --browsecomp-dataset - HuggingFace dataset
#   --browsecomp-top-k   - Number of results
#   --verbose            - Verbose output
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/home/toolkit/.mamba/envs/vllm013/bin/python"
PIP="/home/toolkit/.mamba/envs/vllm013/bin/pip"

: "${VLLM_MODEL:?Must set VLLM_MODEL}"

# Ensure nice_code drbench is installed, not root
# $PIP uninstall -y drbench 2>/dev/null || true
# $PIP install -q -e "$REPO_DIR"
: "${TASKS:=DR0001 DR0002 DR0003 DR0004 DR0005 DR0006 DR0007 DR0008 DR0009 DR0010 DR0011 DR0012 DR0013 DR0014 DR0015}"

# vLLM connection
export VLLM_API_URL="${VLLM_API_URL:-http://127.0.0.1:8000}"

# BrowseComp defaults
: "${BROWSECOMP_INDEX:=/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl}"
: "${BROWSECOMP_MODEL:=Qwen/Qwen3-Embedding-4B}"
: "${BROWSECOMP_DATASET:=Tevatron/browsecomp-plus-corpus}"
: "${BROWSECOMP_TOP_K:=5}"

# Ensure we import from nice_code/drbench, not root drbench
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Start vLLM
LOG_DIR="${VLLM_LOG_DIR:-$REPO_DIR/logs}"
mkdir -p "$LOG_DIR"
export VLLM_LOG_FILE="${VLLM_LOG_FILE:-$LOG_DIR/vllm_browsecomp_$(date +%Y%m%d_%H%M%S).log}"
echo "[INFO] vLLM log: $VLLM_LOG_FILE"
echo "[INFO] Starting vLLM with model: $VLLM_MODEL"
export VLLM_PID=$(bash "$SCRIPT_DIR/start_vllm.sh")
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

bash "$SCRIPT_DIR/wait_vllm.sh"

# Generate run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(basename "$VLLM_MODEL" | cut -c1-20)
RUN_DIR="/home/toolkit/runs/batch_${MODEL_SHORT}_browsecomp_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "[INFO] BrowseComp batch run: $RUN_DIR"
echo "[INFO] Tasks: $TASKS"
echo "[INFO] BrowseComp index: $BROWSECOMP_INDEX"
echo "[INFO] BrowseComp model: $BROWSECOMP_MODEL"

# Run tasks with nohup (safe to disconnect)
TASK_LOG="$RUN_DIR/batch_run.log"
echo "[INFO] Running tasks with BrowseComp offline search..."
echo "[INFO] Output log: $TASK_LOG"
echo "[INFO] Safe to disconnect - tasks will continue running"

nohup "$PYTHON" "$REPO_DIR/experiments/run_tasks.py" $TASKS \
    --model "$VLLM_MODEL" \
    --llm-provider vllm \
    --embedding-provider huggingface \
    --embedding-model "$BROWSECOMP_MODEL" \
    --run-dir "$RUN_DIR" \
    --max-iterations 20 \
    --concurrent-actions 3 \
    --browsecomp \
    --browsecomp-index "$BROWSECOMP_INDEX" \
    --browsecomp-model "$BROWSECOMP_MODEL" \
    --browsecomp-dataset "$BROWSECOMP_DATASET" \
    --browsecomp-top-k "$BROWSECOMP_TOP_K" \
    --verbose \
    > "$TASK_LOG" 2>&1 &

TASK_PID=$!
echo "[INFO] Task PID: $TASK_PID"
echo "[INFO] Monitor with: tail -f $TASK_LOG"

# Wait for tasks to complete
wait $TASK_PID
EXIT_CODE=$?

echo ""
echo "[INFO] BrowseComp batch complete: $RUN_DIR (exit code: $EXIT_CODE)"
