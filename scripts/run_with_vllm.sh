#!/usr/bin/env bash
# Start vLLM server and run DrBench tasks.
#
# Environment:
#   VLLM_MODEL           - Model to serve (optional; derived from --model if not set)
#   VENV_DIR             - Python venv directory (default: ./.venv_vllm)
#   VLLM_LOG_DIR         - vLLM log directory (default: ./logs)
#
# Assumes:
#   The venv already has DrBench installed (e.g., `uv pip install -e .` or `pip install -e .`).
#
# Usage:
#   ./scripts/run_with_vllm.sh DR0001 DR0002 --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --llm-provider vllm
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv_vllm}"

# Extract --model and --embedding-provider from args
MODEL_ARG=""
EMBED_PROVIDER=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--model" ]] && [[ $((i+1)) -lt ${#ARGS[@]} ]]; then
    MODEL_ARG="${ARGS[$((i+1))]}"
  fi
  if [[ "${ARGS[$i]}" == "--embedding-provider" ]] && [[ $((i+1)) -lt ${#ARGS[@]} ]]; then
    EMBED_PROVIDER="${ARGS[$((i+1))]}"
  fi
done

if [[ -z "${VLLM_MODEL:-}" ]]; then
  if [[ -z "$MODEL_ARG" ]]; then
    echo "[ERROR] --model is required (or set VLLM_MODEL)."
    exit 1
  fi
  export VLLM_MODEL="$MODEL_ARG"
fi

# Ensure llm provider is vllm unless explicitly set
HAS_PROVIDER=false
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--llm-provider" ]]; then
    HAS_PROVIDER=true
    break
  fi
done
if [[ "$HAS_PROVIDER" == false ]]; then
  ARGS+=(--llm-provider vllm)
fi

# Setup venv
mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[INFO] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Start vLLM
LOG_DIR="${VLLM_LOG_DIR:-$REPO_DIR/logs}"
mkdir -p "$LOG_DIR"
export VLLM_LOG_FILE="${VLLM_LOG_FILE:-$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log}"
echo "[INFO] vLLM log: $VLLM_LOG_FILE"
echo "[INFO] Starting vLLM with model: $VLLM_MODEL"
export VLLM_PID=$(bash "$SCRIPT_DIR/start_vllm.sh")
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

bash "$SCRIPT_DIR/wait_vllm.sh"

# Run tasks
echo "[INFO] Running tasks..."
python "$REPO_DIR/experiments/run_tasks.py" "${ARGS[@]}"

echo "[INFO] Done."
