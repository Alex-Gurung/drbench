#!/bin/bash
# Wait for vLLM to be ready.
#
# Environment:
#   VLLM_API_URL          - API URL (default: http://localhost:8000)
#   VLLM_API_KEY          - API key (optional)
#   VLLM_WAIT_SECS        - Timeout in seconds (default: 600)
#   VLLM_PID              - Process ID to check (optional)
#   VLLM_LOG_FILE         - Log file for error output (default: /tmp/vllm.log)

echo "[INFO] Waiting for vLLM to be ready..."

WAIT_SECS="${VLLM_WAIT_SECS:-600}"
LOG_FILE="${VLLM_LOG_FILE:-/tmp/vllm.log}"
TAIL_LINES="${VLLM_LOG_TAIL_LINES:-200}"
API_URL="${VLLM_API_URL:-http://localhost:8000}"

CURL_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    CURL_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

for i in $(seq 1 "$WAIT_SECS"); do
    # Check if vLLM process died
    if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "[ERROR] vLLM process (PID $VLLM_PID) died"
        echo "[ERROR] Last ${TAIL_LINES} lines of ${LOG_FILE}:"
        tail -"$TAIL_LINES" "$LOG_FILE" 2>/dev/null || echo "(no log file)"
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${CURL_ARGS[@]}" "$API_URL/v1/models" 2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        echo ""
        echo "[INFO] vLLM ready after ${i}s"
        exit 0
    fi

    if (( i % 10 == 0 )); then
        echo -n "."
    fi
    sleep 1
done

echo ""
echo "[ERROR] vLLM failed to start after ${WAIT_SECS}s"
echo "[ERROR] Last ${TAIL_LINES} lines of ${LOG_FILE}:"
tail -"$TAIL_LINES" "$LOG_FILE" 2>/dev/null || echo "(no log file)"
exit 1
