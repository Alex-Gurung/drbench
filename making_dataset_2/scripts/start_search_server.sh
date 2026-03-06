#!/usr/bin/env bash
# Start the BM25 search server.
#
# Loads local + web chunk indexes into one process (~15-20GB RAM).
# Chain builder workers connect via --search-url http://HOST:PORT.
#
# Usage:
#   bash making_dataset_2/scripts/start_search_server.sh
#
# Environment overrides:
#   PORT=8100 bash ...          # different port (default 8100)
#   HOST=0.0.0.0 bash ...      # bind address (default 0.0.0.0)

set -euo pipefail

PYTHON="${PYTHON:-/home/toolkit/.mamba/envs/vllm013/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

CHUNKS_LOCAL="$ROOT/making_dataset/outputs/chunks_local.jsonl"
CHUNKS_WEB_1="$ROOT/making_dataset/outputs/chunks_web.jsonl"
CHUNKS_WEB_2="$ROOT/making_dataset_2/outputs/chunks_web_drbench_urls.jsonl"

# Build web args from existing files
WEB_ARGS=""
for f in "$CHUNKS_WEB_1" "$CHUNKS_WEB_2"; do
    if [ -f "$f" ]; then
        WEB_ARGS="$WEB_ARGS $f"
    fi
done

echo "Starting BM25 search server on $HOST:$PORT"
echo "  Local chunks: $CHUNKS_LOCAL"
echo "  Web chunks:  $WEB_ARGS"

$PYTHON -m making_dataset_2.retrieval.search_server \
    --chunks-local "$CHUNKS_LOCAL" \
    --chunks-web $WEB_ARGS \
    --host "$HOST" \
    --port "$PORT"
