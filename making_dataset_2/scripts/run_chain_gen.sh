#!/usr/bin/env bash
# Full chain generation run with step-3.5-flash.
#
# Generates 50 chains per pattern for all L/W combos up to length 4 (28 patterns).
# Uses 10 parallel workers and logs progress to both console and file.
#
# Usage:
#   bash making_dataset_2/scripts/run_chain_gen.sh
#
# Environment overrides:
#   N=10 bash making_dataset_2/scripts/run_chain_gen.sh   # 10 per pattern instead of 50
#   WORKERS=4 bash ...                                     # 4 workers instead of 10
#   BASE_URL=http://... bash ...                           # different endpoint
#   PATTERNS="LW WL LWL" bash ...                         # specific patterns only
#   SEARCH_URL=http://localhost:8100 bash ...              # use remote search server

set -euo pipefail

PYTHON="/home/toolkit/.mamba/envs/vllm013/bin/python"
MODEL="${MODEL:-step-3.5-flash}"
BASE_URL="${BASE_URL:-http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-step35flash-vllm:8000/v1}"
N="${N:-50}"
WORKERS="${WORKERS:-20}"
MIN_MAX_TOKENS="${MIN_MAX_TOKENS:-16000}"

# All L/W patterns length 2-4
ALL_PATTERNS="LL LW WL WW LLL LLW LWL LWW WLL WLW WWL WWW LLLL LLLW LLWL LLWW LWLL LWLW LWWL LWWW WLLL WLLW WLWL WLWW WWLL WWLW WWWL WWWW"
PATTERNS="${PATTERNS:-$ALL_PATTERNS}"

# Output paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="making_dataset_2/outputs/chains_${MODEL}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
OUTPUT="$OUTPUT_DIR/chains.jsonl"
LOGFILE="$OUTPUT_DIR/run.log"

# Count patterns and total chains
N_PATTERNS=$(echo $PATTERNS | wc -w)
TOTAL=$((N_PATTERNS * N))

SEARCH_URL="${SEARCH_URL:-}"

cat <<EOF | tee "$LOGFILE"
=== Chain Generation Run ===
Model:      $MODEL
Endpoint:   $BASE_URL
Search:     ${SEARCH_URL:-in-process}
Patterns:   $N_PATTERNS ($PATTERNS)
Per pattern: $N
Total:      $TOTAL chains
Workers:    $WORKERS
Min tokens: $MIN_MAX_TOKENS
Output:     $OUTPUT
Log:        $LOGFILE
Started:    $(date)
============================
EOF

# Run
RESUME_FLAG=""
if [ "${RESUME:-}" = "1" ] || [ "${RESUME:-}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

SEARCH_FLAG=""
if [ -n "$SEARCH_URL" ]; then
    SEARCH_FLAG="--search-url $SEARCH_URL"
fi

$PYTHON -m making_dataset_2.pipeline.chain_builder \
    --patterns $PATTERNS \
    --n "$N" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --output "$OUTPUT" \
    --spacy-model en_core_web_trf \
    --retrieval-mode bm25 --retrieval-k 50 \
    --min-max-tokens "$MIN_MAX_TOKENS" \
    --workers "$WORKERS" \
    $RESUME_FLAG \
    $SEARCH_FLAG \
    --verbose \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "Output:   $OUTPUT" | tee -a "$LOGFILE"
echo "Chains:   $(wc -l < "$OUTPUT")" | tee -a "$LOGFILE"

# Quick summary
$PYTHON -c "
import json, sys
from collections import Counter
valid = Counter()
total = Counter()
for line in open('$OUTPUT'):
    d = json.loads(line)
    p = d['pattern']
    total[p] += 1
    if d.get('verification', {}).get('is_valid'):
        valid[p] += 1
print()
print('Pattern    Valid/Total')
print('-' * 30)
for p in sorted(total):
    print(f'{p:10s} {valid[p]:3d}/{total[p]:3d}')
print('-' * 30)
print(f'{\"TOTAL\":10s} {sum(valid.values()):3d}/{sum(total.values()):3d}')
" | tee -a "$LOGFILE"
