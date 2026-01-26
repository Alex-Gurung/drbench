# Privacy Tools

Opinionated privacy evaluation workflows for DrBench runs.

## Assumptions

- Run commands from the repo root.
- CLI flags control behavior; env vars provide keys/endpoints only.
- Logging is on by default and writes under `--run-dir`.

## Core evaluator

### `privacy/eval.py`
Runs adversary evaluation against web/local search queries.

Required:
- `--model`
- One of: `--batch`, `--latest`

Common args:
- `--runs-dir` (where batches live)
- `--run-dir` (where outputs/logs are written)
- `--search-source` (`web`, `local`, `both`)
- `--runs` (adversary repetitions per question)
- `--batched`, `--batch-size`
- `--llm-provider`, `--data-dir`, `--vector-stores-dir`
- `--no-log*`

Examples:
```bash
python privacy/eval.py --batch ./runs/batch_x --model gpt-4o-mini
python privacy/eval.py --latest --runs-dir ./runs --model gpt-4o-mini --search-source both
```

Outputs (under `--run-dir`):
- `privacy_eval.json`
- `config.json`

## Utilities

### `privacy/roleplay.py`
Interactive human adversary roleplay (queries only).

Examples:
```bash
python privacy/roleplay.py --latest --runs-dir ./runs
python privacy/roleplay.py ./runs/batch_x --company "Acme"
```

### `privacy/summary.py`
Summarize privacy eval outputs across batches.

Examples:
```bash
python privacy/summary.py --latest --runs-dir ./runs
python privacy/summary.py --batch ./runs/batch_x
```

### `privacy/explore_questions.py`
Inspect enterprise-fact questions per task.

Examples:
```bash
python privacy/explore_questions.py DR0001
python privacy/explore_questions.py --list
```

### `privacy/extract_secrets.py`
LLM-based extraction of confidential Q/A pairs from task files.

Required:
- `--model`

Common args:
- `--all` or explicit task IDs
- `--output`, `--run-dir`, `--data-dir`
- `--llm-provider`, `--api-url`
- `--max-tokens`, `--max-chars`, `--concurrency`
- `--no-log*`

Examples:
```bash
python privacy/extract_secrets.py DR0001 --model gpt-4o-mini --run-dir ./runs/secret_extract
python privacy/extract_secrets.py --all --model gpt-4o-mini --output ./runs/secrets.json
```

## Logging outputs

When logging is enabled, runs typically contain:
- `llm_generations.jsonl`
- `internet_searches.jsonl`
- `local_searches.jsonl`
- `prompts/*.txt`

## Reference suites

See `privacy/test_cases/` for curated eval question sets and leakage notes.
