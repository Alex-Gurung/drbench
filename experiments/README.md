# Experiments

Exploratory runners and analysis utilities. These scripts are intentionally
separate from the core library and privacy package so the main code stays
predictable and minimal.

## Assumptions

- Run commands from the repo root.
- Use CLI flags for behavior. Env vars are for secrets/infrastructure only.
- Logging is on by default and writes under `--run-dir`.

## Runners

### `experiments/run_tasks.py`
Batch run one or more tasks using the full DrBench agent.

Required:
- `--model` (LLM name)

Common args:
- `--run-dir` (base output dir; default: `./runs/batch_<model>_<timestamp>`)
- `--llm-provider` (`openai`, `vllm`, `openrouter`, `azure`, `together`)
- `--embedding-provider`, `--embedding-model`
- `--data-dir` (override DRBENCH_DATA_DIR)
- `--question-set` (override DR question)
- `--no-web`, `--no-log*`

Examples:
```bash
python experiments/run_tasks.py DR0001 DR0002 --model gpt-4o-mini --run-dir ./runs/demo
python experiments/run_tasks.py --subset validation --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --llm-provider vllm
```

### `experiments/privacy_question_harness.py`
Runs a question-variant harness for privacy experiments.

Required:
- `--model`
- `--questions-file` (JSON of question variants)

Common args:
- `--run-dir`, `--llm-provider`, `--data-dir`
- `--no-log*`

Example:
```bash
python experiments/privacy_question_harness.py --questions-file ./data/variants.json --model gpt-4o-mini --run-dir ./runs/privacy_harness
```

### `experiments/atomic_privacy_testbed.py`
Fast iteration tool to generate web queries from local findings and evaluate
privacy leakage.

Required:
- `--model`
- One of: `--batch`/`--task`, `--batch`/`--all`, or `--synthetic`

Common args:
- `--prompt` (template name; default: `drbench_adaptive`)
- `--results-dir` (base output dir; default: `./runs/atomic_privacy_testbed`)
- `--llm-provider`, `--api-url`, `--data-dir`
- `--no-eval`, `--show-prompt`, `--show-full-prompt`
- `--no-log*`

Examples:
```bash
python experiments/atomic_privacy_testbed.py --batch ./runs/batch_x --task DR0001 --model gpt-4o-mini
python experiments/atomic_privacy_testbed.py --synthetic ./data/synthetic_case.json --model gpt-4o-mini
```

## Analysis utilities

- `experiments/analyze_run.py` - summarize a run directory or batch.
  ```bash
  python experiments/analyze_run.py --latest
  python experiments/analyze_run.py ./runs/batch_x/DR0001
  ```

- `experiments/pretty_action_plan.py` - render `action_plan_*.json` to Markdown.
  ```bash
  python experiments/pretty_action_plan.py ./runs/batch_x/DR0001/action_plan_final.json --out ./runs/batch_x/DR0001/action_plan.md
  ```

- `experiments/view_web_searches.py` - view `privacy/web_searches.json` outputs.
  ```bash
  python experiments/view_web_searches.py --latest
  python experiments/view_web_searches.py --batch ./runs/batch_x --task DR0001
  ```

- `experiments/view_search_logs.py` - view JSONL log streams.
  ```bash
  python experiments/view_search_logs.py --run-dir ./runs/batch_x/DR0001 --stream internet
  ```

- `experiments/summarize_adversarial_batch.py` - batch performance + privacy summary.

## Prompt templates

Prompt templates for `atomic_privacy_testbed.py` live in `experiments/prompts/`.
The default is `drbench_adaptive.txt` (a static template). If you want to
mirror the live agent prompt, that will require a small refactor to expose
prompt builders from `drbench/agents/drbench_agent/action_planning_system.py`.

## Test cases

Sample and curated synthetic cases live under:
`experiments/test_cases/`

## Logging outputs

When logging is enabled, runs typically contain:
- `llm_generations.jsonl`
- `internet_searches.jsonl`
- `local_searches.jsonl`
- `prompts/*.txt`
- `privacy/web_searches.json` (from `run_tasks.py`)

## VLLM helper

`scripts/run_with_vllm.sh` starts vLLM and delegates to `experiments/run_tasks.py`.
It assumes the venv already has DrBench installed.
