# Experiment Test Cases

These JSON files are inputs for `experiments/atomic_privacy_testbed.py`.
They capture question variants, local findings context, and optional eval questions
for measuring leakage.

## Formats

### 1) Synthetic case (self-contained)
Use when you want to embed local findings directly in the file.

Required fields:
- `task_id`
- `company_name`
- `research_question` OR `research_question_variants`
- `local_findings` (list of `{source, content}`)
- `eval_questions` (optional, for adversary scoring)

Example: `indirect_leaky_dr0001.json`

### 2) Batch-backed case
Use when you want to reuse local findings from a real run.

Required fields:
- `task_id`
- `company_name`
- `base_local_findings` set to:
  - `FROM_BATCH:./runs/<batch_dir>/<task_id>`
- `research_question_variants` (for mosaics)
- `eval_questions` (optional)

Example: `indirect_leaky_dr0001.json`

## Running

```bash
python experiments/atomic_privacy_testbed.py --synthetic experiments/test_cases/indirect_leaky_dr0001.json --model gpt-4o-mini
python experiments/atomic_privacy_testbed.py --synthetic experiments/test_cases/indirect_leaky_dr0001.json --mosaic --model gpt-4o-mini
```

## Notes

- Batch-backed cases require a matching run directory under `./runs/`.
- If you do not have the original batch, update `base_local_findings` to point to
  your own batch path.
- Some files in this folder are analysis artifacts (not direct inputs) but are
  kept for reference.
