# Privacy Test Case References

These files are reference suites for privacy evaluation and leakage analysis.
They are not necessarily direct inputs to `privacy/eval.py`.

## Contents

- `all_tasks_eval_questions.json`
  - A full task->eval question mapping derived from local findings.
- `local_findings_eval_suite.json`
  - A descriptive suite with per-task metrics + question notes.
- `dr0001_local_findings_eval.json`, `dr0006_local_findings_eval.json`, `dr0011_local_findings_eval.json`
  - Focused per-task leakage notes and question sets.

## Usage

Use these as references when:
- building new test cases for `experiments/atomic_privacy_testbed.py`
- comparing leakage patterns across tasks
- documenting evaluation criteria

If you want runnable synthetic cases, see:
`experiments/test_cases/`
