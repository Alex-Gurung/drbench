import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class QuestionSet:
    name: str
    source: str
    questions: Dict[str, str]

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTION_SET_DIRS = [
    REPO_ROOT / "data" / "question_sets",
    REPO_ROOT / "data" / "tasks" / "question_sets",
]


def _load_question_set(path: Path) -> QuestionSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions: Dict[str, str] = {}

    if isinstance(data, dict):
        raw_questions = data.get("questions", {})
        for task_id, entry in raw_questions.items():
            if isinstance(entry, dict):
                question = entry.get("dr_question")
            else:
                question = entry
            if question:
                questions[task_id] = question
        name = data.get("set_name", path.stem)
    else:
        name = path.stem

    return QuestionSet(name=name, source=str(path), questions=questions)


def _resolve_question_set_path(
    question_set: Optional[str],
    question_file: Optional[str],
) -> Optional[Path]:
    if question_file:
        return Path(question_file)

    if not question_set:
        return None

    direct_path = Path(question_set)
    if direct_path.exists():
        return direct_path

    for base_dir in QUESTION_SET_DIRS:
        candidate = base_dir / f"{question_set}.json"
        if candidate.exists():
            return candidate

    return None


def load_question_set(
    question_set: Optional[str] = None,
    question_file: Optional[str] = None,
) -> Optional[QuestionSet]:
    path = _resolve_question_set_path(question_set, question_file)
    if not path:
        return None

    if not path.exists():
        raise FileNotFoundError(
            f"Question set not found: {path} "
            f"(question_set={question_set!r}, question_file={question_file!r})"
        )

    return _load_question_set(path)


def resolve_dr_question(
    task_id: str,
    default_question: str,
    question_set: Optional[str] = None,
    question_file: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    qs = load_question_set(question_set=question_set, question_file=question_file)
    if not qs:
        return default_question, None

    question = qs.questions.get(task_id, default_question)
    return question, qs.name
