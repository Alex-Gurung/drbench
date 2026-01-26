import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from drbench.task_loader import get_data_path


@dataclass(frozen=True)
class QuestionSet:
    name: str
    source: str
    questions: Dict[str, str]


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


def load_question_set(
    question_set: Optional[str] = None,
    question_file: Optional[str] = None,
) -> Optional[QuestionSet]:
    if question_file:
        path = Path(question_file)
    elif question_set:
        path = Path(get_data_path(f"drbench/data/question_sets/{question_set}.json"))
    else:
        return None

    if not path.exists():
        raise FileNotFoundError(f"Question set not found: {path}")

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
