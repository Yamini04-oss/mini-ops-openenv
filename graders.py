"""
Grader functions for MiniOpsEnv.

Each grader receives the raw text string from action.payload["text"]
and the expected output, then returns a float score in [0.0, 1.0].
All graders are deterministic, reproducible, and never crash.
"""
import json
import re
from typing import Any, List


# ---------------------------------------------------------------------------
# Task 0 — Email Classification
# ---------------------------------------------------------------------------

def grade_email_classification(text: str, expected: str) -> float:
    """
    Score: 1.0 for correct label, 0.0 otherwise.
    Accepts any text that contains the correct word.
    """
    if not isinstance(text, str):
        return 0.0
    normalized = text.strip().lower()
    expected_lower = expected.lower()
    # Exact single-word match
    if normalized == expected_lower:
        return 1.0
    # The correct label appears as a word in the response
    if re.search(rf"\b{re.escape(expected_lower)}\b", normalized):
        return 1.0
    # Contains the label somewhere (partial)
    if expected_lower in normalized:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Task 1 — Task Prioritization
# ---------------------------------------------------------------------------

def grade_task_prioritization(text: str, expected: List[str]) -> float:
    """
    Parse a JSON array or line-by-line list from the agent's text,
    then score on ordering accuracy.
    """
    if not isinstance(text, str):
        return 0.0

    # Attempt JSON parse first
    parsed = None
    try:
        clean = text.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        if not isinstance(parsed, list):
            parsed = None
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: one task name per line
    if parsed is None:
        parsed = [
            line.strip().lstrip("-•*123456789. )")
            for line in text.splitlines()
            if line.strip()
        ]

    answer = [str(a).strip() for a in parsed]
    expected_stripped = [e.strip() for e in expected]

    if not answer:
        return 0.0

    # Perfect order
    if answer == expected_stripped:
        return 1.0

    # Right items, wrong order
    if sorted(answer) == sorted(expected_stripped):
        matches = sum(1 for a, e in zip(answer, expected_stripped) if a == e)
        return round(0.5 + 0.5 * (matches / len(expected_stripped)), 4)

    # Partial: correct items present
    correct_items = sum(1 for a in answer if a in expected_stripped)
    if correct_items == 0:
        return 0.0
    return round(0.2 * correct_items / len(expected_stripped), 4)


# ---------------------------------------------------------------------------
# Task 2 — Data Cleaning
# ---------------------------------------------------------------------------

def grade_data_cleaning(text: str, expected: float) -> float:
    """
    Extract a number from the agent's text and compare to expected sum.
    """
    if not isinstance(text, str):
        return 0.0

    # Find the first number-like token in the text
    match = re.search(r"-?\d+(\.\d+)?", text.strip())
    if not match:
        return 0.0

    try:
        numeric_answer = float(match.group())
    except ValueError:
        return 0.0

    diff = abs(numeric_answer - expected)
    if diff < 1e-6:
        return 1.0
    if diff <= 5.0:
        return 0.5
    if diff <= 20.0:
        return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "email_classification": grade_email_classification,
    "task_prioritization":  grade_task_prioritization,
    "data_cleaning":        grade_data_cleaning,
}


def grade(task_type: str, text: Any, expected: Any) -> float:
    """
    Dispatch to the correct grader.
    Always returns a float in [0.0, 1.0]. Never raises.
    """
    grader = GRADERS.get(task_type)
    if grader is None:
        return 0.0
    try:
        score = grader(str(text) if not isinstance(text, str) else text, expected)
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0
