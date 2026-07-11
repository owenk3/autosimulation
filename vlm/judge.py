"""Failure classifier — watches frames and returns a failure_type."""

from vlm._claude import call_claude, frames_to_b64

FAILURE_TYPES = [
    "execution_misalignment",
    "task_misunderstanding",
    "perception_failure",
    "multi_step_planning",
]

_PROMPT = """You are a robotics failure analyst. Classify this robot manipulation failure.

Task: "{task}"

Failure types:
1. execution_misalignment: Robot reaches CORRECT target but grasp/push miscalibrated — too weak, too shallow, slightly off.
2. task_misunderstanding: Robot manipulates WRONG object — misread spatial/relational language (e.g., picks middle bowl when asked for left).
3. perception_failure: Target object IGNORED entirely — robot wanders or grabs completely wrong object.
4. multi_step_planning: FIRST sub-goal succeeds, SECOND abandoned (compound "X and Y" tasks only).

Reply with JSON only: {{"failure_type": "<one of the four>", "reason": "<one sentence>"}}"""


def classify(frames: list, task: str) -> dict:
    """Returns {"failure_type": str, "reason": str}."""
    import json

    raw = call_claude(_PROMPT.format(task=task), frames_to_b64(frames))
    start, end = raw.find("{"), raw.rfind("}") + 1
    result = json.loads(raw[start:end])
    assert result["failure_type"] in FAILURE_TYPES, f"Unknown type: {result['failure_type']}"
    return result
