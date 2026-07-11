"""
train.py — the ONLY file the autonomous agent edits.

Wires vlm + memory into UnifiedCorrector and apply_correction.
The agent tunes prompts in vlm/corrector.py and vlm/judge.py to improve success rate.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.store import CorrectionMemory
from vlm import corrector as vlm_corrector
from vlm import judge as vlm_judge


class UnifiedCorrector:
    """
    Classify failure → memory lookup → call VLM corrector → store result.

    Usage:
        c = UnifiedCorrector()
        result = c.run(frames, task)
        instruction = apply_correction(result["correction"], task)
    """

    def __init__(self, memory_dir: Optional[str] = None):
        self.memory = CorrectionMemory(memory_dir)

    def run(self, frames: list, task: str,
            failure_type: Optional[str] = None) -> dict:
        failure_reason = ""

        if failure_type is None:
            clf = vlm_judge.classify(frames, task)
            failure_type = clf["failure_type"]
            failure_reason = clf.get("reason", "")
            print(f"  [judge] {failure_type}: {failure_reason}")

        cached = self.memory.retrieve(task, failure_type)
        if cached is not None:
            print(f"  [memory] HIT '{task[:60]}' ({failure_type})")
            return {"task": task, "failure_type": failure_type,
                    "failure_reason": failure_reason,
                    "correction": cached["correction"], "from_memory": True}

        print(f"  [corrector] calling vlm for {failure_type}...")
        correction = vlm_corrector.correct(frames, task, failure_type)
        self.memory.store(task, failure_type, correction)

        return {"task": task, "failure_type": failure_type,
                "failure_reason": failure_reason,
                "correction": correction, "from_memory": False}

    def stats(self) -> dict:
        return self.memory.stats()


def apply_correction(correction: dict, task: str) -> str:
    """Return effective task instruction given a correction dict."""
    ctype = correction.get("type")
    if ctype == "action":
        return task
    if ctype in ("instruction", "visual"):
        return correction.get("revised_instruction", task)
    if ctype == "goal_state":
        return correction.get("reminder_instruction", task)
    return task
