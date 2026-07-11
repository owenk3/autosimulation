"""Verifier — scores how close an attempt is to task completion.

Two modes:
  - fast: pixel-diff heuristic (no LLM call, instant)
  - vlm:  asks Claude to score 0-100 (slower, more accurate)
"""

import cv2
import numpy as np

from vlm._claude import call_claude, frames_to_b64

_SCORE_PROMPT = """You are a robotics evaluator. Score how close this robot attempt is to completing the task.

Task: "{task}"

Watch the frames. Give a score from 0 to 100:
  0   = robot did nothing / moved wrong direction entirely
  50  = partially correct (reached object but failed grasp, or wrong target)
  100 = task fully completed

Reply with JSON only: {{"score": <int 0-100>, "reason": "<one sentence>"}}"""


def score_vlm(frames: list, task: str) -> dict:
    """Ask Claude to score attempt quality. Returns {"score": int, "reason": str}."""
    import json

    raw = call_claude(_SCORE_PROMPT.format(task=task), frames_to_b64(frames))
    start, end = raw.find("{"), raw.rfind("}") + 1
    return json.loads(raw[start:end])


def score_fast(prev_frames: list, curr_frames: list) -> float:
    """
    Heuristic: compare last frames of two consecutive attempts.
    Returns 0.0-1.0 — higher means curr attempt changed more (proxy for progress).
    Not task-aware; use as a cheap pre-filter before score_vlm.
    """
    if not prev_frames or not curr_frames:
        return 0.0

    def last_gray(frames):
        f = frames[-1]
        if f.ndim == 3:
            return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return f.astype(np.float32)

    diff = np.abs(last_gray(curr_frames) - last_gray(prev_frames))
    return float(np.mean(diff) / 255.0)


def improved(prev_frames: list, curr_frames: list,
             task: str, mode: str = "fast",
             threshold: float = 0.01) -> bool:
    """
    Returns True if curr attempt is measurably better than prev.
    mode="fast" uses pixel diff; mode="vlm" uses Claude scoring.
    """
    if mode == "vlm":
        prev_score = score_vlm(prev_frames, task)["score"]
        curr_score = score_vlm(curr_frames, task)["score"]
        return curr_score > prev_score
    return score_fast(prev_frames, curr_frames) > threshold
