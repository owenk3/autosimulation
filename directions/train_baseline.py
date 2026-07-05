"""
Unified Corrector — action vector correction via VLM reasoning.

Like correctVLA: the VLM analyzes failure video and outputs timed action biases
(axis, direction, magnitude, envelope) to correct the VLA's actions directly.
Task instruction is NEVER rewritten — only the action vector is modified.

Exploration-exploitation:
  - Attempts 1-5: EXPLORE — try diverse action corrections to learn scene dynamics
  - Attempts 6+: EXPLOIT — refine the most promising action correction

Each attempt saves a reasoning .json with prompt_suggestion for prompt evolution research.
"""

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_CLAUDE_BIN = "claude"
_CORRECTIONS_DIR = Path(__file__).parent / "corrections"

FAILURE_TYPES = [
    "execution_misalignment",
    "task_misunderstanding",
    "perception_failure",
    "multi_step_planning",
]

EXPLORE_CUTOFF = 5
_LEARNINGS_PATH = _CORRECTIONS_DIR / "learnings.json"

# ---------------------------------------------------------------------------
# Cross-iteration learnings
# ---------------------------------------------------------------------------

def _load_learnings() -> str:
    """Load accumulated learnings — focus on proven successful corrections."""
    if not _LEARNINGS_PATH.exists():
        return ""
    data = json.loads(_LEARNINGS_PATH.read_text())
    if not data.get("learnings"):
        return ""

    # Extract all successful corrections across iterations
    proven = {}
    for l in data["learnings"]:
        for s in l.get("successful_corrections", []):
            task = s.get("task", "")
            # Keep the one with lowest attempt number (most efficient)
            if task not in proven or s.get("attempt", 99) < proven[task].get("attempt", 99):
                proven[task] = s

    if not proven:
        return ""

    lines = ["## Proven corrections from previous iterations (use as reference for SIMILAR tasks):"]
    for task, s in proven.items():
        lines.append(f"- {task[:50]}: {s.get('correction', '')[:80]} (attempt {s.get('attempt','?')})")
        if s.get("insight"):
            lines.append(f"  Insight: {s['insight'][:100]}")
    return "\n".join(lines)


def save_iteration_learnings(iteration_num: int, results: dict, corrections_dir: Path = None):
    """Analyze completed iteration and append findings to learnings.json.

    Called after each iteration completes. Reads all attempt docs from this iteration,
    extracts patterns, and saves general findings.

    Args:
        iteration_num: which iteration just completed
        results: dict with keys like {"successes": [...], "failures": [...], "success_rate": float}
    """
    cdir = corrections_dir or _CORRECTIONS_DIR
    if not _LEARNINGS_PATH.exists():
        data = {"learnings": []}
    else:
        data = json.loads(_LEARNINGS_PATH.read_text())

    # Collect all attempt docs from this iteration
    successful_corrections = []
    failed_patterns = []
    prompt_suggestions = []

    for task_dir in sorted(cdir.glob("*/")):
        attempts = []
        for f in sorted(task_dir.glob("attempt_*.json")):
            attempts.append(json.loads(f.read_text()))
        if not attempts:
            continue

        task = attempts[0].get("task", "?")
        success_attempt = None
        for a in attempts:
            if a.get("result") == "success":
                success_attempt = a
                break

        if success_attempt:
            successful_corrections.append({
                "task": task,
                "attempt": success_attempt["attempt"],
                "correction": success_attempt.get("correction_summary", ""),
                "insight": success_attempt.get("insight", ""),
                "strategy": success_attempt.get("strategy", ""),
            })
        else:
            # Collect what was tried and failed
            last = attempts[-1]
            failed_patterns.append({
                "task": task,
                "total_attempts": len(attempts),
                "last_root_cause": last.get("root_cause", "?"),
                "last_insight": last.get("insight", "?"),
            })

        for a in attempts:
            ps = a.get("prompt_suggestion", "")
            if ps and ps != "?":
                prompt_suggestions.append(ps)

    # Build summary finding
    sr = results.get("success_rate", 0)
    n_success = len(successful_corrections)
    n_fail = len(failed_patterns)

    finding_parts = [f"Iteration {iteration_num}: {sr:.0%} success ({n_success}/{n_success + n_fail})."]

    if successful_corrections:
        winning = [f"{s['task'][:40]} (attempt {s['attempt']}): {s['correction'][:80]}"
                   for s in successful_corrections[:3]]
        finding_parts.append("Successful corrections: " + " | ".join(winning))

    if failed_patterns:
        common_causes = {}
        for f in failed_patterns:
            rc = f["last_root_cause"][:60] if f["last_root_cause"] != "?" else "unknown"
            common_causes[rc] = common_causes.get(rc, 0) + 1
        top_causes = sorted(common_causes.items(), key=lambda x: -x[1])[:3]
        finding_parts.append("Common failure causes: " + "; ".join(f"{c} ({n}x)" for c, n in top_causes))

    if prompt_suggestions:
        # Deduplicate and take top 3
        unique_ps = list(dict.fromkeys(prompt_suggestions))[:3]
        finding_parts.append("VLM prompt suggestions: " + " | ".join(unique_ps))

    finding = " ".join(finding_parts)

    data["learnings"].append({
        "iteration": iteration_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success_rate": sr,
        "finding": finding,
        "successful_corrections": successful_corrections,
        "failed_patterns": failed_patterns[:5],
    })

    _LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LEARNINGS_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n[learnings] saved iteration {iteration_num} findings ({len(finding)} chars)")
    return finding


# ---------------------------------------------------------------------------
# Prompts — focused on action vector correction only
# ---------------------------------------------------------------------------

_TIMING_INFO = """
## Episode timing (CRITICAL for correct time windows):
- Total: 400 steps at 30Hz = 13.3s of action time
- You see 16 frames sampled evenly (~0.9s apart): frame 1≈0s, frame 5≈3.5s, frame 9≈7.1s, frame 13≈10.6s, frame 16≈13.3s
- Typical task phases: approach (0~4s), grasp/contact (4~7s), transport/place (7~13s)
- Your time windows MUST fall within 0~13.3s"""

_COORD_INFO = """Video coords (camera rotated 180°): Video-left=+Y, Video-right=-Y, Video-forward=+X, Video-backward=-X, Up=+Z, Down=-Z
Action scale (1× std): X:0.336, Y:0.378, Z:0.445
- "slightly off" → magnitude 0.1-0.3
- "moderately off" → magnitude 0.3-0.6
- "very off" → magnitude 0.6-1.0
IMPORTANT: The VLA is a capable policy. Corrections are gentle nudges, NOT overrides.
- Start with SMALL magnitudes (0.1-0.2)
- NEVER exceed magnitude 0.5 — anything higher destroys the VLA's natural behavior
- If 0.5 isn't enough, the AXIS or TIMING is wrong, not the magnitude"""

_INITIAL_PROMPT = """You are an expert roboticist. A robot failed a manipulation task. Diagnose the failure and propose ACTION VECTOR corrections.

Task: {task}
{learnings}
""" + _TIMING_INFO + """

## Step 1: Map the scene precisely (do this FIRST)
Look at frame 1 carefully and describe EXACTLY:
- Where is each relevant object? (e.g. "black bowl is at video-center, plate is ~3cm to the video-left")
- Where is the robot gripper starting?
- Where must the object end up? Estimate the distance in cm.
- This scene map is your reference for all corrections — be specific, not vague.

## Step 2: Classify failure type
- execution_misalignment: correct object, but grasp/push is off
- task_misunderstanding: wrong object (misread spatial language)
- perception_failure: target ignored entirely
- multi_step_planning: first sub-task ok, second abandoned

## Step 3: Correction strategy (based on type)
- execution_misalignment → small corrections (1-2 axes, 1-3s windows), focus on failure moment
- perception_failure → large redirections (0~6s, magnitude 0.4-0.8)
- task_misunderstanding → map text to scene, sustained lateral correction during approach
- multi_step_planning → corrections only in second phase (7~13s)
""" + _COORD_INFO + """

## Output:
FAILURE_POINT: <frame number and timestamp>
FAILURE_TYPE: <one of four types>
ROOT_CAUSE: <what went wrong and why>
SCENE_MAP: <object positions and target location from frame 1>
INSIGHT: <estimated distance/direction the arm needs to be corrected>
STRATEGY: <correction approach>
PROMPT_SUGGESTION: <how to improve>

Max 2 CORRECTION lines:
CORRECTION: <t_start>~<t_end>s <±axis> magnitude_value=<float> [envelope=flat|ramp|triangle]"""

_EXPLORE_PROMPT = """You are an expert roboticist. The robot failed AGAIN.

Task: {task}
{learnings}
""" + _TIMING_INFO + """

## Previous attempts:
{history_text}

## Attempt {attempt_num} — VISUAL COMPARISON:

Look at the NEW failure video and compare to what you know from previous attempts:

1. Pick a KEY FRAME where the failure is visible. Describe EXACTLY:
   - "At frame 8, the gripper is ~2cm to the left of the bowl" (be specific in cm)
   - Compare to the SAME frame in the previous attempt — did it move? How much?

2. Proportional self-calibration:
   - Example: "magnitude 0.2 moved the arm ~1cm. I need ~2cm more. Try 0.2 × (2/1) = 0.4"
   - NEVER exceed 0.5. If 0.5 isn't enough → wrong axis or wrong timing.
   - If the arm barely moved at 0.3+ → the axis is WRONG, switch to a different one.

3. Change ONE thing. Be precise, not aggressive.
""" + _COORD_INFO + """

## Output:
FAILURE_POINT: <frame and timestamp>
FAILURE_TYPE: <type>
ROOT_CAUSE: <current issue>
VISUAL_DIFF: <what visibly changed from previous attempt at key frames>
INSIGHT: <self-calibration — how far did we move vs how far we need to go>
STRATEGY: <what to adjust and why>
PROMPT_SUGGESTION: <what would help>

Max 2 CORRECTION lines:
CORRECTION: <t_start>~<t_end>s <±axis> magnitude_value=<float> [envelope=flat|ramp|triangle]"""

_EXPLOIT_PROMPT = """You are an expert roboticist. Refine the best action correction found so far.

Task: {task}
{learnings}
""" + _TIMING_INFO + """

## Previous attempts and learnings:
{history_text}

## Best approach so far:
{best_summary}

## EXPLOITATION — Attempt {attempt_num}

Compare the latest failure to the best attempt at the KEY FRAME:
1. Describe EXACTLY how close: "gripper is ~1cm short in +X direction"
2. Proportional adjustment: if current magnitude produced ~2cm and you need ~3cm total, scale by 3/2
3. NEVER exceed magnitude 0.5. If you're at 0.5 and still failing → wrong axis or timing.
""" + _COORD_INFO + """

## Output:
FAILURE_POINT: <frame number and timestamp>
FAILURE_TYPE: <type>
ROOT_CAUSE: <remaining issue>
INSIGHT: <what micro-tuning is needed>
STRATEGY: <specific adjustment and why>
PROMPT_SUGGESTION: <what would help exploitation prompts work better?>

Action corrections:
CORRECTION: <t_start>~<t_end>s <±axis> magnitude_value=<float> [envelope=flat|ramp|triangle]"""

# ---------------------------------------------------------------------------
# Claude CLI
# ---------------------------------------------------------------------------

def _frames_to_b64(frames: list, n: int = 16) -> list[str]:
    if not frames:
        return []
    indices = [int(len(frames) * i / max(n - 1, 1)) for i in range(n)]
    result = []
    for idx in indices:
        frame = frames[min(idx, len(frames) - 1)]
        if not isinstance(frame, np.ndarray):
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        result.append(base64.b64encode(buf).decode())
    return result


def _call_claude(prompt: str, frames_b64: list[str], max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                image_paths = []
                for i, b64 in enumerate(frames_b64):
                    p = Path(tmp) / f"frame_{i:02d}.jpg"
                    p.write_bytes(base64.b64decode(b64))
                    image_paths.append(str(p))
                image_lines = "\n".join(f"![frame {i+1}]({p})" for i, p in enumerate(image_paths))
                full_prompt = f"{image_lines}\n\n{prompt}"
                result = subprocess.run(
                    [_CLAUDE_BIN, "--print"],
                    input=full_prompt, capture_output=True, text=True, timeout=600,
                )
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI error: {result.stderr.strip()}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"  [claude] timeout on attempt {attempt+1}/{max_retries}, retrying...")
            if attempt == max_retries - 1:
                return "FAILURE_TYPE: execution_misalignment\nCORRECTION: 2.0~6.0s -Z magnitude_value=0.3 envelope=flat"


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract tag value, handling markdown formatting (**TAG:**, ## TAG:, etc.)."""
    tag_upper = tag.upper()
    for line in text.splitlines():
        cleaned = re.sub(r'^[\s#*\-]+', '', line).strip()
        if cleaned.upper().startswith(tag_upper + ":"):
            val = cleaned[len(tag) + 1:].strip()
            return re.sub(r'^\*+|\*+$', '', val).strip()  # strip trailing **
        if line.strip().upper().startswith(tag_upper + ":"):
            val = line.strip()[len(tag) + 1:].strip()
            return re.sub(r'^\*+|\*+$', '', val).strip()
    return None


def _parse_response(text: str) -> dict:
    """Parse VLM response → action bias correction dict."""
    correction = {"raw": text, "type": "action"}

    axes = []
    for line in text.splitlines():
        cleaned = re.sub(r'^[\s#*\-]+', '', line).strip()
        if not cleaned.upper().startswith("CORRECTION:"):
            continue
        part = cleaned[len("CORRECTION:"):].strip()
        ax = {"raw": part, "mode": "add"}
        m = re.search(r"([\d.]+)~([\d.]+)s", part)
        if m:
            ax["t_start"], ax["t_end"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"([+-])([xyzXYZ])\b", part)
        if m:
            ax["direction"] = "+1" if m.group(1) == "+" else "-1"
            ax["dimension"] = m.group(2).lower()
        m = re.search(r"magnitude_value=([\d.]+)", part)
        if m:
            ax["magnitude_value"] = float(m.group(1))
        m = re.search(r"envelope=(\w+)", part)
        if m:
            ax["envelope"] = m.group(1)
        m = re.search(r"mode=(\w+)", part)
        if m:
            ax["mode"] = m.group(1)
        axes.append(ax)

    correction["params"] = {"axes": axes}

    for tag in ["FAILURE_POINT", "ROOT_CAUSE", "SCENE_MAP", "VISUAL_DIFF", "INSIGHT", "STRATEGY", "PROMPT_SUGGESTION"]:
        val = _extract_tag(text, tag)
        if val:
            correction[tag.lower()] = val

    return correction


def _extract_failure_type(text: str) -> str:
    for line in text.splitlines():
        cleaned = re.sub(r'^[\s#*\-]+', '', line).strip()
        if cleaned.upper().startswith("FAILURE_TYPE:"):
            ft = cleaned[len("FAILURE_TYPE:"):].strip().lower()
            if ft in FAILURE_TYPES:
                return ft
    return "execution_misalignment"


# ---------------------------------------------------------------------------
# Persistent reasoning docs
# ---------------------------------------------------------------------------

def _task_slug(task: str) -> str:
    return re.sub(r"[^\w]", "_", task)[:80]


def _summarize_correction(c: dict) -> str:
    parts = []
    for ax in c.get("params", {}).get("axes", []):
        parts.append(ax.get("raw", ""))
    return " | ".join(parts) if parts else "no action corrections"


def _save_reasoning(task: str, attempt: int, entry: dict):
    """Save per-attempt reasoning doc."""
    task_dir = _CORRECTIONS_DIR / _task_slug(task)
    task_dir.mkdir(parents=True, exist_ok=True)

    c = entry.get("correction", {})
    phase = entry.get("phase", "explore")
    doc = {
        "task": task,
        "attempt": attempt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "prompt_used": f"{'INITIAL' if attempt == 1 else phase.upper()}_PROMPT",
        "what_changed": (
            "first attempt — vanilla VLA failed, VLM proposes bold action correction"
            if attempt == 1 else
            f"attempt {attempt} — VLM sees {attempt-1} prior attempt(s) + new failure frames, "
            f"{'tries DIFFERENT action correction (explore)' if phase == 'explore' else 'micro-tunes best action correction (exploit)'}"
        ),
        "failure_type": entry.get("failure_type", "unknown"),
        "failure_point": c.get("failure_point", "unknown"),
        "root_cause": c.get("root_cause", "unknown"),
        "scene_map": c.get("scene_map", ""),
        "visual_diff": c.get("visual_diff", ""),
        "insight": c.get("insight", ""),
        "strategy": c.get("strategy", ""),
        "correction_summary": _summarize_correction(c),
        "prompt_suggestion": c.get("prompt_suggestion", ""),
        "result": entry.get("result", "pending"),
    }
    path = task_dir / f"attempt_{attempt:02d}.json"
    path.write_text(json.dumps(doc, indent=2))
    return doc


def _load_task_history(task: str) -> list[dict]:
    task_dir = _CORRECTIONS_DIR / _task_slug(task)
    if not task_dir.exists():
        return []
    history = []
    for f in sorted(task_dir.glob("attempt_*.json")):
        history.append(json.loads(f.read_text()))
    return history


def _build_history_text(history: list[dict]) -> str:
    lines = []
    for h in history:
        lines.append(
            f"Attempt {h['attempt']} [{h.get('phase','?')}] → {h.get('result','?')}\n"
            f"  Failure point: {h.get('failure_point', '?')}\n"
            f"  Root cause: {h.get('root_cause', '?')}\n"
            f"  Visual diff: {h.get('visual_diff', '?')}\n"
            f"  Insight: {h.get('insight', '?')}\n"
            f"  Strategy: {h.get('strategy', '?')}\n"
            f"  Action correction: {h.get('correction_summary', 'none')}")
    return "\n".join(lines)


def _find_best_attempt(history: list[dict]) -> str:
    best = history[-1]
    for h in history:
        if h.get("result") == "success":
            best = h
            break
    return (f"Attempt {best['attempt']}: {best.get('strategy', 'unknown')}\n"
            f"  Action correction: {best.get('correction_summary', 'none')}\n"
            f"  Insight: {best.get('insight', 'none')}")


# ---------------------------------------------------------------------------
# Main corrector
# ---------------------------------------------------------------------------

class UnifiedCorrector:
    def __init__(self, memory_dir: Optional[str] = None):
        self._history: dict[str, list[dict]] = {}

    def _get_proven_correction(self, task: str) -> Optional[dict]:
        """Check if we have a proven correction for this exact task from learnings."""
        if not _LEARNINGS_PATH.exists():
            return None
        data = json.loads(_LEARNINGS_PATH.read_text())
        for l in data.get("learnings", []):
            for s in l.get("successful_corrections", []):
                if s.get("task") == task and s.get("correction"):
                    # Reconstruct correction dict from stored summary
                    return s
        return None

    def run(self, frames: list, task: str, failure_type: Optional[str] = None) -> dict:
        if task not in self._history:
            self._history[task] = _load_task_history(task)
        history = self._history[task]
        attempt_num = len(history) + 1
        phase = "explore" if attempt_num <= EXPLORE_CUTOFF else "exploit"

        # Attempt 1: try proven correction if available (skip VLM call)
        if attempt_num == 1:
            proven = self._get_proven_correction(task)
            if proven:
                print(f"  [proven] reusing successful correction from previous iteration")
                # Parse the stored correction summary back into params
                raw = f"FAILURE_TYPE: execution_misalignment\nUsing proven correction.\n"
                for part in proven.get("correction", "").split(" | "):
                    raw += f"CORRECTION: {part}\n"
                correction = _parse_response(raw)
                correction["insight"] = f"Reusing proven correction: {proven.get('insight', '')[:100]}"
                correction["strategy"] = "Apply proven correction from previous successful iteration"
                entry = {"attempt": attempt_num, "phase": "proven",
                         "failure_type": "execution_misalignment", "correction": correction}
                doc = _save_reasoning(task, attempt_num, entry)
                self._history[task].append(doc)
                return {"task": task, "failure_type": "execution_misalignment",
                        "failure_reason": "", "correction": correction, "from_memory": True}

        b64 = _frames_to_b64(frames)
        learnings = _load_learnings()

        if attempt_num == 1:
            raw = _call_claude(_INITIAL_PROMPT.format(task=task, learnings=learnings), b64)
        elif phase == "explore":
            history_text = _build_history_text(history)
            raw = _call_claude(
                _EXPLORE_PROMPT.format(task=task, learnings=learnings,
                                       history_text=history_text, attempt_num=attempt_num), b64)
        else:
            history_text = _build_history_text(history)
            best_summary = _find_best_attempt(history)
            raw = _call_claude(
                _EXPLOIT_PROMPT.format(task=task, learnings=learnings,
                                       history_text=history_text,
                                       best_summary=best_summary, attempt_num=attempt_num), b64)

        failure_type = _extract_failure_type(raw)
        correction = _parse_response(raw)

        entry = {"attempt": attempt_num, "phase": phase,
                 "failure_type": failure_type, "correction": correction}

        doc = _save_reasoning(task, attempt_num, entry)
        self._history[task].append(doc)

        print(f"  [{phase}] attempt {attempt_num}")
        print(f"  [failure_point] {correction.get('failure_point', '?')}")
        print(f"  [root_cause] {correction.get('root_cause', '?')}")
        print(f"  [insight] {correction.get('insight', '?')}")
        print(f"  [strategy] {correction.get('strategy', '?')}")

        return {"task": task, "failure_type": failure_type,
                "failure_reason": correction.get("root_cause", ""),
                "correction": correction, "from_memory": False}

    def mark_result(self, task: str, attempt: int, result: str):
        task_dir = _CORRECTIONS_DIR / _task_slug(task)
        path = task_dir / f"attempt_{attempt:02d}.json"
        if path.exists():
            doc = json.loads(path.read_text())
            doc["result"] = result
            path.write_text(json.dumps(doc, indent=2))
        for h in self._history.get(task, []):
            if h.get("attempt") == attempt:
                h["result"] = result

    def reset_task(self, task: str):
        self._history.pop(task, None)

    def stats(self) -> dict:
        n_tasks = len(list(_CORRECTIONS_DIR.glob("*/"))) if _CORRECTIONS_DIR.exists() else 0
        return {"persisted_tasks": n_tasks,
                "active_tasks": len(self._history),
                "total_attempts": sum(len(v) for v in self._history.values())}


# ---------------------------------------------------------------------------
# Apply correction — always returns original task (no instruction rewriting)
# ---------------------------------------------------------------------------

def apply_correction(correction: dict, task: str) -> str:
    return task  # action bias applied externally in prepare.py via correction["params"]
