"""
Unified Corrector with memory accumulation.

Workflow:
  1. corrector.run(frames, task) → calls `claude --print` via subprocess (no API key needed,
     uses Claude Code subscription) → parses response → stores in memory.
  2. On subsequent calls for same (task, failure_type) → memory hit, no claude call.
  3. At eval time → apply_correction() returns effective instruction for VLA.

Failure types:
  execution_misalignment → action bias params  (CORRECTION: lines)
  task_misunderstanding  → revised instruction  (REVISED: line)
  perception_failure     → instruction + spatial hint  (REVISED: / HINT: lines)
  multi_step_planning    → sub-goal reminder  (REMINDER: line)
"""

import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_HERE = Path(__file__).parent
_CLAUDE_BIN = "claude"

_EXECUTION_CORRECTION_PROMPT = """You are an expert roboticist. Watch this failure video and diagnose what went wrong, then propose action corrections.

Task: {task_instruction}

You will see ONE video showing a robot arm failing this task.

CRITICAL — Video-to-action coordinate mapping:
The video feed is rotated 180°. Directions AS SEEN IN THE VIDEO map to actions as follows:
  Video-left  = +Y action,   Video-right = -Y action
  Video-forward = +X action,  Video-backward = -X action
  Up (toward camera) = +Z,   Down (away from camera) = -Z

Action scale reference (1× action_std in meters/step):
  X: 0.336,  Y: 0.378,  Z: 0.445

---

## Your reasoning process:

### Step 1: Understand the task
- What is the robot supposed to accomplish?
- What objects are involved and where should they end up?

### Step 2: Diagnose the failure
- Watch the video frame by frame.
- At what timestamp (T_commit) does the arm commit to the wrong trajectory?
- What specifically went wrong? (wrong direction, too high, too low, wrong object, etc.)
- How far off is the trajectory from what's needed?

### Step 3: Propose corrections
- Which axes need correction? (X, Y, Z)
- What direction? (use ±X/±Y/±Z with the video-to-action mapping above)
- What time window? (t_start~t_end in seconds — start BEFORE T_commit)
- What magnitude? Start conservative: 1.0-3.0 × action_std for the relevant axis.
  - "slightly" off → 1.0× action_std
  - "moderately" off → 2.0× action_std
  - "very" off → 3.0-4.0× action_std

### Step 4: Consider envelope
- Use "ramp" if the correction should build gradually (e.g., lifting)
- Use "flat" for sustained corrections (e.g., lateral shift)
- Use "triangle" if the correction should peak in the middle

---

## Output rules:
- Use ±X/±Y/±Z notation (remember: video-left = +Y, video-right = -Y)
- magnitude_value must be an explicit unsigned float (sign encoded in ±axis direction)
- Output one CORRECTION line per axis you want to correct.
- Each CORRECTION line can optionally include envelope= and mode= parameters.
- Default mode is "add" (bias on top of VLA action). Only use "override" for gripper open/close.

Output your full reasoning above, then CORRECTION line(s):
CORRECTION: <t_start>~<t_end>s <±axis> magnitude_value=<float> [envelope=flat|ramp|triangle]"""

FAILURE_TYPES = [
    "execution_misalignment",
    "task_misunderstanding",
    "perception_failure",
    "multi_step_planning",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are a robotics failure analyst. Classify this robot manipulation failure.

Task: "{task}"

Failure types:
1. execution_misalignment: Robot reaches CORRECT target but grasp/push miscalibrated — too weak, too shallow, slightly off.
2. task_misunderstanding: Robot manipulates WRONG object — misread spatial/relational language (e.g., picks middle bowl when asked for left).
3. perception_failure: Target object IGNORED entirely — robot wanders or grabs completely wrong object.
4. multi_step_planning: FIRST sub-goal succeeds, SECOND abandoned (compound "X and Y" tasks only).

Reply with JSON only: {{"failure_type": "<one of the four>", "reason": "<one sentence>"}}"""

_MISUNDERSTANDING_PROMPT = """You are a robot instruction expert.

The robot failed this task due to task_misunderstanding — it manipulated the wrong object.
Task: "{task}"

Watch the failure frames. Rewrite the instruction to unambiguously identify the correct target
using spatial/color/size qualifiers.

End your response with:
REVISED: <rewritten instruction>"""

_PERCEPTION_PROMPT = """You are a robot instruction expert.

The robot failed due to perception_failure — it ignored or could not find the target object.
Task: "{task}"

Watch the failure frames. Identify where the target object is in the scene.
Append a spatial hint to the original instruction.

End your response with:
HINT: <spatial location, e.g. "top-right of scene", "near left edge">
REVISED: <original instruction — target is at [hint]>"""

_GOAL_STATE_PROMPT = """You are a robot task planning expert.

The robot failed due to multi_step_planning — it completed sub-goal 1 but forgot sub-goal 2.
Task: "{task}"

Watch the failure frames. Generate a standalone reminder instruction for sub-goal 2.

End your response with:
REMINDER: <complete actionable instruction for sub-goal 2>"""


# ---------------------------------------------------------------------------
# Claude CLI call
# ---------------------------------------------------------------------------

def _frames_to_b64(frames: list, n: int = 8) -> list[str]:
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


def _call_claude(prompt: str, frames_b64: list[str]) -> str:
    """
    Call `claude --print` via subprocess with text prompt + images saved as temp files.
    Images are embedded in the prompt as file paths — claude CLI reads them automatically.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Save frames as temp jpg files
        image_paths = []
        for i, b64 in enumerate(frames_b64):
            p = Path(tmp) / f"frame_{i:02d}.jpg"
            p.write_bytes(base64.b64decode(b64))
            image_paths.append(str(p))

        # Build prompt with image paths (claude CLI picks up local file refs)
        image_lines = "\n".join(f"![frame {i+1}]({p})" for i, p in enumerate(image_paths))
        full_prompt = f"{image_lines}\n\n{prompt}"

        result = subprocess.run(
            [_CLAUDE_BIN, "--print"],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI error: {result.stderr.strip()}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


def _extract_tag(text: str, tag: str) -> Optional[str]:
    for line in text.splitlines():
        if line.strip().upper().startswith(tag.upper()):
            return line.strip()[len(tag):].strip()
    return None


def _parse_correction_str(text: str) -> dict:
    """Parse CORRECTION: lines → {"axes": [...]} matching correction_params.json schema."""
    axes = []
    for line in text.splitlines():
        if not line.strip().upper().startswith("CORRECTION:"):
            continue
        part = line.strip()[len("CORRECTION:"):].strip()
        ax: dict = {"raw": part, "mode": "add"}
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
        m = re.search(r"mode=(\w+)", part)
        if m:
            ax["mode"] = m.group(1)
        axes.append(ax)
    return {"axes": axes}


# ---------------------------------------------------------------------------
# Per-type correctors
# ---------------------------------------------------------------------------

def _classify(frames: list, task: str) -> dict:
    b64 = _frames_to_b64(frames)
    raw = _call_claude(_CLASSIFY_PROMPT.format(task=task), b64)
    result = _parse_json(raw)
    assert result["failure_type"] in FAILURE_TYPES, f"Unknown type: {result['failure_type']}"
    return result


def _correct_execution(frames: list, task: str) -> dict:
    prompt = _EXECUTION_CORRECTION_PROMPT.replace("{task_instruction}", task)
    b64 = _frames_to_b64(frames)
    raw = _call_claude(prompt, b64)
    params = _parse_correction_str(raw)
    corrections = [l.strip()[len("CORRECTION:"):].strip()
                   for l in raw.splitlines() if l.strip().upper().startswith("CORRECTION:")]
    return {"type": "action", "correction_str": " | ".join(corrections), "params": params, "raw": raw}


def _correct_task_misunderstanding(frames: list, task: str) -> dict:
    raw = _call_claude(_MISUNDERSTANDING_PROMPT.format(task=task), _frames_to_b64(frames))
    revised = _extract_tag(raw, "REVISED:") or raw.splitlines()[-1].strip()
    return {"type": "instruction", "revised_instruction": revised, "raw": raw}


def _correct_perception_failure(frames: list, task: str) -> dict:
    raw = _call_claude(_PERCEPTION_PROMPT.format(task=task), _frames_to_b64(frames))
    revised = _extract_tag(raw, "REVISED:") or raw.splitlines()[-1].strip()
    hint = _extract_tag(raw, "HINT:") or ""
    return {"type": "visual", "revised_instruction": revised, "hint": hint, "raw": raw}


def _correct_multi_step(frames: list, task: str) -> dict:
    raw = _call_claude(_GOAL_STATE_PROMPT.format(task=task), _frames_to_b64(frames))
    reminder = _extract_tag(raw, "REMINDER:") or raw.splitlines()[-1].strip()
    return {"type": "goal_state", "reminder_instruction": reminder, "raw": raw}


_CORRECTORS = {
    "execution_misalignment": _correct_execution,
    "task_misunderstanding":  _correct_task_misunderstanding,
    "perception_failure":     _correct_perception_failure,
    "multi_step_planning":    _correct_multi_step,
}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class CorrectionMemory:
    def __init__(self, memory_dir: Optional[str] = None):
        self.path = Path(memory_dir or (_HERE / "corrections" / "memory"))
        self.path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple, dict] = {}
        self._load_all()

    def _filepath(self, task: str, failure_type: str) -> Path:
        safe = re.sub(r"[^\w]", "_", task)[:80]
        return self.path / f"{safe}__{failure_type}.json"

    def _load_all(self):
        for f in self.path.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self._cache[(data["task"], data["failure_type"])] = data
            except Exception:
                pass

    def retrieve(self, task: str, failure_type: str) -> Optional[dict]:
        return self._cache.get((task, failure_type))

    def store(self, task: str, failure_type: str, correction: dict, success: bool = True):
        key = (task, failure_type)
        entry = {
            "task": task, "failure_type": failure_type,
            "correction": correction, "success": success,
            "uses": self._cache.get(key, {}).get("uses", 0) + 1,
        }
        self._cache[key] = entry
        self._filepath(task, failure_type).write_text(json.dumps(entry, indent=2))
        print(f"  [memory] stored {failure_type} for '{task[:60]}'")

    def stats(self) -> dict:
        from collections import Counter
        return {"total": len(self._cache),
                "by_type": dict(Counter(v["failure_type"] for v in self._cache.values()))}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UnifiedCorrector:
    """
    Autonomous corrector: classify failure → memory lookup → call claude CLI → store.

    Usage:
        corrector = UnifiedCorrector()
        result = corrector.run(frames, task)
        instruction = apply_correction(result["correction"], task)
    """

    def __init__(self, memory_dir: Optional[str] = None):
        self.memory = CorrectionMemory(memory_dir)

    def run(self, frames: list, task: str,
            failure_type: Optional[str] = None) -> dict:
        failure_reason = ""

        if failure_type is None:
            clf = _classify(frames, task)
            failure_type = clf["failure_type"]
            failure_reason = clf.get("reason", "")
            print(f"  [classify] {failure_type}: {failure_reason}")

        cached = self.memory.retrieve(task, failure_type)
        if cached is not None:
            print(f"  [memory] HIT '{task[:60]}' ({failure_type})")
            return {"task": task, "failure_type": failure_type,
                    "failure_reason": failure_reason,
                    "correction": cached["correction"], "from_memory": True}

        print(f"  [correct] calling claude for {failure_type}...")
        correction = _CORRECTORS[failure_type](frames, task)
        self.memory.store(task, failure_type, correction)

        return {"task": task, "failure_type": failure_type,
                "failure_reason": failure_reason,
                "correction": correction, "from_memory": False}

    def stats(self) -> dict:
        return self.memory.stats()


# ---------------------------------------------------------------------------
# Apply correction → effective instruction for VLA
# ---------------------------------------------------------------------------

def apply_correction(correction: dict, task: str) -> str:
    ctype = correction.get("type")
    if ctype == "action":
        return task  # params applied externally as action bias
    elif ctype in ("instruction", "visual"):
        return correction.get("revised_instruction", task)
    elif ctype == "goal_state":
        return correction.get("reminder_instruction", task)
    return task
