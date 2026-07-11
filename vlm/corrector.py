"""Per-failure-type correctors — each returns a correction dict."""

import re
from typing import Optional

from vlm._claude import call_claude, frames_to_b64

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXECUTION_PROMPT = """You are an expert roboticist. Watch this failure video and diagnose what went wrong, then propose action corrections.

Task: {task}

You will see ONE video showing a robot arm failing this task.

CRITICAL — Video-to-action coordinate mapping:
The video feed is rotated 180°. Directions AS SEEN IN THE VIDEO map to actions as follows:
  Video-left  = +Y action,   Video-right = -Y action
  Video-forward = +X action,  Video-backward = -X action
  Up (toward camera) = +Z,   Down (away from camera) = -Z

Action scale reference (1× action_std in meters/step):
  X: 0.336,  Y: 0.378,  Z: 0.445

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

### Step 4: Consider envelope
- Use "ramp" if the correction should build gradually
- Use "flat" for sustained corrections
- Use "triangle" if the correction should peak in the middle

Output your full reasoning above, then CORRECTION line(s):
CORRECTION: <t_start>~<t_end>s <±axis> magnitude_value=<float> [envelope=flat|ramp|triangle]"""

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
# Parsers
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> Optional[str]:
    for line in text.splitlines():
        if line.strip().upper().startswith(tag.upper()):
            return line.strip()[len(tag):].strip()
    return None


def _parse_correction_lines(text: str) -> dict:
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

def correct_execution(frames: list, task: str) -> dict:
    raw = call_claude(_EXECUTION_PROMPT.format(task=task), frames_to_b64(frames))
    params = _parse_correction_lines(raw)
    corrections = [l.strip()[len("CORRECTION:"):].strip()
                   for l in raw.splitlines() if l.strip().upper().startswith("CORRECTION:")]
    return {"type": "action", "correction_str": " | ".join(corrections), "params": params, "raw": raw}


def correct_task_misunderstanding(frames: list, task: str) -> dict:
    raw = call_claude(_MISUNDERSTANDING_PROMPT.format(task=task), frames_to_b64(frames))
    revised = _extract_tag(raw, "REVISED:") or raw.splitlines()[-1].strip()
    return {"type": "instruction", "revised_instruction": revised, "raw": raw}


def correct_perception_failure(frames: list, task: str) -> dict:
    raw = call_claude(_PERCEPTION_PROMPT.format(task=task), frames_to_b64(frames))
    revised = _extract_tag(raw, "REVISED:") or raw.splitlines()[-1].strip()
    hint = _extract_tag(raw, "HINT:") or ""
    return {"type": "visual", "revised_instruction": revised, "hint": hint, "raw": raw}


def correct_multi_step(frames: list, task: str) -> dict:
    raw = call_claude(_GOAL_STATE_PROMPT.format(task=task), frames_to_b64(frames))
    reminder = _extract_tag(raw, "REMINDER:") or raw.splitlines()[-1].strip()
    return {"type": "goal_state", "reminder_instruction": reminder, "raw": raw}


_DISPATCH = {
    "execution_misalignment": correct_execution,
    "task_misunderstanding":  correct_task_misunderstanding,
    "perception_failure":     correct_perception_failure,
    "multi_step_planning":    correct_multi_step,
}


def correct(frames: list, task: str, failure_type: str) -> dict:
    """Dispatch to the right corrector by failure_type."""
    fn = _DISPATCH.get(failure_type)
    if fn is None:
        raise ValueError(f"Unknown failure_type: {failure_type}")
    return fn(frames, task)
