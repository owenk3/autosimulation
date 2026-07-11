"""Shared Claude CLI caller and frame utilities for vlm modules."""

import base64
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

_CLAUDE_BIN = "claude"


def frames_to_b64(frames: list, n: int = 8) -> list[str]:
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


def call_claude(prompt: str, frames_b64: list[str]) -> str:
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
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI error: {result.stderr.strip()}")
    return result.stdout.strip()
