"""Persistent correction memory — store and retrieve per (task, failure_type)."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(__file__).parent.parent / "corrections" / "memory"


class CorrectionMemory:
    def __init__(self, memory_dir: Optional[str] = None):
        self.path = Path(memory_dir or _DEFAULT_DIR)
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
            "task": task,
            "failure_type": failure_type,
            "correction": correction,
            "success": success,
            "uses": self._cache.get(key, {}).get("uses", 0) + 1,
        }
        self._cache[key] = entry
        self._filepath(task, failure_type).write_text(json.dumps(entry, indent=2))
        print(f"  [memory] stored {failure_type} for '{task[:60]}'")

    def stats(self) -> dict:
        return {
            "total": len(self._cache),
            "by_type": dict(Counter(v["failure_type"] for v in self._cache.values())),
        }
