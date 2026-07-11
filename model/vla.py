"""VLA inference client — HTTP calls to remote pi0.5 model server."""

import numpy as np
import requests

DEFAULT_TIMEOUT = 300  # 5 min for slow inference


def get_action(host: str, port: int, obs: dict, task_label: str,
               cfg_overrides: dict | None = None) -> list[np.ndarray]:
    payload = {
        "obs": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items()},
        "task_label": task_label,
        "cfg_overrides": cfg_overrides or {},
    }
    resp = requests.post(f"http://{host}:{port}/get_action", json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return [np.array(a) for a in resp.json()["actions"]]


def health(host: str, port: int) -> dict:
    resp = requests.get(f"http://{host}:{port}/health", timeout=5)
    return resp.json()
