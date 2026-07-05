"""HTTP client for remote model server (FastAPI)."""

import requests
import numpy as np

DEFAULT_TIMEOUT = 300  # 5 min for slow inference


def get_action(host, port, obs, task_label, cfg_overrides=None):
    url = f"http://{host}:{port}/get_action"
    payload = {
        "obs": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items()},
        "task_label": task_label,
        "cfg_overrides": cfg_overrides or {},
    }
    resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return [np.array(a) for a in data["actions"]]


def health(host, port):
    resp = requests.get(f"http://{host}:{port}/health", timeout=5)
    return resp.json()
