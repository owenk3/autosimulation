"""LIBERO simulation adapter."""

import math
import os

import numpy as np

from sim.base import SimEnv

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoEnv(SimEnv):
    """Wraps LIBERO OffScreenRenderEnv behind the SimEnv interface."""

    def __init__(self, task, suite_name: str, resolution: int = 256):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self._suite_name = suite_name
        self._task_description: str = task.language
        self._max_steps: int = TASK_MAX_STEPS[suite_name]

        task_bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self._env.seed(0)

    # --- SimEnv interface ---

    def reset(self):
        return self._env.reset()

    def set_init_state(self, state):
        return self._env.set_init_state(state)

    def step(self, action: list):
        return self._env.step(action)

    def get_image(self, obs) -> np.ndarray:
        return obs["agentview_image"][::-1, ::-1]

    def get_wrist_image(self, obs) -> np.ndarray:
        return obs["robot0_eye_in_hand_image"][::-1, ::-1]

    def get_state(self, obs) -> np.ndarray:
        return np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ))

    def close(self) -> None:
        self._env.close()

    def dummy_action(self) -> list:
        return [0, 0, 0, 0, 0, 0, -1]

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def max_steps(self) -> int:
        return self._max_steps


def _get_suite(suite_name: str):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite_name]()


def suite_n_tasks(suite_name: str) -> int:
    return _get_suite(suite_name).n_tasks


def load_task(suite_name: str, episode: int, num_trials_per_task: int = 50):
    """Return (task, initial_state, task_id) for a global episode index (1-indexed)."""
    idx = episode - 1
    task_id = idx // num_trials_per_task
    episode_idx = idx % num_trials_per_task

    task_suite = _get_suite(suite_name)
    task = task_suite.get_task(task_id)
    initial_state = task_suite.get_task_init_states(task_id)[episode_idx]
    return task, initial_state, task_id
