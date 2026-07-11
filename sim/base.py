"""Abstract base class for simulation environments."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SimEnv(ABC):
    """
    Minimal interface every sim adapter must implement.
    Decouples the episode runner from any specific sim backend.
    """

    @abstractmethod
    def reset(self) -> Any:
        """Reset env, return initial obs."""

    @abstractmethod
    def set_init_state(self, state: Any) -> Any:
        """Set deterministic initial state, return obs."""

    @abstractmethod
    def step(self, action: list) -> tuple[Any, float, bool, dict]:
        """Step env. Returns (obs, reward, done, info)."""

    @abstractmethod
    def get_image(self, obs: Any) -> np.ndarray:
        """Extract agentview RGB frame from obs."""

    @abstractmethod
    def get_wrist_image(self, obs: Any) -> np.ndarray:
        """Extract wrist-cam RGB frame from obs."""

    @abstractmethod
    def get_state(self, obs: Any) -> np.ndarray:
        """Extract robot state vector (eef pos + orientation + gripper)."""

    @abstractmethod
    def close(self) -> None:
        """Clean up env resources."""

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Natural language task instruction."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum episode length in steps."""

    @abstractmethod
    def dummy_action(self) -> list:
        """Zero/neutral action for wait steps."""
