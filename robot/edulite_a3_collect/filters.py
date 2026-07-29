"""Small, dependency-free filters used by the teleoperation controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FilterState:
    position: np.ndarray
    velocity: np.ndarray


class CriticallyDampedJointFilter:
    """Second-order low-pass filter with velocity/acceleration limiting.

    The continuous target dynamics are::

        x'' = omega^2 (target - x) - 2 omega x'

    Semi-implicit Euler integration is used because it is stable at the small
    control periods used here.  Limits are applied independently per joint.
    """

    def __init__(
        self,
        initial_position: Sequence[float],
        omega: float,
        max_velocity: Sequence[float] | float,
        max_acceleration: Sequence[float] | float,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
    ) -> None:
        initial = np.asarray(initial_position, dtype=np.float64)
        if initial.ndim != 1:
            raise ValueError("initial_position must be one-dimensional")
        if omega <= 0:
            raise ValueError("omega must be positive")

        self.omega = float(omega)
        self.max_velocity = self._broadcast(max_velocity, initial.size, "max_velocity")
        self.max_acceleration = self._broadcast(max_acceleration, initial.size, "max_acceleration")
        self.lower_limits = np.asarray(lower_limits, dtype=np.float64)
        self.upper_limits = np.asarray(upper_limits, dtype=np.float64)
        if self.lower_limits.shape != initial.shape or self.upper_limits.shape != initial.shape:
            raise ValueError("joint limits must match initial_position")
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("every lower joint limit must be smaller than its upper limit")

        self.state = FilterState(
            position=np.clip(initial, self.lower_limits, self.upper_limits),
            velocity=np.zeros_like(initial),
        )

    @staticmethod
    def _broadcast(value: Sequence[float] | float, size: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(size, float(arr), dtype=np.float64)
        if arr.shape != (size,):
            raise ValueError(f"{name} must be a scalar or a sequence of length {size}")
        if np.any(arr <= 0):
            raise ValueError(f"{name} entries must be positive")
        return arr

    def reset(self, position: Sequence[float]) -> None:
        pos = np.asarray(position, dtype=np.float64)
        if pos.shape != self.state.position.shape:
            raise ValueError("reset position has the wrong shape")
        self.state.position = np.clip(pos, self.lower_limits, self.upper_limits)
        self.state.velocity.fill(0.0)

    def step(self, target: Sequence[float], dt: float) -> FilterState:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Avoid a debugger pause or scheduler stall creating a large step.
        dt = min(float(dt), 0.05)

        target_arr = np.asarray(target, dtype=np.float64)
        if target_arr.shape != self.state.position.shape:
            raise ValueError("target has the wrong shape")
        target_arr = np.clip(target_arr, self.lower_limits, self.upper_limits)

        error = target_arr - self.state.position
        acceleration = (
            self.omega * self.omega * error
            - 2.0 * self.omega * self.state.velocity
        )
        acceleration = np.clip(
            acceleration,
            -self.max_acceleration,
            self.max_acceleration,
        )
        velocity = self.state.velocity + acceleration * dt
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        position = self.state.position + velocity * dt
        position = np.clip(position, self.lower_limits, self.upper_limits)

        # Do not retain velocity that points farther into a saturated limit.
        hit_lower = (position <= self.lower_limits) & (velocity < 0)
        hit_upper = (position >= self.upper_limits) & (velocity > 0)
        velocity[hit_lower | hit_upper] = 0.0

        self.state = FilterState(position=position, velocity=velocity)
        return FilterState(position=position.copy(), velocity=velocity.copy())
