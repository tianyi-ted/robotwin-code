#!/usr/bin/env python3
"""Safety-gated VLA inference for one physical EDULITE-A3 follower arm.

The model produces the existing 14-D RoboTwin action.  Only action[0:7] is
accepted: six follower joint targets plus one normalized gripper target.  The
synthetic right-arm action is always ignored.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
COLLECTOR_ROOT = PROJECT_ROOT / "robot" / "edulite_a3_collect"
for path in (PROJECT_ROOT, COLLECTOR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _make_pinocchio_visible() -> None:
    """Expose the existing edulite_collect Pinocchio wheel to RoboTwin.

    The EDULITE SDK uses raw SocketCAN and needs no python-can package.  Only
    Pinocchio's Python module is absent from the RoboTwin environment.  The
    existing wheel is ABI-compatible with the current Python 3.10 runtime.
    """

    try:
        import pinocchio  # noqa: F401
        return
    except ImportError:
        pass

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates: list[Path] = []
    explicit = os.environ.get("EDULITE_PINOCCHIO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    # .../envs/RoboTwin/bin/python -> .../envs/edulite_collect
    envs_root = Path(sys.executable).resolve().parents[2]
    candidates.append(
        envs_root
        / "edulite_collect"
        / "lib"
        / pyver
        / "site-packages"
        / "cmeel.prefix"
        / "lib"
        / pyver
        / "site-packages"
    )
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))
            try:
                import pinocchio  # noqa: F401
                return
            except ImportError:
                continue
    raise RuntimeError(
        "Pinocchio is unavailable. Activate RoboTwin and set "
        "EDULITE_PINOCCHIO_PATH to edulite_collect's "
        "cmeel.prefix/lib/python3.10/site-packages directory."
    )


_make_pinocchio_visible()

from cameras import build_cameras  # noqa: E402
from filters import CriticallyDampedJointFilter  # noqa: E402
from hardware import (  # noqa: E402
    DEFAULT_LIMITS,
    GripperCalibration,
    _feedback,
    _make_arm,
    _pose,
    _wait_feedback_ready,
)
from visualization import InferenceVisualizer  # noqa: E402


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_deployment_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("deployment configuration root must be a mapping")
    for section in ("model", "hardware", "runtime", "safety"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")
    base = config_path.parent
    for key in ("config_path", "checkpoint_path", "norm_stats_path"):
        config["model"][key] = str(_resolve(base, config["model"][key]))
    config["hardware"]["collector_config_path"] = str(
        _resolve(base, config["hardware"]["collector_config_path"])
    )
    visualization = config.setdefault("visualization", {})
    if visualization.get("output_root"):
        visualization["output_root"] = str(
            _resolve(base, visualization["output_root"])
        )
    return config, config_path


def load_collector_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("collector configuration root must be a mapping")
    if not bool(config.get("hardware_calibrated", False)):
        raise RuntimeError("collector configuration has hardware_calibrated=false")
    return config


def load_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stats = data.get("robotwin2")
    if not isinstance(stats, dict):
        raise ValueError("normalization file is missing robotwin2")
    action = stats.get("action", {})
    state = stats.get("state", {})
    action_min = np.asarray(action.get("min"), dtype=np.float64)
    action_max = np.asarray(action.get("max"), dtype=np.float64)
    state_min = np.asarray(state.get("min"), dtype=np.float64)
    state_max = np.asarray(state.get("max"), dtype=np.float64)
    if action_min.shape != (14,) or action_max.shape != (14,):
        raise ValueError("normalization action stats must be 14-D")
    if state_min.shape != (16,) or state_max.shape != (16,):
        raise ValueError("normalization state stats must be 16-D")
    for name, value in {
        "action_min": action_min,
        "action_max": action_max,
        "state_min": state_min,
        "state_max": state_max,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if not np.allclose(action_min[7:], 0.0) or not np.allclose(action_max[7:], 0.0):
        raise ValueError("this single-arm deployment requires zero-padded action[7:14]")
    if not np.allclose(state_min[8:], 0.0) or not np.allclose(state_max[8:], 0.0):
        raise ValueError("this single-arm deployment requires zero-padded state[8:16]")
    return action_min, action_max, state_min, state_max, stats.get("meta", {})


def static_check(config: dict) -> dict:
    model_cfg = Path(config["model"]["config_path"])
    checkpoint = Path(config["model"]["checkpoint_path"])
    stats_path = Path(config["model"]["norm_stats_path"])
    collector_path = Path(config["hardware"]["collector_config_path"])
    for label, path in {
        "model config": model_cfg,
        "checkpoint": checkpoint,
        "normalization stats": stats_path,
        "collector config": collector_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    with model_cfg.open("r", encoding="utf-8") as handle:
        train_cfg = yaml.safe_load(handle)
    common = train_cfg.get("common", {})
    if int(common.get("action_dim", -1)) != 14:
        raise ValueError("trained model action_dim is not 14")
    if int(common.get("state_dim", -1)) != 16:
        raise ValueError("trained model state_dim is not 16")
    cameras = train_cfg.get("dataset", {}).get("camera_names", [])
    requested_camera = config["hardware"]["camera_name"]
    if cameras != [requested_camera]:
        raise ValueError(
            f"model cameras {cameras!r} do not match runtime camera {requested_camera!r}"
        )
    state_indices = train_cfg.get("dataset", {}).get("indices_config", {}).get(
        "state_indices", []
    )
    if state_indices != [0]:
        raise ValueError(
            "the real prefetch runtime currently requires state_indices=[0]"
        )

    collector = load_collector_config(collector_path)
    side = str(config["hardware"].get("arm_side", "right"))
    if side not in {"left", "right"}:
        raise ValueError("hardware.arm_side must be left or right")
    camera_configs = [c for c in collector.get("cameras", []) if c.get("enabled", True)]
    if requested_camera not in {str(c.get("name")) for c in camera_configs}:
        raise ValueError(f"camera {requested_camera!r} is absent from collector config")

    action_min, action_max, state_min, state_max, meta = load_stats(stats_path)
    return {
        "train_config": train_cfg,
        "collector_config": collector,
        "action_min": action_min,
        "action_max": action_max,
        "state_min": state_min,
        "state_max": state_max,
        "stats_meta": meta,
    }


class EDULITEVLAController:
    """One-arm controller with an independent 100 Hz safety loop."""

    def __init__(self, deployment: dict, checked: dict):
        self.deployment = deployment
        self.collector = checked["collector_config"]
        self.safety = deployment["safety"]
        self.runtime = deployment["runtime"]
        side = str(deployment["hardware"].get("arm_side", "right"))
        self.arm_cfg = self.collector["arms"][side]
        self.arm = _make_arm(self.arm_cfg, self.collector.get("sdk", {}))
        self.gripper = GripperCalibration(self.arm_cfg["gripper"])

        action_min = checked["action_min"][:7]
        action_max = checked["action_max"][:7]
        margin = float(self.safety.get("joint_limit_margin_rad", 0.15))
        mechanical_lower = DEFAULT_LIMITS[:, 0] + margin
        mechanical_upper = DEFAULT_LIMITS[:, 1] - margin
        if bool(self.safety.get("clamp_to_training_range", True)):
            self.lower = np.maximum(mechanical_lower, action_min[:6])
            self.upper = np.minimum(mechanical_upper, action_max[:6])
        else:
            self.lower, self.upper = mechanical_lower, mechanical_upper
        if np.any(self.lower >= self.upper):
            raise ValueError("runtime joint limits are empty after intersecting training ranges")
        self.raw_lower = DEFAULT_LIMITS[:, 0]
        self.raw_upper = DEFAULT_LIMITS[:, 1]
        self.train_lower = action_min[:6]
        self.train_upper = action_max[:6]

        self.feedback_timeout = float(self.safety.get("feedback_timeout_s", 0.2))
        self.startup_timeout = float(self.safety.get("startup_feedback_timeout_s", 2.0))
        self.control_rate = float(self.runtime.get("control_rate_hz", 100.0))
        self.max_target_step = np.broadcast_to(
            np.asarray(self.safety.get("max_target_step_rad", 0.1), dtype=float), (6,)
        ).copy()
        self.follow_error_limit = float(self.safety.get("max_follow_error_rad", 0.3))
        self.follow_error_cycles = int(self.safety.get("follow_error_cycles", 20))
        self.raw_tolerance = float(self.safety.get("raw_joint_limit_tolerance_rad", 0.25))
        self.raw_gripper_tolerance = float(self.safety.get("raw_gripper_tolerance", 0.2))
        self.gripper_filter_alpha = float(
            self.safety.get("gripper_filter_alpha", 0.45)
        )
        self.gripper_max_target_rate = float(
            self.safety.get("gripper_max_target_rate_norm_s", 1.0)
        )
        self.gripper_command_deadband = float(
            self.safety.get("gripper_command_deadband_norm", 0.005)
        )
        if not 0.0 < self.gripper_filter_alpha <= 1.0:
            raise ValueError("safety.gripper_filter_alpha must be in (0, 1]")
        if self.gripper_max_target_rate <= 0.0:
            raise ValueError(
                "safety.gripper_max_target_rate_norm_s must be positive"
            )
        if not 0.0 <= self.gripper_command_deadband <= 0.2:
            raise ValueError(
                "safety.gripper_command_deadband_norm must be in [0, 0.2]"
            )
        action_rate = float(self.runtime.get("action_rate_hz", 30.0))
        if action_rate <= 0.0:
            raise ValueError("runtime.action_rate_hz must be positive")
        # Model actions arrive at a fixed cadence.  Using that cadence instead
        # of wall-clock time prevents a preview/start delay from admitting one
        # unbounded first L7 step.
        self._gripper_max_step = self.gripper_max_target_rate / action_rate

        grip_cfg = self.collector.get("gripper_control", {})
        force = grip_cfg.get("force_stop", {})
        self.gripper_torque_limit = float(grip_cfg.get("motor_torque_limit_nm", 0.10))
        self.gripper_current_limit = float(grip_cfg.get("motor_current_limit_a", 0.30))
        self.force_enabled = bool(force.get("enabled", True))
        self.force_soft = float(force.get("torque_threshold_nm", 0.12))
        self.force_hard = float(force.get("hard_torque_limit_nm", 0.18))
        self.force_alpha = float(force.get("ema_alpha", 0.5))
        self.force_samples_required = int(force.get("consecutive_samples", 3))
        self.force_contact_velocity = float(force.get("max_contact_velocity_rad_s", 0.05))
        self.force_min_error = float(force.get("min_target_error_norm", 0.015))
        self.force_stall_time = float(force.get("stall_time_s", 0.30))
        self.force_stall_velocity = float(force.get("max_stall_velocity_rad_s", 0.01))
        self.force_stall_torque = float(force.get("stall_torque_threshold_nm", 0.07))
        self.force_backoff = float(force.get("backoff_norm", 0.0))
        self.reopen_margin = float(
            self.collector.get("master_gripper_control", {}).get(
                "reopen_latch_margin_norm", 0.05
            )
        )

        self.connected = False
        self.armed = False
        self.fault: str | None = None
        self._filter: CriticallyDampedJointFilter | None = None
        self._target = np.zeros(6, dtype=np.float64)
        self._sent = np.zeros(6, dtype=np.float64)
        self._last_model_command = 0.0
        # Raw, validated model intent is kept separately from the conditioned
        # L7 target so diagnostics can expose action-chunk discontinuities.
        self._gripper_target = self.gripper.value
        self._gripper_filtered_target = self.gripper.value
        self._gripper_sent = self.gripper.value
        self._gripper_latched = False
        self._gripper_hold = self.gripper.value
        self._gripper_ema = 0.0
        self._gripper_samples = 0
        self._stall_since: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._emergency = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        if self.connected:
            return
        if not self.arm.ConnectPort():
            raise RuntimeError(f"cannot connect task arm on {self.arm_cfg['can_name']}")
        self.connected = True

    def _configure_gripper_limits(self) -> None:
        from el_a3_sdk.protocol import ParamIndex

        for parameter, limit, label, unit in (
            (ParamIndex.LIMIT_TORQUE, self.gripper_torque_limit, "torque", "Nm"),
            (ParamIndex.LIMIT_CUR, self.gripper_current_limit, "current", "A"),
        ):
            if not self.arm.WriteMotorParameter(7, parameter, limit):
                raise RuntimeError(f"failed to write L7 {label} limit")
            time.sleep(0.03)
            result = self.arm.ReadMotorParameter(7, parameter)
            tolerance = max(0.01, 0.1 * limit)
            if (
                result is None
                or not result.success
                or abs(float(result.value) - limit) > tolerance
            ):
                raise RuntimeError(f"failed to verify L7 {label} limit")
            print(f"L7 {label} limit verified: {float(result.value):.3f} {unit}")

    def arm_and_home(self) -> None:
        if self.armed:
            return
        if self._emergency.is_set():
            raise RuntimeError(
                "software emergency stop is latched; restart the program before re-arming"
            )
        if not self.connected:
            self.connect()
        if not self.arm.EnableArm():
            raise RuntimeError("task arm failed to enable")
        try:
            self._configure_gripper_limits()
            sdk_rate = float(self.collector.get("sdk", {}).get("control_rate_hz", 200.0))
            initial = _wait_feedback_ready(
                self.arm, "task arm", self.feedback_timeout, self.startup_timeout
            )
            self.gripper.value = self.gripper.normalize(initial[6])
            self._gripper_target = self.gripper.value
            self._gripper_filtered_target = self.gripper.value
            self._gripper_sent = self.gripper.value
            self.arm.start_control_loop(sdk_rate)
            if not self.arm.JointCtrl(*initial[:6].tolist(), velocities=[0.0] * 6):
                raise RuntimeError("could not establish initial joint hold")
            self.armed = True
            self.home()
            feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
            self._target = feedback[:6].copy()
            self._sent = feedback[:6].copy()
            self._filter = CriticallyDampedJointFilter(
                feedback[:6],
                omega=float(self.safety.get("filter_omega", 10.0)),
                max_velocity=self.safety.get("max_velocity_rad_s", 0.35),
                max_acceleration=self.safety.get("max_acceleration_rad_s2", 1.2),
                lower_limits=self.lower,
                upper_limits=self.upper,
            )
            self._last_model_command = time.monotonic()
            self.fault = None
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._control_loop, name="edulite-vla-safety", daemon=True
            )
            self._thread.start()
        except Exception:
            self.emergency_stop()
            raise

    def home(self) -> None:
        if not self.armed:
            raise RuntimeError("arm is not enabled")
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("stop the runtime controller before homing")
        cfg = self.collector.get("episode_reset", {})
        target = np.deg2rad(np.asarray(cfg.get("target_deg", [0, 5, -5, 0, 0, 0]), dtype=float))
        target = np.clip(target, DEFAULT_LIMITS[:, 0] + 0.03, DEFAULT_LIMITS[:, 1] - 0.03)
        initial = _feedback(self.arm, "task arm", self.feedback_timeout)
        filt = CriticallyDampedJointFilter(
            initial[:6],
            omega=6.0,
            max_velocity=float(cfg.get("max_velocity_rad_s", 0.25)),
            max_acceleration=float(cfg.get("max_acceleration_rad_s2", 0.5)),
            lower_limits=DEFAULT_LIMITS[:, 0] + 0.03,
            upper_limits=DEFAULT_LIMITS[:, 1] - 0.03,
        )
        tolerance = math.radians(float(cfg.get("tolerance_deg", 2.0)))
        timeout = float(cfg.get("timeout_s", 20.0))
        deadline = time.monotonic() + timeout
        previous = time.monotonic()
        print(f"HOMING task arm to {np.rad2deg(target).round(2).tolist()} deg")
        while time.monotonic() < deadline:
            if self._emergency.is_set():
                raise RuntimeError("homing interrupted by emergency stop")
            now = time.monotonic()
            state = filt.step(target, max(now - previous, 1e-4))
            previous = now
            if not self.arm.JointCtrl(*state.position.tolist(), velocities=state.velocity.tolist()):
                raise RuntimeError("home joint command failed")
            feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
            if np.max(np.abs(feedback[:6] - target)) <= tolerance:
                self.arm.JointCtrl(*target.tolist(), velocities=[0.0] * 6)
                print("HOME READY: task arm is holding the trained start pose")
                return
            time.sleep(0.01)
        raise RuntimeError("homing timed out")

    def validate_model_action(self, action: np.ndarray) -> tuple[np.ndarray, list[str]]:
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (14,) or not np.isfinite(raw).all():
            raise RuntimeError("model action must be finite with shape (14,)")
        q_raw = raw[:6]
        if np.any(q_raw < self.raw_lower - self.raw_tolerance) or np.any(
            q_raw > self.raw_upper + self.raw_tolerance
        ):
            raise RuntimeError(f"model proposed joints beyond hard limits: {q_raw.tolist()}")
        grip_raw = float(raw[6])
        if grip_raw < -self.raw_gripper_tolerance or grip_raw > 1.0 + self.raw_gripper_tolerance:
            raise RuntimeError(f"model proposed invalid gripper value: {grip_raw:.3f}")

        warnings: list[str] = []
        q = np.clip(q_raw, self.lower, self.upper)
        if not np.allclose(q, q_raw, atol=1e-6):
            warnings.append("joint target clipped to trained/mechanical range")
        grip = float(np.clip(grip_raw, 0.0, 1.0))
        if abs(grip - grip_raw) > 1e-6:
            warnings.append("gripper target clipped to [0,1]")
        return np.r_[q, grip], warnings

    def set_model_action(self, action: np.ndarray) -> list[str]:
        safe, warnings = self.validate_model_action(action)
        with self._lock:
            stepped = np.clip(
                safe[:6],
                self._target - self.max_target_step,
                self._target + self.max_target_step,
            )
            if not np.allclose(stepped, safe[:6], atol=1e-6):
                warnings.append("joint target rate-clipped")
            self._target = stepped
            self._gripper_target = float(safe[6])
            if self._gripper_latched and safe[6] < self._gripper_hold + self.reopen_margin:
                # Do not accumulate a hidden closing command behind a force
                # latch.  Otherwise releasing the latch could briefly command
                # another close before the filter catches up.
                self._gripper_filtered_target = self._gripper_hold
            else:
                ema_target = (
                    self.gripper_filter_alpha * float(safe[6])
                    + (1.0 - self.gripper_filter_alpha)
                    * self._gripper_filtered_target
                )
                delta = float(
                    np.clip(
                        ema_target - self._gripper_filtered_target,
                        -self._gripper_max_step,
                        self._gripper_max_step,
                    )
                )
                self._gripper_filtered_target = float(
                    np.clip(self._gripper_filtered_target + delta, 0.0, 1.0)
                )
            self._last_model_command = time.monotonic()
        return warnings

    def hold_current(self) -> None:
        if not self.armed:
            return
        feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
        with self._lock:
            self._target = np.clip(feedback[:6], self.lower, self.upper)
            self._sent = self._target.copy()
            self._gripper_target = self.gripper.normalize(feedback[6])
            self._gripper_filtered_target = self._gripper_target
            self._gripper_sent = self._gripper_target
            self._last_model_command = time.monotonic()
            if self._filter is not None:
                self._filter.reset(self._target)
        self.arm.JointCtrl(*self._target.tolist(), velocities=[0.0] * 6)

    def return_home(self) -> None:
        """Pause the runtime loop, home L1-L6, then resume safe holding.

        L7 deliberately keeps its current opening during this transition.  A
        new model preview is required by the UI before the next execution.
        """

        if not self.armed:
            raise RuntimeError("arm is not enabled")
        if self._emergency.is_set():
            raise RuntimeError(
                "software emergency stop is latched; restart before homing"
            )
        if self.fault:
            raise RuntimeError(f"cannot home after controller fault: {self.fault}")

        # Give the homing routine exclusive ownership of JointCtrl.  This is a
        # normal pause, not an emergency stop, so the motors remain enabled.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("runtime control thread did not stop before homing")
        self._thread = None

        try:
            self.home()
            feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
            with self._lock:
                self._target = np.clip(feedback[:6], self.lower, self.upper)
                self._sent = self._target.copy()
                self._gripper_target = self.gripper.normalize(feedback[6])
                self._gripper_filtered_target = self._gripper_target
                self._gripper_sent = self._gripper_target
                self._last_model_command = time.monotonic()
                self._filter = CriticallyDampedJointFilter(
                    feedback[:6],
                    omega=float(self.safety.get("filter_omega", 10.0)),
                    max_velocity=self.safety.get("max_velocity_rad_s", 0.35),
                    max_acceleration=self.safety.get("max_acceleration_rad_s2", 1.2),
                    lower_limits=self.lower,
                    upper_limits=self.upper,
                )
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._control_loop,
                name="edulite-vla-safety",
                daemon=True,
            )
            self._thread.start()
            print(
                "NEXT EPISODE HOME READY: L1-L6 are holding the trained start pose; "
                "press p for a new preview",
                flush=True,
            )
        except Exception:
            self.emergency_stop()
            raise

    def telemetry(self) -> dict[str, Any]:
        """Snapshot model target, filtered command and physical feedback."""

        feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
        efforts = self.arm.GetArmJointEfforts()
        with self._lock:
            return {
                "model_target": np.r_[self._target, self._gripper_target].astype(np.float32),
                "sent": np.r_[self._sent, self._gripper_sent].astype(np.float32),
                "feedback": np.r_[
                    feedback[:6], self.gripper.normalize(feedback[6])
                ].astype(np.float32),
                "gripper_torque_raw": float(efforts.joint_7),
                "gripper_torque_ema": float(self._gripper_ema),
                "gripper_force_latched": bool(self._gripper_latched),
            }

    def predicted_xyz(self, actions7: np.ndarray) -> np.ndarray:
        """Apply SDK FK to a safe action chunk for visualization only."""

        kin = self.arm._get_kinematics()
        if kin is None:
            raise RuntimeError("Pinocchio FK is unavailable for trajectory visualization")
        xyz = []
        for action in np.asarray(actions7):
            pose = kin.forward_kinematics(action[:6].tolist())
            xyz.append([pose.x, pose.y, pose.z])
        return np.asarray(xyz, dtype=np.float32)

    def observation(self, rgb: np.ndarray) -> dict[str, Any]:
        feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
        left_pose = _pose(self.arm, self.arm_cfg)
        left_gripper = self.gripper.normalize(feedback[6])
        return {
            "endpose": {
                "left_endpose": left_pose,
                "left_gripper": left_gripper,
                "right_endpose": np.zeros(7, dtype=np.float32),
                "right_gripper": 0.0,
            },
            "observation": {
                self.deployment["hardware"]["camera_name"]: {"rgb": rgb}
            },
        }

    def _update_gripper(self, feedback: np.ndarray, now: float) -> None:
        actual_norm = self.gripper.normalize(feedback[6])
        efforts = self.arm.GetArmJointEfforts()
        velocities = self.arm.GetArmJointVelocities()
        raw_torque = abs(float(efforts.joint_7))
        velocity = abs(float(velocities.joint_7))
        with self._lock:
            raw_requested = self._gripper_target
            requested = self._gripper_filtered_target
            self._gripper_ema = (
                self.force_alpha * raw_torque
                + (1.0 - self.force_alpha) * self._gripper_ema
            )
            if self._gripper_latched:
                if raw_requested >= self._gripper_hold + self.reopen_margin:
                    self._gripper_latched = False
                    self._gripper_samples = 0
                    self._stall_since = None
                    print("GRIPPER FORCE LATCH RELEASED by an opening model command")
                else:
                    requested = self._gripper_hold

            closing = requested < actual_norm - self.force_min_error
            stalled_contact = velocity <= self.force_contact_velocity
            if self.force_enabled and closing and stalled_contact and self._gripper_ema >= self.force_soft:
                self._gripper_samples += 1
            else:
                self._gripper_samples = 0
            hard = self.force_enabled and closing and raw_torque >= self.force_hard
            soft = self._gripper_samples >= self.force_samples_required
            stalled = (
                self.force_enabled
                and closing
                and velocity <= self.force_stall_velocity
                and self._gripper_ema >= self.force_stall_torque
            )
            if stalled:
                self._stall_since = self._stall_since or now
            else:
                self._stall_since = None
            stall = self._stall_since is not None and now - self._stall_since >= self.force_stall_time
            trigger = hard or soft or stall
            if trigger and not self._gripper_latched:
                self._gripper_hold = min(1.0, actual_norm + self.force_backoff)
                self._gripper_target = self._gripper_hold
                self._gripper_filtered_target = self._gripper_hold
                requested = self._gripper_hold
                self._gripper_latched = True
                reason = "HARD" if hard else ("SOFT" if soft else "STALL")
                print(
                    f"GRIPPER FORCE STOP ({reason}): raw={raw_torque:.3f} Nm, "
                    f"ema={self._gripper_ema:.3f} Nm, velocity={velocity:.3f} rad/s, "
                    f"hold_norm={self._gripper_hold:.3f}; closing LATCHED",
                    flush=True,
                )
            should_send = (
                abs(requested - self._gripper_sent)
                >= self.gripper_command_deadband
            )
            if should_send:
                self.gripper.value = float(np.clip(requested, 0.0, 1.0))
                angle = self.gripper.radians()
                self._gripper_sent = self.gripper.value
        if should_send and not self.arm.GripperCtrl(angle):
            raise RuntimeError("gripper command failed")

    def _control_loop(self) -> None:
        period = 1.0 / self.control_rate
        deadline = time.monotonic()
        previous = deadline
        bad_cycles = 0
        try:
            while not self._stop.is_set() and not self._emergency.is_set():
                now = time.monotonic()
                feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
                with self._lock:
                    target = self._target.copy()
                    last_command = self._last_model_command
                    filt = self._filter
                if filt is None:
                    raise RuntimeError("runtime joint filter is not initialized")
                # Stale inference never extrapolates; it simply holds target.
                if now - last_command > float(self.safety.get("model_command_timeout_s", 1.0)):
                    target = filt.state.position.copy()
                state = filt.step(target, max(now - previous, 1e-4))
                previous = now
                error = float(np.max(np.abs(state.position - feedback[:6])))
                bad_cycles = bad_cycles + 1 if error > self.follow_error_limit else 0
                if bad_cycles >= self.follow_error_cycles:
                    raise RuntimeError(
                        f"following error {error:.3f} rad exceeds {self.follow_error_limit:.3f} rad"
                    )
                if not self.arm.JointCtrl(
                    *state.position.tolist(), velocities=state.velocity.tolist()
                ):
                    raise RuntimeError("joint command failed")
                with self._lock:
                    self._sent = state.position.copy()
                self._update_gripper(feedback, now)
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
                if time.monotonic() - deadline > period:
                    deadline = time.monotonic()
        except Exception as exc:
            self.fault = str(exc)
            print(f"\nVLA SAFETY STOP: {exc}", file=sys.stderr, flush=True)
            self.emergency_stop()

    def check_health(self) -> None:
        if self.fault:
            raise RuntimeError(f"VLA safety stop: {self.fault}")
        if self._emergency.is_set():
            raise RuntimeError("emergency stop is active")

    def emergency_stop(self) -> None:
        self._emergency.set()
        self._stop.set()
        self.armed = False
        try:
            self.arm.EmergencyStop()
        except Exception:
            pass

    def close(self) -> None:
        self.emergency_stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if self.connected:
            try:
                self.arm.stop_control_loop()
            except Exception:
                pass
            try:
                self.arm.DisconnectPort()
            except Exception:
                pass
            self.connected = False


@contextmanager
def raw_terminal(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class KeyReader:
    """Read keys independently so `e` remains responsive during GPU inference."""

    def __init__(self, emergency_callback):
        self.keys: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.stop = threading.Event()
        self.emergency_callback = emergency_callback
        self.thread = threading.Thread(target=self._run, name="vla-keys", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1)
            if key == "e":
                self.emergency_callback()
            self.keys.put(key)

    def get(self) -> str | None:
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.5)


class AsyncActionChunks:
    """Prefetch model chunks without ever bursting delayed actions."""

    def __init__(self, agent, controller: EDULITEVLAController, instruction: str, config: dict):
        self.agent = agent
        self.controller = controller
        self.instruction = instruction
        self.horizon = int(agent.action_execution_horizon)
        self.prefetch_threshold = int(
            config["runtime"].get("prefetch_threshold_actions", self.horizon // 2)
        )
        if not 0 <= self.prefetch_threshold < self.horizon:
            raise ValueError("prefetch_threshold_actions must be in [0, horizon)")
        self.actions: deque[np.ndarray] = deque()
        self.pending: np.ndarray | None = None
        self.future: Future | None = None
        self.future_generation: int | None = None
        self.generation = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-prefetch")

    def _infer(self, observation: dict[str, Any]) -> np.ndarray:
        # static_check enforces state_indices=[0], so every chunk needs only
        # this observation and has no cross-thread history dependency.
        self.agent.reset()
        self.agent.processor.update_state_buffer(observation)
        return self.agent._predict_chunk(observation, self.instruction)

    def _validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        value = np.asarray(chunk, dtype=np.float32)
        expected = (int(self.agent.config.common.action_chunk_size), 14)
        if value.shape != expected or not np.isfinite(value).all():
            raise RuntimeError(
                f"invalid model chunk: expected {expected}, got {value.shape}"
            )
        # Reject a bad future action before admitting any part of the chunk.
        for action in value[: self.horizon]:
            self.controller.validate_model_action(action)
        return value

    def preview(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        self.wait_idle()
        started = time.perf_counter()
        chunk = self._validate_chunk(self._infer(observation))
        return chunk, time.perf_counter() - started

    def start(self, preview_chunk: np.ndarray) -> None:
        self.stop()
        chunk = self._validate_chunk(preview_chunk)
        self.actions.extend(chunk[: self.horizon].copy())

    def _submit(self, observation: dict[str, Any]) -> None:
        if self.future is not None or self.pending is not None:
            return
        generation = self.generation
        self.future_generation = generation
        self.future = self.executor.submit(self._infer, observation)

    def _poll(self) -> None:
        if self.future is None or not self.future.done():
            return
        future = self.future
        generation = self.future_generation
        self.future = None
        self.future_generation = None
        result = future.result()
        if generation != self.generation:
            return
        chunk = self._validate_chunk(result)
        self.pending = chunk[: self.horizon].copy()

    def next_action(self, observation: dict[str, Any]) -> np.ndarray | None:
        self._poll()
        if not self.actions and self.pending is not None:
            self.actions.extend(self.pending)
            self.pending = None
        if not self.actions:
            self._submit(observation)
            return None
        action = self.actions.popleft()
        if (
            len(self.actions) <= self.prefetch_threshold
            and self.pending is None
            and self.future is None
        ):
            self._submit(observation)
        return action

    def stop(self) -> None:
        self.generation += 1
        self.actions.clear()
        self.pending = None
        if self.future is not None and self.future.cancel():
            self.future = None
            self.future_generation = None

    def wait_idle(self) -> None:
        if self.future is not None:
            # Preview is only allowed while the arm is holding. A running
            # inference is allowed to finish, then its stale result is dropped.
            self.future.result()
            self.future = None
            self.future_generation = None
            self.pending = None

    def close(self) -> None:
        self.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)


def build_agent(config: dict):
    import torch
    from robotwin_infer import RobotWinInference

    dtype_name = str(config["model"].get("dtype", "bfloat16"))
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    device = str(config["model"].get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but no CUDA GPU is visible")
    return RobotWinInference(
        config_path=config["model"]["config_path"],
        checkpoint_path=config["model"]["checkpoint_path"],
        norm_stats_path=config["model"]["norm_stats_path"],
        two_stage_mode=bool(config["model"].get("two_stage_mode", True)),
        device=device,
        dtype=dtypes[dtype_name],
        smooth_actions=bool(config["model"].get("smooth_actions", True)),
        smooth_sigma=float(config["model"].get("smooth_sigma", 1.0)),
    )


def print_static_report(config: dict, checked: dict) -> None:
    meta = checked["stats_meta"]
    print("STATIC CHECK PASSED")
    print(f"  checkpoint: {config['model']['checkpoint_path']}")
    print(f"  stats files: {meta.get('valid_files', 'unknown')}/{meta.get('total_files', 'unknown')}")
    print("  model schema: action=14, state=16, camera=head_camera")
    print("  deployed action: action[0:6] joints + action[6] gripper")
    print("  ignored action: action[7:14] synthetic right-arm padding")
    print(
        "  trained left action range: "
        f"min={checked['action_min'][:7].round(4).tolist()}, "
        f"max={checked['action_max'][:7].round(4).tolist()}"
    )


def run_execute(config: dict, checked: dict) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("--execute requires an interactive terminal")

    print("Loading VLA model before enabling any motor...")
    agent = build_agent(config)
    collector = checked["collector_config"]
    cameras = build_cameras(collector.get("cameras", []), dry_run=False)
    camera_name = config["hardware"]["camera_name"]
    camera = cameras[camera_name]
    controller = EDULITEVLAController(config, checked)
    visualizer = InferenceVisualizer(config, controller)
    instruction = str(config["model"]["instruction"])
    policy = AsyncActionChunks(agent, controller, instruction, config)
    camera_max_age = float(config["runtime"].get("camera_max_age_s", 0.25))
    action_period = 1.0 / float(config["runtime"].get("action_rate_hz", 30.0))
    max_episode = float(config["runtime"].get("max_episode_s", 40.0))
    start_delay = float(config["runtime"].get("start_delay_s", 3.0))
    require_preview = bool(config["runtime"].get("require_preview_before_execute", True))

    executing = False
    preview_ok = False
    episode_start = 0.0
    next_action = 0.0
    preview_chunk: np.ndarray | None = None
    stop_requested = threading.Event()

    def emergency() -> None:
        nonlocal executing
        executing = False
        controller.emergency_stop()
        print("\nEMERGENCY STOP: task-arm motors disabled", flush=True)

    def signal_stop(_signum, _frame) -> None:
        emergency()
        stop_requested.set()

    signal.signal(signal.SIGINT, signal_stop)
    signal.signal(signal.SIGTERM, signal_stop)

    key_reader: KeyReader | None = None
    try:
        camera.start()
        # Wait for a real frame before opening CAN.
        deadline = time.monotonic() + 3.0
        while True:
            try:
                frame = camera.latest(camera_max_age)
                print(f"Camera ready: RGB shape={tuple(frame.rgb.shape)}")
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        controller.connect()  # Connection alone never enables motors.
        print(f"Connected to task arm: {controller.arm_cfg['can_name']} (motors disabled)")
        print("Controls: a=enable+home, p=preview one model chunk, r=execute, s=hold, h=home")
        print("          e=E-STOP (independent key thread), q=quit")
        print("Keep the physical emergency-stop reachable at all times.")

        with raw_terminal(True):
            key_reader = KeyReader(emergency)
            key_reader.start()
            while not stop_requested.is_set():
                if controller.armed:
                    controller.check_health()
                key = key_reader.get()
                if key == "a":
                    if controller.armed:
                        print("Already armed")
                    else:
                        controller.arm_and_home()
                        agent.reset()
                        preview_ok = False
                        print("ARMED/HOME READY; press p to preview model output")
                elif key == "p":
                    if not controller.armed or executing:
                        print("Preview requires an armed, non-executing arm")
                    else:
                        frame = camera.latest(camera_max_age)
                        observation = controller.observation(frame.rgb)
                        chunk, inference_s = policy.preview(observation)
                        warning_count = 0
                        for action in chunk:
                            _, warnings = controller.validate_model_action(action)
                            warning_count += len(warnings)
                        print(
                            "PREVIEW PASSED: "
                            f"left min={chunk[:, :7].min(0).round(3).tolist()}, "
                            f"left max={chunk[:, :7].max(0).round(3).tolist()}, "
                            f"clip_warnings={warning_count}, "
                            f"inference={inference_s:.3f}s"
                        )
                        preview_chunk = chunk.copy()
                        preview_ok = True
                        visualizer.save_preview(frame.rgb, chunk)
                elif key == "r":
                    if not controller.armed:
                        print("Press a before execution")
                    elif executing:
                        print("Already executing")
                    elif require_preview and not preview_ok:
                        print("Press p and pass model preview before execution")
                    else:
                        print(f"Execution starts in {start_delay:.1f}s; press e to abort")
                        end = time.monotonic() + start_delay
                        while time.monotonic() < end:
                            if not controller.armed:
                                break
                            time.sleep(0.02)
                        if controller.armed:
                            if preview_chunk is None:
                                raise RuntimeError("preview chunk is unexpectedly missing")
                            policy.start(preview_chunk)
                            # A preview is single-use. After any motion, a new
                            # execution requires a new observation and preview.
                            preview_chunk = None
                            preview_ok = False
                            start_frame = camera.latest(camera_max_age)
                            visualizer.start_episode(start_frame.rgb)
                            executing = True
                            episode_start = time.monotonic()
                            next_action = episode_start
                            print(f"VLA EXECUTION ACTIVE: {instruction}")
                elif key == "s":
                    executing = False
                    policy.stop()
                    if controller.armed:
                        controller.hold_current()
                        visualizer.finalize("operator_stop")
                        print("EXECUTION STOPPED: holding current pose")
                elif key == "h":
                    if executing:
                        print("Press s before homing")
                    elif not controller.armed:
                        print("Press a before homing")
                    else:
                        policy.stop()
                        visualizer.finalize("home_requested")
                        controller.return_home()
                        preview_ok = False
                        preview_chunk = None
                elif key == "e":
                    executing = False
                    visualizer.finalize("emergency_stop")
                elif key == "q":
                    visualizer.finalize("quit")
                    break

                if executing and controller.armed:
                    now = time.monotonic()
                    if now - episode_start >= max_episode:
                        executing = False
                        policy.stop()
                        controller.hold_current()
                        visualizer.finalize("max_episode_time")
                        print(f"MAX EPISODE TIME {max_episode:.1f}s reached; holding pose")
                    elif now >= next_action:
                        frame = camera.latest(camera_max_age)
                        observation = controller.observation(frame.rgb)
                        action = policy.next_action(observation)
                        if action is not None:
                            warnings = controller.set_model_action(action)
                            if warnings:
                                print("ACTION SAFETY: " + "; ".join(sorted(set(warnings))))
                        visualizer.record(
                            frame.rgb,
                            action,
                            controller.telemetry(),
                            np.asarray(observation["endpose"]["left_endpose"][:3]),
                        )
                        next_action += action_period
                        if time.monotonic() - next_action > action_period:
                            # Never catch up by bursting old actions.
                            next_action = time.monotonic() + action_period
                time.sleep(0.002 if executing else 0.01)
        return 0
    finally:
        if key_reader is not None:
            key_reader.close()
        visualizer.finalize("program_exit")
        policy.close()
        controller.close()
        camera.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.real_vla.yaml"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate paths/schema/stats only; never load GPU, camera, or CAN",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow the interactive program to enable and move the physical task arm",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only and args.execute:
        raise ValueError("choose either --check-only or --execute")
    if not args.check_only and not args.execute:
        raise ValueError("no operation selected; use --check-only first, then --execute")
    config, _ = load_deployment_config(args.config)
    checked = static_check(config)
    print_static_report(config, checked)
    if args.check_only:
        return 0
    return run_execute(config, checked)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
