"""EDULITE-A3 rigs for bimanual hand-guiding and one-arm master/slave teleop."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from filters import CriticallyDampedJointFilter


SDK_ROOT = Path(__file__).resolve().parents[1] / "EDULITE_A3" / "el_a3_sdk"
DEFAULT_LIMITS = np.asarray(
    [
        [-2.79253, 2.79253],
        [0.0, 3.66519],
        [-4.01426, 0.0],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RigSnapshot:
    action: np.ndarray
    left_pose: np.ndarray
    left_gripper: float
    right_pose: np.ndarray | None
    right_gripper: float | None
    slave_feedback: np.ndarray
    master_feedback: np.ndarray | None


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Numerically stable 3x3 rotation to normalized [w,x,y,z]."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = [(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s]
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = [(rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = [(rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s]
    result = np.asarray(quat, dtype=np.float64)
    return result / np.linalg.norm(result)


def world_pose(local_xyzrpy: np.ndarray, arm_cfg: dict) -> np.ndarray:
    base_xyz = np.asarray(arm_cfg.get("base_xyz", [0, 0, 0]), dtype=np.float64)
    base_rpy = np.asarray(arm_cfg.get("base_rpy", [0, 0, 0]), dtype=np.float64)
    if base_xyz.shape != (3,) or base_rpy.shape != (3,):
        raise ValueError("base_xyz and base_rpy must each contain three numbers")
    base_rotation = _rotation_from_rpy(base_rpy)
    local_rotation = _rotation_from_rpy(local_xyzrpy[3:6])
    xyz = base_xyz + base_rotation @ local_xyzrpy[:3]
    quaternion = _quaternion_wxyz(base_rotation @ local_rotation)
    return np.concatenate([xyz, quaternion]).astype(np.float32)


class GripperCalibration:
    def __init__(self, cfg: dict):
        self.closed = float(cfg["closed_rad"])
        self.open = float(cfg["open_rad"])
        if abs(self.open - self.closed) < 1e-6:
            raise ValueError("gripper open_rad and closed_rad must differ")
        self.value = float(np.clip(cfg.get("initial_norm", 1.0), 0.0, 1.0))

    def normalize(self, angle: float) -> float:
        return float(np.clip((angle - self.closed) / (self.open - self.closed), 0.0, 1.0))

    def radians(self) -> float:
        return self.closed + self.value * (self.open - self.closed)

    def adjust(self, delta: float) -> float:
        self.value = float(np.clip(self.value + delta, 0.0, 1.0))
        return self.radians()


def _import_sdk():
    if str(SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT))
    from el_a3_sdk import ELA3Interface, LogLevel
    return ELA3Interface, LogLevel


def _make_arm(arm_cfg: dict, sdk_cfg: dict):
    interface, log_level = _import_sdk()
    inertia = sdk_cfg.get("inertia_config_path")
    if inertia and not Path(inertia).is_absolute():
        inertia = str(SDK_ROOT / inertia)
    return interface(
        can_name=str(arm_cfg["can_name"]),
        backend=str(arm_cfg.get("backend", "socketcan")),
        serial_port=arm_cfg.get("serial_port"),
        default_kp=float(sdk_cfg.get("position_kp", 50.0)),
        default_kd=float(sdk_cfg.get("position_kd", 3.0)),
        control_rate_hz=float(sdk_cfg.get("control_rate_hz", 200.0)),
        smoothing_alpha=float(sdk_cfg.get("smoothing_alpha", 0.35)),
        max_velocity=float(sdk_cfg.get("max_velocity", 1.0)),
        max_acceleration=float(sdk_cfg.get("max_acceleration", 5.0)),
        gravity_feedforward_ratio=float(sdk_cfg.get("gravity_feedforward_ratio", 1.0)),
        limit_margin=float(sdk_cfg.get("limit_margin", 0.15)),
        limit_stop_margin=float(sdk_cfg.get("limit_stop_margin", 0.03)),
        pp_velocity=float(sdk_cfg.get("gripper_velocity", 0.8)),
        pp_acceleration=float(sdk_cfg.get("gripper_acceleration", 2.0)),
        inertia_config_path=inertia,
        logger_level=getattr(log_level, str(sdk_cfg.get("log_level", "WARNING")).upper()),
    )


def _feedback(
    arm,
    name: str,
    timeout: float,
    allowed_disabled_joints: set[int] | None = None,
    required_disabled_joints: set[int] | None = None,
) -> np.ndarray:
    msg = arm.GetArmJointMsgs()
    values = np.asarray(msg.to_list(), dtype=np.float64)
    status = arm.GetArmStatus()
    allowed_disabled = allowed_disabled_joints or set()
    required_disabled = required_disabled_joints or set()
    age = time.time() - float(msg.timestamp)
    if msg.timestamp <= 0 or age > timeout:
        raise RuntimeError(f"{name} feedback stale/missing (age={age:.3f}s)")
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError(f"{name} feedback is invalid")
    if status.has_fault:
        raise RuntimeError(f"{name} motor fault codes: {status.joint_faults}")
    unexpected_disabled = [
        index
        for index, enabled in enumerate(status.joint_enabled, start=1)
        if not enabled and index not in allowed_disabled
    ]
    if unexpected_disabled:
        raise RuntimeError(
            f"{name} has unexpectedly disabled motors {unexpected_disabled}: "
            f"{status.joint_enabled}"
        )
    unexpectedly_enabled = [
        index
        for index in required_disabled
        if index <= len(status.joint_enabled) and status.joint_enabled[index - 1]
    ]
    if unexpectedly_enabled:
        raise RuntimeError(
            f"{name} motors must remain disabled but are enabled: "
            f"{unexpectedly_enabled}"
        )
    return values


def _wait_feedback_ready(
    arm,
    name: str,
    feedback_timeout: float,
    wait_timeout: float,
    allowed_disabled_joints: set[int] | None = None,
    required_disabled_joints: set[int] | None = None,
) -> np.ndarray:
    """Wait through mode-switch transients for fresh, fully-enabled feedback."""
    deadline = time.monotonic() + wait_timeout
    last_error = "no feedback attempt"
    while time.monotonic() < deadline:
        try:
            return _feedback(
                arm,
                name,
                feedback_timeout,
                allowed_disabled_joints=allowed_disabled_joints,
                required_disabled_joints=required_disabled_joints,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(0.02)
    raise RuntimeError(
        f"{name} did not become ready within {wait_timeout:.2f}s; "
        f"last check: {last_error}"
    )


def _pose(arm, cfg: dict) -> np.ndarray:
    pose = arm.GetArmEndPoseMsgs()
    local = np.asarray([pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz], dtype=np.float64)
    if not np.isfinite(local).all():
        raise RuntimeError("forward kinematics returned NaN/Inf")
    if np.allclose(local, 0.0):
        raise RuntimeError(
            "forward kinematics is unavailable or returned an all-zero pose; "
            "install Pinocchio and verify the EDULITE URDF"
        )
    return world_pose(local, cfg)


class DryRunRig:
    def __init__(self, mode: str):
        self.mode = mode
        self.start_time = time.monotonic()
        self.armed = False
        self.teleoperation_active = False
        self.left_gripper = 1.0
        self.right_gripper = 1.0

    def connect(self) -> None:
        return

    def arm(self) -> None:
        self.armed = True
        self.teleoperation_active = self.mode == "bimanual_kinesthetic"
        self.start_time = time.monotonic()

    def start_teleoperation(self) -> None:
        if not self.armed:
            raise RuntimeError("dry-run rig is not armed")
        self.teleoperation_active = True

    def reset_to_home(self) -> None:
        if not self.armed:
            raise RuntimeError("dry-run rig is not armed")
        self.teleoperation_active = False
        self.start_time = time.monotonic()

    def adjust_gripper(self, side: str, delta: float) -> None:
        attr = "right_gripper" if side == "right" else "left_gripper"
        setattr(self, attr, float(np.clip(getattr(self, attr) + delta, 0.0, 1.0)))

    def snapshot(self) -> RigSnapshot:
        if not self.armed:
            raise RuntimeError("dry-run rig is not armed")
        t = time.monotonic() - self.start_time
        left = 0.2 * np.sin(t + np.arange(6) * 0.4)
        left_pose = np.asarray([0.35, 0.18, 0.25 + 0.02 * math.sin(t), 1, 0, 0, 0], np.float32)
        left7 = np.r_[left, self.left_gripper].astype(np.float32)
        if self.mode == "bimanual_kinesthetic":
            right = 0.2 * np.sin(t + 0.5 + np.arange(6) * 0.4)
            right7 = np.r_[right, self.right_gripper].astype(np.float32)
            right_pose = np.asarray([0.35, -0.18, 0.25 + 0.02 * math.cos(t), 1, 0, 0, 0], np.float32)
            return RigSnapshot(np.r_[left7, right7], left_pose, self.left_gripper,
                               right_pose, self.right_gripper, np.r_[left7, right7], None)
        return RigSnapshot(left7, left_pose, self.left_gripper, None, None, left7, left7.copy())

    def check_health(self) -> None:
        return

    def gripper_status(self) -> str:
        return (
            f"dry-run gripper: left_norm={self.left_gripper:.3f}, "
            f"right_norm={self.right_gripper:.3f}"
        )

    def emergency_stop(self) -> None:
        self.armed = False
        self.teleoperation_active = False

    def close(self) -> None:
        self.armed = False
        self.teleoperation_active = False


class BimanualKinestheticRig:
    """Both physical arms are gravity-compensated and guided by hand."""

    def __init__(self, config: dict):
        self.config = config
        self.left_cfg = config["arms"]["left"]
        self.right_cfg = config["arms"]["right"]
        self.left_gripper = GripperCalibration(self.left_cfg["gripper"])
        self.right_gripper = GripperCalibration(self.right_cfg["gripper"])
        self.left = _make_arm(self.left_cfg, config.get("sdk", {}))
        self.right = _make_arm(self.right_cfg, config.get("sdk", {}))
        self.timeout = float(config.get("safety", {}).get("feedback_timeout_s", 0.2))
        self.startup_timeout = float(
            config.get("safety", {}).get("startup_feedback_timeout_s", 2.0)
        )
        self.armed = False

    def connect(self) -> None:
        if not self.left.ConnectPort():
            raise RuntimeError("cannot connect left arm")
        if not self.right.ConnectPort():
            self.left.DisconnectPort()
            raise RuntimeError("cannot connect right arm")

    def arm(self) -> None:
        if not self.left.EnableArm() or not self.right.EnableArm():
            self.emergency_stop()
            raise RuntimeError("one or more arms failed to enable")
        rate = float(self.config.get("sdk", {}).get("control_rate_hz", 200.0))
        left_feedback = _feedback(self.left, "left", self.startup_timeout)
        right_feedback = _feedback(self.right, "right", self.startup_timeout)
        # Engage without moving either gripper.  The configured endpoints are
        # calibration values, not a startup target.
        self.left_gripper.value = self.left_gripper.normalize(left_feedback[6])
        self.right_gripper.value = self.right_gripper.normalize(right_feedback[6])
        kd = float(self.config.get("kinesthetic", {}).get("zero_torque_kd", 0.08))
        # Switch motor modes before starting the 200 Hz loops.  Otherwise the
        # loops can send Type-1 commands while ZeroTorqueMode is concurrently
        # disabling/reconfiguring individual motors.
        if not self.left.ZeroTorqueMode(True, kd=kd):
            raise RuntimeError("left arm could not enter zero-torque mode")
        if not self.right.ZeroTorqueMode(True, kd=kd):
            self.emergency_stop()
            raise RuntimeError("right arm could not enter zero-torque mode")
        self.left.start_control_loop(rate)
        self.right.start_control_loop(rate)
        _wait_feedback_ready(
            self.left, "left", self.timeout, self.startup_timeout
        )
        _wait_feedback_ready(
            self.right, "right", self.timeout, self.startup_timeout
        )
        self.armed = True

    def adjust_gripper(self, side: str, delta: float) -> None:
        gripper, arm = ((self.right_gripper, self.right) if side == "right"
                        else (self.left_gripper, self.left))
        if not arm.GripperCtrl(gripper.adjust(delta)):
            raise RuntimeError(f"{side} gripper command failed")

    def snapshot(self) -> RigSnapshot:
        if not self.armed:
            raise RuntimeError("rig is not armed")
        left = _feedback(self.left, "left", self.timeout)
        right = _feedback(self.right, "right", self.timeout)
        lg = self.left_gripper.normalize(left[6])
        rg = self.right_gripper.normalize(right[6])
        left7, right7 = np.r_[left[:6], lg], np.r_[right[:6], rg]
        return RigSnapshot(np.r_[left7, right7].astype(np.float32), _pose(self.left, self.left_cfg),
                           lg, _pose(self.right, self.right_cfg), rg,
                           np.r_[left7, right7].astype(np.float32), None)

    def check_health(self) -> None:
        if not self.armed:
            return
        _feedback(self.left, "left", self.timeout)
        _feedback(self.right, "right", self.timeout)

    def gripper_status(self) -> str:
        left = self.left.GetArmJointMsgs()
        right = self.right.GetArmJointMsgs()
        left_effort = self.left.GetArmJointEfforts()
        right_effort = self.right.GetArmJointEfforts()
        return (
            f"left L7: angle={math.degrees(left.joint_7):.2f} deg, "
            f"torque={left_effort.joint_7:.3f} Nm; "
            f"right L7: angle={math.degrees(right.joint_7):.2f} deg, "
            f"torque={right_effort.joint_7:.3f} Nm"
        )

    def emergency_stop(self) -> None:
        self.armed = False
        for arm in (self.left, self.right):
            try:
                arm.EmergencyStop()
            except Exception:
                pass

    def close(self) -> None:
        self.armed = False
        for arm in (self.left, self.right):
            try:
                arm.EmergencyStop()
                arm.DisconnectPort()
            except Exception:
                pass


class MasterSlaveSingleRig:
    """One master arm guides one follower; output is intentionally single-arm 7D."""

    def __init__(self, config: dict):
        self.config = config
        self.master_cfg = config["arms"]["left"]
        self.slave_cfg = config["arms"]["right"]
        self.master = _make_arm(self.master_cfg, config.get("sdk", {}))
        self.slave = _make_arm(self.slave_cfg, config.get("sdk", {}))
        self.master_gripper = GripperCalibration(self.master_cfg["gripper"])
        self.gripper = GripperCalibration(self.slave_cfg["gripper"])
        self.timeout = float(config.get("safety", {}).get("feedback_timeout_s", 0.2))
        self.startup_timeout = float(
            config.get("safety", {}).get("startup_feedback_timeout_s", 2.0)
        )
        self.teleop = config.get("master_slave", {})
        self.episode_reset = config.get("episode_reset", {})
        self._home_enabled = bool(self.episode_reset.get("enabled", False))
        self._home_on_arm = bool(self.episode_reset.get("home_on_arm", True))
        self._home_target = np.deg2rad(
            np.asarray(
                self.episode_reset.get(
                    "target_deg", [0.0, 5.0, -5.0, 0.0, 0.0, 0.0]
                ),
                dtype=np.float64,
            )
        )
        self._home_velocity = float(
            self.episode_reset.get("max_velocity_rad_s", 0.25)
        )
        self._home_acceleration = float(
            self.episode_reset.get("max_acceleration_rad_s2", 0.50)
        )
        self._home_tolerance = math.radians(
            float(self.episode_reset.get("tolerance_deg", 2.0))
        )
        self._home_timeout = float(self.episode_reset.get("timeout_s", 20.0))
        if self._home_target.shape != (6,) or not np.isfinite(self._home_target).all():
            raise ValueError("episode_reset.target_deg must contain six finite values")
        if np.any(self._home_target < DEFAULT_LIMITS[:, 0]) or np.any(
            self._home_target > DEFAULT_LIMITS[:, 1]
        ):
            raise ValueError("episode_reset.target_deg is outside EDULITE joint limits")
        if self._home_velocity <= 0 or self._home_acceleration <= 0:
            raise ValueError("episode_reset velocity and acceleration must be positive")
        if self._home_tolerance <= 0 or self._home_timeout <= 0:
            raise ValueError("episode_reset tolerance and timeout must be positive")
        self.gripper_control = config.get("gripper_control", {})
        master_gripper_cfg = config.get("master_gripper_control", {})
        self._master_gripper_enabled = bool(
            master_gripper_cfg.get("enabled", False)
        )
        self._master_gripper_rate = float(
            master_gripper_cfg.get("command_rate_hz", 50.0)
        )
        self._master_gripper_feedback_refresh_rate = float(
            master_gripper_cfg.get("feedback_refresh_rate_hz", 20.0)
        )
        self._master_gripper_feedback_refresh_timeout = float(
            master_gripper_cfg.get("feedback_refresh_timeout_s", 0.04)
        )
        self._master_gripper_position_margin = float(
            master_gripper_cfg.get("position_limit_margin_rad", 0.15)
        )
        self._master_gripper_teaching_kd = float(
            master_gripper_cfg.get("teaching_kd", 0.05)
        )
        self._master_gripper_torque_limit = float(
            master_gripper_cfg.get("motor_torque_limit_nm", 0.10)
        )
        self._master_gripper_current_limit = float(
            master_gripper_cfg.get("motor_current_limit_a", 0.30)
        )
        self._master_gripper_filter_alpha = float(
            master_gripper_cfg.get("filter_alpha", 0.35)
        )
        self._master_gripper_deadband = float(
            master_gripper_cfg.get("deadband_norm", 0.005)
        )
        self._master_gripper_max_rate = float(
            master_gripper_cfg.get("max_target_rate_norm_s", 0.50)
        )
        self._master_gripper_activation_open = float(
            master_gripper_cfg.get("activation_open_norm", 0.85)
        )
        self._master_gripper_full_open_input = float(
            master_gripper_cfg.get("input_full_open_norm", 1.0)
        )
        self._master_gripper_reopen_margin = float(
            master_gripper_cfg.get("reopen_latch_margin_norm", 0.05)
        )
        self._master_gripper_feedback_timeout = float(
            master_gripper_cfg.get("feedback_timeout_s", self.timeout)
        )
        if self._master_gripper_rate <= 0:
            raise ValueError("master_gripper_control.command_rate_hz must be positive")
        if self._master_gripper_feedback_refresh_rate <= 0:
            raise ValueError(
                "master_gripper_control.feedback_refresh_rate_hz must be positive"
            )
        if self._master_gripper_feedback_refresh_timeout <= 0:
            raise ValueError(
                "master_gripper_control.feedback_refresh_timeout_s must be positive"
            )
        if self._master_gripper_position_margin < 0:
            raise ValueError(
                "master_gripper_control.position_limit_margin_rad must be non-negative"
            )
        if not 0 <= self._master_gripper_teaching_kd <= 0.30:
            raise ValueError(
                "master_gripper_control.teaching_kd must be in [0, 0.30]"
            )
        if not 0 < self._master_gripper_torque_limit <= 0.30:
            raise ValueError(
                "master_gripper_control.motor_torque_limit_nm must be in (0, 0.30]"
            )
        if not 0 < self._master_gripper_current_limit <= 0.50:
            raise ValueError(
                "master_gripper_control.motor_current_limit_a must be in (0, 0.50]"
            )
        if not 0 < self._master_gripper_filter_alpha <= 1:
            raise ValueError("master_gripper_control.filter_alpha must be in (0, 1]")
        if not 0 <= self._master_gripper_deadband <= 0.2:
            raise ValueError("master_gripper_control.deadband_norm must be in [0, 0.2]")
        if self._master_gripper_max_rate <= 0:
            raise ValueError(
                "master_gripper_control.max_target_rate_norm_s must be positive"
            )
        if not 0 <= self._master_gripper_activation_open <= 1:
            raise ValueError(
                "master_gripper_control.activation_open_norm must be in [0, 1]"
            )
        if not 0 < self._master_gripper_full_open_input <= 1:
            raise ValueError(
                "master_gripper_control.input_full_open_norm must be in (0, 1]"
            )
        if self._master_gripper_activation_open > self._master_gripper_full_open_input:
            raise ValueError(
                "master_gripper_control.activation_open_norm must not exceed "
                "input_full_open_norm"
            )
        if not 0 < self._master_gripper_reopen_margin <= 0.5:
            raise ValueError(
                "master_gripper_control.reopen_latch_margin_norm must be in (0, 0.5]"
            )
        if self._master_gripper_feedback_timeout <= 0:
            raise ValueError(
                "master_gripper_control.feedback_timeout_s must be positive"
            )
        self._gripper_motor_torque_limit = float(
            self.gripper_control.get("motor_torque_limit_nm", 0.10)
        )
        self._gripper_motor_current_limit = float(
            self.gripper_control.get("motor_current_limit_a", 0.30)
        )
        if not 0 < self._gripper_motor_torque_limit <= 6.0:
            raise ValueError(
                "gripper_control.motor_torque_limit_nm must be in (0, 6] for EL05"
            )
        if not 0 < self._gripper_motor_current_limit <= 11.0:
            raise ValueError(
                "gripper_control.motor_current_limit_a must be in (0, 11] for EL05"
            )
        self.initial_gripper_state = str(
            self.gripper_control.get("initial_state", "hold_current")
        )
        if self.initial_gripper_state not in {"hold_current", "closed", "open"}:
            raise ValueError(
                "gripper_control.initial_state must be hold_current, closed or open"
            )
        force_cfg = self.gripper_control.get("force_stop", {})
        self._force_stop_enabled = bool(force_cfg.get("enabled", False))
        self._force_threshold = float(force_cfg.get("torque_threshold_nm", 0.3))
        self._force_hard_limit = float(
            force_cfg.get("hard_torque_limit_nm", self._force_threshold * 1.5)
        )
        self._force_ema_alpha = float(force_cfg.get("ema_alpha", 0.25))
        self._force_required_samples = int(
            force_cfg.get("consecutive_samples", 5)
        )
        self._force_min_closing_time = float(
            force_cfg.get("min_closing_time_s", 0.10)
        )
        self._force_backoff_norm = float(force_cfg.get("backoff_norm", 0.01))
        self._force_contact_velocity = float(
            force_cfg.get("max_contact_velocity_rad_s", 0.05)
        )
        self._force_min_target_error = float(
            force_cfg.get("min_target_error_norm", 0.015)
        )
        self._force_stall_time = float(
            force_cfg.get("stall_time_s", 0.30)
        )
        self._force_stall_velocity = float(
            force_cfg.get("max_stall_velocity_rad_s", 0.01)
        )
        self._force_stall_torque = float(
            force_cfg.get("stall_torque_threshold_nm", 0.07)
        )
        if self._force_stop_enabled and self._force_threshold <= 0:
            raise ValueError("force_stop.torque_threshold_nm must be positive")
        if self._force_stop_enabled and self._force_hard_limit < self._force_threshold:
            raise ValueError(
                "force_stop.hard_torque_limit_nm must be >= torque_threshold_nm"
            )
        if not 0 < self._force_ema_alpha <= 1:
            raise ValueError("force_stop.ema_alpha must be in (0, 1]")
        if self._force_required_samples < 1:
            raise ValueError("force_stop.consecutive_samples must be >= 1")
        if not 0 <= self._force_backoff_norm <= 0.2:
            raise ValueError("force_stop.backoff_norm must be in [0, 0.2]")
        if self._force_contact_velocity <= 0:
            raise ValueError("force_stop.max_contact_velocity_rad_s must be positive")
        if not 0 < self._force_min_target_error <= 0.2:
            raise ValueError("force_stop.min_target_error_norm must be in (0, 0.2]")
        if self._force_stall_time <= 0:
            raise ValueError("force_stop.stall_time_s must be positive")
        if self._force_stall_velocity <= 0:
            raise ValueError("force_stop.max_stall_velocity_rad_s must be positive")
        if self._force_stall_torque <= 0:
            raise ValueError("force_stop.stall_torque_threshold_nm must be positive")
        self.armed = False
        self._running = False
        self.teleoperation_active = False
        self._filter: CriticallyDampedJointFilter | None = None
        self._thread: threading.Thread | None = None
        self._home_lock = threading.Lock()
        self._lock = threading.Lock()
        self._master_gripper_poll_lock = threading.Lock()
        self._fault: str | None = None
        self._command = np.zeros(6, dtype=np.float64)
        self._master0 = np.zeros(6, dtype=np.float64)
        self._slave0 = np.zeros(6, dtype=np.float64)
        self._gripper_closing = False
        self._gripper_command_time = 0.0
        self._gripper_torque_raw = 0.0
        self._gripper_torque_ema = 0.0
        self._gripper_force_samples = 0
        self._gripper_force_latched = False
        self._gripper_stall_since: float | None = None
        self._master_gripper_active = False
        self._master_gripper_input_norm = self.master_gripper.value
        self._master_gripper_filtered_norm = self.gripper.value
        self._master_gripper_last_update = 0.0
        self._master_gripper_last_feedback_refresh = 0.0
        self._master_gripper_last_position_success = 0.0
        self._master_gripper_position = self.master_gripper.radians()
        self._master_gripper_position_failures = 0
        self._master_gripper_teaching_torque = 0.0
        self._master_gripper_last_rejected_position: float | None = None

    def connect(self) -> None:
        if not self.master.ConnectPort():
            raise RuntimeError("cannot connect master arm")
        if not self.slave.ConnectPort():
            self.master.DisconnectPort()
            raise RuntimeError("cannot connect slave arm")

    def _master_feedback(self) -> np.ndarray:
        """Read J1-J6 feedback and poll the physically disabled master L7."""
        if not self._master_gripper_enabled:
            return _feedback(self.master, "master", self.timeout)
        with self._master_gripper_poll_lock:
            now = time.monotonic()
            refresh_period = 1.0 / self._master_gripper_feedback_refresh_rate
            if (
                now - self._master_gripper_last_feedback_refresh
                >= refresh_period
            ):
                self._master_gripper_last_feedback_refresh = now
                result = self.master.RefreshGripperMotionTeachingFeedback(
                    kd=self._master_gripper_teaching_kd,
                    timeout=self._master_gripper_feedback_refresh_timeout
                )
                position = result[0] if result is not None else None
                torque = result[1] if result is not None else None
                lower = min(self.master_gripper.closed, self.master_gripper.open)
                upper = max(self.master_gripper.closed, self.master_gripper.open)
                if (
                    position is not None
                    and math.isfinite(position)
                    and lower - self._master_gripper_position_margin
                    <= position
                    <= upper + self._master_gripper_position_margin
                ):
                    self._master_gripper_position = float(position)
                    self._master_gripper_teaching_torque = float(torque)
                    self._master_gripper_last_position_success = time.monotonic()
                    self._master_gripper_position_failures = 0
                    self._master_gripper_last_rejected_position = None
                else:
                    self._master_gripper_position_failures += 1
                    if position is not None and math.isfinite(position):
                        self._master_gripper_last_rejected_position = float(position)
            gripper_position = self._master_gripper_position
            last_position_success = self._master_gripper_last_position_success
            position_failures = self._master_gripper_position_failures
            rejected_position = self._master_gripper_last_rejected_position
        values = _feedback(
            self.master,
            "master",
            self.timeout,
        )
        position_age = (
            time.monotonic() - last_position_success
            if last_position_success > 0
            else math.inf
        )
        if position_age > self._master_gripper_feedback_timeout:
            raise RuntimeError(
                "master L7 zero-torque feedback refresh stale/missing "
                f"(age={position_age:.3f}s, "
                f"failures={position_failures}, "
                f"last_rejected_position={rejected_position})"
            )
        values[6] = gripper_position
        return values

    def _wait_master_feedback_ready(self) -> np.ndarray:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "no feedback attempt"
        while time.monotonic() < deadline:
            try:
                return self._master_feedback()
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(0.02)
        raise RuntimeError(
            "master passive gripper did not become ready within "
            f"{self.startup_timeout:.2f}s; last check: {last_error}"
        )

    def _configure_gripper_motor_limit(self) -> None:
        """Apply and verify volatile motor-side L7 torque/current limits."""
        from el_a3_sdk.protocol import ParamIndex

        settings = (
            (
                ParamIndex.LIMIT_TORQUE,
                self._gripper_motor_torque_limit,
                "torque",
                "Nm",
            ),
            (
                ParamIndex.LIMIT_CUR,
                self._gripper_motor_current_limit,
                "current",
                "A",
            ),
        )
        verified = []
        for parameter, limit, label, unit in settings:
            if not self.slave.WriteMotorParameter(7, parameter, limit):
                raise RuntimeError(
                    f"failed to write follower L7 {label} limit "
                    f"{limit:.3f} {unit}"
                )
            time.sleep(0.03)
            result = self.slave.ReadMotorParameter(7, parameter)
            if result is None or not result.success:
                raise RuntimeError(f"cannot read back follower L7 {label} limit")
            tolerance = max(0.01, limit * 0.10)
            if abs(float(result.value) - limit) > tolerance:
                raise RuntimeError(
                    f"follower L7 {label}-limit verification failed: "
                    f"requested={limit:.3f} {unit}, "
                    f"readback={float(result.value):.3f} {unit}"
                )
            verified.append(f"{label}={float(result.value):.3f} {unit}")
        print(
            f"Follower L7 motor limits verified: {', '.join(verified)} (volatile)",
            flush=True,
        )

    def _move_both_home(self) -> tuple[np.ndarray, np.ndarray]:
        """Move both active-position arms to the configured episode start pose."""
        if not self._home_enabled:
            return (
                _feedback(self.master, "master", self.timeout),
                _feedback(self.slave, "slave", self.timeout),
            )
        if not self.master.control_loop_running or not self.slave.control_loop_running:
            raise RuntimeError("both SDK control loops must be running before homing")

        target_deg = np.rad2deg(self._home_target)
        print(
            "HOMING BOTH ARMS: "
            f"target_deg={[round(float(v), 3) for v in target_deg]}, "
            f"v_max={self._home_velocity:.3f} rad/s, "
            f"a_max={self._home_acceleration:.3f} rad/s^2. "
            "Keep the workspace clear and the physical E-stop reachable.",
            flush=True,
        )

        deadline = time.monotonic() + self._home_timeout
        master_queued = self.master.MoveJ(
            self._home_target.tolist(),
            v_max=self._home_velocity,
            a_max=self._home_acceleration,
            block=False,
        )
        slave_queued = self.slave.MoveJ(
            self._home_target.tolist(),
            v_max=self._home_velocity,
            a_max=self._home_acceleration,
            block=False,
        )
        if not master_queued or not slave_queued:
            self.master.cancel_motion()
            self.slave.cancel_motion()
            raise RuntimeError(
                "could not queue the home trajectory on both arms "
                f"(master={master_queued}, slave={slave_queued})"
            )

        for name, arm in (("master", self.master), ("slave", self.slave)):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not arm.wait_for_motion(timeout=remaining):
                self.master.cancel_motion()
                self.slave.cancel_motion()
                raise RuntimeError(f"{name} home trajectory timed out")

        last_master = last_slave = None
        while time.monotonic() < deadline:
            last_master = _feedback(self.master, "master", self.timeout)
            last_slave = _feedback(self.slave, "slave", self.timeout)
            master_error = float(
                np.max(np.abs(last_master[:6] - self._home_target))
            )
            slave_error = float(
                np.max(np.abs(last_slave[:6] - self._home_target))
            )
            if (
                master_error <= self._home_tolerance
                and slave_error <= self._home_tolerance
            ):
                print(
                    "HOME REACHED: "
                    f"master_max_error={math.degrees(master_error):.2f} deg, "
                    f"slave_max_error={math.degrees(slave_error):.2f} deg",
                    flush=True,
                )
                return last_master, last_slave
            time.sleep(0.02)

        master_error = (
            math.inf
            if last_master is None
            else math.degrees(
                float(np.max(np.abs(last_master[:6] - self._home_target)))
            )
        )
        slave_error = (
            math.inf
            if last_slave is None
            else math.degrees(
                float(np.max(np.abs(last_slave[:6] - self._home_target)))
            )
        )
        raise RuntimeError(
            "home verification timed out: "
            f"master_max_error={master_error:.2f} deg, "
            f"slave_max_error={slave_error:.2f} deg"
        )

    def _enter_master_teaching(
        self, sdk_rate: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Switch the master from active positioning to gravity-compensated teaching."""
        kd = float(self.config.get("kinesthetic", {}).get("zero_torque_kd", 0.08))
        if not self.master.ZeroTorqueMode(True, kd=kd):
            raise RuntimeError("master could not enter zero-torque mode")
        self.master.start_control_loop(sdk_rate)

        master = _wait_feedback_ready(
            self.master, "master", self.timeout, self.startup_timeout
        )
        slave = _wait_feedback_ready(
            self.slave, "slave", self.timeout, self.startup_timeout
        )

        if self._master_gripper_enabled:
            if not self.master.SetGripperMotionTeachingMode(
                kd=self._master_gripper_teaching_kd,
                torque_limit_nm=self._master_gripper_torque_limit,
                current_limit_a=self._master_gripper_current_limit,
            ):
                raise RuntimeError(
                    "could not put master L7 into verified motion teaching mode"
                )
            self._master_gripper_last_feedback_refresh = 0.0
            self._master_gripper_last_position_success = 0.0
            self._master_gripper_position_failures = 0
            self._master_gripper_last_rejected_position = None
            master = self._wait_master_feedback_ready()
            self._master_gripper_input_norm = self.master_gripper.normalize(master[6])
            self._master_gripper_filtered_norm = self.gripper.value
            self._master_gripper_last_update = time.monotonic()
            self._master_gripper_active = False
            print(
                "Master L7 is MOTION ZERO-TORQUE PASSIVE "
                f"(Kp=0, Kd={self._master_gripper_teaching_kd:.3f}, "
                f"feedback={self._master_gripper_feedback_refresh_rate:.1f} Hz). "
                "Open the master gripper to "
                f"norm>={self._master_gripper_activation_open:.2f} "
                "to activate follower gripper control; "
                f"master norm 0.00..{self._master_gripper_full_open_input:.2f} "
                "maps to follower norm 0.00..1.00.",
                flush=True,
            )
        return master, slave

    def arm(self) -> None:
        if not self.master.EnableArm() or not self.slave.EnableArm():
            self.emergency_stop()
            raise RuntimeError("master or slave failed to enable")
        try:
            self._configure_gripper_motor_limit()
        except Exception:
            self.emergency_stop()
            raise
        sdk_rate = float(self.config.get("sdk", {}).get("control_rate_hz", 200.0))

        # Enabling two arms is intentionally slow and may leave the master's
        # last feedback older than the strict runtime watchdog threshold.
        # These values are used only to establish safe hold commands.
        master_initial = _feedback(self.master, "master", self.startup_timeout)
        slave_initial = _feedback(self.slave, "slave", self.startup_timeout)

        # Hold the follower's current gripper position at engagement; do not
        # jump to initial_norm from a possibly uncalibrated config.
        self.gripper.value = self.gripper.normalize(slave_initial[6])

        # Put both arms into ordinary active position hold.  Pressing 'a'
        # deliberately does not enter zero-torque teaching; that transition
        # is deferred until the operator presses 'r'.
        self.slave.start_control_loop(sdk_rate)
        if not self.slave.JointCtrl(
            *slave_initial[:6].tolist(), velocities=[0.0] * 6
        ):
            self.emergency_stop()
            raise RuntimeError("slave could not enter position hold")
        self.master.start_control_loop(sdk_rate)
        if not self.master.JointCtrl(
            *master_initial[:6].tolist(), velocities=[0.0] * 6
        ):
            self.emergency_stop()
            raise RuntimeError("master could not enter position hold")

        if self._home_enabled and self._home_on_arm:
            try:
                master_initial, slave_initial = self._move_both_home()
            except Exception:
                self.emergency_stop()
                raise
        else:
            master_initial = _feedback(self.master, "master", self.timeout)
            slave_initial = _feedback(self.slave, "slave", self.timeout)

        master, slave = master_initial, slave_initial
        self._master0, self._slave0 = master[:6].copy(), slave[:6].copy()
        self._command = self._slave0.copy()

        margin = float(self.teleop.get("limit_margin", 0.12))
        tolerance = float(
            self.teleop.get("startup_limit_tolerance_rad", 0.08)
        )
        hard_lower, hard_upper = DEFAULT_LIMITS[:, 0], DEFAULT_LIMITS[:, 1]
        outside = (
            (self._slave0 < hard_lower - tolerance)
            | (self._slave0 > hard_upper + tolerance)
        )
        if np.any(outside):
            bad = (np.flatnonzero(outside) + 1).tolist()
            self.emergency_stop()
            raise RuntimeError(
                f"slave startup joints outside hard limits beyond "
                f"{tolerance:.3f} rad tolerance: joints={bad}, "
                f"q={self._slave0.tolist()}"
            )

        # A soft margin must not itself create a startup move.  If the robot
        # begins inside the margin (common around calibrated zero), include its
        # current position as the innermost allowed boundary while preventing
        # commands that move farther outward.
        soft_lower = hard_lower + margin
        soft_upper = hard_upper - margin
        lower = np.minimum(soft_lower, self._slave0)
        upper = np.maximum(soft_upper, self._slave0)
        self._filter = CriticallyDampedJointFilter(
            self._slave0,
            omega=float(self.teleop.get("filter_omega", 8.0)),
            max_velocity=self.teleop.get("max_velocity", 0.7),
            max_acceleration=self.teleop.get("max_acceleration", 3.0),
            lower_limits=lower,
            upper_limits=upper,
        )
        self.slave.JointCtrl(*self._slave0.tolist(), velocities=[0.0] * 6)
        self._fault = None
        self._running = False
        self.teleoperation_active = False
        self.armed = True
        if self.initial_gripper_state == "closed":
            self._command_gripper_norm(0.0)
        elif self.initial_gripper_state == "open":
            self._command_gripper_norm(1.0)
        print(
            "HOME READY: both arms are holding the episode start pose. "
            "Press 'r' to enter zero-torque teleoperation and start recording.",
            flush=True,
        )

    def start_teleoperation(self) -> None:
        """Enter master teaching mode and start relative follower control."""
        if not self.armed:
            raise RuntimeError("press 'a' before starting teleoperation")
        if self.teleoperation_active:
            return

        with self._home_lock:
            try:
                # The master is actively holding the home pose.  Stop its loop
                # before the driver's mode transition to avoid competing frames.
                self.master.stop_control_loop()
                sdk_rate = float(
                    self.config.get("sdk", {}).get("control_rate_hz", 200.0)
                )
                master, slave = self._enter_master_teaching(sdk_rate)
                with self._lock:
                    self._master0 = master[:6].copy()
                    self._slave0 = slave[:6].copy()
                    self._command = self._slave0.copy()
                    if self._filter is None:
                        raise RuntimeError("joint filter is not initialized")
                    self._filter.reset(self._slave0)
                    self.gripper.value = self.gripper.normalize(slave[6])
                    self._master_gripper_input_norm = (
                        self.master_gripper.normalize(master[6])
                    )
                    self._master_gripper_filtered_norm = self.gripper.value
                    self._master_gripper_last_update = time.monotonic()
                    self._master_gripper_active = False
                    self._fault = None

                self._running = True
                self.teleoperation_active = True
                self._thread = threading.Thread(
                    target=self._teleop_loop,
                    daemon=True,
                    name="master-slave",
                )
                self._thread.start()
                print(
                    "TELEOP ACTIVE: master is zero-torque, follower tracking "
                    "is running, and recording may begin",
                    flush=True,
                )
            except Exception:
                self.emergency_stop()
                raise

    def _teleop_loop(self) -> None:
        rate = float(self.teleop.get("rate_hz", 100.0))
        period = 1.0 / rate
        scale = np.broadcast_to(np.asarray(self.teleop.get("scale", 1.0), dtype=float), (6,))
        direction = np.broadcast_to(np.asarray(self.teleop.get("direction", 1.0), dtype=float), (6,))
        threshold = float(self.teleop.get("max_follow_error_rad", 0.35))
        allowed_cycles = int(self.teleop.get("follow_error_cycles", 10))
        bad_cycles = 0
        previous = time.monotonic()
        deadline = previous
        try:
            while self._running:
                now = time.monotonic()
                master = self._master_feedback()
                slave = _feedback(self.slave, "slave", self.timeout)
                target = self._slave0 + scale * direction * (master[:6] - self._master0)
                state = self._filter.step(target, max(now - previous, 1e-4))
                previous = now
                error = float(np.max(np.abs(state.position - slave[:6])))
                bad_cycles = bad_cycles + 1 if error > threshold else 0
                if bad_cycles >= allowed_cycles:
                    raise RuntimeError(f"following error {error:.3f} rad exceeds {threshold:.3f} rad")
                if not self.slave.JointCtrl(*state.position.tolist(), velocities=state.velocity.tolist()):
                    raise RuntimeError("slave joint command failed")
                with self._lock:
                    self._command = state.position.copy()
                self._update_master_gripper_control(master, now)
                self._update_gripper_force_guard(slave, now)
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
                if time.monotonic() - deadline > period:
                    deadline = time.monotonic()
        except Exception as exc:
            with self._lock:
                self._fault = str(exc)
            print(
                f"\nTELEOP SAFETY STOP: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._running = False
            self.teleoperation_active = False
            self.armed = False
            try:
                self.slave.EmergencyStop()
                self.master.EmergencyStop()
            except Exception:
                pass

    def _update_master_gripper_control(
        self,
        master_feedback: np.ndarray,
        now: float,
    ) -> None:
        if not self._master_gripper_enabled:
            return
        period = 1.0 / self._master_gripper_rate
        if now - self._master_gripper_last_update < period:
            return
        dt = max(now - self._master_gripper_last_update, period)
        self._master_gripper_last_update = now
        master_norm = self.master_gripper.normalize(float(master_feedback[6]))
        mapped_norm = float(
            np.clip(master_norm / self._master_gripper_full_open_input, 0.0, 1.0)
        )

        release_latch = False
        with self._lock:
            self._master_gripper_input_norm = master_norm
            if not self._master_gripper_active:
                # Requiring an explicit open pose makes engagement intentional
                # and prevents an arbitrary master startup angle from closing
                # the follower as soon as 'a' is pressed.
                if master_norm < self._master_gripper_activation_open:
                    return
                self._master_gripper_active = True
                self._master_gripper_filtered_norm = self.gripper.value
                print(
                    "MASTER GRIPPER CONTROL ACTIVE: "
                    f"input_norm={master_norm:.3f}, "
                    f"mapped_norm={mapped_norm:.3f}",
                    flush=True,
                )

            current_target = self.gripper.value
            latched = self._gripper_force_latched
            if latched:
                release_threshold = min(
                    1.0,
                    current_target + self._master_gripper_reopen_margin,
                )
                if mapped_norm < release_threshold:
                    return
                # Once the operator has explicitly opened beyond the held
                # follower position, discard the previous closed-side filter
                # history so the first post-latch command can only open.
                self._master_gripper_filtered_norm = mapped_norm
                release_latch = True
            else:
                alpha = self._master_gripper_filter_alpha
                self._master_gripper_filtered_norm = (
                    alpha * mapped_norm
                    + (1.0 - alpha) * self._master_gripper_filtered_norm
                )

            max_step = self._master_gripper_max_rate * dt
            delta = float(
                np.clip(
                    self._master_gripper_filtered_norm - current_target,
                    -max_step,
                    max_step,
                )
            )
            if abs(delta) < self._master_gripper_deadband:
                return
            candidate = float(np.clip(current_target + delta, 0.0, 1.0))
            opening = candidate > current_target
            closing = candidate < current_target
            if latched and not opening:
                return

        self._command_gripper_norm(
            candidate,
            clear_force_latch=release_latch or opening,
            close_intent=closing,
        )
        if release_latch:
            print(
                "GRIPPER FORCE LATCH CLEARED by explicit master opening",
                flush=True,
            )

    def _command_gripper_norm(
        self,
        target_norm: float,
        bypass_force_latch: bool = False,
        clear_force_latch: bool = False,
        close_intent: bool = False,
    ) -> None:
        target = float(np.clip(target_norm, 0.0, 1.0))
        actual_angle = float(self.slave.GetArmJointMsgs().joint_7)
        actual_norm = self.gripper.normalize(actual_angle)
        send_command = True
        with self._lock:
            closing = target < actual_norm - self._force_min_target_error
            opening = target > actual_norm + self._force_min_target_error
            if (
                (closing or close_intent)
                and self._gripper_force_latched
                and not bypass_force_latch
            ):
                print(
                    "GRIPPER CLOSE BLOCKED: force stop is latched; "
                    "open the master gripper or press ]/o before closing again",
                    flush=True,
                )
                return
            if (opening or clear_force_latch) and not bypass_force_latch:
                self._gripper_force_latched = False
            duplicate_target = abs(target - self.gripper.value) < 1e-9
            continued_closing = closing and self._gripper_closing
            self.gripper.value = target
            self._gripper_closing = closing
            # Keyboard auto-repeat can produce many identical 'c' commands.
            # Do not restart contact detection while the same closing motion
            # is in progress, otherwise consecutive samples never accumulate.
            if not continued_closing:
                self._gripper_command_time = time.monotonic()
                self._gripper_force_samples = 0
                self._gripper_stall_since = None
            send_command = not duplicate_target
            angle = self.gripper.radians()
        if send_command and not self.slave.GripperCtrl(angle):
            raise RuntimeError("slave gripper command failed")

    def _update_gripper_force_guard(
        self,
        slave_feedback: np.ndarray,
        now: float,
    ) -> None:
        efforts = self.slave.GetArmJointEfforts()
        velocities = self.slave.GetArmJointVelocities()
        raw = abs(float(efforts.joint_7))
        velocity = abs(float(velocities.joint_7))
        current_norm = self.gripper.normalize(float(slave_feedback[6]))
        with self._lock:
            self._gripper_torque_raw = raw
            self._gripper_torque_ema = (
                self._force_ema_alpha * raw
                + (1.0 - self._force_ema_alpha) * self._gripper_torque_ema
            )
            closing_requested = (
                self.gripper.value
                < current_norm - self._force_min_target_error
            )
            active = (
                self._force_stop_enabled
                and closing_requested
                and not self._gripper_force_latched
                and now - self._gripper_command_time >= self._force_min_closing_time
            )
            hard_trigger = active and raw >= self._force_hard_limit
            stalled_contact = velocity <= self._force_contact_velocity
            if (
                active
                and stalled_contact
                and self._gripper_torque_ema >= self._force_threshold
            ):
                self._gripper_force_samples += 1
            else:
                self._gripper_force_samples = 0
            soft_trigger = (
                self._gripper_force_samples >= self._force_required_samples
            )
            sustained_stall = (
                active
                and velocity <= self._force_stall_velocity
                and self._gripper_torque_ema >= self._force_stall_torque
            )
            if sustained_stall:
                if self._gripper_stall_since is None:
                    self._gripper_stall_since = now
            else:
                self._gripper_stall_since = None
            stall_trigger = (
                self._gripper_stall_since is not None
                and now - self._gripper_stall_since >= self._force_stall_time
            )
            trigger = hard_trigger or soft_trigger or stall_trigger
            filtered = self._gripper_torque_ema

        if trigger:
            hold_norm = min(1.0, current_norm + self._force_backoff_norm)
            self._command_gripper_norm(hold_norm, bypass_force_latch=True)
            with self._lock:
                self._gripper_force_latched = True
                self._gripper_closing = False
                self._gripper_stall_since = None
            reason = "HARD" if hard_trigger else ("SOFT" if soft_trigger else "STALL")
            print(
                f"\nGRIPPER FORCE STOP ({reason}): "
                f"raw={raw:.3f} Nm, ema={filtered:.3f} Nm, "
                f"velocity={velocity:.3f} rad/s, hold_norm={hold_norm:.3f}; "
                f"closing is now LATCHED",
                flush=True,
            )

    def adjust_gripper(self, side: str, delta: float) -> None:
        del side
        with self._lock:
            target = self.gripper.value + delta
        self._command_gripper_norm(
            target,
            clear_force_latch=delta > 0,
            close_intent=delta < 0,
        )

    def reset_to_home(self) -> None:
        """Stop teaching, home both arms, and wait in active position hold."""
        if not self._home_enabled:
            print("Episode homing is disabled in the configuration", flush=True)
            return
        if not self.armed:
            raise RuntimeError("press 'a' before homing")

        with self._home_lock:
            try:
                print(
                    "PAUSING TELEOP FOR HOME: do not guide either arm during "
                    "the active return motion",
                    flush=True,
                )
                self._running = False
                if self._thread is not None:
                    self._thread.join(timeout=2.0)
                    if self._thread.is_alive():
                        raise RuntimeError("teleoperation thread did not stop for homing")
                    self._thread = None

                sdk_rate = float(
                    self.config.get("sdk", {}).get("control_rate_hz", 200.0)
                )
                if self.teleoperation_active:
                    # The master is in gravity-compensated teaching mode.
                    # Stop its loop before restoring active motion control.
                    self.master.stop_control_loop()
                    if not self.master.ZeroTorqueMode(False):
                        raise RuntimeError(
                            "master could not leave zero-torque mode for homing"
                        )
                    self.master.start_control_loop(sdk_rate)
                elif not self.master.control_loop_running:
                    self.master.start_control_loop(sdk_rate)

                master_hold = _feedback(
                    self.master, "master", self.startup_timeout
                )
                if not self.master.JointCtrl(
                    *master_hold[:6].tolist(), velocities=[0.0] * 6
                ):
                    raise RuntimeError("master could not enter active position hold")
                if not self.slave.control_loop_running:
                    self.slave.start_control_loop(sdk_rate)

                master, slave = self._move_both_home()

                self.gripper.value = self.gripper.normalize(slave[6])
                with self._lock:
                    self._master0 = master[:6].copy()
                    self._slave0 = slave[:6].copy()
                    self._command = self._slave0.copy()
                    self._filter.reset(self._slave0)
                    self._master_gripper_input_norm = (
                        self.master_gripper.normalize(master[6])
                    )
                    self._master_gripper_filtered_norm = self.gripper.value
                    self._master_gripper_last_update = time.monotonic()
                    self._master_gripper_active = False
                    self._gripper_closing = False
                    self._gripper_force_samples = 0
                    self._gripper_stall_since = None
                    self._fault = None

                self._running = False
                self.teleoperation_active = False
                self.armed = True
                print(
                    "HOME COMPLETE: both arms are at the episode start pose; "
                    "active position hold is enabled. Press 'r' to start the "
                    "next teleoperation recording.",
                    flush=True,
                )
            except Exception:
                self.emergency_stop()
                raise

    def gripper_status(self) -> str:
        joints = self.slave.GetArmJointMsgs()
        efforts = self.slave.GetArmJointEfforts()
        iq_text = "unavailable"
        try:
            from el_a3_sdk.protocol import ParamIndex

            iq_result = self.slave.ReadMotorParameter(7, ParamIndex.IQF)
            if (
                iq_result is not None
                and iq_result.success
                and math.isfinite(float(iq_result.value))
            ):
                iq = float(iq_result.value)
                iq_text = (
                    f"{iq:+.3f} A (abs={abs(iq):.3f}, "
                    f"limit={self._gripper_motor_current_limit:.3f} A)"
                )
        except Exception:
            # Diagnostics must not emergency-stop teleoperation merely because
            # a one-shot parameter query timed out on a busy CAN bus.
            pass
        with self._lock:
            target = self.gripper.value
            filtered = self._gripper_torque_ema
            enabled = self._force_stop_enabled
            latched = self._gripper_force_latched
            master_input = self._master_gripper_input_norm
            master_active = self._master_gripper_active
            master_position_failures = self._master_gripper_position_failures
            master_teaching_torque = self._master_gripper_teaching_torque
            master_position = self._master_gripper_position
            master_rejected_position = self._master_gripper_last_rejected_position
            master_mode = (
                "OFF"
                if not self._master_gripper_enabled
                else ("ACTIVE" if master_active else "WAIT_OPEN")
            )
        return (
            f"follower L7: angle={math.degrees(joints.joint_7):.2f} deg, "
            f"norm={self.gripper.normalize(joints.joint_7):.3f}, "
            f"torque={efforts.joint_7:.3f} Nm "
            f"(abs_ema={filtered:.3f}), iq={iq_text}, "
            f"target_norm={target:.3f}, "
            f"force_stop={'ON' if enabled else 'OFF'}, "
            f"soft={self._force_threshold:.3f} Nm, "
            f"hard={self._force_hard_limit:.3f} Nm, "
            f"latched={latched}, "
            f"master_angle={math.degrees(master_position):.2f} deg, "
            f"master_gripper_norm={master_input:.3f}, "
            f"master_teaching_torque={master_teaching_torque:.3f} Nm, "
            f"master_feedback_refresh_failures={master_position_failures}, "
            f"master_last_rejected_position={master_rejected_position}, "
            f"master_control={master_mode}"
        )

    def snapshot(self) -> RigSnapshot:
        with self._lock:
            fault, command = self._fault, self._command.copy()
        if fault:
            raise RuntimeError(f"teleoperation safety stop: {fault}")
        if not self.armed:
            raise RuntimeError("rig is not armed")
        if not self.teleoperation_active:
            raise RuntimeError("press 'r' to start teleoperation before recording")
        slave = _feedback(self.slave, "slave", self.timeout)
        master = self._master_feedback()
        grip_feedback = self.gripper.normalize(slave[6])
        action = np.r_[command, self.gripper.value].astype(np.float32)
        slave7 = np.r_[slave[:6], grip_feedback].astype(np.float32)
        master7 = np.r_[master[:6], master[6]].astype(np.float32)
        return RigSnapshot(action, _pose(self.slave, self.slave_cfg), grip_feedback,
                           None, None, slave7, master7)

    def check_health(self) -> None:
        with self._lock:
            fault = self._fault
        if fault:
            raise RuntimeError(f"teleoperation safety stop: {fault}")
        if self.armed and not self.teleoperation_active:
            _feedback(self.master, "master", self.timeout)
            _feedback(self.slave, "slave", self.timeout)

    def emergency_stop(self) -> None:
        self._running = False
        self.teleoperation_active = False
        self.armed = False
        for arm in (self.slave, self.master):
            try:
                arm.EmergencyStop()
            except Exception:
                pass

    def close(self) -> None:
        self._running = False
        self.teleoperation_active = False
        self.armed = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for arm in (self.slave, self.master):
            try:
                arm.EmergencyStop()
                arm.DisconnectPort()
            except Exception:
                pass


def build_rig(config: dict, dry_run: bool):
    mode = str(config.get("mode", "bimanual_kinesthetic"))
    if mode not in {"bimanual_kinesthetic", "master_slave_single"}:
        raise ValueError("mode must be bimanual_kinesthetic or master_slave_single")
    if dry_run:
        return DryRunRig(mode)
    if not bool(config.get("hardware_calibrated", False)):
        raise RuntimeError(
            "real hardware is locked: calibrate CAN, joint directions, base transforms and "
            "gripper endpoints, then set hardware_calibrated: true"
        )
    left_can = str(config["arms"]["left"]["can_name"])
    right_can = str(config["arms"]["right"]["can_name"])
    if left_can == right_can:
        raise ValueError("left/master and right/slave arms must use different CAN interfaces")
    return (BimanualKinestheticRig(config) if mode == "bimanual_kinesthetic"
            else MasterSlaveSingleRig(config))
