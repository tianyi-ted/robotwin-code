#!/usr/bin/env python3
"""Interactive, safety-gated EDULITE-A3 demonstration collector."""

from __future__ import annotations

import argparse
import math
import select
import signal
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cameras import build_cameras
from dataset_io import EpisodeBuffer, EpisodeSample, RoboTwinEpisodeWriter
from hardware import build_rig


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


def read_key() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    return sys.stdin.read(1) if readable else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.example.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="never open CAN/cameras; synthesize data")
    parser.add_argument("--auto-seconds", type=float, default=None,
                        help="dry-run only: automatically record/save this many seconds")
    parser.add_argument("--output-root", help="override collection.output_root")
    parser.add_argument("--task", help="override collection.task_name")
    parser.add_argument("--split", choices=["demo_clean", "demo_randomized"])
    parser.add_argument("--mode", choices=["bimanual_kinesthetic", "master_slave_single"],
                        help="override the configured collection mode")
    parser.add_argument("--single-arm-padding", choices=["none", "zero_right"],
                        help="override collection.single_arm_padding")
    return parser.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    collection = config.setdefault("collection", {})
    if args.output_root:
        collection["output_root"] = args.output_root
    elif not Path(str(collection["output_root"])).is_absolute():
        collection["output_root"] = str((config_path.parent / collection["output_root"]).resolve())
    if args.task:
        collection["task_name"] = args.task
    if args.split:
        collection["split"] = args.split
    if args.mode:
        config["mode"] = args.mode
    if args.single_arm_padding:
        collection["single_arm_padding"] = args.single_arm_padding
    return config


def make_sample(
    rig,
    cameras: dict,
    max_camera_age: float,
    synthetic_zero_right: bool = False,
) -> EpisodeSample:
    snapshot = rig.snapshot()
    action = np.asarray(snapshot.action, dtype=np.float32)
    right_pose = snapshot.right_pose
    right_gripper = snapshot.right_gripper
    if synthetic_zero_right:
        if action.shape != (7,) or right_pose is not None:
            raise RuntimeError("zero-right padding is only valid for 7D master_slave_single data")
        action = np.concatenate([action, np.zeros(7, dtype=np.float32)])
        right_pose = np.zeros(7, dtype=np.float32)
        right_gripper = 0.0
    images, camera_times = {}, {}
    for name, camera in cameras.items():
        frame = camera.latest(max_camera_age)
        images[name] = frame.rgb
        camera_times[name] = frame.monotonic_time
    return EpisodeSample(
        monotonic_time=time.monotonic(),
        wall_time=time.time(),
        action=action,
        left_pose=np.asarray(snapshot.left_pose, dtype=np.float32),
        left_gripper=float(snapshot.left_gripper),
        right_pose=None if right_pose is None else np.asarray(right_pose, dtype=np.float32),
        right_gripper=right_gripper,
        slave_feedback=np.asarray(snapshot.slave_feedback, dtype=np.float32),
        master_feedback=(None if snapshot.master_feedback is None
                         else np.asarray(snapshot.master_feedback, dtype=np.float32)),
        images=images,
        camera_times=camera_times,
    )


def print_controls(mode: str) -> None:
    print("\nControls: a=enable/home, r=schedule teleop+recording, s=save, d=discard, e=E-STOP, q=quit")
    if mode == "bimanual_kinesthetic":
        print("Grippers: 1/2=left close/open, 3/4=right close/open")
    else:
        print("a=enable and home both arms; r=enter teleop and start recording")
        print("Episode reset: h=home both J1-J6 (also runs after s/d when enabled)")
        print("Master L7 controls follower gripper after the configured open activation")
        print("Follower keyboard fallback: [ / ]=close/open step, c/o=fully close/open")
        print("Gripper diagnostics: t=print L7 angle/torque/current and protection state")
    print("Keep the physical emergency-stop reachable. Software 'e' is only a second layer.\n")


def main() -> int:
    args = parse_args()
    if args.auto_seconds is not None and not args.dry_run:
        raise ValueError("--auto-seconds is restricted to --dry-run")
    if args.auto_seconds is not None and args.auto_seconds <= 0:
        raise ValueError("--auto-seconds must be positive")
    config = load_config(args.config, args)
    collection = config["collection"]
    rate_hz = float(collection.get("rate_hz", 30.0))
    if rate_hz <= 0:
        raise ValueError("collection.rate_hz must be positive")
    min_frames = int(collection.get("min_frames", 40))
    camera_max_age = float(collection.get("camera_max_age_s", 0.25))
    gripper_step = float(collection.get("gripper_step", 0.05))
    mode = str(config.get("mode", "bimanual_kinesthetic"))
    reset_cfg = config.get("episode_reset", {})
    reset_enabled = bool(reset_cfg.get("enabled", False))
    reset_after_save = reset_enabled and bool(
        reset_cfg.get("return_after_save", True)
    )
    reset_after_discard = reset_enabled and bool(
        reset_cfg.get("return_after_discard", True)
    )
    teleop_start_delay = float(
        config.get("master_slave", {}).get("start_delay_s", 0.0)
    )
    if teleop_start_delay < 0:
        raise ValueError("master_slave.start_delay_s must be non-negative")
    single_arm_padding = str(collection.get("single_arm_padding", "none"))
    if single_arm_padding not in {"none", "zero_right"}:
        raise ValueError("collection.single_arm_padding must be none or zero_right")
    synthetic_zero_right = single_arm_padding == "zero_right"
    if synthetic_zero_right and mode != "master_slave_single":
        raise ValueError("zero_right padding is only valid in master_slave_single mode")

    writer = RoboTwinEpisodeWriter(
        output_root=collection["output_root"],
        task_name=collection["task_name"],
        split=collection.get("split", "demo_clean"),
        instructions_seen=collection.get("instructions", {}).get("seen", []),
        instructions_unseen=collection.get("instructions", {}).get("unseen", []),
        rate_hz=rate_hz,
        mode=mode,
        jpeg_quality=int(collection.get("jpeg_quality", 95)),
        synthetic_zero_right=synthetic_zero_right,
    )
    cameras = build_cameras(config.get("cameras", []), args.dry_run)
    rig = build_rig(config, args.dry_run)
    buffer = EpisodeBuffer()
    recording = False
    armed = False
    stop_requested = False
    auto_start = None
    pending_start_deadline: float | None = None
    pending_countdown_second: int | None = None

    def request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    def begin_teleop_recording() -> None:
        nonlocal recording, deadline
        if mode == "master_slave_single":
            rig.start_teleoperation()
        buffer.clear()
        recording = True
        deadline = time.monotonic()
        print("TELEOP + RECORDING")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        for camera in cameras.values():
            camera.start()
        rig.connect()  # Connection alone does not enable motors.
        print(f"Connected. mode={mode}, output={writer.base}")
        print_controls(mode)

        if args.auto_seconds is not None:
            rig.arm()
            armed = True
            if mode == "master_slave_single":
                rig.start_teleoperation()
            recording = True
            auto_start = time.monotonic()
            print(f"Dry-run auto recording for {args.auto_seconds:.2f}s...")

        deadline = time.monotonic()
        with raw_terminal(sys.stdin.isatty() and args.auto_seconds is None):
            while not stop_requested:
                if armed:
                    # Surface asynchronous controller faults even when the
                    # operator is testing motion without recording an episode.
                    rig.check_health()
                if pending_start_deadline is not None:
                    remaining = pending_start_deadline - time.monotonic()
                    if remaining <= 0:
                        pending_start_deadline = None
                        pending_countdown_second = None
                        begin_teleop_recording()
                    else:
                        countdown_second = max(1, int(math.ceil(remaining)))
                        if countdown_second != pending_countdown_second:
                            pending_countdown_second = countdown_second
                            print(f"Starting teleoperation in {countdown_second}...")
                key = None if args.auto_seconds is not None else read_key()
                if key:
                    try:
                        if key == "a":
                            if not armed:
                                rig.arm()
                                armed = True
                                if mode == "master_slave_single":
                                    print(
                                        "ARMED/HOME READY: both arms are holding; "
                                        "press 'r' to start teleoperation"
                                    )
                                else:
                                    print("ARMED: gravity compensation engaged")
                        elif key == "r":
                            if not armed:
                                print("Press 'a' before recording")
                            elif recording:
                                print("Already recording")
                            elif pending_start_deadline is not None:
                                print("Teleoperation start countdown is already running")
                            else:
                                if (
                                    mode == "master_slave_single"
                                    and teleop_start_delay > 0
                                ):
                                    pending_start_deadline = (
                                        time.monotonic() + teleop_start_delay
                                    )
                                    pending_countdown_second = int(
                                        math.ceil(teleop_start_delay)
                                    )
                                    print(
                                        "START SCHEDULED: both arms remain in "
                                        f"position hold for {teleop_start_delay:.1f}s"
                                    )
                                    print(
                                        f"Starting teleoperation in "
                                        f"{pending_countdown_second}..."
                                    )
                                else:
                                    begin_teleop_recording()
                        elif key == "s":
                            if pending_start_deadline is not None:
                                pending_start_deadline = None
                                pending_countdown_second = None
                                print("Teleoperation start countdown cancelled")
                            elif not recording:
                                print("Not recording")
                            elif len(buffer) < min_frames:
                                print(f"Too short: {len(buffer)} frames; need at least {min_frames}")
                            else:
                                recording = False
                                path = writer.write(buffer)
                                print(f"SAVED {len(buffer)} frames -> {path}")
                                buffer.clear()
                                if mode == "master_slave_single" and reset_after_save:
                                    print("Returning both arms to the episode start pose...")
                                    rig.reset_to_home()
                        elif key == "d":
                            if pending_start_deadline is not None:
                                pending_start_deadline = None
                                pending_countdown_second = None
                                print("Teleoperation start countdown cancelled")
                            had_episode = recording or len(buffer) > 0
                            recording = False
                            print(f"DISCARDED {len(buffer)} buffered frames")
                            buffer.clear()
                            if (
                                had_episode
                                and armed
                                and mode == "master_slave_single"
                                and reset_after_discard
                            ):
                                print("Returning both arms to the episode start pose...")
                                rig.reset_to_home()
                        elif key == "e":
                            pending_start_deadline = None
                            pending_countdown_second = None
                            recording = False
                            buffer.clear()
                            rig.emergency_stop()
                            armed = False
                            print("EMERGENCY STOP: all configured motors disabled")
                        elif key == "q":
                            pending_start_deadline = None
                            pending_countdown_second = None
                            if len(buffer):
                                print(f"Discarding {len(buffer)} unsaved frames on quit")
                            break
                        elif key == "t" and armed:
                            print(rig.gripper_status())
                        elif key == "h":
                            if mode != "master_slave_single":
                                print("'h' is only available in master_slave_single mode")
                            elif not armed:
                                print("Press 'a' before homing")
                            elif recording:
                                print("Stop/save/discard the current recording before homing")
                            else:
                                if pending_start_deadline is not None:
                                    pending_start_deadline = None
                                    pending_countdown_second = None
                                    print("Teleoperation start countdown cancelled")
                                rig.reset_to_home()
                        elif key in {"1", "2", "3", "4", "[", "]", "c", "o"} and armed:
                            mapping = {
                                "1": ("left", -gripper_step), "2": ("left", gripper_step),
                                "3": ("right", -gripper_step), "4": ("right", gripper_step),
                                "[": ("left", -gripper_step), "]": ("left", gripper_step),
                                "c": ("left", -1.0), "o": ("left", 1.0),
                            }
                            side, delta = mapping[key]
                            rig.adjust_gripper(side, delta)
                    except Exception:
                        rig.emergency_stop()
                        armed = False
                        recording = False
                        raise

                if recording:
                    now = time.monotonic()
                    if now >= deadline:
                        buffer.append(
                            make_sample(rig, cameras, camera_max_age, synthetic_zero_right)
                        )
                        deadline += 1.0 / rate_hz
                        if now - deadline > 1.0 / rate_hz:
                            print("Warning: collection loop missed a period; resynchronizing")
                            deadline = now + 1.0 / rate_hz

                if auto_start is not None and time.monotonic() - auto_start >= args.auto_seconds:
                    recording = False
                    if len(buffer) < min_frames:
                        raise RuntimeError(f"dry-run produced only {len(buffer)} frames (< {min_frames})")
                    path = writer.write(buffer)
                    print(f"SAVED {len(buffer)} frames -> {path}")
                    buffer.clear()
                    break
                time.sleep(0.001 if recording else 0.01)
        return 0
    except Exception as exc:
        try:
            rig.emergency_stop()
        except Exception:
            pass
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            rig.close()
        finally:
            for camera in cameras.values():
                camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
