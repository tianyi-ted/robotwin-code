"""Atomic RoboTwin-compatible HDF5 episode writer and validator."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import h5py
import numpy as np


@dataclass
class EpisodeSample:
    monotonic_time: float
    wall_time: float
    action: np.ndarray
    left_pose: np.ndarray
    left_gripper: float
    right_pose: np.ndarray | None
    right_gripper: float | None
    slave_feedback: np.ndarray
    master_feedback: np.ndarray | None
    images: Mapping[str, np.ndarray]
    camera_times: Mapping[str, float]


@dataclass
class EpisodeBuffer:
    samples: list[EpisodeSample] = field(default_factory=list)

    def append(self, sample: EpisodeSample) -> None:
        self.samples.append(sample)

    def clear(self) -> None:
        self.samples.clear()

    def __len__(self) -> int:
        return len(self.samples)


def _stack(samples: Sequence[EpisodeSample], attr: str, dtype=np.float32) -> np.ndarray:
    return np.stack([np.asarray(getattr(s, attr), dtype=dtype) for s in samples])


def _encode_rgb_frames(frames: Sequence[np.ndarray], quality: int) -> tuple[list[bytes], int]:
    encoded: list[bytes] = []
    max_len = 0
    for index, rgb in enumerate(frames):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"RGB frame {index} must have shape [H,W,3] and dtype uint8")
        # The project loader decodes with cv2 and passes the numeric array to PIL
        # without BGR->RGB conversion.  Encoding an RGB numeric array with cv2 and
        # decoding it with cv2 preserves that numeric channel order.
        ok, buf = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(rgb),
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for frame {index}")
        payload = buf.tobytes()
        encoded.append(payload)
        max_len = max(max_len, len(payload))
    return encoded, max_len


class RoboTwinEpisodeWriter:
    def __init__(
        self,
        output_root: str | Path,
        task_name: str,
        split: str,
        instructions_seen: Sequence[str],
        instructions_unseen: Sequence[str],
        rate_hz: float,
        mode: str,
        jpeg_quality: int = 95,
        synthetic_zero_right: bool = False,
    ) -> None:
        if split not in {"demo_clean", "demo_randomized"}:
            raise ValueError("split must be demo_clean or demo_randomized")
        if not instructions_seen:
            raise ValueError("at least one seen instruction is required")
        self.base = Path(output_root).expanduser().resolve() / task_name / split
        self.data_dir = self.base / "data"
        self.instructions_dir = self.base / "instructions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.instructions_dir.mkdir(parents=True, exist_ok=True)
        self.instructions_seen = [str(v) for v in instructions_seen]
        self.instructions_unseen = [str(v) for v in instructions_unseen]
        self.rate_hz = float(rate_hz)
        self.mode = mode
        self.jpeg_quality = int(jpeg_quality)
        self.synthetic_zero_right = bool(synthetic_zero_right)

    def next_episode_id(self) -> int:
        ids = []
        for path in self.data_dir.glob("episode*.hdf5"):
            match = re.fullmatch(r"episode_?(\d+)\.hdf5", path.name)
            if match:
                ids.append(int(match.group(1)))
        return max(ids, default=-1) + 1

    def write(self, episode: EpisodeBuffer, episode_id: int | None = None) -> Path:
        if not episode.samples:
            raise ValueError("cannot write an empty episode")
        eid = self.next_episode_id() if episode_id is None else int(episode_id)
        target = self.data_dir / f"episode{eid}.hdf5"
        instruction_target = self.instructions_dir / f"episode{eid}.json"
        if target.exists() or instruction_target.exists():
            raise FileExistsError(f"episode {eid} already exists under {self.base}")

        temporary = target.with_suffix(".partial.hdf5")
        instruction_tmp = instruction_target.with_suffix(".partial.json")
        try:
            self._write_hdf5(temporary, episode.samples)
            with instruction_tmp.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"seen": self.instructions_seen, "unseen": self.instructions_unseen},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            os.replace(instruction_tmp, instruction_target)
        except Exception:
            temporary.unlink(missing_ok=True)
            instruction_tmp.unlink(missing_ok=True)
            raise
        return target

    def _write_hdf5(self, path: Path, samples: Sequence[EpisodeSample]) -> None:
        action = _stack(samples, "action")
        left_pose = _stack(samples, "left_pose")
        left_gripper = np.asarray([s.left_gripper for s in samples], dtype=np.float32)
        dual = samples[0].right_pose is not None

        with h5py.File(path, "w") as root:
            root.attrs["sim"] = False
            root.attrs["real_robot"] = "EDULITE_A3"
            root.attrs["teleop_mode"] = self.mode
            root.attrs["sample_rate_hz"] = self.rate_hz
            root.attrs["action_semantics"] = "joint_position_target_rad_plus_normalized_gripper"
            root.attrs["structural_dual_arm_schema"] = bool(dual and action.shape[1] == 14)
            root.attrs["dual_arm_compatible"] = bool(
                dual and action.shape[1] == 14 and not self.synthetic_zero_right
            )
            root.attrs["synthetic_zero_right"] = self.synthetic_zero_right
            if self.synthetic_zero_right:
                root.attrs["inactive_arm"] = "right"

            joint_action = root.create_group("joint_action")
            joint_action.create_dataset("vector", data=action, compression="gzip", compression_opts=1)
            if action.shape[1] == 14:
                joint_action.create_dataset("left_arm", data=action[:, :6])
                joint_action.create_dataset("left_gripper", data=action[:, 6])
                joint_action.create_dataset("right_arm", data=action[:, 7:13])
                joint_action.create_dataset("right_gripper", data=action[:, 13])
            elif action.shape[1] == 7:
                joint_action.create_dataset("left_arm", data=action[:, :6])
                joint_action.create_dataset("left_gripper", data=action[:, 6])

            endpose = root.create_group("endpose")
            endpose.create_dataset("left_endpose", data=left_pose)
            endpose.create_dataset("left_gripper", data=left_gripper)
            if dual:
                endpose.create_dataset("right_endpose", data=_stack(samples, "right_pose"))
                endpose.create_dataset(
                    "right_gripper",
                    data=np.asarray([s.right_gripper for s in samples], dtype=np.float32),
                )

            observation = root.create_group("observation")
            camera_names = list(samples[0].images.keys())
            for name in camera_names:
                group = observation.create_group(name)
                payloads, max_len = _encode_rgb_frames(
                    [s.images[name] for s in samples], self.jpeg_quality
                )
                group.create_dataset("rgb", data=payloads, dtype=f"S{max_len}")

            timing = root.create_group("timing")
            timing.create_dataset(
                "monotonic",
                data=np.asarray([s.monotonic_time for s in samples], dtype=np.float64),
            )
            timing.create_dataset(
                "wall_time",
                data=np.asarray([s.wall_time for s in samples], dtype=np.float64),
            )
            for name in camera_names:
                timing.create_dataset(
                    f"camera_{name}",
                    data=np.asarray([s.camera_times[name] for s in samples], dtype=np.float64),
                )

            diagnostic = root.create_group("teleop")
            diagnostic.create_dataset("slave_joint_feedback", data=_stack(samples, "slave_feedback"))
            if samples[0].master_feedback is not None:
                diagnostic.create_dataset("master_joint_feedback", data=_stack(samples, "master_feedback"))

        # Make sure the file reaches disk before it is atomically renamed.
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def validate_episode(path: str | Path, require_dual: bool = False) -> dict:
    path = Path(path)
    errors: list[str] = []
    with h5py.File(path, "r") as root:
        required = ["joint_action/vector", "endpose/left_endpose", "endpose/left_gripper"]
        for key in required:
            if key not in root:
                errors.append(f"missing dataset: {key}")
        if errors:
            return {"path": str(path), "valid": False, "errors": errors}

        action = root["joint_action/vector"][:]
        length = action.shape[0]
        if action.ndim != 2 or action.shape[1] not in {7, 14}:
            errors.append(f"action must be [T,7] or [T,14], got {action.shape}")
        if require_dual and action.shape[1] != 14:
            errors.append("current VLA requires a 14-dimensional dual-arm action")
        if not np.isfinite(action).all():
            errors.append("action contains NaN/Inf")

        dual_keys = ["endpose/right_endpose", "endpose/right_gripper"]
        if require_dual:
            for key in dual_keys:
                if key not in root:
                    errors.append(f"missing dual-arm dataset: {key}")

        for key in ["endpose/left_endpose", "endpose/left_gripper", *dual_keys]:
            if key in root:
                value = root[key][:]
                if value.shape[0] != length:
                    errors.append(f"length mismatch: {key} has {value.shape[0]}, action has {length}")
                if not np.isfinite(value).all():
                    errors.append(f"{key} contains NaN/Inf")

        if "observation" not in root or not list(root["observation"].keys()):
            errors.append("no camera observations found")
        else:
            for name in root["observation"].keys():
                key = f"observation/{name}/rgb"
                if key not in root:
                    errors.append(f"missing camera RGB: {key}")
                    continue
                ds = root[key]
                if ds.shape[0] != length:
                    errors.append(f"camera {name} length mismatch")
                    continue
                for index in sorted(set([0, length - 1])):
                    arr = np.frombuffer(ds[index], dtype=np.uint8)
                    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if image is None:
                        errors.append(f"camera {name} frame {index} cannot be decoded")

    return {
        "path": str(path),
        "valid": not errors,
        "errors": errors,
        "length": length,
        "action_dim": int(action.shape[1]),
    }
