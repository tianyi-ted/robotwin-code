"""Saved, headless visualizations for EDULITE-A3 VLA inference."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np


class InferenceVisualizer:
    """Persist previews, execution videos, CSV traces and comparison plots."""

    def __init__(self, config: dict, controller):
        cfg = config.get("visualization", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.root = Path(cfg.get("output_root", "./inference_outputs")).expanduser().resolve()
        self.save_video = bool(cfg.get("save_video", True))
        self.video_fps = float(cfg.get("video_fps", 30.0))
        self.controller = controller
        self.records: list[np.ndarray] = []
        self.run_dir: Path | None = None
        self.video = None
        self.started = 0.0
        self.frame_index = 0
        self.last_raw = np.full(14, np.nan, dtype=np.float32)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pyplot():
        cache = Path("/tmp/edulite-vla-matplotlib")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    @staticmethod
    def _stamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _unique_dir(self, prefix: str) -> Path:
        candidate = self.root / f"{prefix}_{self._stamp()}"
        suffix = 1
        while candidate.exists():
            candidate = self.root / f"{prefix}_{self._stamp()}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True)
        return candidate

    def save_preview(self, rgb: np.ndarray, chunk: np.ndarray) -> Path | None:
        """Save exactly what the model saw and what its 32-step chunk means."""

        if not self.enabled:
            return None
        import cv2

        preview_dir = self._unique_dir("preview")
        cv2.imwrite(
            str(preview_dir / "model_input_rgb.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        safe = np.stack(
            [self.controller.validate_model_action(action)[0] for action in chunk]
        )
        xyz = self.controller.predicted_xyz(safe)
        columns = [f"raw_action_{i}" for i in range(14)]
        columns += [f"safe_action_{i}" for i in range(7)]
        columns += ["fk_x_m", "fk_y_m", "fk_z_m"]
        np.savetxt(
            preview_dir / "preview_actions.csv",
            np.c_[chunk, safe, xyz],
            delimiter=",",
            header=",".join(columns),
            comments="",
        )

        plt = self._pyplot()
        fig = plt.figure(figsize=(16, 10))
        image_ax = fig.add_subplot(2, 2, 1)
        image_ax.imshow(rgb)
        image_ax.set_title("Exact RGB input seen by the VLA")
        image_ax.axis("off")

        joint_ax = fig.add_subplot(2, 2, 2)
        steps = np.arange(len(chunk))
        for joint in range(6):
            joint_ax.plot(steps, chunk[:, joint], "--", alpha=0.35)
            joint_ax.plot(steps, safe[:, joint], label=f"L{joint + 1}")
        joint_ax.set_title("Joints: dashed=raw, solid=safety-clipped")
        joint_ax.set_xlabel("chunk step")
        joint_ax.set_ylabel("rad")
        joint_ax.grid(True, alpha=0.3)
        joint_ax.legend(ncol=3, fontsize=8)

        grip_ax = fig.add_subplot(2, 2, 3)
        grip_ax.plot(steps, chunk[:, 6], "--", label="raw gripper")
        grip_ax.plot(steps, safe[:, 6], label="safe gripper")
        grip_ax.set_ylim(-0.1, 1.1)
        grip_ax.set_title("Gripper: 0=closed, 1=open")
        grip_ax.set_xlabel("chunk step")
        grip_ax.grid(True, alpha=0.3)
        grip_ax.legend()

        trajectory_ax = fig.add_subplot(2, 2, 4, projection="3d")
        trajectory_ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "o-")
        trajectory_ax.scatter(*xyz[0], color="green", s=70, label="start")
        trajectory_ax.scatter(*xyz[-1], color="red", s=70, label="end")
        trajectory_ax.set_xlabel("X (m)")
        trajectory_ax.set_ylabel("Y (m)")
        trajectory_ax.set_zlabel("Z (m)")
        trajectory_ax.set_title("Predicted end-effector path (robot base frame)")
        trajectory_ax.legend()
        fig.tight_layout()
        fig.savefig(preview_dir / "preview_summary.png", dpi=150)
        plt.close(fig)
        print(f"VISUAL PREVIEW SAVED: {preview_dir}", flush=True)
        return preview_dir

    def start_episode(self, rgb: np.ndarray) -> Path | None:
        if not self.enabled:
            return None
        self.finalize("restarted")
        self.run_dir = self._unique_dir("run")
        self.records = []
        self.started = time.monotonic()
        self.frame_index = 0
        self.last_raw.fill(np.nan)
        if self.save_video:
            import cv2

            height, width = rgb.shape[:2]
            video = cv2.VideoWriter(
                str(self.run_dir / "execution_overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
            )
            if video.isOpened():
                self.video = video
            else:
                video.release()
                print("WARNING: video writer unavailable; CSV/plots remain enabled")
        print(f"INFERENCE RECORDING: {self.run_dir}", flush=True)
        return self.run_dir

    def record(
        self,
        rgb: np.ndarray,
        raw_action: np.ndarray | None,
        telemetry: dict[str, Any],
        endpose_xyz: np.ndarray,
    ) -> None:
        if not self.enabled or self.run_dir is None:
            return
        available = raw_action is not None
        if available:
            self.last_raw = np.asarray(raw_action, dtype=np.float32).copy()
        elapsed = time.monotonic() - self.started
        row = np.r_[
            elapsed,
            float(available),
            self.last_raw,
            telemetry["model_target"],
            telemetry["sent"],
            telemetry["feedback"],
            np.asarray(endpose_xyz, dtype=np.float32),
            telemetry["gripper_torque_raw"],
            telemetry["gripper_torque_ema"],
            float(telemetry["gripper_force_latched"]),
        ]
        self.records.append(row.astype(np.float32))
        if self.video is not None:
            self._write_video_frame(rgb, available, elapsed, telemetry)
        self.frame_index += 1

    def _write_video_frame(
        self, rgb: np.ndarray, available: bool, elapsed: float, telemetry: dict[str, Any]
    ) -> None:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        target = telemetry["model_target"]
        feedback = telemetry["feedback"]
        lines = [
            f"t={elapsed:5.2f}s frame={self.frame_index} model={'YES' if available else 'WAIT'}",
            "target q=" + " ".join(f"{v:+.2f}" for v in target[:6]),
            "actual q=" + " ".join(f"{v:+.2f}" for v in feedback[:6]),
            f"grip raw/cmd/actual={target[6]:.3f}/{telemetry['sent'][6]:.3f}/"
            f"{feedback[6]:.3f} "
            f"torque={telemetry['gripper_torque_raw']:+.3f}Nm",
        ]
        for index, line in enumerate(lines):
            y = 25 + index * 24
            cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(
                bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
        self.video.write(bgr)

    def finalize(self, reason: str) -> Path | None:
        if self.run_dir is None:
            return None
        run_dir = self.run_dir
        if self.video is not None:
            self.video.release()
            self.video = None
        if not self.records:
            (run_dir / "EMPTY.txt").write_text(reason + "\n", encoding="utf-8")
            self.run_dir = None
            return run_dir

        data = np.stack(self.records)
        names = ["time_s", "new_model_action"]
        names += [f"raw_action_{i}" for i in range(14)]
        names += [f"safe_target_{i}" for i in range(7)]
        names += [f"filtered_command_{i}" for i in range(7)]
        names += [f"feedback_{i}" for i in range(7)]
        names += ["ee_x_m", "ee_y_m", "ee_z_m"]
        names += ["gripper_torque_raw_nm", "gripper_torque_ema_nm", "gripper_latched"]
        np.savetxt(
            run_dir / "execution_trace.csv",
            data,
            delimiter=",",
            header=",".join(names),
            comments="",
        )
        (run_dir / "result.txt").write_text(
            f"stop_reason={reason}\nframes={len(data)}\nduration_s={data[-1, 0]:.3f}\n",
            encoding="utf-8",
        )
        self._save_execution_plot(run_dir, data, reason)
        print(f"INFERENCE VISUALIZATION SAVED: {run_dir}", flush=True)
        self.run_dir = None
        self.records = []
        return run_dir

    def _save_execution_plot(self, run_dir: Path, data: np.ndarray, reason: str) -> None:
        plt = self._pyplot()
        fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
        axes = axes.ravel()
        t = data[:, 0]
        raw0, target0, command0, feedback0 = 2, 16, 23, 30
        for joint in range(6):
            ax = axes[joint]
            ax.plot(t, data[:, raw0 + joint], "--", alpha=0.4, label="raw model")
            ax.plot(t, data[:, target0 + joint], label="safe target")
            ax.plot(t, data[:, command0 + joint], label="filtered command")
            ax.plot(t, data[:, feedback0 + joint], label="feedback")
            ax.set_title(f"L{joint + 1}")
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)
        grip_ax = axes[6]
        grip_ax.plot(t, data[:, raw0 + 6], "--", alpha=0.4, label="raw model")
        grip_ax.plot(t, data[:, target0 + 6], label="safe target")
        grip_ax.plot(t, data[:, command0 + 6], label="filtered command")
        grip_ax.plot(t, data[:, feedback0 + 6], label="feedback")
        grip_ax.set_title("Gripper (0=closed, 1=open)")
        grip_ax.set_ylim(-0.1, 1.1)
        grip_ax.grid(True, alpha=0.3)
        error_ax = axes[7]
        error = np.max(
            np.abs(data[:, command0 : command0 + 6] - data[:, feedback0 : feedback0 + 6]),
            axis=1,
        )
        error_ax.plot(t, error, label="max |command-feedback|")
        error_ax.axhline(
            self.controller.follow_error_limit,
            color="red",
            linestyle="--",
            label="safety limit",
        )
        error_ax.set_title("Following error")
        error_ax.set_ylabel("rad")
        error_ax.grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel("time (s)")
        axes[0].legend(ncol=2, fontsize=8)
        grip_ax.legend(ncol=2, fontsize=8)
        error_ax.legend(fontsize=8)
        fig.suptitle(f"VLA execution trace — stop reason: {reason}")
        fig.tight_layout()
        fig.savefig(run_dir / "execution_summary.png", dpi=150)
        plt.close(fig)
