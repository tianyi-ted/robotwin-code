"""Timestamped camera capture for the EDULITE-A3 dataset collector."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    monotonic_time: float


class OpenCVCamera:
    """Continuously reads a V4L2/file camera and exposes the newest RGB frame."""

    def __init__(self, name: str, source: Any, width: int, height: int, fps: float):
        self.name = name
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._capture: cv2.VideoCapture | None = None
        self._frame: CameraFrame | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"camera {self.name!r} cannot open source {self.source!r}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"camera-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._capture is not None
        while self._running:
            ok, bgr = self._capture.read()
            if not ok:
                self._error = RuntimeError(f"camera {self.name!r} read failed")
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = CameraFrame(np.ascontiguousarray(rgb), time.monotonic())
            with self._lock:
                self._frame = frame

    def latest(self, max_age_s: float) -> CameraFrame:
        if self._error is not None and self._frame is None:
            raise self._error
        with self._lock:
            frame = self._frame
        if frame is None:
            raise RuntimeError(f"camera {self.name!r} has not produced a frame")
        age = time.monotonic() - frame.monotonic_time
        if age > max_age_s:
            raise RuntimeError(
                f"camera {self.name!r} frame is stale: {age:.3f}s > {max_age_s:.3f}s"
            )
        return CameraFrame(frame.rgb.copy(), frame.monotonic_time)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class SyntheticCamera:
    """Deterministic RGB source used by dry-run and offline tests."""

    def __init__(self, name: str, width: int, height: int, fps: float):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._start = time.monotonic()

    def start(self) -> None:
        self._start = time.monotonic()

    def latest(self, max_age_s: float) -> CameraFrame:
        del max_age_s
        t = time.monotonic() - self._start
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = int((40.0 * t) % 255)
        image[..., 1] = np.linspace(0, 255, self.width, dtype=np.uint8)[None, :]
        image[..., 2] = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
        cv2.putText(
            image,
            f"dry-run {self.name} {t:5.2f}s",
            (12, min(40, self.height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return CameraFrame(image, time.monotonic())

    def stop(self) -> None:
        return


def build_cameras(configs: list[dict], dry_run: bool):
    cameras = {}
    for cfg in configs:
        if not cfg.get("enabled", True):
            continue
        name = str(cfg["name"])
        if name in cameras:
            raise ValueError(f"duplicate camera name: {name}")
        cls = SyntheticCamera if dry_run else OpenCVCamera
        kwargs = {
            "name": name,
            "width": int(cfg.get("width", 640)),
            "height": int(cfg.get("height", 480)),
            "fps": float(cfg.get("fps", 30.0)),
        }
        if not dry_run:
            kwargs["source"] = cfg.get("source", 0)
        cameras[name] = cls(**kwargs)
    if not cameras:
        raise ValueError("at least one enabled camera is required")
    return cameras
