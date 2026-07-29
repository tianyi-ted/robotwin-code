#!/usr/bin/env python3
"""Preview and snapshot one configured camera without touching robot hardware."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cameras import build_cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.example.yaml"))
    parser.add_argument("--camera", default="head_camera", help="configured camera name")
    parser.add_argument("--source", help="temporarily override camera source, e.g. 0 or /dev/video2")
    parser.add_argument("--snapshot-dir", default=str(HERE / "camera_snapshots"))
    parser.add_argument("--dry-run", action="store_true", help="use a synthetic image source")
    parser.add_argument("--max-frames", type=int, help="exit after N frames (useful for headless tests)")
    parser.add_argument("--no-gui", action="store_true", help="capture without opening a window")
    parser.add_argument("--save-one", help="save one annotated frame to this path and exit")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    camera_cfgs = [dict(value) for value in config.get("cameras", [])]
    selected = [value for value in camera_cfgs if str(value.get("name")) == args.camera]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one enabled/configured camera named {args.camera!r}")
    selected[0]["enabled"] = True
    if args.source is not None:
        selected[0]["source"] = args.source

    camera = build_cameras(selected, args.dry_run)[args.camera]
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    count = 0
    camera.start()
    try:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                frame = camera.latest(max_age_s=0.5)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.03)

        print(f"Previewing {args.camera!r}. This program never opens CAN or enables motors.")
        if not args.no_gui:
            print("Keys: s=save RGB snapshot, q/Esc=quit")
        while True:
            frame = camera.latest(max_age_s=0.5)
            # Camera API returns RGB; OpenCV windows/files require BGR.
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            cv2.drawMarker(
                bgr,
                (width // 2, height // 2),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                max(20, min(width, height) // 12),
                2,
            )
            cv2.putText(
                bgr,
                f"{args.camera}  {width}x{height}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if args.save_one:
                output = Path(args.save_one).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), bgr):
                    raise RuntimeError(f"failed to save frame to {output}")
                print(f"Saved {output}")
                break
            key = -1
            if not args.no_gui:
                cv2.imshow(f"EDULITE camera preview - {args.camera}", bgr)
                key = cv2.waitKey(1) & 0xFF
            count += 1
            if key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = snapshot_dir / time.strftime(f"{args.camera}_%Y%m%d_%H%M%S.jpg")
                if not cv2.imwrite(str(path), bgr):
                    raise RuntimeError(f"failed to save snapshot to {path}")
                print(f"Saved {path}")
            if key in {ord("q"), 27} or (args.max_frames and count >= args.max_frames):
                break
            time.sleep(0.005)
        return 0
    finally:
        camera.stop()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
