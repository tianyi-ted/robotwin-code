#!/usr/bin/env python3
"""Validate collected HDF5 episodes and paired instruction JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_io import validate_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="episode .hdf5 file or collection root")
    parser.add_argument("--require-dual", action="store_true",
                        help="enforce current RoboTwin VLA 14-action/16-state schema")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    paths = [root] if root.is_file() else sorted(root.rglob("*.hdf5"))
    if not paths:
        print(f"No HDF5 files found under {root}")
        return 2
    failed = 0
    for path in paths:
        result = validate_episode(path, require_dual=args.require_dual)
        instruction = path.parent.parent / "instructions" / f"{path.stem}.json"
        if not instruction.exists():
            result["valid"] = False
            result["errors"].append(f"missing paired instruction: {instruction}")
        else:
            try:
                data = json.loads(instruction.read_text(encoding="utf-8"))
                if not data.get("seen"):
                    raise ValueError("non-empty 'seen' list is required")
            except Exception as exc:
                result["valid"] = False
                result["errors"].append(f"bad instruction JSON: {exc}")
        state = "PASS" if result["valid"] else "FAIL"
        print(f"{state} {path}: T={result.get('length')} action_dim={result.get('action_dim')}")
        for error in result["errors"]:
            print(f"  - {error}")
        failed += int(not result["valid"])
    print(f"Checked {len(paths)} episode(s), failures={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
