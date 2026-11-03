#!/usr/bin/env python3
"""Verify the published predictive EV/HEMS input bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "inputs/predictive/MANIFEST.json"


def main() -> int:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    failures: list[str] = []
    total_bytes = 0
    for record in records:
        path = REPO_ROOT / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
            continue
        total_bytes += path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            failures.append(f"changed: {record['path']}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(records)} predictive input files ({total_bytes:,} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
