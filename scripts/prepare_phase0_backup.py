"""Create and validate a Phase 0 backup of local SCT Camera data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"} or path.name.endswith((".yaml.disabled", ".yml.disabled")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return {"format": "yaml", "records": 1}
    if suffix == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object: {path}")
                count += 1
        return {"format": "jsonl", "records": count}
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            required = {"event_id", "label", "notes"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"CSV missing columns {sorted(required)}: {path}")
        return {"format": "csv", "records": len(rows)}
    raise ValueError(f"Unsupported backup file: {path}")


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (args.output or root / "data/backups" / f"phase0_{timestamp}").resolve()
    sources = [root / "config/settings.yaml"]
    sources.extend(sorted((root / "config/cameras").glob("*.yaml*")))
    sources.extend(
        path
        for path in (root / "data/behavior_events.jsonl", root / "data/behavior_labels.csv")
        if path.exists()
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "files": [],
    }
    for source in sources:
        relative = source.relative_to(root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        validation = _validate(destination)
        manifest["files"].append(
            {
                "path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "validation": validation,
            }
        )

    manifest_path = output / "backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Backup valid: {output}")
    print(f"Files: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
