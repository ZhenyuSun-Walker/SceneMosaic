from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(paths: Sequence[str | Path]) -> dict[str, str]:
    """Hash every regular file below the supplied paths."""

    snapshot: dict[str, str] = {}
    for raw_root in paths:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            snapshot[str(root)] = "<missing>"
            continue
        if root.is_file():
            snapshot[str(root)] = sha256_file(root)
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            snapshot[str(path.resolve())] = sha256_file(path)
    return snapshot


def compare_snapshots(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    changed = sorted(
        key for key in before_keys & after_keys if before[key] != after[key]
    )
    return {
        "unchanged": before == after,
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": changed,
        "before_file_count": len(before),
        "after_file_count": len(after),
    }
