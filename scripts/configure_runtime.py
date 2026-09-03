#!/usr/bin/env python3
"""Create the machine-local runtime profile from installed Conda environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "runtime.example.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "runtime.local.json"


class ConfigurationError(RuntimeError):
    """Raised when an installed environment cannot be resolved safely."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{label} must contain a JSON object: {path}")
    return payload


def _find_conda_executable() -> Path:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        shutil.which("mamba"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise ConfigurationError("conda or mamba executable was not found")


def _list_environment_prefixes(executable: Path) -> tuple[Path, ...]:
    try:
        completed = subprocess.run(
            [str(executable), "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError("cannot query Conda environments") from exc
    if completed.returncode != 0:
        raise ConfigurationError("conda env list --json failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("conda env list returned invalid JSON") from exc
    raw_prefixes = payload.get("envs") if isinstance(payload, dict) else None
    if not isinstance(raw_prefixes, list):
        raise ConfigurationError("conda env list returned no environment list")
    return tuple(
        Path(item).expanduser().resolve()
        for item in raw_prefixes
        if isinstance(item, str) and item
    )


def _python_from_prefix(prefix: Path, label: str) -> Path:
    resolved_prefix = prefix.expanduser().resolve()
    python = (resolved_prefix / "bin" / "python").resolve()
    if (
        not resolved_prefix.is_dir()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ConfigurationError(
            f"{label} is not a usable Conda environment: {resolved_prefix}"
        )
    return python


def _resolve_named_environment(
    prefixes: tuple[Path, ...],
    name: str,
    label: str,
) -> Path:
    normalized = name.strip()
    if not normalized or "/" in normalized or "\\" in normalized:
        raise ConfigurationError(f"{label} environment name is invalid")
    matches = [prefix for prefix in prefixes if prefix.name == normalized]
    if not matches:
        raise ConfigurationError(
            f"Conda environment {normalized!r} was not found for {label}"
        )
    if len(matches) > 1:
        raise ConfigurationError(
            f"multiple Conda environments named {normalized!r}; use --{label}-prefix"
        )
    return _python_from_prefix(matches[0], label)


def _atomic_write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.parent != (PROJECT_ROOT / "configs").resolve():
        raise ConfigurationError("runtime profile must be written below configs/")
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise ConfigurationError(
                f"runtime profile already exists: {path}; pass --overwrite"
            )
        if path.is_symlink() or not path.is_file():
            raise ConfigurationError(
                f"refusing to overwrite unsafe runtime profile path: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve Conda environments and create configs/runtime.local.json"
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sam3 = parser.add_mutually_exclusive_group()
    sam3.add_argument("--sam3-env")
    sam3.add_argument("--sam3-prefix", type=Path)
    scenemosaic = parser.add_mutually_exclusive_group()
    scenemosaic.add_argument("--scenemosaic-env")
    scenemosaic.add_argument("--scenemosaic-prefix", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    template = _read_object(args.template.expanduser().resolve(), "runtime template")
    conda_envs = template.pop("conda_envs", None)
    if not isinstance(conda_envs, dict):
        raise ConfigurationError("runtime template is missing conda_envs")

    sam3_name = args.sam3_env or conda_envs.get("perception")
    sam3d_name = args.scenemosaic_env or conda_envs.get("sam3d")
    prefixes: tuple[Path, ...] = ()
    if args.sam3_prefix is None or args.scenemosaic_prefix is None:
        prefixes = _list_environment_prefixes(_find_conda_executable())

    perception_python = (
        _python_from_prefix(args.sam3_prefix, "sam3")
        if args.sam3_prefix is not None
        else _resolve_named_environment(prefixes, str(sam3_name or ""), "sam3")
    )
    sam3d_python = (
        _python_from_prefix(args.scenemosaic_prefix, "scenemosaic")
        if args.scenemosaic_prefix is not None
        else _resolve_named_environment(
            prefixes, str(sam3d_name or ""), "scenemosaic"
        )
    )

    template.update(
        {
            "perception_python": str(perception_python),
            "sam3d_python": str(sam3d_python),
            "relation_precompute_python": str(sam3d_python),
            "relation_runner_python": str(sam3d_python),
            "scenemosaic_python": str(sam3d_python),
        }
    )
    _atomic_write(args.output, template, args.overwrite)
    print(f"Wrote runtime profile: {args.output.expanduser().resolve()}")
    print(f"sam3 Python: {perception_python}")
    print(f"scenemosaic Python: {sam3d_python}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigurationError as exc:
        raise SystemExit(f"configuration failed: {exc}") from None
