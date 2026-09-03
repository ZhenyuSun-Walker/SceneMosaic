#!/usr/bin/env python3
"""Run the public SceneMosaic pipeline from one image to final evolution."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGES = {
    "scene_018": "scene_018.png",
    "scene_031": "scene_031.png",
    "scene_092": "scene_092.png",
    "gaminglounge": "gaminglounge.png",
}
SCENE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SceneMosaic from an input image through perception, "
            "reconstruction, relation extraction, and scene evolution."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--example",
        choices=sorted(EXAMPLE_IMAGES),
        help="Use one of the repository's public example images.",
    )
    source.add_argument(
        "--input-image",
        type=Path,
        help="Use a custom input image instead of a bundled example.",
    )
    parser.add_argument(
        "--scene-name",
        required=True,
        help="New scene identifier; use a fresh value to start a new run.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "runtime.local.json",
        help=(
            "Secret-free runtime profile (default: "
            "configs/runtime.local.json)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not SCENE_NAME_RE.fullmatch(args.scene_name):
        raise SystemExit(
            "error: --scene-name may contain only letters, digits, '_' and '-'."
        )

    image = (
        REPOSITORY_ROOT / "examples" / "images" / EXAMPLE_IMAGES[args.example]
        if args.example
        else args.input_image
    )
    assert image is not None
    runtime_config = args.runtime_config
    if not image.is_file():
        raise SystemExit(f"error: input image is missing: {image}")
    if not runtime_config.is_file():
        raise SystemExit(
            "error: runtime configuration is missing: "
            f"{runtime_config}. Run python scripts/configure_runtime.py first."
        )

    command = [
        sys.executable,
        "-m",
        "vlm_agent.cli",
        "--runtime-config",
        str(runtime_config),
        "run",
        "--input-image",
        str(image),
        "--scene-name",
        args.scene_name,
    ]
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
