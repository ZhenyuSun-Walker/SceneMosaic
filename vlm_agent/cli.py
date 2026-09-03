from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from .pipeline import (
    FullPipelineError,
    FullRunOptions,
    FullSceneOrchestrator,
)
from .config import RuntimeConfig, RuntimeConfigError


SECRET_ENV_RE = re.compile(
    r"(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION)", re.I
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SceneMosaic pipeline: image -> perception -> SAM3D -> "
            "relation structuring -> scene evolution"
        )
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help=(
            "Secret-free deployment profile; defaults to "
            "SCENEMOSAIC_RUNTIME_CONFIG or configs/runtime.local.json"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify", help="Read-only release, path and renderer verification"
    )
    verify.add_argument(
        "--require-gpu",
        action="store_true",
        help="Also verify that an NVIDIA GPU is visible; no model is loaded",
    )

    status = commands.add_parser(
        "status", help="Read-only status and immutable fact validation"
    )
    status.add_argument("--scene-root", type=Path, required=True)

    run = commands.add_parser(
        "run", help="Start from one image or resume a validated scene run"
    )
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-image", type=Path)
    source.add_argument("--scene-root", type=Path)
    run.add_argument(
        "--scene-name",
        help="Required with --input-image; ignored values are never inferred",
    )
    run.add_argument(
        "--run-root",
        type=Path,
        help="Optional new run root below configured workspace/runs",
    )
    run.add_argument(
        "--perception-reuse-local-pre-pass-b-cache-root",
        type=Path,
        help=(
            "Reuse one validated failed perception attempt below the trusted "
            "attempts root without reissuing Pass A or SAM3-A"
        ),
    )
    run.add_argument(
        "--perception-pass-b-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
    )
    return parser


def _redact(value: str) -> str:
    result = str(value)
    for name, secret in os.environ.items():
        if SECRET_ENV_RE.search(name) and len(secret) >= 4:
            result = result.replace(secret, "<redacted>")
    return result


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        runtime = RuntimeConfig.load(args.runtime_config)
        orchestrator = FullSceneOrchestrator(runtime)
        if args.command == "verify":
            report = orchestrator.verify(require_gpu=args.require_gpu)
            _print(report)
            return 0 if report.get("valid") else 4
        if args.command == "status":
            report = orchestrator.status(args.scene_root)
            _print(report)
            return 0 if report.get("facts_valid") else 4
        if args.input_image is not None and not args.scene_name:
            raise FullPipelineError(
                "--input-image requires an explicit --scene-name"
            )
        if args.scene_root is not None and args.scene_name is not None:
            raise FullPipelineError(
                "--scene-name is valid only with --input-image"
            )
        report = orchestrator.run(
            FullRunOptions(
                input_image=args.input_image,
                scene_name=args.scene_name,
                scene_root=args.scene_root,
                run_root=args.run_root,
                perception_reuse_local_pre_pass_b_cache_root=(
                    args.perception_reuse_local_pre_pass_b_cache_root
                ),
                perception_pass_b_reasoning_effort=(
                    args.perception_pass_b_reasoning_effort
                ),
            ),
            sanitized_argv=raw_argv,
        )
        _print(report)
        return 0
    except RuntimeConfigError as exc:
        parser.exit(2, f"error: {_redact(str(exc))}\n")
    except FullPipelineError as exc:
        parser.exit(exc.exit_code, f"error: {_redact(str(exc))}\n")
    except Exception as exc:
        parser.exit(4, f"error: {_redact(str(exc))}\n")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
