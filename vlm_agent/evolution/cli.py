from __future__ import annotations

import argparse
import os
from pathlib import Path

from .orchestrator import PipelineOrchestrator
from .paths import ProjectPaths
from .tool_runner import ToolRunner
from .vlm_client import VLMClient


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM-based scene optimization framework")
    parser.add_argument("--project_root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional output directory name or path")
    parser.add_argument("--max_rounds", type=int, default=3, help="How many VLM optimization rounds to run")
    parser.add_argument("--log_path", type=Path, default=None, help="Optional path for JSON output")
    parser.add_argument("--model", type=str, default="gpt-5.5", help="VLM model name")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="VLM API key")
    parser.add_argument("--base_url", type=str, default="None", help="VLM base URL (optional)")
    parser.add_argument(
        "--collision_policy",
        type=str,
        choices=["strict", "repair"],
        default="strict",
        help="Policy for handling illegal objects in original scene: 'strict' to discard, 'repair' to attempt recovery"
    )
    return parser


def main_full_pipeline(args) -> int:
    """Optimize, generate, and compose an existing scene."""
    if args.api_key is None:
        raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY environment variable.")
    paths = ProjectPaths(args.project_root, output_dir=args.output_dir)
    tool_runner = ToolRunner(args.project_root)
    orchestrator = PipelineOrchestrator(
        paths=paths,
        tool_runner=tool_runner,
        vlm_client=VLMClient(
            url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        ),
        max_rounds=args.max_rounds,
        collision_policy=args.collision_policy,
    )

    report = orchestrator.run_full_pipeline()
    report_path = orchestrator.save_report(report, args.log_path)
    print(f"Pipeline completed. Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(main_full_pipeline(args))