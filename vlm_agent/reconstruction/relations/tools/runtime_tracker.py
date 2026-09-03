from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from relations.agents.state import write_json, write_md


class RuntimeTracker:
    """Track execution closure without token, cost, or timing accounting."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, Any]] = {}
        self.api_calls: list[dict[str, Any]] = []
        self.render_calls: list[dict[str, Any]] = []


    def start(self, stage_name: str) -> None:
        stage = self.stages.setdefault(
            stage_name, {"status": "running", "count": 0}
        )
        stage["status"] = "running"

    def end(
        self, stage_name: str, status: str = "success", **metadata: Any
    ) -> None:
        stage = self.stages.setdefault(
            stage_name, {"status": status, "count": 0}
        )
        stage["status"] = status
        stage["count"] = int(stage.get("count", 0)) + 1
        if metadata:
            stage.setdefault("metadata", {}).update(metadata)

    def mark_skipped(self, stage_name: str, reason: str | None = None) -> None:
        stage = self.stages.setdefault(
            stage_name, {"status": "skipped", "count": 0}
        )
        stage["status"] = "skipped"
        if reason:
            stage["reason"] = reason


    @contextmanager
    def stage(self, stage_name: str, **metadata: Any) -> Iterator[None]:
        self.start(stage_name)
        try:
            yield
        except Exception as exc:
            self.end(
                stage_name, status="failed", error=type(exc).__name__, **metadata
            )
            raise
        else:
            self.end(stage_name, status="success", **metadata)


    @contextmanager
    def api_call(self, stage_name: str, **metadata: Any) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            self.record_api_call(
                stage_name, success=False,
                error=type(exc).__name__, **metadata
            )
            raise
        else:
            self.record_api_call(stage_name, success=True, **metadata)

    def record_api_call(
        self, stage: str, success: bool = True, **metadata: Any
    ) -> None:
        item = {"stage": stage, "success": bool(success)}
        item.update(metadata)
        self.api_calls.append(item)
        self.record_stage_result(
            stage, status="success" if success else "failed"
        )

    def record_stage_result(
        self, stage: str, status: str = "success",
        **metadata: Any
    ) -> None:
        stage_data = self.stages.setdefault(
            stage, {"status": status, "count": 0}
        )
        stage_data["status"] = status
        stage_data["count"] = int(stage_data.get("count", 0)) + 1
        if metadata:
            stage_data.setdefault("metadata", {}).update(metadata)

    def record_render_call(
        self, stage: str, success: bool = True, **metadata: Any
    ) -> None:
        item = {"stage": stage, "success": bool(success)}
        item.update(metadata)
        self.render_calls.append(item)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "relation_execution_summary",
            "api_calls": len(self.api_calls),
            "render_calls": len(self.render_calls),
            "stages": self.stages,
        }

    def to_markdown_lines(self) -> list[str]:
        data = self.to_json()
        lines = [
            f"- api_calls: `{data['api_calls']}`",
            f"- render_calls: `{data['render_calls']}`",
            "",
            "## Stages",
        ]
        for name, stage in data["stages"].items():
            lines.append(
                f"- `{name}`: status=`{stage.get('status')}` "
                f"count=`{stage.get('count', 0)}`"
            )
        return lines

    def write(self, output_dir: Path) -> dict[str, Any]:
        data = self.to_json()
        json_path = output_dir / "debug" / "agents" / "runtime_summary.json"
        md_path = output_dir / "debug" / "agents" / "runtime_summary.md"
        write_json(json_path, data)
        write_md(md_path, "Execution Summary", self.to_markdown_lines())
        data["json_path"] = str(json_path)
        data["md_path"] = str(md_path)
        return data
