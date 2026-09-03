from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AUDIT_FILES = {
    "walls": ["wall_audit"],
    "ceiling": ["ceiling_audit"],
    "objects": ["object_grounding_audit", "object_revision_audit"],
    "attach": ["attach_decision_audit"],
    "rendering": ["topdown_render_log"],
    "agents": ["agent_task_log"],
    "supervisor": ["supervisor_log"],
    "final_validation": ["schema_validation"],
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    body.extend(lines)
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")


def ensure_output_tree(output_dir: Path) -> None:
    dirs = [
        "scene_states",
        "policy",
        "manager_plans",
        "agent_outputs/task_0001",
        "generated_evidence/topdown",
        "final_outputs",
    ]
    for rel in dirs:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)
    for group in AUDIT_FILES:
        (output_dir / "debug" / group).mkdir(parents=True, exist_ok=True)


def write_audit_pair(output_dir: Path, group: str, name: str, data: dict[str, Any], title: str | None = None) -> None:
    json_path = output_dir / "debug" / group / f"{name}.json"
    md_path = output_dir / "debug" / group / f"{name}.md"
    write_json(json_path, data)
    lines = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, indent=2, ensure_ascii=False)
            lines.append(f"## {key}")
            lines.append("")
            lines.append("```json")
            lines.append(rendered)
            lines.append("```")
        else:
            lines.append(f"- **{key}**: {value}")
    write_md(md_path, title or name.replace("_", " ").title(), lines)


def initialize_empty_audits(output_dir: Path, dry_run: bool) -> None:
    base = {
        "dry_run": dry_run,
        "status": "initialized",
        "note": "Placeholder audit created by the multi-agent scene delivery scaffold.",
        "items": [],
    }
    for group, names in AUDIT_FILES.items():
        for name in names:
            if group == "final_validation":
                continue
            write_audit_pair(output_dir, group, name, dict(base), title=name.replace("_", " ").title())


def sorted_object_id_key(object_id: str) -> tuple[int, str]:
    try:
        return (int(str(object_id).split("_")[-1]), str(object_id))
    except Exception:
        return (10**9, str(object_id))
