from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.state import write_json, write_md


HYPOTHESES_JSON_NAME = "wall_object_hypotheses.json"
HYPOTHESES_MD_NAME = "wall_object_hypotheses.md"


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def extract_wall_object_hypotheses(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    likely_wall_objects: list[dict[str, Any]] = []
    wall_line_hypotheses: list[dict[str, Any]] = []
    source_tasks: list[str] = []

    for root in task_results:
        if not isinstance(root, dict):
            continue
        source_task_id = str(root.get("task_id") or "unknown_task")
        source_task_type = str(root.get("task_type") or "")
        root_is_hypothesis = source_task_type == "wall_object_hypothesis" or "wall_object_hypothesis" in source_task_id
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            object_items = _list_of_dicts(node.get("likely_wall_objects"))
            line_items = _list_of_dicts(node.get("wall_line_hypotheses"))
            if not object_items and not line_items and not root_is_hypothesis:
                continue
            if object_items or line_items:
                if source_task_id not in source_tasks:
                    source_tasks.append(source_task_id)
            for item in object_items:
                enriched = dict(item)
                enriched.setdefault("source_task_id", source_task_id)
                enriched.setdefault("hypothesis_role", "likely_wall_object")
                likely_wall_objects.append(enriched)
            for item in line_items:
                enriched = dict(item)
                enriched.setdefault("source_task_id", source_task_id)
                enriched.setdefault("source", item.get("source") or "wall_attached_object_distribution")
                enriched.setdefault("may_be_used_as_final_wall", False)
                wall_line_hypotheses.append(enriched)

    return {
        "available": bool(likely_wall_objects or wall_line_hypotheses),
        "likely_wall_objects_count": len(likely_wall_objects),
        "wall_line_hypotheses_count": len(wall_line_hypotheses),
        "source_task_ids": source_tasks,
        "likely_wall_objects": likely_wall_objects,
        "wall_line_hypotheses": wall_line_hypotheses,
        "final_wall_source_allowed": False,
        "coordinate_source_definition": "Wall object hypotheses are layout-XZ support evidence for final adjudication only; they are not final wall coordinates.",
        "notes": "These hypotheses may support visibility and wall-attach review in wall_final_adjudication, but final walls must still use obb_from_mesh_footprint_min_rect or obb_from_mesh_footprint_min_rect.",
    }


def write_wall_object_hypotheses(output_dir: Path, hypotheses: dict[str, Any]) -> dict[str, Any]:
    wall_debug_dir = output_dir / "debug" / "walls"
    json_path = wall_debug_dir / HYPOTHESES_JSON_NAME
    md_path = wall_debug_dir / HYPOTHESES_MD_NAME
    data = dict(hypotheses)
    data["json_path"] = str(json_path)
    data["md_path"] = str(md_path)
    write_json(json_path, data)

    lines = [
        f"- available: `{data.get('available')}`",
        f"- likely_wall_objects_count: `{data.get('likely_wall_objects_count')}`",
        f"- wall_line_hypotheses_count: `{data.get('wall_line_hypotheses_count')}`",
        f"- source_task_ids: `{data.get('source_task_ids', [])}`",
        "- final_wall_source_allowed: `False`",
        "- coordinate_source_definition: Wall object hypotheses support final adjudication only; they are not final walls.",
        "",
        "## Likely Wall Objects",
    ]
    for item in data.get("likely_wall_objects", []):
        lines.append(
            f"- `{item.get('object_id')}` role=`{item.get('role')}` side=`{item.get('candidate_wall_side')}` confidence=`{item.get('confidence')}`"
        )
    lines.extend(["", "## Wall Line Hypotheses"])
    for item in data.get("wall_line_hypotheses", []):
        line = item.get("estimated_line", {}) if isinstance(item.get("estimated_line"), dict) else {}
        lines.append(
            f"- `{item.get('hypothesis_id')}` candidate_wall_id=`{item.get('candidate_wall_id')}` line_type=`{item.get('line_type')}` "
            f"point_a=`{line.get('point_a')}` point_b=`{line.get('point_b')}` confidence=`{item.get('confidence')}` final_ok=`{item.get('may_be_used_as_final_wall', False)}`"
        )
    write_md(md_path, "Wall Object Hypotheses", lines)
    return data


def build_and_write_wall_object_hypotheses(output_dir: Path, task_results: list[dict[str, Any]]) -> dict[str, Any]:
    return write_wall_object_hypotheses(output_dir, extract_wall_object_hypotheses(task_results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and write wall object hypotheses from task results JSON.")
    parser.add_argument("--task_results", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    task_results = json.loads(args.task_results.read_text(encoding="utf-8"))
    if isinstance(task_results, dict) and isinstance(task_results.get("task_results"), list):
        task_results = task_results["task_results"]
    if not isinstance(task_results, list):
        raise SystemExit("task_results must be a list or an object containing task_results list")
    data = build_and_write_wall_object_hypotheses(args.output_dir, task_results)
    print(json.dumps({"json_path": data["json_path"], "wall_line_hypotheses_count": data["wall_line_hypotheses_count"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
