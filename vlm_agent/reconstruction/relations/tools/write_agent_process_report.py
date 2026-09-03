from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.schema_guard import wall_axis_alignment_summary
from relations.agents.state import sorted_object_id_key, write_json, write_md


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_or(value: Any, path: Path | None, default: Any) -> Any:
    if value is not None:
        return value
    loaded = _load_json(path)
    return loaded if loaded is not None else default


def _object_ids(scene_state: dict[str, Any]) -> list[str]:
    items = scene_state.get("object_candidates", {}).get("items", []) if isinstance(scene_state, dict) else []
    ids = [str(item.get("object_id")) for item in items if isinstance(item, dict) and item.get("object_id")]
    return sorted(ids, key=sorted_object_id_key)


def _task_purpose(task: dict[str, Any]) -> str:
    for key in ["purpose", "instructions", "reason", "objective"]:
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")[:220]
    return ""


def _task_summary(manager_plan: dict[str, Any], api_called: bool | None) -> list[dict[str, Any]]:
    tasks = manager_plan.get("tasks", []) if isinstance(manager_plan, dict) else []
    result = []
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, dict):
            continue
        result.append(
            {
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "purpose": _task_purpose(task),
                "api_used": bool(api_called),
                "vision_used": bool(task.get("use_vision_model") or task.get("wall_visual_image_paths") or "wall" in str(task.get("task_type", "")).lower()),
            }
        )
    return result


def _count_relations(relations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"floor": 0, "ceiling": 0, "wall": 0, "object": 0}
    for rel in relations if isinstance(relations, list) else []:
        if not isinstance(rel, dict) or rel.get("relation") != "attach":
            continue
        anchor = str(rel.get("parameters", {}).get("object_anchor"))
        if anchor == "floor":
            counts["floor"] += 1
        elif anchor == "ceiling":
            counts["ceiling"] += 1
        elif anchor.startswith("wall_"):
            counts["wall"] += 1
        elif anchor.startswith("object_"):
            counts["object"] += 1
    return counts


def _final_relations(output_dir: Path) -> list[dict[str, Any]]:
    relations = _load_json(output_dir / "final_outputs" / "relations.json")
    return relations if isinstance(relations, list) else []


def _final_walls(output_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(output_dir / "final_outputs" / "objects.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("walls"), list):
        return []
    return [wall for wall in payload["walls"] if isinstance(wall, dict)]


def _glob_first(output_dir: Path, pattern: str) -> Path | None:
    matches = sorted(output_dir.glob(pattern))
    return matches[0] if matches else None


def write_final_process_report(
    output_dir: Path,
    scene_state: dict[str, Any] | None = None,
    manager_plan: dict[str, Any] | None = None,
    task_results: list[dict[str, Any]] | None = None,
    validation: dict[str, Any] | None = None,
    wall_evidence: dict[str, Any] | None = None,
    wall_visual_evidence: dict[str, Any] | None = None,
    final_wall_overlay: dict[str, Any] | None = None,
    final_wall_overlay_skipped_reason: str | None = None,
    api_called: bool | None = None,
    api_call_count: int | None = None,
    finalization_status: str | None = None,
    runtime_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    scene_state_path = output_dir / "scene_states" / "scene_state.json"
    manager_plan_path = output_dir / "manager_plans" / "manager_plan_000.json"
    validation_path = output_dir / "debug" / "final_validation" / "schema_validation.json"
    wall_evidence_path = output_dir / "debug" / "walls" / "wall_evidence.json"
    wall_visual_manifest_path = _glob_first(output_dir, "debug/walls/visual_review/*_wall_visual_review_manifest.json")
    wall_audit_path = output_dir / "debug" / "walls" / "wall_audit.json"
    wall_object_hypotheses_path = output_dir / "debug" / "walls" / "wall_object_hypotheses.json"
    attach_audit_path = output_dir / "debug" / "attach" / "attach_decision_audit.json"
    coverage_audit_path = output_dir / "debug" / "attach" / "coverage_repair_audit.json"
    accepted_merge_path = output_dir / "debug" / "attach" / "accepted_attach_merge_check.json"
    runtime_summary_path = output_dir / "debug" / "agents" / "runtime_summary.json"

    scene_state = _load_or(scene_state, scene_state_path, {})
    manager_plan = _load_or(manager_plan, manager_plan_path, {})
    validation = _load_or(validation, validation_path, {})
    wall_evidence = _load_or(wall_evidence, wall_evidence_path, {})
    wall_visual_evidence = _load_or(wall_visual_evidence, wall_visual_manifest_path, {})
    wall_audit = _load_json(wall_audit_path) or {}
    wall_object_hypotheses = _load_json(wall_object_hypotheses_path) or {}
    attach_audit = _load_json(attach_audit_path) or {}
    coverage_audit = _load_json(coverage_audit_path) or {}
    accepted_merge = _load_json(accepted_merge_path) or {}
    runtime_summary = runtime_summary or _load_json(runtime_summary_path) or {}

    ids = _object_ids(scene_state)
    relation_counts = _count_relations(_final_relations(output_dir))
    final_walls = _final_walls(output_dir)
    tasks = _task_summary(manager_plan, api_called)
    candidate_walls = wall_evidence.get("candidate_walls", []) if isinstance(wall_evidence, dict) else []
    candidate_sources = sorted({str(item.get("source")) for item in candidate_walls if isinstance(item, dict) and item.get("source")})
    candidate_trust = sorted({str(item.get("trust_level")) for item in candidate_walls if isinstance(item, dict) and item.get("trust_level")})
    wall_adjudication = validation.get("wall_final_adjudication") or wall_audit.get("wall_final_adjudication") or {}
    promotion = wall_audit.get("candidate_wall_promotion") or {}
    weak_wall_completion = wall_audit.get("weak_wall_completion") or validation.get("weak_wall_completion") or {}
    candidate_modified = bool(wall_adjudication.get("candidate_modified") or promotion.get("candidate_modified"))
    revision_reasons = wall_adjudication.get("revision_reasons") or promotion.get("revision_reasons") or []
    overlay_png = None
    if isinstance(final_wall_overlay, dict):
        overlay_png = final_wall_overlay.get("output_overlay_png") or final_wall_overlay.get("output")
    overlay_png = overlay_png or str(output_dir / "debug" / "walls" / "final_walls_overlay_on_canonical_topdown.png")
    overlay_exists = Path(overlay_png).exists() if overlay_png else False
    if overlay_exists:
        final_wall_overlay_skipped_reason = None
    elif not final_wall_overlay_skipped_reason:
        final_wall_overlay_skipped_reason = "final wall overlay skipped because final walls were not generated"

    objects_path = output_dir / "final_outputs" / "objects.json"
    relations_path = output_dir / "final_outputs" / "relations.json"
    objects_valid = validation.get("objects", {}).get("valid") if isinstance(validation.get("objects"), dict) else None
    relations_valid = validation.get("relations", {}).get("valid") if isinstance(validation.get("relations"), dict) else None
    target_coverage = validation.get("relations", {}).get("target_coverage", {}) if isinstance(validation.get("relations"), dict) else {}
    wall_geometry = validation.get("objects", {}).get("wall_geometry") if isinstance(validation.get("objects"), dict) else wall_audit.get("geometry")
    wall_geometry_details = wall_geometry.get("details", {}) if isinstance(wall_geometry, dict) and isinstance(wall_geometry.get("details"), dict) else {}
    orientation_summary = wall_axis_alignment_summary(final_walls) if final_walls else {}
    axis_aligned_final_walls = wall_geometry_details.get("axis_aligned", orientation_summary.get("axis_aligned"))
    oriented_boundary_inferred = wall_geometry_details.get("oriented_boundary", orientation_summary.get("oriented_boundary"))
    edge_orientations = wall_geometry_details.get("edge_orientations", orientation_summary.get("edge_orientations", []))
    wall_object_hypotheses_influenced_orientation = bool(
        wall_object_hypotheses.get("available") and (oriented_boundary_inferred or candidate_modified)
    )

    report = {
        "scene_summary": {
            "scene_name": scene_state.get("scene_name"),
            "object_candidate_count": len(ids),
            "object_id_range": [ids[0], ids[-1]] if ids else [],
            "output_dir": str(output_dir),
        },
        "evidence_preparation": {
            "scene_state_path": str(scene_state_path),
            "scene_state_exists": scene_state_path.exists(),
            "canonical_topdown_path": wall_visual_evidence.get("canonical_topdown_path") or scene_state.get("topdown_grid_mapping", {}).get("canonical_topdown_image_path"),
            "canonical_grid_mapping_path": wall_visual_evidence.get("canonical_grid_mapping_path") or scene_state.get("topdown_grid_mapping", {}).get("canonical_mapping_path"),
            "fixed_layout_views_manifest": scene_state.get("evidence", {}).get("fixed_layout_views", {}).get("manifest_path"),
            "wall_evidence_path": str(wall_evidence_path),
            "wall_visual_contact_sheet": wall_visual_evidence.get("contact_sheet_path"),
            "original_image_path": wall_visual_evidence.get("original_image_path") or scene_state.get("evidence", {}).get("original_image_path"),
            "mask_overlay_path": wall_visual_evidence.get("mask_overlay_path") or scene_state.get("evidence", {}).get("mask_overlay_path"),
        },
        "manager_plan": {"task_count": len(tasks), "tasks": tasks},
        "wall_reasoning": {
            "candidate_walls_count": len(candidate_walls) if isinstance(candidate_walls, list) else 0,
            "candidate_walls_source": candidate_sources,
            "candidate_walls_trust_level": candidate_trust,
            "promotion_rubric": {
                "object_occupied_bounds": "interior_occupancy_evidence_not_wall_boundary",
                "room_wall_boundary": "may_be_larger_than_object_extents_and_enclose_empty_floor_area",
                "camera_grid_extent": "reject_only_when_no_visual_room_boundary_support_or_excessive_unrelated_padding",
            },
            "wall_final_adjudication": wall_adjudication,
            "decision": wall_adjudication.get("decision"),
            "candidate_modified": candidate_modified,
            "revision_reasons": revision_reasons,
            "candidate_promotion": promotion,
            "axis_aligned_final_walls": axis_aligned_final_walls,
            "oriented_boundary_inferred": oriented_boundary_inferred,
            "edge_orientations": edge_orientations,
            "wall_object_hypotheses_influenced_orientation": wall_object_hypotheses_influenced_orientation,
            "grid_camera_extent_usage": "weak_reference_only; reject when it is only a loose extent rectangle and does not follow the actual room footprint",
            "weak_wall_completion": weak_wall_completion,
            "weak_wall_completion_used": bool(weak_wall_completion.get("used") or weak_wall_completion.get("weak_wall_completion_used")),
            "weak_wall_ids": weak_wall_completion.get("weak_wall_ids", []),
            "weak_wall_completion_methods": weak_wall_completion.get("completion_methods", []),
            "weak_wall_completion_reasons": weak_wall_completion.get("completion_reasons", []),
            "oriented_boundary_rule": "Axis-aligned walls remain allowed when they match the true room footprint; oblique walls are allowed and preferred when the footprint/object layout is oblique.",
            "final_wall_source": "obb_from_mesh_footprint_min_rect" if promotion.get("promotion_from_candidate") or wall_adjudication.get("accepted_final_walls") else None,
            "overlay_png": overlay_png if overlay_exists else None,
            "overlay_skipped_reason": final_wall_overlay_skipped_reason,
        },
        "wall_object_hypotheses": {
            "available": bool(wall_object_hypotheses.get("available")),
            "likely_wall_objects_count": int(wall_object_hypotheses.get("likely_wall_objects_count") or 0),
            "wall_line_hypotheses_count": int(wall_object_hypotheses.get("wall_line_hypotheses_count") or 0),
            "source_task_ids": wall_object_hypotheses.get("source_task_ids", []),
            "used_in_final_adjudication": bool(wall_object_hypotheses.get("available") and wall_adjudication.get("executed")),
            "likely_wall_objects": wall_object_hypotheses.get("likely_wall_objects", []),
            "wall_line_hypotheses": wall_object_hypotheses.get("wall_line_hypotheses", []),
            "json_path": str(wall_object_hypotheses_path),
        },
        "attach_reasoning": {
            "floor_attach_count": relation_counts["floor"],
            "ceiling_attach_count": relation_counts["ceiling"],
            "ceiling_confirmed": bool(validation.get("ceiling_confirmed") or attach_audit.get("ceiling_confirmed")),
            "wall_attach_count": relation_counts["wall"],
            "object_to_object_attach_count": relation_counts["object"],
            "accepted_candidate_merge_valid": accepted_merge.get("valid"),
            "accepted_candidate_merge_items": len(accepted_merge.get("items", [])) if isinstance(accepted_merge.get("items"), list) else 0,
            "skipped_candidate_count": len(attach_audit.get("skipped_candidates", [])) if isinstance(attach_audit.get("skipped_candidates"), list) else 0,
            "common_skip_reasons": sorted({str(item.get("reason")) for item in attach_audit.get("skipped_candidates", []) if isinstance(item, dict) and item.get("reason")})[:10],
        },
        "coverage_repair": {
            "missing_before": coverage_audit.get("missing_before", []),
            "missing_after": coverage_audit.get("missing_after", []),
            "coverage_repair_used": bool(coverage_audit.get("coverage_repair_used")),
            "coverage_repair_count": int(coverage_audit.get("coverage_repair_count") or 0),
            "missing_object_count_before_repair": len(coverage_audit.get("missing_before", [])) if isinstance(coverage_audit.get("missing_before"), list) else 0,
            "items": coverage_audit.get("items", []),
            "coverage_repair_audit_path": str(coverage_audit_path),
        },
        "final_validation": {
            "objects_json_generated": objects_path.exists(),
            "relations_json_generated": relations_path.exists(),
            "schema_validation_valid": bool(validation.get("valid")),
            "objects_valid": objects_valid,
            "relations_valid": relations_valid,
            "wall_geometry_valid": wall_geometry.get("valid") if isinstance(wall_geometry, dict) else None,
            "object_target_coverage_complete": target_coverage.get("complete") if isinstance(target_coverage, dict) else None,
            "accepted_attach_merge_valid": accepted_merge.get("valid"),
            "finalization_status": finalization_status or validation.get("finalization_status"),
            "errors": validation.get("errors", []),
        },
        "important_debug_files": {
            "final_objects": str(objects_path),
            "final_relations": str(relations_path),
            "schema_validation_md": str(output_dir / "debug" / "final_validation" / "schema_validation.md"),
            "wall_audit_md": str(output_dir / "debug" / "walls" / "wall_audit.md"),
            "wall_object_hypotheses_md": str(output_dir / "debug" / "walls" / "wall_object_hypotheses.md"),
            "final_wall_overlay": str(output_dir / "debug" / "walls" / "final_walls_overlay_on_canonical_topdown.png"),
            "coverage_repair_audit_md": str(output_dir / "debug" / "attach" / "coverage_repair_audit.md"),
            "accepted_attach_merge_check_md": str(output_dir / "debug" / "attach" / "accepted_attach_merge_check.md"),
            "runtime_summary_json": str(runtime_summary_path),
            "runtime_summary_md": str(output_dir / "debug" / "agents" / "runtime_summary.md"),
            "manager_plan": str(manager_plan_path),
            "agent_outputs": str(output_dir / "agent_outputs"),
        },
        "human_review_notes": [
            "Check whether final_walls_overlay_on_canonical_topdown.png fits the true room boundary.",
            "Review whether wall attach relations are physically plausible after final wall ids are stable.",
            "Review coverage repair anchors for plausible physical support.",
            "Review object descriptions for obvious image or mask conflicts.",
        ],
        "runtime": {"api_called": bool(api_called), "api_call_count": api_call_count, "summary": runtime_summary},
    }

    lines = [
        "## Execution Summary",
        f"- api_calls: `{runtime_summary.get('api_calls', api_call_count)}`",
        f"- render_calls: `{runtime_summary.get('render_calls')}`",
        f"- runtime_summary_json: `{runtime_summary_path}`",
        "",
        "## 1. Scene Summary",
        f"- scene_name: `{report['scene_summary']['scene_name']}`",
        f"- object_candidate_count: `{report['scene_summary']['object_candidate_count']}`",
        f"- object_id_range: `{report['scene_summary']['object_id_range']}`",
        f"- output_dir: `{output_dir}`",
        "",
        "## 2. Evidence Preparation",
    ]
    for key, value in report["evidence_preparation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## 3. Manager Plan"])
    for task in tasks:
        lines.append(f"- `{task.get('task_id')}` {task.get('task_type')}: API=`{task.get('api_used')}` vision=`{task.get('vision_used')}` purpose={task.get('purpose')}")
    if not tasks:
        lines.append("- No manager tasks recorded.")
    lines.extend([
        "",
        "## 4. Wall Reasoning",
        f"- candidate_walls_count: `{report['wall_reasoning']['candidate_walls_count']}`",
        f"- candidate_walls_source: `{candidate_sources}`",
        f"- candidate_walls_trust_level: `{candidate_trust}`",
        "- object_occupied_bounds: `interior_occupancy_evidence_not_wall_boundary`",
        "- room_wall_boundary: `may_be_larger_than_object_extents_and_enclose_empty_floor_area`",
        "- camera_grid_extent: `reject_only_when_no_visual_room_boundary_support_or_excessive_unrelated_padding`",
        "- rubric_note: Candidate wall promotion judges plausible room boundary alignment, not tight fit to object occupied bounds.",
        "- oriented_boundary_rule: Axis-aligned walls are allowed if they match the true room footprint; oblique walls are allowed and preferred if the room footprint/object layout is oblique.",
        f"- axis_aligned_final_walls: `{report['wall_reasoning']['axis_aligned_final_walls']}`",
        f"- oriented_boundary_inferred: `{report['wall_reasoning']['oriented_boundary_inferred']}`",
        f"- edge_orientations: `{report['wall_reasoning']['edge_orientations']}`",
        f"- wall_object_hypotheses_influenced_orientation: `{report['wall_reasoning']['wall_object_hypotheses_influenced_orientation']}`",
        f"- grid_camera_extent_usage: `{report['wall_reasoning']['grid_camera_extent_usage']}`",
        f"- weak_wall_completion_used: `{report['wall_reasoning']['weak_wall_completion_used']}`",
        f"- weak_wall_ids: `{report['wall_reasoning']['weak_wall_ids']}`",
        f"- weak_wall_completion_methods: `{report['wall_reasoning']['weak_wall_completion_methods']}`",
        f"- wall_final_adjudication: `{wall_adjudication}`",
        f"- candidate_promotion: `{promotion}`",
        f"- final_wall_source: `{report['wall_reasoning']['final_wall_source']}`",
        f"- overlay_png: `{report['wall_reasoning']['overlay_png']}`",
        f"- overlay_skipped_reason: `{report['wall_reasoning']['overlay_skipped_reason']}`",
        "",
        "## Wall Object Hypotheses",
        f"- available: `{report['wall_object_hypotheses']['available']}`",
        f"- likely_wall_objects_count: `{report['wall_object_hypotheses']['likely_wall_objects_count']}`",
        f"- wall_line_hypotheses_count: `{report['wall_object_hypotheses']['wall_line_hypotheses_count']}`",
        f"- source_task_ids: `{report['wall_object_hypotheses']['source_task_ids']}`",
        f"- used_in_final_adjudication: `{report['wall_object_hypotheses']['used_in_final_adjudication']}`",
        f"- json_path: `{wall_object_hypotheses_path}`",
        "",
        "## Hidden / Weak Wall Completion",
        f"- used: `{report['wall_reasoning']['weak_wall_completion_used']}`",
        f"- weak_wall_ids: `{report['wall_reasoning']['weak_wall_ids']}`",
        f"- completion_methods: `{report['wall_reasoning']['weak_wall_completion_methods']}`",
        f"- completion_reasons: `{report['wall_reasoning']['weak_wall_completion_reasons']}`",
        "",
        "## Wall Final Adjudication",
        f"- decision: `{wall_adjudication.get('decision')}`",
        f"- candidate_modified: `{candidate_modified}`",
        f"- revision_reasons: `{revision_reasons}`",
        f"- final_walls_accepted: `{wall_adjudication.get('accepted_final_walls')}`",
        "",
        "## 5. Attach Reasoning",
        f"- floor_attach_count: `{relation_counts['floor']}`",
        f"- ceiling_confirmed: `{report['attach_reasoning']['ceiling_confirmed']}`",
        f"- ceiling_attach_count: `{relation_counts['ceiling']}`",
        f"- wall_attach_count: `{relation_counts['wall']}`",
        f"- object_to_object_attach_count: `{relation_counts['object']}`",
        f"- accepted_candidate_merge_valid: `{accepted_merge.get('valid')}`",
        f"- skipped_candidate_count: `{report['attach_reasoning']['skipped_candidate_count']}`",
        f"- common_skip_reasons: `{report['attach_reasoning']['common_skip_reasons']}`",
        "",
        "## 6. Coverage Repair",
        "## Attach Target Coverage",
        f"- missing_object_count_before_repair: `{report['coverage_repair']['missing_object_count_before_repair']}`",
        f"- coverage_repair_used: `{report['coverage_repair']['coverage_repair_used']}`",
        f"- coverage_repair_count: `{report['coverage_repair']['coverage_repair_count']}`",
        f"- missing_before: `{coverage_audit.get('missing_before', [])}`",
        f"- missing_after: `{coverage_audit.get('missing_after', [])}`",
        f"- coverage_repair_used: `{coverage_audit.get('coverage_repair_used')}`",
        f"- coverage_repair_count: `{coverage_audit.get('coverage_repair_count')}`",
        f"- coverage_repair_audit: `{coverage_audit_path}`",
        "",
        "## 7. Final Validation",
    ])
    for key, value in report["final_validation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## 8. Important Debug Files"])
    for key, value in report["important_debug_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## 9. Human Review Notes"])
    lines.extend(f"- {item}" for item in report["human_review_notes"])

    json_path = output_dir / "debug" / "agents" / "final_process_report.json"
    md_path = output_dir / "debug" / "agents" / "final_process_report.md"
    write_json(json_path, report)
    write_md(md_path, "Final Process Report", lines)
    return {"json_path": str(json_path), "md_path": str(md_path), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a human-readable final multi-agent process report.")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    result = write_final_process_report(args.output_dir)
    print(json.dumps({"md_path": result["md_path"], "json_path": result["json_path"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
