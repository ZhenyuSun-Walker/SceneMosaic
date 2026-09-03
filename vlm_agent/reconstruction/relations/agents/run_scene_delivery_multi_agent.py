from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
RELATIONS_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from relations.agents.api_client import APIConfig, APIClientError, SceneDeliveryAPIClient
from relations.agents.final_assembler import (
    assemble_final_payloads,
    collect_contain_conflicts,
    compact_contain_conflicts_for_guard,
)
from relations.agents.schema_guard import validate_files
from relations.agents.state import ensure_output_tree, initialize_empty_audits, write_audit_pair, write_json, write_md
from relations.agents.orchestration.delivery_io import (
    POLICY_PATH,
    RELATION_LAYOUT_DIR_ENV,
    _load_json_file,
    apply_render_grid_precompute_to_scene_state,
    configure_wall_candidate_environment,
    copy_policy_to_run,
    load_prompt,
    load_relation_layout_manifest,
    load_static_policy,
    relation_layout_expected_manifest_path,
    update_agent_audits,
    wait_for_fixed_views_precompute_dir,
    wait_for_relation_precompute,
    write_dry_run_final_marker,
    write_manager_plan,
    write_render_audit,
    write_task_output,
)
from relations.agents.supervisor import supervise_execution
from relations.agents.tool_runner import ensure_canonical_topdown_mapping, scan_render_script_capabilities
from relations.agents.task_registry import (
    compact_manager_capability_catalog,
    compile_compact_manager_plan,
    task_image_artifact_profiles,
)
from relations.tools.build_scene_state import build_scene_state, write_scene_state
from relations.tools.extract_wall_evidence import build_wall_evidence, write_wall_evidence
from relations.tools.wall_orientation_candidates import resolve_orientation_candidate_from_wall_objects
from relations.tools.generate_fixed_layout_views import reuse_fixed_layout_views_cache
from relations.tools.build_wall_visual_evidence import ensure_wall_visual_evidence
from relations.tools.visualize_final_walls_on_topdown import visualize as visualize_final_walls_on_topdown
from relations.tools.write_agent_process_report import write_final_process_report
from relations.tools.runtime_tracker import RuntimeTracker
from relations.tools.build_wall_object_hypotheses import build_and_write_wall_object_hypotheses, extract_wall_object_hypotheses, write_wall_object_hypotheses
from relations.agents.wall_attach_prefilter import (
    build_wall_attach_prefilter,
    compact_wall_evidence_for_recall,
    formal_wall_references,
    render_wall_attach_prefilter_overlay,
    wall_geometry_key,
)

from relations.agents.orchestration.planning import (
    CANONICAL_WALL_IDS,
    PROVISIONAL_WALL_VISIBILITY_CAPABILITY,
    CANONICAL_WALL_NAMING_INSTRUCTION,
    CONTAIN_SCOPE_CONFIDENCE_THRESHOLD,
    CONTAIN_SCOPE_MAX_HIGH_CONFIDENCE_PAIRS_PER_ANCHOR,
    CONTAIN_TASK_MAX_REVIEW_PAIRS,
    HIGH_CONFIDENCE_EXCLUSIVE_MODES,
    HIGH_CONFIDENCE_NON_WALL_MODES,
    _as_instruction_list,
    append_task_instruction,
    task_identifier,
    task_capabilities,
    task_provides_capability,
    task_requires_capability,
    is_wall_attach_recall_task,
    hydrate_wall_attach_candidate_scope,
    annotate_semantic_visibility_geometry_conflict_resolutions,
    _normalized_contain_pair,
    select_contain_review_pairs,
    hydrate_contain_candidate_scope,
    _object_anchor_routing_by_id,
    _scene_state_object_ids,
    hydrate_task_scope_from_anchor_routing,
    hydrate_support_capability_scope,
    task_control_text,
    is_support_contact_task,
    _as_float_pair,
    _point_in_polygon_xz,
    _polygon_area_abs_xz,
    _obb_polygon_from_wall_evidence,
    build_obb_wall_risk_audit,
    write_obb_wall_risk_audit,
    resolve_wall_review_mode,
    local_wall_reasoning_signal,
    wall_orientation_disagreement_signal,
    configure_wall_alignment_task,
    configure_provisional_wall_visibility_task,
    ensure_wall_alignment_object_review_task,
    ensure_post_orientation_wall_visibility_task,
    apply_wall_task_policy,
    ensure_wall_attach_recall_task,
)

def extract_wall_attach_candidate_evidence(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    def walk(value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for root in task_results:
        if not isinstance(root, dict):
            continue
        for node in walk(root):
            if not isinstance(node, dict):
                continue
            for key in ["wall_attach_candidates", "wall_attach_recall", "wall_attach_decisions", "attach_decisions", "accepted_attach_candidates"]:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            evidence.append({
                                "source_task_id": root.get("task_id"),
                                "source_key": key,
                                "candidate": item,
                            })
                elif isinstance(value, dict) and isinstance(value.get("candidates"), list):
                    for item in value["candidates"]:
                        if isinstance(item, dict):
                            evidence.append({
                                "source_task_id": root.get("task_id"),
                                "source_key": f"{key}.candidates",
                                "candidate": item,
                            })
    return evidence


def build_and_write_wall_evidence(scene_state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    wall_evidence = build_wall_evidence(scene_state)
    write_wall_evidence(wall_evidence, output_dir / "debug" / "walls")
    return wall_evidence


def compact_formal_wall_selection(wall_evidence: dict[str, Any]) -> dict[str, Any]:
    source = wall_evidence.get("render_grid_wall_candidates") if isinstance(wall_evidence, dict) else {}
    source = source if isinstance(source, dict) else {}
    audit = wall_evidence.get("wall_orientation_selection_audit") if isinstance(wall_evidence, dict) else {}
    audit = audit if isinstance(audit, dict) else {}
    selected_id = (
        audit.get("selected_candidate_id")
        or source.get("selected_orientation_candidate_id")
    )
    return {
        "selected_orientation_candidate_id": selected_id,
        "selection_source": (
            audit.get("selection_source")
            or source.get("wall_orientation_selection_source")
        ),
        "selection_is_final": bool(selected_id),
        "agent_may_select_candidate": False,
        "agent_may_change_geometry": False,
    }


def compact_provisional_wall_visibility_catalog(wall_evidence: dict[str, Any]) -> dict[str, Any]:
    """Expose candidate-edge identities without forwarding local coordinates."""
    source = wall_evidence.get("render_grid_wall_candidates") if isinstance(wall_evidence, dict) else {}
    source = source if isinstance(source, dict) else {}
    candidates = []
    for candidate in source.get("orientation_candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        edges = []
        for index, wall in enumerate(candidate.get("wall_segments", []), start=1):
            if not isinstance(wall, dict):
                continue
            provisional_wall_id = str(wall.get("id") or wall.get("provisional_wall_id") or "")
            if provisional_wall_id not in CANONICAL_WALL_IDS:
                continue
            edges.append({
                "provisional_wall_id": provisional_wall_id,
                "geometry_key": wall_geometry_key(wall),
                "screen_clockwise_order_index": wall.get("screen_clockwise_order_index") or index,
            })
        candidates.append({
            "candidate_id": candidate_id,
            "orientation_source": (candidate.get("orientation") or {}).get("source") if isinstance(candidate.get("orientation"), dict) else None,
            "provisional_wall_ids": [item["provisional_wall_id"] for item in edges],
            "edges": edges,
        })
    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "agent_contract": "Use the isolated labelled candidate panels. Return candidate_id plus provisional_wall_id semantics only; local code maps the selected candidate by immutable edge geometry after orientation scoring.",
    }


def _walk_mapping_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mapping_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mapping_nodes(child)


def materialize_selected_provisional_wall_visibility(
    wall_evidence: dict[str, Any],
    selected_candidate_id: str | None,
    task_results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bind candidate-panel semantics to immutable final OBB edges after selection."""
    catalog = compact_provisional_wall_visibility_catalog(wall_evidence)
    selected_candidate = next(
        (item for item in catalog.get("candidates", []) if item.get("candidate_id") == selected_candidate_id),
        None,
    )
    audit: dict[str, Any] = {
        "schema_version": "provisional_wall_visibility_binding",
        "selected_candidate_id": selected_candidate_id,
        "candidate_panel_count": catalog.get("candidate_count", 0),
        "binding_policy": "candidate_id_exact_then_geometry_key_exact",
        "status": "not_requested",
        "source_task_ids": [],
        "matched_provisional_records": [],
        "unmatched_provisional_records": [],
        "missing_final_wall_ids": list(CANONICAL_WALL_IDS),
    }
    if not selected_candidate_id or not isinstance(selected_candidate, dict):
        audit["status"] = "selected_candidate_unavailable"
        return None, audit

    candidate_edges = {
        str(edge.get("provisional_wall_id")): edge
        for edge in selected_candidate.get("edges", [])
        if isinstance(edge, dict) and edge.get("provisional_wall_id")
    }
    final_references = formal_wall_references(wall_evidence)
    final_by_geometry = {
        str(reference.get("geometry_key")): reference
        for reference in final_references if reference.get("geometry_key")
    }
    seen: set[tuple[str, str]] = set()
    materialized: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for root in task_results:
        if not isinstance(root, dict):
            continue
        provided = root.get("provided_capabilities") if isinstance(root.get("provided_capabilities"), list) else []
        if PROVISIONAL_WALL_VISIBILITY_CAPABILITY not in provided:
            continue
        source_task_id = str(root.get("task_id") or "unknown_task")
        if source_task_id not in source_ids:
            source_ids.append(source_task_id)
        for node in _walk_mapping_nodes(root):
            items = node.get("provisional_wall_visibility_review") if isinstance(node, dict) else None
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                candidate_id = str(item.get("candidate_id") or "").strip()
                provisional_wall_id = str(item.get("provisional_wall_id") or item.get("wall_id") or item.get("id") or "").strip()
                if candidate_id != selected_candidate_id:
                    continue
                edge = candidate_edges.get(provisional_wall_id)
                final_reference = final_by_geometry.get(str(edge.get("geometry_key"))) if isinstance(edge, dict) and edge.get("geometry_key") else None
                if not final_reference:
                    audit["unmatched_provisional_records"].append({
                        "source_task_id": source_task_id,
                        "candidate_id": candidate_id,
                        "provisional_wall_id": provisional_wall_id or None,
                        "reason": "candidate_edge_or_final_geometry_key_unavailable",
                    })
                    continue
                key = (source_task_id, provisional_wall_id)
                if key in seen:
                    continue
                seen.add(key)
                record = {
                    "wall_id": final_reference.get("wall_id"),
                    "geometry_key": final_reference.get("geometry_key"),
                    "original_image_visibility": item.get("original_image_visibility"),
                    "weak_wall": item.get("weak_wall"),
                    "closure_only": item.get("closure_only"),
                    "can_anchor_wall_attach": item.get("can_anchor_wall_attach"),
                    "reason": item.get("reason") or item.get("reason_code"),
                    "reason_code": item.get("reason_code"),
                    "candidate_id": candidate_id,
                    "provisional_wall_id": provisional_wall_id,
                    "semantic_binding_source_task_id": source_task_id,
                }
                record = {key: value for key, value in record.items() if value is not None}
                materialized.append(record)
                audit["matched_provisional_records"].append({
                    "source_task_id": source_task_id,
                    "candidate_id": candidate_id,
                    "provisional_wall_id": provisional_wall_id,
                    "final_wall_id": final_reference.get("wall_id"),
                    "geometry_key": final_reference.get("geometry_key"),
                })
    audit["source_task_ids"] = source_ids
    matched_final_ids = {str(record.get("wall_id")) for record in materialized if record.get("wall_id")}
    audit["missing_final_wall_ids"] = [wall_id for wall_id in CANONICAL_WALL_IDS if wall_id not in matched_final_ids]
    if not materialized:
        audit["status"] = "no_matching_selected_candidate_semantics"
        return None, audit
    audit["status"] = "applied" if len(matched_final_ids) == len(CANONICAL_WALL_IDS) else "partially_applied"
    result = {
        "task_id": "local_provisional_wall_visibility_binding",
        "task_type": "local_provisional_wall_visibility_binding",
        "canonical_task_kind": "local_provisional_wall_visibility_binding",
        "provided_capabilities": ["wall_visibility_semantics_ready"],
        "semantic_authority": "selected_candidate_provisional_visibility",
        "wall_semantic_priority": 300,
        "selected_candidate_id": selected_candidate_id,
        "wall_visibility_review": materialized,
    }
    return result, audit


def materialize_selected_candidate_object_wall_bindings(
    wall_evidence: dict[str, Any],
    selected_candidate_id: str | None,
    task_results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Map Agent candidate-panel object bindings onto final canonical wall ids."""
    catalog = compact_provisional_wall_visibility_catalog(wall_evidence)
    selected_candidate = next(
        (item for item in catalog.get("candidates", []) if item.get("candidate_id") == selected_candidate_id),
        None,
    )
    audit: dict[str, Any] = {
        "schema_version": "selected_candidate_object_wall_binding",
        "selected_candidate_id": selected_candidate_id,
        "binding_policy": "candidate_id_exact_then_provisional_edge_geometry_key_exact",
        "status": "not_requested",
        "matched_bindings": [],
        "unmatched_bindings": [],
    }
    if not selected_candidate_id or not isinstance(selected_candidate, dict):
        audit["status"] = "selected_candidate_unavailable"
        return None, audit

    candidate_edges = {
        str(edge.get("provisional_wall_id")): edge
        for edge in selected_candidate.get("edges", [])
        if isinstance(edge, dict) and edge.get("provisional_wall_id")
    }
    final_by_geometry = {
        str(reference.get("geometry_key")): reference
        for reference in formal_wall_references(wall_evidence)
        if reference.get("geometry_key")
    }
    hypotheses: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_task_ids: list[str] = []
    for root in task_results:
        if not isinstance(root, dict):
            continue
        source_task_id = str(root.get("task_id") or "unknown_task")
        for node in _walk_mapping_nodes(root):
            items = node.get("candidate_wall_bindings") if isinstance(node, dict) else None
            if not isinstance(items, list):
                continue
            if source_task_id not in source_task_ids:
                source_task_ids.append(source_task_id)
            for item in items:
                if not isinstance(item, dict):
                    continue
                candidate_id = str(item.get("candidate_id") or "").strip()
                provisional_wall_id = str(item.get("provisional_wall_id") or "").strip()
                object_id = str(item.get("object_id") or "").strip()
                confidence = str(item.get("confidence") or "").strip().lower()
                if candidate_id != selected_candidate_id or not object_id or confidence not in {"high", "medium"}:
                    continue
                edge = candidate_edges.get(provisional_wall_id)
                reference = final_by_geometry.get(str(edge.get("geometry_key"))) if isinstance(edge, dict) else None
                if not reference:
                    audit["unmatched_bindings"].append({
                        "source_task_id": source_task_id,
                        "object_id": object_id,
                        "candidate_id": candidate_id,
                        "provisional_wall_id": provisional_wall_id,
                    })
                    continue
                final_wall_id = str(reference.get("wall_id"))
                key = (object_id, final_wall_id)
                if key in seen:
                    continue
                seen.add(key)
                hypothesis = {
                    "hypothesis_id": f"selected_panel_binding:{object_id}:{final_wall_id}",
                    "object_id": object_id,
                    "supporting_objects": [object_id],
                    "candidate_wall_id": final_wall_id,
                    "geometry_key": reference.get("geometry_key"),
                    "role": item.get("role"),
                    "confidence": confidence,
                    "evidence_code": item.get("evidence_code"),
                    "source": "selected_candidate_agent_semantic_binding",
                    "candidate_id": candidate_id,
                    "provisional_wall_id": provisional_wall_id,
                    "may_be_used_as_final_wall": False,
                }
                hypotheses.append(hypothesis)
                audit["matched_bindings"].append({
                    "source_task_id": source_task_id,
                    "object_id": object_id,
                    "candidate_id": candidate_id,
                    "provisional_wall_id": provisional_wall_id,
                    "final_wall_id": final_wall_id,
                    "geometry_key": reference.get("geometry_key"),
                })
    audit["source_task_ids"] = source_task_ids
    if not hypotheses:
        audit["status"] = "no_matching_selected_candidate_bindings"
        return None, audit
    audit["status"] = "applied"
    return {
        "task_id": "local_selected_candidate_object_wall_binding",
        "task_type": "local_selected_candidate_object_wall_binding",
        "canonical_task_kind": "local_selected_candidate_object_wall_binding",
        "provided_capabilities": ["wall_object_hypotheses_ready"],
        "selected_candidate_id": selected_candidate_id,
        "wall_line_hypotheses": hypotheses,
    }, audit


def scene_object_topdown_rotations(scene_state: dict[str, Any]) -> dict[str, float]:
    candidates = scene_state.get("object_candidates") if isinstance(scene_state, dict) else {}
    items = candidates.get("items") if isinstance(candidates, dict) else []
    rotations: dict[str, float] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or item.get("id") or "").strip()
        layout = item.get("layout_reference") if isinstance(item.get("layout_reference"), dict) else {}
        orientation = layout.get("orientation_for_agent") if isinstance(layout.get("orientation_for_agent"), dict) else {}
        rotation = orientation.get("rotation_degrees_topdown")
        if object_id and isinstance(rotation, (int, float)) and not isinstance(rotation, bool):
            rotations[object_id] = float(rotation)
    return rotations


def _discard_pre_orientation_wall_semantics(results: list[dict[str, Any]]) -> list[str]:
    """Remove semantic labels that were keyed to a superseded wall candidate."""
    removed: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for key in ("wall_visibility_review", "walls", "accepted_walls"):
            if key in result:
                result.pop(key, None)
                removed.append(f"{result.get('task_id', 'unknown')}:{key}")
    return removed


def apply_wall_orientation_selection(
    scene_state: dict[str, Any],
    semantic_results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    precompute = evidence.get("render_grid_wall_precompute") if isinstance(evidence.get("render_grid_wall_precompute"), dict) else {}
    method_path = Path(str(precompute.get("method_candidates_path") or ""))
    method_data = _load_json_file(method_path) if method_path.exists() else None
    raw_candidates = [
        item for item in (method_data.get("candidates", []) if isinstance(method_data, dict) else [])
        if isinstance(item, dict) and item.get("method") == "obb_from_mesh_footprint_min_rect"
    ]
    candidate_ids = [
        str(item.get("candidate_id") or f"candidate_{index:02d}")
        for index, item in enumerate(raw_candidates, start=1)
    ]
    review = method_data.get("wall_orientation_review") if isinstance(method_data, dict) and isinstance(method_data.get("wall_orientation_review"), dict) else {}
    baseline_id = review.get("baseline_candidate_id")
    if baseline_id not in candidate_ids:
        baseline_id = candidate_ids[0] if candidate_ids else None

    object_score_resolution = resolve_orientation_candidate_from_wall_objects(
        semantic_results,
        raw_candidates,
        scene_object_topdown_rotations(scene_state),
    )
    local_choice = object_score_resolution.get("resolved_candidate_id")
    selected_id = local_choice or baseline_id
    previous_id = precompute.get("selected_orientation_candidate_id")
    if selected_id:
        precompute["selected_orientation_candidate_id"] = selected_id
        if local_choice:
            precompute["wall_orientation_selection_source"] = "local_agent_object_rotation_score"
        else:
            precompute["wall_orientation_selection_source"] = "precompute_baseline_fallback"

    invalidated_semantics: list[str] = []
    if (
        local_choice
        and previous_id
        and selected_id != previous_id
    ):
        invalidated_semantics = _discard_pre_orientation_wall_semantics(semantic_results)

    audit = {
        "schema_version": "wall_orientation_selection_audit",
        "status": "applied" if selected_id else "unavailable",
        "method_candidates_path": str(method_path) if method_path else None,
        "available_candidate_ids": candidate_ids,
        "selection_required": len(candidate_ids) > 1,
        "selected_candidate_id": selected_id,
        "previous_selected_candidate_id": previous_id,
        "selection_source": precompute.get("wall_orientation_selection_source"),
        "local_object_rotation_scoring": object_score_resolution,
        "agent_selected_candidate_id_is_authoritative": False,
        "pre_orientation_wall_semantics_invalidated": invalidated_semantics,
        "geometry_policy": "candidate extent comes from all posed mesh footprints; Agent selects semantic wall-following objects and local code scores only finite precomputed axes",
        "wall_attach_dependency": "selected geometry is applied and wall evidence is rebuilt before visibility and wall-attach review",
        "orientation_support_does_not_imply_attach": True,
    }
    write_audit_pair(output_dir, "walls", "wall_orientation_selection", audit, "Wall Orientation Selection")
    return audit


TASK_IMAGE_ARTIFACT_PROFILES = task_image_artifact_profiles()

CAPABILITY_IMAGE_ARTIFACT_PROFILES = {
    "wall_alignment_objects_ready": ("original_image", "mask_overlay"),
    "provisional_wall_visibility_ready": ("wall_orientation_overlay",),
    "wall_visibility_semantics_ready": ("original_image", "wall_attach_prefilter_overlay"),
    "wall_attach_decisions_ready": ("original_image", "mask_overlay", "wall_attach_prefilter_overlay"),
    "object_physical_properties_ready": ("original_image", "mask_overlay"),
    "room_materials_ready": ("original_image",),
}

NO_IMAGE_TASK_KINDS = {"relation_reconciliation", "final_validation", "final_consistency_validation"}

IMAGE_ARTIFACT_ALIASES = {
    "original_image_path": "original_image",
    "reviewed_overlay": "mask_overlay",
    "reviewed_overlay_path": "mask_overlay",
    "canonical_topdown_path": "canonical_topdown",
    "wall_orientation_candidates_overlay": "wall_orientation_overlay",
    "wall_orientation_candidates_overlay_path": "wall_orientation_overlay",
}


def is_wall_visual_task(task: dict[str, Any]) -> bool:
    text = task_control_text(task)
    return any(token in text for token in ["wall", "infer_four_clockwise_walls", "repair_missing_final_walls"])


def is_contain_relation_task(task: dict[str, Any]) -> bool:
    capabilities = task_capabilities(task, "provides_capabilities")
    if capabilities:
        return "contain_relations_ready" in capabilities
    identity = " ".join(
        str(task.get(key) or "")
        for key in ("task_id", "id", "task_type", "canonical_task_kind")
    ).lower()
    return any(marker in identity for marker in ("contain_relation", "contain relation", "containment", "container"))


def compact_image_paths(paths: list[Any], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not path:
            continue
        path_s = str(path)
        if path_s in seen:
            continue
        if not Path(path_s).exists():
            continue
        seen.add(path_s)
        result.append(path_s)
        if len(result) >= limit:
            break
    return result


def _task_binding_path(task: dict[str, Any], key: str) -> str | None:
    binding = task.get("evidence_binding")
    if isinstance(binding, dict) and binding.get(key):
        return str(binding[key])
    return None


def _wall_image_artifact_paths(task: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    prefilter = task.get("wall_attach_prefilter") if isinstance(task.get("wall_attach_prefilter"), dict) else {}
    prefilter_overlay = prefilter.get("overlay_path") if prefilter.get("overlay_status") == "available" else None
    prefilter_overlay = (
        prefilter_overlay
        or evidence.get("wall_attach_prefilter_overlay_path")
        or evidence.get("primary_candidate_overlay_path")
        or evidence.get("candidate_walls_overlay_path")
    )
    return {
        "original_image": _task_binding_path(task, "original_image") or evidence.get("original_image_path"),
        "mask_overlay": _task_binding_path(task, "mask_overlay") or evidence.get("mask_overlay_path") or evidence.get("reviewed_overlay_path"),
        "wall_contact_sheet": evidence.get("contact_sheet_path"),
        "candidate_wall_overlay": evidence.get("candidate_walls_overlay_path"),
        "primary_wall_overlay": evidence.get("primary_candidate_overlay_path") or evidence.get("candidate_walls_overlay_path"),
        "wall_orientation_overlay": evidence.get("render_grid_wall_orientation_candidates_overlay_path") or evidence.get("primary_candidate_overlay_path") or evidence.get("candidate_walls_overlay_path"),
        "canonical_topdown": _task_binding_path(task, "canonical_topdown") or evidence.get("canonical_topdown_path"),
        "top_view_clean": evidence.get("top_view_clean_path"),
        "top_view_grid": evidence.get("top_view_grid_path"),
        "wall_attach_prefilter_overlay": prefilter_overlay,
    }


def resolve_task_image_evidence(task: dict[str, Any], wall_visual_evidence: dict[str, Any] | None) -> dict[str, Any]:
    evidence = wall_visual_evidence if isinstance(wall_visual_evidence, dict) else {}
    available = _wall_image_artifact_paths(task, evidence)
    canonical_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").strip().lower()
    text = task_control_text(task)
    no_image_markers = ["reconciliation", "coverage_and_repair", "merge_and_coverage", "coverage_repair", "final_validation", "consistency_validation", "relations_validation_and_repair"]
    if canonical_kind in NO_IMAGE_TASK_KINDS or any(marker in text for marker in no_image_markers):
        return {"matched": True, "image_paths": [], "selected_artifacts": [], "routing_source": "structured_no_images", "keyword_fallback_used": False}

    artifact_names: list[str] = []
    routing_sources: list[str] = []

    def add_artifacts(names: Any, source: str) -> None:
        for name in names or ():
            normalized = IMAGE_ARTIFACT_ALIASES.get(str(name).strip().lower().replace("-", "_").replace(" ", "_"), str(name).strip().lower().replace("-", "_").replace(" ", "_"))
            if normalized in available and normalized not in artifact_names:
                artifact_names.append(normalized)
        if source not in routing_sources:
            routing_sources.append(source)

    if canonical_kind == "wall_final_adjudication":
        if str(task.get("wall_review_mode", "heavy")).lower() == "light":
            add_artifacts(("original_image", "wall_orientation_overlay", "mask_overlay"), "canonical_task_kind")
        else:
            add_artifacts(("wall_contact_sheet", "candidate_wall_overlay", "canonical_topdown", "top_view_clean", "top_view_grid", "original_image", "mask_overlay"), "canonical_task_kind")
    elif canonical_kind in TASK_IMAGE_ARTIFACT_PROFILES:
        add_artifacts(TASK_IMAGE_ARTIFACT_PROFILES[canonical_kind], "canonical_task_kind")

    required_artifacts = task_capabilities(task, "requires_artifacts")
    recognized_required = []
    for item in required_artifacts:
        normalized = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        normalized = IMAGE_ARTIFACT_ALIASES.get(normalized, normalized)
        if normalized in available:
            recognized_required.append(normalized)
    if recognized_required:
        known_profile = canonical_kind in TASK_IMAGE_ARTIFACT_PROFILES or canonical_kind == "wall_final_adjudication"
        permitted_required = (
            [name for name in recognized_required if name in artifact_names] if known_profile else recognized_required
        )
        if permitted_required:
            add_artifacts(permitted_required, "requires_artifacts")

    for capability in task_capabilities(task, "provides_capabilities"):
        profile = CAPABILITY_IMAGE_ARTIFACT_PROFILES.get(str(capability).strip().lower())
        if profile:
            add_artifacts(profile, "provides_capabilities")

    matched = bool(routing_sources) or canonical_kind in TASK_IMAGE_ARTIFACT_PROFILES
    selected_artifacts = [name for name in artifact_names if available.get(name)]
    return {
        "matched": matched,
        "image_paths": [str(available[name]) for name in selected_artifacts],
        "selected_artifacts": selected_artifacts,
        "routing_source": "+".join(routing_sources) if routing_sources else "task_keyword_fallback",
        "keyword_fallback_used": not matched,
    }


def contain_task_image_paths(task: dict[str, Any], scene_state: dict[str, Any]) -> list[str]:
    if not is_contain_relation_task(task):
        return []
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    return compact_image_paths(
        [
            _task_binding_path(task, "original_image") or evidence.get("original_image_path"),
            _task_binding_path(task, "mask_overlay") or evidence.get("mask_overlay_path"),
        ],
        2,
    )


def load_contain_candidates_for_task(
    task: dict[str, Any],
    scene_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project recall pairs with image localization, never heuristic evidence."""
    path = _task_binding_path(task, "contain_candidates")
    data = _load_json_file(Path(path)) if path else None
    if not isinstance(data, dict):
        return {"available": False}

    bbox_by_id: dict[str, list[float]] = {}
    candidates = (
        scene_state.get("object_candidates", {})
        if isinstance(scene_state, dict) and isinstance(scene_state.get("object_candidates"), dict)
        else {}
    )
    for item in candidates.get("items", []) if isinstance(candidates.get("items"), list) else []:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id") or item.get("id")
        bbox = item.get("bbox_2d")
        if not (object_id and isinstance(bbox, list) and len(bbox) >= 4):
            continue
        try:
            bbox_by_id[str(object_id)] = [round(float(value), 6) for value in bbox[:4]]
        except (TypeError, ValueError):
            continue

    raw_pairs = data.get("candidate_pairs") if isinstance(data.get("candidate_pairs"), list) else []
    selection = select_contain_review_pairs(data)
    pairs: list[dict[str, Any]] = []
    for selected in selection["selected_pairs"]:
        item = selected["source"]
        anchor = selected["object_anchor"]
        target = selected["object_target"]
        pair = {
            "object_anchor": anchor,
            "object_target": target,
            "anchor_render_direction": item.get("anchor_render_direction"),
            "candidate_origin": "local_high_recall_scan",
            "candidate_is_hint_only": True,
            "image_bboxes_are_localization_only": True,
        }
        if anchor in bbox_by_id:
            pair["anchor_bbox_2d"] = bbox_by_id[anchor]
        if target in bbox_by_id:
            pair["target_bbox_2d"] = bbox_by_id[target]
        pairs.append(pair)

    possible_containers: list[dict[str, Any]] = []
    raw_containers = data.get("possible_containers") if isinstance(data.get("possible_containers"), list) else []
    for item in raw_containers:
        if not isinstance(item, dict):
            continue
        possible_containers.append({
            key: item.get(key)
            for key in (
                "object_id",
                "category",
                "description",
                "render_direction",
                "anchor_render_direction",
            )
            if item.get(key) is not None
        })

    return {
        "available": True,
        "possible_containers": possible_containers,
        "candidate_pairs": pairs,
        "candidate_pair_count": selection["valid_pair_count"],
        "retained_candidate_pair_count": len(pairs),
        "dropped_candidate_pair_count": len(selection["dropped_pairs"]),
        "candidate_selection_policy": selection["selection_policy"],
        "note": (
            "These pairs only select what to inspect. Per-pair bbox_2d values locate object ids in the original image and "
            "carry zero positive evidence for contain. Accept only from positive original-image/mask evidence of interior "
            "membership; add directly visible omissions."
        ),
    }

def wall_visual_image_paths(task: dict[str, Any], wall_visual_evidence: dict[str, Any] | None) -> list[str]:
    routing = resolve_task_image_evidence(task, wall_visual_evidence)
    if routing.get("matched"):
        return list(routing.get("image_paths", []))
    if not isinstance(wall_visual_evidence, dict) or not is_wall_visual_task(task):
        return []
    task_type = str(task.get("task_type", "")).lower()
    task_id = task_identifier(task).lower()
    text = task_control_text(task)

    def compact(paths: list[Any], limit: int) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for path in paths:
            if not path:
                continue
            path_s = str(path)
            if path_s in seen:
                continue
            seen.add(path_s)
            result.append(path_s)
            if len(result) >= limit:
                break
        return result

    original = wall_visual_evidence.get("original_image_path")
    mask = wall_visual_evidence.get("mask_overlay_path") or wall_visual_evidence.get("reviewed_overlay_path")
    contact = wall_visual_evidence.get("contact_sheet_path")
    candidate_overlay = wall_visual_evidence.get("candidate_walls_overlay_path")
    canonical = wall_visual_evidence.get("canonical_topdown_path")
    top_clean = wall_visual_evidence.get("top_view_clean_path")
    top_grid = wall_visual_evidence.get("top_view_grid_path")

    # These are structured merge/check tasks. Sending images again is expensive and usually redundant.
    if any(marker in text for marker in ["reconciliation", "coverage_and_repair", "merge_and_coverage", "coverage_repair", "final_validation", "final_scene_delivery_validation", "final_consistency_validation", "consistency_validation", "relations_validation_and_repair"]):
        return []
    if "support_contact" in text:
        return compact([original, mask, top_clean, canonical, candidate_overlay], 4)
    if task_type == "wall_object_hypothesis":
        return compact([original, mask, contact, canonical, top_clean, candidate_overlay], 6)
    if task_type == "wall_final_adjudication":
        if str(task.get("wall_review_mode", "heavy")).lower() == "light":
            primary_overlay = wall_visual_evidence.get("primary_candidate_overlay_path")
            return compact([original, primary_overlay or candidate_overlay, canonical, mask], 4)
        return compact([contact, candidate_overlay, canonical, top_clean, top_grid, original, mask], 7)
    if "wall_attach" in task_type or "wall_attach" in task_id:
        prefilter = task.get("wall_attach_prefilter") if isinstance(task.get("wall_attach_prefilter"), dict) else {}
        prefilter_overlay = prefilter.get("overlay_path") if prefilter.get("overlay_status") == "available" else None
        if prefilter_overlay:
            return compact([original, mask, prefilter_overlay], 3)
        return compact([original, mask, contact, canonical, candidate_overlay], 5)

    paths = wall_visual_evidence.get("image_paths_for_wall_repair") or []
    if not isinstance(paths, list):
        return []
    return compact(paths, 5)

def wall_visual_payload(wall_visual_evidence: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(wall_visual_evidence, dict):
        return {"available": False}
    keys = [
        "manifest_path",
        "contact_sheet_path",
        "candidate_walls_overlay_path",
        "canonical_topdown_path",
        "canonical_grid_mapping_path",
        "render_grid_method_candidates_path",
        "render_grid_affine_path",
        "render_grid_topdown_path",
        "render_grid_topdown_grid_path",
        "render_grid_foreground_mask_path",
        "render_grid_candidate_overlay_path",
        "render_grid_wall_orientation_candidates_overlay_path",
        "render_grid_primary_candidate_overlay_path",
        "top_view_clean_path",
        "top_view_grid_path",
        "front_view_path",
        "back_view_path",
        "left_view_path",
        "right_view_path",
        "original_image_path",
        "mask_overlay_path",
        "reviewed_overlay_path",
        "image_paths_for_wall_repair",
        "wall_evidence_path",
        "candidate_walls_count",
        "candidate_walls_source",
        "candidate_walls_trust_level",
        "candidate_walls_may_be_final",
        "candidate_room_polygon_hint",
        "failure_summary",
        "next_suggestions",
    ]
    return {key: wall_visual_evidence.get(key) for key in keys}


def compact_object_summary(scene_state: dict[str, Any]) -> dict[str, Any]:
    items = scene_state.get("object_candidates", {}).get("items", []) if isinstance(scene_state.get("object_candidates"), dict) else []
    compact_items = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                "object_id": item.get("object_id"),
                "category": item.get("category"),
                "description": item.get("description"),
                "has_mask": bool(item.get("mask_path")),
                "has_layout_bbox": bool(item.get("bbox_3d")),
                "bbox_3d": item.get("bbox_3d"),
            }
        )
    return {"count": len(compact_items), "items": compact_items}


def build_manager_planning_payload(
    args: argparse.Namespace,
    scene_state: dict[str, Any],
    render_capabilities: dict[str, Any],
    wall_evidence: dict[str, Any],
    wall_visual_evidence: dict[str, Any],
) -> dict[str, Any]:
    fixed = scene_state.get("evidence", {}).get("fixed_layout_views", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    topdown = scene_state.get("topdown_grid_mapping", {}) if isinstance(scene_state.get("topdown_grid_mapping"), dict) else {}
    return {
        "planning_payload_mode": "light",
        "dry_run": args.dry_run,
        "scene_summary": {
            "scene_name": scene_state.get("scene_name"),
            "run_root": scene_state.get("run_root"),
            "objects": compact_object_summary(scene_state),
            "topdown_status": topdown.get("status"),
            "topdown_mapping_path": topdown.get("canonical_mapping_path"),
            "fixed_layout_views_status": fixed.get("status"),
            "fixed_layout_views_manifest_path": fixed.get("manifest_path"),
        },
        "available_evidence_summary": {
            "original_image_path": scene_state.get("evidence", {}).get("original_image_path"),
            "mask_overlay_path": scene_state.get("evidence", {}).get("mask_overlay_path"),
            "reviewed_objects_path": scene_state.get("evidence", {}).get("reviewed_objects_path"),
            "wall_visual_manifest_path": wall_visual_evidence.get("manifest_path") if isinstance(wall_visual_evidence, dict) else None,
            "candidate_walls_overlay_path": wall_visual_evidence.get("candidate_walls_overlay_path") if isinstance(wall_visual_evidence, dict) else None,
            "image_paths_for_wall_repair_count": len(wall_visual_evidence.get("image_paths_for_wall_repair", [])) if isinstance(wall_visual_evidence, dict) else 0,
        },
        "wall_policy_summary": {
            "wall_candidate_mode": wall_evidence.get("wall_candidate_mode"),
            "active_wall_candidate_source": wall_evidence.get("active_wall_candidate_source"),
            "candidate_walls_count": len(wall_evidence.get("candidate_walls", [])),
            "render_grid_wall_status": wall_evidence.get("render_grid_wall_candidates", {}).get("status"),
            "runner_controls_wall_tasks": True,
            "manager_should_not_generate_wall_extraction_tasks": True,
            "runner_injected_wall_tasks": [
                "task_wall_object_hypothesis",
                "task_wall_final_adjudication",
                "task_wall_attach_candidate_recall",
            ],
        },
        "relation_policy_summary": {
            "allowed_relation_types": ["attach", "contain"],
            "contain_is_same_level_as_attach": True,
            "same_anchor_target_pair_attach_and_contain_are_mutually_exclusive": True,
            "multi_attach_allowed": True,
            "floor_support_is_primary_but_not_exclusive": True,
            "floor_supported_against_wall_objects_may_have_both_floor_and_wall_attach": True,
            "weak_or_inferred_hidden_walls_cannot_anchor_attach": True,
            "each_object_must_appear_at_least_once_as_object_target": True,
            "ceiling_anchor_requires_positive_judgement": True,
        },
        "render_capabilities_summary": {
            "select_partial_objects": render_capabilities.get("capabilities", {}).get("select_partial_objects"),
            "topdown_render": render_capabilities.get("capabilities", {}).get("topdown_render"),
            "reuse_grid_mapping": render_capabilities.get("capabilities", {}).get("reuse_grid_mapping"),
        },
        "manager_instructions": [
            "Plan only non-wall support tasks that are useful for final attach extraction; the runner will inject and control A-line wall tasks.",
            "Preserve multi-agent autonomy: decide whether ceiling judgement, object revision, support-contact extraction, and validation/repair tasks are needed for this scene.",
            "For every planned task, include depends_on when the task needs outputs from earlier tasks. Use [] for independent tasks.",
            "Prefer support_contact_extraction as an independent first-layer task when object_list and object_footprints are sufficient. Add an object-revision dependency only when the support decision genuinely requires a corrected object identity.",
            "For every planned task, include stable routing fields: canonical_task_kind, runner_action, and runtime_class. Use runner_action=local_schema_validation and runtime_class=local_redundant_validation for final validation/coverage/merge checks that the local assembler or schema guard can perform instead of an API task.",
            "Do not use old wall extraction methods or request layout_walls_from_06_layout as a final wall source.",
            "Allow an object to have both floor attach and wall attach when floor support and wall contact are both physically supported.",
            "Plan contain_relation_extraction when the original image or object list positively shows a concrete target inside an ordinary container or on/within any functional level of multi-level furniture. For bunk beds, bookcases/shelving units, multi-tier service carts, and multi-level racks, every occupied level including the top is contain and the same pair must not emit attach. For ordinary single-level furniture, a plain exterior top remains attach. Anchor existence alone is insufficient when empty.",
        ],
        "policy_excerpt": {
            "final_objects_keep_sam3d_object_ids": True,
            "final_relations_allow_attach_and_contain": True,
            "category_description_changes_require_object_revision_audit": True,
        },
    }




def load_expected_relation_manifest(run_root: Path | None) -> dict[str, Any] | None:
    path = relation_layout_expected_manifest_path(run_root)
    if not path:
        return None
    manifest = _load_json_file(path)
    if isinstance(manifest, dict):
        manifest = dict(manifest)
        manifest["expected_manifest_path"] = str(path)
        return manifest
    return None


def expected_relation_manifest_readiness(
    manifest: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Require a real object index before asynchronous Manager planning starts."""
    if not isinstance(manifest, dict):
        return False, "expected_manifest_missing_or_invalid"
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        return False, "expected_manifest_objects_empty"
    object_ids = [
        str(item.get("object_id") or "").strip()
        for item in objects
        if isinstance(item, dict)
    ]
    if len(object_ids) != len(objects) or any(not object_id for object_id in object_ids):
        return False, "expected_manifest_object_id_missing"
    try:
        object_count = int(manifest.get("object_count"))
    except (TypeError, ValueError):
        return False, "expected_manifest_object_count_invalid"
    if object_count <= 0 or object_count != len(objects):
        return False, "expected_manifest_object_count_mismatch"
    return True, "planning_ready_with_nonempty_object_index"


def wait_for_expected_relation_manifest(
    args: argparse.Namespace,
    run_root: Path | None,
    *,
    poll_seconds: float = 1.0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Poll the lightweight manifest only; full relation preprocessing stays async."""
    wait_seconds = max(0.0, float(getattr(args, "relation_precompute_wait_seconds", 0) or 0))
    deadline = time.time() + wait_seconds
    attempts = 0
    latest: dict[str, Any] | None = None
    ready = False
    reason = "expected_manifest_not_checked"
    while True:
        attempts += 1
        latest = load_expected_relation_manifest(run_root)
        ready, reason = expected_relation_manifest_readiness(latest)
        if ready or time.time() >= deadline:
            break
        time.sleep(max(0.1, poll_seconds))
    return latest, {
        "schema_version": "expected_manifest_readiness",
        "status": "ready" if ready else "not_ready",
        "reason": reason,
        "poll_interval_seconds": max(0.1, poll_seconds),
        "wait_budget_seconds": wait_seconds,
        "poll_attempts": attempts,
        "object_count": latest.get("object_count") if isinstance(latest, dict) else None,
        "expected_manifest_path": latest.get("expected_manifest_path") if isinstance(latest, dict) else None,
        "policy": "Manager planning and Physics Sidecar may start before full preprocessing, but never before a nonempty, internally consistent object index exists.",
    }


def launch_async_physics_sidecar(
    args: argparse.Namespace,
    expected_manifest: dict[str, Any] | None,
    output_dir: Path,
) -> tuple[subprocess.Popen[Any] | None, dict[str, Any]]:
    sidecar_dir = output_dir / "physics_sidecar" / "integrated_async"
    manifest_path = (
        Path(expected_manifest["expected_manifest_path"])
        if isinstance(expected_manifest, dict) and expected_manifest.get("expected_manifest_path")
        else None
    )
    metadata: dict[str, Any] = {
        "mode": "async",
        "relation_critical_path_dependency": False,
        "output_dir": str(sidecar_dir),
        "object_physics_path": str(sidecar_dir / "object_physics.json"),
        "started": False,
    }
    if not manifest_path or not manifest_path.exists():
        metadata["status"] = "skipped_missing_expected_manifest"
        return None, metadata

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RELATIONS_ROOT / "object_physics" / "run_sidecar.py"),
        "--scene_name", args.scene_name,
        "--run_root", str(args.run_root.resolve()),
        "--expected_manifest", str(manifest_path),
        "--sidecar_id", "integrated_async",
        "--output_dir", str(sidecar_dir),
        "--model", str(getattr(args, "physics_model", None) or args.model),
        "--vision_model", str(getattr(args, "physics_vision_model", None) or args.vision_model),
        "--api_key_env", args.api_key_env,
        "--vision_image_detail", str(getattr(args, "physics_vision_image_detail", "low")),
    ]
    if args.api_base:
        command.extend(["--api_base", args.api_base])
    if getattr(args, "physics_include_original", False):
        command.append("--include_original")
    if args.dry_run:
        command.append("--dry_run")

    stdout_path = sidecar_dir / "process_stdout.log"
    stderr_path = sidecar_dir / "process_stderr.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CODE_ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            env=environment,
            start_new_session=True,
        )
    metadata.update({
        "status": "running",
        "started": True,
        "pid": process.pid,
        "expected_manifest_path": str(manifest_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
    })
    write_json(sidecar_dir / "launch_manifest.json", metadata)
    return process, metadata


def snapshot_physics_sidecar(
    process: subprocess.Popen[Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result = dict(metadata)
    if process is not None:
        return_code = process.poll()
        result["return_code"] = return_code
        result["status"] = "running" if return_code is None else ("completed" if return_code == 0 else "failed")
    physics_path = Path(result["object_physics_path"]) if result.get("object_physics_path") else None
    result["object_physics_available"] = bool(physics_path and physics_path.exists())
    if result.get("output_dir"):
        output_dir = Path(result["output_dir"])
        runtime_path = output_dir / "runtime.json"
        result["runtime_path"] = str(runtime_path)
        if runtime_path.exists():
            try:
                result["sidecar_runtime"] = json.loads(runtime_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result["runtime_read_error"] = str(exc)
        write_json(output_dir / "integration_status.json", result)
    return result


def build_compact_manager_skeleton_payload(
    args: argparse.Namespace,
    expected_manifest: dict[str, Any] | None,
    scene_state: dict[str, Any],
    render_capabilities: dict[str, Any],
) -> dict[str, Any]:
    objects = []
    if isinstance(expected_manifest, dict) and isinstance(expected_manifest.get("objects"), list):
        objects = expected_manifest.get("objects", [])
    elif isinstance(scene_state.get("object_candidates"), dict):
        objects = compact_object_summary(scene_state).get("items", [])
    artifacts = expected_manifest.get("artifacts", {}) if isinstance(expected_manifest, dict) and isinstance(expected_manifest.get("artifacts"), dict) else {}
    compact_artifacts = {
        str(key): {
            "status": value.get("status"),
            "description": value.get("description"),
        }
        for key, value in artifacts.items()
        if isinstance(value, dict)
    }
    return {
        "manager_plan_format": "compact",
        "planning_payload_mode": "compact_skeleton_async",
        "scene_summary": {
            "scene_name": args.scene_name,
            "objects": {"count": len(objects), "items": objects},
        },
        "expected_artifact_contract": compact_artifacts,
        "capability_catalog": compact_manager_capability_catalog(),
        "runner_contract": {
            "formal_wall_method": "obb_from_mesh_footprint_min_rect",
            "runner_controlled_wall_capabilities": [
                "wall_object_hypotheses_ready",
                "wall_visibility_semantics_ready",
                "wall_attach_decisions_ready",
            ],
            "local_capabilities": [
                "wall_attach_candidate_prefilter",
                "final_assembly",
                "schema_validation",
                "coverage_fallback",
            ],
            "manager_must_not_plan_local_validation": True,
        },
        "relation_contract": {
            "allowed_relation_types": ["attach", "contain"],
            "same_pair_attach_contain_requires_schema_guard_arbitration": True,
            "floor_support_and_wall_contact_may_coexist": True,
            "task_object_scopes_are_not_mutually_exclusive": True,
            "object_anchor_routing_modes": [
                "ceiling_only", "wall_only", "floor_wall",
                "support_only", "contain_candidate", "uncertain",
            ],
            "high_confidence_exclusive_routes_only": True,
            "floor_wall_routes_to_support_and_wall": True,
            "every_object_requires_target_coverage": True,
            "ceiling_requires_positive_judgement": True,
        },
        "compact_task_contract": {
            "required_fields": [
                "node_id",
                "canonical_task_kind",
                "provides_capabilities",
                "depends_on",
                "reason",
                "object_scope",
                "scene_focus",
            ],
            "registered_tasks_must_not_repeat_execution_boilerplate": True,
            "custom_tasks_require": ["custom_instructions", "expected_output_schema"],
        },
        "render_capabilities_summary": {
            "select_partial_objects": render_capabilities.get("capabilities", {}).get("select_partial_objects"),
            "topdown_render": render_capabilities.get("capabilities", {}).get("topdown_render"),
            "reuse_grid_mapping": render_capabilities.get("capabilities", {}).get("reuse_grid_mapping"),
        },
    }



def build_manager_skeleton_payload(
    args: argparse.Namespace,
    expected_manifest: dict[str, Any] | None,
    scene_state: dict[str, Any],
    render_capabilities: dict[str, Any],
) -> dict[str, Any]:
    payload = build_compact_manager_skeleton_payload(
        args,
        expected_manifest,
        scene_state,
        render_capabilities,
    )
    payload["planning_mode"] = "skeleton_async"
    payload["available_evidence_summary"] = {
        "original_image_available": bool(
            scene_state.get("evidence", {}).get("original_image_path")
        ),
        "mask_overlay_available": bool(
            scene_state.get("evidence", {}).get("mask_overlay_path")
        ),
        "full_evidence_available": bool(
            scene_state.get("evidence", {}).get("wall_visual_evidence")
        ),
    }
    return payload


def manager_skeleton_image_paths(args: argparse.Namespace, expected_manifest: dict[str, Any] | None) -> list[str]:
    artifact_path = None
    if isinstance(expected_manifest, dict):
        artifacts = expected_manifest.get("artifacts")
        if isinstance(artifacts, dict) and isinstance(artifacts.get("original_image"), dict):
            artifact_path = artifacts["original_image"].get("path")
    fallback = Path(args.run_root) / "input" / "image.png" if getattr(args, "run_root", None) else None
    return compact_image_paths([artifact_path, str(fallback) if fallback else None], 1)


def write_manager_skeleton_plan(output_dir: Path, plan: dict[str, Any]) -> None:
    write_json(output_dir / "manager_plans" / "manager_plan_skeleton_000.json", plan)
    tasks = plan.get("tasks", []) if isinstance(plan.get("tasks"), list) else []
    lines = [
        "- planning_payload_mode: `skeleton_async`",
        f"- task_count: `{len(tasks)}`",
        "",
        "## Skeleton Tasks",
    ]
    for task in tasks:
        if isinstance(task, dict):
            lines.append(f"- `{task.get('task_id')}`: {task.get('canonical_task_kind') or task.get('task_type')} requires={task.get('requires_artifacts', [])}")
    write_md(output_dir / "manager_plans" / "manager_plan_skeleton_000.md", "Manager Plan Skeleton 000", lines)


def compile_manager_plan_response(
    args: argparse.Namespace,
    raw_plan: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    write_json(output_dir / "manager_plans" / "manager_plan_compact_000.json", raw_plan)
    compiled, audit = compile_compact_manager_plan(raw_plan)
    audit["source_path"] = str(output_dir / "manager_plans" / "manager_plan_compact_000.json")
    audit["compiled_plan_path"] = str(output_dir / "manager_plans" / "manager_plan_000.json")
    write_json(output_dir / "manager_plans" / "manager_plan_compilation_audit.json", audit)
    return compiled, audit


def update_manager_compilation_audit_with_wall_policy(
    output_dir: Path,
    manager_plan: dict[str, Any],
) -> None:
    path = output_dir / "manager_plans" / "manager_plan_compilation_audit.json"
    if not path.exists():
        return
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        audit = {}
    if not isinstance(audit, dict):
        audit = {}
    audit["wall_capability_policy"] = {
        "wall_task_policy": manager_plan.get("wall_task_policy"),
        "manager_plan_repaired": bool(manager_plan.get("manager_plan_repaired")),
        "manager_plan_repairs": manager_plan.get("manager_plan_repairs", []),
        "formal_wall_task_control": manager_plan.get("formal_wall_task_control", {}),
        "field_source": "manager_then_runner_policy",
    }
    write_json(path, audit)


def manager_prompt_name() -> str:
    return "scene_delivery_manager_compact_system.txt"


def annotate_manager_api_call(
    runtime_tracker: RuntimeTracker,
    manager_client: SceneDeliveryAPIClient,
    payload: dict[str, Any],
    prompt_text: str,
    args: argparse.Namespace,
    response_plan: dict[str, Any] | None = None,
) -> None:
    """Attach compact-manager size/routing metadata to the latest API call."""
    if not runtime_tracker.api_calls:
        return
    item = runtime_tracker.api_calls[-1]
    item.update({
        **manager_client.usage_metadata(),
        "manager_plan_format": "compact",
        "manager_image_detail": getattr(manager_client.config, "image_detail", None),
        "manager_output_task_count": (
            len(response_plan.get("tasks", []))
            if isinstance(response_plan, dict) and isinstance(response_plan.get("tasks"), list)
            else None
        ),
    })



def available_relation_artifacts(run_root: Path | None, scene_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = load_expected_relation_manifest(run_root)
    manifest = load_relation_layout_manifest(run_root)
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) and isinstance(manifest.get("outputs"), dict) else {}
    artifacts: dict[str, dict[str, Any]] = {}
    if isinstance(expected, dict) and isinstance(expected.get("artifacts"), dict):
        for key, raw in expected["artifacts"].items():
            if isinstance(raw, dict):
                artifacts[str(key)] = dict(raw)
    path_by_key = {
        "object_list": outputs.get("objects_path") or outputs.get("layout_for_relation_path"),
        "object_footprints": outputs.get("object_footprints_path"),
        "formal_obb_walls": outputs.get("obb_walls_path"),
        "canonical_topdown": outputs.get("canonical_topdown_image_path"),
        "render_grid_affine": outputs.get("render_grid_affine_path"),
        "wall_evidence": outputs.get("wall_evidence_path"),
        "contain_candidates": outputs.get("contain_candidates_path"),
        "contain_visual_evidence": outputs.get("contain_visual_evidence_path"),
        "fixed_views": outputs.get("fixed_views_manifest_path"),
        "scene_state": outputs.get("scene_state_path"),
    }
    for key, path in path_by_key.items():
        artifact = artifacts.setdefault(key, {"description": f"Relation artifact {key}"})
        if path:
            artifact["path"] = str(path)
            artifact["status"] = "available" if Path(str(path)).exists() else "pending"
        else:
            artifact.setdefault("status", "missing")
    topdown = scene_state.get("topdown_grid_mapping", {}) if isinstance(scene_state.get("topdown_grid_mapping"), dict) else {}
    if topdown.get("canonical_mapping_path"):
        artifacts.setdefault("canonical_grid_mapping", {"description": "Canonical top-down grid mapping"})
        artifacts["canonical_grid_mapping"].update({
            "path": topdown.get("canonical_mapping_path"),
            "status": "available" if Path(str(topdown.get("canonical_mapping_path"))).exists() else "pending",
        })
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    visual = evidence.get("wall_visual_evidence", {}) if isinstance(evidence.get("wall_visual_evidence"), dict) else {}
    fixed = evidence.get("fixed_layout_views", {}) if isinstance(evidence.get("fixed_layout_views"), dict) else {}
    alias_paths = {
        "original_image": evidence.get("original_image_path"),
        "mask_overlay": evidence.get("mask_overlay_path"),
        "fixed_views": fixed.get("manifest_path"),
        "wall_visual_evidence": visual.get("manifest_path"),
        "wall_visual_manifest": visual.get("manifest_path"),
        "contact_sheet": visual.get("contact_sheet_path"),
        "candidate_walls_overlay": visual.get("candidate_walls_overlay_path"),
        "primary_candidate_overlay": visual.get("primary_candidate_overlay_path"),
    }
    for key, path in alias_paths.items():
        if path:
            artifacts.setdefault(key, {"description": f"Hydrated evidence alias {key}"})
            artifacts[key].update({
                "path": str(path),
                "status": "available" if Path(str(path)).exists() else "pending",
            })
    return artifacts


def hydrate_manager_plan(manager_plan: dict[str, Any], run_root: Path | None, scene_state: dict[str, Any]) -> dict[str, Any]:
    plan = json.loads(json.dumps(manager_plan))
    artifacts = available_relation_artifacts(run_root, scene_state)
    tasks = [task for task in plan.get("tasks", []) if isinstance(task, dict)]
    object_ids = _scene_state_object_ids(scene_state)
    object_anchor_routing = plan.get("object_anchor_routing", [])
    planned_anchor_modes: set[str] = set()
    if any(is_support_contact_task(task) for task in tasks):
        planned_anchor_modes.add("support_only")
    if any(
        any(capability.startswith("wall_") for capability in task_capabilities(task, "provides_capabilities"))
        or str(task.get("canonical_task_kind") or task.get("task_type") or "").lower().startswith("wall_")
        for task in tasks
    ):
        planned_anchor_modes.add("wall_only")
    if any(
        "ceiling_judgement_ready" in task_capabilities(task, "provides_capabilities")
        or str(task.get("canonical_task_kind") or task.get("task_type") or "").lower() == "ceiling_judgement"
        for task in tasks
    ):
        planned_anchor_modes.add("ceiling_only")
    if any(is_contain_relation_task(task) for task in tasks):
        planned_anchor_modes.add("contain_candidate")
    contain_scope_hydrations: list[dict[str, Any]] = []
    support_scope_hydrations: list[dict[str, Any]] = []
    anchor_routing_scope_hydrations: list[dict[str, Any]] = []
    for task in tasks:
        routing_scope_audit = hydrate_task_scope_from_anchor_routing(
            task,
            object_ids,
            object_anchor_routing,
            planned_anchor_modes,
        )
        if isinstance(routing_scope_audit, dict):
            anchor_routing_scope_hydrations.append({
                "task_id": task_identifier(task),
                **routing_scope_audit,
            })
        required = task.get("requires_artifacts")
        if not isinstance(required, list):
            required = task.get("evidence_needed") if isinstance(task.get("evidence_needed"), list) else []
        normalized_required = [str(item) for item in required]
        bindings: dict[str, Any] = {}
        missing: list[str] = []
        for key in normalized_required:
            artifact = artifacts.get(key)
            if isinstance(artifact, dict) and artifact.get("status") == "available" and artifact.get("path"):
                bindings[key] = artifact.get("path")
            else:
                missing.append(key)
        task["requires_artifacts"] = normalized_required
        task["evidence_binding"] = bindings
        task["evidence_readiness"] = {
            "status": "ready" if not missing else "missing_optional_or_pending",
            "missing_or_pending_artifacts": missing,
        }
        if bindings:
            append_task_instruction(task, "Use the hydrated evidence_binding paths as the concrete artifact locations for this task.")
        if is_contain_relation_task(task):
            candidate_path = bindings.get("contain_candidates")
            candidate_data = _load_json_file(Path(str(candidate_path))) if candidate_path else None
            if isinstance(candidate_data, dict):
                contain_scope_hydrations.append({
                    "task_id": task_identifier(task),
                    **hydrate_contain_candidate_scope(task, candidate_data),
                })
        if is_support_contact_task(task):
            support_scope_hydrations.append({
                "task_id": task_identifier(task),
                **hydrate_support_capability_scope(
                    task,
                    object_ids,
                    object_anchor_routing,
                    planned_anchor_modes,
                ),
            })
    plan["tasks"] = tasks
    plan["manager_planning_mode"] = "skeleton_async_hydrated"
    plan["hydration"] = {
        "status": "applied",
        "artifact_count": len(artifacts),
        "available_artifacts": sorted(key for key, value in artifacts.items() if isinstance(value, dict) and value.get("status") == "available"),
        "artifacts": artifacts,
        "object_anchor_routing": object_anchor_routing,
        "planned_anchor_modes": sorted(planned_anchor_modes),
        "anchor_routing_scope_hydrations": anchor_routing_scope_hydrations,
        "contain_scope_hydrations": contain_scope_hydrations,
        "support_scope_hydrations": support_scope_hydrations,
    }
    return plan


def _unique_runner_task_id(tasks: list[dict[str, Any]], base_id: str) -> str:
    used = {task_identifier(task) for task in tasks}
    if base_id not in used:
        return base_id
    suffix = 2
    while f"{base_id}_{suffix}" in used:
        suffix += 1
    return f"{base_id}_{suffix}"


def _compile_runner_capability_task(
    tasks: list[dict[str, Any]],
    canonical_task_kind: str,
    base_task_id: str,
    reason: str,
) -> dict[str, Any]:
    node_id = _unique_runner_task_id(tasks, base_task_id)
    compiled, _ = compile_compact_manager_plan({
        "plan_id": "runner_capability_completion",
        "objective": "Complete missing required relation capabilities without replacing the Manager DAG.",
        "tasks": [{
            "node_id": node_id,
            "canonical_task_kind": canonical_task_kind,
            "depends_on": [],
            "reason": reason,
            "object_scope": [],
            "scene_focus": reason,
        }],
    })
    task = compiled["tasks"][0]
    task["manager_plan_repair"] = {
        "source": "runner_capability_completion",
        "reason": reason,
    }
    return task


def ensure_required_relation_capabilities(
    manager_plan: dict[str, Any],
    run_root: Path | None,
    scene_state: dict[str, Any],
) -> dict[str, Any]:
    """Repair only missing relation capabilities, leaving Manager grouping intact."""
    plan = json.loads(json.dumps(manager_plan))
    tasks = [task for task in plan.get("tasks", []) if isinstance(task, dict)]
    object_ids = _scene_state_object_ids(scene_state)
    repairs: list[dict[str, Any]] = []

    if not any(task_provides_capability(task, "room_materials_ready") for task in tasks):
        task = _compile_runner_capability_task(
            tasks,
            "room_material_classification",
            "task_room_materials_capability_completion",
            "Final objects.json requires image-grounded walls, floor, and ceiling material descriptions; the Manager omitted a provider for room_materials_ready.",
        )
        tasks.append(task)
        repairs.append({
            "repair_reason": "final_output_requires_room_materials_capability",
            "capability": "room_materials_ready",
            "injected_task_ids": [task_identifier(task)],
        })

    if object_ids and not any(task_provides_capability(task, "support_relations_ready") for task in tasks):
        task = _compile_runner_capability_task(
            tasks,
            "support_contact_extraction",
            "task_support_contact_capability_completion",
            "Every reconstructed object requires an explicit gravity-support verdict; the Manager omitted a provider for support_relations_ready.",
        )
        tasks.append(task)
        repairs.append({
            "repair_reason": "nonempty_scene_requires_gravity_support_capability",
            "capability": "support_relations_ready",
            "injected_task_ids": [task_identifier(task)],
        })

    artifacts = available_relation_artifacts(run_root, scene_state)
    contain_artifact = artifacts.get("contain_candidates") if isinstance(artifacts.get("contain_candidates"), dict) else {}
    contain_path = contain_artifact.get("path") if contain_artifact.get("status") == "available" else None
    contain_data = _load_json_file(Path(str(contain_path))) if contain_path else None
    selection = select_contain_review_pairs(contain_data) if isinstance(contain_data, dict) else {
        "selected_pairs": [],
        "valid_pair_count": 0,
        "selection_policy": "no_available_candidates",
    }
    strong_pairs = [
        pair for pair in selection.get("selected_pairs", [])
        if float(pair.get("confidence_hint", 0.0)) >= CONTAIN_SCOPE_CONFIDENCE_THRESHOLD
    ]
    routing = plan.get("object_anchor_routing") if isinstance(plan.get("object_anchor_routing"), list) else []
    routing_object_ids = [
        str(item.get("object_id"))
        for item in routing
        if isinstance(item, dict)
        and str(item.get("anchor_mode") or "").lower() == "contain_candidate"
        and str(item.get("confidence") or "low").lower() in {"high", "medium"}
        and item.get("object_id")
    ]
    contain_signal = bool(strong_pairs or routing_object_ids)
    if contain_signal and not any(task_provides_capability(task, "contain_relations_ready") for task in tasks):
        task = _compile_runner_capability_task(
            tasks,
            "contain_relation_extraction",
            "task_contain_relation_capability_completion",
            "Strong local contain recall or Manager anchor routing requires semantic contain review; candidates remain hints and require original-image verification.",
        )
        tasks.append(task)
        repairs.append({
            "repair_reason": "strong_contain_signal_requires_semantic_review_capability",
            "capability": "contain_relations_ready",
            "injected_task_ids": [task_identifier(task)],
            "strong_candidate_pair_count": len(strong_pairs),
            "routing_object_ids": routing_object_ids,
        })

    plan["tasks"] = tasks
    plan["relation_capability_completion"] = {
        "policy": "capability_floor_only_manager_retains_task_grouping_and_scene_reasoning",
        "nonempty_object_count": len(object_ids),
        "contain_candidate_path": str(contain_path) if contain_path else None,
        "contain_valid_pair_count": int(selection.get("valid_pair_count", 0)),
        "strong_contain_pair_count": len(strong_pairs),
        "contain_routing_object_ids": routing_object_ids,
        "repairs": repairs,
    }
    if repairs:
        plan["manager_plan_repaired"] = True
        existing_repairs = plan.get("manager_plan_repairs") if isinstance(plan.get("manager_plan_repairs"), list) else []
        plan["manager_plan_repairs"] = [*existing_repairs, *repairs]
    return plan


def _add_matching_task_ids(deps: list[str], known_task_ids: set[str], current_task_id: str, *markers: str) -> None:
    for candidate in sorted(known_task_ids):
        candidate_l = candidate.lower()
        if candidate != current_task_id and any(marker in candidate_l for marker in markers) and candidate not in deps:
            deps.append(candidate)


def infer_task_dependencies(
    task: dict[str, Any],
    known_task_ids: set[str],
    capability_providers: dict[str, list[str]] | None = None,
) -> list[str]:
    raw = task.get("depends_on")
    task_id = task_identifier(task)
    task_type = str(task.get("task_type") or "").lower()
    text = task_control_text(task)
    is_support_task = is_support_contact_task(task)
    is_wall_final_task = task_id == "task_wall_final_adjudication" or task_type == "wall_final_adjudication"
    provides_alignment = task_provides_capability(task, "wall_alignment_objects_ready")
    provides_provisional_visibility = task_provides_capability(task, PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
    provides_visibility = task_provides_capability(task, "wall_visibility_semantics_ready")
    provides_wall_attach = task_provides_capability(task, "wall_attach_decisions_ready")
    deps: list[str] = []
    capability_providers = capability_providers or {}

    def add(dep: str) -> None:
        if dep in known_task_ids and dep != task_id and dep not in deps:
            deps.append(dep)

    if isinstance(raw, list):
        for item in raw:
            dep = str(item)
            dep_l = dep.lower()
            # Support/contact extraction is the gravity/local-contact branch. It does not need wall-attach recall.
            # Keeping manager's wall-attach dependency here serializes an otherwise independent task by ~40s.
            if is_support_task and ("wall_attach" in dep_l or dep_l == "task_wall_final_adjudication"):
                continue
            if is_wall_final_task and "wall_attach" in dep_l:
                continue
            add(dep)
    if (is_wall_final_task or provides_visibility or provides_wall_attach) and not provides_alignment:
        alignment_providers = capability_providers.get("wall_alignment_objects_ready", [])
        for provider_id in alignment_providers:
            add(provider_id)
        if not alignment_providers:
            add("task_wall_alignment_object_review")
    if task_id == "task_wall_attach_candidate_recall" or "wall_attach_candidate_recall" in task_type or "wall_attach_candidate_recall" in text or provides_wall_attach:
        # Candidate-panel visibility is optional Manager-selected evidence. If
        # present, it must finish before local prefiltering/attach review so
        # weak or inferred candidate edges can be excluded deterministically.
        provisional_providers = capability_providers.get(PROVISIONAL_WALL_VISIBILITY_CAPABILITY, [])
        for provider_id in provisional_providers:
            add(provider_id)
        # Recall tasks that do not provide final visibility depend on a
        # separate visibility provider.
        if not provides_visibility:
            semantic_providers = capability_providers.get("wall_visibility_semantics_ready", [])
            for provider_id in semantic_providers:
                add(provider_id)
            if not semantic_providers:
                add("task_wall_final_adjudication")
    elif is_support_task:
        # Gravity/local-contact extraction is independently useful from the original
        # object list and footprints. Do not infer object-revision, ceiling-review,
        # or wall dependencies; preserve them only when the Manager explicitly plans them.
        pass
    elif (
        "relations_validation" in task_id
        or "validation_and_repair" in task_id
        or "final_validation" in text
        or ("validation" in text and "relation" in text)
    ):
        add("task_wall_attach_candidate_recall")
        _add_matching_task_ids(deps, known_task_ids, task_id, "ceiling")
        _add_matching_task_ids(deps, known_task_ids, task_id, "object_revision")
        _add_matching_task_ids(deps, known_task_ids, task_id, "support_contact", "support_relation")
    return deps

def apply_relation_policy_and_dependencies(manager_plan: dict[str, Any]) -> dict[str, Any]:
    tasks = [task for task in manager_plan.get("tasks", []) if isinstance(task, dict)]
    known_ids = {task_identifier(task) for task in tasks}
    capability_providers: dict[str, list[str]] = {}
    for provider_task in tasks:
        for capability in task_capabilities(provider_task, "provides_capabilities"):
            capability_providers.setdefault(capability, []).append(task_identifier(provider_task))
    dual_attach_instruction = (
        "Allow multi-attach: floor support is usually the primary support relation, but it does not exclude an additional wall attach when a floor-supported object's back/side is physically against a verified wall."
    )
    weak_wall_attach_instruction = (
        "Do not attach objects to weak, hidden, not_visible, closure_only, or inferred-by-room-closure walls when the source image cannot show the actual wall surface/contact; those walls may be valid room-closure geometry but not valid attach anchors. Use the canonical final wall ids after wall_final_adjudication, not provisional candidate ids if they were renamed."
    )
    wall_visibility_evidence_instruction = (
        "An image crop boundary, furniture side panel, occlusion contour, or room-closure edge is not evidence that a wall is visible. Mark visible or partially_visible only when the actual wall surface or a wall-floor/wall-ceiling junction is directly visible in the source image."
    )
    for task in tasks:
        task_id = task_identifier(task)
        task_type = str(task.get("task_type") or "").lower()
        text = " ".join(_as_instruction_list(task.get("instructions"))).lower()
        if "wall_attach" in task_type or "support_contact" in task_id or "support contact" in text or "reconciliation" in task_id:
            append_task_instruction(task, dual_attach_instruction)
            append_task_instruction(task, weak_wall_attach_instruction)
        if (
            "wall_attach" in task_type
            or task_provides_capability(task, "wall_attach_decisions_ready")
            or task_provides_capability(task, "wall_visibility_semantics_ready")
            or task_provides_capability(task, PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
        ):
            append_task_instruction(task, wall_visibility_evidence_instruction)
        deps = infer_task_dependencies(task, known_ids, capability_providers)
        task["depends_on"] = deps
    manager_plan["tasks"] = tasks
    manager_plan["task_dependency_policy"] = {
        "mode": "runner_inferred_dag_dependencies",
        "multi_attach_allowed": True,
        "floor_support_is_primary_but_not_exclusive": True,
        "wall_attach_requires_verification": True,
        "support_contact_default_independent_of_object_revision": True,
    }
    return manager_plan


FAST_PRESET_PRUNE_MARKERS = (
    "wall_attach_reconciliation",
    "relation_reconciliation",
    "reconciliation_after_wall_tasks",
    "relation_coverage_and_repair",
    "relation_merge_and_coverage_repair",
    "wall_aware_attach_merge_and_repair",
    "attach_merge_and_repair",
    "merge_and_repair",
    "merge_and_coverage",
    "coverage_repair",
    "final_validation",
    "final_scene_delivery_validation",
    "final_consistency_validation",
    "consistency_validation",
    "relations_validation_and_repair",
)
FAST_PRESET_LOCAL_RUNNER_ACTIONS = {
    "local_attach_merge_check",
    "local_final_assembly_validation",
    "local_final_validation",
    "local_relation_coverage_check",
    "local_schema_validation",
}
FAST_PRESET_PRUNE_RUNTIME_CLASSES = {
    "coverage_repair",
    "final_validation",
    "local_redundant_validation",
    "merge_check",
    "optional_review",
    "reconciliation",
}


def apply_runtime_preset(manager_plan: dict[str, Any]) -> dict[str, Any]:
    tasks = [task for task in manager_plan.get("tasks", []) if isinstance(task, dict)]
    retained: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task_identifier(task).lower()
        task_type = str(task.get("task_type") or "").lower()
        text = task_control_text(task)
        runner_action = str(task.get("runner_action") or "").strip().lower()
        runtime_class = str(task.get("runtime_class") or "").strip().lower()
        if (
            runner_action in FAST_PRESET_LOCAL_RUNNER_ACTIONS
            or runtime_class in FAST_PRESET_PRUNE_RUNTIME_CLASSES
            or any(marker in text for marker in FAST_PRESET_PRUNE_MARKERS)
        ):
            pruned.append(task)
        else:
            retained.append(task)
    retained_ids = {task_identifier(task) for task in retained}
    for task in retained:
        deps = task.get("depends_on", []) if isinstance(task.get("depends_on"), list) else []
        task["depends_on"] = [dep for dep in deps if dep in retained_ids]
    manager_plan["tasks"] = retained
    manager_plan["task_runtime_preset"] = {
        "mode": "fast",
        "target_seconds": 240,
        "pruned_task_ids": [task_identifier(task) for task in pruned],
        "pruned_task_types": [task.get("task_type") for task in pruned],
        "pruned_runner_actions": [task.get("runner_action") for task in pruned],
        "pruned_runtime_classes": [task.get("runtime_class") for task in pruned],
        "policy": "Preserve manager planning, but prune optional reconciliation/coverage/final-validation DynamicTaskAgent calls that local assembler, schema guard, and validation can cover under the 4-minute runtime target.",
    }
    return manager_plan


def task_dependency_layers(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pending = {task_identifier(task): task for task in tasks if isinstance(task, dict)}
    completed: set[str] = set()
    layers: list[list[dict[str, Any]]] = []
    while pending:
        ready = [task for task_id, task in pending.items() if all(dep in completed for dep in task.get("depends_on", []))]
        if not ready:
            # Break dependency cycles conservatively by running the remaining tasks in their current order.
            ready = list(pending.values())
        ready_ids = {task_identifier(task) for task in ready}
        layers.append(ready)
        completed.update(ready_ids)
        pending = {task_id: task for task_id, task in pending.items() if task_id not in ready_ids}
    return layers



def _compact_object_record(item: dict[str, Any]) -> dict[str, Any]:
    """Keep relation-relevant geometry, discard local paths and mesh details."""
    bbox_3d = item.get("bbox_3d") if isinstance(item.get("bbox_3d"), dict) else {}
    layout = item.get("layout_reference", {}) if isinstance(item.get("layout_reference"), dict) else {}
    orientation = layout.get("orientation_for_agent", {}) if isinstance(layout.get("orientation_for_agent"), dict) else {}
    return {
        "object_id": item.get("object_id") or item.get("id"),
        "category": item.get("category"),
        "description": item.get("description"),
        "bbox_2d": item.get("bbox_2d"),
        "bbox_3d": {
            "center": bbox_3d.get("center"),
            "extent": bbox_3d.get("extent"),
            "min": bbox_3d.get("min"),
            "max": bbox_3d.get("max"),
        },
        "rotation_degrees_topdown": orientation.get("rotation_degrees_topdown"),
    }


def compact_object_index(scene_state: dict[str, Any], object_ids: set[str] | None = None) -> dict[str, Any]:
    candidates = scene_state.get("object_candidates", {}) if isinstance(scene_state.get("object_candidates"), dict) else {}
    items = candidates.get("items", []) if isinstance(candidates.get("items"), list) else []
    compact_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or item.get("id") or "")
        if object_ids is not None and object_id not in object_ids:
            continue
        compact_items.append(_compact_object_record(item))
    return {
        "coordinate_system": "OpenGL original XZ; Y is vertical",
        "items": compact_items,
        "full_object_count": len(items),
        "included_object_count": len(compact_items),
    }


def compact_wall_metrics(records: Any, object_ids: set[str] | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in records if isinstance(records, list) else []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if object_ids is not None and object_id not in object_ids:
            continue
        flat = item.get("flat_contact_edge_analysis") if isinstance(item.get("flat_contact_edge_analysis"), dict) else {}
        best_edge = flat.get("best_edge") if isinstance(flat.get("best_edge"), dict) else {}
        result.append({
            "object_id": object_id,
            "category": item.get("category"),
            "wall_id": item.get("wall_id"),
            "footprint_min_distance_to_wall": item.get("footprint_min_distance_to_wall"),
            "bbox_side_near_wall": item.get("bbox_side_near_wall"),
            "possible_against_wall_contact": item.get("possible_against_wall_contact"),
            "is_floor_supported": item.get("is_floor_supported"),
            "is_thin_planar_candidate": item.get("is_thin_planar_candidate"),
            "flat_contact_edge": {
                "analysis_available": flat.get("analysis_available"),
                "has_valid_flat_contact_edge": flat.get("has_valid_flat_contact_edge"),
                "support_type": best_edge.get("support_type"),
            },
        })
    return result


def compact_prior_task_results(task: dict[str, Any], prior_task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only keep dependency conclusions that a later task can consume."""
    canonical_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").lower()
    selected_keys = {
        "wall_final_adjudication": {"wall_line_hypotheses", "likely_wall_objects", "wall_visibility_review", "walls", "decision"},
        "wall_attach_candidate_recall": {"wall_visibility_review", "walls", "decision", "wall_line_hypotheses"},
    }.get(canonical_kind, set())
    if not selected_keys:
        return []
    compact: list[dict[str, Any]] = []
    for result in prior_task_results:
        if not isinstance(result, dict):
            continue
        payload = {key: result.get(key) for key in selected_keys if result.get(key) is not None}
        if payload:
            compact.append({"task_id": result.get("task_id"), "task_type": result.get("task_type"), "result": payload})
    return compact


def compact_output_schema(value: Any) -> Any:
    """Replace verbose audit prose fields in the transport schema only."""
    if isinstance(value, dict):
        compact = {key: compact_output_schema(item) for key, item in value.items() if key not in {"reason", "evidence"}}
        if "reason" in value:
            compact["reason_code"] = "short_snake_case_code"
        if "evidence" in value:
            compact["evidence_codes"] = ["short_snake_case_code"]
        return compact
    if isinstance(value, list):
        return [compact_output_schema(item) for item in value]
    return value


def compact_task_definition(task: dict[str, Any]) -> dict[str, Any]:
    """The Agent needs task intent, not local artifact paths or hydration audits."""
    return {
        "task_id": task.get("task_id"),
        "task_type": task.get("task_type"),
        "canonical_task_kind": task.get("canonical_task_kind"),
        "depends_on": task.get("depends_on", []),
        "provides_capabilities": task_capabilities(task, "provides_capabilities"),
        "object_scope": task.get("object_scope", []),
        "scene_focus": task.get("scene_focus"),
        "scene_focus_evidence_status": "planning_hypothesis_only" if task.get("scene_focus") else None,
        "instructions": _as_instruction_list(task.get("instructions")),
        "expected_output_schema": compact_output_schema(task.get("expected_output_schema", {})),
    }


def compact_task_api_context(
    task: dict[str, Any],
    prior_task_results: list[dict[str, Any]],
    scene_state: dict[str, Any],
    wall_evidence: dict[str, Any],
) -> dict[str, Any]:
    canonical_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").lower()
    prefilter = task.get("wall_attach_prefilter") if isinstance(task.get("wall_attach_prefilter"), dict) else {}
    candidate_ids = {
        str(item.get("object_target"))
        for item in prefilter.get("candidate_pairs", [])
        if isinstance(item, dict) and item.get("object_target")
    }
    candidate_ids.update(str(item) for item in prefilter.get("supplemental_visual_review_object_ids", []) if item)
    object_ids = candidate_ids if canonical_kind == "wall_attach_candidate_recall" and candidate_ids else None
    context: dict[str, Any] = {
        "schema_version": "task_api_context_balanced",
        "scene_name": scene_state.get("scene_name"),
        "task": compact_task_definition(task),
        "object_index": compact_object_index(scene_state, object_ids),
        "prior_task_results": compact_prior_task_results(task, prior_task_results),
    }
    provides_provisional_visibility = task_provides_capability(task, PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
    if canonical_kind in {"wall_alignment_object_review", "wall_object_hypothesis", "provisional_wall_visibility_review"}:
        for item in context["object_index"].get("items", []):
            if isinstance(item, dict):
                item.pop("rotation_degrees_topdown", None)
    if canonical_kind == "contain_relation_extraction":
        context["contain_candidates"] = load_contain_candidates_for_task(task, scene_state)
    if canonical_kind in WALL_CONTEXT_TASK_KINDS:
        if canonical_kind in {"wall_alignment_object_review", "wall_object_hypothesis"}:
            context["wall_alignment_contract"] = {
                "agent_output": "semantic object ids plus candidate-panel object-to-provisional-wall bindings",
                "local_inputs_not_sent": "finite candidate coordinates and rotation_degrees_topdown remain local",
                "local_next_step": "score finite OBB axes, then bind only the selected panel's object-wall records by immutable geometry",
                "candidate_id_selection_allowed": False,
                "final_wall_id_or_attach_decision_allowed": False,
            }
            context["provisional_wall_candidate_catalog"] = compact_provisional_wall_visibility_catalog(wall_evidence)
            if provides_provisional_visibility:
                context["provisional_wall_visibility_contract"] = {
                    "agent_output": "candidate_id plus provisional_wall_id visibility semantics only",
                    "final_wall_id_selection_allowed": False,
                    "local_next_step": "bind only records belonging to the locally selected finite candidate by immutable edge geometry",
                }
                context["provisional_wall_candidate_catalog"] = compact_provisional_wall_visibility_catalog(wall_evidence)
        elif canonical_kind == "provisional_wall_visibility_review" or provides_provisional_visibility:
            context["provisional_wall_visibility_contract"] = {
                "agent_output": "candidate_id plus provisional_wall_id visibility semantics only",
                "final_wall_id_selection_allowed": False,
                "local_next_step": "bind only records belonging to the locally selected finite candidate by immutable edge geometry",
            }
            context["provisional_wall_candidate_catalog"] = compact_provisional_wall_visibility_catalog(wall_evidence)
        elif canonical_kind == "wall_attach_candidate_recall" and prefilter:
            context["wall_context"] = compact_wall_evidence_for_recall(wall_evidence, prefilter)
            context["wall_attach_prefilter"] = {
                "candidate_pairs": prefilter.get("candidate_pairs", []),
                "supplemental_visual_review_object_ids": prefilter.get("supplemental_visual_review_object_ids", []),
                "candidate_pair_count": prefilter.get("candidate_pair_count", len(prefilter.get("candidate_pairs", []))),
                "raw_object_wall_pair_count": prefilter.get("raw_object_wall_pair_count"),
                "blocked_wall_ids": prefilter.get("blocked_wall_ids", {}),
            }
        else:
            context["wall_context"] = {
                "coordinate_source": "obb_from_mesh_footprint_min_rect_selected_orientation",
                "formal_wall_references": formal_wall_references(wall_evidence),
                "wall_orientation_selection": compact_formal_wall_selection(wall_evidence),
                "wall_attach_metric_summary": compact_wall_metrics(wall_evidence.get("wall_attach_geometry")),
            }
    return context


SCHEMA_GUARD_WALL_FIELDS = (
    "wall_id",
    "original_image_visibility",
    "weak_wall",
    "closure_only",
    "can_anchor_wall_attach",
    "can_anchor_attach",
    "wall_attach_allowed",
    "decision",
    "reason_code",
    "evidence_codes",
)
SCHEMA_GUARD_OBJECT_PATCH_KEYS = (
    "object_revisions",
    "object_revision_candidates",
    "revision_patches",
    "object_revision_patches",
    "accepted_object_revision_patches",
)
SCHEMA_GUARD_PHYSICAL_KEYS = (
    "object_physical_properties",
    "physical_properties",
    "object_properties",
    "object_property_assessments",
)


def _short_codes(value: Any, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:96] for item in value if item is not None][:limit]


def _guard_relation_from_node(node: dict[str, Any], task_result: dict[str, Any], source_path: str) -> dict[str, Any] | None:
    parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
    anchor = (
        parameters.get("object_anchor")
        or parameters.get("anchor")
        or node.get("object_anchor")
        or node.get("anchor")
        or node.get("candidate_wall_id")
    )
    target = parameters.get("object_target") or parameters.get("target") or node.get("object_target") or node.get("target")
    if not anchor or not target:
        return None
    relation = str(node.get("relation") or "").lower()
    if relation not in {"attach", "contain"}:
        task_kind = str(task_result.get("canonical_task_kind") or task_result.get("task_type") or "").lower()
        relation = "contain" if "contain" in task_kind else "attach"
    decision = node.get("decision") or node.get("verdict") or node.get("status") or "proposed"
    reason_code = node.get("reason_code") or node.get("reason") or node.get("rationale")
    evidence_codes = node.get("evidence_codes") or node.get("evidence") or []
    return {
        "relation": relation,
        "object_anchor": str(anchor),
        "object_target": str(target),
        "decision": str(decision)[:48],
        "reason_code": str(reason_code)[:120] if reason_code else None,
        "evidence_codes": _short_codes(evidence_codes),
        "source_task_id": task_result.get("task_id"),
        "source_task_kind": task_result.get("canonical_task_kind") or task_result.get("task_type"),
        "source_path": source_path,
    }


def _walk_guard_nodes(value: Any, path: str = "$"):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            if key in {"mesh_floor_footprint_hull_xz", "object_footprints", "raw_geometry_evidence", "vertices", "faces"}:
                continue
            yield from _walk_guard_nodes(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_guard_nodes(child, f"{path}[{index}]")


def compact_schema_guard_relation_catalog(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every task-produced relation decision without forwarding task geometry."""
    catalog: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for task_result in task_results:
        if not isinstance(task_result, dict):
            continue
        for path, node in _walk_guard_nodes(task_result):
            if not isinstance(node, dict):
                continue
            candidate = _guard_relation_from_node(node, task_result, path)
            if candidate is None:
                continue
            key = (
                candidate["relation"],
                candidate["object_anchor"],
                candidate["object_target"],
                candidate["decision"],
                str(candidate["source_task_id"]),
            )
            if key not in seen:
                seen.add(key)
                catalog.append(candidate)
    return catalog


def _compact_wall_semantic_records(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for task_result in task_results:
        if not isinstance(task_result, dict):
            continue
        for path, node in _walk_guard_nodes(task_result):
            if not isinstance(node, dict) or not node.get("wall_id"):
                continue
            if not any(field in node for field in SCHEMA_GUARD_WALL_FIELDS[1:]):
                continue
            record = {field: node.get(field) for field in SCHEMA_GUARD_WALL_FIELDS if node.get(field) is not None}
            record["source_task_id"] = task_result.get("task_id")
            record["source_path"] = path
            key = (str(record.get("wall_id")), str(record.get("source_task_id")))
            if key not in seen:
                seen.add(key)
                records.append(record)
    return records


def _compact_named_task_values(task_result: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in keys:
        value = task_result.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            compact[key] = [
                {
                    item_key: item.get(item_key)
                    for item_key in (
                        "object_id", "id", "new_category", "new_description",
                        "is_floor_covering", "mass", "linear_damping", "angular_damping",
                        "static_friction", "dynamic_friction", "decision", "reason_code",
                    )
                    if item.get(item_key) is not None
                }
                for item in value if isinstance(item, dict)
            ]
        elif isinstance(value, (str, bool, int, float)):
            compact[key] = value
    return compact


def compact_schema_guard_context(
    scene_state: dict[str, Any],
    task_results: list[dict[str, Any]],
    supervisor_result: dict[str, Any],
    wall_evidence: dict[str, Any],
    contain_conflict_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the API projection for DeliverySchemaGuard.

    Complete evidence remains local for final assembly, geometry checks, and
    audit. SchemaGuard receives the conclusions it needs to reconcile rather
    than megabytes of mesh-footprint samples.
    """
    task_summaries: list[dict[str, Any]] = []
    for task_result in task_results:
        if not isinstance(task_result, dict):
            continue
        summary = {
            "task_id": task_result.get("task_id"),
            "task_type": task_result.get("task_type"),
            "canonical_task_kind": task_result.get("canonical_task_kind"),
            "provided_capabilities": task_result.get("provided_capabilities", []),
        }
        summary.update(_compact_named_task_values(task_result, SCHEMA_GUARD_OBJECT_PATCH_KEYS))
        summary.update(_compact_named_task_values(task_result, SCHEMA_GUARD_PHYSICAL_KEYS))
        for key in ("ceiling_confirmed", "ceiling_anchor_allowed", "ceiling_present_for_attach", "decision"):
            if task_result.get(key) is not None:
                summary[key] = task_result.get(key)
        task_summaries.append(summary)

    object_items = compact_object_index(scene_state).get("items", [])
    object_index = [
        {
            "object_id": item.get("object_id"),
            "category": item.get("category"),
            "description": item.get("description"),
        }
        for item in object_items
    ]
    supervisor = supervisor_result if isinstance(supervisor_result, dict) else {}
    contain_conflicts = compact_contain_conflicts_for_guard(contain_conflict_report)
    instructions = (
        "Reconcile only the supplied candidate catalog and conclusion summaries. "
        "Do not infer a new object-wall pair or geometry not present in this context. "
        "The local final assembler owns formal OBB geometry and deterministic schema validation."
    )
    if contain_conflicts:
        instructions += (
            " For every contain_conflicts item, including a high-confidence local pair that the contain specialist marked uncertain, return one explicit contain_conflict_resolutions verdict: "
            "accept_contain with support_surface_role=internal_shelf when the original-image crop positively shows "
            "membership in an open or closed shelf/cubby/interior volume, or reject_contain_use_support with "
            "support_surface_role=outer_top|external_surface and a concrete support_anchor (floor or object_###). "
            "For multi-level furniture, every occupied functional level remains contain even though it supplies gravity support. "
            "This includes both bunks and every level of a bookcase/shelving unit, multi-tier service cart, or rack, including "
            "a plain top level. Do not preserve attach for the same pair; use support_surface_role=multi_level_tier. For an "
            "ordinary single-level anchor, a plain exterior top remains attach. Use unknown when tier membership is ambiguous. "
            "Neither relation type has an unconditional priority, and local candidate scores or bbox overlap are not proof. "
            "The attached crop is limited to these listed pairs; do not add new pairs."
        )
    return {
        "schema_version": "delivery_schema_guard_context_balanced",
        "scene_name": scene_state.get("scene_name"),
        "object_index": object_index,
        "canonical_walls": formal_wall_references(wall_evidence),
        "wall_semantic_records": _compact_wall_semantic_records(task_results),
        "relation_candidate_catalog": compact_schema_guard_relation_catalog(task_results),
        "contain_conflicts": contain_conflicts,
        "task_conclusion_summaries": task_summaries,
        "supervisor": {
            "decision": supervisor.get("decision"),
            "ready_for_schema_guard": supervisor.get("ready_for_schema_guard"),
            "reason": str(supervisor.get("reason") or "")[:320] or None,
            "requested_next_tasks": [str(item)[:180] for item in supervisor.get("requested_next_tasks", []) if item][:8],
        },
        "instructions": instructions,
    }


def _normalized_image_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= item <= 1.0 for item in (x0, y0, x1, y1)):
        return None
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def build_contain_conflict_contact_sheet(
    scene_state: dict[str, Any],
    contain_conflict_report: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Create one small original-image crop sheet only when a tie-break is needed."""
    conflicts = (contain_conflict_report or {}).get("conflicts", [])
    if not isinstance(conflicts, list) or not conflicts:
        audit = {
            "schema_version": "contain_conflict_review_manifest",
            "status": "not_needed",
            "conflict_count": 0,
            "contact_sheet_path": None,
            "items": [],
        }
        write_audit_pair(output_dir, "relations", "contain_conflict_review_manifest", audit, "Contain Conflict Review Manifest")
        return audit

    evidence = scene_state.get("evidence", {}) if isinstance(scene_state, dict) else {}
    original_path = evidence.get("original_image_path") if isinstance(evidence, dict) else None
    if not original_path or not Path(str(original_path)).exists():
        audit = {
            "schema_version": "contain_conflict_review_manifest",
            "status": "unavailable",
            "reason": "original_image_path unavailable",
            "conflict_count": len(conflicts),
            "contact_sheet_path": None,
            "items": [],
        }
        write_audit_pair(output_dir, "relations", "contain_conflict_review_manifest", audit, "Contain Conflict Review Manifest")
        return audit

    try:
        from PIL import Image, ImageDraw
        original = Image.open(str(original_path)).convert("RGB")
    except Exception as exc:  # Pillow/image decode is optional evidence only.
        audit = {
            "schema_version": "contain_conflict_review_manifest",
            "status": "unavailable",
            "reason": f"could not render local crop: {type(exc).__name__}: {exc}",
            "conflict_count": len(conflicts),
            "contact_sheet_path": None,
            "items": [],
        }
        write_audit_pair(output_dir, "relations", "contain_conflict_review_manifest", audit, "Contain Conflict Review Manifest")
        return audit

    width, height = original.size
    tiles: list[tuple[Any, str]] = []
    items: list[dict[str, Any]] = []
    for index, conflict in enumerate(conflicts[:4], start=1):
        if not isinstance(conflict, dict):
            continue
        anchor_bbox = _normalized_image_bbox(conflict.get("anchor_image_bbox"))
        target_bbox = _normalized_image_bbox(conflict.get("target_image_bbox"))
        if not anchor_bbox or not target_bbox:
            items.append(
                {
                    "conflict_id": conflict.get("conflict_id"),
                    "status": "skipped",
                    "reason": "normalized anchor/target image bbox unavailable",
                }
            )
            continue
        x0 = max(0.0, min(anchor_bbox[0], target_bbox[0]) - 0.08)
        y0 = max(0.0, min(anchor_bbox[1], target_bbox[1]) - 0.08)
        x1 = min(1.0, max(anchor_bbox[2], target_bbox[2]) + 0.08)
        y1 = min(1.0, max(anchor_bbox[3], target_bbox[3]) + 0.08)
        crop_box = (
            int(x0 * width),
            int(y0 * height),
            max(int(x1 * width), int(x0 * width) + 1),
            max(int(y1 * height), int(y0 * height) + 1),
        )
        crop = original.crop(crop_box)
        draw = ImageDraw.Draw(crop)
        for bbox, color, label in ((anchor_bbox, "#2f7df6", "A"), (target_bbox, "#ee4b3c", "T")):
            left = int((bbox[0] - x0) / max(x1 - x0, 1e-6) * crop.width)
            top = int((bbox[1] - y0) / max(y1 - y0, 1e-6) * crop.height)
            right = int((bbox[2] - x0) / max(x1 - x0, 1e-6) * crop.width)
            bottom = int((bbox[3] - y0) / max(y1 - y0, 1e-6) * crop.height)
            draw.rectangle((left, top, right, bottom), outline=color, width=max(2, crop.width // 180))
            draw.text((left + 3, max(0, top - 14)), label, fill=color)
        crop.thumbnail((720, 560))
        label = f"{index}. A={conflict.get('object_anchor')}  T={conflict.get('object_target')}"
        tiles.append((crop, label))
        items.append(
            {
                "conflict_id": conflict.get("conflict_id"),
                "status": "rendered",
                "crop_box_normalized": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                "label": label,
            }
        )

    if not tiles:
        audit = {
            "schema_version": "contain_conflict_review_manifest",
            "status": "unavailable",
            "reason": "no conflict carried usable normalized image boxes",
            "conflict_count": len(conflicts),
            "contact_sheet_path": None,
            "items": items,
        }
        write_audit_pair(output_dir, "relations", "contain_conflict_review_manifest", audit, "Contain Conflict Review Manifest")
        return audit

    sheet_width = max(tile.width for tile, _ in tiles)
    sheet_height = sum(tile.height + 28 for tile, _ in tiles)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    cursor = 0
    for tile, label in tiles:
        sheet.paste(tile, (0, cursor))
        draw.rectangle((0, cursor + tile.height, sheet_width, cursor + tile.height + 28), fill="white")
        draw.text((4, cursor + tile.height + 6), label, fill="black")
        cursor += tile.height + 28
    contact_sheet_path = output_dir / "debug" / "relations" / "contain_conflict_contact_sheet.png"
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet_path)
    audit = {
        "schema_version": "contain_conflict_review_manifest",
        "status": "available",
        "conflict_count": len(conflicts),
        "rendered_crop_count": len(tiles),
        "original_image_path": str(original_path),
        "contact_sheet_path": str(contact_sheet_path),
        "items": items,
        "policy": "The crop is attached only to the existing DeliverySchemaGuard request when a high-confidence support/contain conflict exists.",
    }
    write_audit_pair(output_dir, "relations", "contain_conflict_review_manifest", audit, "Contain Conflict Review Manifest")
    return audit

def dynamic_task_payload(
    task: dict[str, Any],
    prior_task_results: list[dict[str, Any]],
    scene_state: dict[str, Any],
    wall_evidence: dict[str, Any],
    wall_visual_evidence: dict[str, Any],
    image_paths: list[str],
) -> dict[str, Any]:
    """Serialize only the evidence needed by this Agent request.

    Complete scene and wall evidence remains on disk for local assembly and
    audit; API requests receive task-specific compact context.
    """
    task_type_l = str(task.get("task_type", "")).lower()
    canonical_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").lower()
    is_recall = is_wall_attach_recall_task(task)
    prefilter = task.get("wall_attach_prefilter") if isinstance(task.get("wall_attach_prefilter"), dict) else {}

    context = compact_task_api_context(task, prior_task_results, scene_state, wall_evidence)
    task_view = context.pop("task")
    return {
        "task": task_view,
        "scene_context": context,
        "relation_policy": {
            "allowed_relation_types": ["attach", "contain"],
            "floor_support_is_primary_but_not_exclusive": True,
            "wall_contact_can_be_additional_to_floor_support": True,
            "same_anchor_target_pair_attach_and_contain_are_mutually_exclusive": True,
            "weak_or_inferred_hidden_walls_cannot_anchor_attach": True,
        },
        "wall_review_mode": task.get("wall_review_mode") if canonical_kind == "wall_final_adjudication" else None,
        "wall_visual_image_paths": image_paths,
        "response_policy": {
            "format": "compact_structured_json",
            "reason_field": "Use a short snake_case reason_code or evidence_code instead of a prose explanation.",
            "avoid": ["long visual narration", "duplicate geometry already supplied by local context", "repeating accepted relations in multiple fields"],
        },
    }


def normalize_compact_task_response(response: dict[str, Any]) -> dict[str, Any]:
    """Make compact reason/evidence codes compatible with existing audits."""
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("reason") is None and value.get("reason_code"):
                value["reason"] = str(value["reason_code"])
            if value.get("evidence") is None and isinstance(value.get("evidence_codes"), list):
                value["evidence"] = [str(item) for item in value["evidence_codes"]]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(response)
    return response

DEFAULT_BASE_MODEL = "gpt-5.5"
DEFAULT_WALL_SPECIALIST_MODEL = "gpt-5.5"

WALL_SPECIALIST_TASK_KINDS = {
    "wall_alignment_object_review",
    "provisional_wall_visibility_review",
    "wall_visibility_semantics_review",
    "wall_final_adjudication",
    "wall_attach_candidate_recall",
}


# `balanced` is deliberately task-aware. It keeps high-detail vision only for
# decisions that depend on fine wall/contact or containment evidence.
BALANCED_TASK_IMAGE_DETAILS = {
    "support_contact_extraction": "low",
    "contain_relation_extraction": "high",
    "object_revision_check": "low",
    "object_physical_property_reasoning": "low",
    "room_material_classification": "low",
    "ceiling_judgement": "low",
    "wall_alignment_object_review": "high",
    "provisional_wall_visibility_review": "high",
    "wall_object_hypothesis": "high",
    "wall_visibility_semantics_review": "high",
    "wall_final_adjudication": "high",
    "wall_attach_candidate_recall": "high",
}

WALL_CONTEXT_TASK_KINDS = {
    "wall_alignment_object_review",
    "wall_object_hypothesis",
    "provisional_wall_visibility_review",
    "wall_visibility_semantics_review",
    "wall_final_adjudication",
    "wall_attach_candidate_recall",
}

def configured_wall_specialist_task_kinds(args: argparse.Namespace) -> set[str]:
    configured = getattr(args, "wall_specialist_task_kinds", None)
    if configured is None:
        return set(WALL_SPECIALIST_TASK_KINDS)
    values = [configured] if isinstance(configured, str) else list(configured)
    return {
        part.strip().lower()
        for item in values for part in str(item).split(",") if part.strip()
    }


def resolve_dynamic_task_model_route(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, str]:
    canonical_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").lower()
    task_id = task_identifier(task).lower()
    specialist_task_kinds = configured_wall_specialist_task_kinds(args)
    specialist_enabled = bool(
        getattr(args, "wall_specialist_model", None)
        or getattr(args, "wall_specialist_vision_model", None)
    )
    is_wall_specialist_task = (
        canonical_kind in specialist_task_kinds
        or (task_provides_capability(task, PROVISIONAL_WALL_VISIBILITY_CAPABILITY) and (
            "provisional_wall_visibility_review" in specialist_task_kinds
            or "wall_alignment_object_review" in specialist_task_kinds
        ))
        or (task_provides_capability(task, "wall_visibility_semantics_ready") and "wall_visibility_semantics_review" in specialist_task_kinds)
        or (task_id == "task_wall_final_adjudication" and "wall_final_adjudication" in specialist_task_kinds)
        or ("wall_attach_candidate_recall" in task_id and "wall_attach_candidate_recall" in specialist_task_kinds)
    )
    if specialist_enabled and is_wall_specialist_task:
        specialist_model = getattr(args, "wall_specialist_model", None) or args.model
        specialist_vision_model = (
            getattr(args, "wall_specialist_vision_model", None)
            or getattr(args, "wall_specialist_model", None)
            or args.vision_model
        )
        return {
            "model": specialist_model,
            "vision_model": specialist_vision_model,
            "model_route": "wall_specialist",
            "canonical_task_kind": canonical_kind,
        }
    return {
        "model": args.model,
        "vision_model": args.vision_model,
        "model_route": "base",
        "canonical_task_kind": canonical_kind,
    }



def task_vision_detail_overrides(args: argparse.Namespace) -> dict[str, str]:
    raw = str(getattr(args, "task_vision_detail_overrides", "") or "").strip()
    overrides: dict[str, str] = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        key, value = key.strip().lower(), value.strip().lower()
        if separator and key and value in {"low", "high", "auto"}:
            overrides[key] = value
    return overrides


def resolve_task_image_detail(args: argparse.Namespace, task: dict[str, Any]) -> str | None:
    # A global detail override takes precedence over task-aware defaults.
    if getattr(args, "vision_image_detail", None) is not None:
        return args.vision_image_detail
    canonical_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").lower()
    return task_vision_detail_overrides(args).get(canonical_kind, BALANCED_TASK_IMAGE_DETAILS.get(canonical_kind, "low"))

def execute_dynamic_task_once(
    args: argparse.Namespace,
    task: dict[str, Any],
    prior_task_results: list[dict[str, Any]],
    scene_state: dict[str, Any],
    wall_evidence: dict[str, Any],
    wall_visual_evidence: dict[str, Any],
) -> dict[str, Any]:
    image_routing = resolve_task_image_evidence(task, wall_visual_evidence)
    wall_image_paths = list(image_routing.get("image_paths", [])) if image_routing.get("matched") else wall_visual_image_paths(task, wall_visual_evidence)
    contain_image_paths = contain_task_image_paths(task, scene_state)
    image_paths = compact_image_paths(wall_image_paths + contain_image_paths, 8)
    if contain_image_paths:
        current_source = str(image_routing.get("routing_source") or "")
        image_routing["routing_source"] = "+".join(item for item in [current_source, "contain_task_profile"] if item)
        image_routing["selected_artifacts"] = list(image_routing.get("selected_artifacts", [])) + ["contain_task_images"]
    task_type_l = str(task.get("task_type", "")).lower()
    task_id = task_identifier(task)
    execution_stage = "wall_final_adjudication" if task_type_l == "wall_final_adjudication" else "dynamic_task_api"
    use_vision_model = bool(task.get("use_vision_model") or image_paths)
    payload = dynamic_task_payload(
        task, prior_task_results, scene_state, wall_evidence, wall_visual_evidence, image_paths,
    )
    model_route = resolve_dynamic_task_model_route(args, task)
    local_client = SceneDeliveryAPIClient(
        APIConfig(
            model=model_route["model"],
            vision_model=model_route["vision_model"],
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            dry_run=args.dry_run,
            image_detail=resolve_task_image_detail(args, task),
        )
    )
    response = local_client.call_json(
        "DynamicTaskAgent",
        load_prompt("dynamic_task_agent_compact_system.txt" if task.get("compiled_from_compact_manager_plan") else "dynamic_task_agent_system.txt"),
        payload,
        use_vision_model=use_vision_model,
        image_paths=image_paths,
    )
    if isinstance(response, dict):
        response = normalize_compact_task_response(response)
    return {
        "task": task,
        "task_id": task_id,
        "task_type": task_type_l,
        "execution_stage": execution_stage,
        "response": response,
        "use_vision_model": use_vision_model,
        "image_count": len(image_paths),
        "image_routing_source": image_routing.get("routing_source"),
        "selected_image_artifacts": image_routing.get("selected_artifacts", []),
        "keyword_image_fallback_used": bool(image_routing.get("keyword_fallback_used")),
        "image_detail": local_client.config.image_detail,
        **local_client.usage_metadata(),
        "retry_events": [{**event, "task_id": task_id, "canonical_task_kind": model_route["canonical_task_kind"]} for event in local_client.retry_events],
        "model": model_route["vision_model"] if use_vision_model else model_route["model"],
        "model_route": model_route["model_route"],
        "canonical_task_kind": model_route["canonical_task_kind"],
    }


def prior_results_for_task(task: dict[str, Any], task_results_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    deps = task.get("depends_on", []) if isinstance(task.get("depends_on"), list) else []
    return [task_results_by_id[dep] for dep in deps if dep in task_results_by_id]


def write_final_wall_overlay_or_skip(output_dir: Path, scene_state: dict[str, Any], final_outputs_generated: bool) -> tuple[dict[str, Any] | None, str | None]:
    overlay_path = output_dir / "debug" / "walls" / "final_walls_overlay_on_canonical_topdown.png"
    check_json = output_dir / "debug" / "walls" / "final_walls_overlay_check.json"
    check_md = output_dir / "debug" / "walls" / "final_walls_overlay_check.md"
    objects_json = output_dir / "final_outputs" / "objects.json"
    relations_json = output_dir / "final_outputs" / "relations.json"
    stale_final_outputs_detected = bool(
        not final_outputs_generated
        and (objects_json.exists() or relations_json.exists() or overlay_path.exists())
    )
    topdown = scene_state.get("topdown_grid_mapping", {}) if isinstance(scene_state.get("topdown_grid_mapping"), dict) else {}
    topdown_png = Path(str(topdown.get("canonical_topdown_image_path", ""))) if topdown.get("canonical_topdown_image_path") else None
    grid_mapping = Path(str(topdown.get("canonical_mapping_path", ""))) if topdown.get("canonical_mapping_path") else None
    if not final_outputs_generated or not objects_json.exists():
        reason = "final outputs not generated in current run"
    elif not topdown_png or not topdown_png.exists():
        reason = "final wall overlay skipped because canonical top-down image was not available"
    elif not grid_mapping or not grid_mapping.exists():
        reason = "final wall overlay skipped because canonical grid mapping was not available"
    else:
        report = visualize_final_walls_on_topdown(topdown_png, grid_mapping, objects_json, overlay_path)
        return report, None
    data = {
        "status": "skipped",
        "reason": reason,
        "output_overlay_png": str(overlay_path),
        "coordinate_source": "layout_xz",
        "topdown_usage": "visual_audit_only",
        "candidate_modified": False,
        "revision_used": False,
        "stale_final_outputs_detected": stale_final_outputs_detected,
        "stale_paths_checked": {
            "objects_json_exists": objects_json.exists(),
            "relations_json_exists": relations_json.exists(),
            "overlay_png_exists": overlay_path.exists(),
        },
        "note": "This overlay is for human verification and is generated only after final walls exist in the current run.",
    }
    write_json(check_json, data)
    write_md(
        check_md,
        "Final Walls Overlay Check",
        [
            "- status: `skipped`",
            f"- reason: {reason}",
            f"- output_overlay_png: `{overlay_path}`",
            "- coordinate_source: `layout_xz`",
            "- topdown_usage: `visual_audit_only`",
            "- candidate_modified: `False`",
            "- revision_used: `False`",
            f"- stale_final_outputs_detected: `{stale_final_outputs_detected}`",
            "- this overlay is for human verification",
        ],
    )
    return None, reason


REPAIRABLE_TASK_TYPES = {
    "infer_four_clockwise_walls_from_topdown",
    "repair_missing_final_walls",
    "wall_final_adjudication",
    "accepted_attach_merge_repair",
    "repair_missing_attach_targets",
    "repair_missing_relation_targets",
    "room_material_classification",
}


def repair_task_signature(task: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_type": task.get("task_type"),
            "reason": task.get("reason"),
            "inputs": task.get("inputs"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def strengthen_schema_repair_task(task: dict[str, Any], index: int, wall_evidence: dict[str, Any]) -> dict[str, Any]:
    repair_task = dict(task)
    task_type = repair_task.get("task_type", "schema_repair_task")
    repair_task.setdefault("task_id", f"schema_repair_{index:02d}_{task_type}")
    repair_task.setdefault("agent_role", "DynamicTaskAgent")
    repair_task.setdefault("needs_tool", False)
    if task_type in {"infer_four_clockwise_walls_from_topdown", "repair_missing_final_walls", "wall_final_adjudication"}:
        repair_task["original_task_type"] = task_type
        repair_task["task_type"] = "wall_final_adjudication"
        repair_task["task_id"] = repair_task.get("task_id") or f"schema_repair_{index:02d}_wall_final_adjudication"
        if "wall_final_adjudication" not in str(repair_task.get("task_id", "")):
            repair_task["task_id"] = f"{repair_task['task_id']}_wall_final_adjudication"
        repair_task["use_vision_model"] = True
        repair_task["schema_repair_prompt"] = (
            "You are executing the final wall adjudication step. This is not a candidate generation task. "
            "Decide whether available layout-XZ wall candidates can be promoted to final wall_001 through wall_004. "
            "Output exactly one of: A. {\"decision\":\"verified_final_walls\",\"walls\":[...]} with source obb_from_mesh_footprint_min_rect and topdown_verification passed for every wall; "
            "or B. {\"decision\":\"missing_or_invalid_formal_obb_wall_evidence\",\"reason\":\"...\"}. "
            "A layout-XZ candidate wall may be promoted or revised only when visual/layout evidence verifies the final room boundary; the visual evidence does not create coordinates. "
            "If revising candidates, compare grid/camera-like candidates against wall_line_hypotheses from likely wall-attached or against-wall objects when provided, prefer an oblique/oriented quadrilateral when it better follows the true room footprint, then return candidate_modified=true and revision_reason. "
            "Final walls may be oblique and are not required to be x-constant or z-constant. A weak/hidden wall may be completed by room closure when other walls and footprint evidence support closure; audit evidence_strength=weak, completion_method=inferred_by_room_closure, weak_wall=true, original_image_visibility=not_visible or inferred, and completion_reason. "
            + CANONICAL_WALL_NAMING_INSTRUCTION
            + " Wall object hypotheses are support evidence only; source wall_attached_object_distribution is not a final wall source. "
            "Every final wall must be a segment with first_point [x,z] and second_point [x,z]. Do not output needs_check walls, single-coordinate summaries, candidate-only walls, grid_extent_hint, layout_bounds_hint, object_footprint_bounds_hint, or candidate_room_polygon_hint as final walls."
        )
        repair_task["requirements"] = {
            "final_wall_ids": ["wall_001", "wall_002", "wall_003", "wall_004"],
            "coordinate_format": "[x, z]",
            "coordinate_source": "layout_xz",
            "must_be_clockwise": True,
            "adjacent_endpoints_must_match": True,
            "final_eligible_source": "obb_from_mesh_footprint_min_rect",
            "allowed_decisions": ["verified_final_walls", "missing_or_invalid_formal_obb_wall_evidence"],
            "canonical_naming_policy": {
                "wall_001_when_three_visible_walls": "middle visible wall",
                "wall_001_when_two_visible_walls": "left visible wall",
                "remaining_walls": "clockwise from wall_001 in top-down XZ",
                "required_audit_fields": ["original_visible_wall_count", "visible_role", "original_image_visibility", "clockwise_order_index", "wall_001_anchor_reason"],
            },
        }
        repair_task["wall_output_schema"] = {
            "verified_final_walls": {
                "decision": "verified_final_walls",
                "walls": [
                    {"id": "wall_001", "first_point": ["x", "z"], "second_point": ["x", "z"], "source": "obb_from_mesh_footprint_min_rect", "evidence": "visual/layout-XZ verification", "topdown_verification": "passed", "original_visible_wall_count": 2, "visible_role": "left_visible|middle_visible|right_visible|hidden_closure", "original_image_visibility": "visible|partly_visible|not_visible|inferred", "canonical_naming": {"wall_001_anchor_reason": "middle visible wall or left visible wall according to original-image visibility", "clockwise_order_index": 1}}
                ],
            },
            "insufficient": {"decision": "missing_or_invalid_formal_obb_wall_evidence", "reason": "why verification failed"},
        }
        repair_task["wall_evidence_payload_for_task"] = {
            "candidate_walls": wall_evidence.get("candidate_walls", []),
            "candidate_room_polygon_hint": wall_evidence.get("candidate_room_polygon_hint", {}),
            "canonical_topdown": wall_evidence.get("canonical_topdown", {}),
            "grid_mapping_summary": wall_evidence.get("grid_mapping_summary", {}),
            "hint_only_sources": wall_evidence.get("hint_only_sources", {}),
            "visual_wall_extraction": wall_evidence.get("visual_wall_extraction", {}),
            "wall_output_requirements": wall_evidence.get("wall_output_requirements", {}),
            "wall_canonical_naming_policy": wall_evidence.get("wall_canonical_naming_policy", {}),
        }
    elif task_type in {"repair_missing_attach_targets", "repair_missing_relation_targets"}:
        repair_task["use_vision_model"] = True
        repair_task["schema_repair_prompt"] = (
            "You are repairing missing attach target coverage. Every final object must appear at least once as object_target. "
            "Choose the most plausible physical anchor for each missing object. Prefer local physical contact over generic floor fallback. "
            "Use image/mask, canonical top-down, fixed layout views, layout bbox/footprint evidence, prior attach candidates, and selective render evidence if available. "
            "Return coverage_repair_relations or coverage_repair_candidates with clean attach relation/parameters plus repair_reason, evidence_used, and coverage_repair=true."
        )
        repair_task["coverage_repair_output_schema"] = {
            "coverage_repair_relations": [
                {
                    "relation": "attach",
                    "parameters": {"object_anchor": "floor/object/wall/ceiling", "object_target": "object_xxx"},
                    "repair_reason": "physical anchor reasoning",
                    "evidence_used": ["mask_overlay", "layout_bbox", "fixed_layout_views"],
                    "coverage_repair": True,
                }
            ]
        }
    elif task_type == "room_material_classification":
        repair_task["use_vision_model"] = True
        repair_task["canonical_task_kind"] = "room_material_classification"
        repair_task["provides_capabilities"] = ["room_materials_ready"]
        repair_task["requires_artifacts"] = ["original_image"]
        repair_task["schema_repair_prompt"] = (
            "Inspect the supplied original scene image and return exactly one room_materials object. "
            "It must contain the keys walls, floor, and ceiling, each mapped to a concise non-empty English material/finish phrase suitable for rendering. "
            "Do not copy examples unless they visually match; do not return relations, wall coordinates, objects, or unsupported hidden construction claims."
        )
        repair_task["expected_output_schema"] = {
            "room_materials": {
                "walls": "concise visible wall material/finish",
                "floor": "concise visible floor material/finish",
                "ceiling": "concise visible ceiling material/finish",
            }
        }
    return repair_task


def compact_missing_relation_repair_payload(
    repair_task: dict[str, Any],
    scene_state: dict[str, Any],
    task_results: list[dict[str, Any]],
    wall_evidence: dict[str, Any],
    assembly: dict[str, Any],
) -> dict[str, Any]:
    """Project missing-target repair evidence without forwarding raw geometry."""
    missing_ids = {
        str(object_id)
        for object_id in (repair_task.get("missing_object_targets") or assembly.get("missing_object_targets") or [])
        if object_id
    }
    relation_catalog = compact_schema_guard_relation_catalog(task_results)
    relevant_catalog = [
        item
        for item in relation_catalog
        if str(item.get("object_target")) in missing_ids
        or str(item.get("object_anchor")) in missing_ids
    ]
    current_relations = assembly.get("relations_payload") or []
    related_object_ids = set(missing_ids)
    for item in relevant_catalog:
        for field in ("object_anchor", "object_target"):
            value = str(item.get(field) or "")
            if value.startswith("object_"):
                related_object_ids.add(value)
    for relation in current_relations:
        parameters = relation.get("parameters", {}) if isinstance(relation, dict) else {}
        for field in ("object_anchor", "object_target"):
            value = str(parameters.get(field) or "")
            if value in missing_ids or (field == "object_anchor" and str(parameters.get("object_target") or "") in missing_ids):
                if value.startswith("object_"):
                    related_object_ids.add(value)

    return {
        "schema_version": "missing_relation_repair_context_compact",
        "scene_name": scene_state.get("scene_name"),
        "schema_repair": True,
        "task": {
            "task_id": repair_task.get("task_id"),
            "task_type": repair_task.get("task_type"),
            "missing_object_targets": sorted(missing_ids),
            "instruction": repair_task.get("instruction"),
        },
        "schema_repair_prompt": repair_task.get("schema_repair_prompt"),
        "missing_object_index": compact_object_index(scene_state, missing_ids),
        "related_object_index": compact_object_index(scene_state, related_object_ids),
        "canonical_walls": formal_wall_references(wall_evidence),
        "wall_semantic_records": _compact_wall_semantic_records(task_results),
        "relation_candidate_catalog": relevant_catalog,
        "current_accepted_relations": current_relations,
        "coverage_repair_output_schema": repair_task.get("coverage_repair_output_schema", {}),
        "policy": [
            "Return a relation only for each listed missing object target.",
            "Prefer an existing supported candidate and preserve canonical wall ids.",
            "Do not invent geometry or return objects/walls.",
            "Use floor only when no stronger physical object, wall, ceiling, or contain anchor is supported.",
        ],
    }


def build_schema_repair_payload(
    repair_task: dict[str, Any],
    scene_state: dict[str, Any],
    task_results: list[dict[str, Any]],
    wall_evidence: dict[str, Any],
    assembly: dict[str, Any],
    wall_visual_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task": repair_task,
        "task_type": repair_task.get("task_type"),
        "requirements": repair_task.get("requirements", {}),
        "schema_repair": True,
        "schema_repair_prompt": repair_task.get("schema_repair_prompt"),
        "previous_finalization_errors": assembly.get("errors", []),
        "previous_wall_errors": assembly.get("wall_errors", []),
        "previous_repair_tasks": assembly.get("repair_tasks", []),
        "scene_state": scene_state,
        "wall_evidence": wall_evidence,
        "candidate_walls": wall_evidence.get("candidate_walls", []),
        "candidate_room_polygon_hint": wall_evidence.get("candidate_room_polygon_hint", {}),
        "wall_final_adjudication_prompt": load_prompt("wall_final_adjudication_task.txt") if repair_task.get("task_type") == "wall_final_adjudication" else None,
        "wall_object_hypotheses": extract_wall_object_hypotheses(task_results),
        "candidate_wall_attach_evidence": extract_wall_attach_candidate_evidence(task_results),
        "canonical_topdown": wall_evidence.get("canonical_topdown", {}),
        "object_candidates": scene_state.get("object_candidates", {}),
        "object_footprints": wall_evidence.get("object_footprints", []),
        "current_accepted_relations": assembly.get("relations_payload") or [],
        "missing_object_targets": repair_task.get("missing_object_targets") or assembly.get("missing_object_targets", []),
        "candidate_anchor_objects": scene_state.get("object_candidates", {}).get("items", []),
        "original_image_path": scene_state.get("evidence", {}).get("original_image_path"),
        "mask_overlay_path": scene_state.get("evidence", {}).get("mask_overlay_path"),
        "fixed_layout_views_manifest": scene_state.get("evidence", {}).get("fixed_layout_views", {}).get("manifest_path"),
        "wall_visual_evidence": wall_visual_payload(wall_visual_evidence),
        "wall_visual_image_paths": wall_visual_image_paths(repair_task, wall_visual_evidence),
        "selective_render_tool_capability": "target object plus candidate anchor objects; top/front/back/left/right evidence may be requested by task",
        "wall_output_schema": repair_task.get("wall_output_schema", {}),
        "coverage_repair_output_schema": repair_task.get("coverage_repair_output_schema", {}),
        "prior_task_results": task_results,
        "policy_snippets": [
            "This is a schema repair task, not general planning.",
            "Return exactly four final-eligible layout-XZ walls or missing_or_invalid_formal_obb_wall_evidence when repairing walls.",
            "For missing attach target coverage, return clean attach proposals with coverage_repair=true and repair_reason/evidence_used.",
            "Do not cite hint-only sources as final wall source and do not default to floor when local physical contact evidence suggests an object anchor.",
            "Wall object hypotheses may support wall_final_adjudication, but wall_attached_object_distribution is not final-eligible.",
            "Final walls may be oblique; infer the oriented room footprint rather than accepting a loose grid/camera extent rectangle.",
            "A hidden/weak wall may be completed by room closure when the room footprint is otherwise well supported; audit weak_wall completion fields.",
            "Every final wall must have first_point and second_point layout-XZ endpoints; single-coordinate summaries are invalid.",
        ],
    }


def write_schema_repair_task_results(output_dir: Path, data: dict[str, Any]) -> None:
    write_json(output_dir / "debug" / "agents" / "schema_repair_task_results.json", data)
    lines = [
        f"- triggered: `{data.get('triggered')}`",
        f"- executed_count: `{data.get('executed_count')}`",
        f"- skipped_count: `{data.get('skipped_count')}`",
        f"- reassembled: `{data.get('reassembled')}`",
        f"- final_objects_after_repair: `{data.get('final_objects_after_repair')}`",
        f"- final_relations_after_repair: `{data.get('final_relations_after_repair')}`",
        "",
        "## Results",
    ]
    for item in data.get("items", []):
        lines.append(f"- `{item.get('task_id')}` ({item.get('task_type')}): {item.get('status')} final_eligible_walls=`{item.get('final_eligible_walls')}`")
    write_md(output_dir / "debug" / "agents" / "schema_repair_task_results.md", "Schema Repair Task Results", lines)


def write_api_retry_log(output_dir: Path, client: SceneDeliveryAPIClient) -> None:
    events = list(getattr(client, "retry_events", []) or [])
    if not events:
        return
    data = {
        "status": "retry_events_recorded",
        "event_count": len(events),
        "events": events,
        "api_key_redacted": True,
    }
    write_audit_pair(output_dir, "agents", "api_retry_log", data, "API Retry Log")


def write_working_state_snapshot(
    output_dir: Path,
    stage: str,
    scene_state: dict[str, Any],
    manager_plan: dict[str, Any] | None,
    task_results: list[dict[str, Any]],
    supervisor_result: dict[str, Any] | None,
    api_call_count: int,
) -> None:
    data = {
        "stage": stage,
        "api_call_count_so_far": api_call_count,
        "scene_name": scene_state.get("scene_name"),
        "object_candidate_count": len(scene_state.get("object_candidates", {}).get("items", [])),
        "manager_plan_task_count": len(manager_plan.get("tasks", [])) if isinstance(manager_plan, dict) else None,
        "task_result_count": len(task_results),
        "task_result_ids": [result.get("task_id") for result in task_results if isinstance(result, dict)],
        "supervisor_result": supervisor_result,
    }
    write_json(output_dir / "debug" / "agents" / "working_state_latest.json", data)
    write_md(
        output_dir / "debug" / "agents" / "working_state_latest.md",
        "Working State Snapshot",
        [
            f"- stage: `{stage}`",
            f"- api_call_count_so_far: `{api_call_count}`",
            f"- object_candidate_count: `{data['object_candidate_count']}`",
            f"- task_result_count: `{len(task_results)}`",
        ],
    )


def write_api_failure_report(
    output_dir: Path,
    failed_stage: str,
    exc: BaseException,
    api_call_count_so_far: int,
    local_schema_fallback_attempted: bool,
    next_suggested_command: str,
) -> dict[str, Any]:
    data = {
        "failed_stage": failed_stage,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "api_call_count_so_far": api_call_count_so_far,
        "final_objects_existed": (output_dir / "final_outputs" / "objects.json").exists(),
        "final_relations_existed": (output_dir / "final_outputs" / "relations.json").exists(),
        "local_schema_fallback_attempted": local_schema_fallback_attempted,
        "api_retry_log_path": str(output_dir / "debug" / "agents" / "api_retry_log.json"),
        "next_suggested_command": next_suggested_command,
        "api_key_redacted": True,
    }
    if isinstance(exc, APIClientError):
        data["retry_events"] = exc.retry_events
        data["raw_response_saved"] = bool(exc.raw_response)
    write_audit_pair(output_dir, "agents", "api_failure_report", data, "API Failure Report")
    return data


def response_has_final_eligible_walls(response: dict[str, Any]) -> bool:
    walls = response.get("walls") or response.get("accepted_walls") or response.get("wall_candidates")
    if not isinstance(walls, list):
        return False
    required = {"wall_001", "wall_002", "wall_003", "wall_004"}
    ids = {wall.get("id") for wall in walls if isinstance(wall, dict)}
    if not required.issubset(ids):
        return False
    return all(
        isinstance(wall, dict)
        and wall.get("id") in required
        and wall.get("source") in {"obb_from_mesh_footprint_min_rect", "obb_from_mesh_footprint_min_rect"}
        and bool(wall.get("evidence"))
        for wall in walls
        if isinstance(wall, dict) and wall.get("id") in required
    )


def execute_schema_repair_loop(
    client: SceneDeliveryAPIClient,
    output_dir: Path,
    scene_state: dict[str, Any],
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any],
    wall_evidence: dict[str, Any],
    assembly: dict[str, Any],
    wall_visual_evidence: dict[str, Any] | None = None,
    runtime_tracker: RuntimeTracker | None = None,
) -> tuple[dict[str, Any], int]:
    executed_signatures: set[str] = set()
    repair_log: dict[str, Any] = {
        "triggered": bool(assembly.get("repair_tasks")),
        "items": [],
        "executed_count": 0,
        "skipped_count": 0,
        "reassembled": False,
        "initial_errors": assembly.get("errors", []),
    }
    current_assembly = assembly
    while current_assembly.get("repair_tasks"):
        runnable = []
        for index, task in enumerate(current_assembly.get("repair_tasks", []), start=1):
            if not isinstance(task, dict):
                continue
            task_type = task.get("task_type")
            signature = repair_task_signature(task)
            if task_type not in REPAIRABLE_TASK_TYPES:
                repair_log["items"].append({"task_type": task_type, "status": "skipped_unrepairable_task_type"})
                repair_log["skipped_count"] += 1
                continue
            if signature in executed_signatures:
                repair_log["items"].append({"task_type": task_type, "status": "skipped_no_new_repair_signature"})
                repair_log["skipped_count"] += 1
                continue
            runnable.append((index, task, signature))
        if not runnable:
            break
        for index, task, signature in runnable:
            executed_signatures.add(signature)
            repair_task = strengthen_schema_repair_task(task, index, wall_evidence)
            if repair_task.get("task_type") in {"repair_missing_attach_targets", "repair_missing_relation_targets"}:
                payload = compact_missing_relation_repair_payload(
                    repair_task, scene_state, task_results, wall_evidence, current_assembly
                )
                image_paths = []
            else:
                payload = build_schema_repair_payload(
                    repair_task, scene_state, task_results, wall_evidence, current_assembly, wall_visual_evidence
                )
                image_paths = wall_visual_image_paths(repair_task, wall_visual_evidence)
            if runtime_tracker is not None and not client.config.dry_run:
                with runtime_tracker.api_call(
                    "schema_repair_loop",
                    role="DynamicTaskAgent",
                    task_id=str(repair_task.get("task_id") or f"schema_repair_{index:02d}"),
                    task_type=str(repair_task.get("task_type") or "schema_repair"),
                    use_vision_model=bool(repair_task.get("use_vision_model") or image_paths),
                    image_count=len(image_paths),
                ):
                    response = client.call_json(
                        "DynamicTaskAgent",
                        load_prompt("dynamic_task_agent_system.txt"),
                        payload,
                        use_vision_model=bool(repair_task.get("use_vision_model") or image_paths),
                        image_paths=image_paths,
                    )
            else:
                with runtime_tracker.stage("schema_repair_loop") if runtime_tracker is not None else nullcontext():
                    response = client.call_json(
                        "DynamicTaskAgent",
                        load_prompt("dynamic_task_agent_system.txt"),
                        payload,
                        use_vision_model=bool(repair_task.get("use_vision_model") or image_paths),
                        image_paths=image_paths,
                    )
            if runtime_tracker is not None and not client.config.dry_run and runtime_tracker.api_calls:
                runtime_tracker.api_calls[-1].update(
                    client.usage_metadata(),                )
            if isinstance(response, dict):
                response.setdefault("task_id", repair_task.get("task_id"))
                response.setdefault("task_type", repair_task.get("task_type"))
                response.setdefault("canonical_task_kind", repair_task.get("canonical_task_kind") or repair_task.get("task_type"))
                response.setdefault("provided_capabilities", task_capabilities(repair_task, "provides_capabilities"))
            write_task_output(output_dir, repair_task, response)
            task_results.append(response)
            repair_log["items"].append(
                {
                    "task_id": repair_task.get("task_id"),
                    "task_type": repair_task.get("task_type"),
                    "status": response.get("status"),
                    "prompt_payload": payload,
                    "response": response,
                    "final_eligible_walls": response_has_final_eligible_walls(response),
                }
            )
            repair_log["executed_count"] += 1
        next_assembly = assemble_final_payloads(
            scene_state=scene_state,
            task_results=task_results,
            guard_result=guard_result,
            output_dir=output_dir,
            dry_run=False,
            wall_evidence=wall_evidence,
        )
        repair_log["reassembled"] = True
        if next_assembly.get("objects_payload") is not None or next_assembly.get("errors") != current_assembly.get("errors"):
            current_assembly = next_assembly
            if current_assembly.get("objects_payload") is not None:
                break
            continue
        current_assembly = next_assembly
        break
    repair_log["final_errors"] = current_assembly.get("errors", [])
    repair_log["final_objects_after_repair"] = current_assembly.get("objects_payload") is not None
    repair_log["final_relations_after_repair"] = current_assembly.get("relations_payload") is not None
    write_schema_repair_task_results(output_dir, repair_log)
    return current_assembly, repair_log.get("executed_count", 0)


def run_delivery(args: argparse.Namespace) -> dict[str, Any]:
    wall_candidate_config = configure_wall_candidate_environment(args)
    output_dir = args.output_dir.resolve()
    run_root = args.run_root.resolve()
    render_script = args.render_script.resolve()
    if getattr(args, "relation_layout_dir", None):
        os.environ[RELATION_LAYOUT_DIR_ENV] = str(args.relation_layout_dir.resolve())
    else:
        os.environ.pop(RELATION_LAYOUT_DIR_ENV, None)
    ensure_output_tree(output_dir)
    runtime_tracker = RuntimeTracker()

    with runtime_tracker.stage("policy_generation"):
        policy_result = load_static_policy()
        copy_policy_to_run(output_dir)

    with runtime_tracker.stage("scene_state_build"):
        scene_state = build_scene_state(args.scene_name, run_root, render_script)
        write_scene_state(scene_state, output_dir)

    initialize_empty_audits(output_dir, dry_run=args.dry_run)
    render_capabilities = scan_render_script_capabilities(render_script)
    write_render_audit(output_dir, render_capabilities, args.dry_run)

    client = SceneDeliveryAPIClient(
        APIConfig(
            model=args.model,
            vision_model=args.vision_model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            dry_run=args.dry_run,
            image_detail=args.vision_image_detail,
        )
    )
    api_call_count = 0
    manager_client = SceneDeliveryAPIClient(
        APIConfig(
            model=getattr(args, "manager_model", None) or args.model,
            vision_model=getattr(args, "manager_vision_model", None) or args.vision_model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            dry_run=args.dry_run,
            image_detail=getattr(args, "manager_vision_image_detail", "low"),
        )
    )
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    manager_plan_skeleton: dict[str, Any] | None = None
    physics_sidecar_process: subprocess.Popen[Any] | None = None
    physics_sidecar: dict[str, Any] = {
        "mode": "async",
        "status": "not_started",
        "relation_critical_path_dependency": False,
    }
    with runtime_tracker.stage("expected_manifest_readiness"):
        expected_manifest, expected_manifest_readiness_audit = wait_for_expected_relation_manifest(
            args,
            run_root,
        )
    write_json(
        output_dir / "debug" / "agents" / "expected_manifest_readiness.json",
        expected_manifest_readiness_audit,
    )
    if expected_manifest_readiness_audit.get("status") != "ready":
        raise RuntimeError(
            "Manager planning refused an incomplete expected_manifest: "
            f"{expected_manifest_readiness_audit.get('reason')}"
        )
    with runtime_tracker.stage("physics_sidecar_launch"):
        physics_sidecar_process, physics_sidecar = launch_async_physics_sidecar(
            args,
            expected_manifest,
            output_dir,
        )
    skeleton_payload = build_manager_skeleton_payload(args, expected_manifest, scene_state, render_capabilities)
    skeleton_image_paths = manager_skeleton_image_paths(args, expected_manifest)
    if skeleton_image_paths:
        skeleton_payload["manager_planning_images"] = {
            "original_image_path": skeleton_image_paths[0],
            "purpose": "Use this single original image with the object list to decide whether contain_relation_extraction is needed before slower precompute artifacts finish.",
        }
    if args.dry_run:
        with runtime_tracker.stage("manager_planning_skeleton"):
            manager_plan_skeleton = manager_client.call_json(
                "SceneDeliveryManager",
                load_prompt(manager_prompt_name()),
                skeleton_payload,
                use_vision_model=bool(skeleton_image_paths),
                image_paths=skeleton_image_paths,
            )
    else:
        with runtime_tracker.api_call("manager_planning_skeleton", role="SceneDeliveryManager", task_id="manager_plan_skeleton", model=(getattr(args, "manager_model", None) or args.model)):
            manager_plan_skeleton = manager_client.call_json(
                "SceneDeliveryManager",
                load_prompt(manager_prompt_name()),
                skeleton_payload,
                use_vision_model=bool(skeleton_image_paths),
                image_paths=skeleton_image_paths,
            )
        api_call_count += 1
        annotate_manager_api_call(
            runtime_tracker,
            manager_client,
            skeleton_payload,
            load_prompt(manager_prompt_name()),
            args,
            manager_plan_skeleton,
        )
    if isinstance(manager_plan_skeleton, dict):
        manager_plan_skeleton, _manager_compilation_audit = compile_manager_plan_response(args, manager_plan_skeleton, output_dir)
        manager_plan_skeleton["manager_planning_mode"] = "skeleton_async"
        manager_plan_skeleton["relation_layout_expected_manifest_path"] = (expected_manifest or {}).get("expected_manifest_path") if isinstance(expected_manifest, dict) else None
        write_manager_skeleton_plan(output_dir, manager_plan_skeleton)
    client.retry_events.extend(manager_client.retry_events)
    write_api_retry_log(output_dir, client)

    render_grid_precompute_result: dict[str, Any] | None = None
    with runtime_tracker.stage("render_grid_wall_precompute_load"):
        render_grid_precompute_result = wait_for_relation_precompute(args)
    if isinstance(render_grid_precompute_result, dict) and render_grid_precompute_result.get("status") == "available":
        scene_state = apply_render_grid_precompute_to_scene_state(scene_state, render_grid_precompute_result)
        wall_candidate_config["render_grid_wall_precompute_path"] = render_grid_precompute_result.get("method_candidates_path")
        wall_candidate_config["render_grid_wall_precompute_dir"] = render_grid_precompute_result.get("precompute_dir")
        wall_candidate_config["render_grid_wall_precompute_status"] = render_grid_precompute_result.get("status")
        wall_candidate_config["render_grid_wall_precompute_source"] = "external_precompute"
        write_scene_state(scene_state, output_dir)
    else:
        runtime_tracker.mark_skipped("render_grid_wall_precompute", "formal external precompute is required")
        raise RuntimeError(
            "Formal relation extraction requires a candidate-local relation layout manifest, "
            "--render_grid_wall_precompute_dir, or --render_grid_wall_precompute_path. "
            "Generate it first with "
            "vlm_agent.reconstruction.relations.tools.build_relation_layout_from_sam3d."
        )

    runtime_tracker.start("canonical_topdown_generation")
    if isinstance(render_grid_precompute_result, dict) and render_grid_precompute_result.get("status") in {"available", "missing_primary"} and render_grid_precompute_result.get("canonical_mapping_path"):
        canonical_result = {
            "status": "available",
            "generated": False,
            "canonical_mapping_path": render_grid_precompute_result.get("canonical_mapping_path"),
            "canonical_topdown_image_path": render_grid_precompute_result.get("canonical_topdown_image_path"),
            "source": "render_grid_wall_precompute",
            "fallback_used": False,
            "reused_from_render_grid_wall_precompute": True,
            "precompute_manifest_path": render_grid_precompute_result.get("manifest_path"),
        }
    else:
        canonical_result = ensure_canonical_topdown_mapping(
            scene_state,
            dry_run=args.dry_run,
        )
    runtime_tracker.end("canonical_topdown_generation", status="success" if canonical_result.get("status") == "available" or args.dry_run else "failed", result_status=canonical_result.get("status"))
    runtime_tracker.record_render_call("canonical_topdown_generation", success=canonical_result.get("status") == "available" or args.dry_run, result_status=canonical_result.get("status"), cache_status="reused_from_render_grid_wall_precompute" if canonical_result.get("reused_from_render_grid_wall_precompute") else None)
    if not args.dry_run and canonical_result.get("status") != "available":
        validation = {
            "dry_run": False,
            "valid": False,
            "finalization_status": "canonical_topdown_regeneration_failed",
            "canonical_result": canonical_result,
        }
        write_audit_pair(output_dir, "final_validation", "schema_validation", validation, "Schema Validation")
        runtime_tracker.mark_skipped("fixed_layout_views_generation", "canonical topdown regeneration failed")
        runtime_tracker.mark_skipped("wall_evidence_build", "canonical topdown regeneration failed")
        runtime_tracker.mark_skipped("wall_visual_evidence_build", "canonical topdown regeneration failed")
        runtime_summary = runtime_tracker.write(output_dir)
        final_process_report = write_final_process_report(
            output_dir=output_dir,
            scene_state=scene_state,
            manager_plan={},
            task_results=[],
            validation=validation,
            wall_evidence={},
            wall_visual_evidence={},
            api_called=False,
            api_call_count=0,
            runtime_summary=runtime_summary,
            finalization_status="canonical_topdown_regeneration_failed",
        )
        return {
            "scene_name": args.scene_name,
            "output_dir": str(output_dir),
            "dry_run": args.dry_run,
            "api_called": False,
            "api_call_count": 0,
            "final_outputs_generated": False,
            "canonical_topdown_regeneration": canonical_result,
            "schema_validation": validation,
            "runtime_summary_path": runtime_summary.get("json_path"),
            "final_process_report_path": final_process_report.get("md_path"),
        }
    with runtime_tracker.stage("scene_state_build"):
        scene_state = build_scene_state(args.scene_name, run_root, render_script)
        scene_state = apply_render_grid_precompute_to_scene_state(scene_state, render_grid_precompute_result)
        write_scene_state(scene_state, output_dir)
    fixed_views_output_dir = output_dir / "generated_evidence" / "views"
    fixed_views_result: dict[str, Any] | None = None
    fixed_views_source_dir = wait_for_fixed_views_precompute_dir(args)
    if fixed_views_source_dir and not args.disable_evidence_cache:
        runtime_tracker.start("fixed_layout_views_load")
        fixed_views_result = reuse_fixed_layout_views_cache(
            scene_state,
            fixed_views_source_dir,
            fixed_views_output_dir,
        )
        runtime_tracker.end(
            "fixed_layout_views_load",
            status="success" if fixed_views_result.get("status") == "available" else "failed",
            result_status=fixed_views_result.get("status"),
            cache_status=fixed_views_result.get("cache_status"),
            source_dir=str(fixed_views_source_dir),
        )
        if fixed_views_result.get("status") != "available":
            fixed_views_result = None
    elif fixed_views_source_dir and args.disable_evidence_cache:
        runtime_tracker.mark_skipped("fixed_layout_views_load", "--disable_evidence_cache ignored explicit fixed views precompute")
    else:
        runtime_tracker.mark_skipped("fixed_layout_views_load", "no external fixed views precompute configured")

    if fixed_views_result is not None:
        runtime_tracker.mark_skipped("fixed_layout_views_generation", "external fixed views precompute loaded")
    else:
        runtime_tracker.mark_skipped(
            "fixed_layout_views_generation",
            "optional fixed views were not precomputed; the formal runner does not render evidence in-run",
        )
    with runtime_tracker.stage("scene_state_build"):
        scene_state = build_scene_state(args.scene_name, run_root, render_script)
        scene_state = apply_render_grid_precompute_to_scene_state(scene_state, render_grid_precompute_result)
        if isinstance(fixed_views_result, dict) and fixed_views_result.get("status") == "available":
            scene_state.setdefault("evidence", {})["fixed_layout_views"] = fixed_views_result
        write_scene_state(scene_state, output_dir)
    with runtime_tracker.stage("wall_evidence_build"):
        wall_evidence = build_and_write_wall_evidence(scene_state, output_dir)
    with runtime_tracker.stage("obb_wall_risk_audit"):
        obb_wall_risk_audit = build_obb_wall_risk_audit(wall_evidence)
        write_obb_wall_risk_audit(output_dir, obb_wall_risk_audit)
        wall_evidence["formal_obb_risk_audit"] = obb_wall_risk_audit
        resolved_wall_review_mode = resolve_wall_review_mode(args.wall_review_mode, obb_wall_risk_audit)
    with runtime_tracker.stage("wall_visual_evidence_build"):
        wall_visual_evidence = ensure_wall_visual_evidence(
            scene_state,
            output_dir / "debug" / "walls" / "visual_review",
        )
    scene_state.setdefault("evidence", {})["wall_visual_evidence"] = wall_visual_payload(wall_visual_evidence)
    write_scene_state(scene_state, output_dir)

    if isinstance(manager_plan_skeleton, dict):
        with runtime_tracker.stage("manager_plan_hydration"):
            manager_plan = hydrate_manager_plan(manager_plan_skeleton, run_root, scene_state)
    elif args.dry_run:
        with runtime_tracker.stage("manager_planning"):
            manager_plan = manager_client.call_json(
                "SceneDeliveryManager",
                load_prompt(manager_prompt_name()),
                build_manager_planning_payload(args, scene_state, render_capabilities, wall_evidence, wall_visual_evidence),
            )
    else:
        with runtime_tracker.api_call("manager_planning", role="SceneDeliveryManager", task_id="manager_plan", model=(getattr(args, "manager_model", None) or args.model)):
            manager_plan = manager_client.call_json(
                "SceneDeliveryManager",
                load_prompt(manager_prompt_name()),
                build_manager_planning_payload(args, scene_state, render_capabilities, wall_evidence, wall_visual_evidence),
            )
        api_call_count += 1
        manager_payload_for_metrics = build_manager_planning_payload(
            args, scene_state, render_capabilities, wall_evidence, wall_visual_evidence
        )
        annotate_manager_api_call(
            runtime_tracker,
            manager_client,
            manager_payload_for_metrics,
            load_prompt(manager_prompt_name()),
            args,
            manager_plan,
        )
    if manager_plan_skeleton is None:
        manager_plan, _manager_compilation_audit = compile_manager_plan_response(args, manager_plan, output_dir)
    client.retry_events.extend(manager_client.retry_events)
    write_api_retry_log(output_dir, client)
    manager_plan = apply_wall_task_policy(
        manager_plan,
        wall_evidence,
        wall_candidate_config["wall_candidate_mode"],
        wall_review_mode=resolved_wall_review_mode,
        obb_risk_audit=obb_wall_risk_audit,
    )
    manager_plan = ensure_required_relation_capabilities(manager_plan, run_root, scene_state)
    update_manager_compilation_audit_with_wall_policy(output_dir, manager_plan)
    manager_plan["wall_review_routing"] = {
        "requested_mode": args.wall_review_mode,
        "resolved_mode": resolved_wall_review_mode,
        "risk_level": obb_wall_risk_audit.get("risk_level"),
        "risk_score": obb_wall_risk_audit.get("risk_score"),
        "recommendation": obb_wall_risk_audit.get("recommendation"),
        "audit_path": str(output_dir / "debug" / "walls" / "obb_wall_risk_audit.json"),
    }
    manager_plan = apply_relation_policy_and_dependencies(manager_plan)
    manager_plan = apply_runtime_preset(manager_plan)
    manager_plan = hydrate_manager_plan(manager_plan, run_root, scene_state)
    dependency_layers = task_dependency_layers(manager_plan.get("tasks", []))
    manager_plan["task_model_routing"] = {
        "base_model": args.model,
        "base_vision_model": args.vision_model,
        "manager_plan_format": "compact",
        "manager_model": getattr(args, "manager_model", None) or args.model,
        "manager_vision_model": getattr(args, "manager_vision_model", None) or args.vision_model,
        "manager_vision_image_detail": getattr(args, "manager_vision_image_detail", "low"),
        "wall_specialist_enabled": bool(args.wall_specialist_model or args.wall_specialist_vision_model),
        "wall_specialist_model": args.wall_specialist_model,
        "wall_specialist_vision_model": args.wall_specialist_vision_model or args.wall_specialist_model,
        "wall_specialist_task_kinds": sorted(configured_wall_specialist_task_kinds(args)),
        "policy": "Manager DAG is unchanged; only the listed dynamic task kinds use the specialist route.",
    }
    manager_plan["task_execution_policy"] = {
        "mode": "dependency_layers",
        "max_parallel_tasks": args.max_parallel_tasks,
        "layer_count": len(dependency_layers),
        "layers": [[task_identifier(task) for task in layer] for layer in dependency_layers],
    }
    manager_plan["wall_candidate_config"] = wall_candidate_config
    write_manager_plan(output_dir, manager_plan)

    task_results: list[dict[str, Any]] = []
    task_results_by_id: dict[str, dict[str, Any]] = {}
    runtime_tracker.start("dynamic_tasks_total")
    task_order = {
        task_identifier(task): index
        for index, task in enumerate(manager_plan.get("tasks", []))
        if isinstance(task, dict)
    }
    wall_orientation_selection_done = False
    selected_orientation_candidate_id: str | None = None
    provisional_wall_visibility_binding_result: dict[str, Any] | None = None
    provisional_wall_visibility_binding_audit: dict[str, Any] | None = None
    for layer_index, layer in enumerate(dependency_layers, start=1):
        layer_stage = f"dynamic_task_layer_{layer_index:02d}"
        recall_tasks = [
            task for task in layer
            if is_wall_attach_recall_task(task)
            or task_requires_capability(task, "wall_attach_candidates_ready")
        ]
        if recall_tasks:
            runtime_tracker.start("wall_attach_local_prefilter")
            semantic_provider_ids = [
                task_identifier(task)
                for task in manager_plan.get("tasks", [])
                if isinstance(task, dict)
                and task_provides_capability(task, "wall_visibility_semantics_ready")
            ]
            semantic_provider_results = [
                task_results_by_id[task_id]
                for task_id in semantic_provider_ids
                if task_id in task_results_by_id
            ]
            if provisional_wall_visibility_binding_result is not None:
                semantic_provider_results.append(provisional_wall_visibility_binding_result)
            prefilter = build_wall_attach_prefilter(
                scene_state,
                wall_evidence,
                semantic_provider_results,
                extract_wall_object_hypotheses(list(task_results_by_id.values())),
                object_anchor_routing=manager_plan.get("object_anchor_routing", []),
            )
            prefilter_json_path = output_dir / "debug" / "relations" / "wall_attach_prefilter.json"
            prefilter_overlay_path = output_dir / "debug" / "walls" / "wall_attach_prefilter_overlay.png"
            overlay_result = render_wall_attach_prefilter_overlay(
                wall_visual_evidence.get("canonical_topdown_path"),
                wall_visual_evidence.get("render_grid_affine_path"),
                prefilter_overlay_path,
                wall_evidence,
                prefilter,
            )
            prefilter["artifact_path"] = str(prefilter_json_path)
            prefilter["overlay_status"] = overlay_result.get("status")
            prefilter["overlay_path"] = overlay_result.get("output_path")
            prefilter["overlay_result"] = overlay_result
            if overlay_result.get("status") == "available" and overlay_result.get("output_path"):
                wall_visual_evidence["wall_attach_prefilter_overlay_path"] = overlay_result["output_path"]
            compact_prefilter = {
                key: prefilter.get(key)
                for key in [
                    "schema_version", "status", "capability_contract", "policy",
                    "formal_wall_references", "wall_semantics", "semantic_binding_audit",
                    "blocked_wall_ids", "raw_object_wall_pair_count", "candidate_pair_count",
                    "candidate_object_count", "candidate_pairs",
                    "supplemental_visual_review_object_ids", "artifact_path",
                    "semantic_visibility_geometry_conflicts",
                    "overlay_status", "overlay_path",
                ]
            }
            scope_hydration_audits: list[dict[str, Any]] = []
            for recall_task in recall_tasks:
                recall_task["wall_attach_prefilter"] = compact_prefilter
                recall_task.setdefault("evidence_binding", {})["wall_attach_prefilter"] = str(prefilter_json_path)
                scope_hydration_audits.append({
                    "task_id": task_identifier(recall_task),
                    **hydrate_wall_attach_candidate_scope(recall_task, prefilter),
                })
            prefilter["scope_hydration_audits"] = scope_hydration_audits
            write_audit_pair(output_dir, "relations", "wall_attach_prefilter", prefilter, "Wall Attach Local Prefilter")
            write_manager_plan(output_dir, manager_plan)
            runtime_tracker.end(
                "wall_attach_local_prefilter",
                status="success",
                raw_pair_count=prefilter.get("raw_object_wall_pair_count", 0),
                candidate_pair_count=prefilter.get("candidate_pair_count", 0),
                scope_added_object_count=sum(
                    len(item.get("added_candidate_object_ids", []))
                    for item in scope_hydration_audits
                ),
                blocked_wall_ids=sorted(prefilter.get("blocked_wall_ids", {})),
                overlay_status=overlay_result.get("status"),
            )
        runtime_tracker.start(layer_stage)
        layer_results: list[dict[str, Any]] = []
        future_to_task: dict[Any, dict[str, Any]] = {}
        failed_task_id = "unknown_task"

        def run_one_layer_task(layer_task: dict[str, Any]) -> dict[str, Any]:
            return execute_dynamic_task_once(
                args,
                layer_task,
                prior_results_for_task(layer_task, task_results_by_id),
                scene_state,
                wall_evidence,
                wall_visual_evidence,
            )

        try:
            if args.max_parallel_tasks > 1 and len(layer) > 1:
                with ThreadPoolExecutor(max_workers=min(args.max_parallel_tasks, len(layer))) as executor:
                    future_to_task = {executor.submit(run_one_layer_task, task): task for task in layer}
                    for future in as_completed(future_to_task):
                        failed_task_id = task_identifier(future_to_task[future])
                        layer_results.append(future.result())
            else:
                for task in layer:
                    failed_task_id = task_identifier(task)
                    layer_results.append(run_one_layer_task(task))
        except APIClientError as exc:
            retry_events = list(getattr(exc, "retry_events", []) or [])
            if retry_events:
                client.retry_events.extend(retry_events)
            write_api_retry_log(output_dir, client)
            write_working_state_snapshot(
                output_dir,
                f"api_failed_{failed_task_id}",
                scene_state,
                manager_plan,
                task_results,
                None,
                api_call_count,
            )
            write_api_failure_report(
                output_dir=output_dir,
                failed_stage=f"DynamicTaskAgent:{failed_task_id}",
                exc=exc,
                api_call_count_so_far=api_call_count,
                local_schema_fallback_attempted=False,
                next_suggested_command="Rerun through the SceneMosaic pipeline after checking network/API gateway stability; each attempt uses its own output directory.",
            )
            runtime_tracker.end(layer_stage, status="failed", failed_task_id=failed_task_id)
            runtime_tracker.end("dynamic_tasks_total", status="failed", failed_task_id=failed_task_id)
            runtime_tracker.write(output_dir)
            raise

        layer_results.sort(key=lambda item: task_order.get(item.get("task_id"), 10_000))
        for item in layer_results:
            task = item["task"]
            response = item["response"]
            if isinstance(response, dict):
                response.setdefault("task_id", item["task_id"])
                response.setdefault("task_type", item["task_type"])
                response.setdefault("canonical_task_kind", item["canonical_task_kind"])
                response.setdefault("provided_capabilities", task_capabilities(task, "provides_capabilities"))
                annotate_semantic_visibility_geometry_conflict_resolutions(task, response)
            if item.get("retry_events"):
                client.retry_events.extend(item["retry_events"])
            if args.dry_run:
                runtime_tracker.record_stage_result(
                    item["execution_stage"],
                    status="success",
                    task_id=item["task_id"],
                    task_type=item["task_type"],
                )
            else:
                runtime_tracker.record_api_call(
                    item["execution_stage"],
                    success=True,
                    role="DynamicTaskAgent",
                    task_id=item["task_id"],
                    task_type=item["task_type"],
                    model=item["model"],
                    model_route=item["model_route"],
                    canonical_task_kind=item["canonical_task_kind"],
                    use_vision_model=item["use_vision_model"],
                    image_count=item["image_count"],
                    image_routing_source=item.get("image_routing_source"),
                    selected_image_artifacts=item.get("selected_image_artifacts", []),
                    keyword_image_fallback_used=item.get("keyword_image_fallback_used", False),
                    image_detail=item.get("image_detail"),
                    response_model=item.get("response_model"),
                    finish_reason=item.get("finish_reason"),
                )
            write_api_retry_log(output_dir, client)
            api_call_count += 0 if args.dry_run else 1
            write_task_output(output_dir, task, response)
            task_results_by_id[item["task_id"]] = response
            task_results.append(response)
        orientation_provider_capability = "wall_alignment_objects_ready"
        if (
            not wall_orientation_selection_done
            and any(
                task_provides_capability(item.get("task", {}), orientation_provider_capability)
                for item in layer_results
            )
        ):
            runtime_tracker.start("wall_orientation_selection_apply")
            semantic_provider_results = [
                task_results_by_id[task_identifier(task)]
                for task in manager_plan.get("tasks", [])
                if isinstance(task, dict)
                and task_provides_capability(task, orientation_provider_capability)
                and task_identifier(task) in task_results_by_id
            ]
            selection_audit = apply_wall_orientation_selection(
                scene_state,
                semantic_provider_results,
                output_dir,
            )
            if selection_audit.get("status") == "applied":
                wall_evidence = build_and_write_wall_evidence(scene_state, output_dir)
                wall_evidence["formal_obb_risk_audit"] = obb_wall_risk_audit
                wall_evidence["wall_orientation_selection_audit"] = selection_audit
                wall_reference_overlay = render_wall_attach_prefilter_overlay(
                    wall_visual_evidence.get("canonical_topdown_path"),
                    wall_visual_evidence.get("render_grid_affine_path"),
                    output_dir / "debug" / "walls" / "formal_wall_reference_overlay.png",
                    wall_evidence,
                    {
                        "blocked_wall_ids": {},
                        "wall_semantics": {},
                        "candidate_pairs": [],
                        "candidate_pair_count": 0,
                    },
                )
                if wall_reference_overlay.get("status") == "available":
                    wall_visual_evidence["wall_attach_prefilter_overlay_path"] = wall_reference_overlay.get("output_path")
                current_wall_references = formal_wall_references(wall_evidence)
                for planned_task in manager_plan.get("tasks", []):
                    if not isinstance(planned_task, dict):
                        continue
                    if any(
                        capability.startswith("wall_")
                        for capability in task_capabilities(planned_task, "provides_capabilities")
                    ):
                        planned_task["formal_wall_references"] = current_wall_references
                scene_state.setdefault("evidence", {})["wall_orientation_selection"] = selection_audit
                write_scene_state(scene_state, output_dir)
                write_manager_plan(output_dir, manager_plan)
            wall_orientation_selection_done = True
            selected_orientation_candidate_id = str(selection_audit.get("selected_candidate_id") or "") or None
            runtime_tracker.end(
                "wall_orientation_selection_apply",
                status=selection_audit.get("status"),
                selected_candidate_id=selection_audit.get("selected_candidate_id"),
                selection_source=selection_audit.get("selection_source"),
            )
        if any(
            item.get("task_type") in {"wall_object_hypothesis", "wall_alignment_object_review"}
            or task_provides_capability(item.get("task", {}), "wall_alignment_objects_ready")
            for item in layer_results
        ):
            build_and_write_wall_object_hypotheses(output_dir, task_results)

        # This local binding is deliberately after orientation selection and
        # before the next recall layer. It never changes coordinates or ids:
        # it binds only the selected candidate panel's Agent semantics to
        # matching immutable final edge geometry keys.
        if wall_orientation_selection_done:
            bound_result, binding_audit = materialize_selected_provisional_wall_visibility(
                wall_evidence,
                selected_orientation_candidate_id,
                task_results,
            )
            provisional_wall_visibility_binding_audit = binding_audit
            if bound_result is not None:
                binding_task_id = str(bound_result["task_id"])
                task_results = [
                    result for result in task_results
                    if not (isinstance(result, dict) and result.get("task_id") == binding_task_id)
                ]
                task_results.append(bound_result)
                task_results_by_id[binding_task_id] = bound_result
                provisional_wall_visibility_binding_result = bound_result
            else:
                task_results_by_id.pop("local_provisional_wall_visibility_binding", None)
                provisional_wall_visibility_binding_result = None
            write_audit_pair(
                output_dir,
                "walls",
                "provisional_wall_visibility_binding",
                binding_audit,
                "Provisional Wall Visibility Binding",
            )
            object_wall_binding_result, object_wall_binding_audit = materialize_selected_candidate_object_wall_bindings(
                wall_evidence,
                selected_orientation_candidate_id,
                task_results,
            )
            object_binding_task_id = "local_selected_candidate_object_wall_binding"
            task_results = [
                result for result in task_results
                if not (isinstance(result, dict) and result.get("task_id") == object_binding_task_id)
            ]
            task_results_by_id.pop(object_binding_task_id, None)
            if object_wall_binding_result is not None:
                task_results.append(object_wall_binding_result)
                task_results_by_id[object_binding_task_id] = object_wall_binding_result
            write_audit_pair(
                output_dir,
                "walls",
                "selected_candidate_object_wall_binding",
                object_wall_binding_audit,
                "Selected Candidate Object Wall Binding",
            )
            build_and_write_wall_object_hypotheses(output_dir, task_results)
        runtime_tracker.end(
            layer_stage,
            status="success",
            task_count=len(layer_results),
            task_ids=[item.get("task_id") for item in layer_results],
        )
    if not any(isinstance(result, dict) and (result.get("task_type") == "wall_object_hypothesis" or "wall_object_hypothesis" in str(result.get("task_id", ""))) for result in task_results):
        write_wall_object_hypotheses(output_dir, extract_wall_object_hypotheses(task_results))
    runtime_tracker.end("dynamic_tasks_total", status="success", layer_count=len(dependency_layers), max_parallel_tasks=args.max_parallel_tasks)

    if args.dry_run:
        with runtime_tracker.stage("execution_supervisor"):
            supervisor_result = supervise_execution(
                client,
                load_prompt("execution_supervisor_system.txt"),
                scene_state,
                task_results,
            )
    else:
        with runtime_tracker.api_call("execution_supervisor", role="ExecutionSupervisor", task_id="execution_supervisor", model=args.model):
            supervisor_result = supervise_execution(
                client,
                load_prompt("execution_supervisor_system.txt"),
                scene_state,
                task_results,
            )
    if not args.dry_run and runtime_tracker.api_calls:
        runtime_tracker.api_calls[-1].update(
            client.usage_metadata(),        )
    write_api_retry_log(output_dir, client)
    api_call_count += 0 if args.dry_run else 1
    update_agent_audits(output_dir, manager_plan, task_results, supervisor_result)

    if args.dry_run:
        write_dry_run_final_marker(output_dir)
        runtime_tracker.mark_skipped("schema_guard", "dry_run")
        runtime_tracker.mark_skipped("final_assembly", "dry_run")
        runtime_tracker.mark_skipped("schema_repair_loop", "dry_run")
        with runtime_tracker.stage("schema_validation"):
            validation = validate_files(
                output_dir / "final_outputs" / "objects.json",
                output_dir / "final_outputs" / "relations.json",
                output_dir,
                ceiling_confirmed=False,
                dry_run=True,
                scene_state=scene_state,
                scene_state_path=output_dir / "scene_states" / "scene_state.json",
            )
    else:
        write_working_state_snapshot(
            output_dir,
            "before_delivery_schema_guard",
            scene_state,
            manager_plan,
            task_results,
            supervisor_result,
            api_call_count,
        )
        guard_result: dict[str, Any]
        guard_api_failed = False
        api_failure_report: dict[str, Any] | None = None
        contain_conflict_report = collect_contain_conflicts(scene_state, task_results)
        contain_conflict_visual = build_contain_conflict_contact_sheet(
            scene_state,
            contain_conflict_report,
            output_dir,
        )
        guard_image_paths = compact_image_paths(
            [contain_conflict_visual.get("contact_sheet_path")],
            1,
        )
        guard_payload = compact_schema_guard_context(
            scene_state,
            task_results,
            supervisor_result,
            wall_evidence,
            contain_conflict_report=contain_conflict_report,
        )
        guard_payload["contain_conflict_review_image_attached"] = bool(guard_image_paths)
        try:
            with runtime_tracker.api_call(
                "schema_guard",
                role="DeliverySchemaGuard",
                task_id="schema_guard",
                model=args.model,
                use_vision_model=bool(guard_image_paths),
                image_count=len(guard_image_paths),
                image_routing_source="contain_conflict_only" if guard_image_paths else "none",
            ):
                guard_result = client.call_json(
                    "DeliverySchemaGuard",
                    load_prompt("delivery_schema_guard_system.txt"),
                    guard_payload,
                    use_vision_model=bool(guard_image_paths),
                    image_paths=guard_image_paths,
                )
            if runtime_tracker.api_calls:
                runtime_tracker.api_calls[-1].update(
                    client.usage_metadata(),                    contain_conflict_count=len(contain_conflict_report.get("conflicts", [])),
                    contain_conflict_contact_sheet_path=contain_conflict_visual.get("contact_sheet_path"),
                )
            write_api_retry_log(output_dir, client)
            api_call_count += 1
        except APIClientError as exc:
            guard_api_failed = True
            write_api_retry_log(output_dir, client)
            guard_result = {
                "status": "api_failed_local_schema_fallback",
                "ceiling_confirmed": False,
                "api_error": str(exc),
                "failed_stage": "DeliverySchemaGuard",
                "local_schema_fallback": True,
            }
            api_failure_report = write_api_failure_report(
                output_dir=output_dir,
                failed_stage="DeliverySchemaGuard",
                exc=exc,
                api_call_count_so_far=api_call_count,
                local_schema_fallback_attempted=True,
                next_suggested_command=(
                    "python -m vlm_agent.reconstruction.relations.agents.run_scene_delivery_multi_agent "
                    f"--scene_name {args.scene_name} --run_root {run_root} --model {args.model} --vision_model {args.vision_model} "
                    f"--api_key_env {args.api_key_env} --api_base <OPEN_BASE_URL> --render_script {render_script} --output_dir {output_dir} "
                    f"--relation_layout_dir {getattr(args, 'relation_layout_dir', None) or relation_layout_expected_manifest_path(run_root).parent}"
                ),
            )
        guard_output_dir = output_dir / "agent_outputs" / "delivery_schema_guard"
        write_json(guard_output_dir / "response.json", guard_result)
        write_md(
            guard_output_dir / "response.md",
            "Delivery Schema Guard Response",
            ["```json", json.dumps(guard_result, ensure_ascii=False, indent=2), "```"],
        )
        write_working_state_snapshot(
            output_dir,
            "after_delivery_schema_guard_or_local_fallback",
            scene_state,
            manager_plan,
            task_results,
            supervisor_result,
            api_call_count,
        )
        with runtime_tracker.stage("final_assembly"):
            assembly = assemble_final_payloads(
                scene_state=scene_state,
                task_results=task_results,
                guard_result=guard_result,
                output_dir=output_dir,
                dry_run=False,
                wall_evidence=wall_evidence,
            )
        if assembly.get("repair_tasks") and not assembly.get("objects_payload"):
            assembly, repair_api_calls = execute_schema_repair_loop(
                client=client,
                output_dir=output_dir,
                scene_state=scene_state,
                task_results=task_results,
                guard_result=guard_result,
                wall_evidence=wall_evidence,
                assembly=assembly,
                wall_visual_evidence=wall_visual_evidence,
                runtime_tracker=runtime_tracker,
            )
            api_call_count += repair_api_calls
        else:
            runtime_tracker.mark_skipped("schema_repair_loop", "no repair tasks were executed")
        if assembly["objects_payload"] is not None and assembly["relations_payload"] is not None:
            write_json(output_dir / "final_outputs" / "objects.json", assembly["objects_payload"])
            assembly["final_objects_written"] = True
            write_json(output_dir / "final_outputs" / "relations.json", assembly["relations_payload"])
            assembly["final_relations_written"] = True
            with runtime_tracker.stage("schema_validation"):
                validation = validate_files(
                    output_dir / "final_outputs" / "objects.json",
                    output_dir / "final_outputs" / "relations.json",
                    output_dir,
                    ceiling_confirmed=bool(guard_result.get("ceiling_confirmed") or assembly.get("ceiling_confirmed")),
                    dry_run=False,
                    scene_state=scene_state,
                    scene_state_path=output_dir / "scene_states" / "scene_state.json",
                )
            if guard_api_failed:
                validation["local_schema_fallback"] = True
                validation["api_failure_report_path"] = str(output_dir / "debug" / "agents" / "api_failure_report.json")
                write_audit_pair(output_dir, "final_validation", "schema_validation", validation, "Schema Validation")
        else:
            with runtime_tracker.stage("schema_validation"):
                validation = {
                    "dry_run": False,
                    "valid": False,
                    "finalization_status": "finalization_failed_with_partial_outputs",
                    "errors": assembly.get("errors", []),
                    "wall_errors": assembly.get("wall_errors", []),
                    "wall_final_adjudication": assembly.get("wall_final_adjudication"),
                    "partial_objects_path": assembly.get("partial_objects_path"),
                    "repair_tasks": assembly.get("repair_tasks", []),
                    "objects": None,
                    "relations": None,
                    "local_schema_fallback": bool(guard_api_failed),
                    "api_failure_report_path": str(output_dir / "debug" / "agents" / "api_failure_report.json") if guard_api_failed else None,
                }
                write_audit_pair(output_dir, "final_validation", "schema_validation", validation, "Schema Validation")

    current_final_outputs_generated = bool(
        (not args.dry_run)
        and validation.get("valid")
        and (output_dir / "final_outputs" / "objects.json").exists()
        and (output_dir / "final_outputs" / "relations.json").exists()
    )
    stale_final_outputs_detected = bool(
        not current_final_outputs_generated
        and (
            (output_dir / "final_outputs" / "objects.json").exists()
            or (output_dir / "final_outputs" / "relations.json").exists()
            or (output_dir / "debug" / "walls" / "final_walls_overlay_on_canonical_topdown.png").exists()
        )
    )
    with runtime_tracker.stage("final_wall_overlay"):
        final_wall_overlay, final_wall_overlay_skipped_reason = write_final_wall_overlay_or_skip(
            output_dir,
            scene_state,
            final_outputs_generated=current_final_outputs_generated,
        )
    runtime_summary_for_report = runtime_tracker.to_json()
    with runtime_tracker.stage("final_process_report"):
        final_process_report = write_final_process_report(
            output_dir=output_dir,
            scene_state=scene_state,
            manager_plan=manager_plan,
            task_results=task_results,
            validation=validation,
            wall_evidence=wall_evidence,
            wall_visual_evidence=wall_visual_evidence,
            final_wall_overlay=final_wall_overlay,
            final_wall_overlay_skipped_reason=final_wall_overlay_skipped_reason,
            api_called=not args.dry_run,
            api_call_count=api_call_count,
            finalization_status=validation.get("finalization_status") or ("completed" if validation.get("valid") else "failed"),
            runtime_summary=runtime_summary_for_report,
        )
    runtime_summary = runtime_tracker.write(output_dir)
    final_process_report = write_final_process_report(
        output_dir=output_dir,
        scene_state=scene_state,
        manager_plan=manager_plan,
        task_results=task_results,
        validation=validation,
        wall_evidence=wall_evidence,
        wall_visual_evidence=wall_visual_evidence,
        final_wall_overlay=final_wall_overlay,
        final_wall_overlay_skipped_reason=final_wall_overlay_skipped_reason,
        api_called=not args.dry_run,
        api_call_count=api_call_count,
        finalization_status=validation.get("finalization_status") or ("completed" if validation.get("valid") else "failed"),
        runtime_summary=runtime_summary,
    )

    physics_sidecar = snapshot_physics_sidecar(physics_sidecar_process, physics_sidecar)

    return {
        "scene_name": args.scene_name,
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "policy": policy_result,
        "scene_state_path": str(output_dir / "scene_states" / "scene_state.json"),
        "object_candidate_count": len(scene_state["object_candidates"]["items"]),
        "render_capabilities": render_capabilities,
        "manager_plan_path": str(output_dir / "manager_plans" / "manager_plan_000.json"),
        "final_outputs_generated": current_final_outputs_generated,
        "stale_final_outputs_detected": stale_final_outputs_detected,
        "schema_validation": validation,
        "api_called": not args.dry_run,
        "api_call_count": api_call_count,
        "guard_api_failed": bool(locals().get("guard_api_failed", False)),
        "api_failure_report": locals().get("api_failure_report"),
        "wall_evidence_path": str(output_dir / "debug" / "walls" / "wall_evidence.json"),
        "wall_visual_evidence_manifest_path": str(output_dir / "debug" / "walls" / "visual_review" / f"{args.scene_name}_wall_visual_review_manifest.json"),
        "wall_candidate_config": wall_candidate_config,
        "wall_evidence_candidate_walls": len(wall_evidence.get("candidate_walls", [])),
        "final_wall_overlay": final_wall_overlay,
        "final_wall_overlay_skipped_reason": final_wall_overlay_skipped_reason,
        "final_process_report_path": final_process_report.get("md_path"),
        "runtime_summary_path": runtime_summary.get("json_path"),
        "canonical_topdown_regeneration": canonical_result,
        "render_grid_wall_precompute": render_grid_precompute_result,
        "fixed_layout_views": fixed_views_result,
        "physics_sidecar": physics_sidecar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent scene delivery for final objects/relations outputs.")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help=f"Base text model for ordinary tasks; default: {DEFAULT_BASE_MODEL}.")
    parser.add_argument("--vision_model", default=DEFAULT_BASE_MODEL, help=f"Base vision model for ordinary tasks; default: {DEFAULT_BASE_MODEL}.")
    parser.add_argument(
        "--wall_specialist_model",
        default=DEFAULT_WALL_SPECIALIST_MODEL,
        help=f"Model for configured wall semantic/attach task kinds; default: {DEFAULT_WALL_SPECIALIST_MODEL}.",
    )
    parser.add_argument(
        "--wall_specialist_vision_model",
        default=None,
        help="Vision model for wall specialist tasks; defaults to --wall_specialist_model.",
    )
    parser.add_argument(
        "--wall_specialist_task_kinds",
        nargs="+",
        default=sorted(WALL_SPECIALIST_TASK_KINDS),
        help=(
            "Canonical task kinds routed to the wall specialist when it is enabled. "
            "Accepts space- or comma-separated values. The default covers wall visibility, "
            "final wall adjudication, and wall attach recall."
        ),
    )
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--render_script", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--disable_evidence_cache", action="store_true", help="Disable reuse of evidence caches. Formal render_grid_only runs should normally pass explicit precompute dirs instead.")
    parser.add_argument("--fixed_views_precompute_dir", type=Path, default=None, help="Directory containing a compatible precomputed fixed-view cache. Formal render_grid_only runs consume this evidence instead of rendering fixed views inside the multi-agent run.")
    parser.add_argument("--relation_precompute_wait_seconds", type=float, default=0.0, help="When relation precompute is running concurrently, poll its manifest for up to this many seconds before failing.")
    parser.add_argument("--fixed_views_precompute_wait_seconds", type=float, default=None, help="Poll for precomputed fixed views for this many seconds. Defaults to --relation_precompute_wait_seconds. Use 0 for dry-run smoke tests that intentionally skip fixed views.")
    parser.add_argument("--relation_layout_dir", type=Path, default=None, help="Explicit candidate-local relation layout directory to consume.")
    parser.add_argument("--manager_model", default=None, help="Optional model for Manager planning; defaults to --model.")
    parser.add_argument("--manager_vision_model", default=None, help="Optional vision model for Manager planning; defaults to --vision_model.")
    parser.add_argument("--manager_vision_image_detail", choices=["low", "high", "auto"], default="low", help="Image detail used by compact Manager planning.")
    parser.add_argument("--vision_image_detail", choices=["low", "high", "auto"], default=None, help="Optional image_url.detail for vision inputs; low uses a smaller image representation when fine visual detail is not needed.")
    parser.add_argument("--task_vision_detail_overrides", default="", help="Optional comma-separated task-kind overrides, e.g. support_contact_extraction=high,contain_relation_extraction=low. An explicit --vision_image_detail still overrides every task.")
    parser.add_argument("--max_parallel_tasks", type=int, default=4, help="Maximum DynamicTaskAgent calls to run concurrently within the same dependency layer.")
    parser.add_argument("--physics_model", default=None, help="Optional text model for the independent Physics Sidecar; defaults to --model.")
    parser.add_argument("--physics_vision_model", default=None, help="Optional vision model for the independent Physics Sidecar; defaults to --vision_model.")
    parser.add_argument("--physics_include_original", action="store_true", help="Also provide the original image to Physics Sidecar. Object-list-only mode is the faster default.")
    parser.add_argument("--physics_vision_image_detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument(
        "--wall_candidate_mode",
        choices=["render_grid_only"],
        default="render_grid_only",
        help="Formal path: consume the precomputed OBB render-grid package as the only wall coordinate source.",
    )
    parser.add_argument(
        "--wall_review_mode",
        choices=["auto", "light", "heavy"],
        default="auto",
        help="auto routes formal OBB wall review through local risk audit: low risk uses light visibility/weak-wall review, medium/high risk keeps heavy final wall adjudication. Use heavy to always run the comprehensive multi-image adjudication.",
    )
    parser.add_argument("--render_grid_wall_precompute_dir", type=Path, default=None, help="Directory produced by the SceneMosaic relation render-grid precompute. Formal render_grid_only runs consume this precomputed evidence instead of rendering inside the multi-agent run.")
    parser.add_argument("--render_grid_wall_precompute_path", type=Path, default=None, help="Path to a precomputed method_candidates.json produced by the SceneMosaic relation render-grid precompute.")
    args = parser.parse_args()
    if args.max_parallel_tasks < 1:
        parser.error("--max_parallel_tasks must be >= 1")

    try:
        result = run_delivery(args)
    except APIClientError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if (
        not args.dry_run
        and not result.get("final_outputs_generated")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
