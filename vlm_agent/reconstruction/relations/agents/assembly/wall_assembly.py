from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from relations.agents.assembly.object_properties import _walk
from relations.agents.schema_guard import WALL_IDS, validate_wall_geometry
from relations.agents.state import write_audit_pair
from relations.agents.wall_attach_prefilter import wall_geometry_key
from relations.tools.wall_canonical_naming import canonical_wall_naming_policy, final_wall_naming_audit

FORMAL_OBB_WALL_METHOD = "obb_from_mesh_footprint_min_rect"
FORMAL_OBB_WALL_SOURCE = "obb_from_mesh_footprint_min_rect"

ALLOWED_FINAL_WALL_SOURCES = {
    FORMAL_OBB_WALL_SOURCE,
}
INTERMEDIATE_WALL_SOURCES = {
    "layout_xz_boundary_points",
}
HINT_ONLY_WALL_SOURCES = {
    "layout_bounds_hint",
    "object_footprint_bounds_hint",
    "aggregate_bbox_hint",
    "grid_extent_hint",
    "candidate_room_polygon_hint",
    "canonical_grid_mapping.grid_points_extent_hint",
    "canonical_grid_mapping.x_range_z_range_hint",
    "wall_attached_object_distribution",
    "topdown_camera_config.scene_bbox_global_xz_hint",
    "topdown_camera_config.x_range_z_range_hint",
    "scene_state.object_candidates.bbox_3d_aggregate_hint",
}
DEFAULT_WALL_PROMOTION_EVIDENCE = [
    "contact_sheet",
    "candidate_walls_overlay",
    "canonical_topdown",
    "fixed_layout_views",
    "original_image",
    "mask_overlay",
    "object_distribution",
]

def _wall_source_from_context(wall: dict[str, Any], node: dict[str, Any], root: dict[str, Any]) -> str | None:
    return (
        wall.get("source")
        or wall.get("wall_source")
        or node.get("source")
        or node.get("wall_source")
        or root.get("source")
        or root.get("wall_source")
    )


def _wall_verification_status(wall: dict[str, Any], node: dict[str, Any], root: dict[str, Any]) -> str | None:
    raw = wall.get("topdown_verification") or wall.get("topdown_verification_status") or node.get("topdown_verification") or root.get("topdown_verification")
    if isinstance(raw, dict):
        raw = raw.get("status") or raw.get("result")
    return str(raw).lower() if raw is not None else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "promoted", "candidate_promotion"}
    return bool(value)


def _listify_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _candidate_promotion_audit(
    wall: dict[str, Any],
    node: dict[str, Any],
    root: dict[str, Any],
    original_source: str | None,
    normalized_source: str | None,
    topdown_verification: str | None,
    evidence: Any,
    source_normalized: bool,
) -> dict[str, Any]:
    evidence_text = str(evidence or "")
    evidence_l = evidence_text.lower()
    candidate_original_source = (
        wall.get("candidate_original_source")
        or wall.get("promoted_from_source")
        or wall.get("promotion_original_source")
        or node.get("candidate_original_source")
        or node.get("promoted_from_source")
        or root.get("candidate_original_source")
        or root.get("promoted_from_source")
    )
    if not candidate_original_source:
        if original_source == "layout_xz_boundary_points":
            candidate_original_source = "layout_xz_boundary_points"
        elif "layout_xz_boundary_points" in evidence_l:
            candidate_original_source = "layout_xz_boundary_points"
    explicit_promotion = any(
        _truthy(value)
        for value in [
            wall.get("promotion_from_candidate"),
            wall.get("promoted_from_candidate"),
            node.get("promotion_from_candidate"),
            node.get("promoted_from_candidate"),
            root.get("promotion_from_candidate"),
            root.get("promoted_from_candidate"),
        ]
    )
    text_promotion = "promot" in evidence_l and ("candidate" in evidence_l or "layout_xz_boundary_points" in evidence_l)
    source_promotion = source_normalized and original_source == "layout_xz_boundary_points"
    promotion = bool(
        normalized_source == "obb_from_mesh_footprint_min_rect"
        and topdown_verification == "passed"
        and (explicit_promotion or text_promotion or source_promotion or candidate_original_source)
    )
    verification_evidence_used = (
        _listify_strings(wall.get("verification_evidence_used"))
        or _listify_strings(wall.get("visual_evidence_used"))
        or _listify_strings(node.get("verification_evidence_used"))
        or _listify_strings(root.get("verification_evidence_used"))
        or (DEFAULT_WALL_PROMOTION_EVIDENCE if promotion else [])
    )
    promotion_reason = (
        wall.get("promotion_reason")
        or node.get("promotion_reason")
        or root.get("promotion_reason")
        or evidence_text
        or "layout-XZ candidate wall coordinates were visually verified before promotion"
    )
    candidate_modified = any(
        _truthy(value)
        for value in [
            wall.get("candidate_modified"),
            node.get("candidate_modified"),
            root.get("candidate_modified"),
        ]
    )
    revision_reason = wall.get("revision_reason") or node.get("revision_reason") or root.get("revision_reason")
    if candidate_modified:
        promotion = bool(normalized_source == "obb_from_mesh_footprint_min_rect" and topdown_verification == "passed")
        if not candidate_original_source:
            candidate_original_source = "layout_xz_boundary_points"
        if not revision_reason:
            revision_reason = evidence_text or "candidate wall coordinates were revised using layout/visual evidence"
    return {
        "promotion_from_candidate": promotion,
        "candidate_modified": candidate_modified,
        "revision_reason": revision_reason,
        "candidate_original_source": str(candidate_original_source) if candidate_original_source else None,
        "normalized_source": normalized_source,
        "verification_evidence_used": verification_evidence_used,
        "promotion_reason": promotion_reason if promotion else None,
    }


def _normalize_wall_source(source: str | None, topdown_verification: str | None) -> tuple[str | None, bool, str | None]:
    return source, False, None


def _wall_source_allowed(source: str | None, emergency_fallback: bool = False) -> bool:
    if source in ALLOWED_FINAL_WALL_SOURCES:
        return True
    if emergency_fallback:
        return True
    return False


def _normalize_wall_set(walls: list[Any], node: dict[str, Any], root: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        original_source = _wall_source_from_context(wall, node, root)
        topdown_verification = _wall_verification_status(wall, node, root)
        source, source_normalized, normalization_reason = _normalize_wall_source(original_source, topdown_verification)
        emergency = bool(wall.get("emergency_fallback") or node.get("emergency_fallback") or root.get("emergency_fallback"))
        evidence = wall.get("evidence") or wall.get("audit_evidence") or node.get("evidence") or node.get("evidence_used")
        promotion_audit = _candidate_promotion_audit(wall, node, root, original_source, source, topdown_verification, evidence, source_normalized)
        evidence_strength = wall.get("evidence_strength") or node.get("evidence_strength") or root.get("evidence_strength")
        completion_method = wall.get("completion_method") or node.get("completion_method") or root.get("completion_method")
        completion_reason = wall.get("completion_reason") or node.get("completion_reason") or root.get("completion_reason")
        canonical_naming = wall.get("canonical_naming") if isinstance(wall.get("canonical_naming"), dict) else {}
        candidate_or_geometry_wall_id = (
            wall.get("candidate_or_geometry_wall_id")
            or wall.get("candidate_wall_id")
            or wall.get("geometry_wall_id")
            or wall.get("provisional_wall_id")
            or canonical_naming.get("candidate_or_geometry_wall_id")
            or canonical_naming.get("geometry_wall_id")
        )
        original_image_visibility = (
            wall.get("original_image_visibility")
            or wall.get("visibility")
            or canonical_naming.get("original_image_visibility")
            or node.get("original_image_visibility")
            or root.get("original_image_visibility")
        )
        visible_role = wall.get("visible_role") or canonical_naming.get("visible_role") or wall.get("wall_visible_role")
        original_visible_wall_count = (
            wall.get("original_visible_wall_count")
            or canonical_naming.get("original_visible_wall_count")
            or node.get("original_visible_wall_count")
            or root.get("original_visible_wall_count")
        )
        visibility_text = str(original_image_visibility or "").lower()
        closure_only = _truthy(wall.get("closure_only") or node.get("closure_only") or root.get("closure_only"))
        weak_wall = bool(
            _truthy(wall.get("weak_wall") or node.get("weak_wall") or root.get("weak_wall"))
            or closure_only
            or str(evidence_strength or "").lower() == "weak"
            or str(completion_method or "").lower() == "inferred_by_room_closure"
            or visibility_text in {"hidden", "not_visible", "not directly visible", "inferred", "occluded"}
        )
        record = {
            "wall_id": wall.get("id"),
            "id": wall.get("id"),
            "candidate_or_geometry_wall_id": candidate_or_geometry_wall_id,
            "candidate_wall_id": wall.get("candidate_wall_id"),
            "geometry_wall_id": wall.get("geometry_wall_id"),
            "provisional_wall_id": wall.get("provisional_wall_id"),
            "first_point": wall.get("first_point"),
            "second_point": wall.get("second_point"),
            "original_source": original_source,
            "normalized_source": source,
            "source": source,
            "source_normalized": source_normalized,
            "evidence": evidence,
            "topdown_verification": topdown_verification,
            "evidence_strength": evidence_strength,
            "completion_method": completion_method,
            "completion_reason": completion_reason,
            "weak_wall": weak_wall,
            "closure_only": closure_only,
            "original_image_visibility": original_image_visibility,
            "visible_role": visible_role,
            "original_visible_wall_count": original_visible_wall_count,
            "canonical_naming": canonical_naming,
            "fixed_views_used": wall.get("fixed_views_used") or node.get("fixed_views_used") or [],
            "promotion_from_candidate": promotion_audit["promotion_from_candidate"],
            "candidate_modified": promotion_audit["candidate_modified"],
            "revision_reason": promotion_audit["revision_reason"],
            "candidate_original_source": promotion_audit["candidate_original_source"],
            "verification_evidence_used": promotion_audit["verification_evidence_used"],
            "promotion_reason": promotion_audit["promotion_reason"],
            "emergency_fallback": emergency,
            "accepted": False,
            "accepted_as_final_wall": False,
            "reason": normalization_reason,
        }
        if normalization_reason:
            warnings.append(f"{normalization_reason}: `{wall.get('id')}`.")
        if not _wall_source_allowed(source, emergency_fallback=emergency):
            reason = "missing_source" if not source else "hint_only_or_untrusted_source"
            if original_source in HINT_ONLY_WALL_SOURCES or source in HINT_ONLY_WALL_SOURCES:
                reason = "hint_only_source_rejected"
            if original_source == "layout_xz_boundary_points" and topdown_verification != "passed":
                reason = "layout_xz_boundary_points_requires_topdown_verification_passed"
            record["rejection_reason"] = reason
            record["reason"] = record.get("reason") or reason
            warnings.append(f"Rejected wall `{wall.get('id')}` as final wall because source `{original_source}` normalized to `{source}` is not final-eligible.")
        else:
            record["accepted"] = True
            record["accepted_as_final_wall"] = True
            record["reason"] = record.get("reason") or "final wall source is layout XZ coordinates with topdown verification or manual layout XZ walls"
            normalized.append({"id": wall.get("id"), "first_point": wall.get("first_point"), "second_point": wall.get("second_point")})
            if emergency:
                warnings.append(f"Accepted wall `{wall.get('id')}` only as emergency fallback; final validation should treat this as warning.")
        records.append(record)
    return normalized, records, warnings


def _is_wall_final_adjudication_node(node: dict[str, Any], root: dict[str, Any]) -> bool:
    task_type = str(node.get("task_type") or "").lower()
    task_id = str(node.get("task_id") or "").lower()
    decision = str(node.get("decision") or "").lower()
    if node is root:
        task_type = task_type or str(root.get("task_type") or "").lower()
        task_id = task_id or str(root.get("task_id") or "").lower()
        decision = decision or str(root.get("decision") or "").lower()
    return (
        task_type == "wall_final_adjudication"
        or "wall_final_adjudication" in task_id
        or decision in {"verified_final_walls", "insufficient_layout_xz_wall_evidence"}
    )


def _wall_final_adjudication_decision(node: dict[str, Any], root: dict[str, Any]) -> str | None:
    decision = node.get("decision") or root.get("decision")
    if decision is None:
        return None
    decision_s = str(decision).lower()
    if decision_s in {"accepted_candidate", "accept_candidate", "accepted_relation"}:
        return "accept"
    return decision_s


def summarize_wall_final_adjudication(task_results: list[dict[str, Any]], guard_result: dict[str, Any] | None = None, accepted_walls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)
    summary = {
        "executed": False,
        "decision": None,
        "accepted_final_walls": bool(accepted_walls and len(accepted_walls) == 4),
        "invalid_intermediate_output": False,
        "items": [],
    }
    for root in roots:
        if not isinstance(root, dict):
            continue
        for node in _walk(root):
            if not isinstance(node, dict) or not _is_wall_final_adjudication_node(node, root):
                continue
            decision = _wall_final_adjudication_decision(node, root)
            walls = node.get("walls") or node.get("accepted_walls") or node.get("wall_candidates") or []
            wall_count = len(walls) if isinstance(walls, list) else 0
            accepted_count = 0
            if isinstance(walls, list) and wall_count:
                normalized, _, _ = _normalize_wall_set(walls, node, root)
                accepted_ids = {wall.get("id") for wall in normalized if isinstance(wall, dict)}
                if set(WALL_IDS).issubset(accepted_ids):
                    accepted_count = 4
                else:
                    accepted_count = len(accepted_ids.intersection(WALL_IDS))
            candidate_modified = any(isinstance(wall, dict) and _truthy(wall.get("candidate_modified")) for wall in walls) if isinstance(walls, list) else False
            revision_reasons = [str(wall.get("revision_reason")) for wall in walls if isinstance(wall, dict) and wall.get("revision_reason")] if isinstance(walls, list) else []
            item = {
                "task_id": node.get("task_id") or root.get("task_id"),
                "task_type": node.get("task_type") or root.get("task_type"),
                "decision": decision,
                "wall_count": wall_count,
                "accepted_final_wall_count": accepted_count,
                "candidate_modified": candidate_modified,
                "revision_used": candidate_modified,
                "revision_reasons": revision_reasons,
                "reason": node.get("reason") or root.get("reason"),
            }
            summary["executed"] = True
            summary["items"].append(item)
            if decision == "verified_final_walls":
                if accepted_count == 4:
                    summary["decision"] = "verified_final_walls"
                else:
                    summary["decision"] = "invalid_intermediate_output"
                    summary["invalid_intermediate_output"] = True
            elif decision == "insufficient_layout_xz_wall_evidence":
                summary["decision"] = "insufficient_layout_xz_wall_evidence"
            elif wall_count:
                summary["decision"] = summary.get("decision") or "invalid_intermediate_output"
                summary["invalid_intermediate_output"] = True
    summary["candidate_modified"] = any(item.get("candidate_modified") for item in summary.get("items", []))
    summary["revision_used"] = bool(summary["candidate_modified"])
    summary["revision_reasons"] = [reason for item in summary.get("items", []) for reason in item.get("revision_reasons", [])]
    if summary["accepted_final_walls"] and summary.get("decision") in {None, "verified_final_walls"}:
        summary["decision"] = "verified_final_walls"
    return summary

def collect_walls(
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None = None,
    wall_evidence: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)
    warnings: list[str] = []
    all_records: list[dict[str, Any]] = []
    for root in roots:
        if not isinstance(root, dict):
            continue
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            walls = node.get("accepted_walls") or node.get("walls") or node.get("wall_candidates")
            visibility_review = node.get("wall_visibility_review")
            if isinstance(visibility_review, list):
                for item in visibility_review:
                    if not isinstance(item, dict):
                        continue
                    wall_id = item.get("wall_id") or item.get("id")
                    if wall_id:
                        all_records.append(
                            {
                                "id": wall_id,
                                "wall_id": wall_id,
                                "accepted_as_final_wall": False,
                                "visibility_review_only": True,
                                "original_image_visibility": item.get("original_image_visibility") or item.get("visibility"),
                                "visible_role": item.get("visible_role"),
                                "weak_wall": _truthy(item.get("weak_wall")),
                                "closure_only": _truthy(item.get("closure_only")),
                                "reason": item.get("reason"),
                                "source_task_id": root.get("task_id"),
                            }
                        )
            is_final_adjudication = _is_wall_final_adjudication_node(node, root)
            decision = _wall_final_adjudication_decision(node, root)
            if is_final_adjudication and decision == "insufficient_layout_xz_wall_evidence":
                reason = node.get("reason") or root.get("reason") or "wall final adjudication returned insufficient evidence"
                warnings.append(f"wall_final_adjudication returned insufficient_layout_xz_wall_evidence: {reason}")
                all_records.append({
                    "task_id": node.get("task_id") or root.get("task_id"),
                    "task_type": node.get("task_type") or root.get("task_type"),
                    "decision": decision,
                    "accepted_as_final_wall": False,
                    "rejection_reason": "wall_final_adjudication_insufficient",
                    "reason": reason,
                })
                continue
            if isinstance(walls, list) and (len(walls) >= 4 or is_final_adjudication):
                normalized, records, wall_warnings = _normalize_wall_set(walls, node, root)
                warnings.extend(wall_warnings)
                all_records.extend(records)
                wall_ids = {wall.get("id") for wall in normalized}
                if set(WALL_IDS).issubset(wall_ids):
                    normalized = sorted([w for w in normalized if w.get("id") in WALL_IDS], key=lambda w: WALL_IDS.index(w["id"]))
                    return normalized[:4], warnings, all_records
                if is_final_adjudication:
                    warnings.append("wall_final_adjudication_returned_intermediate_state")
                    all_records.append({
                        "task_id": node.get("task_id") or root.get("task_id"),
                        "task_type": node.get("task_type") or root.get("task_type"),
                        "decision": decision or "invalid_intermediate_output",
                        "accepted_as_final_wall": False,
                        "rejection_reason": "wall_final_adjudication_returned_intermediate_state",
                    })
    if isinstance(wall_evidence, dict) and wall_evidence.get("candidate_walls"):
        warnings.append("Structured wall_evidence.candidate_walls are hint_only and are not accepted as final walls without a final-eligible source.")
        for wall in wall_evidence.get("candidate_walls", []):
            if isinstance(wall, dict):
                all_records.append(
                    {
                        "id": wall.get("id"),
                        "first_point": wall.get("first_point"),
                        "second_point": wall.get("second_point"),
                        "source": wall.get("source"),
                        "trust_level": wall.get("trust_level", "hint_only"),
                        "may_be_used_as_final_wall": wall.get("may_be_used_as_final_wall", False),
                        "accepted_as_final_wall": False,
                        "rejection_reason": "wall_evidence_hint_only",
                    }
                )
    warnings.append("No complete wall_001..wall_004 set with final-eligible source was found.")
    return [], warnings, all_records


WALL_SEMANTIC_CONFLICT_RESOLUTION_PRIORITY = 350


def _wall_semantic_source_priority(root: dict[str, Any]) -> int:
    """Prefer candidate-panel semantics once local code binds them to final geometry."""
    text = " ".join(
        str(root.get(key) or "")
        for key in ("task_id", "task_type", "canonical_task_kind", "semantic_authority")
    ).lower()
    if "selected_candidate_provisional_visibility" in text or "local_provisional_wall_visibility_binding" in text:
        return 300
    if "wall_final_adjudication" in text:
        return 200
    provided = root.get("provided_capabilities") if isinstance(root.get("provided_capabilities"), list) else []
    if "wall_visibility_semantics_ready" in provided:
        return 100
    if "wall_attach" in text:
        return 50
    return 25


def collect_wall_final_semantics(
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read visibility metadata with explicit source priority.

    The local provisional binding normally has the highest priority because it is
    Agent semantic evidence tied to the selected finite candidate by immutable
    geometry keys. The only higher-priority exception is one wall that the later
    recall Agent both reclassifies as visible and accepts for an exact locally
    flagged semantic/geometry conflict pair.
    """
    semantics: dict[str, dict[str, Any]] = {}
    roots = list(task_results) + ([guard_result] if isinstance(guard_result, dict) else [])
    semantic_keys = [
        "original_image_visibility",
        "visible_role",
        "original_visible_wall_count",
        "weak_wall",
        "inferred",
        "closure_only",
        "can_anchor_wall_attach",
        "attach_eligible",
        "evidence_strength",
        "completion_method",
        "completion_reason",
        "reason",
        "reason_code",
        "evidence_code",
        "semantic_visibility_geometry_conflict_resolved",
        "semantic_visibility_geometry_conflict_object_ids",
    ]
    for root in roots:
        if not isinstance(root, dict):
            continue
        task_text = " ".join(str(root.get(key) or "") for key in ["task_id", "task_type", "canonical_task_kind"]).lower()
        provided_capabilities = root.get("provided_capabilities") if isinstance(root.get("provided_capabilities"), list) else []
        if "wall_final_adjudication" not in task_text and "wall_visibility_semantics_ready" not in provided_capabilities:
            continue
        source_priority = _wall_semantic_source_priority(root)
        numbered_semantic_sets = [
            node.get("wall_visibility_semantics")
            for node in _walk(root)
            if isinstance(node, dict) and isinstance(node.get("wall_visibility_semantics"), list)
        ]
        has_complete_geometry_bound_semantics = any(
            {str(item.get("wall_id") or item.get("id") or "") for item in items if isinstance(item, dict)} == set(WALL_IDS)
            and all(
                isinstance(item, dict) and bool(item.get("geometry_key"))
                for item in items
            )
            for items in numbered_semantic_sets
        )
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            for list_key in ["walls", "accepted_walls", "wall_visibility_review", "wall_visibility_semantics"]:
                items = node.get(list_key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    wall_id = item.get("wall_id") or item.get("id")
                    if wall_id not in WALL_IDS:
                        continue
                    item_priority = (
                        WALL_SEMANTIC_CONFLICT_RESOLUTION_PRIORITY
                        if (
                            item.get("semantic_visibility_geometry_conflict_resolved") is True
                            or (list_key == "wall_visibility_semantics" and has_complete_geometry_bound_semantics)
                        )
                        else source_priority
                    )
                    record = semantics.setdefault(str(wall_id), {"wall_id": str(wall_id), "source_lists": [], "_field_priorities": {}})
                    priorities = record["_field_priorities"]
                    geometry_key = item.get("geometry_key") or wall_geometry_key(item)
                    if geometry_key and item_priority >= int(priorities.get("geometry_key", -1)):
                        record["geometry_key"] = geometry_key
                        priorities["geometry_key"] = item_priority
                    if item.get("first_point") is not None and item.get("second_point") is not None and item_priority >= int(priorities.get("reported_geometry", -1)):
                        record["reported_geometry"] = {"first_point": item.get("first_point"), "second_point": item.get("second_point")}
                        priorities["reported_geometry"] = item_priority
                    if list_key not in record["source_lists"]:
                        record["source_lists"].append(list_key)
                    for key in semantic_keys:
                        if key not in item or item.get(key) is None:
                            continue
                        target_key = {
                            "reason": "semantic_reason",
                            "evidence_code": "reason_code",
                            "attach_eligible": "can_anchor_wall_attach",
                        }.get(key, key)
                        if item_priority >= int(priorities.get(target_key, -1)):
                            record[target_key] = item.get(key)
                            priorities[target_key] = item_priority
                    record["semantic_source_priority"] = max(int(record.get("semantic_source_priority", -1)), item_priority)
    for record in semantics.values():
        record.pop("_field_priorities", None)
    return semantics

def apply_wall_final_semantics(
    wall_records: list[dict[str, Any]],
    semantics: dict[str, dict[str, Any]],
    wall_id_remap: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Overlay Agent semantics only when wall id and immutable geometry agree."""
    wall_id_remap = wall_id_remap or {}
    final_by_id = {
        str(record.get("id") or record.get("wall_id")): record
        for record in wall_records
        if isinstance(record, dict) and record.get("accepted_as_final_wall")
    }
    remapped: dict[str, dict[str, Any]] = {}
    by_geometry_key: dict[str, dict[str, Any]] = {}
    rejected_records: list[dict[str, Any]] = []
    for source_wall_id, metadata in semantics.items():
        if not isinstance(metadata, dict):
            continue
        final_id = str(wall_id_remap.get(source_wall_id, source_wall_id))
        final_record = final_by_id.get(final_id)
        if final_id not in WALL_IDS or final_record is None:
            rejected_records.append({
                "reported_wall_id": source_wall_id,
                "remapped_wall_id": final_id,
                "geometry_key": metadata.get("geometry_key"),
                "reason": "semantic_wall_id_not_in_final_walls",
            })
            continue
        expected_key = wall_geometry_key(final_record)
        reported_key = metadata.get("geometry_key")
        if reported_key and expected_key and str(reported_key) != str(expected_key):
            rejected_records.append({
                "reported_wall_id": source_wall_id,
                "remapped_wall_id": final_id,
                "geometry_key": reported_key,
                "expected_geometry_key": expected_key,
                "reason": "wall_id_geometry_key_mismatch",
            })
            continue
        remapped[final_id] = metadata
        if expected_key:
            by_geometry_key[str(expected_key)] = metadata

    updated_records = []
    applied_ids = []
    for record in wall_records:
        updated = dict(record)
        wall_id = str(record.get("id") or record.get("wall_id") or "")
        geometry_key = wall_geometry_key(record)
        metadata = None
        if record.get("accepted_as_final_wall"):
            metadata = by_geometry_key.get(str(geometry_key)) if geometry_key else None
            if metadata is None:
                metadata = remapped.get(wall_id)
        if geometry_key:
            updated["geometry_key"] = geometry_key
        if metadata:
            for key in [
                "original_image_visibility", "visible_role", "original_visible_wall_count",
                "weak_wall", "closure_only", "can_anchor_wall_attach", "evidence_strength", "completion_method", "completion_reason",
            ]:
                if key in metadata:
                    updated[key] = metadata[key]
            updated["agent_wall_semantics_applied"] = True
            updated["wall_semantic_reason"] = metadata.get("semantic_reason")
            updated["wall_semantic_source_lists"] = metadata.get("source_lists", [])
            applied_ids.append(wall_id)
        updated_records.append(updated)
    return updated_records, {
        "applied": bool(applied_ids),
        "source": "prioritized Agent wall semantics with strict canonical id and immutable geometry binding",
        "coordinates_accepted_from_agent": False,
        "binding_policy": "geometry_key_and_remapped_wall_id_must_agree_when_both_are_available",
        "applied_wall_ids": sorted(set(applied_ids)),
        "missing_wall_ids": [wall_id for wall_id in WALL_IDS if wall_id not in set(applied_ids)],
        "rejected_semantic_records": rejected_records,
    }


def summarize_candidate_wall_promotion(wall_records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [record for record in wall_records if record.get("accepted_as_final_wall")]
    promoted = [record for record in accepted if record.get("promotion_from_candidate")]
    modified = [record for record in accepted if record.get("candidate_modified")]
    evidence_used = sorted({item for record in promoted + modified for item in record.get("verification_evidence_used", []) if item})
    return {
        "promotion_from_candidate": bool(promoted or modified),
        "candidate_modified": bool(modified),
        "revision_used": bool(modified),
        "promoted_wall_ids": [record.get("id") for record in promoted or modified],
        "modified_wall_ids": [record.get("id") for record in modified],
        "candidate_original_sources": sorted({str(record.get("candidate_original_source")) for record in promoted + modified if record.get("candidate_original_source")}),
        "normalized_source": "obb_from_mesh_footprint_min_rect" if (promoted or modified) else None,
        "verification_evidence_used": evidence_used,
        "revision_reasons": [record.get("revision_reason") for record in modified if record.get("revision_reason")],
        "items": [
            {
                "wall_id": record.get("id"),
                "promotion_from_candidate": bool(record.get("promotion_from_candidate") or record.get("candidate_modified")),
                "candidate_modified": bool(record.get("candidate_modified")),
                "revision_reason": record.get("revision_reason"),
                "candidate_original_source": record.get("candidate_original_source"),
                "normalized_source": record.get("normalized_source"),
                "verification_evidence_used": record.get("verification_evidence_used", []),
                "promotion_reason": record.get("promotion_reason"),
            }
            for record in promoted + [record for record in modified if record not in promoted]
        ],
    }


def summarize_weak_wall_completion(wall_records: list[dict[str, Any]]) -> dict[str, Any]:
    weak_records = [record for record in wall_records if record.get("accepted_as_final_wall") and record.get("weak_wall")]
    return {
        "used": bool(weak_records),
        "weak_wall_ids": [record.get("id") or record.get("wall_id") for record in weak_records],
        "completion_methods": sorted({str(record.get("completion_method")) for record in weak_records if record.get("completion_method")}),
        "evidence_strengths": sorted({str(record.get("evidence_strength")) for record in weak_records if record.get("evidence_strength")}),
        "completion_reasons": [record.get("completion_reason") for record in weak_records if record.get("completion_reason")],
        "items": [
            {
                "wall_id": record.get("id") or record.get("wall_id"),
                "evidence_strength": record.get("evidence_strength"),
                "completion_method": record.get("completion_method"),
                "completion_reason": record.get("completion_reason"),
                "weak_wall": bool(record.get("weak_wall")),
                "original_image_visibility": record.get("original_image_visibility"),
                "visible_role": record.get("visible_role"),
            }
            for record in weak_records
        ],
        "rule": "Weak or hidden walls may be completed by room closure only when other walls, room footprint, and layout evidence support a closed boundary.",
    }

def weak_or_inferred_wall_attach_blocks(wall_records: list[dict[str, Any]]) -> dict[str, str]:
    """Final walls may be weakly completed for room closure, but weak/hidden walls should not anchor attach relations."""
    blocked: dict[str, str] = {}
    for record in wall_records:
        if not isinstance(record, dict) or not record.get("accepted_as_final_wall"):
            continue
        wall_id = record.get("id") or record.get("wall_id")
        if not wall_id:
            continue
        evidence_strength = str(record.get("evidence_strength") or "").lower()
        completion_method = str(record.get("completion_method") or "").lower()
        completion_reason = str(record.get("completion_reason") or record.get("evidence") or "").lower()
        visibility = str(record.get("original_image_visibility") or record.get("visibility") or "").lower()
        weak_or_inferred = bool(
            record.get("can_anchor_wall_attach") is False
            or
            record.get("weak_wall")
            or record.get("closure_only")
            or evidence_strength == "weak"
            or completion_method == "inferred_by_room_closure"
            or visibility in {"hidden", "not_visible", "not directly visible", "inferred", "occluded"}
            or "hidden" in completion_reason
            or "not directly visible" in completion_reason
        )
        if weak_or_inferred:
            blocked[str(wall_id)] = (
                "Weak/hidden/inferred/closure-only wall or agent-disallowed attach anchor is not allowed as an attach anchor; "
                "attach requires a wall surface or wall-footline/contact visible in source evidence."
            )
    return blocked

def shift_walls_from_attach_policy(
    walls: list[dict[str, Any]],
    wall_records: list[dict[str, Any]],
    merge_report: dict[str, Any],
    relations: list[dict[str, Any]],
    offset_ratio: float = 0.05,
    max_area_growth_ratio: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Move only unattached virtual walls outward after final relation decisions."""
    filtered = [
        item for item in merge_report.get("skipped_candidates", [])
        if isinstance(item, dict) and item.get("reason") == "weak_or_inferred_wall_anchor_not_allowed_for_attach"
    ]
    triggered_targets: dict[str, set[str]] = {}
    for item in filtered:
        relation = item.get("relation") if isinstance(item.get("relation"), dict) else {}
        parameters = relation.get("parameters") if isinstance(relation.get("parameters"), dict) else {}
        wall_id = parameters.get("object_anchor")
        target = parameters.get("object_target")
        if wall_id in WALL_IDS:
            triggered_targets.setdefault(str(wall_id), set()).add(str(target or "unknown"))
    attached_wall_ids: set[str] = set()
    for relation in relations:
        if not isinstance(relation, dict) or relation.get("relation") != "attach":
            continue
        parameters = relation.get("parameters") if isinstance(relation.get("parameters"), dict) else {}
        anchor = parameters.get("object_anchor")
        if anchor in WALL_IDS:
            attached_wall_ids.add(str(anchor))
    unattached_wall_ids = [wall_id for wall_id in WALL_IDS if wall_id not in attached_wall_ids]
    virtual_wall_ids = set(weak_or_inferred_wall_attach_blocks(wall_records))
    # Run only after the complete relation set is final. An empty visible wall
    # is not evidence that its OBB boundary is too tight.
    triggered_wall_ids = [
        wall_id for wall_id in WALL_IDS
        if wall_id in virtual_wall_ids and wall_id in unattached_wall_ids
    ]
    trigger_reasons = {
        wall_id: [
            reason
            for condition, reason in [
                (wall_id in unattached_wall_ids, "virtual_wall_without_retained_wall_attach"),
                (wall_id in triggered_targets, "filtered_weak_wall_attach"),
            ]
            if condition
        ]
        for wall_id in triggered_wall_ids
    }
    base_audit = {
        "schema_version": "wall_attach_policy_outward_shift",
        "decision_source": "local_final_wall_attach_policy",
        "offset_ratio": offset_ratio,
        "attached_wall_ids": sorted(attached_wall_ids),
        "unattached_wall_ids": unattached_wall_ids,
        "virtual_wall_ids": sorted(virtual_wall_ids),
        "triggered_wall_ids": triggered_wall_ids,
        "trigger_reasons": trigger_reasons,
        "triggered_object_targets": {key: sorted(value) for key, value in triggered_targets.items()},
        "filtered_candidate_count": len(filtered),
    }
    if not triggered_wall_ids:
        return walls, wall_records, {
            **base_audit,
            "status": "no_change",
            "reason": "no_unattached_virtual_wall_requires_outward_shift",
        }
    by_id = {str(wall.get("id")): wall for wall in walls if isinstance(wall, dict)}
    if set(by_id) != set(WALL_IDS):
        return walls, wall_records, {**base_audit, "status": "rejected", "reason": "requires_four_canonical_walls"}
    ordered = [by_id[wall_id] for wall_id in WALL_IDS]
    points = [_safe_float_pair(wall.get("first_point")) for wall in ordered]
    second_points = [_safe_float_pair(wall.get("second_point")) for wall in ordered]
    if any(point is None for point in points + second_points):
        return walls, wall_records, {**base_audit, "status": "rejected", "reason": "invalid_wall_endpoint"}
    assert all(point is not None for point in points + second_points)
    if any(math.dist(second_points[index], points[(index + 1) % 4]) > 1e-5 for index in range(4)):
        return walls, wall_records, {**base_audit, "status": "rejected", "reason": "wall_segments_not_connected"}

    def area(vertices: list[list[float]]) -> float:
        return abs(sum(
            vertices[index][0] * vertices[(index + 1) % 4][1]
            - vertices[(index + 1) % 4][0] * vertices[index][1]
            for index in range(4)
        )) * 0.5

    center = [sum(point[axis] for point in points) / 4.0 for axis in range(2)]
    lines: list[dict[str, Any]] = []
    for index, first in enumerate(points):
        second = points[(index + 1) % 4]
        dx, dz = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dz)
        if length <= 1e-12:
            return walls, wall_records, {**base_audit, "status": "rejected", "reason": "degenerate_wall_segment"}
        midpoint = [(first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5]
        away = [midpoint[0] - center[0], midpoint[1] - center[1]]
        normals = [(-dz / length, dx / length), (dz / length, -dx / length)]
        normal = max(normals, key=lambda value: value[0] * away[0] + value[1] * away[1])
        lines.append({"normal": normal, "constant": normal[0] * first[0] + normal[1] * first[1], "length": length})
    short_side = min(line["length"] for line in lines)
    offset = short_side * offset_ratio
    revised = [dict(line) for line in lines]
    for index, wall_id in enumerate(WALL_IDS):
        if wall_id in triggered_wall_ids:
            revised[index]["constant"] += offset

    def intersection(first: dict[str, Any], second: dict[str, Any]) -> list[float]:
        a1, b1 = first["normal"]
        a2, b2 = second["normal"]
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) <= 1e-10:
            raise ValueError("adjacent wall support lines are parallel")
        c1, c2 = first["constant"], second["constant"]
        return [(c1 * b2 - c2 * b1) / determinant, (a1 * c2 - a2 * c1) / determinant]

    try:
        after_points = [intersection(revised[index - 1], revised[index]) for index in range(4)]
    except ValueError as exc:
        return walls, wall_records, {**base_audit, "status": "rejected", "reason": str(exc)}
    before_area, after_area = area(points), area(after_points)
    growth = after_area / before_area - 1.0 if before_area > 1e-12 else float("inf")
    original_contained = all(
        all(line["normal"][0] * point[0] + line["normal"][1] * point[1] <= line["constant"] + 1e-7 for line in revised)
        for point in points
    )
    errors = []
    if after_area <= before_area + 1e-10:
        errors.append("area_did_not_increase")
    if not original_contained:
        errors.append("original_obb_not_contained")
    if growth > max_area_growth_ratio + 1e-9:
        errors.append("area_growth_exceeds_limit")
    if not all(math.isfinite(value) for point in after_points for value in point):
        errors.append("non_finite_intersection")
    if errors:
        return walls, wall_records, {**base_audit, "status": "rejected", "validation_errors": errors}

    revised_by_id: dict[str, dict[str, Any]] = {}
    for index, wall_id in enumerate(WALL_IDS):
        revised_wall = dict(by_id[wall_id])
        revised_wall["first_point"] = [round(value, 6) for value in after_points[index]]
        revised_wall["second_point"] = [round(value, 6) for value in after_points[(index + 1) % 4]]
        revised_by_id[wall_id] = revised_wall
    updated_walls = [revised_by_id.get(str(wall.get("id")), wall) for wall in walls]
    updated_records = []
    for record in wall_records:
        updated = dict(record)
        wall_id = str(record.get("id") or record.get("wall_id") or "")
        if wall_id in revised_by_id:
            updated["first_point"] = revised_by_id[wall_id]["first_point"]
            updated["second_point"] = revised_by_id[wall_id]["second_point"]
            updated["virtual_wall_shift"] = {
                "triggered_by_filtered_attach": wall_id in triggered_targets,
                "triggered_by_no_retained_attach": wall_id in virtual_wall_ids and wall_id in unattached_wall_ids,
                "trigger_reasons": trigger_reasons.get(wall_id, []),
                "support_line_offset_xz_units": round(offset if wall_id in triggered_wall_ids else 0.0, 6),
            }
        updated_records.append(updated)
    return updated_walls, updated_records, {
        **base_audit,
        "status": "applied",
        "shifted_support_wall_ids": triggered_wall_ids,
        "offset_xz_units": round(offset, 6),
        "before_area_xz": round(before_area, 8),
        "after_area_xz": round(after_area, 8),
        "area_growth_ratio": round(growth, 8),
        "original_obb_contained": original_contained,
        "policy": "After all relation decisions are finalized, only canonical weak/inferred/closure-only walls without a retained wall attach move outward once. Visible walls remain fixed even when no object attaches to them.",
    }


def _safe_float_pair(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except Exception:
        return None


def _load_xz_to_pixel_affine_from_wall_evidence(wall_evidence: dict[str, Any] | None) -> list[list[float]] | None:
    if not isinstance(wall_evidence, dict):
        return None
    candidates: list[Any] = []
    precompute = wall_evidence.get("render_grid_wall_candidates")
    if isinstance(precompute, dict):
        candidates.append(precompute.get("render_grid_affine_path"))
    evidence = wall_evidence.get("topdown_verification")
    if isinstance(evidence, dict):
        candidates.append(evidence.get("render_grid_affine_path"))
    topdown = wall_evidence.get("topdown")
    if isinstance(topdown, dict):
        candidates.append(topdown.get("render_grid_affine_path"))
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw))
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        affine = data.get("xz_to_pixel_affine")
        if isinstance(affine, list) and len(affine) == 3:
            return affine
    return None


def _xz_to_screen_xy(point: list[float], affine: list[list[float]] | None) -> list[float]:
    x, z = float(point[0]), float(point[1])
    if affine:
        return [
            float(affine[0][0]) * x + float(affine[1][0]) * z + float(affine[2][0]),
            float(affine[0][1]) * x + float(affine[1][1]) * z + float(affine[2][1]),
        ]
    # PNG/screen coordinates grow downward, so use -z as a conservative fallback.
    return [x, -z]


def _signed_area_xy(points: list[list[float]]) -> float:
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return total / 2.0


def _segment_midpoint_xy(first: list[float], second: list[float], affine: list[list[float]] | None) -> list[float]:
    a = _xz_to_screen_xy(first, affine)
    b = _xz_to_screen_xy(second, affine)
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]


def _point_close(a: list[float], b: list[float], eps: float = 1e-6) -> bool:
    return abs(float(a[0]) - float(b[0])) <= eps and abs(float(a[1]) - float(b[1])) <= eps


def _match_old_wall_id(first: list[float], second: list[float], original_walls: list[dict[str, Any]]) -> str | None:
    for wall in original_walls:
        old_first = _safe_float_pair(wall.get("first_point"))
        old_second = _safe_float_pair(wall.get("second_point"))
        if not old_first or not old_second:
            continue
        if (_point_close(first, old_first) and _point_close(second, old_second)) or (_point_close(first, old_second) and _point_close(second, old_first)):
            return str(wall.get("id"))
    return None


def _record_by_old_wall_id(wall_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in wall_records:
        if isinstance(record, dict) and record.get("accepted_as_final_wall") and record.get("id"):
            result[str(record["id"])] = record
    return result


def _wall_record_aliases(record: dict[str, Any]) -> list[str]:
    aliases: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            value_s = value.strip()
            if value_s and value_s not in aliases:
                aliases.append(value_s)
        elif isinstance(value, (int, float)):
            value_s = str(value)
            if value_s not in aliases:
                aliases.append(value_s)

    for key in [
        "candidate_or_geometry_wall_id",
        "candidate_wall_id",
        "geometry_wall_id",
        "provisional_wall_id",
        "original_wall_id_before_canonicalization",
        "id",
        "wall_id",
    ]:
        add(record.get(key))
    formal = record.get("formal_obb_segment")
    if isinstance(formal, dict):
        for key in ["geometry_wall_id", "provisional_wall_id"]:
            add(formal.get(key))
    return aliases


def _wall_geometry_aliases(record: dict[str, Any]) -> list[str]:
    aliases: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        value_s = str(value).strip()
        if value_s and value_s not in aliases:
            aliases.append(value_s)

    for key in [
        "candidate_or_geometry_wall_id",
        "candidate_wall_id",
        "geometry_wall_id",
        "provisional_wall_id",
    ]:
        add(record.get(key))
    formal = record.get("formal_obb_segment")
    if isinstance(formal, dict):
        for key in ["geometry_wall_id", "provisional_wall_id"]:
            add(formal.get(key))
    return aliases


def _record_by_wall_geometry_alias(wall_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in wall_records:
        if not isinstance(record, dict) or not record.get("accepted_as_final_wall"):
            continue
        for alias in _wall_geometry_aliases(record):
            result.setdefault(alias, record)
    return result


def build_wall_relation_id_remap(wall_records: list[dict[str, Any]], wall_canonicalization_gate: dict[str, Any] | None = None) -> dict[str, str]:
    remap: dict[str, str] = {}
    gate_map = wall_canonicalization_gate.get("old_to_new_wall_id") if isinstance(wall_canonicalization_gate, dict) else None
    if isinstance(gate_map, dict):
        for old_id, new_id in gate_map.items():
            if old_id and new_id:
                remap[str(old_id)] = str(new_id)
    for record in wall_records:
        if not isinstance(record, dict) or not record.get("accepted_as_final_wall"):
            continue
        final_id = record.get("id") or record.get("wall_id")
        if final_id not in WALL_IDS:
            continue
        for alias in _wall_record_aliases(record):
            if alias in WALL_IDS:
                remap.setdefault(alias, str(final_id))
            else:
                remap[alias] = str(final_id)
    return remap


def _visible_count_from_records(records_by_id: dict[str, dict[str, Any]]) -> int | None:
    counts: list[int] = []
    for record in records_by_id.values():
        raw = record.get("original_visible_wall_count")
        try:
            counts.append(int(raw))
        except Exception:
            continue
    if not counts:
        return None
    return max(set(counts), key=counts.count)


def _formal_obb_primary_candidate(wall_evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(wall_evidence, dict):
        return None
    for key in ("render_grid_wall_candidates",):
        source = wall_evidence.get(key)
        if not isinstance(source, dict):
            continue
        primary = source.get("primary_candidate")
        if isinstance(primary, dict) and primary.get("method") == FORMAL_OBB_WALL_METHOD:
            return primary
    return None


def _authoritative_obb_walls_from_evidence(
    walls: list[dict[str, Any]],
    wall_records: list[dict[str, Any]],
    wall_evidence: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replace final wall coordinates with the formal OBB candidate.

    Agents may still adjudicate visibility and weak/hidden wall metadata, but
    the coordinate source for final wall_001..wall_004 is fixed locally to
    ``obb_from_mesh_footprint_min_rect``. This prevents old layout/mask/LSD
    wall coordinates from re-entering final objects.json.
    """
    primary = _formal_obb_primary_candidate(wall_evidence)
    if not primary:
        return walls, wall_records, {
            "applied": False,
            "required_method": FORMAL_OBB_WALL_METHOD,
            "reason": "formal_obb_primary_candidate_missing",
        }
    selected_orientation_candidate_id = primary.get("candidate_id") or "formal_obb_primary"
    segments = primary.get("wall_segments")
    if not isinstance(segments, list) or len(segments) < 4:
        return walls, wall_records, {
            "applied": False,
            "required_method": FORMAL_OBB_WALL_METHOD,
            "reason": "formal_obb_primary_candidate_has_no_four_segments",
        }
    normalized_segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments[:4]):
        if not isinstance(segment, dict):
            continue
        first = _safe_float_pair(segment.get("first_point"))
        second = _safe_float_pair(segment.get("second_point"))
        if not first or not second:
            continue
        normalized_segments.append(
            {
                "id": segment.get("id") if segment.get("id") in WALL_IDS else WALL_IDS[idx],
                "first_point": [round(float(first[0]), 6), round(float(first[1]), 6)],
                "second_point": [round(float(second[0]), 6), round(float(second[1]), 6)],
                "segment": segment,
            }
        )
    if len(normalized_segments) != 4:
        return walls, wall_records, {
            "applied": False,
            "required_method": FORMAL_OBB_WALL_METHOD,
            "reason": "formal_obb_primary_candidate_segments_invalid",
            "valid_segment_count": len(normalized_segments),
        }

    # When formal OBB is authoritative, inherit only geometry/candidate metadata
    # from the agent output. Agent final wall ids are unstable across serial and
    # skeleton-async runs, so matching by id/wall_id can copy weak/visibility
    # metadata onto the wrong OBB segment.
    previous_records_by_alias = _record_by_wall_geometry_alias(wall_records)
    authoritative_source_path = "wall_evidence.render_grid_wall_candidates.primary_candidate.wall_segments"

    new_walls: list[dict[str, Any]] = []
    new_records: list[dict[str, Any]] = []
    matched_aliases: dict[str, str | None] = {}
    for idx, item in enumerate(normalized_segments):
        wall_id = item["id"]
        segment = item["segment"]
        segment_aliases = [
            segment.get("geometry_wall_id"),
            segment.get("provisional_wall_id"),
            segment.get("id"),
            wall_id,
        ]
        old_record: dict[str, Any] = {}
        matched_alias: str | None = None
        for alias in segment_aliases:
            if alias and str(alias) in previous_records_by_alias:
                old_record = previous_records_by_alias[str(alias)]
                matched_alias = str(alias)
                break
        matched_aliases[wall_id] = matched_alias
        new_walls.append({"id": wall_id, "first_point": item["first_point"], "second_point": item["second_point"]})
        record = dict(old_record) if isinstance(old_record, dict) else {}
        previous_evidence_used = record.get("verification_evidence_used")
        if isinstance(previous_evidence_used, list):
            verification_evidence_used = [str(item) for item in previous_evidence_used if item]
        elif previous_evidence_used:
            verification_evidence_used = [str(previous_evidence_used)]
        else:
            verification_evidence_used = []
        verification_evidence_used.extend([
            "render_grid_wall_candidates.primary_candidate",
            FORMAL_OBB_WALL_METHOD,
            f"wall_orientation_candidate:{selected_orientation_candidate_id}",
        ])
        record.update(
            {
                "id": wall_id,
                "wall_id": wall_id,
                "first_point": item["first_point"],
                "second_point": item["second_point"],
                "source": FORMAL_OBB_WALL_SOURCE,
                "original_source": FORMAL_OBB_WALL_SOURCE,
                "normalized_source": FORMAL_OBB_WALL_SOURCE,
                "topdown_verification": "local_formal_obb_gate",
                "evidence": "Final wall coordinates are locally overridden from render_grid_wall_candidates.primary_candidate using obb_from_mesh_footprint_min_rect.",
                "accepted": True,
                "accepted_as_final_wall": True,
                "reason": "formal OBB wall coordinate source is authoritative for final wall geometry",
                "candidate_original_source": FORMAL_OBB_WALL_SOURCE,
                "verification_evidence_used": sorted(set(verification_evidence_used)),
                "formal_wall_orientation_candidate_id": selected_orientation_candidate_id,
                "formal_wall_orientation": primary.get("orientation", {}),
                "formal_obb_segment": {
                    "provisional_wall_id": segment.get("provisional_wall_id"),
                    "geometry_wall_id": segment.get("geometry_wall_id"),
                    "clockwise_order_index": segment.get("clockwise_order_index"),
                    "source_method": segment.get("source_method"),
                    "matched_agent_wall_alias": matched_alias,
                    "matched_agent_wall_id": old_record.get("id") if isinstance(old_record, dict) else None,
                },
            }
        )
        record.setdefault("canonical_naming", segment.get("canonical_naming") or {})
        new_records.append(record)

    return new_walls, new_records, {
        "applied": True,
        "method": FORMAL_OBB_WALL_METHOD,
        "selected_orientation_candidate_id": selected_orientation_candidate_id,
        "selected_orientation": primary.get("orientation", {}),
        "coordinate_frame": "OpenGL original XZ; Y is vertical",
        "source": authoritative_source_path,
        "previous_wall_count": len(walls),
        "previous_accepted_record_count": len([r for r in wall_records if isinstance(r, dict) and r.get("accepted_as_final_wall")]),
        "new_wall_count": len(new_walls),
        "matched_agent_record_aliases": matched_aliases,
        "policy": "local OBB coordinate authority before screen-clockwise wall-id canonicalization",
        "note": "Agent wall coordinates are never used. Local code selects one finite orientation using Agent-identified wall-following objects; final coordinates come from that selected candidate and only semantic metadata is preserved.",
    }


def canonicalize_walls_for_screen_clockwise(
    walls: list[dict[str, Any]],
    wall_records: list[dict[str, Any]],
    wall_evidence: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Make final wall ids follow the human top-down clockwise convention.

    The model may output a geometrically valid wall_001..wall_004 set that is
    clockwise in OpenGL XZ but visually counter-clockwise in the rendered PNG.
    This gate keeps the coordinates, but renumbers/reverses the final wall
    records so downstream attach relations use the canonical human-facing wall
    ids. Candidate/provisional ids are allowed to be wrong here; final ids are
    not.
    """
    if len(walls) != 4 or any(wall.get("id") not in WALL_IDS for wall in walls):
        return walls, wall_records, {"applied": False, "reason": "requires_complete_wall_001_to_wall_004"}
    original_walls = [dict(wall) for wall in sorted(walls, key=lambda wall: WALL_IDS.index(wall["id"]))]
    vertices = []
    for wall in original_walls:
        point = _safe_float_pair(wall.get("first_point"))
        if not point:
            return walls, wall_records, {"applied": False, "reason": "invalid_wall_first_point"}
        vertices.append(point)
    affine = _load_xz_to_pixel_affine_from_wall_evidence(wall_evidence)
    screen_vertices = [_xz_to_screen_xy(point, affine) for point in vertices]
    screen_area_before = _signed_area_xy(screen_vertices)
    # In image/screen coordinates y grows downward; positive shoelace area is
    # the human-visible clockwise order.
    if screen_area_before < 0:
        vertices = list(reversed(vertices))
        screen_vertices = [_xz_to_screen_xy(point, affine) for point in vertices]
    screen_area_after = _signed_area_xy(screen_vertices)
    segments: list[dict[str, Any]] = []
    for idx, first in enumerate(vertices):
        second = vertices[(idx + 1) % len(vertices)]
        old_id = _match_old_wall_id(first, second, original_walls)
        segments.append(
            {
                "old_wall_id": old_id,
                "first_point": [round(float(first[0]), 6), round(float(first[1]), 6)],
                "second_point": [round(float(second[0]), 6), round(float(second[1]), 6)],
                "midpoint_screen_xy": _segment_midpoint_xy(first, second, affine),
            }
        )
    records_by_id = _record_by_old_wall_id(wall_records)
    visible_count = _visible_count_from_records(records_by_id)
    # Keep wall ids deterministic across serial/async runs. Earlier versions
    # rotated the screen-clockwise segment list according to agent-provided
    # visible_role metadata. That made wall ids vary when two API calls described
    # the same OBB edges differently. We now preserve the local OBB/screen order
    # and use agent visibility only as metadata, not as the numbering anchor.
    start_idx = 0
    ordered_segments = segments
    old_to_new: dict[str, str] = {}
    new_walls: list[dict[str, Any]] = []
    for idx, segment in enumerate(ordered_segments):
        new_id = WALL_IDS[idx]
        old_id = segment.get("old_wall_id")
        if old_id:
            old_to_new[str(old_id)] = new_id
        new_walls.append({"id": new_id, "first_point": segment["first_point"], "second_point": segment["second_point"]})
    # Fill any unmatched old ids conservatively to keep remap total for audits.
    for old_id in WALL_IDS:
        old_to_new.setdefault(old_id, old_id)
    by_new_id = {wall["id"]: wall for wall in new_walls}
    new_records: list[dict[str, Any]] = []
    for record in wall_records:
        if not isinstance(record, dict) or not record.get("id"):
            new_records.append(record)
            continue
        old_id = str(record.get("id"))
        new_record = dict(record)
        new_id = old_to_new.get(old_id, old_id)
        if old_id in WALL_IDS:
            new_record["original_wall_id_before_canonicalization"] = old_id
            new_record["id"] = new_id
            new_record["wall_id"] = new_id
            if new_id in by_new_id:
                new_record["first_point"] = by_new_id[new_id]["first_point"]
                new_record["second_point"] = by_new_id[new_id]["second_point"]
        new_records.append(new_record)
    return new_walls, new_records, {
        "applied": True,
        "policy": "screen_clockwise_wall_ids_deterministic_obb_order",
        "screen_coordinate_source": "render_grid_affine.xz_to_pixel_affine" if affine else "fallback_x_neg_z",
        "screen_signed_area_before": screen_area_before,
        "screen_signed_area_after": screen_area_after,
        "screen_order_interpretation": "clockwise" if screen_area_after > 0 else "counter_clockwise_or_degenerate",
        "original_visible_wall_count_used": visible_count,
        "start_segment_old_wall_id": ordered_segments[0].get("old_wall_id") if ordered_segments else None,
        "start_segment_policy": "deterministic_first_screen_clockwise_segment_no_agent_visible_role_rotation",
        "old_to_new_wall_id": old_to_new,
        "changed_wall_ids": {old: new for old, new in old_to_new.items() if old != new},
        "note": "Final wall ids are canonicalized for the rendered top-down image, not for raw XZ signed-area order.",
    }



def write_wall_audit(
    output_dir: Path,
    walls: list[dict[str, Any]],
    wall_records: list[dict[str, Any]],
    warnings: list[str],
    dry_run: bool,
    wall_final_adjudication: dict[str, Any] | None = None,
    wall_canonicalization_gate: dict[str, Any] | None = None,
    authoritative_wall_geometry_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    geometry = validate_wall_geometry(walls) if walls else {"valid": False, "errors": ["No complete final-eligible wall set available."], "warnings": warnings, "details": {}}
    accepted_wall_records = [record for record in wall_records if record.get("accepted_as_final_wall")]
    naming_audit_source = accepted_wall_records if accepted_wall_records else walls
    data = {
        "dry_run": dry_run,
        "status": "completed" if geometry.get("valid") else "failed",
        "walls": walls,
        "wall_source_records": wall_records,
        "coordinate_source_definition": "final wall coordinates are layout XZ coordinates; topdown/grid/fixed views are verification and auxiliary evidence",
        "orientation_rule": "Final walls may be axis-aligned or oblique; final assembly requires final-eligible source, closure, adjacent endpoint consistency, and clockwise geometry, not x-constant/z-constant edges.",
        "wall_canonical_naming_policy": canonical_wall_naming_policy(),
        "wall_canonical_naming_audit": final_wall_naming_audit(naming_audit_source),
        "allowed_final_wall_sources": sorted(ALLOWED_FINAL_WALL_SOURCES),
        "intermediate_wall_sources": sorted(INTERMEDIATE_WALL_SOURCES),
        "hint_only_sources": sorted(HINT_ONLY_WALL_SOURCES),
        "geometry": geometry,
        "warnings": warnings,
        "wall_final_adjudication": wall_final_adjudication or {"executed": False, "decision": None, "accepted_final_walls": bool(walls and len(walls) == 4)},
        "authoritative_wall_geometry_gate": authoritative_wall_geometry_gate or {"applied": False},
        "wall_canonicalization_gate": wall_canonicalization_gate or {"applied": False},
        "candidate_wall_promotion": summarize_candidate_wall_promotion(wall_records),
        "weak_wall_completion": summarize_weak_wall_completion(wall_records),
    }
    write_audit_pair(output_dir, "walls", "wall_audit", data, "Wall Audit")
    return geometry
