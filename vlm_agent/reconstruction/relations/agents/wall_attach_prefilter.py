from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


WALL_IDS = ("wall_001", "wall_002", "wall_003", "wall_004")
WALL_SEMANTIC_LIST_KEYS = (
    "walls",
    "accepted_walls",
    "wall_visibility_review",
    "wall_visibility_semantics",
)
BLOCKED_VISIBILITY = {"hidden", "not_visible", "not visible", "inferred", "occluded"}
WALL_MOUNTED_ROLES = {"wall_mounted", "wall_embedded", "mounted", "embedded", "hung"}
POINT_GEOMETRY_WALL_MOUNT_ROLES = {"wall_mounted", "mounted", "hung"}
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
SEMANTIC_RECONSTRUCTION_DISTANCE_MULTIPLIER = 1.5
SELECTED_PANEL_BINDING_DISTANCE_MULTIPLIER = 2.0
SEMANTIC_CONFLICT_MIN_EDGE_LENGTH_RATIO = 0.50
SEMANTIC_CONFLICT_EDGE_RATIO_MARGIN = 0.20
SEMANTIC_CONFLICT_EDGE_LENGTH_MULTIPLIER = 1.50
SEMANTIC_CONFLICT_CENTER_DISTANCE_MULTIPLIER = 0.75
FLOOR_COVERING_TERMS = {
    "area rug",
    "area_rug",
    "carpet",
    "floor covering",
    "floor_covering",
    "floor mat",
    "floor_mat",
    "rug",
}


def _pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def wall_geometry_key(wall: dict[str, Any]) -> str | None:
    """Return an orientation-independent identity for one formal wall segment."""
    first = _pair(wall.get("first_point") or wall.get("first_point_xz"))
    second = _pair(wall.get("second_point") or wall.get("second_point_xz"))
    if first is None or second is None:
        return None
    endpoints = sorted((first, second))
    payload = "|".join(f"{x:.6f},{z:.6f}" for x, z in endpoints)
    return f"wall_edge_{hashlib.sha1(payload.encode('ascii')).hexdigest()[:12]}"


def formal_wall_references(wall_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        str(wall.get("id")): wall
        for wall in wall_evidence.get("candidate_walls", [])
        if isinstance(wall, dict) and wall.get("id") in WALL_IDS
    }
    references = []
    for wall_id in WALL_IDS:
        wall = by_id.get(wall_id)
        if not wall:
            continue
        references.append({
            "wall_id": wall_id,
            "geometry_key": wall_geometry_key(wall),
            "first_point": wall.get("first_point"),
            "second_point": wall.get("second_point"),
            "coordinate_source": "obb_from_mesh_footprint_min_rect",
        })
    return references


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def collect_wall_semantics_by_geometry(
    provider_results: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Bind Agent visibility semantics to immutable local wall geometry."""
    reference_by_key = {item.get("geometry_key"): item for item in references if item.get("geometry_key")}
    reference_by_id = {item.get("wall_id"): item for item in references if item.get("wall_id")}
    semantics: dict[str, dict[str, Any]] = {}
    unmatched = []
    for root in provider_results:
        if not isinstance(root, dict):
            continue
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            for list_key in WALL_SEMANTIC_LIST_KEYS:
                items = node.get(list_key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    explicit_key = item.get("geometry_key")
                    reported_key = explicit_key or wall_geometry_key(item)
                    reported_id = item.get("wall_id") or item.get("id")
                    reference = None
                    mismatch_reason = None
                    if reported_key:
                        reference = reference_by_key.get(reported_key)
                        id_reference = reference_by_id.get(reported_id)
                        if reference is None:
                            mismatch_reason = "geometry_key_not_in_formal_wall_references"
                        elif id_reference and id_reference.get("geometry_key") != reported_key:
                            mismatch_reason = "wall_id_geometry_key_mismatch"
                            reference = None
                    else:
                        reference = reference_by_id.get(reported_id)
                    if not reference:
                        unmatched.append({
                            "reported_wall_id": reported_id,
                            "geometry_key": reported_key,
                            "reason": mismatch_reason or "wall_reference_not_found",
                        })
                        continue
                    wall_id = str(reference["wall_id"])
                    record = semantics.setdefault(wall_id, {
                        "wall_id": wall_id,
                        "geometry_key": reference.get("geometry_key"),
                        "semantic_sources": [],
                    })
                    if list_key not in record["semantic_sources"]:
                        record["semantic_sources"].append(list_key)
                    for key in [
                        "original_image_visibility",
                        "visible_role",
                        "weak_wall",
                        "closure_only",
                        "can_anchor_wall_attach",
                        "evidence_strength",
                        "completion_method",
                        "completion_reason",
                        "reason",
                    ]:
                        if item.get(key) is not None:
                            record[key] = item.get(key)
    return semantics, {
        "binding_policy": (
            "strict_geometry_key_when_present_else_canonical_wall_id; "
            "conflicting wall_id/geometry_key records are rejected"
        ),
        "matched_wall_ids": sorted(semantics),
        "missing_wall_ids": [wall_id for wall_id in WALL_IDS if wall_id not in semantics],
        "unmatched_agent_records": unmatched,
    }


def _wall_is_blocked(semantic: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not semantic:
        return False, None
    visibility = str(semantic.get("original_image_visibility") or "").strip().lower()
    completion_method = str(semantic.get("completion_method") or "").strip().lower()
    blocked = bool(
        semantic.get("weak_wall")
        or semantic.get("closure_only")
        or semantic.get("can_anchor_wall_attach") is False
        or visibility in BLOCKED_VISIBILITY
        or completion_method == "inferred_by_room_closure"
    )
    if not blocked:
        return False, None
    return True, "agent_visibility_semantics_block_wall_attach_anchor"


def _objects_by_id(scene_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    candidates = scene_state.get("object_candidates", {})
    items = candidates.get("items", []) if isinstance(candidates, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id") or item.get("id")
        if object_id:
            result[str(object_id)] = {
                "object_id": str(object_id),
                "category": item.get("category"),
                "description": item.get("description"),
            }
    return result


def _hypotheses_by_object(wall_object_hypotheses: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    items = wall_object_hypotheses.get("likely_wall_objects", []) if isinstance(wall_object_hypotheses, dict) else []
    for item in items:
        if not isinstance(item, dict) or not item.get("object_id"):
            continue
        object_id = str(item["object_id"])
        current = result.get(object_id)
        score = CONFIDENCE_RANK.get(str(item.get("confidence") or "").lower(), 0)
        current_score = CONFIDENCE_RANK.get(str(current.get("confidence") or "").lower(), 0) if current else -1
        if current is None or score > current_score:
            result[object_id] = dict(item)
    return result


def _wall_line_hypotheses_by_object(
    wall_object_hypotheses: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    items = wall_object_hypotheses.get("wall_line_hypotheses", []) if isinstance(wall_object_hypotheses, dict) else []
    for item in items:
        if not isinstance(item, dict) or item.get("candidate_wall_id") not in WALL_IDS:
            continue
        object_ids = item.get("supporting_objects", [])
        if item.get("object_id"):
            object_ids = [*object_ids, item["object_id"]] if isinstance(object_ids, list) else [item["object_id"]]
        if not isinstance(object_ids, list):
            continue
        for object_id in object_ids:
            if object_id:
                result.setdefault(str(object_id), []).append(dict(item))
    return result


def _has_semantic_near_wall_geometry(metric: dict[str, Any]) -> bool:
    # Floor-supported semantic matches need an edge, not isolated point contact.
    if metric.get("possible_against_wall_contact") is not True:
        return False
    if metric.get("is_floor_supported") is not True:
        return True
    analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
    if analysis.get("has_valid_flat_contact_edge") is True:
        return True
    best_edge = analysis.get("best_edge") if isinstance(analysis.get("best_edge"), dict) else {}
    return (
        best_edge.get("flat_segment_candidate") is True
        and best_edge.get("parallel_to_wall") is True
        and best_edge.get("whole_edge_near_wall") is True
    )


def _is_semantic_reconstruction_uncertain_edge(metric: dict[str, Any]) -> bool:
    if metric.get("is_floor_supported") is not True or metric.get("possible_against_wall_contact") is not True:
        return False
    analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
    best_edge = analysis.get("best_edge") if isinstance(analysis.get("best_edge"), dict) else {}
    if analysis.get("analysis_available") is not True or best_edge.get("parallel_to_wall") is not True:
        return False
    try:
        distance_threshold = float(analysis.get("distance_threshold"))
        maximum_distance = float(best_edge.get("maximum_distance_to_wall"))
    except (TypeError, ValueError):
        return False
    return (
        distance_threshold > 0.0
        and maximum_distance > distance_threshold
        and maximum_distance <= distance_threshold * SEMANTIC_RECONSTRUCTION_DISTANCE_MULTIPLIER
    )


def _is_selected_panel_semantic_near_edge(metric: dict[str, Any]) -> bool:
    """Allow a review-only pair when image semantics bind a nearby parallel mesh edge."""
    if metric.get("is_floor_supported") is not True:
        return False
    analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
    best_edge = analysis.get("best_edge") if isinstance(analysis.get("best_edge"), dict) else {}
    if analysis.get("analysis_available") is not True or best_edge.get("parallel_to_wall") is not True:
        return False
    try:
        distance_threshold = float(analysis.get("distance_threshold"))
        maximum_distance = float(best_edge.get("maximum_distance_to_wall"))
    except (TypeError, ValueError):
        return False
    return bool(
        distance_threshold > 0.0
        and maximum_distance <= distance_threshold * SELECTED_PANEL_BINDING_DISTANCE_MULTIPLIER
    )


def _is_floor_covering(obj: dict[str, Any]) -> bool:
    text = " ".join(str(obj.get(key) or "").strip().lower() for key in ("category", "description"))
    return any(term in text for term in FLOOR_COVERING_TERMS)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flat_edge_signature(metric: dict[str, Any]) -> dict[str, Any]:
    analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
    best_edge = analysis.get("best_edge") if isinstance(analysis.get("best_edge"), dict) else {}
    return {
        "wall_id": str(metric.get("wall_id") or ""),
        "valid": analysis.get("has_valid_flat_contact_edge") is True,
        "edge_length": _float_or_none(best_edge.get("length")),
        "edge_length_ratio": _float_or_none(best_edge.get("length_ratio")),
        "maximum_distance_to_wall": _float_or_none(best_edge.get("maximum_distance_to_wall")),
        "distance_threshold": _float_or_none(analysis.get("distance_threshold")),
        "footprint_min_distance_to_wall": _float_or_none(metric.get("footprint_min_distance_to_wall")),
        "center_to_wall_distance": _float_or_none(metric.get("center_to_wall_distance")),
        "possible_against_wall_contact": metric.get("possible_against_wall_contact") is True,
        "is_floor_supported": metric.get("is_floor_supported") is True,
    }


def _walls_are_adjacent(first: str, second: str) -> bool:
    if first not in WALL_IDS or second not in WALL_IDS:
        return False
    delta = abs(WALL_IDS.index(first) - WALL_IDS.index(second))
    return delta in {1, len(WALL_IDS) - 1}


def _semantic_visibility_geometry_conflicts(
    object_id: str,
    role: str,
    confidence: str,
    metrics: list[dict[str, Any]],
    blocked_walls: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Reopen only strong wall-id conflicts for downstream Agent comparison.

    This does not change wall visibility or accept a relation locally. It keeps
    one blocked candidate in the existing recall call when object geometry
    strongly contradicts an adjacent visible-wall binding.
    """
    if CONFIDENCE_RANK.get(confidence, 0) < CONFIDENCE_RANK["high"]:
        return {}
    if role not in {"wall_mounted", "wall_embedded", "against_wall", "wall_aligned", "mounted", "embedded", "hung"}:
        return {}

    metric_by_wall = {
        str(metric.get("wall_id")): metric
        for metric in metrics
        if isinstance(metric, dict) and metric.get("wall_id") in WALL_IDS
    }
    signatures = {
        wall_id: _flat_edge_signature(metric)
        for wall_id, metric in metric_by_wall.items()
    }
    nearest_wall_id = min(
        signatures,
        key=lambda wall_id: (
            signatures[wall_id].get("footprint_min_distance_to_wall")
            if signatures[wall_id].get("footprint_min_distance_to_wall") is not None
            else float("inf"),
            wall_id,
        ),
        default=None,
    )
    unblocked_valid = [
        signature
        for wall_id, signature in signatures.items()
        if wall_id not in blocked_walls and signature.get("valid")
    ]
    unblocked_valid.sort(key=lambda signature: (
        -(signature.get("edge_length_ratio") or 0.0),
        signature.get("center_to_wall_distance")
        if signature.get("center_to_wall_distance") is not None
        else float("inf"),
        signature.get("wall_id") or "",
    ))
    best_unblocked = unblocked_valid[0] if unblocked_valid else None

    conflicts: list[dict[str, Any]] = []
    for wall_id in sorted(blocked_walls):
        signature = signatures.get(wall_id)
        if not signature or signature.get("possible_against_wall_contact") is not True:
            continue

        if role in WALL_MOUNTED_ROLES:
            if wall_id != nearest_wall_id:
                continue
            conflicts.append({
                "object_id": object_id,
                "wall_id": wall_id,
                "reason": "high_confidence_wall_mount_nearest_geometry_conflicts_with_blocked_visibility",
                "prior_block_reason": blocked_walls[wall_id],
                "geometry": signature,
                "competing_unblocked_wall": best_unblocked,
                "local_action": "reopen_for_existing_wall_attach_agent_review_only",
            })
            continue

        if signature.get("valid") is not True:
            continue
        edge_length = signature.get("edge_length") or 0.0
        edge_ratio = signature.get("edge_length_ratio") or 0.0
        max_distance = signature.get("maximum_distance_to_wall")
        threshold = signature.get("distance_threshold")
        within_edge_threshold = bool(
            max_distance is not None
            and threshold is not None
            and threshold > 0.0
            and max_distance <= threshold * SEMANTIC_RECONSTRUCTION_DISTANCE_MULTIPLIER
        )
        if not within_edge_threshold:
            continue

        if best_unblocked is None:
            strong_conflict = edge_ratio >= SEMANTIC_CONFLICT_MIN_EDGE_LENGTH_RATIO
        else:
            competing_wall_id = str(best_unblocked.get("wall_id") or "")
            if not _walls_are_adjacent(wall_id, competing_wall_id):
                continue
            competing_length = best_unblocked.get("edge_length") or 0.0
            competing_ratio = best_unblocked.get("edge_length_ratio") or 0.0
            center_distance = signature.get("center_to_wall_distance")
            competing_center_distance = best_unblocked.get("center_to_wall_distance")
            footprint_distance = signature.get("footprint_min_distance_to_wall")
            competing_footprint_distance = best_unblocked.get("footprint_min_distance_to_wall")
            stronger_edge = bool(
                edge_ratio >= max(
                    SEMANTIC_CONFLICT_MIN_EDGE_LENGTH_RATIO,
                    competing_ratio + SEMANTIC_CONFLICT_EDGE_RATIO_MARGIN,
                )
                and edge_length >= competing_length * SEMANTIC_CONFLICT_EDGE_LENGTH_MULTIPLIER
            )
            closer_center = bool(
                center_distance is not None
                and competing_center_distance is not None
                and center_distance <= competing_center_distance * SEMANTIC_CONFLICT_CENTER_DISTANCE_MULTIPLIER
            )
            no_farther_footprint = bool(
                footprint_distance is not None
                and competing_footprint_distance is not None
                and footprint_distance <= competing_footprint_distance
            )
            strong_conflict = stronger_edge and closer_center and no_farther_footprint
        if not strong_conflict:
            continue
        conflicts.append({
            "object_id": object_id,
            "wall_id": wall_id,
            "reason": "dominant_adjacent_flat_edge_conflicts_with_blocked_visibility",
            "prior_block_reason": blocked_walls[wall_id],
            "geometry": signature,
            "competing_unblocked_wall": best_unblocked,
            "local_action": "reopen_for_existing_wall_attach_agent_review_only",
        })

    # One conflict option exposes a likely adjacent-wall swap without restoring
    # an exhaustive object-wall scan.
    conflicts.sort(key=lambda item: (
        -(item.get("geometry", {}).get("edge_length_ratio") or 0.0),
        item.get("geometry", {}).get("footprint_min_distance_to_wall")
        if item.get("geometry", {}).get("footprint_min_distance_to_wall") is not None
        else float("inf"),
        str(item.get("wall_id") or ""),
    ))
    return {
        str(item["wall_id"]): item
        for item in conflicts[:1]
        if item.get("wall_id") in WALL_IDS
    }


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
    return {
        "object_id": metric.get("object_id"),
        "wall_id": metric.get("wall_id"),
        "footprint_min_distance_to_wall": metric.get("footprint_min_distance_to_wall"),
        "center_to_wall_distance": metric.get("center_to_wall_distance"),
        "possible_against_wall_contact": metric.get("possible_against_wall_contact"),
        "is_floor_supported": metric.get("is_floor_supported"),
        "is_thin_planar_candidate": metric.get("is_thin_planar_candidate"),
        "geometry_classification": metric.get("geometry_classification"),
        "flat_contact_edge_analysis": {
            "analysis_available": analysis.get("analysis_available"),
            "has_valid_flat_contact_edge": analysis.get("has_valid_flat_contact_edge"),
            "distance_threshold": analysis.get("distance_threshold"),
            "semantic_reconstruction_distance_multiplier": SEMANTIC_RECONSTRUCTION_DISTANCE_MULTIPLIER,
            "reason": analysis.get("reason"),
            "best_edge": analysis.get("best_edge"),
        },
    }


def build_wall_attach_prefilter(
    scene_state: dict[str, Any],
    wall_evidence: dict[str, Any],
    semantic_provider_results: list[dict[str, Any]],
    wall_object_hypotheses: dict[str, Any] | None,
    object_anchor_routing: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    references = formal_wall_references(wall_evidence)
    semantics, semantic_binding = collect_wall_semantics_by_geometry(semantic_provider_results, references)
    blocked_walls = {}
    for reference in references:
        wall_id = str(reference["wall_id"])
        blocked, reason = _wall_is_blocked(semantics.get(wall_id))
        if blocked:
            blocked_walls[wall_id] = reason

    objects = _objects_by_id(scene_state)
    hypotheses = _hypotheses_by_object(wall_object_hypotheses)
    wall_line_hypotheses = _wall_line_hypotheses_by_object(wall_object_hypotheses)
    routing_by_object = {
        str(item["object_id"]): {
            "anchor_mode": str(item.get("anchor_mode") or "uncertain").lower(),
            "confidence": str(item.get("confidence") or "low").lower(),
        }
        for item in object_anchor_routing or []
        if isinstance(item, dict) and item.get("object_id")
    }
    metrics_by_object: dict[str, list[dict[str, Any]]] = {}
    raw_pair_count = 0
    for metric in wall_evidence.get("wall_attach_geometry", []):
        if not isinstance(metric, dict) or metric.get("wall_id") not in WALL_IDS or not metric.get("object_id"):
            continue
        raw_pair_count += 1
        metrics_by_object.setdefault(str(metric["object_id"]), []).append(metric)

    candidates = []
    object_evaluations = []
    supplemental_visual_review_object_ids = []
    semantic_visibility_geometry_conflicts = []
    for object_id in sorted(objects):
        if _is_floor_covering(objects[object_id]):
            object_evaluations.append({
                "object_id": object_id,
                "decision": "excluded_floor_covering",
                "hypothesis_role": None,
                "hypothesis_confidence": None,
                "rejected_reasons": [{"reason": "floor_covering_cannot_anchor_or_target_wall_attach"}],
            })
            continue
        hypothesis = hypotheses.get(object_id, {})
        role = str(hypothesis.get("role") or "").lower()
        confidence = str(hypothesis.get("confidence") or "").lower()
        explicit_visual_mount = role in WALL_MOUNTED_ROLES
        high_confidence_visual_mount = explicit_visual_mount and CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK["high"]
        high_confidence_hanging_mount = (
            role in POINT_GEOMETRY_WALL_MOUNT_ROLES
            and CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK["high"]
        )
        routing = routing_by_object.get(object_id, {})
        routing_mode = str(routing.get("anchor_mode") or "uncertain")
        routing_confidence = str(routing.get("confidence") or "low")
        semantic_wall_alignment = (
            role in {"wall_mounted", "wall_embedded", "against_wall", "wall_aligned"}
            and CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK["medium"]
        )
        hard_ceiling_exclusion = routing_confidence == "high" and routing_mode == "ceiling_only"
        soft_non_wall_exclusion = (
            routing_confidence == "high"
            and routing_mode in {"support_only", "contain_candidate"}
            and not semantic_wall_alignment
        )
        if hard_ceiling_exclusion or soft_non_wall_exclusion:
            object_evaluations.append({
                "object_id": object_id,
                "decision": "excluded_by_high_confidence_anchor_routing",
                "anchor_routing": routing,
                "hypothesis_role": role or None,
                "hypothesis_confidence": confidence or None,
                "rejected_reasons": [{
                    "reason": (
                        "anchor_routing_ceiling_only_excludes_wall_review"
                        if hard_ceiling_exclusion
                        else f"anchor_routing_{routing_mode}_soft_exclusion_without_agent_wall_alignment"
                    )
                }],
            })
            continue
        object_wall_lines = wall_line_hypotheses.get(object_id, [])
        semantic_line_wall_ids = {
            str(item.get("candidate_wall_id"))
            for item in object_wall_lines
            if item.get("candidate_wall_id") in WALL_IDS
        }
        conflict_reopenings = _semantic_visibility_geometry_conflicts(
            object_id,
            role,
            confidence,
            metrics_by_object.get(object_id, []),
            blocked_walls,
        )
        semantic_visibility_geometry_conflicts.extend(conflict_reopenings.values())
        object_metrics = [
            metric for metric in metrics_by_object.get(object_id, [])
            if (
                str(metric.get("wall_id")) not in blocked_walls
                or str(metric.get("wall_id")) in conflict_reopenings
            )
        ]
        object_metrics.sort(key=lambda metric: (
            float(metric.get("footprint_min_distance_to_wall") or 1e9),
            str(metric.get("wall_id")),
        ))
        nearest_final_wall_id = str(object_metrics[0].get("wall_id")) if object_metrics else None
        nearest_neighbor_wall_ids = {
            str(metric.get("wall_id")) for metric in object_metrics[:2]
        }
        semantic_alignment_object = (
            role in {"wall_mounted", "wall_embedded", "against_wall", "wall_aligned"}
            and CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK["medium"]
        )
        eligible_metrics = []
        rejected_reasons = []
        for metric in metrics_by_object.get(object_id, []):
            wall_id = str(metric.get("wall_id"))
            semantic_geometry_conflict = conflict_reopenings.get(wall_id)
            if wall_id in blocked_walls and semantic_geometry_conflict is None:
                rejected_reasons.append({"wall_id": wall_id, "reason": blocked_walls[wall_id]})
                continue
            analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
            floor_supported = metric.get("is_floor_supported") is True
            valid_flat_edge = analysis.get("analysis_available") is True and analysis.get("has_valid_flat_contact_edge") is True
            possible_contact = metric.get("possible_against_wall_contact") is True
            thin_planar = metric.get("is_thin_planar_candidate") is True
            semantic_line_support = wall_id in semantic_line_wall_ids
            selected_panel_line_support = any(
                item.get("candidate_wall_id") == wall_id
                and item.get("source") == "selected_candidate_agent_semantic_binding"
                and CONFIDENCE_RANK.get(str(item.get("confidence") or "").lower(), 0) >= CONFIDENCE_RANK["medium"]
                for item in object_wall_lines
            )
            selected_panel_high_confidence_support = any(
                item.get("candidate_wall_id") == wall_id
                and item.get("source") == "selected_candidate_agent_semantic_binding"
                and CONFIDENCE_RANK.get(str(item.get("confidence") or "").lower(), 0) >= CONFIDENCE_RANK["high"]
                for item in object_wall_lines
            )
            local_nearest_alignment_support = (
                semantic_alignment_object
                and wall_id == nearest_final_wall_id
            )
            semantic_reconstruction_uncertain = (
                (semantic_line_support or local_nearest_alignment_support)
                and _is_semantic_reconstruction_uncertain_edge(metric)
            )
            if floor_supported:
                if semantic_geometry_conflict is not None and valid_flat_edge:
                    eligible = True
                    reason = "semantic_visibility_geometry_conflict_flat_edge_review"
                elif selected_panel_line_support and _is_selected_panel_semantic_near_edge(metric):
                    eligible = True
                    reason = "selected_candidate_semantic_wall_binding_with_near_parallel_edge"
                elif selected_panel_high_confidence_support and semantic_alignment_object:
                    # Exact original-image semantics may survive a missing SAM3D
                    # contact edge. This creates review input, never local acceptance.
                    eligible = True
                    reason = "selected_candidate_semantic_wall_binding_reconstruction_override"
                elif semantic_reconstruction_uncertain:
                    eligible = True
                    reason = (
                        "agent_wall_line_parallel_edge_within_reconstruction_uncertainty"
                        if semantic_line_support
                        else "agent_alignment_nearest_wall_parallel_edge_within_reconstruction_uncertainty"
                    )
                elif (
                    high_confidence_hanging_mount
                    and (semantic_line_support or wall_id in nearest_neighbor_wall_ids)
                ):
                    # A genuinely wall-mounted/hung object can have unreliable SAM3D
                    # contact points on one real wall. Keep at most the two nearest
                    # non-blocked alternatives for visual comparison; this remains
                    # candidate recall only, never a local final acceptance.
                    eligible = True
                    if possible_contact:
                        reason = (
                            "high_confidence_wall_mount_with_semantic_line_and_near_point_geometry"
                            if semantic_line_support
                            else "high_confidence_wall_mount_with_nearest_wall_point_geometry"
                        )
                    else:
                        reason = (
                            "high_confidence_wall_mount_with_semantic_line_geometry_uncertain_second_option"
                            if semantic_line_support
                            else "high_confidence_wall_mount_with_nearest_wall_geometry_uncertain_second_option"
                        )
                elif high_confidence_visual_mount and possible_contact:
                    eligible = _has_semantic_near_wall_geometry(metric)
                    reason = (
                        "high_confidence_wall_mounted_or_embedded_with_near_wall_edge"
                        if eligible else "high_confidence_wall_hypothesis_with_corner_only_geometry"
                    )
                else:
                    eligible = valid_flat_edge
                    reason = "floor_supported_real_lower_mesh_flat_edge" if eligible else "floor_supported_without_valid_lower_mesh_flat_edge"
            else:
                eligible = possible_contact and (explicit_visual_mount or thin_planar)
                reason = "visual_wall_mount_or_thin_planar_near_wall" if eligible else "non_floor_object_without_wall_mount_contact_evidence"
            if eligible:
                eligible_metrics.append((metric, reason))
            else:
                rejected_reasons.append({"wall_id": wall_id, "reason": reason})

        if eligible_metrics:
            semantic_wall_rank = {
                str(item["candidate_wall_id"]): index
                for index, item in enumerate(object_wall_lines)
                if item.get("candidate_wall_id") in WALL_IDS
            }
            semantic_eligible_metrics = [
                pair for pair in eligible_metrics
                if str(pair[0].get("wall_id")) in semantic_wall_rank
                and (
                    _has_semantic_near_wall_geometry(pair[0])
                    or _is_semantic_reconstruction_uncertain_edge(pair[0])
                    or (
                        any(
                            item.get("candidate_wall_id") == str(pair[0].get("wall_id"))
                            and item.get("source") == "selected_candidate_agent_semantic_binding"
                            for item in object_wall_lines
                        )
                        and _is_selected_panel_semantic_near_edge(pair[0])
                    )
                    or pair[1] == "selected_candidate_semantic_wall_binding_reconstruction_override"
                )
            ]
            semantic_preference_applied = bool(semantic_eligible_metrics)
            selection_pool = semantic_eligible_metrics or eligible_metrics
            selection_pool.sort(key=lambda pair: (
                semantic_wall_rank.get(str(pair[0].get("wall_id")), len(semantic_wall_rank)),
                0 if pair[0].get("possible_against_wall_contact") is True else 1,
                float(pair[0].get("footprint_min_distance_to_wall") or 1e9),
                str(pair[0].get("wall_id")),
            ))
            selected_pairs = selection_pool[:2]
            for option_index, (metric, reason) in enumerate(selected_pairs, start=1):
                wall_id = str(metric["wall_id"])
                selected_wall_line = next(
                    (item for item in object_wall_lines if item.get("candidate_wall_id") == wall_id),
                    None,
                )
                candidate = {
                    "object_target": object_id,
                    "candidate_wall_id": wall_id,
                    "candidate_group_id": object_id,
                    "candidate_option_index": option_index,
                    "candidate_option_count": len(selected_pairs),
                    "wall_geometry_key": next((item.get("geometry_key") for item in references if item.get("wall_id") == wall_id), None),
                    "candidate_source": (
                        reason
                        if reason in {
                            "agent_wall_line_parallel_edge_within_reconstruction_uncertainty",
                            "agent_alignment_nearest_wall_parallel_edge_within_reconstruction_uncertainty",
                            "selected_candidate_semantic_wall_binding_reconstruction_override",
                        }
                        else "agent_wall_line_hypothesis_with_near_wall_edge"
                        if semantic_preference_applied else reason
                    ),
                    "geometry_eligibility_source": reason,
                    "reconstruction_uncertainty_used": reason in {
                        "agent_wall_line_parallel_edge_within_reconstruction_uncertainty",
                        "agent_alignment_nearest_wall_parallel_edge_within_reconstruction_uncertainty",
                        "selected_candidate_semantic_wall_binding_reconstruction_override",
                    },
                    "object": objects[object_id],
                    "object_anchor_routing": routing or None,
                    "wall_object_hypothesis": hypothesis or None,
                    "wall_line_hypothesis": selected_wall_line,
                    "semantic_visibility_geometry_conflict": conflict_reopenings.get(wall_id),
                    "semantic_wall_preference_applied": semantic_preference_applied,
                    "geometry_evidence": _compact_metric(metric),
                    "agent_action": "compare_wall_options_then_accept_at_most_one",
                }
                candidates.append(candidate)
            decision = "candidate"
        else:
            semantic_review_line = next(
                (
                    item for item in object_wall_lines
                    if item.get("candidate_wall_id") in WALL_IDS
                    and item.get("candidate_wall_id") not in blocked_walls
                ),
                None,
            )
            if high_confidence_visual_mount and semantic_review_line is not None:
                wall_id = str(semantic_review_line["candidate_wall_id"])
                metric = next(
                    (
                        item for item in metrics_by_object.get(object_id, [])
                        if str(item.get("wall_id")) == wall_id
                    ),
                    None,
                )
                candidate = {
                    "object_target": object_id,
                    "candidate_wall_id": wall_id,
                    "wall_geometry_key": next(
                        (item.get("geometry_key") for item in references if item.get("wall_id") == wall_id),
                        None,
                    ),
                    "candidate_source": "high_confidence_visual_mount_with_semantic_wall_line_review",
                    "geometry_eligibility_source": "visual_semantic_mount_without_reliable_reconstruction_contact",
                    "reconstruction_uncertainty_used": True,
                    "object": objects[object_id],
                    "wall_object_hypothesis": hypothesis or None,
                    "wall_line_hypothesis": semantic_review_line,
                    "semantic_wall_preference_applied": True,
                    "geometry_evidence": (
                        _compact_metric(metric)
                        if isinstance(metric, dict)
                        else {
                            "object_id": object_id,
                            "wall_id": wall_id,
                            "geometry_classification": "visual_review_required_without_metric",
                        }
                    ),
                    "agent_action": "judge_accept_or_reject",
                }
                candidates.append(candidate)
                decision = "candidate"
            else:
                decision = "excluded_from_exhaustive_pair_scan"
                if explicit_visual_mount and object_id not in supplemental_visual_review_object_ids:
                    supplemental_visual_review_object_ids.append(object_id)
        object_evaluations.append({
            "object_id": object_id,
            "decision": decision,
            "hypothesis_role": role or None,
            "hypothesis_confidence": confidence or None,
            "rejected_reasons": rejected_reasons,
        })

    candidate_metrics = [candidate["geometry_evidence"] for candidate in candidates]
    return {
        "schema_version": "wall_attach_prefilter",
        "status": "available",
        "capability_contract": {
            "requires": ["formal_wall_geometry_ready", "wall_visibility_semantics_ready"],
            "provides": ["wall_attach_candidates_ready"],
            "policy": "Capability dependency is fixed; Agent task names and task grouping remain Manager-controlled.",
        },
        "policy": {
            "object_anchor_routing": "High-confidence ceiling_only is a hard wall-review exclusion. support_only and contain_candidate are soft routing hints: a medium/high Agent wall-mounted, embedded, against-wall, or wall-aligned finding may reopen local geometry review. floor_wall enters both Support and Wall.",
            "candidate_wall_options": "Keep up to two geometrically plausible neighboring walls for one object so the Agent, not nearest-edge ordering, selects the actual contact wall.",
            "ordinary_floor_supported": "Require a finite wall-parallel flat contact edge from the real lower-mesh XZ hull near a non-blocked wall; merge collinear fragments and reject isolated point/corner contact.",
            "semantic_wall_line_priority": "Prefer an exact selected-candidate Agent object-wall binding. Medium confidence still requires a nearby parallel reconstructed edge. A high-confidence binding may open one review pair when SAM3D loses that edge, but it never creates local acceptance. Generic nearest-wall/corner evidence cannot use this override.",
            "semantic_reconstruction_uncertainty": "When SAM3D placement leaves a small gap, either an Agent wall-line hypothesis or a high/medium-confidence Agent wall-alignment object mapped locally to its nearest final wall may enter review up to 1.5 times the adaptive edge-distance threshold, but only with a real parallel hull edge. This creates a review candidate, not a final attach.",
            "wall_mounted": "A high-confidence wall-mounted/embedded hypothesis with a non-blocked semantic wall line becomes an explicit Agent review pair even when SAM3D contact geometry is weak; local code does not auto-accept it.",
            "floor_covering": "Exclude rugs, carpets, and floor coverings from wall-attach review.",
            "weak_wall": "Visibility/weak semantics come from an Agent provider and are bound geometry-key-first. Missing semantics remain reviewable rather than locally blocked.",
            "semantic_geometry_conflict": "A high-confidence wall-following object may reopen one blocked adjacent wall only when its real geometry strongly contradicts the prior wall-id visibility binding. Reopening adds a pair to the existing Agent review; it never changes wall semantics or accepts attach locally.",
            "open_supplement": "The recall Agent may add an immediately obvious wall-mounted object, but must not exhaustively rescan every object-wall pair.",
        },
        "formal_wall_references": references,
        "wall_semantics": semantics,
        "object_anchor_routing": routing_by_object,
        "semantic_binding_audit": semantic_binding,
        "blocked_wall_ids": blocked_walls,
        "raw_object_wall_pair_count": raw_pair_count,
        "candidate_pair_count": len(candidates),
        "candidate_object_count": len({item["object_target"] for item in candidates}),
        "candidate_pairs": candidates,
        "candidate_wall_attach_geometry": candidate_metrics,
        "semantic_visibility_geometry_conflicts": semantic_visibility_geometry_conflicts,
        "supplemental_visual_review_object_ids": supplemental_visual_review_object_ids,
        "object_evaluations": object_evaluations,
    }


def compact_wall_evidence_for_recall(wall_evidence: dict[str, Any], prefilter: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_name": wall_evidence.get("scene_name"),
        "coordinate_source": "obb_from_mesh_footprint_min_rect",
        "canonical_topdown": wall_evidence.get("canonical_topdown", {}),
        "formal_wall_references": prefilter.get("formal_wall_references", []),
        "wall_semantics": prefilter.get("wall_semantics", {}),
        "blocked_wall_ids": prefilter.get("blocked_wall_ids", {}),
        "wall_attach_geometry": prefilter.get("candidate_wall_attach_geometry", []),
        "semantic_visibility_geometry_conflicts": prefilter.get(
            "semantic_visibility_geometry_conflicts", []
        ),
        "input_reduction": {
            "raw_object_wall_pair_count": prefilter.get("raw_object_wall_pair_count", 0),
            "candidate_pair_count": prefilter.get("candidate_pair_count", len(prefilter.get("candidate_pairs", []))),
        },
    }


def render_wall_attach_prefilter_overlay(
    canonical_topdown: str | Path | None,
    affine_path: str | Path | None,
    output_path: str | Path,
    wall_evidence: dict[str, Any],
    prefilter: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_path)
    if not canonical_topdown or not Path(canonical_topdown).exists():
        return {"status": "unavailable", "reason": "canonical_topdown_missing", "output_path": str(output)}
    if not affine_path or not Path(affine_path).exists():
        return {"status": "unavailable", "reason": "render_grid_affine_missing", "output_path": str(output)}
    try:
        import cv2
    except ImportError:
        return {"status": "unavailable", "reason": "opencv_missing", "output_path": str(output)}
    image = cv2.imread(str(canonical_topdown))
    if image is None:
        return {"status": "unavailable", "reason": "canonical_topdown_unreadable", "output_path": str(output)}
    affine_data = json.loads(Path(affine_path).read_text(encoding="utf-8"))
    affine = affine_data.get("xz_to_pixel_affine")
    if not (isinstance(affine, list) and len(affine) == 3):
        return {"status": "unavailable", "reason": "xz_to_pixel_affine_missing", "output_path": str(output)}

    def project(point: Any) -> tuple[int, int] | None:
        pair = _pair(point)
        if pair is None:
            return None
        x, z = pair
        px = x * float(affine[0][0]) + z * float(affine[1][0]) + float(affine[2][0])
        py = x * float(affine[0][1]) + z * float(affine[1][1]) + float(affine[2][1])
        return int(round(px)), int(round(py))

    blocked = set(prefilter.get("blocked_wall_ids", {}))
    semantic_by_wall = prefilter.get("wall_semantics", {})
    wall_midpoints = {}
    for wall in wall_evidence.get("candidate_walls", []):
        if not isinstance(wall, dict) or wall.get("id") not in WALL_IDS:
            continue
        wall_id = str(wall["id"])
        first, second = project(wall.get("first_point")), project(wall.get("second_point"))
        if first is None or second is None:
            continue
        color = (120, 120, 120) if wall_id in blocked else (30, 180, 30)
        cv2.arrowedLine(image, first, second, color, 4, cv2.LINE_AA, tipLength=0.035)
        midpoint = ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
        wall_midpoints[wall_id] = midpoint
        visibility = str(semantic_by_wall.get(wall_id, {}).get("original_image_visibility") or "unknown")
        label = f"{wall_id} {'BLOCKED' if wall_id in blocked else visibility}"
        cv2.putText(image, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    footprint_by_id = {
        str(item.get("object_id")): item
        for item in wall_evidence.get("object_footprints", [])
        if isinstance(item, dict) and item.get("object_id")
    }
    for index, candidate in enumerate(prefilter.get("candidate_pairs", [])):
        object_id = str(candidate.get("object_target"))
        wall_id = str(candidate.get("candidate_wall_id"))
        footprint = footprint_by_id.get(object_id, {})
        center = project(footprint.get("center_xz"))
        if center is None:
            metric = candidate.get("geometry_evidence", {})
            bbox_min, bbox_max = _pair(metric.get("bbox_xz_min")), _pair(metric.get("bbox_xz_max"))
            if bbox_min and bbox_max:
                center = project([(bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5])
        if center is None:
            continue
        color = (0, 80 + (index * 53) % 160, 255)
        cv2.circle(image, center, 9, color, -1, cv2.LINE_AA)
        cv2.putText(image, object_id, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        if wall_id in wall_midpoints:
            cv2.line(image, center, wall_midpoints[wall_id], color, 1, cv2.LINE_AA)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        return {"status": "unavailable", "reason": "overlay_write_failed", "output_path": str(output)}
    return {
        "status": "available",
        "output_path": str(output),
        "candidate_pair_count": prefilter.get("candidate_pair_count", 0),
        "blocked_wall_ids": sorted(blocked),
    }
