from __future__ import annotations

from pathlib import Path
from typing import Any

from relations.agents.state import sorted_object_id_key, write_json, write_md
from relations.agents.wall_attach_prefilter import formal_wall_references
from relations.tools.wall_orientation_candidates import axis_angle_distance_degrees

CANONICAL_WALL_IDS = ("wall_001", "wall_002", "wall_003", "wall_004")
PROVISIONAL_WALL_VISIBILITY_CAPABILITY = "provisional_wall_visibility_ready"
CANONICAL_WALL_NAMING_INSTRUCTION = (
    "Preserve the provided canonical screen-clockwise wall ids exactly. The formal OBB preprocessing and local final assembler own wall coordinates and id order. "
    "Judge original-image visibility and weak/inferred status for each existing wall, but do not rotate, renumber, or replace wall ids based on visibility. "
    "Candidate wall ids are provisional only until the local canonical remap is applied. Weak/hidden/inferred closure walls may complete room geometry, "
    "but mark them weak_wall=true and original_image_visibility=not_visible or inferred; do not use them as ordinary attach anchors without visible wall contact."
)

def _as_instruction_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def append_task_instruction(task: dict[str, Any], instruction: str) -> None:
    instructions = _as_instruction_list(task.get("instructions"))
    if instruction not in instructions:
        instructions.append(instruction)
    task["instructions"] = instructions


def task_identifier(task: dict[str, Any]) -> str:
    return str(task.get("task_id") or task.get("id") or "unknown_task")


def task_capabilities(task: dict[str, Any], field: str) -> list[str]:
    value = task.get(field)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def task_provides_capability(task: dict[str, Any], capability: str) -> bool:
    return capability in task_capabilities(task, "provides_capabilities")


def task_requires_capability(task: dict[str, Any], capability: str) -> bool:
    return capability in task_capabilities(task, "requires_capabilities")


def is_wall_attach_recall_task(task: dict[str, Any]) -> bool:
    canonical_kind = str(task.get("canonical_task_kind") or "").lower()
    task_type = str(task.get("task_type") or "").lower()
    task_id = task_identifier(task).lower()
    return bool(
        canonical_kind == "wall_attach_candidate_recall"
        or "wall_attach_candidate_recall" in task_type
        or "wall_attach_candidate_recall" in task_id
    )


def hydrate_wall_attach_candidate_scope(
    task: dict[str, Any],
    prefilter: dict[str, Any],
) -> dict[str, Any]:
    """Expand a Manager-selected scope with candidates created after planning."""
    manager_scope = [
        str(object_id)
        for object_id in task.get("object_scope", [])
        if isinstance(object_id, str) and object_id.strip()
    ]
    candidate_scope: list[str] = []
    for pair in prefilter.get("candidate_pairs", []):
        if not isinstance(pair, dict):
            continue
        object_id = pair.get("object_target")
        if isinstance(object_id, str) and object_id.strip() and object_id not in candidate_scope:
            candidate_scope.append(object_id)

    effective_scope = list(dict.fromkeys([*manager_scope, *candidate_scope]))
    added_candidate_object_ids = [object_id for object_id in candidate_scope if object_id not in manager_scope]
    task["object_scope"] = effective_scope
    scope_audit = {
        "policy": "manager_scope_union_local_prefilter_candidates",
        "manager_object_scope": manager_scope,
        "prefilter_candidate_object_ids": candidate_scope,
        "added_candidate_object_ids": added_candidate_object_ids,
        "effective_object_scope": effective_scope,
        "candidate_pairs_require_explicit_decision": True,
    }
    task["wall_attach_scope_hydration"] = scope_audit
    append_task_instruction(
        task,
        "The effective object_scope is the union of the Manager-selected scope and objects produced later by the local wall-attach prefilter. "
        "Return an explicit accept or reject decision for every pair in wall_attach_prefilter.candidate_pairs, including candidates outside the Manager's original scope. "
        "Do not silently omit a prefiltered pair.",
    )
    if prefilter.get("semantic_visibility_geometry_conflicts"):
        append_task_instruction(
            task,
            "Some candidate pairs carry semantic_visibility_geometry_conflict because "
            "a high-confidence wall-following object has substantially stronger local "
            "contact geometry on an adjacent wall that an earlier visibility record blocked. "
            "For those pairs, re-check the original image against the numbered final-wall "
            "overlay instead of treating the earlier visibility label as authoritative. "
            "If the wall-id binding was wrong, correct that wall's wall_visibility_review "
            "record and then judge attach; geometry alone is never sufficient for acceptance.",
        )
    return scope_audit


def annotate_semantic_visibility_geometry_conflict_resolutions(
    task: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    """Grant correction authority only to an explicitly accepted conflict pair."""
    prefilter = task.get("wall_attach_prefilter") if isinstance(task.get("wall_attach_prefilter"), dict) else {}
    conflicts = prefilter.get("semantic_visibility_geometry_conflicts", [])
    if not isinstance(conflicts, list) or not conflicts:
        return response

    accepted_pairs: set[tuple[str, str]] = set()
    for item in response.get("wall_attach_candidates", []):
        if not isinstance(item, dict) or str(item.get("decision") or "").lower() != "accept":
            continue
        object_id = item.get("object_target")
        wall_id = item.get("candidate_wall_id")
        if isinstance(object_id, str) and wall_id in CANONICAL_WALL_IDS:
            accepted_pairs.add((object_id, str(wall_id)))
    for item in response.get("accepted_attach_candidates", []):
        if not isinstance(item, dict):
            continue
        relation = item.get("relation") if isinstance(item.get("relation"), dict) else item
        parameters = relation.get("parameters") if isinstance(relation.get("parameters"), dict) else {}
        object_id = parameters.get("object_target")
        wall_id = parameters.get("object_anchor")
        if isinstance(object_id, str) and wall_id in CANONICAL_WALL_IDS:
            accepted_pairs.add((object_id, str(wall_id)))

    reviews = response.get("wall_visibility_review", [])
    review_by_wall = {
        str(item.get("wall_id") or item.get("id")): item
        for item in reviews
        if isinstance(item, dict) and (item.get("wall_id") or item.get("id")) in CANONICAL_WALL_IDS
    } if isinstance(reviews, list) else {}
    resolutions = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        object_id = str(conflict.get("object_id") or "")
        wall_id = str(conflict.get("wall_id") or "")
        review = review_by_wall.get(wall_id)
        if (object_id, wall_id) not in accepted_pairs or not isinstance(review, dict):
            continue
        visibility = str(review.get("original_image_visibility") or "").strip().lower()
        if (
            visibility not in {"visible", "partially_visible"}
            or review.get("can_anchor_wall_attach") is not True
            or bool(review.get("weak_wall"))
            or bool(review.get("closure_only"))
        ):
            continue
        review["semantic_visibility_geometry_conflict_resolved"] = True
        object_ids = review.setdefault("semantic_visibility_geometry_conflict_object_ids", [])
        if isinstance(object_ids, list) and object_id not in object_ids:
            object_ids.append(object_id)
        resolutions.append({
            "object_id": object_id,
            "wall_id": wall_id,
            "decision": "accepted_pair_and_visible_wall_correct_prior_binding",
            "prior_block_reason": conflict.get("prior_block_reason"),
        })
    if resolutions:
        response["semantic_visibility_geometry_conflict_resolutions"] = resolutions
    return response


CONTAIN_SCOPE_CONFIDENCE_THRESHOLD = 0.90
CONTAIN_SCOPE_MAX_HIGH_CONFIDENCE_PAIRS_PER_ANCHOR = 32
CONTAIN_TASK_MAX_REVIEW_PAIRS = 80


def _normalized_contain_pair(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else item
    anchor = parameters.get("object_anchor")
    target = parameters.get("object_target")
    if not (
        isinstance(anchor, str)
        and anchor.startswith("object_")
        and isinstance(target, str)
        and target.startswith("object_")
        and anchor != target
    ):
        return None
    try:
        confidence = float(item.get("confidence_hint", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "object_anchor": anchor,
        "object_target": target,
        "confidence_hint": confidence,
        "source": item,
    }


def select_contain_review_pairs(
    candidates: dict[str, Any],
    max_pairs: int = CONTAIN_TASK_MAX_REVIEW_PAIRS,
) -> dict[str, Any]:
    """Select one deterministic, anchor-balanced pair set for scope and API payload."""
    by_anchor: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    invalid_pair_count = 0
    for item in candidates.get("candidate_pairs", []):
        pair = _normalized_contain_pair(item)
        if pair is None:
            invalid_pair_count += 1
            continue
        key = (pair["object_anchor"], pair["object_target"])
        if key in seen:
            continue
        seen.add(key)
        by_anchor.setdefault(pair["object_anchor"], []).append(pair)
    per_anchor_overflow: list[dict[str, Any]] = []
    for anchor, anchor_pairs in by_anchor.items():
        anchor_pairs.sort(key=lambda pair: (-float(pair["confidence_hint"]), pair["object_target"]))
        per_anchor_overflow.extend(anchor_pairs[CONTAIN_SCOPE_MAX_HIGH_CONFIDENCE_PAIRS_PER_ANCHOR:])
        by_anchor[anchor] = anchor_pairs[:CONTAIN_SCOPE_MAX_HIGH_CONFIDENCE_PAIRS_PER_ANCHOR]

    selected: list[dict[str, Any]] = []
    anchor_ids = sorted(by_anchor, key=sorted_object_id_key)
    round_index = 0
    limit = max(0, int(max_pairs))
    while len(selected) < limit:
        added = False
        for anchor in anchor_ids:
            anchor_pairs = by_anchor[anchor]
            if round_index >= len(anchor_pairs):
                continue
            selected.append(anchor_pairs[round_index])
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        round_index += 1

    selected_keys = {(pair["object_anchor"], pair["object_target"]) for pair in selected}
    dropped = [
        *per_anchor_overflow,
        *[
            pair
            for anchor in anchor_ids
            for pair in by_anchor[anchor]
            if (pair["object_anchor"], pair["object_target"]) not in selected_keys
        ],
    ]
    return {
        "selected_pairs": selected,
        "dropped_pairs": dropped,
        "valid_pair_count": len(seen),
        "invalid_pair_count": invalid_pair_count,
        "max_pairs": limit,
        "selection_policy": "deterministic_round_robin_by_anchor_then_confidence",
    }


def hydrate_contain_candidate_scope(
    task: dict[str, Any],
    candidates: dict[str, Any],
    confidence_threshold: float = CONTAIN_SCOPE_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Use the exact transmitted contain pair set to hydrate the review scope."""
    manager_scope = [
        str(object_id)
        for object_id in task.get("object_scope", [])
        if isinstance(object_id, str) and object_id.strip()
    ]
    selection = select_contain_review_pairs(candidates)
    review_pairs = [
        {
            "object_anchor": pair["object_anchor"],
            "object_target": pair["object_target"],
            "confidence_hint": pair["confidence_hint"],
        }
        for pair in selection["selected_pairs"]
    ]
    dropped_review_pairs = [
        {
            "object_anchor": pair["object_anchor"],
            "object_target": pair["object_target"],
            "confidence_hint": pair["confidence_hint"],
        }
        for pair in selection["dropped_pairs"]
    ]
    candidate_scope: list[str] = []
    for pair in review_pairs:
        for object_id in (pair["object_anchor"], pair["object_target"]):
            if object_id not in candidate_scope:
                candidate_scope.append(object_id)

    high_confidence_pairs = [
        pair for pair in review_pairs
        if float(pair["confidence_hint"]) >= confidence_threshold
    ]
    dropped_high_confidence_pairs = [
        pair for pair in dropped_review_pairs
        if float(pair["confidence_hint"]) >= confidence_threshold
    ]
    effective_scope = list(dict.fromkeys([*manager_scope, *candidate_scope]))
    task["object_scope"] = effective_scope
    scope_audit = {
        "policy": "manager_priority_scope_union_exact_transmitted_local_contain_candidates",
        "selection_policy": selection["selection_policy"],
        "confidence_threshold": confidence_threshold,
        "max_transmitted_candidate_pairs": selection["max_pairs"],
        "manager_object_scope": manager_scope,
        "transmitted_candidate_pairs": review_pairs,
        "dropped_candidate_pairs": dropped_review_pairs,
        "high_confidence_candidate_pairs": high_confidence_pairs,
        "dropped_high_confidence_candidate_pairs": dropped_high_confidence_pairs,
        "candidate_object_ids": candidate_scope,
        "added_candidate_object_ids": [object_id for object_id in candidate_scope if object_id not in manager_scope],
        "effective_object_scope": effective_scope,
        "candidate_pairs_require_explicit_decision": True,
    }
    task["contain_scope_hydration"] = scope_audit
    append_task_instruction(
        task,
        "The Manager-selected object_scope and scene_focus are priorities, not a closed contain search set. Review every retained local recall pair, including pairs outside the Manager original scope, and return an explicit accept, reject, or uncertain decision for each one. The local selection score is not semantic evidence and is not exposed to the Agent.",
    )
    return scope_audit


HIGH_CONFIDENCE_EXCLUSIVE_MODES = {"ceiling_only", "wall_only"}
HIGH_CONFIDENCE_NON_WALL_MODES = {"ceiling_only", "support_only", "contain_candidate"}


def _object_anchor_routing_by_id(value: Any) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not item.get("object_id"):
            continue
        result[str(item["object_id"])] = {
            "object_id": str(item["object_id"]),
            "anchor_mode": str(item.get("anchor_mode") or "uncertain").lower(),
            "confidence": str(item.get("confidence") or "low").lower(),
        }
    return result


def _scene_state_object_ids(scene_state: dict[str, Any]) -> list[str]:
    candidates = scene_state.get("object_candidates", {})
    items = candidates.get("items", []) if isinstance(candidates, dict) else []
    return list(dict.fromkeys(
        str(item.get("object_id") or item.get("id"))
        for item in items
        if isinstance(item, dict) and (item.get("object_id") or item.get("id"))
    ))


def hydrate_task_scope_from_anchor_routing(
    task: dict[str, Any],
    object_ids: list[str],
    object_anchor_routing: list[dict[str, Any]] | None,
    available_anchor_modes: set[str] | None = None,
) -> dict[str, Any] | None:
    """Compile Manager semantic routes into non-exclusive task scopes."""
    if is_support_contact_task(task):
        return None
    routing = _object_anchor_routing_by_id(object_anchor_routing)
    if not routing:
        return None
    available_anchor_modes = available_anchor_modes or {
        "ceiling_only", "wall_only", "support_only", "contain_candidate",
    }
    manager_scope = [
        str(object_id)
        for object_id in task.get("manager_priority_object_scope", task.get("object_scope", []))
        if isinstance(object_id, str) and object_id.strip()
    ]
    task.setdefault("manager_priority_object_scope", list(manager_scope))
    capabilities = set(task_capabilities(task, "provides_capabilities"))
    kind = str(task.get("canonical_task_kind") or task.get("task_type") or "").lower()
    is_wall = any(capability.startswith("wall_") for capability in capabilities) or kind.startswith("wall_")
    is_ceiling = "ceiling_judgement_ready" in capabilities or kind == "ceiling_judgement"
    is_contain = "contain_relations_ready" in capabilities or kind == "contain_relation_extraction"
    if not (is_wall or is_ceiling or is_contain):
        return None

    routed: list[str] = []
    excluded: list[str] = []
    for object_id in object_ids:
        record = routing.get(object_id)
        if record is None:
            continue
        mode = record["anchor_mode"]
        confidence = record["confidence"]
        high = confidence == "high"
        if is_wall:
            if high and mode in HIGH_CONFIDENCE_NON_WALL_MODES and mode in available_anchor_modes:
                excluded.append(object_id)
                continue
            if mode in {"wall_only", "floor_wall", "uncertain"} or not high:
                routed.append(object_id)
        elif is_ceiling:
            if mode == "ceiling_only":
                routed.append(object_id)
        elif is_contain:
            if mode == "contain_candidate":
                routed.append(object_id)

    effective_scope = [
        object_id
        for object_id in dict.fromkeys([*manager_scope, *routed])
        if object_id not in excluded
    ]
    task["object_scope"] = effective_scope
    task["object_anchor_routing"] = [
        routing[object_id] for object_id in effective_scope if object_id in routing
    ]
    audit = {
        "policy": "manager_priority_scope_plus_nonexclusive_anchor_routing",
        "manager_priority_object_scope": manager_scope,
        "routed_object_ids": routed,
        "high_confidence_excluded_object_ids": excluded,
        "effective_object_scope": effective_scope,
    }
    task["anchor_routing_scope_hydration"] = audit
    return audit


def hydrate_support_capability_scope(
    task: dict[str, Any],
    object_ids: list[str] | None = None,
    object_anchor_routing: list[dict[str, Any]] | None = None,
    available_anchor_modes: set[str] | None = None,
) -> dict[str, Any]:
    """Give Support a complete gravity baseline while preserving Manager priorities."""
    manager_scope = [
        str(object_id)
        for object_id in task.get("manager_priority_object_scope", task.get("object_scope", []))
        if isinstance(object_id, str) and object_id.strip()
    ]
    task.setdefault("manager_priority_object_scope", list(manager_scope))
    all_object_ids = list(dict.fromkeys(object_ids or manager_scope))
    routing = _object_anchor_routing_by_id(object_anchor_routing)
    available_anchor_modes = available_anchor_modes or HIGH_CONFIDENCE_EXCLUSIVE_MODES
    excluded: list[str] = []
    gravity_scope: list[str] = []
    for object_id in all_object_ids:
        record = routing.get(object_id)
        if (
            record
            and record.get("confidence") == "high"
            and record.get("anchor_mode") in HIGH_CONFIDENCE_EXCLUSIVE_MODES
            and record.get("anchor_mode") in available_anchor_modes
        ):
            excluded.append(object_id)
            continue
        gravity_scope.append(object_id)

    task["object_scope"] = gravity_scope
    task["gravity_coverage_scope"] = gravity_scope
    task["object_anchor_routing"] = [
        routing[object_id] for object_id in all_object_ids if object_id in routing
    ]
    audit = {
        "policy": "complete_gravity_baseline_plus_manager_priority_scope",
        "manager_priority_object_scope": manager_scope,
        "gravity_coverage_scope": gravity_scope,
        "high_confidence_exclusive_object_ids": excluded,
        "support_anchor_requires_own_gravity_verdict": True,
        "every_gravity_scope_object_requires_explicit_verdict": True,
    }
    task["support_scope_hydration"] = audit
    append_task_instruction(
        task,
        "Gravity-direction support is the first relation pass. Return exactly one concise support_contact_candidates verdict for every object in gravity_coverage_scope; do not silently omit obvious floor-standing furniture. The Manager priority scope is not a closed support set and only identifies difficult cases; when an object supports another object, also decide its own gravity-direction support. Floor support may coexist with a separate wall attach for floor_wall objects.",
    )
    return audit


def task_control_text(task: dict[str, Any]) -> str:
    keys = [
        "task_id",
        "id",
        "task_type",
        "canonical_task_kind",
        "runner_action",
        "runtime_class",
        "runtime_policy",
        "execution_stage_hint",
        "instructions",
        "reason",
    ]
    return " ".join(str(task.get(key, "")) for key in keys).lower()


def is_support_contact_task(task: dict[str, Any]) -> bool:
    capabilities = task_capabilities(task, "provides_capabilities")
    if capabilities:
        return "support_relations_ready" in capabilities
    identity = " ".join(
        str(task.get(key) or "")
        for key in ("task_id", "id", "task_type", "canonical_task_kind")
    ).lower()
    return any(marker in identity for marker in ("support_contact", "support contact", "support_relation", "support relation"))



def _as_float_pair(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _point_in_polygon_xz(point: list[float], polygon: list[list[float]]) -> bool:
    x, z = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        intersects = (zi > z) != (zj > z)
        if intersects:
            x_at_z = (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi
            if x < x_at_z:
                inside = not inside
        j = i
    return inside


def _polygon_area_abs_xz(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(polygon):
        nxt = polygon[(idx + 1) % len(polygon)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) * 0.5


def _obb_polygon_from_wall_evidence(wall_evidence: dict[str, Any]) -> list[list[float]]:
    render_grid = wall_evidence.get("render_grid_wall_candidates") if isinstance(wall_evidence, dict) else None
    primary = render_grid.get("primary_candidate") if isinstance(render_grid, dict) else None
    segments = primary.get("wall_segments") if isinstance(primary, dict) else None
    polygon: list[list[float]] = []
    if isinstance(segments, list):
        for segment in segments[:4]:
            if not isinstance(segment, dict):
                continue
            point = _as_float_pair(segment.get("first_point"))
            if point:
                polygon.append(point)
    if len(polygon) == 4:
        return polygon
    candidate_walls = wall_evidence.get("candidate_walls") if isinstance(wall_evidence, dict) else None
    if isinstance(candidate_walls, list):
        polygon = []
        for wall in candidate_walls[:4]:
            if not isinstance(wall, dict):
                continue
            point = _as_float_pair(wall.get("first_point"))
            if point:
                polygon.append(point)
        if len(polygon) == 4:
            return polygon
    return []


def build_obb_wall_risk_audit(wall_evidence: dict[str, Any]) -> dict[str, Any]:
    """Route formal OBB wall review to light or heavy visual adjudication."""
    render_grid = wall_evidence.get("render_grid_wall_candidates", {}) if isinstance(wall_evidence, dict) else {}
    primary = render_grid.get("primary_candidate") if isinstance(render_grid, dict) else None
    method = None
    if isinstance(primary, dict):
        method = primary.get("method") or primary.get("source_method")
    policy_method = render_grid.get("method_policy", {}).get("primary_method") if isinstance(render_grid, dict) else None
    polygon = _obb_polygon_from_wall_evidence(wall_evidence if isinstance(wall_evidence, dict) else {})
    object_footprints = wall_evidence.get("object_footprints", []) if isinstance(wall_evidence, dict) else []
    risk_score = 0
    triggered: list[dict[str, Any]] = []

    def trigger(check: str, severity: int, reason: str, details: dict[str, Any] | None = None) -> None:
        nonlocal risk_score
        risk_score += severity
        triggered.append({"check": check, "severity": severity, "reason": reason, "details": details or {}})

    if not isinstance(primary, dict):
        trigger("primary_candidate_missing", 5, "No formal OBB primary candidate is available.")
    elif (method or policy_method) != "obb_from_mesh_footprint_min_rect":
        trigger(
            "unexpected_primary_method",
            4,
            "Formal wall flow expects obb_from_mesh_footprint_min_rect as the primary method.",
            {"method": method, "policy_primary_method": policy_method},
        )
    if len(polygon) != 4:
        trigger("invalid_polygon", 5, "Formal OBB candidate does not expose four valid wall corners.", {"corner_count": len(polygon)})

    inside_count = 0
    outside_objects: list[str] = []
    if len(polygon) == 4 and isinstance(object_footprints, list):
        for obj in object_footprints:
            if not isinstance(obj, dict):
                continue
            center = _as_float_pair(obj.get("center_xz"))
            if not center:
                continue
            if _point_in_polygon_xz(center, polygon):
                inside_count += 1
            else:
                outside_objects.append(str(obj.get("object_id")))
        total = inside_count + len(outside_objects)
        outside_ratio = (len(outside_objects) / total) if total else 0.0
        if total and outside_ratio > 0.15:
            trigger(
                "object_centers_outside_obb",
                4 if outside_ratio > 0.3 else 2,
                "Many object footprint centers fall outside the formal OBB candidate.",
                {"outside_ratio": round(outside_ratio, 4), "outside_objects": outside_objects[:12], "object_count": total},
            )

        xs: list[float] = []
        zs: list[float] = []
        for obj in object_footprints:
            if not isinstance(obj, dict):
                continue
            center = _as_float_pair(obj.get("center_xz"))
            if center:
                xs.append(center[0])
                zs.append(center[1])
        if xs and zs:
            object_extent_area = max((max(xs) - min(xs)) * (max(zs) - min(zs)), 1e-8)
            obb_area = _polygon_area_abs_xz(polygon)
            area_ratio = obb_area / object_extent_area
            if area_ratio > 8.0 or area_ratio < 0.6:
                trigger(
                    "obb_area_ratio_unusual",
                    2,
                    "OBB area is unusually large or small relative to object center spread.",
                    {"obb_area": round(obb_area, 6), "object_center_extent_area": round(object_extent_area, 6), "area_ratio": round(area_ratio, 3)},
                )

    segments = primary.get("wall_segments") if isinstance(primary, dict) else None
    lengths: list[float] = []
    if isinstance(segments, list):
        for segment in segments[:4]:
            if not isinstance(segment, dict):
                continue
            first = _as_float_pair(segment.get("first_point"))
            second = _as_float_pair(segment.get("second_point"))
            if first and second:
                lengths.append(((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5)
    if len(lengths) == 4 and min(lengths) > 1e-8:
        length_ratio = max(lengths) / min(lengths)
        if length_ratio > 8.0:
            trigger("extreme_wall_length_ratio", 3, "OBB wall length ratio is extreme.", {"length_ratio": round(length_ratio, 3), "lengths": [round(v, 6) for v in lengths]})

    near_contact_count = 0
    near_geometry = wall_evidence.get("render_grid_wall_attach_geometry") or wall_evidence.get("wall_attach_geometry") if isinstance(wall_evidence, dict) else None
    if isinstance(near_geometry, list):
        for item in near_geometry:
            if not isinstance(item, dict):
                continue
            try:
                if float(item.get("footprint_min_distance_to_wall")) <= 0.035:
                    near_contact_count += 1
            except (TypeError, ValueError):
                pass
    if isinstance(object_footprints, list) and object_footprints and near_contact_count == 0:
        trigger("no_near_wall_object_support", 2, "No object footprints are near any OBB wall; wall placement may be suspect.")

    orientation_disagreement = wall_orientation_disagreement_signal(wall_evidence)
    if orientation_disagreement.get("requires_agent_wall_alignment_objects"):
        trigger(
            "mesh_min_area_disagrees_with_stable_object_axis",
            3,
            "The minimum-area OBB direction differs materially from a dominant multi-object orientation consensus.",
            orientation_disagreement,
        )

    if risk_score >= 5:
        risk_level = "high"
        recommendation = "heavy_review"
    elif risk_score >= 3:
        risk_level = "medium"
        recommendation = "heavy_review"
    else:
        risk_level = "low"
        recommendation = "light_review"
    return {
        "schema_version": "obb_wall_risk_audit",
        "status": "available",
        "formal_method": "obb_from_mesh_footprint_min_rect",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "triggered_checks": triggered,
        "metrics": {
            "corner_count": len(polygon),
            "object_center_inside_count": inside_count,
            "object_center_outside_count": len(outside_objects),
            "near_wall_geometry_count": near_contact_count,
            "wall_lengths": [round(v, 6) for v in lengths],
            "orientation_disagreement": orientation_disagreement,
        },
        "policy": {
            "low_risk": "Use light wall visibility / weak-wall review.",
            "medium_or_high_risk": "Use heavy wall final adjudication with richer visual evidence.",
            "coordinates": "Final coordinates are still locally generated by obb_from_mesh_footprint_min_rect; this audit only routes review effort.",
        },
    }


def write_obb_wall_risk_audit(output_dir: Path, audit: dict[str, Any]) -> None:
    write_json(output_dir / "debug" / "walls" / "obb_wall_risk_audit.json", audit)
    lines = [
        f"- risk_level: `{audit.get('risk_level')}`",
        f"- risk_score: `{audit.get('risk_score')}`",
        f"- recommendation: `{audit.get('recommendation')}`",
        f"- formal_method: `{audit.get('formal_method')}`",
        "",
        "## Triggered Checks",
    ]
    for item in audit.get("triggered_checks", []):
        lines.append(f"- `{item.get('check')}` severity={item.get('severity')}: {item.get('reason')}")
    if not audit.get("triggered_checks"):
        lines.append("- none")
    write_md(output_dir / "debug" / "walls" / "obb_wall_risk_audit.md", "OBB Wall Risk Audit", lines)


def resolve_wall_review_mode(requested: str, audit: dict[str, Any] | None) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode in {"light", "heavy"}:
        return mode
    if isinstance(audit, dict) and audit.get("recommendation") == "light_review":
        return "light"
    return "heavy"


def local_wall_reasoning_signal(wall_evidence: dict[str, Any]) -> dict[str, Any]:
    """Return geometry-only evidence that justifies repairing a missing wall reasoning chain."""
    raw_geometry = wall_evidence.get("render_grid_wall_attach_geometry")
    if not isinstance(raw_geometry, list):
        raw_geometry = wall_evidence.get("wall_attach_geometry")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_geometry if isinstance(raw_geometry, list) else []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        wall_id = str(item.get("wall_id") or "")
        if not object_id or not wall_id:
            continue
        try:
            distance = float(item.get("footprint_min_distance_to_wall"))
        except (TypeError, ValueError):
            continue
        floor_supported = bool(item.get("is_floor_supported"))
        flat_analysis = item.get("flat_contact_edge_analysis")
        has_flat_edge = bool(item.get("has_valid_flat_contact_edge"))
        if isinstance(flat_analysis, dict):
            has_flat_edge = has_flat_edge or bool(flat_analysis.get("has_valid_flat_contact_edge"))
        thin_planar = bool(item.get("is_thin_planar_candidate"))
        close_to_wall = distance <= 0.035
        reason = None
        if close_to_wall and not floor_supported:
            reason = "non_floor_supported_object_near_formal_wall"
        elif close_to_wall and floor_supported and has_flat_edge:
            reason = "floor_supported_object_has_valid_flat_contact_edge"
        elif close_to_wall and thin_planar:
            reason = "thin_planar_object_near_formal_wall"
        if not reason:
            continue
        key = (object_id, wall_id)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "object_id": object_id,
            "wall_id": wall_id,
            "category": item.get("category"),
            "footprint_min_distance_to_wall": distance,
            "is_floor_supported": floor_supported,
            "is_thin_planar_candidate": thin_planar,
            "has_valid_flat_contact_edge": has_flat_edge,
            "reason": reason,
        })
    return {
        "status": "candidate_signal" if candidates else "no_candidate_signal",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "policy": "Geometry is a high-recall trigger only. Agent wall semantics and attach review still decide the relation.",
    }


def wall_orientation_disagreement_signal(wall_evidence: dict[str, Any]) -> dict[str, Any]:
    """Detect when minimum-area geometry and a stable object-axis consensus disagree."""
    render_grid = wall_evidence.get("render_grid_wall_candidates") if isinstance(wall_evidence, dict) else {}
    review = render_grid.get("wall_orientation_review") if isinstance(render_grid, dict) else {}
    candidates = render_grid.get("orientation_candidates") if isinstance(render_grid, dict) else []
    candidate_by_id = {
        str(item.get("candidate_id")): item
        for item in candidates if isinstance(item, dict) and item.get("candidate_id")
    }
    baseline_id = str(review.get("baseline_candidate_id") or "")
    baseline = candidate_by_id.get(baseline_id)
    baseline_orientation = baseline.get("orientation") if isinstance(baseline, dict) and isinstance(baseline.get("orientation"), dict) else {}
    baseline_angle = baseline_orientation.get("angle_degrees_xz")
    clusters = []
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        orientation = item.get("orientation") if isinstance(item.get("orientation"), dict) else {}
        if orientation.get("source") != "robust_object_rotation_consensus":
            continue
        clusters.append({
            "candidate_id": item.get("candidate_id"),
            "angle_degrees_xz": orientation.get("angle_degrees_xz"),
            "member_count": int(orientation.get("member_count") or 0),
            "total_weight": float(orientation.get("total_weight") or 0.0),
        })
    clusters.sort(key=lambda item: (-item["total_weight"], -item["member_count"], str(item["candidate_id"])))
    dominant = clusters[0] if clusters else None
    total_cluster_weight = sum(item["total_weight"] for item in clusters)
    dominance = (
        dominant["total_weight"] / total_cluster_weight
        if dominant and total_cluster_weight > 0 else 0.0
    )
    angle_difference = None
    if dominant and isinstance(baseline_angle, (int, float)) and isinstance(dominant.get("angle_degrees_xz"), (int, float)):
        angle_difference = axis_angle_distance_degrees(float(baseline_angle), float(dominant["angle_degrees_xz"]))
    requires_review = bool(
        len(candidate_by_id) > 1
        and dominant
        and dominant["member_count"] >= 2
        and dominance >= 0.55
        and angle_difference is not None
        and angle_difference >= 10.0
    )
    return {
        "status": "orientation_review_required" if requires_review else "no_strong_orientation_disagreement",
        "requires_agent_wall_alignment_objects": requires_review,
        "baseline_candidate_id": baseline_id or None,
        "baseline_angle_degrees_xz": baseline_angle,
        "dominant_object_axis_candidate_id": dominant.get("candidate_id") if dominant else None,
        "dominant_object_axis_angle_degrees_xz": dominant.get("angle_degrees_xz") if dominant else None,
        "dominant_member_count": dominant.get("member_count") if dominant else 0,
        "dominant_weight_share": round(dominance, 6),
        "axis_difference_degrees": round(angle_difference, 6) if angle_difference is not None else None,
        "policy": "A strong disagreement triggers semantic object selection; it never directly selects a wall candidate.",
    }


def configure_wall_alignment_task(task: dict[str, Any]) -> dict[str, Any]:
    """Convert a pre-orientation wall node into the compact alignment contract.

    A Manager may merge ``provisional_wall_visibility_ready`` into this node.
    That optional capability judges labelled edges in finite candidate panels;
    it remains distinct from final wall-id semantics and therefore stays valid
    when local code selects a different finite orientation after this task.
    """
    previous_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "")
    focus = str(task.get("scene_focus") or "").strip()
    scope = [str(item) for item in task.get("object_scope", []) if item]
    preserve_provisional = task_provides_capability(task, PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
    task["task_type"] = "wall_alignment_object_review"
    task["canonical_task_kind"] = "wall_alignment_object_review"
    task["runner_action"] = "api_dynamic_task"
    task["runtime_class"] = "required_reasoning"
    task["requires_artifacts"] = ["object_list", "original_image", "mask_overlay", "wall_orientation_candidates_overlay"]
    task["requires_capabilities"] = []
    task["provides_capabilities"] = ["wall_alignment_objects_ready", "wall_object_hypotheses_ready"]
    task["image_artifact_profile"] = ["original_image", "mask_overlay", "wall_orientation_candidates_overlay"]
    if preserve_provisional:
        task["provides_capabilities"].append(PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
    task["needs_tool"] = False
    task["use_vision_model"] = True
    instructions = [
        "Identify only objects whose real pose reliably reveals a room-wall axis: wall-mounted, wall-embedded, physically against a wall, or clearly parallel/aligned with a wall.",
        "Use the original image for semantics and the mask overlay for object ids. Treat object_scope as a priority, not a closed set.",
        "Exclude central furniture, round objects, small clutter, and uncertain headings. Return high or medium confidence only.",
        "Do not choose the winning room-orientation candidate, edit coordinates/angles, emit final wall ids, or emit attach relations. Local code scores the finite OBB axes from each selected object's canonical rotation_degrees_topdown.",
        "For each likely wall object, use the isolated labelled candidate panels to return candidate_wall_bindings for every panel where the corresponding real wall edge is identifiable. Copy candidate_id and provisional_wall_id exactly. These bindings are semantic review evidence, not final wall selection or attach acceptance.",
    ]
    expected_output_schema: dict[str, Any] = {
        "likely_wall_objects": [{
            "object_id": "object_###",
            "role": "wall_mounted|wall_embedded|against_wall|wall_aligned",
            "confidence": "high|medium",
            "evidence_code": "short_visual_evidence_code",
        }],
        "candidate_wall_bindings": [{
            "object_id": "object_###",
            "candidate_id": "copy exactly from candidate panel",
            "provisional_wall_id": "wall_001|wall_002|wall_003|wall_004",
            "role": "wall_mounted|wall_embedded|against_wall|wall_aligned",
            "confidence": "high|medium",
            "evidence_code": "short_visual_binding_code",
        }],
    }
    if preserve_provisional:
        instructions.extend([
            "The isolated wall-orientation candidate panels are labelled independently. For each panel edge, you may judge only whether it is a directly visible real wall versus inferred, weak, or closure-only geometry.",
            "Return candidate_id plus provisional_wall_id exactly as printed on the panel. Do not select the winning candidate or emit final wall ids/relations; local code binds records only for the selected candidate after orientation scoring.",
        ])
        expected_output_schema["provisional_wall_visibility_review"] = [{
            "candidate_id": "copy exactly from candidate panel",
            "provisional_wall_id": "wall_001|wall_002|wall_003|wall_004",
            "original_image_visibility": "visible|partially_visible|not_visible|inferred",
            "weak_wall": False,
            "closure_only": False,
            "can_anchor_wall_attach": True,
            "reason_code": "short_reason_code",
        }]
        task["provisional_wall_visibility_merged_with_alignment"] = True
    if focus:
        instructions.append(f"Manager scene focus: {focus}")
    if scope:
        instructions.append("Manager priority object scope: " + ", ".join(scope))
    task["instructions"] = instructions
    task["expected_output_schema"] = expected_output_schema
    task["wall_alignment_contract_normalized_by_runner"] = True
    task.setdefault("wall_alignment_previous_task_kind", previous_kind)
    return task


def configure_provisional_wall_visibility_task(task: dict[str, Any]) -> dict[str, Any]:
    """Normalize a standalone pre-orientation candidate-edge semantic node."""
    previous_kind = str(task.get("canonical_task_kind") or task.get("task_type") or "")
    focus = str(task.get("scene_focus") or "").strip()
    scope = [str(item) for item in task.get("object_scope", []) if item]
    task["task_type"] = "provisional_wall_visibility_review"
    task["canonical_task_kind"] = "provisional_wall_visibility_review"
    task["runner_action"] = "api_dynamic_task"
    task["runtime_class"] = "required_reasoning"
    task["requires_artifacts"] = ["original_image", "mask_overlay", "wall_orientation_candidates_overlay"]
    task["provides_capabilities"] = [PROVISIONAL_WALL_VISIBILITY_CAPABILITY]
    task["image_artifact_profile"] = ["original_image", "mask_overlay", "wall_orientation_candidates_overlay"]
    task["needs_tool"] = False
    task["use_vision_model"] = True
    instructions = [
        "Use the isolated finite wall-orientation candidate panels to judge directly visible real-wall edges versus inferred, weak, or closure-only room boundaries.",
        "Return compact candidate_id plus provisional_wall_id records only. Do not select an orientation candidate, create coordinates, alter edges, emit final wall ids, or emit attach relations.",
        "Mark visible or partially_visible only with direct source-image wall-surface or wall-junction evidence. If uncertain, use inferred/not_visible and can_anchor_wall_attach=false.",
    ]
    if focus:
        instructions.append(f"Manager scene focus: {focus}")
    if scope:
        instructions.append("Manager priority object scope: " + ", ".join(scope))
    task["instructions"] = instructions
    task["expected_output_schema"] = {
        "provisional_wall_visibility_review": [{
            "candidate_id": "copy exactly from candidate panel",
            "provisional_wall_id": "wall_001|wall_002|wall_003|wall_004",
            "original_image_visibility": "visible|partially_visible|not_visible|inferred",
            "weak_wall": False,
            "closure_only": False,
            "can_anchor_wall_attach": True,
            "reason_code": "short_reason_code",
        }]
    }
    task["provisional_wall_visibility_contract_normalized_by_runner"] = True
    task.setdefault("provisional_wall_visibility_previous_task_kind", previous_kind)
    return task

def ensure_wall_alignment_object_review_task(
    manager_plan: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    tasks = manager_plan.setdefault("tasks", [])
    for task in tasks:
        if isinstance(task, dict) and task_provides_capability(task, "wall_alignment_objects_ready"):
            configure_wall_alignment_task(task)
            return manager_plan
    task = {
        "task_id": "task_wall_alignment_object_review",
        "reason": reason,
        "object_scope": [],
        "scene_focus": "Identify only reliable wall-following objects for local finite-axis scoring.",
    }
    tasks.append(configure_wall_alignment_task(task))
    manager_plan["wall_alignment_task_injected_by_runner"] = True
    return manager_plan


def ensure_post_orientation_wall_visibility_task(
    manager_plan: dict[str, Any],
) -> dict[str, Any]:
    tasks = manager_plan.setdefault("tasks", [])
    if any(
        isinstance(task, dict)
        and task_provides_capability(task, "wall_visibility_semantics_ready")
        and not task_provides_capability(task, "wall_alignment_objects_ready")
        for task in tasks
    ):
        return manager_plan
    tasks.append({
        "task_id": "task_wall_visibility_semantics_review",
        "task_type": "wall_visibility_semantics_review",
        "canonical_task_kind": "wall_visibility_semantics_review",
        "agent_role": "DynamicTaskAgent",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["formal_obb_walls", "original_image", "mask_overlay"],
        "requires_capabilities": ["wall_alignment_objects_ready"],
        "provides_capabilities": ["wall_visibility_semantics_ready"],
        "image_artifact_profile": ["original_image", "mask_overlay"],
        "needs_tool": False,
        "use_vision_model": True,
        "instructions": [
            "The local scorer has already selected and applied the formal OBB orientation. Judge visibility, weak/inferred or closure-only status, and attach eligibility for all four final wall references.",
            "Copy wall_id and geometry_key exactly. Do not choose another candidate or edit coordinates, angles, ids, or methods.",
        ],
        "expected_output_schema": {
            "wall_visibility_review": [{
                "wall_id": "wall_001",
                "geometry_key": "copy supplied wall_edge_... exactly",
                "original_image_visibility": "visible|partially_visible|not_visible|inferred",
                "weak_wall": False,
                "closure_only": False,
                "can_anchor_wall_attach": True,
                "reason_code": "short_reason_code",
            }]
        },
    })
    manager_plan["post_orientation_visibility_task_injected_by_runner"] = True
    return manager_plan


def apply_wall_task_policy(
    manager_plan: dict[str, Any],
    wall_evidence: dict[str, Any],
    wall_candidate_mode: str,
    wall_review_mode: str = "heavy",
    obb_risk_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:

    tasks = [task for task in manager_plan.get("tasks", []) if isinstance(task, dict)]
    manager_plan["tasks"] = tasks
    local_signal = local_wall_reasoning_signal(wall_evidence)
    orientation_signal = wall_orientation_disagreement_signal(wall_evidence)
    repairs: list[dict[str, Any]] = []
    manager_wall_task_ids = [
        task_identifier(task)
        for task in tasks
        if any(capability.startswith("wall_") for capability in task_capabilities(task, "provides_capabilities"))
    ]
    visibility_requested_by_manager = any(
        task_provides_capability(task, "wall_visibility_semantics_ready")
        for task in tasks
    )
    provisional_visibility_requested_by_manager = any(
        task_provides_capability(task, PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
        for task in tasks
    )
    split_attach_scopes: list[str] = []

    # A pre-orientation node cannot also provide semantics keyed to final wall
    # ids. Split that incompatible merge while preserving Manager scope/focus.
    for task in tasks:
        caps = task_capabilities(task, "provides_capabilities")
        if (
            "wall_object_hypotheses_ready" in caps
            and "wall_alignment_objects_ready" not in caps
        ):
            caps.append("wall_alignment_objects_ready")
            task["provides_capabilities"] = caps
        if (
            task_provides_capability(task, "wall_alignment_objects_ready")
            and task_provides_capability(task, "wall_attach_decisions_ready")
        ):
            split_attach_scopes.extend(str(item) for item in task.get("object_scope", []) if item)
            configure_wall_alignment_task(task)
            repairs.append({
                "repair_reason": "pre_orientation_alignment_and_post_orientation_wall_attach_must_be_separate",
                "converted_task_id": task_identifier(task),
            })
        elif (
            task_provides_capability(task, "wall_alignment_objects_ready")
            and task_provides_capability(task, "wall_visibility_semantics_ready")
        ):
            configure_wall_alignment_task(task)
            repairs.append({
                "repair_reason": "pre_orientation_alignment_and_post_orientation_visibility_must_be_separate",
                "converted_task_id": task_identifier(task),
            })

    if split_attach_scopes:
        before = {task_identifier(task) for task in manager_plan.get("tasks", [])}
        manager_plan = ensure_wall_attach_recall_task(manager_plan, wall_evidence)
        for task in manager_plan.get("tasks", []):
            if task_identifier(task) not in before and is_wall_attach_recall_task(task):
                task["object_scope"] = sorted(set(split_attach_scopes))
        repairs.append({
            "repair_reason": "split_combined_alignment_and_wall_attach_node",
            "injected_task_ids": [
                task_identifier(task)
                for task in manager_plan.get("tasks", [])
                if task_identifier(task) not in before
            ],
        })

    def providers(capability: str) -> list[dict[str, Any]]:
        return [task for task in manager_plan.get("tasks", []) if task_provides_capability(task, capability)]

    # Candidate-edge visibility is pre-orientation evidence. It cannot be
    # produced by the downstream attach node because the local prefilter needs
    # it before that node is executed. When a compact Manager accidentally
    # groups both capabilities there, preserve its semantic intent by moving
    # only the provisional capability into an existing alignment node.
    misplaced_provisional_tasks = [
        task for task in providers(PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
        if task_provides_capability(task, "wall_attach_decisions_ready")
    ]
    if misplaced_provisional_tasks:
        for task in misplaced_provisional_tasks:
            task["provides_capabilities"] = [
                capability
                for capability in task_capabilities(task, "provides_capabilities")
                if capability != PROVISIONAL_WALL_VISIBILITY_CAPABILITY
            ]
        if not providers("wall_alignment_objects_ready"):
            before = {task_identifier(task) for task in manager_plan.get("tasks", [])}
            manager_plan = ensure_wall_alignment_object_review_task(
                manager_plan,
                reason="provisional_candidate_visibility_requires_preorientation_alignment_context",
            )
            repairs.append({
                "repair_reason": "moved_provisional_visibility_from_downstream_attach_to_alignment",
                "injected_task_ids": [
                    task_identifier(task)
                    for task in manager_plan.get("tasks", [])
                    if task_identifier(task) not in before
                ],
            })
        for alignment_task in providers("wall_alignment_objects_ready"):
            provided = task_capabilities(alignment_task, "provides_capabilities")
            if PROVISIONAL_WALL_VISIBILITY_CAPABILITY not in provided:
                provided.append(PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
                alignment_task["provides_capabilities"] = provided
            configure_wall_alignment_task(alignment_task)
        repairs.append({
            "repair_reason": "provisional_visibility_must_precede_wall_attach_prefilter",
            "moved_from_task_ids": [task_identifier(task) for task in misplaced_provisional_tasks],
            "target_task_ids": [task_identifier(task) for task in providers("wall_alignment_objects_ready")],
        })

    has_wall_intent = bool(manager_wall_task_ids)
    attach_requested = bool(providers("wall_attach_decisions_ready"))
    needs_attach_chain = attach_requested or bool(local_signal.get("candidate_count"))

    if bool(local_signal.get("candidate_count")) and not attach_requested:
        before = {task_identifier(task) for task in manager_plan.get("tasks", [])}
        manager_plan = ensure_wall_attach_recall_task(manager_plan, wall_evidence)
        added = [
            task_identifier(task)
            for task in manager_plan.get("tasks", [])
            if task_identifier(task) not in before
        ]
        repairs.append({
            "repair_reason": "local_geometry_candidates_require_wall_attach_decision",
            "injected_task_ids": added,
            "trigger": "local_geometry_candidate_signal",
        })
        attach_requested = bool(providers("wall_attach_decisions_ready"))
        needs_attach_chain = True

    if orientation_signal.get("requires_agent_wall_alignment_objects") and not attach_requested:
        before = {task_identifier(task) for task in manager_plan.get("tasks", [])}
        manager_plan = ensure_wall_attach_recall_task(manager_plan, wall_evidence)
        added = [
            task_identifier(task)
            for task in manager_plan.get("tasks", [])
            if task_identifier(task) not in before
        ]
        repairs.append({
            "repair_reason": "orientation_risk_review_requires_post_orientation_wall_attach_and_visibility_decision",
            "injected_task_ids": added,
            "trigger": orientation_signal.get("status"),
        })
        attach_requested = bool(providers("wall_attach_decisions_ready"))
        needs_attach_chain = True

    needs_alignment = bool(
        has_wall_intent
        or needs_attach_chain
        or bool(providers(PROVISIONAL_WALL_VISIBILITY_CAPABILITY))
        or orientation_signal.get("requires_agent_wall_alignment_objects")
    )
    if needs_alignment and not providers("wall_alignment_objects_ready"):
        before = {task_identifier(task) for task in manager_plan.get("tasks", [])}
        manager_plan = ensure_wall_alignment_object_review_task(
            manager_plan,
            reason=(
                "strong_local_orientation_disagreement"
                if orientation_signal.get("requires_agent_wall_alignment_objects")
                else "wall_reasoning_requires_semantic_alignment_objects"
            ),
        )
        added = [
            task_identifier(task)
            for task in manager_plan.get("tasks", [])
            if task_identifier(task) not in before
        ]
        repairs.append({
            "repair_reason": "formal_wall_orientation_requires_semantic_alignment_objects",
            "injected_task_ids": added,
            "trigger": orientation_signal.get("status"),
        })

    # In the normal path the downstream attach task also performs final wall
    # visibility review, avoiding a third wall API call.
    for task in providers("wall_attach_decisions_ready"):
        provided = task_capabilities(task, "provides_capabilities")
        if "wall_visibility_semantics_ready" not in provided:
            provided.append("wall_visibility_semantics_ready")
        task["provides_capabilities"] = provided
        required = task_capabilities(task, "requires_capabilities")
        for capability in ("wall_alignment_objects_ready", "wall_attach_candidates_ready"):
            if capability not in required:
                required.append(capability)
        if providers(PROVISIONAL_WALL_VISIBILITY_CAPABILITY) and PROVISIONAL_WALL_VISIBILITY_CAPABILITY not in required:
            required.append(PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
        task["requires_capabilities"] = required

    if visibility_requested_by_manager and not providers("wall_attach_decisions_ready") and not providers("wall_visibility_semantics_ready"):
        manager_plan = ensure_post_orientation_wall_visibility_task(manager_plan)
        repairs.append({
            "repair_reason": "manager_requested_visibility_moved_after_local_orientation_selection",
            "injected_task_ids": ["task_wall_visibility_semantics_review"],
        })

    for task in manager_plan.get("tasks", []):
        if not isinstance(task, dict):
            continue
        wall_caps = {
            capability
            for capability in task_capabilities(task, "provides_capabilities")
            if capability.startswith("wall_")
        }
        if not wall_caps:
            continue
        if "wall_alignment_objects_ready" in wall_caps:
            configure_wall_alignment_task(task)
            continue
        if PROVISIONAL_WALL_VISIBILITY_CAPABILITY in wall_caps:
            configure_provisional_wall_visibility_task(task)
            continue
        task.setdefault("formal_wall_references", formal_wall_references(wall_evidence))
        task.setdefault("formal_wall_geometry_source", "obb_from_mesh_footprint_min_rect")
        append_task_instruction(task, CANONICAL_WALL_NAMING_INSTRUCTION)
        append_task_instruction(
            task,
            "The final formal OBB orientation is selected by local scoring from Agent-identified wall-following objects before this task. Do not choose another candidate or edit wall coordinates, ids, angles, or methods.",
        )
        if "wall_visibility_semantics_ready" in wall_caps:
            task.setdefault("wall_review_mode", wall_review_mode)
            task.setdefault("obb_wall_risk_audit", obb_risk_audit or {})
            append_task_instruction(
                task,
                "Return visibility/weak/inferred/closure-only attach eligibility for all four current formal_wall_references, copying each geometry_key exactly.",
            )
        if "wall_attach_decisions_ready" in wall_caps:
            task["needs_tool"] = True
            append_task_instruction(
                task,
                "Return one explicit accept or reject verdict for every local wall_attach_prefilter candidate pair after reviewing final-wall visibility.",
            )

    manager_plan["wall_task_policy"] = "compact_manager_semantic_objects_then_local_orientation"
    manager_plan["manager_plan_repaired"] = bool(repairs)
    manager_plan["manager_plan_repairs"] = repairs
    manager_plan["formal_wall_task_control"] = {
        "mode": wall_candidate_mode,
        "manager_plan_format": "compact",
        "manager_selected_wall_task_ids": manager_wall_task_ids,
        "final_wall_task_ids": [
            task_identifier(task)
            for task in manager_plan.get("tasks", [])
            if any(capability.startswith("wall_") for capability in task_capabilities(task, "provides_capabilities"))
        ],
        "manager_had_wall_intent": has_wall_intent,
        "manager_requested_provisional_wall_visibility": provisional_visibility_requested_by_manager,
        "provisional_wall_visibility_provider_task_ids": [
            task_identifier(task) for task in providers(PROVISIONAL_WALL_VISIBILITY_CAPABILITY)
        ],
        "local_wall_reasoning_signal": local_signal,
        "local_wall_orientation_disagreement_signal": orientation_signal,
        "repairs": repairs,
        "note": "Manager selects wall capabilities and grouping. When requested, provisional candidate-edge visibility is bound locally only after finite orientation selection; the runner enforces data dependencies, not a fixed wall-task roster.",
        "wall_review_mode": wall_review_mode,
        "obb_wall_risk_level": obb_risk_audit.get("risk_level") if isinstance(obb_risk_audit, dict) else None,
        "obb_wall_risk_recommendation": obb_risk_audit.get("recommendation") if isinstance(obb_risk_audit, dict) else None,
    }
    return manager_plan



def ensure_wall_attach_recall_task(manager_plan: dict[str, Any], wall_evidence: dict[str, Any]) -> dict[str, Any]:
    tasks = manager_plan.setdefault("tasks", [])
    has_recall_task = any(is_wall_attach_recall_task(task) for task in tasks if isinstance(task, dict))
    if has_recall_task:
        for task in tasks:
            if isinstance(task, dict) and is_wall_attach_recall_task(task):
                required = task_capabilities(task, "requires_capabilities")
                for capability in ("wall_alignment_objects_ready", "wall_attach_candidates_ready"):
                    if capability not in required:
                        required.append(capability)
                task["requires_capabilities"] = required
                provided = task_capabilities(task, "provides_capabilities")
                for capability in ("wall_visibility_semantics_ready", "wall_attach_decisions_ready"):
                    if capability not in provided:
                        provided.append(capability)
                task["provides_capabilities"] = provided
        return manager_plan
    tasks.append(
        {
            "task_id": "task_wall_attach_candidate_recall",
            "agent_role": "DynamicTaskAgent",
            "task_type": "wall_attach_candidate_recall",
            "canonical_task_kind": "wall_attach_candidate_recall",
            "runner_action": "api_dynamic_task",
            "runtime_class": "required_reasoning",
            "requires_capabilities": ["wall_alignment_objects_ready", "wall_attach_candidates_ready"],
            "provides_capabilities": ["wall_visibility_semantics_ready", "wall_attach_decisions_ready"],
            "needs_tool": True,
            "use_vision_model": True,
            "candidate_categories_include_but_not_limited_to": [
                "picture",
                "painting",
                "glass_door",
                "window",
                "whiteboard",
                "wall_shelf",
                "wall_panel",
                "mounted_screen",
                "wall_lamp",
            ],
            "requires_artifacts": [
                "object_list",
                "object_footprints",
                "formal_obb_walls",
                "canonical_topdown",
                "canonical_grid_mapping",
                "wall_evidence",
                "original_image",
                "mask_overlay",
            ],
            "evidence_needed": [
                "original image and mask overlay paths from scene_state",
                "object masks and object footprints from wall_evidence.object_footprints",
                "wall_evidence.wall_attach_geometry footprint-to-wall metrics",
                "wall_evidence.render_grid_wall_candidates and render_grid_wall_attach_geometry from obb_from_mesh_footprint_min_rect",
                "canonical topdown image and grid mapping",
                "final wall coordinates will be locally overridden from obb_from_mesh_footprint_min_rect; candidate_walls are formal OBB geometry evidence",
                "selective topdown render if wall id or contact is ambiguous",
            ],
            "instructions": (
                "The local scorer has already selected and applied the final formal OBB orientation. First return visibility/weak/inferred/closure-only semantics for all four supplied formal_wall_references, copying wall_id and geometry_key exactly. "
                "Recall and judge possible wall-mounted, wall-embedded, or verified against-wall horizontal contact attach relations. Do not rely on category or description alone. "
                "Do not reject using object center distance alone: thick objects can touch a wall with their back/side while their center is away from it. "
                "Floor-supported or object-supported objects may also be wall-attached; gravity-direction support and horizontal wall contact are different physical relations. A floor/object support relation must not replace a second wall attach when the object back/side is visibly or geometrically in contact with a verified non-weak wall. "
                "For each candidate, combine image/overlay/mask evidence, canonical topdown, "
                "object footprint/bbox side contact, wall_attach_geometry footprint_min_distance_to_wall, render_grid_wall_attach_geometry from the formal OBB wall candidate, and final canonical wall coordinates. "
                "Generated wall candidates are provisional geometry evidence; their candidate/provisional ids may be renamed by wall_final_adjudication and the local canonicalization gate. Do not choose between wall extraction algorithms. "
                "Return wall_attach_candidates with decision accept or reject, object_target, candidate_wall_id using the final canonical wall id after wall_final_adjudication when available, wall_contact_type, visual_evidence, "
                "geometry_evidence, description_used_as, reason, and selective_topdown_used. Reject decisions must cite concrete visual or footprint evidence. "
                "Accepted wall attaches must use wall_001..wall_004 only; never unknown_wall or unresolved_anchor. "
                "Do not accept attach to a weak, hidden, or inferred-by-room-closure wall when the original image cannot show the actual wall surface/contact; such walls may close the room geometry but cannot anchor attach relations."
                " Consume the local wall_attach_prefilter candidate pairs as the primary review set. Do not exhaustively rescan every object-wall combination. "
                "The prefilter has already applied deterministic lower-mesh flat-edge checks for ordinary floor-supported objects. It may deliberately retain candidates on walls whose final visibility is not known yet. "
                "You may add a missing relation only when the original image directly shows an obvious wall-mounted or wall-embedded object omitted by the local candidates; explain that supplement explicitly."
            ),
            "wall_evidence_summary": {
                "canonical_topdown_image_path": wall_evidence.get("topdown", {}).get("canonical_topdown_image_path"),
                "canonical_grid_mapping_path": wall_evidence.get("topdown", {}).get("canonical_grid_mapping_path"),
                "candidate_walls_count": len(wall_evidence.get("candidate_walls", [])),
                "candidate_walls_trust_level": wall_evidence.get("candidate_room_polygon_hint", {}).get("trust_level"),
                "object_footprints_count": len(wall_evidence.get("object_footprints", [])),
                "wall_attach_geometry_count": len(wall_evidence.get("wall_attach_geometry", [])),
                "render_grid_wall_status": wall_evidence.get("render_grid_wall_candidates", {}).get("status"),
                "render_grid_wall_primary_method": wall_evidence.get("render_grid_wall_candidates", {}).get("method_policy", {}).get("primary_method"),
                "render_grid_wall_attach_geometry_count": len(wall_evidence.get("render_grid_wall_attach_geometry", [])),
            },
            "output_schema": {
                "wall_visibility_review": [
                    {
                        "wall_id": "wall_001",
                        "geometry_key": "copy supplied wall_edge_... exactly",
                        "original_image_visibility": "visible|partially_visible|not_visible|inferred",
                        "weak_wall": False,
                        "closure_only": False,
                        "can_anchor_wall_attach": True,
                        "reason_code": "short_reason_code",
                    }
                ],
                "wall_attach_candidates": [
                    {
                        "decision": "accept|reject",
                        "object_target": "object_###",
                        "candidate_wall_id": "wall_001|wall_002|wall_003|wall_004",
                        "evidence": ["..."],
                        "reason": "...",
                        "selective_topdown_used": False,
                    }
                ],
                "accepted_attach_candidates": [
                    {"relation": "attach", "parameters": {"object_anchor": "wall_001", "object_target": "object_###"}}
                ],
            },
        }
    )
    manager_plan["wall_attach_recall_task_injected_by_runner"] = True
    return manager_plan
