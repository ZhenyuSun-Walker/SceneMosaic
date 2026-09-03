from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from relations.agents.assembly.object_properties import _walk_with_path
from relations.agents.assembly.relation_candidates import collect_accepted_attach_candidates
from relations.agents.state import sorted_object_id_key

def _target_coverage(object_ids: list[str], relations: list[dict[str, Any]]) -> dict[str, Any]:
    covered = {
        rel.get("parameters", {}).get("object_target")
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") in {"attach", "contain"}
        and isinstance(rel.get("parameters"), dict)
    }
    missing = sorted(set(object_ids) - covered, key=sorted_object_id_key)
    return {
        "object_ids": object_ids,
        "covered_object_targets": sorted([target for target in covered if target in object_ids], key=sorted_object_id_key),
        "missing_object_targets": missing,
        "complete": not missing,
    }



def _contain_geometry_index(scene_state: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], str | None, str | None]:
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(scene_state, dict):
        candidates = scene_state.get("object_candidates", {})
        for item in candidates.get("items", []) if isinstance(candidates, dict) else []:
            if not (isinstance(item, dict) and item.get("object_id")):
                continue
            by_id[str(item["object_id"])] = {
                "object_id": str(item["object_id"]),
                "bbox_global": item.get("bbox_3d") if isinstance(item.get("bbox_3d"), dict) else None,
                "orientation_for_agent": (
                    item.get("layout_reference", {}).get("orientation_for_agent")
                    if isinstance(item.get("layout_reference"), dict)
                    else None
                ),
            }
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state, dict) else {}
    relation_precompute = evidence.get("relation_precompute", {}) if isinstance(evidence, dict) else {}
    outputs = relation_precompute.get("outputs", {}) if isinstance(relation_precompute, dict) else {}
    raw_path = outputs.get("object_footprints_path") if isinstance(outputs, dict) else None
    if not raw_path:
        return by_id, None, "object_footprints_path unavailable"
    path = Path(str(raw_path))
    if not path.exists():
        return by_id, str(path), "object_footprints_path does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return by_id, str(path), str(exc)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    for item in items:
        if not (isinstance(item, dict) and item.get("object_id")):
            continue
        object_id = str(item["object_id"])
        merged = dict(by_id.get(object_id, {"object_id": object_id}))
        for key in [
            "bbox_global",
            "center_xz",
            "extent_xz",
            "bottom_y",
            "top_y",
            "height_y",
            "mesh_floor_footprint_hull_xz",
            "footprint_polygon_xz",
            "topdown_grid_stats",
        ]:
            if item.get(key) is not None:
                merged[key] = item.get(key)
        by_id[object_id] = merged
    return by_id, str(path), None


def _contain_point_xz(record: dict[str, Any]) -> tuple[float, float] | None:
    center_xz = record.get("center_xz")
    if isinstance(center_xz, list) and len(center_xz) >= 2:
        try:
            return float(center_xz[0]), float(center_xz[1])
        except (TypeError, ValueError):
            pass
    bbox = record.get("bbox_global")
    center = bbox.get("center") if isinstance(bbox, dict) else None
    if isinstance(center, list) and len(center) >= 3:
        try:
            return float(center[0]), float(center[2])
        except (TypeError, ValueError):
            return None
    return None


def _contain_center_y(record: dict[str, Any]) -> float | None:
    bbox = record.get("bbox_global")
    center = bbox.get("center") if isinstance(bbox, dict) else None
    if isinstance(center, list) and len(center) >= 3:
        try:
            return float(center[1])
        except (TypeError, ValueError):
            pass
    bottom = record.get("bottom_y")
    top = record.get("top_y")
    try:
        if bottom is not None and top is not None:
            return (float(bottom) + float(top)) * 0.5
    except (TypeError, ValueError):
        pass
    return None


def _contain_y_bounds(record: dict[str, Any]) -> tuple[float, float] | None:
    bbox = record.get("bbox_global")
    minimum = bbox.get("min") if isinstance(bbox, dict) else None
    maximum = bbox.get("max") if isinstance(bbox, dict) else None
    if isinstance(minimum, list) and isinstance(maximum, list) and len(minimum) >= 3 and len(maximum) >= 3:
        try:
            return min(float(minimum[1]), float(maximum[1])), max(float(minimum[1]), float(maximum[1]))
        except (TypeError, ValueError):
            pass
    try:
        bottom = float(record["bottom_y"])
        top = float(record["top_y"])
        return min(bottom, top), max(bottom, top)
    except (KeyError, TypeError, ValueError):
        return None


def _contain_hull_xz(record: dict[str, Any]) -> list[tuple[float, float]]:
    for key in ["mesh_floor_footprint_hull_xz", "footprint_polygon_xz"]:
        raw = record.get(key)
        if not isinstance(raw, list):
            continue
        points: list[tuple[float, float]] = []
        for point in raw:
            if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if len(points) >= 3:
            return points
    return []


def _contain_rotate_xz(point: tuple[float, float], angle: float) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, z = point
    return x * cosine + z * sine, -x * sine + z * cosine


def _contain_min_area_angle(points: list[tuple[float, float]]) -> float:
    angles = {
        round(math.atan2(second[1] - first[1], second[0] - first[0]) % (math.pi / 2.0), 12)
        for first, second in zip(points, points[1:] + points[:1])
        if math.dist(first, second) > 1e-9
    }
    if not angles:
        return 0.0
    best_angle = 0.0
    best_area = float("inf")
    for angle in sorted(angles):
        rotated = [_contain_rotate_xz(point, angle) for point in points]
        area = (
            (max(point[0] for point in rotated) - min(point[0] for point in rotated))
            * (max(point[1] for point in rotated) - min(point[1] for point in rotated))
        )
        if area < best_area:
            best_area = area
            best_angle = angle
    return best_angle


def _axis_outside_distance(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum - value
    if value > maximum:
        return value - maximum
    return 0.0


def _contain_geometry_decision(
    anchor: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(anchor, dict) or not isinstance(target, dict):
        return {
            "decision": "uncertain",
            "reason": "anchor or target geometry is unavailable",
            "geometry_source": "unavailable",
        }
    target_xz = _contain_point_xz(target)
    if target_xz is None:
        return {
            "decision": "uncertain",
            "reason": "target center_xz is unavailable",
            "geometry_source": "unavailable",
        }

    horizontal_decision = "uncertain"
    horizontal_metrics: dict[str, Any] = {}
    geometry_source = "bbox_global_aabb"
    hull = _contain_hull_xz(anchor)
    if hull:
        angle = _contain_min_area_angle(hull)
        rotated = [_contain_rotate_xz(point, angle) for point in hull]
        target_uv = _contain_rotate_xz(target_xz, angle)
        min_u = min(point[0] for point in rotated)
        max_u = max(point[0] for point in rotated)
        min_v = min(point[1] for point in rotated)
        max_v = max(point[1] for point in rotated)
        extent_u = max(max_u - min_u, 1e-6)
        extent_v = max(max_v - min_v, 1e-6)
        outside_u = _axis_outside_distance(target_uv[0], min_u, max_u)
        outside_v = _axis_outside_distance(target_uv[1], min_v, max_v)
        relaxed_u = max(extent_u * 0.08, 0.015)
        relaxed_v = max(extent_v * 0.08, 0.015)
        clear_u = max(extent_u * 0.18, 0.03)
        clear_v = max(extent_v * 0.18, 0.03)
        if outside_u > clear_u or outside_v > clear_v:
            horizontal_decision = "reject"
        elif outside_u <= relaxed_u and outside_v <= relaxed_v:
            horizontal_decision = "pass"
        geometry_source = "mesh_footprint_min_area_obb"
        horizontal_metrics = {
            "target_center_xz": [round(target_xz[0], 6), round(target_xz[1], 6)],
            "target_center_local_uv": [round(target_uv[0], 6), round(target_uv[1], 6)],
            "anchor_local_bounds_uv": [round(min_u, 6), round(min_v, 6), round(max_u, 6), round(max_v, 6)],
            "outside_distance_uv": [round(outside_u, 6), round(outside_v, 6)],
            "relaxed_margin_uv": [round(relaxed_u, 6), round(relaxed_v, 6)],
            "clear_reject_margin_uv": [round(clear_u, 6), round(clear_v, 6)],
            "obb_angle_degrees": round(math.degrees(angle), 6),
        }
    else:
        bbox = anchor.get("bbox_global")
        minimum = bbox.get("min") if isinstance(bbox, dict) else None
        maximum = bbox.get("max") if isinstance(bbox, dict) else None
        if isinstance(minimum, list) and isinstance(maximum, list) and len(minimum) >= 3 and len(maximum) >= 3:
            try:
                min_x, max_x = sorted([float(minimum[0]), float(maximum[0])])
                min_z, max_z = sorted([float(minimum[2]), float(maximum[2])])
                extent_x = max(max_x - min_x, 1e-6)
                extent_z = max(max_z - min_z, 1e-6)
                outside_x = _axis_outside_distance(target_xz[0], min_x, max_x)
                outside_z = _axis_outside_distance(target_xz[1], min_z, max_z)
                clear_x = max(extent_x * 0.18, 0.03)
                clear_z = max(extent_z * 0.18, 0.03)
                if outside_x > clear_x or outside_z > clear_z:
                    horizontal_decision = "reject"
                horizontal_metrics = {
                    "target_center_xz": [round(target_xz[0], 6), round(target_xz[1], 6)],
                    "anchor_aabb_xz": [round(min_x, 6), round(min_z, 6), round(max_x, 6), round(max_z, 6)],
                    "outside_distance_xz": [round(outside_x, 6), round(outside_z, 6)],
                    "clear_reject_margin_xz": [round(clear_x, 6), round(clear_z, 6)],
                }
            except (TypeError, ValueError):
                pass

    vertical_decision = "uncertain"
    vertical_metrics: dict[str, Any] = {}
    y_bounds = _contain_y_bounds(anchor)
    target_y = _contain_center_y(target)
    if y_bounds is not None and target_y is not None:
        min_y, max_y = y_bounds
        extent_y = max(max_y - min_y, 1e-6)
        outside_y = _axis_outside_distance(target_y, min_y, max_y)
        relaxed_y = max(extent_y * 0.08, 0.02)
        clear_y = max(extent_y * 0.15, 0.04)
        if outside_y > clear_y:
            vertical_decision = "reject"
        elif outside_y <= relaxed_y:
            vertical_decision = "pass"
        vertical_metrics = {
            "target_center_y": round(target_y, 6),
            "anchor_bounds_y": [round(min_y, 6), round(max_y, 6)],
            "outside_distance_y": round(outside_y, 6),
            "relaxed_margin_y": round(relaxed_y, 6),
            "clear_reject_margin_y": round(clear_y, 6),
        }

    if horizontal_decision == "reject" or vertical_decision == "reject":
        decision = "reject"
        reason = "target center is clearly outside the container OBB or vertical range"
    elif horizontal_decision == "pass" and vertical_decision in {"pass", "uncertain"}:
        decision = "pass"
        reason = "target center is consistent with the container OBB and available vertical range"
    else:
        decision = "uncertain"
        reason = "geometry is incomplete or close to the container boundary; preserve Agent judgement"
    return {
        "decision": decision,
        "reason": reason,
        "geometry_source": geometry_source,
        "horizontal_decision": horizontal_decision,
        "vertical_decision": vertical_decision,
        "metrics": {**horizontal_metrics, **vertical_metrics},
    }


def filter_contain_relations_by_geometry(
    relations: list[dict[str, Any]],
    scene_state: dict[str, Any] | None,
    trusted_agent_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    geometry_by_id, footprint_path, read_error = _contain_geometry_index(scene_state)
    trusted_agent_pairs = trusted_agent_pairs or set()
    kept: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for rel in relations:
        if not (isinstance(rel, dict) and rel.get("relation") == "contain" and isinstance(rel.get("parameters"), dict)):
            kept.append(rel)
            continue
        anchor = str(rel["parameters"].get("object_anchor") or "")
        target = str(rel["parameters"].get("object_target") or "")
        result = _contain_geometry_decision(geometry_by_id.get(anchor), geometry_by_id.get(target))
        rejected = result.get("decision") == "reject"
        if rejected and (anchor, target) in trusted_agent_pairs:
            result = {
                **result,
                "local_geometry_decision": "reject",
                "decision": "uncertain",
                "reason": (
                    "Explicit original-image semantic evidence places the target on an internal multi-level tier; "
                    "preserve it when SAM3D containment geometry disagrees."
                ),
                "trusted_agent_override": True,
            }
            rejected = False
        item = {
            "relation": rel,
            "object_anchor": anchor,
            "object_target": target,
            **result,
            "final_action": "remove" if rejected else "keep",
        }
        items.append(item)
        if not rejected:
            kept.append(rel)
    counts = {
        decision: len([item for item in items if item.get("decision") == decision])
        for decision in ["pass", "uncertain", "reject"]
    }
    audit = {
        "schema_version": "contain_geometry_audit",
        "status": "completed",
        "object_footprints_path": footprint_path,
        "object_footprints_read_error": read_error,
        "counts": counts,
        "items": items,
        "policy": (
            "Agent remains responsible for open-world contain discovery. Local geometry removes only clear contradictions: "
            "a target center materially outside the anchor mesh-footprint OBB or vertical range. Boundary or missing-data "
            "cases remain uncertain. An explicit accepted semantic verdict for an internal multi-level tier may "
            "override a contradictory SAM3D footprint because reconstruction placement is advisory rather than image truth."
        ),
    }
    return kept, audit


# Local contain candidates remain high-recall review selectors, not semantic
# evidence. Conflicting attach/contain verdicts, including a strong local
# selector that disagrees with Agent semantics, are escalated to the existing
# DeliverySchemaGuard; normal scenes add no task or API call.
CONTAIN_CONFLICT_MIN_LOCAL_CONFIDENCE = 0.90
CONTAIN_INTERNAL_SURFACE_ROLE = "internal_shelf"
CONTAIN_EXTERNAL_SURFACE_ROLES = {"outer_top", "external_surface"}


def _normalized_contain_surface_role(
    resolution: dict[str, Any],
    anchor_category: Any = None,
) -> tuple[str, str]:
    """Normalize the Guard's structured role, with a narrow text-reason fallback."""
    raw_role = resolution.get("support_surface_role") or resolution.get("surface_role")
    if raw_role is not None:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(raw_role).strip().lower()).strip("_")
        normalized_anchor = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(anchor_category or "").strip().lower(),
        ).strip("_")
        reason = " ".join(
            str(resolution.get(key) or "")
            for key in ("reason_code", "reason", "explanation")
        ).strip().lower().replace("_", " ")
        singular_exposed_shelf = normalized_anchor in {
            "shelf",
            "shelf_board",
            "wall_shelf",
            "floating_shelf",
            "single_shelf",
        }
        bench_top = (
            normalized_anchor.rsplit("_", 1)[-1] in {"bench", "workbench"}
            and re.search(r"\b(?:top|topmost|uppermost)(?: level| surface)?\b", reason)
        )
        if normalized in {"multi_level_tier", "multilevel_tier"} and (
            singular_exposed_shelf or bench_top
        ):
            return "external_surface", "structured_role_corrected_by_anchor_category"
        aliases = {
            "internal": CONTAIN_INTERNAL_SURFACE_ROLE,
            "inside": CONTAIN_INTERNAL_SURFACE_ROLE,
            "inside_shelf": CONTAIN_INTERNAL_SURFACE_ROLE,
            "internal_shelf_board": CONTAIN_INTERNAL_SURFACE_ROLE,
            "bounded_tier": CONTAIN_INTERNAL_SURFACE_ROLE,
            "multi_level_tier": CONTAIN_INTERNAL_SURFACE_ROLE,
            "multilevel_tier": CONTAIN_INTERNAL_SURFACE_ROLE,
            "bounded_tray": CONTAIN_INTERNAL_SURFACE_ROLE,
            "rail_enclosed_deck": CONTAIN_INTERNAL_SURFACE_ROLE,
            "framed_bunk": CONTAIN_INTERNAL_SURFACE_ROLE,
            "cubby": CONTAIN_INTERNAL_SURFACE_ROLE,
            "outermost_top": "outer_top",
            "outer_top_surface": "outer_top",
            "top": "outer_top",
            "sleep_surface": "external_surface",
            "sleeping_surface": "external_surface",
            "bed_surface": "external_surface",
            "work_surface": "external_surface",
            "exposed_single_shelf": "external_surface",
            "single_shelf_top": "external_surface",
            "floating_shelf_top": "external_surface",
            "exterior": "external_surface",
            "exterior_surface": "external_surface",
            "unknown": "unknown",
            "uncertain": "unknown",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {CONTAIN_INTERNAL_SURFACE_ROLE, *CONTAIN_EXTERNAL_SURFACE_ROLES, "unknown"}:
            return normalized, "structured_support_surface_role"
        return "unknown", "invalid_structured_support_surface_role"

    reason = " ".join(
        str(resolution.get(key) or "")
        for key in ("reason_code", "reason", "explanation")
    ).strip().lower().replace("_", "-")
    reason = re.sub(r"[-\s]+", " ", reason)
    if re.search(r"\b(?:outermost|outer|exterior) top\b", reason) or re.search(
        r"\b(?:top of (?:the )?(?:cabinet|fridge|refrigerator|bookshelf|bookcase|shelving unit)|(?:cabinet|fridge|refrigerator|bookshelf|bookcase|shelving unit) top)\b",
        reason,
    ):
        return "outer_top", "text_reason_fallback"
    if re.search(
        r"\b(?:sleep(?:ing)?|bed|bench|work) surface\b|\b(?:exposed|single|floating) shelf (?:top|surface)\b",
        reason,
    ):
        return "external_surface", "text_reason_fallback"
    if re.search(r"\b(?:internal|inside|lower|middle|upper)(?: open)? shelf(?: board| surface| bay)?\b", reason) or any(
        marker in reason
        for marker in ("display cubby", "shelf bay", "shelf compartment", "cabinet compartment")
    ):
        return CONTAIN_INTERNAL_SURFACE_ROLE, "text_reason_fallback"
    if any(marker in reason for marker in ("external surface", "exterior surface", "outside the container")):
        return "external_surface", "text_reason_fallback"
    return "unknown", "surface_role_missing"


def _contain_conflict_id(anchor: str, target: str) -> str:
    return f"contain_conflict:{anchor}:{target}"


def _contain_candidate_pairs_from_scene_state(
    scene_state: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state, dict) else {}
    precompute = evidence.get("relation_precompute", {}) if isinstance(evidence, dict) else {}
    outputs = precompute.get("outputs", {}) if isinstance(precompute, dict) else {}
    raw_path = outputs.get("contain_candidates_path") if isinstance(outputs, dict) else None
    if not raw_path and isinstance(precompute, dict) and precompute.get("precompute_dir"):
        raw_path = str(Path(str(precompute["precompute_dir"])) / "contain_candidates.json")
    if not raw_path:
        return [], None, "contain_candidates_path unavailable"
    path = Path(str(raw_path))
    if not path.exists():
        return [], str(path), "contain_candidates_path does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], str(path), str(exc)
    pairs = payload.get("candidate_pairs", []) if isinstance(payload, dict) else []
    return [item for item in pairs if isinstance(item, dict)], str(path), None


def _positive_containment_hypothesis(candidate: dict[str, Any]) -> dict[str, Any] | None:
    decision = str(candidate.get("decision") or "").lower()
    mode = str(candidate.get("support_mode") or "").lower()
    anchor, target = candidate.get("object_anchor"), candidate.get("object_target")
    hypothesis = candidate.get("containment_hypothesis")
    hypothesis_decision = ""
    if isinstance(hypothesis, dict):
        hypothesis_decision = str(hypothesis.get("decision") or hypothesis.get("status") or "").lower()
        anchor = hypothesis.get("object_anchor") or hypothesis.get("anchor") or anchor
        target = hypothesis.get("object_target") or hypothesis.get("target") or target
    positive_decisions = {"accept", "accepted", "true", "yes", "positive"}
    positive_hypotheses = {"inside", "accept", "accepted", "true", "yes", "positive"}
    accepted_positive = decision in positive_decisions and (
        mode in {"contain", "inside", "container"} or hypothesis_decision in positive_hypotheses
    )
    deferred_inside = (
        decision in {"defer", "deferred"}
        and mode == "defer_to_contain"
        and hypothesis_decision in positive_hypotheses
    )
    if not (accepted_positive or deferred_inside) or not (
        isinstance(anchor, str) and isinstance(target, str)
    ):
        return None
    return {
        "object_anchor": anchor,
        "object_target": target,
        "reason_code": candidate.get("reason_code") or candidate.get("reason"),
        "support_mode": mode,
    }


def _contain_rejections_from_task_results(
    task_results: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    rejected: dict[tuple[str, str], dict[str, Any]] = {}
    for root in task_results:
        if not isinstance(root, dict):
            continue
        kind = str(root.get("canonical_task_kind") or root.get("task_type") or "").lower()
        capabilities = root.get("provided_capabilities", [])
        if "contain" not in kind and not (isinstance(capabilities, list) and "contain_relations_ready" in capabilities):
            continue
        for path, node in _walk_with_path(root, str(root.get("task_id") or "contain_task")):
            values = node.get("contain_candidates") if isinstance(node, dict) else None
            if not isinstance(values, list):
                continue
            for candidate in values:
                if not isinstance(candidate, dict) or str(candidate.get("decision") or "").lower() not in {"reject", "rejected", "false", "no"}:
                    continue
                anchor, target = candidate.get("object_anchor"), candidate.get("object_target")
                if isinstance(anchor, str) and isinstance(target, str):
                    rejected.setdefault(
                        (anchor, target),
                        {
                            "source_task_id": root.get("task_id"),
                            "source_path": f"{path}.contain_candidates",
                            "reason_code": candidate.get("reason_code") or candidate.get("reason"),
                            "evidence_codes": candidate.get("evidence_codes") or candidate.get("evidence") or [],
                        },
                    )
    return rejected


def collect_multi_level_support_contain_promotions(
    task_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Promote only explicit Support Agent multi-level containment verdicts."""
    rejected = _contain_rejections_from_task_results(task_results)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    positive = {"inside", "accept", "accepted", "true", "yes", "positive"}
    accepted_decisions = {"accept", "accepted", "true", "yes"}
    tier_roles = {"multi_level_tier", "multilevel_tier", "internal_shelf", "bounded_tier"}
    for root in task_results:
        if not isinstance(root, dict):
            continue
        kind = str(root.get("canonical_task_kind") or root.get("task_type") or "").lower()
        capabilities = root.get("provided_capabilities", [])
        if "support" not in kind and not (
            isinstance(capabilities, list) and "support_relations_ready" in capabilities
        ):
            continue
        for path, node in _walk_with_path(root, str(root.get("task_id") or "support_task")):
            values = node.get("support_contact_candidates") if isinstance(node, dict) else None
            if not isinstance(values, list):
                continue
            for candidate in values:
                if not isinstance(candidate, dict):
                    continue
                hypothesis = candidate.get("containment_hypothesis")
                hypothesis = hypothesis if isinstance(hypothesis, dict) else {}
                anchor = hypothesis.get("object_anchor") or hypothesis.get("anchor") or candidate.get("object_anchor")
                target = hypothesis.get("object_target") or hypothesis.get("target") or candidate.get("object_target")
                pair = (str(anchor or ""), str(target or ""))
                role = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    str(candidate.get("support_surface_role") or "").strip().lower(),
                ).strip("_")
                qualifies = (
                    str(candidate.get("decision") or "").lower() in accepted_decisions
                    and str(candidate.get("support_mode") or "").lower() == "defer_to_contain"
                    and str(hypothesis.get("decision") or hypothesis.get("status") or "").lower() in positive
                    and role in tier_roles
                    and pair[0].startswith("object_")
                    and pair[1].startswith("object_")
                    and pair[0] != pair[1]
                )
                if not qualifies or pair in seen:
                    continue
                base = {
                    "object_anchor": pair[0],
                    "object_target": pair[1],
                    "source_task_id": root.get("task_id"),
                    "source_path": f"{path}.support_contact_candidates",
                    "support_surface_role": role,
                    "reason_code": candidate.get("reason_code") or candidate.get("reason"),
                    "raw_candidate": candidate,
                }
                if pair in rejected:
                    skipped.append({
                        **base,
                        "reason": "explicit_contain_specialist_rejection_prevents_support_promotion",
                        "contain_rejection": rejected[pair],
                    })
                    continue
                seen.add(pair)
                accepted.append({
                    **base,
                    "relation": {
                        "relation": "contain",
                        "parameters": {"object_anchor": pair[0], "object_target": pair[1]},
                    },
                    "decision": "accept",
                })
    return {
        "schema_version": "multi_level_support_contain_promotion",
        "accepted": accepted,
        "skipped": skipped,
        "policy": (
            "Promote only an explicit accepted Support Agent defer_to_contain verdict with an inside "
            "hypothesis and a multi-level/internal tier role; an explicit contain-specialist rejection wins."
        ),
    }


def _contain_uncertainties_from_task_results(
    task_results: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Collect explicit specialist uncertainty for later visual SchemaGuard arbitration."""
    uncertain: dict[tuple[str, str], dict[str, Any]] = {}
    for root in task_results:
        if not isinstance(root, dict):
            continue
        kind = str(root.get("canonical_task_kind") or root.get("task_type") or "").lower()
        capabilities = root.get("provided_capabilities", [])
        if "contain" not in kind and not (isinstance(capabilities, list) and "contain_relations_ready" in capabilities):
            continue
        for path, node in _walk_with_path(root, str(root.get("task_id") or "contain_task")):
            values = node.get("contain_candidates") if isinstance(node, dict) else None
            if not isinstance(values, list):
                continue
            for candidate in values:
                if not isinstance(candidate, dict) or str(candidate.get("decision") or "").lower() not in {"uncertain", "unknown", "ambiguous"}:
                    continue
                anchor, target = candidate.get("object_anchor"), candidate.get("object_target")
                if isinstance(anchor, str) and isinstance(target, str) and anchor != target:
                    uncertain.setdefault(
                        (anchor, target),
                        {
                            "source_task_id": root.get("task_id"),
                            "source_path": f"{path}.contain_candidates",
                            "reason_code": candidate.get("reason_code") or candidate.get("reason"),
                            "evidence_codes": candidate.get("evidence_codes") or candidate.get("evidence") or [],
                        },
                    )
    return uncertain


def _accepted_object_relation_hypotheses(
    task_results: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Collect independent accepted object attach/contain verdicts."""
    accepted_attach: dict[tuple[str, str], dict[str, Any]] = {}
    accepted_contain: dict[tuple[str, str], dict[str, Any]] = {}
    collection = collect_accepted_attach_candidates(task_results)
    for item in collection.get("accepted", []):
        relation = item.get("relation") if isinstance(item, dict) else None
        params = relation.get("parameters") if isinstance(relation, dict) and isinstance(relation.get("parameters"), dict) else {}
        relation_type = relation.get("relation") if isinstance(relation, dict) else None
        anchor, target = params.get("object_anchor"), params.get("object_target")
        if not (
            relation_type in {"attach", "contain"}
            and isinstance(anchor, str)
            and anchor.startswith("object_")
            and isinstance(target, str)
            and target.startswith("object_")
            and anchor != target
        ):
            continue
        raw = item.get("raw_candidate") if isinstance(item.get("raw_candidate"), dict) else {}
        evidence = raw.get("evidence_codes") or raw.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [evidence] if evidence else []
        record = {
            "object_anchor": anchor,
            "object_target": target,
            "source_path": item.get("source_path"),
            "reason_code": raw.get("reason_code") or raw.get("reason") or raw.get("rationale"),
            "evidence_codes": [str(value)[:120] for value in evidence][:8],
        }
        destination = accepted_attach if relation_type == "attach" else accepted_contain
        destination.setdefault((anchor, target), record)
    return accepted_attach, accepted_contain


def _contain_conflict_object_metadata(
    scene_state: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    candidates = scene_state.get("object_candidates", {}) if isinstance(scene_state, dict) else {}
    items = candidates.get("items", []) if isinstance(candidates, dict) else []
    result: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id") or item.get("id")
        if not object_id:
            continue
        result[str(object_id)] = {
            "category": item.get("category"),
            "image_bbox": item.get("bbox_2d") or item.get("image_bbox"),
        }
    return result


def collect_contain_conflicts(
    scene_state: dict[str, Any] | None,
    task_results: list[dict[str, Any]],
    minimum_local_confidence: float = CONTAIN_CONFLICT_MIN_LOCAL_CONFIDENCE,
) -> dict[str, Any]:
    """Return semantic attach/contain disagreements for the existing guard call."""
    candidates, candidate_path, read_error = _contain_candidate_pairs_from_scene_state(scene_state)
    local: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        params = candidate.get("parameters") if isinstance(candidate.get("parameters"), dict) else {}
        anchor, target = params.get("object_anchor"), params.get("object_target")
        try:
            confidence = float(candidate.get("confidence_hint", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), list) else []
        evidence_text = " ".join(str(item).lower() for item in evidence)
        if not (isinstance(anchor, str) and isinstance(target, str) and confidence >= minimum_local_confidence):
            continue
        if "outside_anchor" in evidence_text or "outside_container" in evidence_text:
            continue
        anchor_data = candidate.get("anchor") if isinstance(candidate.get("anchor"), dict) else {}
        target_data = candidate.get("target") if isinstance(candidate.get("target"), dict) else {}
        local.setdefault(
            (anchor, target),
            {
                "confidence_hint": round(confidence, 4),
                "evidence": [str(item)[:120] for item in evidence][:8],
                "anchor_category": anchor_data.get("category"),
                "target_category": target_data.get("category"),
                "anchor_image_bbox": anchor_data.get("image_bbox"),
                "target_image_bbox": target_data.get("image_bbox"),
            },
        )

    support_inside: dict[tuple[str, str], dict[str, Any]] = {}
    for root in task_results:
        if not isinstance(root, dict) or "support_contact" not in str(root.get("canonical_task_kind") or root.get("task_type") or "").lower():
            continue
        for path, node in _walk_with_path(root, str(root.get("task_id") or "support_task")):
            values = node.get("support_contact_candidates") if isinstance(node, dict) else None
            if not isinstance(values, list):
                continue
            for candidate in values:
                hypothesis = _positive_containment_hypothesis(candidate) if isinstance(candidate, dict) else None
                if hypothesis:
                    key = (hypothesis["object_anchor"], hypothesis["object_target"])
                    support_inside.setdefault(
                        key,
                        {
                            "source_task_id": root.get("task_id"),
                            "source_path": f"{path}.support_contact_candidates",
                            **hypothesis,
                        },
                    )

    rejected = _contain_rejections_from_task_results(task_results)
    uncertain = _contain_uncertainties_from_task_results(task_results)
    accepted_attach, accepted_contain = _accepted_object_relation_hypotheses(task_results)
    # Two independent semantic agents explicitly disagree here. Do not require
    # the local high-recall selector to exceed a geometry-derived confidence
    # threshold before asking the existing SchemaGuard to arbitrate the image.
    support_reject_conflict_pairs = set(support_inside) & set(rejected)
    accepted_conflict_pairs = set(accepted_attach) & set(accepted_contain)
    localized_disagreement_pairs = set(local) & set(accepted_attach) & set(rejected)
    uncertain_high_recall_pairs = (set(local) & set(uncertain)) - set(rejected) - set(accepted_contain)
    conflict_pairs = support_reject_conflict_pairs | accepted_conflict_pairs | localized_disagreement_pairs | uncertain_high_recall_pairs
    object_metadata = _contain_conflict_object_metadata(scene_state)

    conflicts: list[dict[str, Any]] = []
    for anchor, target in sorted(
        conflict_pairs,
        key=lambda pair: (sorted_object_id_key(pair[0]), sorted_object_id_key(pair[1])),
    ):
        local_item = local.get((anchor, target), {})
        anchor_metadata = object_metadata.get(anchor, {})
        target_metadata = object_metadata.get(target, {})
        conflict_types: list[str] = []
        if (anchor, target) in accepted_conflict_pairs:
            conflict_types.append("accepted_attach_vs_accepted_contain")
        if (anchor, target) in support_reject_conflict_pairs:
            conflict_types.append("support_inside_vs_contain_reject")
        if (anchor, target) in localized_disagreement_pairs:
            conflict_types.append("accepted_attach_vs_high_recall_contain_reject")
        if (anchor, target) in uncertain_high_recall_pairs:
            conflict_types.append("high_recall_contain_uncertain")
        conflict: dict[str, Any] = {
            "conflict_id": _contain_conflict_id(anchor, target),
            "conflict_type": conflict_types[0],
            "conflict_types": conflict_types,
            "object_anchor": anchor,
            "object_target": target,
            "anchor_category": local_item.get("anchor_category") or anchor_metadata.get("category"),
            "target_category": local_item.get("target_category") or target_metadata.get("category"),
            "anchor_image_bbox": local_item.get("anchor_image_bbox") or anchor_metadata.get("image_bbox"),
            "target_image_bbox": local_item.get("target_image_bbox") or target_metadata.get("image_bbox"),
            "geometry_status": "semantic_image_tie_break_required",
        }
        if local_item:
            conflict["local_candidate_confidence"] = local_item.get("confidence_hint")
            conflict["local_evidence"] = local_item.get("evidence", [])
        if (anchor, target) in accepted_conflict_pairs:
            conflict["accepted_attach"] = accepted_attach[(anchor, target)]
            conflict["accepted_contain"] = accepted_contain[(anchor, target)]
        if (anchor, target) in localized_disagreement_pairs:
            conflict["accepted_attach"] = accepted_attach[(anchor, target)]
            conflict["contain_rejection"] = rejected[(anchor, target)]
        if (anchor, target) in support_reject_conflict_pairs:
            conflict["support_hypothesis"] = support_inside[(anchor, target)]
            conflict["contain_rejection"] = rejected[(anchor, target)]
        if (anchor, target) in uncertain_high_recall_pairs:
            conflict["contain_uncertainty"] = uncertain[(anchor, target)]
            if (anchor, target) in support_inside:
                conflict["support_hypothesis"] = support_inside[(anchor, target)]
        conflicts.append(conflict)

    return {
        "schema_version": "contain_conflict_candidates",
        "status": "available" if conflicts or not read_error else "unavailable",
        "contain_candidates_path": candidate_path,
        "read_error": read_error,
        "minimum_local_confidence": minimum_local_confidence,
        "high_confidence_candidate_pair_count": len(local),
        "positive_support_hypothesis_count": len(support_inside),
        "contain_rejection_count": len(rejected),
        "contain_uncertain_pair_count": len(uncertain),
        "uncertain_high_recall_conflict_count": len(uncertain_high_recall_pairs),
        "accepted_attach_pair_count": len(accepted_attach),
        "accepted_contain_pair_count": len(accepted_contain),
        "accepted_pair_conflict_count": len(accepted_conflict_pairs),
        "support_contain_reject_conflict_count": len(support_reject_conflict_pairs),
        "localized_attach_reject_conflict_count": len(localized_disagreement_pairs),
        "conflicts": conflicts,
    }


def compact_contain_conflicts_for_guard(conflict_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Keep semantic verdicts; local recall scores and bbox arithmetic stay local."""
    compact: list[dict[str, Any]] = []
    for item in (conflict_report or {}).get("conflicts", []):
        if not isinstance(item, dict):
            continue
        support = item.get("support_hypothesis") if isinstance(item.get("support_hypothesis"), dict) else {}
        rejected = item.get("contain_rejection") if isinstance(item.get("contain_rejection"), dict) else {}
        accepted_attach = item.get("accepted_attach") if isinstance(item.get("accepted_attach"), dict) else {}
        accepted_contain = item.get("accepted_contain") if isinstance(item.get("accepted_contain"), dict) else {}
        uncertain = item.get("contain_uncertainty") if isinstance(item.get("contain_uncertainty"), dict) else {}
        compact.append(
            {
                "conflict_id": item.get("conflict_id"),
                "conflict_type": item.get("conflict_type"),
                "conflict_types": item.get("conflict_types", []),
                "object_anchor": item.get("object_anchor"),
                "object_target": item.get("object_target"),
                "anchor_category": item.get("anchor_category"),
                "target_category": item.get("target_category"),
                "accepted_attach": {
                    "reason_code": accepted_attach.get("reason_code"),
                    "evidence_codes": accepted_attach.get("evidence_codes", []),
                } if accepted_attach else None,
                "accepted_contain": {
                    "reason_code": accepted_contain.get("reason_code"),
                    "evidence_codes": accepted_contain.get("evidence_codes", []),
                } if accepted_contain else None,
                "support_hypothesis": {
                    "reason_code": support.get("reason_code"),
                    "support_mode": support.get("support_mode"),
                } if support else None,
                "contain_rejection": {
                    "reason_code": rejected.get("reason_code"),
                    "evidence_codes": rejected.get("evidence_codes", []),
                } if rejected else None,
                "contain_uncertainty": {
                    "reason_code": uncertain.get("reason_code"),
                    "evidence_codes": uncertain.get("evidence_codes", []),
                } if uncertain else None,
                "geometry_status": item.get("geometry_status"),
            }
        )
    return compact

def collect_contain_conflict_resolutions(guard_result: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    resolutions: dict[str, dict[str, Any]] = {}
    if not isinstance(guard_result, dict):
        return resolutions
    for _, node in _walk_with_path(guard_result, "DeliverySchemaGuard"):
        values = node.get("contain_conflict_resolutions") if isinstance(node, dict) else None
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            conflict_id = item.get("conflict_id")
            if not isinstance(conflict_id, str) or not conflict_id:
                anchor, target = item.get("object_anchor"), item.get("object_target")
                if isinstance(anchor, str) and isinstance(target, str):
                    conflict_id = _contain_conflict_id(anchor, target)
            if isinstance(conflict_id, str) and conflict_id:
                normalized = dict(item)
                if not normalized.get("decision") and normalized.get("verdict"):
                    normalized["decision"] = normalized["verdict"]
                if not normalized.get("reason") and not normalized.get("reason_code") and normalized.get("explanation"):
                    normalized["reason"] = normalized["explanation"]
                resolutions.setdefault(conflict_id, normalized)
    return resolutions


def resolve_attach_contain_conflicts(
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove unresolved same-pair conflicts and cross-furniture support contradictions."""
    attach_pairs: set[tuple[Any, Any]] = set()
    contain_pairs: set[tuple[Any, Any]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_type = relation.get("relation")
        if relation_type not in {"attach", "contain"}:
            continue
        params = relation.get("parameters")
        params = params if isinstance(params, dict) else {}
        pair = (params.get("object_anchor"), params.get("object_target"))
        if relation_type == "attach":
            attach_pairs.add(pair)
        else:
            contain_pairs.add(pair)

    if not attach_pairs or not contain_pairs:
        return relations, []
    conflict_pairs = attach_pairs & contain_pairs
    contain_anchors_by_target: dict[Any, set[Any]] = {}
    for anchor, target in contain_pairs:
        contain_anchors_by_target.setdefault(target, set()).add(anchor)
    cross_furniture_attach_pairs = {
        pair
        for pair in attach_pairs
        if isinstance(pair[0], str)
        and pair[0].startswith("object_")
        and pair[1] in contain_anchors_by_target
        and pair[0] not in contain_anchors_by_target[pair[1]]
    }
    if not conflict_pairs and not cross_furniture_attach_pairs:
        return relations, []

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for relation in relations:
        params = relation.get("parameters", {}) if isinstance(relation, dict) and isinstance(relation.get("parameters"), dict) else {}
        pair = (params.get("object_anchor"), params.get("object_target"))
        if pair in conflict_pairs and relation.get("relation") in {"attach", "contain"}:
            removed.append(
                {
                    "relation": relation,
                    "reason": "unresolved_attach_contain_same_pair_requires_schema_guard",
                    "policy": (
                        "Attach and contain are mutually exclusive for one anchor-target pair, "
                        "but neither relation type has a fixed priority. Both are removed when "
                        "the earlier SchemaGuard arbitration did not produce exactly one verdict."
                    ),
                }
            )
            continue
        if pair in cross_furniture_attach_pairs and relation.get("relation") == "attach":
            removed.append(
                {
                    "relation": relation,
                    "reason": "object_attach_conflicts_with_other_contain_anchor",
                    "policy": (
                        "A target retained inside one object container cannot also use a different object as "
                        "its physical support. Structural floor, wall, and ceiling relations are unaffected."
                    ),
                    "contain_anchors": sorted(str(value) for value in contain_anchors_by_target[pair[1]]),
                }
            )
            continue
        kept.append(relation)
    return kept, removed

def filter_redundant_floor_attaches_for_object_supported_targets(
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep object support instead of a contradictory floor fallback."""
    object_supported_targets = {
        str(params.get("object_target"))
        for relation in relations
        if isinstance(relation, dict)
        and relation.get("relation") == "attach"
        and isinstance((params := relation.get("parameters")), dict)
        and isinstance(params.get("object_anchor"), str)
        and str(params.get("object_anchor")).startswith("object_")
        and params.get("object_target")
    }
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for relation in relations:
        params = relation.get("parameters", {}) if isinstance(relation, dict) else {}
        if (
            relation.get("relation") == "attach"
            and params.get("object_anchor") == "floor"
            and str(params.get("object_target")) in object_supported_targets
        ):
            removed.append(
                {
                    "relation": relation,
                    "reason": "object_supported_target_redundant_floor_attach",
                    "policy": "One target has one gravity support path; object support outranks a generic floor fallback.",
                }
            )
            continue
        kept.append(relation)
    return kept, removed


def normalize_floor_covering_relations(
    relations: list[dict[str, Any]],
    object_properties: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    floor_covering_ids = set((object_properties or {}).get("floor_covering_object_ids", []))
    if not floor_covering_ids:
        return relations, {
            "schema_version": "relation_normalization_audit",
            "floor_covering_object_ids": [],
            "rewritten_count": 0,
            "items": [],
        }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    items: list[dict[str, Any]] = []
    for rel in relations:
        if not isinstance(rel, dict) or not isinstance(rel.get("parameters"), dict):
            normalized.append(rel)
            continue
        relation_type = rel.get("relation")
        params = rel.get("parameters", {})
        anchor = params.get("object_anchor")
        target = params.get("object_target")
        new_rel = rel
        if relation_type == "attach" and anchor in floor_covering_ids and target != anchor:
            new_rel = {"relation": "attach", "parameters": {"object_anchor": "floor", "object_target": target}}
            items.append(
                {
                    "original_relation": rel,
                    "normalized_relation": new_rel,
                    "reason": "floor_covering_anchor_normalized_to_floor",
                    "floor_covering_anchor": anchor,
                }
            )
        key = (new_rel.get("relation"), new_rel.get("parameters", {}).get("object_anchor"), new_rel.get("parameters", {}).get("object_target"))
        if key in seen:
            items.append(
                {
                    "original_relation": rel,
                    "normalized_relation": new_rel,
                    "reason": "duplicate_after_floor_covering_normalization_dropped",
                }
            )
            continue
        seen.add(key)
        normalized.append(new_rel)
    return normalized, {
        "schema_version": "relation_normalization_audit",
        "floor_covering_object_ids": sorted(floor_covering_ids, key=sorted_object_id_key),
        "rewritten_count": len([item for item in items if item.get("reason") == "floor_covering_anchor_normalized_to_floor"]),
        "items": items,
        "policy": "Objects marked as floor covering are not final support anchors; their supported targets are normalized to floor attach.",
    }
