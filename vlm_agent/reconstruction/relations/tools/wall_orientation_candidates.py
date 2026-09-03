"""Local wall-orientation candidate utilities.

The room extent still comes from the complete posed-mesh footprint.  This
module only proposes robust rectangle axes from object rotations already
computed in the canonical top-down grid.  A vision task may choose one of the
finite candidates, but it never supplies coordinates or an arbitrary angle.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def normalize_axis_angle_degrees(value: float) -> float:
    """Normalize an undirected rectangular axis to ``[-45, 45)`` degrees."""
    return (float(value) + 45.0) % 90.0 - 45.0


def axis_angle_distance_degrees(first: float, second: float) -> float:
    """Return the smallest distance between two axes with a 90-degree period."""
    return abs(normalize_axis_angle_degrees(float(first) - float(second)))


def _weighted_axis_mean(records: Iterable[dict[str, Any]]) -> float:
    sine = 0.0
    cosine = 0.0
    for record in records:
        angle = math.radians(float(record["axis_angle_degrees"]) * 4.0)
        weight = float(record.get("weight") or 1.0)
        sine += math.sin(angle) * weight
        cosine += math.cos(angle) * weight
    if abs(sine) <= 1e-12 and abs(cosine) <= 1e-12:
        return 0.0
    return normalize_axis_angle_degrees(math.degrees(math.atan2(sine, cosine)) / 4.0)


def _orientation_record(item: dict[str, Any]) -> dict[str, Any] | None:
    object_id = str(item.get("object_id") or "").strip()
    rotation = item.get("rotation_degrees")
    size = item.get("size_2d")
    if not object_id or not isinstance(rotation, (int, float)) or isinstance(rotation, bool):
        return None
    if not isinstance(size, list) or len(size) != 2:
        return None
    try:
        width = abs(float(size[0]))
        height = abs(float(size[1]))
    except (TypeError, ValueError):
        return None
    if width <= 1e-9 or height <= 1e-9:
        return None
    area = width * height
    aspect_ratio = max(width, height) / max(min(width, height), 1e-9)
    # Square-like and tiny footprints have an unstable semantic heading.  The
    # area square root prevents a single large central object from completely
    # suppressing several agreeing wall-aligned objects.
    weight = math.sqrt(area) * math.sqrt(min(aspect_ratio, 4.0))
    return {
        "object_id": object_id,
        "rotation_degrees_topdown": round(float(rotation), 6),
        "axis_angle_degrees": normalize_axis_angle_degrees(float(rotation)),
        "size_2d": [width, height],
        "area_2d": area,
        "aspect_ratio": aspect_ratio,
        "weight": weight,
    }


def cluster_object_orientation_axes(
    objects: list[dict[str, Any]],
    *,
    grid_size: int,
    cluster_tolerance_degrees: float = 10.0,
    max_clusters: int = 3,
) -> dict[str, Any]:
    """Cluster reliable object headings into a small set of room-axis hints."""
    raw_records = [record for item in objects if isinstance(item, dict) if (record := _orientation_record(item))]
    max_area = max((float(record["area_2d"]) for record in raw_records), default=0.0)
    grid_area = float(max(1, grid_size - 1) ** 2)
    minimum_area = max(1.0, max_area * 0.01, grid_area * 0.0025)
    reliable = [
        record
        for record in raw_records
        if float(record["area_2d"]) >= minimum_area and float(record["aspect_ratio"]) >= 1.2
    ]
    if not reliable:
        reliable = [record for record in raw_records if float(record["aspect_ratio"]) >= 1.05]

    clusters: list[dict[str, Any]] = []
    for record in sorted(reliable, key=lambda item: (-float(item["weight"]), item["object_id"])):
        nearest = None
        nearest_distance = float("inf")
        for cluster in clusters:
            distance = axis_angle_distance_degrees(record["axis_angle_degrees"], cluster["angle_degrees_xz"])
            if distance < nearest_distance:
                nearest = cluster
                nearest_distance = distance
        if nearest is None or nearest_distance > cluster_tolerance_degrees:
            clusters.append({"members": [record], "angle_degrees_xz": record["axis_angle_degrees"]})
        else:
            nearest["members"].append(record)
            nearest["angle_degrees_xz"] = _weighted_axis_mean(nearest["members"])

    clusters.sort(
        key=lambda cluster: (
            -sum(float(item["weight"]) for item in cluster["members"]),
            -len(cluster["members"]),
            abs(float(cluster["angle_degrees_xz"])),
        )
    )
    output_clusters = []
    for index, cluster in enumerate(clusters[:max_clusters], start=1):
        members = sorted(cluster["members"], key=lambda item: (-float(item["weight"]), item["object_id"]))
        output_clusters.append({
            "cluster_id": f"object_axis_cluster_{index:02d}",
            "angle_degrees_xz": round(_weighted_axis_mean(members), 6),
            "supporting_object_ids": [item["object_id"] for item in members],
            "member_count": len(members),
            "total_weight": round(sum(float(item["weight"]) for item in members), 6),
            "members": [
                {
                    "object_id": item["object_id"],
                    "rotation_degrees_topdown": item["rotation_degrees_topdown"],
                    "axis_angle_degrees": round(float(item["axis_angle_degrees"]), 6),
                    "size_2d": [round(float(value), 6) for value in item["size_2d"]],
                    "area_2d": round(float(item["area_2d"]), 6),
                    "aspect_ratio": round(float(item["aspect_ratio"]), 6),
                    "weight": round(float(item["weight"]), 6),
                }
                for item in members
            ],
        })
    return {
        "schema_version": "object_orientation_axis_clusters",
        "coordinate_frame": "OpenGL original XZ; rectangular axes have a 90-degree period",
        "grid_size": grid_size,
        "cluster_tolerance_degrees": cluster_tolerance_degrees,
        "minimum_reliable_area_2d": round(minimum_area, 6),
        "raw_object_count": len(raw_records),
        "reliable_object_count": len(reliable),
        "clusters": output_clusters,
    }






WALL_ALIGNMENT_ROLES = {
    "wall_mounted": 1.20,
    "wall_embedded": 1.20,
    "against_wall": 1.00,
    "wall_aligned": 0.85,
}
WALL_ALIGNMENT_CONFIDENCE = {
    "high": 1.00,
    "medium": 0.65,
}


def wall_alignment_object_records(
    task_results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract compact Agent judgements about objects that reveal wall axes.

    Candidate ids and wall ids are intentionally ignored here. The Agent's
    job is semantic: identify objects whose real pose follows a room wall.
    Local code later reads each selected object's canonical top-down rotation.
    """
    best_by_object: dict[str, dict[str, Any]] = {}

    def walk(value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for result in task_results:
        for node in walk(result):
            for key in ("wall_alignment_objects", "likely_wall_objects"):
                items = node.get(key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    object_id = str(item.get("object_id") or "").strip()
                    role = str(item.get("role") or "").strip().lower()
                    confidence = str(item.get("confidence") or "").strip().lower()
                    if (
                        not object_id
                        or role not in WALL_ALIGNMENT_ROLES
                        or confidence not in WALL_ALIGNMENT_CONFIDENCE
                    ):
                        continue
                    score = WALL_ALIGNMENT_ROLES[role] * WALL_ALIGNMENT_CONFIDENCE[confidence]
                    current = best_by_object.get(object_id)
                    if current is None or score > float(current["semantic_weight"]):
                        best_by_object[object_id] = {
                            "object_id": object_id,
                            "role": role,
                            "confidence": confidence,
                            "semantic_weight": score,
                        }
    return [best_by_object[key] for key in sorted(best_by_object)]


def resolve_orientation_candidate_from_wall_objects(
    task_results: Iterable[dict[str, Any]],
    candidates: list[dict[str, Any]],
    object_rotations_topdown: dict[str, float],
    *,
    agreement_limit_degrees: float = 22.5,
    minimum_score: float = 0.35,
    minimum_margin: float = 0.05,
) -> dict[str, Any]:
    """Score finite wall axes from Agent-selected objects and local rotations.

    The Agent never chooses a candidate in this path. It only identifies
    semantically reliable wall-following objects. The returned choice is a
    deterministic function of their rotation_degrees_topdown and the already
    generated finite candidate angles.
    """
    semantic_records = wall_alignment_object_records(task_results)
    # Candidate generation records an axis reliability weight for each object.
    # Use it only to break semantic-score ties among Agent-selected objects.
    reliability_by_object: dict[str, float] = {}
    for candidate in candidates:
        evidence = candidate.get("evidence") if isinstance(candidate, dict) else None
        cluster = evidence.get("orientation_cluster") if isinstance(evidence, dict) else None
        members = cluster.get("members") if isinstance(cluster, dict) else None
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, dict) or not member.get("object_id"):
                continue
            try:
                weight = max(float(member.get("weight", 1.0)), 1e-6)
            except (TypeError, ValueError):
                weight = 1.0
            object_id = str(member["object_id"])
            reliability_by_object[object_id] = max(reliability_by_object.get(object_id, 0.0), weight)

    usable_records: list[dict[str, Any]] = []
    missing_rotation_ids: list[str] = []
    for item in semantic_records:
        object_id = item["object_id"]
        rotation = object_rotations_topdown.get(object_id)
        if not isinstance(rotation, (int, float)) or isinstance(rotation, bool):
            missing_rotation_ids.append(object_id)
            continue
        usable_records.append({
            **item,
            "rotation_degrees_topdown": round(float(rotation), 6),
            "axis_angle_degrees": round(normalize_axis_angle_degrees(float(rotation)), 6),
            "orientation_reliability_weight": round(reliability_by_object.get(object_id, 1.0), 6),
        })

    candidate_scores: list[dict[str, Any]] = []
    total_semantic_weight = sum(float(item["semantic_weight"]) for item in usable_records)
    total_reliability_weight = sum(
        float(item["semantic_weight"]) * float(item["orientation_reliability_weight"])
        for item in usable_records
    )
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
            continue
        orientation = candidate.get("orientation") if isinstance(candidate.get("orientation"), dict) else {}
        angle = orientation.get("angle_degrees_xz")
        if not isinstance(angle, (int, float)) or isinstance(angle, bool):
            continue
        object_scores = []
        weighted_agreement = 0.0
        reliability_weighted_agreement = 0.0
        for item in usable_records:
            distance = axis_angle_distance_degrees(item["axis_angle_degrees"], float(angle))
            agreement = max(0.0, 1.0 - distance / max(agreement_limit_degrees, 1e-6))
            contribution = float(item["semantic_weight"]) * agreement
            reliability_contribution = (
                float(item["semantic_weight"])
                * float(item["orientation_reliability_weight"])
                * agreement
            )
            weighted_agreement += contribution
            reliability_weighted_agreement += reliability_contribution
            object_scores.append({
                "object_id": item["object_id"],
                "axis_distance_degrees": round(distance, 6),
                "agreement": round(agreement, 6),
                "weighted_contribution": round(contribution, 6),
                "orientation_reliability_weight": item["orientation_reliability_weight"],
                "reliability_weighted_contribution": round(reliability_contribution, 6),
            })
        normalized_score = weighted_agreement / total_semantic_weight if total_semantic_weight > 0 else 0.0
        reliability_normalized_score = (
            reliability_weighted_agreement / total_reliability_weight
            if total_reliability_weight > 0 else normalized_score
        )
        candidate_scores.append({
            "candidate_id": str(candidate["candidate_id"]),
            "angle_degrees_xz": round(float(angle), 6),
            "normalized_score": round(normalized_score, 6),
            "weighted_score": round(weighted_agreement, 6),
            "reliability_normalized_score": round(reliability_normalized_score, 6),
            "reliability_weighted_score": round(reliability_weighted_agreement, 6),
            "object_scores": object_scores,
        })

    candidate_scores.sort(key=lambda item: (-float(item["normalized_score"]), item["candidate_id"]))
    best = candidate_scores[0] if candidate_scores else None
    runner_up = candidate_scores[1] if len(candidate_scores) > 1 else None
    margin = (
        float(best["normalized_score"]) - float(runner_up["normalized_score"])
        if best and runner_up else float(best["normalized_score"]) if best else 0.0
    )
    resolved_id = None
    status = "no_usable_agent_wall_alignment_objects"
    reliability_tiebreak: dict[str, Any] = {
        "used": False,
        "resolved_candidate_id": None,
        "best_score": None,
        "runner_up_score": None,
        "score_margin": None,
    }
    if best and usable_records:
        if float(best["normalized_score"]) < minimum_score:
            status = "agent_objects_do_not_match_any_finite_candidate"
        elif runner_up and margin < minimum_margin:
            reliability_ranked = sorted(
                candidate_scores,
                key=lambda item: (-float(item["reliability_normalized_score"]), item["candidate_id"]),
            )
            reliability_best = reliability_ranked[0]
            reliability_runner_up = reliability_ranked[1] if len(reliability_ranked) > 1 else None
            reliability_margin = (
                float(reliability_best["reliability_normalized_score"])
                - float(reliability_runner_up["reliability_normalized_score"])
                if reliability_runner_up else float(reliability_best["reliability_normalized_score"])
            )
            reliability_tiebreak = {
                "used": True,
                "resolved_candidate_id": None,
                "best_candidate_id": reliability_best["candidate_id"],
                "best_score": float(reliability_best["reliability_normalized_score"]),
                "runner_up_candidate_id": reliability_runner_up["candidate_id"] if reliability_runner_up else None,
                "runner_up_score": (
                    float(reliability_runner_up["reliability_normalized_score"])
                    if reliability_runner_up else None
                ),
                "score_margin": round(reliability_margin, 6),
                "minimum_margin": minimum_margin,
                "policy": "Break semantic-score ties only with local geometric reliability weights for the Agent-selected wall-following objects.",
            }
            if (
                float(reliability_best["reliability_normalized_score"]) >= minimum_score
                and reliability_margin >= minimum_margin
            ):
                resolved_id = str(reliability_best["candidate_id"])
                reliability_tiebreak["resolved_candidate_id"] = resolved_id
                status = "selected_by_local_agent_object_rotation_reliability_tiebreak"
            else:
                status = "finite_candidates_ambiguous_for_agent_objects"
        else:
            resolved_id = str(best["candidate_id"])
            status = "selected_by_local_agent_object_rotation_score"

    return {
        "resolved_candidate_id": resolved_id,
        "resolution_source": (
            "local_agent_object_rotation_reliability_tiebreak"
            if resolved_id and reliability_tiebreak.get("resolved_candidate_id")
            else "local_agent_object_rotation_score" if resolved_id
            else "no_local_object_score_selection"
        ),
        "status": status,
        "wall_alignment_objects": usable_records,
        "missing_rotation_object_ids": missing_rotation_ids,
        "candidate_scores": candidate_scores,
        "best_score": float(best["normalized_score"]) if best else 0.0,
        "runner_up_score": float(runner_up["normalized_score"]) if runner_up else None,
        "score_margin": round(margin, 6),
        "reliability_tiebreak": reliability_tiebreak,
        "agreement_limit_degrees": agreement_limit_degrees,
        "minimum_score": minimum_score,
        "minimum_margin": minimum_margin,
        "policy": "Agent selects semantic wall-alignment objects; local canonical rotations score finite formal OBB axes.",
    }
