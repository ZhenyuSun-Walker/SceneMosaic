from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.state import sorted_object_id_key, write_json, write_md
from relations.tools.wall_canonical_naming import canonical_wall_naming_policy, provisional_segments_from_polygon

WALL_IDS = ["wall_001", "wall_002", "wall_003", "wall_004"]
WALL_CONTACT_EPSILON = 0.035
FLAT_CONTACT_EDGE_MIN_LENGTH_RATIO = 0.008
FLAT_CONTACT_EDGE_MIN_ABSOLUTE_LENGTH = 0.003
FLAT_CONTACT_EDGE_MAX_ABSOLUTE_LENGTH = 0.008
FLAT_CONTACT_EDGE_PARALLEL_TOLERANCE_DEGREES = 8.0
FLAT_CONTACT_EDGE_COLLINEAR_TOLERANCE_DEGREES = 6.0
FLAT_CONTACT_EDGE_STANDALONE_CORNER_DEGREES = 20.0
FLAT_CONTACT_EDGE_LONG_RUN_RATIO = 0.25
RENDER_GRID_WALL_PRECOMPUTE_PATH_ENV = "RENDER_GRID_WALL_PRECOMPUTE_PATH"
FORMAL_WALL_METHOD = "obb_from_mesh_footprint_min_rect"


def _load_json(path: Path | None) -> Any | None:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _bounds_from_points(points: list[list[float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    return {"x_min": min(xs), "x_max": max(xs), "z_min": min(zs), "z_max": max(zs)}


def _bounds_from_mapping(mapping: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    points: list[list[float]] = []
    for point in mapping.get("grid_points", []) if isinstance(mapping.get("grid_points"), list) else []:
        if not isinstance(point, dict):
            continue
        raw = point.get("point_3d_original_xz")
        if isinstance(raw, list) and len(raw) == 2:
            x, z = _number(raw[0]), _number(raw[1])
            if x is not None and z is not None:
                points.append([x, z])
                continue
        raw3 = point.get("point_3d_original") or point.get("point_3d_normalized")
        if isinstance(raw3, list) and len(raw3) >= 3:
            x, z = _number(raw3[0]), _number(raw3[2])
            if x is not None and z is not None:
                points.append([x, z])
    bounds = _bounds_from_points(points)
    if bounds:
        bounds["source"] = "canonical_grid_mapping.grid_points_extent_hint"
        return bounds
    x_range = mapping.get("x_range")
    z_range = mapping.get("z_range")
    if isinstance(x_range, list) and isinstance(z_range, list) and len(x_range) == 2 and len(z_range) == 2:
        vals = [_number(x_range[0]), _number(x_range[1]), _number(z_range[0]), _number(z_range[1])]
        if all(v is not None for v in vals):
            return {"x_min": min(vals[0], vals[1]), "x_max": max(vals[0], vals[1]), "z_min": min(vals[2], vals[3]), "z_max": max(vals[2], vals[3]), "source": "canonical_grid_mapping.x_range_z_range_hint"}
    return None


def _bounds_from_camera_config(camera_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(camera_config, dict):
        return None
    scene_bbox = camera_config.get("scene_bbox_global_xz")
    if isinstance(scene_bbox, dict):
        vals = {k: _number(scene_bbox.get(k)) for k in ["x_min", "x_max", "z_min", "z_max"]}
        if all(v is not None for v in vals.values()):
            vals["source"] = "topdown_camera_config.scene_bbox_global_xz_hint"
            return vals
    x_range = camera_config.get("x_range") or camera_config.get("world_to_pixel", {}).get("x_range")
    z_range = camera_config.get("z_range") or camera_config.get("world_to_pixel", {}).get("z_range")
    if isinstance(x_range, list) and isinstance(z_range, list) and len(x_range) == 2 and len(z_range) == 2:
        vals = [_number(x_range[0]), _number(x_range[1]), _number(z_range[0]), _number(z_range[1])]
        if all(v is not None for v in vals):
            return {"x_min": min(vals[0], vals[1]), "x_max": max(vals[0], vals[1]), "z_min": min(vals[2], vals[3]), "z_max": max(vals[2], vals[3]), "source": "topdown_camera_config.x_range_z_range_hint"}
    return None


def _bounds_from_objects(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    points: list[list[float]] = []
    for item in items:
        bbox = item.get("bbox_3d") if isinstance(item, dict) else None
        if not isinstance(bbox, dict):
            continue
        mn = bbox.get("min")
        mx = bbox.get("max")
        if isinstance(mn, list) and len(mn) >= 3 and isinstance(mx, list) and len(mx) >= 3:
            x0, z0 = _number(mn[0]), _number(mn[2])
            x1, z1 = _number(mx[0]), _number(mx[2])
            if None not in (x0, z0, x1, z1):
                points.extend([[x0, z0], [x1, z1]])
    bounds = _bounds_from_points(points)
    if bounds:
        bounds["source"] = "scene_state.object_candidates.bbox_3d_aggregate_hint"
        return bounds
    return None


def _clockwise_walls_from_bounds(bounds: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not bounds:
        return []
    vals = {k: _number(bounds.get(k)) for k in ["x_min", "x_max", "z_min", "z_max"]}
    if not all(v is not None for v in vals.values()):
        return []
    x_min, x_max, z_min, z_max = vals["x_min"], vals["x_max"], vals["z_min"], vals["z_max"]
    if abs(x_max - x_min) <= 1e-8 or abs(z_max - z_min) <= 1e-8:
        return []
    source = str(bounds.get("source") or "bounds_hint")
    return [
        {"id": "wall_001", "first_point": [x_min, z_min], "second_point": [x_min, z_max], "source": f"{source}", "trust_level": "hint_only", "may_be_used_as_final_wall": False},
        {"id": "wall_002", "first_point": [x_min, z_max], "second_point": [x_max, z_max], "source": f"{source}", "trust_level": "hint_only", "may_be_used_as_final_wall": False},
        {"id": "wall_003", "first_point": [x_max, z_max], "second_point": [x_max, z_min], "source": f"{source}", "trust_level": "hint_only", "may_be_used_as_final_wall": False},
        {"id": "wall_004", "first_point": [x_max, z_min], "second_point": [x_min, z_min], "source": f"{source}", "trust_level": "hint_only", "may_be_used_as_final_wall": False},
    ]


def _clean_polygon_xz(raw: Any) -> list[list[float]]:
    points: list[list[float]] = []
    if not isinstance(raw, list):
        return points
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            return []
        x, z = _number(point[0]), _number(point[1])
        if x is None or z is None:
            return []
        points.append([x, z])
    return points


def _polygon_signed_area_xz(points: list[list[float]]) -> float | None:
    if len(points) < 3:
        return None
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return total / 2.0


def _segments_from_polygon_xz(points: list[list[float]], *, source: str, method: str, role: str, xz_to_pixel_affine: Any = None) -> list[dict[str, Any]]:
    return provisional_segments_from_polygon(
        points,
        xz_to_pixel_affine=xz_to_pixel_affine,
        source=source,
        method=method,
        role=role,
        trust_level="candidate",
        may_be_used_as_final_wall=False,
    )


def _candidate_record_from_render_grid(
    raw: dict[str, Any],
    rank_label: int,
    xz_to_pixel_affine: Any = None,
    *,
    selected: bool = False,
) -> dict[str, Any] | None:
    method = str(raw.get("method") or "")
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    polygon = _clean_polygon_xz(metrics.get("polygon_xz"))
    if len(polygon) < 3:
        return None
    candidate_id = str(raw.get("candidate_id") or f"candidate_{rank_label:02d}")
    role = "primary_wall_candidate" if selected else "wall_orientation_candidate"
    source = "render_grid_selected_wall_orientation_candidate" if selected else "render_grid_wall_orientation_candidate"
    signed_area = _polygon_signed_area_xz(polygon)
    return {
        "candidate_id": candidate_id,
        "rank_label": rank_label,
        "method": method,
        "role": role,
        "polygon_xz": polygon,
        "orientation": raw.get("orientation") if isinstance(raw.get("orientation"), dict) else {},
        "evidence": raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {},
        "polygon_orientation_xz": "counter_clockwise_or_xz_positive" if signed_area and signed_area > 0 else "clockwise_or_xz_negative",
        "signed_area_xz": signed_area,
        "wall_segments": _segments_from_polygon_xz(
            polygon,
            source=source,
            method=method,
            role=role,
            xz_to_pixel_affine=xz_to_pixel_affine,
        ),
        "metrics": {
            "area_xz": metrics.get("area_xz"),
            "axis_deviation_degrees": metrics.get("axis_deviation_degrees"),
            "foreground_coverage": metrics.get("foreground_coverage"),
            "empty_pixel_ratio_in_candidate": metrics.get("empty_pixel_ratio_in_candidate"),
        },
        "selected_by_wall_semantic_review": selected,
        "trust_level": "candidate",
        "may_be_used_as_final_wall": False,
    }


def _discover_render_grid_method_candidates(scene_state: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    precompute = evidence.get("render_grid_wall_precompute") if isinstance(evidence.get("render_grid_wall_precompute"), dict) else {}
    candidate_paths = []
    for raw in [
        precompute.get("method_candidates_path") if isinstance(precompute, dict) else None,
        os.environ.get(RENDER_GRID_WALL_PRECOMPUTE_PATH_ENV),
    ]:
        if isinstance(raw, str) and raw:
            candidate_paths.append(Path(raw))
    for candidate_path in candidate_paths:
        data = _load_json(candidate_path)
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            return candidate_path, data, None
    if candidate_paths:
        return candidate_paths[0], None, "configured render-grid method_candidates.json is missing or invalid"
    return None, None, "formal render-grid precompute was not bound to scene_state"


def _extract_render_grid_wall_candidates(scene_state: dict[str, Any]) -> dict[str, Any]:
    source_path, data, error = _discover_render_grid_method_candidates(scene_state)
    if not isinstance(data, dict):
        return {
            "status": "missing",
            "source_path": str(source_path) if source_path else None,
            "reason": error,
            "primary_method": FORMAL_WALL_METHOD,
            "supporting_methods": [],
            "may_be_used_as_final_wall": False,
        }
    raw_candidates = [
        candidate
        for candidate in data.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("method") == FORMAL_WALL_METHOD
    ]
    assets = data.get("assets") if isinstance(data.get("assets"), dict) else {}
    affine_data = _load_json(Path(str(assets.get("render_grid_affine_path")))) if assets.get("render_grid_affine_path") else None
    xz_to_pixel_affine = affine_data.get("xz_to_pixel_affine") if isinstance(affine_data, dict) else None
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    precompute = evidence.get("render_grid_wall_precompute") if isinstance(evidence.get("render_grid_wall_precompute"), dict) else {}
    requested_id = precompute.get("selected_orientation_candidate_id")
    review = data.get("wall_orientation_review") if isinstance(data.get("wall_orientation_review"), dict) else {}
    baseline_id = review.get("baseline_candidate_id")
    available_ids = [
        str(candidate.get("candidate_id") or f"candidate_{index:02d}")
        for index, candidate in enumerate(raw_candidates, start=1)
    ]
    if requested_id not in available_ids:
        requested_id = baseline_id if baseline_id in available_ids else (available_ids[0] if available_ids else None)

    orientation_candidates = []
    primary = None
    primary_rank = None
    for index, candidate in enumerate(raw_candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index:02d}")
        record = _candidate_record_from_render_grid(
            candidate,
            index,
            xz_to_pixel_affine=xz_to_pixel_affine,
            selected=candidate_id == requested_id,
        )
        if not record:
            continue
        orientation_candidates.append(record)
        if candidate_id == requested_id:
            primary = record
            primary_rank = index

    method_policy = data.get("method_policy") if isinstance(data.get("method_policy"), dict) else {}
    return {
        "status": "available" if primary else "missing_primary",
        "source_path": str(source_path),
        "source_run_id": data.get("run_id"),
        "source_scene": data.get("scene"),
        "assets": assets,
        "render_grid_affine_path": assets.get("render_grid_affine_path"),
        "wall_id_naming_source": "deterministic_screen_clockwise_wall_naming",
        "canonical_topdown_image_path": assets.get("canonical_topdown_path"),
        "canonical_grid_mapping_path": assets.get("canonical_grid_mapping_path"),
        "primary_candidate_overlay_path": assets.get("primary_candidate_overlay_path"),
        "candidate_overlay_path": assets.get("candidate_overlay_path"),
        "wall_orientation_candidates_overlay_path": assets.get("wall_orientation_candidates_overlay_path"),
        "integration_mode": "formal_render_grid_precompute_with_local_orientation_selection",
        "method_family": "render_grid_wall_precompute",
        "method_policy": {
            **method_policy,
            "agent_should_not_choose_method": True,
            "orientation_selected_locally": True,
            "primary_method": FORMAL_WALL_METHOD,
            "supporting_methods": [],
        },
        "wall_orientation_review": review,
        "orientation_candidates": orientation_candidates,
        "selected_orientation_candidate_id": requested_id,
        "selection_source": (
            str(precompute.get("wall_orientation_selection_source") or "local_object_rotation_score")
            if precompute.get("selected_orientation_candidate_id") in available_ids
            else "precompute_baseline_fallback"
        ),
        "primary_candidate": primary,
        "primary_rank": primary_rank,
        "supporting_candidates": [
            candidate for candidate in orientation_candidates
            if candidate.get("candidate_id") != requested_id
        ],
        "may_be_used_as_final_wall": False,
    }


def _point_segment_distance(point: list[float], a: list[float], b: list[float]) -> float:
    px, pz = point
    ax, az = a
    bx, bz = b
    dx = bx - ax
    dz = bz - az
    denom = dx * dx + dz * dz
    if denom <= 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / denom))
    qx = ax + t * dx
    qz = az + t * dz
    return math.hypot(px - qx, pz - qz)


def _bbox_wall_distance(min_xz: list[float] | None, max_xz: list[float] | None, wall: dict[str, Any]) -> tuple[float | None, str | None]:
    if not min_xz or not max_xz:
        return None, None
    x0, z0 = min(min_xz[0], max_xz[0]), min(min_xz[1], max_xz[1])
    x1, z1 = max(min_xz[0], max_xz[0]), max(min_xz[1], max_xz[1])
    a = wall.get("first_point")
    b = wall.get("second_point")
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == 2 and len(b) == 2):
        return None, None
    ax, az = float(a[0]), float(a[1])
    bx, bz = float(b[0]), float(b[1])
    if abs(ax - bx) <= 1e-8:
        wall_x = ax
        overlap = not (z1 < min(az, bz) or z0 > max(az, bz))
        side = "left" if wall_x <= (x0 + x1) / 2 else "right"
        if overlap:
            return min(abs(x0 - wall_x), abs(x1 - wall_x)), side
    if abs(az - bz) <= 1e-8:
        wall_z = az
        overlap = not (x1 < min(ax, bx) or x0 > max(ax, bx))
        side = "back" if wall_z <= (z0 + z1) / 2 else "front"
        if overlap:
            return min(abs(z0 - wall_z), abs(z1 - wall_z)), side
    corners = [[x0, z0], [x0, z1], [x1, z0], [x1, z1]]
    return min(_point_segment_distance(corner, [ax, az], [bx, bz]) for corner in corners), None


def analyze_flat_contact_edge(
    hull_xz: list[list[float]] | None,
    wall: dict[str, Any],
) -> dict[str, Any]:
    """Detect a finite flat wall-contact edge while rejecting point/corner contact."""
    points = [
        [float(point[0]), float(point[1])]
        for point in hull_xz or []
        if isinstance(point, list) and len(point) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in point)
    ]
    wall_a = wall.get("first_point")
    wall_b = wall.get("second_point")
    if len(points) < 3 or not (
        isinstance(wall_a, list) and len(wall_a) == 2
        and isinstance(wall_b, list) and len(wall_b) == 2
    ):
        return {
            "analysis_available": False,
            "has_valid_flat_contact_edge": None,
            "reason": "mesh_floor_footprint_hull_or_wall_missing",
        }

    wall_dx = float(wall_b[0]) - float(wall_a[0])
    wall_dz = float(wall_b[1]) - float(wall_a[1])
    wall_length = math.hypot(wall_dx, wall_dz)
    if wall_length <= 1e-12:
        return {
            "analysis_available": False,
            "has_valid_flat_contact_edge": None,
            "reason": "degenerate_wall_segment",
        }
    wall_unit = (wall_dx / wall_length, wall_dz / wall_length)

    def undirected_angle_degrees(first: tuple[float, float], second: tuple[float, float]) -> float:
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        if first_length <= 1e-12 or second_length <= 1e-12:
            return 90.0
        cosine = abs((first[0] * second[0] + first[1] * second[1]) / (first_length * second_length))
        return math.degrees(math.acos(max(0.0, min(1.0, cosine))))

    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    footprint_span = max(max(xs) - min(xs), max(zs) - min(zs), 1e-9)
    minimum_contact_length = max(
        FLAT_CONTACT_EDGE_MIN_ABSOLUTE_LENGTH,
        min(FLAT_CONTACT_EDGE_MAX_ABSOLUTE_LENGTH, footprint_span * FLAT_CONTACT_EDGE_MIN_LENGTH_RATIO),
    )
    distance_threshold = min(WALL_CONTACT_EPSILON, max(0.006, footprint_span * 0.10))
    edges: list[dict[str, Any]] = []
    for index, start_point in enumerate(points):
        end_point = points[(index + 1) % len(points)]
        vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        length = math.hypot(*vector)
        midpoint = [(start_point[0] + end_point[0]) / 2.0, (start_point[1] + end_point[1]) / 2.0]
        distances = [
            _point_segment_distance(start_point, wall_a, wall_b),
            _point_segment_distance(midpoint, wall_a, wall_b),
            _point_segment_distance(end_point, wall_a, wall_b),
        ]
        parallel_error = undirected_angle_degrees(vector, wall_unit)
        whole_edge_near_wall = max(distances) <= distance_threshold
        parallel_to_wall = parallel_error <= FLAT_CONTACT_EDGE_PARALLEL_TOLERANCE_DEGREES
        edges.append({
            "index": index,
            "start": start_point,
            "end": end_point,
            "vector": vector,
            "length": length,
            "length_ratio": length / footprint_span,
            "endpoint_and_midpoint_distances": distances,
            "maximum_distance_to_wall": max(distances),
            "minimum_distance_to_wall": min(distances),
            "parallel_error_degrees": parallel_error,
            "parallel_to_wall": parallel_to_wall,
            "whole_edge_near_wall": whole_edge_near_wall,
            "flat_segment_candidate": length > 1e-9 and parallel_to_wall and whole_edge_near_wall,
        })

    groups: list[list[dict[str, Any]]] = []
    for edge in edges:
        if not edge["flat_segment_candidate"]:
            continue
        if (
            groups
            and edge["index"] == groups[-1][-1]["index"] + 1
            and undirected_angle_degrees(groups[-1][-1]["vector"], edge["vector"])
            <= FLAT_CONTACT_EDGE_COLLINEAR_TOLERANCE_DEGREES
        ):
            groups[-1].append(edge)
        else:
            groups.append([edge])
    if (
        len(groups) > 1
        and groups[0][0]["index"] == 0
        and groups[-1][-1]["index"] == len(edges) - 1
        and undirected_angle_degrees(groups[-1][-1]["vector"], groups[0][0]["vector"])
        <= FLAT_CONTACT_EDGE_COLLINEAR_TOLERANCE_DEGREES
    ):
        groups = [groups[-1] + groups[0], *groups[1:-1]]

    flat_runs: list[dict[str, Any]] = []
    for group in groups:
        projected_length = sum(
            abs(edge["vector"][0] * wall_unit[0] + edge["vector"][1] * wall_unit[1])
            for edge in group
        )
        standalone_corner_angles = None
        standalone_shape_supported = True
        long_contact_run = projected_length / footprint_span >= FLAT_CONTACT_EDGE_LONG_RUN_RATIO
        if len(group) == 1:
            edge_index = group[0]["index"]
            previous_edge = edges[(edge_index - 1) % len(edges)]
            next_edge = edges[(edge_index + 1) % len(edges)]
            standalone_corner_angles = [
                undirected_angle_degrees(previous_edge["vector"], group[0]["vector"]),
                undirected_angle_degrees(group[0]["vector"], next_edge["vector"]),
            ]
            standalone_shape_supported = (
                min(standalone_corner_angles) >= FLAT_CONTACT_EDGE_STANDALONE_CORNER_DEGREES
                or long_contact_run
            )
        contact_extent_sufficient = projected_length >= minimum_contact_length
        valid = contact_extent_sufficient and standalone_shape_supported
        flat_runs.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "length": projected_length,
            "length_ratio": projected_length / footprint_span,
            "source_edge_count": len(group),
            "source_edge_indices": [edge["index"] for edge in group],
            "maximum_distance_to_wall": max(edge["maximum_distance_to_wall"] for edge in group),
            "minimum_distance_to_wall": min(edge["minimum_distance_to_wall"] for edge in group),
            "maximum_parallel_error_degrees": max(edge["parallel_error_degrees"] for edge in group),
            "standalone_corner_angles_degrees": standalone_corner_angles,
            "standalone_shape_supported": standalone_shape_supported,
            "long_contact_run": long_contact_run,
            "contact_extent_sufficient": contact_extent_sufficient,
            "whole_edge_near_wall": True,
            "parallel_to_wall": True,
            "valid_flat_contact_run": valid,
            "support_type": "fragmented_collinear_flat_edge" if len(group) > 1 else "standalone_flat_edge",
        })

    valid_runs = [run for run in flat_runs if run["valid_flat_contact_run"]]
    nearest = min(edges, key=lambda edge: edge["maximum_distance_to_wall"])
    nearest_public = {key: value for key, value in nearest.items() if key != "vector"}
    best = max(valid_runs, key=lambda run: run["length"]) if valid_runs else nearest_public
    return {
        "analysis_available": True,
        "has_valid_flat_contact_edge": bool(valid_runs),
        "mesh_floor_footprint_vertex_count": len(points),
        "footprint_span": footprint_span,
        "minimum_contact_length": minimum_contact_length,
        "distance_threshold": distance_threshold,
        "parallel_tolerance_degrees": FLAT_CONTACT_EDGE_PARALLEL_TOLERANCE_DEGREES,
        "collinear_merge_tolerance_degrees": FLAT_CONTACT_EDGE_COLLINEAR_TOLERANCE_DEGREES,
        "flat_contact_runs": flat_runs,
        "best_edge": best,
        "reason": "valid_flat_edge_contact_not_point_or_corner" if valid_runs else "no_flat_edge_contact_or_only_point_corner_contact",
    }

def _relation_footprint_records(scene_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state, dict) else {}
    relation_precompute = evidence.get("relation_precompute", {}) if isinstance(evidence, dict) else {}
    outputs = relation_precompute.get("outputs", {}) if isinstance(relation_precompute, dict) else {}
    candidate = outputs.get("object_footprints_path") if isinstance(outputs, dict) else None
    if not candidate:
        layout_path = evidence.get("layout_for_relation_path") if isinstance(evidence, dict) else None
        if layout_path:
            candidate = str(Path(layout_path).parent / "object_footprints.json")
    data = _load_json(Path(candidate)) if candidate else None
    items = data.get("items", []) if isinstance(data, dict) else []
    return {
        str(item["object_id"]): item
        for item in items
        if isinstance(item, dict) and item.get("object_id")
    }


def classify_wall_contact_geometry(metric: dict[str, Any]) -> str:
    if metric.get("possible_against_wall_contact") and metric.get("is_floor_supported"):
        return "possible floor_supported_and_wall_contact"
    if metric.get("possible_against_wall_contact"):
        return "needs_more_evidence"
    return "not_wall_attach_by_geometry_only"


def _object_footprints(
    items: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    footprint_records: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    footprints = []
    metrics = []
    footprint_records = footprint_records or {}
    for item in sorted(items, key=lambda x: sorted_object_id_key(x.get("object_id", ""))):
        object_id = str(item.get("object_id") or "")
        footprint_record = footprint_records.get(object_id, {})
        floor_hull = footprint_record.get("mesh_floor_footprint_hull_xz")
        bbox = item.get("bbox_3d") if isinstance(item, dict) else None
        center_xz = None
        min_xz = None
        max_xz = None
        bottom_y = None
        extent_xz = None
        if isinstance(bbox, dict):
            center = bbox.get("center")
            mn = bbox.get("min")
            mx = bbox.get("max")
            if isinstance(center, list) and len(center) >= 3:
                x, z = _number(center[0]), _number(center[2])
                if x is not None and z is not None:
                    center_xz = [x, z]
            if isinstance(mn, list) and len(mn) >= 3:
                x, y, z = _number(mn[0]), _number(mn[1]), _number(mn[2])
                if x is not None and z is not None:
                    min_xz = [x, z]
                bottom_y = y
            if isinstance(mx, list) and len(mx) >= 3:
                x, z = _number(mx[0]), _number(mx[2])
                if x is not None and z is not None:
                    max_xz = [x, z]
            if min_xz and max_xz:
                extent_xz = [abs(max_xz[0] - min_xz[0]), abs(max_xz[1] - min_xz[1])]
        is_floor_supported = bool(bottom_y is not None and bottom_y <= -0.05)
        object_metrics = []
        for wall in walls:
            wall_id = wall.get("id")
            if not center_xz:
                center_distance = None
            else:
                center_distance = _point_segment_distance(center_xz, wall.get("first_point"), wall.get("second_point"))
            footprint_distance, bbox_side = _bbox_wall_distance(min_xz, max_xz, wall)
            if isinstance(floor_hull, list) and floor_hull:
                hull_distances = [
                    _point_segment_distance(point, wall.get("first_point"), wall.get("second_point"))
                    for point in floor_hull
                    if isinstance(point, list) and len(point) == 2
                ]
                if hull_distances:
                    footprint_distance = min(hull_distances)
            possible_contact = bool(footprint_distance is not None and footprint_distance <= WALL_CONTACT_EPSILON)
            flat_edge = analyze_flat_contact_edge(floor_hull, wall)
            metric = {
                "object_id": item.get("object_id"),
                "category": item.get("category"),
                "wall_id": wall_id,
                "center_to_wall_distance": center_distance,
                "footprint_min_distance_to_wall": footprint_distance,
                "bbox_xz_min": min_xz,
                "bbox_xz_max": max_xz,
                "bbox_side_near_wall": possible_contact,
                "bbox_side": bbox_side,
                "object_depth_toward_wall": min(extent_xz) if extent_xz else None,
                "is_thin_planar_candidate": bool(extent_xz and min(extent_xz) <= 0.04),
                "is_floor_supported": is_floor_supported,
                "bottom_y": bottom_y,
                "possible_against_wall_contact": possible_contact,
                "mesh_floor_footprint_hull_xz": floor_hull,
                "flat_contact_edge_analysis": flat_edge,
                "flat_contact_edge_analysis_available": flat_edge.get("analysis_available"),
                "has_valid_flat_contact_edge": flat_edge.get("has_valid_flat_contact_edge"),
            }
            metric["geometry_classification"] = classify_wall_contact_geometry(metric)
            metrics.append(metric)
            object_metrics.append(metric)
        nearest = min(object_metrics, key=lambda m: m["footprint_min_distance_to_wall"] if m["footprint_min_distance_to_wall"] is not None else 10**9, default=None)
        footprints.append({
            "object_id": item.get("object_id"),
            "category": item.get("category"),
            "center_xz": center_xz,
            "bbox_xz_min": min_xz,
            "bbox_xz_max": max_xz,
            "nearest_wall_id_by_footprint": nearest.get("wall_id") if nearest else None,
            "nearest_wall_distance_by_footprint": nearest.get("footprint_min_distance_to_wall") if nearest else None,
            "possible_against_wall_contact": nearest.get("possible_against_wall_contact") if nearest else None,
            "is_floor_supported": is_floor_supported,
            "bottom_y": bottom_y,
            "mesh_floor_footprint_hull_xz": floor_hull,
            "mesh_floor_footprint_source": footprint_record.get("mesh_floor_footprint_source"),
        })
    return footprints, metrics



def _grid_point_xz(point: dict[str, Any]) -> list[float] | None:
    raw = point.get("point_3d_original_xz")
    if isinstance(raw, list) and len(raw) == 2:
        x, z = _number(raw[0]), _number(raw[1])
        if x is not None and z is not None:
            return [x, z]
    raw3 = point.get("point_3d_original") or point.get("point_3d_normalized")
    if isinstance(raw3, list) and len(raw3) >= 3:
        x, z = _number(raw3[0]), _number(raw3[2])
        if x is not None and z is not None:
            return [x, z]
    return None


def _grid_mapping_summary(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {
            "available": False,
            "coordinate_range": None,
            "grid_points_sample": [],
            "pixel_to_world_or_world_to_pixel_notes": "No canonical grid mapping JSON was available.",
        }
    points = mapping.get("grid_points", []) if isinstance(mapping.get("grid_points"), list) else []
    xz_points = []
    samples = []
    for point in points:
        if not isinstance(point, dict):
            continue
        xz = _grid_point_xz(point)
        if xz is not None:
            xz_points.append(xz)
            if len(samples) < 8:
                samples.append({
                    "grid_i": point.get("grid_i"),
                    "grid_j": point.get("grid_j"),
                    "point_xz": xz,
                    "point_3d_original": point.get("point_3d_original"),
                    "point_3d_normalized": point.get("point_3d_normalized"),
                })
    coordinate_range = _bounds_from_points(xz_points) if xz_points else None
    return {
        "available": bool(xz_points),
        "grid_size": mapping.get("grid_size"),
        "grid_point_count": len(points),
        "coordinate_range": coordinate_range,
        "grid_points_sample": samples,
        "pixel_to_world_or_world_to_pixel_notes": (
            "Grid points are canonical top-down world [x,z] samples produced with the canonical PNG. "
            "The extent rectangle is a hint only; final walls are layout XZ coordinates. Canonical top-down/grid mapping is used for verification and visualization, not as the primary coordinate source."
        ),
        "source": mapping.get("source"),
        "fallback_used": mapping.get("fallback_used"),
    }

def build_wall_evidence(scene_state: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    evidence = scene_state.get("evidence", {})
    topdown_assets = evidence.get("topdown_assets", {}) if isinstance(evidence.get("topdown_assets"), dict) else {}
    topdown_jsons = [Path(p) for p in topdown_assets.get("json_files", []) if isinstance(p, str)]
    camera_config_path = None
    for path in topdown_jsons:
        if path.name == "topdown_camera_config.json" and path.exists():
            camera_config_path = path
            break
    topdown_mapping_state = scene_state.get("topdown_grid_mapping", {}) if isinstance(scene_state.get("topdown_grid_mapping"), dict) else {}
    mapping_path = topdown_mapping_state.get("canonical_mapping_path")
    image_path = topdown_mapping_state.get("canonical_topdown_image_path")
    mapping = _load_json(Path(mapping_path)) if mapping_path else None
    grid_summary = _grid_mapping_summary(mapping)
    camera_config = _load_json(camera_config_path)
    object_items = scene_state.get("object_candidates", {}).get("items", [])

    mapping_bounds = _bounds_from_mapping(mapping)
    camera_bounds = _bounds_from_camera_config(camera_config)
    object_bounds = _bounds_from_objects(object_items if isinstance(object_items, list) else [])

    layout_bounds = camera_bounds or object_bounds or mapping_bounds
    if layout_bounds is None:
        layout_bounds = {"x_min": None, "x_max": None, "z_min": None, "z_max": None, "source": None}
        warnings.append("Could not infer layout bounds hints from topdown camera config, object bbox, or grid mapping.")

    polygon_bounds = mapping_bounds or camera_bounds or object_bounds
    candidate_walls = _clockwise_walls_from_bounds(polygon_bounds)
    if candidate_walls:
        for wall in candidate_walls:
            wall["source"] = "layout_xz_boundary_points"
            wall["topdown_verification_status"] = "needs_check"
            wall["trust_level"] = "candidate"
            wall["may_be_used_as_final_wall"] = False
        warnings.append("candidate_walls are layout XZ boundary point candidates from grid/camera/object extents; they remain candidate-only until topdown_verification=passed.")
    else:
        warnings.append("Could not derive a four-wall clockwise hint from available bounds.")
    polygon_points = [wall["first_point"] for wall in candidate_walls] if candidate_walls else []
    relation_footprint_records = _relation_footprint_records(scene_state)
    footprints, wall_attach_geometry = _object_footprints(
        object_items if isinstance(object_items, list) else [],
        candidate_walls,
        relation_footprint_records,
    )

    render_grid_wall_candidates = _extract_render_grid_wall_candidates(scene_state)
    primary_render_grid = render_grid_wall_candidates.get("primary_candidate") if isinstance(render_grid_wall_candidates, dict) else None
    primary_render_grid_walls = primary_render_grid.get("wall_segments", []) if isinstance(primary_render_grid, dict) else []
    if primary_render_grid_walls:
        render_grid_object_footprints, render_grid_wall_attach_geometry = _object_footprints(
            object_items if isinstance(object_items, list) else [],
            primary_render_grid_walls,
            relation_footprint_records,
        )
        warnings.append("Formal render-grid wall evidence uses only obb_from_mesh_footprint_min_rect.")
    else:
        render_grid_object_footprints, render_grid_wall_attach_geometry = [], []
        reason = render_grid_wall_candidates.get("reason") if isinstance(render_grid_wall_candidates, dict) else "unknown"
        warnings.append(f"No usable formal render-grid primary wall candidate was loaded: {reason}")

    active_candidate_walls = primary_render_grid_walls
    active_polygon_points = (
        primary_render_grid.get("polygon_xz") or [wall.get("first_point") for wall in primary_render_grid_walls]
        if isinstance(primary_render_grid, dict)
        else []
    )
    active_polygon_source = (
        primary_render_grid_walls[0].get("source", "render_grid_generated_primary_wall_candidate")
        if primary_render_grid_walls
        else None
    )
    active_polygon_trust_level = "candidate" if primary_render_grid_walls else "missing"
    active_footprints = render_grid_object_footprints
    active_wall_attach_geometry = render_grid_wall_attach_geometry
    non_authoritative_wall_hints = {
        "reason": "Non-authoritative layout/grid/bbox wall hints are retained for audit only; formal tasks require render-grid OBB walls.",
        "candidate_walls_count": len(candidate_walls),
        "wall_attach_geometry_count": len(wall_attach_geometry),
        "candidate_room_polygon_hint": {
            "available": bool(polygon_points),
            "points_xz": polygon_points,
            "source": polygon_bounds.get("source") if isinstance(polygon_bounds, dict) else None,
            "trust_level": "hint_only",
            "may_be_used_as_final_walls": False,
        },
    }
    wall_candidate_mode = "render_grid_only"

    fixed_views = evidence.get("fixed_layout_views") if isinstance(evidence.get("fixed_layout_views"), dict) else {}
    return {
        "scene_name": scene_state.get("scene_name"),
        "layout_xz_evidence": {
            "source_candidates": ["layout_for_relation", "layout_params", "object_footprints", "topdown_camera_config"],
            "coordinate_system": "layout_xz",
            "notes": ["Final wall coordinates should be expressed in OpenGL original XZ and generated by obb_from_mesh_footprint_min_rect."],
        },
        "topdown_verification": {
            "canonical_topdown_image_path": image_path,
            "canonical_grid_mapping_path": mapping_path,
            "topdown_camera_config_path": str(camera_config_path) if camera_config_path else None,
            "render_grid_affine_path": render_grid_wall_candidates.get("render_grid_affine_path") if isinstance(render_grid_wall_candidates, dict) else None,
            "usage": "verification_only",
            "source": topdown_mapping_state.get("source"),
            "fallback_used": topdown_mapping_state.get("fallback_used"),
            "notes": [
                "Canonical top-down/grid mapping helps verify and visualize layout XZ wall coordinates, but is not the primary source of final wall coordinates."
            ],
        },
        "fixed_layout_views_verification": {
            "manifest_path": fixed_views.get("manifest_path"),
            "usage": "auxiliary_visual_verification_only",
            "views": [item.get("view") for item in fixed_views.get("views", []) if isinstance(item, dict)],
            "notes": [
                "Fixed layout views can support wall visibility, wall-id consistency, wall attach reasoning, ceiling/floor evidence, object placement checks, and description revision, but they are not primary wall coordinate sources."
            ],
        },
        "topdown": {
            "canonical_topdown_image_path": image_path,
            "canonical_grid_mapping_path": mapping_path,
            "usage": "verification_only",
            "source": topdown_mapping_state.get("source"),
            "fallback_used": topdown_mapping_state.get("fallback_used"),
            "topdown_camera_config_path": str(camera_config_path) if camera_config_path else None,
            "render_grid_affine_path": render_grid_wall_candidates.get("render_grid_affine_path") if isinstance(render_grid_wall_candidates, dict) else None,
            "topdown_assets": topdown_assets.get("png_files", []) + topdown_assets.get("json_files", []),
        },
        "canonical_topdown": {
            "image_path": image_path,
            "grid_mapping_path": mapping_path,
            "usage": "verification_only",
            "source": topdown_mapping_state.get("source"),
            "fallback_used": topdown_mapping_state.get("fallback_used"),
        },
        "grid_mapping_summary": grid_summary,
        "visual_wall_extraction": {
            "source": "obb_from_mesh_footprint_min_rect",
            "status": "formal_obb_wall_coordinates_required",
            "canonical_topdown_image_path": image_path,
            "canonical_grid_mapping_path": mapping_path,
            "fallback_used": topdown_mapping_state.get("fallback_used"),
            "note": "Final wall coordinates are generated by obb_from_mesh_footprint_min_rect in OpenGL XZ. Top-down/grid/fixed views verify and audit those coordinates; they are not alternate wall coordinate sources.",
        },
        "layout_bounds": layout_bounds,
        "grid_mapping_bounds": mapping_bounds,
        "camera_config_bounds": camera_bounds,
        "object_bbox_bounds": object_bounds,
        "wall_candidate_mode": wall_candidate_mode,
        "active_wall_candidate_source": active_polygon_source,
        "non_authoritative_wall_hints": non_authoritative_wall_hints,
        "object_footprints": active_footprints,
        "wall_attach_geometry": active_wall_attach_geometry,
        "render_grid_wall_candidates": render_grid_wall_candidates,
        "render_grid_object_footprints": render_grid_object_footprints,
        "render_grid_wall_attach_geometry": render_grid_wall_attach_geometry,
        "candidate_room_polygon_hint": {
            "available": bool(active_polygon_points),
            "points_xz": active_polygon_points,
            "source": active_polygon_source,
            "trust_level": active_polygon_trust_level,
            "may_be_used_as_final_walls": False,
        },
        "candidate_walls": active_candidate_walls,
        "hint_only_sources": {
            "candidate_room_polygon_hint": {
                "available": bool(active_polygon_points),
                "points_xz": active_polygon_points,
                "source": active_polygon_source,
                "trust_level": active_polygon_trust_level,
                "may_be_used_as_final_walls": False,
            },
            "candidate_walls": active_candidate_walls,
            "render_grid_wall_candidates": render_grid_wall_candidates,
            "layout_bounds": layout_bounds,
            "object_bbox_bounds": object_bounds,
            "may_be_used_as_final_walls": False,
        },
        "wall_canonical_naming_policy": canonical_wall_naming_policy(),
        "wall_output_requirements": {
            "wall_canonical_naming_policy": canonical_wall_naming_policy(),
            "required_wall_ids": WALL_IDS,
            "coordinate_format": "[x, z]",
            "clockwise_required": True,
            "fields": ["id", "first_point", "second_point"],
            "adjacent_endpoint_consistency_required": True,
            "required_fields_for_task_output": ["id", "first_point", "second_point", "source", "evidence", "topdown_verification"],
            "final_eligible_sources": [FORMAL_WALL_METHOD],
            "intermediate_sources": ["wall_attached_object_distribution", "agent_visibility_review"],
            "formal_authoritative_wall_method": FORMAL_WALL_METHOD,
            "insufficient_evidence_status": "missing_formal_render_grid_wall_evidence",
            "hint_only_sources_cannot_be_final_source": True,
        },
        "warnings": warnings,
    }


def write_wall_evidence(wall_evidence: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "wall_evidence.json", wall_evidence)
    lines = [
        f"- scene_name: `{wall_evidence.get('scene_name')}`",
        "- coordinate_source_definition: `final wall coordinates come from the formal OBB mesh-footprint method in OpenGL XZ; visual assets provide verification`",
        f"- layout_xz_coordinate_system: `{wall_evidence.get('layout_xz_evidence', {}).get('coordinate_system')}`",
        f"- canonical_topdown_image_path: `{wall_evidence.get('topdown_verification', {}).get('canonical_topdown_image_path')}`",
        f"- canonical_grid_mapping_path: `{wall_evidence.get('topdown_verification', {}).get('canonical_grid_mapping_path')}`",
        f"- topdown_usage: `{wall_evidence.get('topdown_verification', {}).get('usage')}`",
        f"- fixed_views_manifest: `{wall_evidence.get('fixed_layout_views_verification', {}).get('manifest_path')}`",
        f"- fixed_views_usage: `{wall_evidence.get('fixed_layout_views_verification', {}).get('usage')}`",
        f"- topdown_camera_config_path: `{wall_evidence.get('topdown_verification', {}).get('topdown_camera_config_path')}`",
        f"- canonical_source: `{wall_evidence.get('topdown_verification', {}).get('source')}`",
        f"- fallback_used: `{wall_evidence.get('topdown_verification', {}).get('fallback_used')}`",
        f"- visual_wall_extraction_status: `{wall_evidence.get('visual_wall_extraction', {}).get('status')}`",
        f"- wall_candidate_mode: `{wall_evidence.get('wall_candidate_mode')}`",
        f"- active_wall_candidate_source: `{wall_evidence.get('active_wall_candidate_source')}`",
        f"- non_authoritative_wall_hints: `{bool(wall_evidence.get('non_authoritative_wall_hints'))}`",
        f"- grid_mapping_available: `{wall_evidence.get('grid_mapping_summary', {}).get('available')}`",
        f"- grid_mapping_coordinate_range: `{wall_evidence.get('grid_mapping_summary', {}).get('coordinate_range')}`",
        f"- candidate_polygon_source: `{wall_evidence.get('candidate_room_polygon_hint', {}).get('source')}`",
        f"- candidate_polygon_trust: `{wall_evidence.get('candidate_room_polygon_hint', {}).get('trust_level')}`",
        f"- candidate_may_be_final: `{wall_evidence.get('candidate_room_polygon_hint', {}).get('may_be_used_as_final_walls')}`",
        f"- candidate_walls_count: `{len(wall_evidence.get('candidate_walls', []))}`",
        f"- object_footprints_count: `{len(wall_evidence.get('object_footprints', []))}`",
        f"- wall_attach_geometry_count: `{len(wall_evidence.get('wall_attach_geometry', []))}`",
        f"- wall_canonical_naming_policy: `{wall_evidence.get('wall_canonical_naming_policy', {}).get('schema_version')}`",
        "- wall_001_final_rule: `3 visible walls -> middle visible wall; 2 visible walls -> left visible wall; remaining walls clockwise`",
        f"- render_grid_status: `{wall_evidence.get('render_grid_wall_candidates', {}).get('status')}`",
        f"- render_grid_source_path: `{wall_evidence.get('render_grid_wall_candidates', {}).get('source_path')}`",
        f"- render_grid_primary_method: `{wall_evidence.get('render_grid_wall_candidates', {}).get('method_policy', {}).get('primary_method')}`",
        f"- render_grid_attach_geometry_count: `{len(wall_evidence.get('render_grid_wall_attach_geometry', []))}`",
        "",
        "## Wall Canonical Naming Policy",
        "```json",
        json.dumps(wall_evidence.get("wall_canonical_naming_policy", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Candidate Wall Hints",
        "```json",
        json.dumps(wall_evidence.get("candidate_walls", []), indent=2, ensure_ascii=False),
        "```",
        "",
        "## External A-Line Wall Candidates",
        "```json",
        json.dumps(wall_evidence.get("render_grid_wall_candidates", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Warnings",
    ]
    warnings = wall_evidence.get("warnings") or []
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    write_md(output_dir / "wall_evidence.md", "Wall Evidence", lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured wall evidence for multi-agent scene delivery.")
    parser.add_argument("--scene_state", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    scene_state = json.loads(args.scene_state.read_text(encoding="utf-8"))
    wall_evidence = build_wall_evidence(scene_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_wall_evidence(wall_evidence, args.output_dir)
    print(json.dumps({"wall_evidence_path": str(args.output_dir / "wall_evidence.json"), "candidate_walls": len(wall_evidence.get("candidate_walls", [])), "warnings": wall_evidence.get("warnings", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
