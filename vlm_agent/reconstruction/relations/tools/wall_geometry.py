"""Production 2D wall geometry for relation preprocessing.

This module owns the formal minimum-area OBB implementation and related
polygon diagnostics. It intentionally has no dependency on exploratory
exploratory relation packages or SceneMosaic runtime modules.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from relations.tools.wall_canonical_naming import canonical_screen_clockwise_segments_from_polygon

Point = tuple[float, float]
FORMAL_WALL_METHOD = "obb_from_mesh_footprint_min_rect"
WALL_IDS = ["wall_001", "wall_002", "wall_003", "wall_004"]


def signed_area(points: Sequence[Point]) -> float:
    total = 0.0
    for index, (x1, z1) in enumerate(points):
        x2, z2 = points[(index + 1) % len(points)]
        total += x1 * z2 - x2 * z1
    return total * 0.5


def polygon_area(points: Sequence[Point]) -> float:
    return abs(signed_area(points))


def convex_hull(points: Iterable[Point]) -> list[Point]:
    unique = sorted({(round(float(x), 12), round(float(z), 12)) for x, z in points})
    if len(unique) <= 1:
        return list(unique)

    def cross(origin: Point, first: Point, second: Point) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def ensure_clockwise(points: Sequence[Point]) -> list[Point]:
    result = [(float(x), float(z)) for x, z in points]
    if signed_area(result) > 0:
        result.reverse()
    return result


def canonical_clockwise_rect(points: Sequence[Point]) -> list[Point]:
    if len(points) != 4:
        raise ValueError("canonical_clockwise_rect requires exactly four points")
    ordered = ensure_clockwise(points)
    edges = []
    for index in range(4):
        first = ordered[index]
        second = ordered[(index + 1) % 4]
        midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
        edges.append((midpoint[1], midpoint[0], -math.dist(first, second), index))
    start = min(edges)[3]
    return [ordered[(start + index) % 4] for index in range(4)]


def _rotate_point(point: Point, angle: float) -> Point:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, z = point
    return x * cosine + z * sine, -x * sine + z * cosine


def _unrotate_point(point: Point, angle: float) -> Point:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    u, v = point
    return u * cosine - v * sine, u * sine + v * cosine


def rect_from_angle(points: Sequence[Point], angle: float, margin: float = 0.0) -> list[Point]:
    if not points:
        raise ValueError("rect_from_angle requires at least one point")
    rotated = [_rotate_point(point, angle) for point in points]
    us = [point[0] for point in rotated]
    vs = [point[1] for point in rotated]
    min_u, max_u = min(us) - margin, max(us) + margin
    min_v, max_v = min(vs) - margin, max(vs) + margin
    rectangle = [
        (min_u, min_v),
        (max_u, min_v),
        (max_u, max_v),
        (min_u, max_v),
    ]
    return canonical_clockwise_rect([_unrotate_point(point, angle) for point in rectangle])


def min_area_rect(points: Sequence[Point], margin: float = 0.0) -> tuple[list[Point], float, float]:
    if not points:
        raise ValueError("min_area_rect requires at least one point")
    hull = convex_hull(points)
    if len(hull) < 3:
        rectangle = rect_from_angle(hull, 0.0, margin=margin)
        return rectangle, polygon_area(rectangle), 0.0
    angles = {
        round(math.atan2(hull[(index + 1) % len(hull)][1] - point[1], hull[(index + 1) % len(hull)][0] - point[0]) % (math.pi / 2.0), 12)
        for index, point in enumerate(hull)
    }
    best_rectangle: list[Point] | None = None
    best_area = float("inf")
    best_angle = 0.0
    for angle in sorted(angles):
        rectangle = rect_from_angle(hull, angle, margin=margin)
        area = polygon_area(rectangle)
        if area < best_area:
            best_rectangle = rectangle
            best_area = area
            best_angle = angle
    if best_rectangle is None:
        best_rectangle = rect_from_angle(hull, 0.0, margin=margin)
        best_area = polygon_area(best_rectangle)
    return best_rectangle, best_area, best_angle


def layout_wall_polygon(layout: dict[str, Any] | None) -> list[Point] | None:
    if not isinstance(layout, dict):
        return None
    points: list[Point] = []
    for wall in layout.get("walls") or []:
        if not isinstance(wall, dict):
            continue
        first = wall.get("first_point_xz") or wall.get("first_point")
        second = wall.get("second_point_xz") or wall.get("second_point")
        if first and second:
            points.extend([(float(first[0]), float(first[1])), (float(second[0]), float(second[1]))])
    if len(points) < 3:
        return None
    hull = convex_hull(points)
    return canonical_clockwise_rect(hull) if len(hull) == 4 else hull


def dominant_rect_angle_degrees(points: Sequence[Point]) -> float:
    lengths = [math.dist(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
    index = max(range(len(lengths)), key=lengths.__getitem__)
    first, second = points[index], points[(index + 1) % len(points)]
    angle = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])) % 180.0
    return angle - 90.0 if angle >= 90.0 else angle


def axis_deviation_degrees(angle_degrees: float) -> float:
    value = abs(angle_degrees) % 90.0
    return min(value, 90.0 - value)


def _inside_halfplane(point: Point, first: Point, second: Point, clockwise: bool) -> bool:
    cross = (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])
    return cross <= 1e-9 if clockwise else cross >= -1e-9


def _line_intersection(start: Point, end: Point, first: Point, second: Point) -> Point:
    x1, y1 = start
    x2, y2 = end
    x3, y3 = first
    x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return end
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return float(px), float(py)


def _clip_intersection_area(subject: Sequence[Point], clipper: Sequence[Point], clockwise: bool) -> float:
    output = list(subject)
    for index, first in enumerate(clipper):
        second = clipper[(index + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            end_inside = _inside_halfplane(end, first, second, clockwise)
            start_inside = _inside_halfplane(start, first, second, clockwise)
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(start, end, first, second))
                output.append(end)
            elif start_inside:
                output.append(_line_intersection(start, end, first, second))
            start = end
    return polygon_area(output) if len(output) >= 3 else 0.0


def polygon_iou(first: Sequence[Point] | None, second: Sequence[Point] | None) -> float | None:
    if not first or not second:
        return None
    intersection = max(
        _clip_intersection_area(first, second, clockwise=True),
        _clip_intersection_area(first, second, clockwise=False),
    )
    union = polygon_area(first) + polygon_area(second) - intersection
    return None if union <= 1e-12 else intersection / union


def xz_to_pixel(points: Sequence[Point], matrix: np.ndarray) -> list[tuple[int, int]]:
    rows = np.asarray([[float(x), float(z), 1.0] for x, z in points], dtype=np.float64)
    projected = rows @ matrix
    return [(int(round(x)), int(round(y))) for x, y in projected]


def candidate_metrics(
    polygon_xz: Sequence[Point],
    layout_polygon: Sequence[Point] | None,
    foreground: np.ndarray,
    xz_to_pixel_matrix: np.ndarray,
) -> dict[str, Any]:
    polygon = canonical_clockwise_rect(polygon_xz) if len(polygon_xz) == 4 else list(polygon_xz)
    points = np.asarray(xz_to_pixel(polygon, xz_to_pixel_matrix), dtype=np.int32)
    polygon_mask = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [points], 255)
    foreground_total = max(1, int(np.count_nonzero(foreground)))
    foreground_inside = int(np.count_nonzero(cv2.bitwise_and(foreground, foreground, mask=polygon_mask)))
    polygon_area_pixels = max(1, int(np.count_nonzero(polygon_mask)))
    axis_deviation = axis_deviation_degrees(dominant_rect_angle_degrees(polygon)) if len(polygon) == 4 else None
    return {
        "polygon_xz": [[round(float(x), 6), round(float(z), 6)] for x, z in polygon],
        "area_xz": round(polygon_area(polygon), 6),
        "axis_deviation_degrees": None if axis_deviation is None else round(axis_deviation, 3),
        "iou_vs_layout_walls": None if layout_polygon is None else round(float(polygon_iou(polygon, layout_polygon) or 0.0), 4),
        "foreground_coverage": round(foreground_inside / foreground_total, 4),
        "foreground_fill_in_candidate": round(foreground_inside / polygon_area_pixels, 4),
        "empty_pixel_ratio_in_candidate": round(1.0 - foreground_inside / polygon_area_pixels, 4),
    }


def build_obb_walls(method_candidates: dict[str, Any], render_grid_affine: dict[str, Any]) -> dict[str, Any]:
    candidates = method_candidates.get("candidates", []) if isinstance(method_candidates.get("candidates"), list) else []
    primary = next((item for item in candidates if isinstance(item, dict) and item.get("method") == FORMAL_WALL_METHOD), None)
    if not primary:
        return {"status": "missing", "method": FORMAL_WALL_METHOD, "walls": [], "reason": "formal candidate missing"}
    polygon = primary.get("metrics", {}).get("polygon_xz")
    if not isinstance(polygon, list) or len(polygon) < 4:
        return {"status": "missing", "method": FORMAL_WALL_METHOD, "walls": [], "reason": "polygon_xz missing"}
    segments, naming_audit = canonical_screen_clockwise_segments_from_polygon(
        polygon[:4],
        xz_to_pixel_affine=render_grid_affine.get("xz_to_pixel_affine"),
        source=FORMAL_WALL_METHOD,
        method=FORMAL_WALL_METHOD,
        role="formal_obb_wall",
        trust_level="final_coordinate_source",
        may_be_used_as_final_wall=True,
    )
    if len(segments) != 4:
        return {"status": "invalid", "method": FORMAL_WALL_METHOD, "walls": [], "reason": "canonical segment generation failed", "canonical_naming_audit": naming_audit}
    walls = []
    for index, segment in enumerate(segments):
        walls.append({
            "id": WALL_IDS[index],
            "type": "wall",
            "first_point_xz": segment["first_point"],
            "second_point_xz": segment["second_point"],
            "source": FORMAL_WALL_METHOD,
            "coordinate_frame": "OpenGL original XZ; Y is vertical",
            "screen_clockwise_order_index": index + 1,
            "geometry_wall_id": segment.get("geometry_wall_id"),
            "midpoint_screen_xy": segment.get("midpoint_screen_xy"),
            "canonical_naming": segment.get("canonical_naming"),
        })
    return {
        "status": "available",
        "schema_version": "obb_walls",
        "method": FORMAL_WALL_METHOD,
        "coordinate_frame": "OpenGL original XZ; Y is vertical",
        "screen_coordinate_source": naming_audit.get("screen_coordinate_source"),
        "screen_signed_area_before": naming_audit.get("screen_signed_area_before"),
        "screen_signed_area_after": naming_audit.get("screen_signed_area_after"),
        "screen_order_interpretation": naming_audit.get("screen_order_interpretation"),
        "start_segment_policy": naming_audit.get("start_segment_policy"),
        "model_visible_role_can_rotate_ids": False,
        "walls": walls,
        "primary_candidate": primary,
        "canonical_naming_audit": naming_audit,
    }
