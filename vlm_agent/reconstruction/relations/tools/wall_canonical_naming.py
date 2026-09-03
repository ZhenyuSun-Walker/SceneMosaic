from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

WALL_IDS = ["wall_001", "wall_002", "wall_003", "wall_004"]

CANONICAL_WALL_NAMING_POLICY: dict[str, Any] = {
    "schema_version": "wall_canonical_naming",
    "final_policy": {
        "if_original_image_shows_three_walls": "wall_001 is the middle visible wall; wall_002..wall_004 continue clockwise in top-down XZ order.",
        "if_original_image_shows_two_walls": "wall_001 is the left visible wall; wall_002..wall_004 continue clockwise in top-down XZ order.",
        "remaining_walls": "Assign the remaining room-closure walls clockwise in top-down XZ order from wall_001.",
        "weak_hidden_wall_rule": "Weak/hidden/inferred walls may close the room geometry, but they must be marked weak/hidden and are not valid attach anchors unless the actual wall surface/contact is visible.",
    },
    "candidate_fallback_policy": {
        "purpose": "Used only for local candidate evidence before the vision agent decides original-image visibility.",
        "rule": "When original-image visibility is unknown, name the leftmost top-down segment wall_001, then continue clockwise. This is provisional and may be renamed by final wall adjudication.",
    },
    "required_final_audit_fields": [
        "original_visible_wall_count",
        "wall_001_anchor_reason",
        "visible_role",
        "original_image_visibility",
        "clockwise_order_index",
        "weak_wall",
    ],
}


def canonical_wall_naming_policy() -> dict[str, Any]:
    return deepcopy(CANONICAL_WALL_NAMING_POLICY)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _clean_points(points: Sequence[Any]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            return []
        x = _number(point[0])
        z = _number(point[1])
        if x is None or z is None:
            return []
        if cleaned and abs(cleaned[-1][0] - x) <= 1e-12 and abs(cleaned[-1][1] - z) <= 1e-12:
            continue
        cleaned.append([x, z])
    if len(cleaned) >= 2 and abs(cleaned[0][0] - cleaned[-1][0]) <= 1e-12 and abs(cleaned[0][1] - cleaned[-1][1]) <= 1e-12:
        cleaned.pop()
    return cleaned


def _centroid(points: Sequence[Sequence[float]]) -> list[float]:
    return [sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)]


def _bounds(points: Sequence[Sequence[float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    return {"x_min": min(xs), "x_max": max(xs), "z_min": min(zs), "z_max": max(zs)}


def classify_segment_geometry_role(segment: dict[str, Any], center: Sequence[float], bounds: dict[str, float]) -> str:
    first = segment.get("first_point")
    second = segment.get("second_point")
    if not (isinstance(first, list) and isinstance(second, list) and len(first) == 2 and len(second) == 2):
        return "unknown"
    mx = (float(first[0]) + float(second[0])) / 2.0
    mz = (float(first[1]) + float(second[1])) / 2.0
    span_x = max(abs(bounds["x_max"] - bounds["x_min"]), 1e-9)
    span_z = max(abs(bounds["z_max"] - bounds["z_min"]), 1e-9)
    dx = (mx - float(center[0])) / span_x
    dz = (mz - float(center[1])) / span_z
    if abs(dx) >= abs(dz):
        return "left" if dx < 0 else "right"
    return "back" if dz < 0 else "front"


def _segment_midpoint(segment: dict[str, Any]) -> list[float]:
    first = segment["first_point"]
    second = segment["second_point"]
    return [(float(first[0]) + float(second[0])) / 2.0, (float(first[1]) + float(second[1])) / 2.0]


def _valid_affine(affine: Any) -> list[list[float]] | None:
    if not (isinstance(affine, Sequence) and len(affine) == 3):
        return None
    rows: list[list[float]] = []
    for row in affine:
        if not (isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) >= 2):
            return None
        x = _number(row[0])
        y = _number(row[1])
        if x is None or y is None:
            return None
        rows.append([x, y])
    return rows


def xz_to_screen_xy(point: Sequence[float], affine: Any = None) -> list[float]:
    """Project an OpenGL XZ point to top-down screen coordinates.

    When the render-grid affine is available we use it, so wall naming follows
    the exact image shown to humans/agents. The fallback is only for
    calls that lack a render grid; it remains deterministic but is less
    semantically rich.
    """
    x = float(point[0])
    z = float(point[1])
    matrix = _valid_affine(affine)
    if matrix:
        return [
            matrix[0][0] * x + matrix[1][0] * z + matrix[2][0],
            matrix[0][1] * x + matrix[1][1] * z + matrix[2][1],
        ]
    return [x, -z]


def _signed_area_xy(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        total += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return total / 2.0


def _rotate(items: list[Any], start_idx: int) -> list[Any]:
    if not items:
        return items
    return items[start_idx:] + items[:start_idx]


def canonical_screen_clockwise_segments_from_polygon(
    points: Sequence[Any],
    *,
    xz_to_pixel_affine: Any = None,
    source: str,
    method: str,
    role: str,
    trust_level: str = "candidate",
    may_be_used_as_final_wall: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one deterministic wall-id order for a polygon.

    This is the single local naming rule used by precompute, wall evidence, and
    final assembly. It first orients the polygon clockwise in rendered top-down
    screen coordinates, then chooses the first wall by a stable screen midpoint
    sort. Model-provided visible-role metadata is audit-only and must not rotate
    the final wall ids.
    """
    cleaned = _clean_points(points)
    if len(cleaned) < 3:
        return [], {"status": "invalid", "reason": "not_enough_polygon_points"}
    affine = _valid_affine(xz_to_pixel_affine)
    screen_points = [xz_to_screen_xy(point, affine) for point in cleaned]
    screen_area_before = _signed_area_xy(screen_points)
    clockwise_points = list(cleaned)
    if screen_area_before < 0:
        clockwise_points = list(reversed(clockwise_points))
        screen_points = [xz_to_screen_xy(point, affine) for point in clockwise_points]
    screen_area_after = _signed_area_xy(screen_points)
    center = _centroid(clockwise_points)
    bounds = _bounds(clockwise_points)
    raw_segments: list[dict[str, Any]] = []
    for idx, point in enumerate(clockwise_points):
        nxt = clockwise_points[(idx + 1) % len(clockwise_points)]
        first_screen = xz_to_screen_xy(point, affine)
        second_screen = xz_to_screen_xy(nxt, affine)
        raw_segments.append(
            {
                "provisional_original_index": idx,
                "first_point": [round(float(point[0]), 6), round(float(point[1]), 6)],
                "second_point": [round(float(nxt[0]), 6), round(float(nxt[1]), 6)],
                "source": source,
                "source_method": method,
                "role": role,
                "trust_level": trust_level,
                "topdown_verification_status": "needs_check",
                "may_be_used_as_final_wall": may_be_used_as_final_wall,
                "midpoint_screen_xy": [
                    (first_screen[0] + second_screen[0]) / 2.0,
                    (first_screen[1] + second_screen[1]) / 2.0,
                ],
            }
        )
    for segment in raw_segments:
        segment["geometry_role"] = classify_segment_geometry_role(segment, center, bounds)
        segment["midpoint_xz"] = _segment_midpoint(segment)
    start_idx = min(
        range(len(raw_segments)),
        key=lambda idx: (
            raw_segments[idx]["midpoint_screen_xy"][1],
            raw_segments[idx]["midpoint_screen_xy"][0],
            raw_segments[idx]["midpoint_xz"][0],
            raw_segments[idx]["midpoint_xz"][1],
        ),
    )
    ordered = _rotate(raw_segments, start_idx)
    audit = {
        "schema_version": "deterministic_screen_clockwise_wall_naming",
        "status": "available",
        "screen_coordinate_source": "render_grid_affine.xz_to_pixel_affine" if affine else "fallback_x_neg_z",
        "screen_signed_area_before": screen_area_before,
        "screen_signed_area_after": screen_area_after,
        "screen_order_interpretation": "clockwise" if screen_area_after > 0 else "counter_clockwise_or_degenerate",
        "start_segment_policy": "topmost_then_leftmost_screen_midpoint",
        "model_visible_role_can_rotate_ids": False,
    }
    for idx, segment in enumerate(ordered):
        segment["geometry_wall_id"] = f"geom_wall_{idx + 1:03d}"
        segment["id"] = WALL_IDS[idx] if idx < len(WALL_IDS) else f"wall_{idx + 1:03d}"
        segment["provisional_wall_id"] = segment["id"]
        segment["clockwise_order_index"] = idx + 1
        segment["screen_clockwise_order_index"] = idx + 1
        segment["canonical_naming"] = {
            "schema_version": CANONICAL_WALL_NAMING_POLICY["schema_version"],
            "status": "deterministic_screen_clockwise_geometry",
            "candidate_fallback_rule": "topmost_then_leftmost_screen_midpoint, then continue screen-clockwise",
            "final_policy_required": CANONICAL_WALL_NAMING_POLICY["final_policy"],
            "wall_001_anchor_reason": "deterministic topmost/leftmost rendered top-down segment; original-image visibility is audit metadata only",
            "geometry_role": segment["geometry_role"],
            "original_image_visibility": "unknown_until_final_adjudication",
            "must_be_confirmed_or_renamed_by_final_adjudication": False,
            "screen_coordinate_source": audit["screen_coordinate_source"],
            "start_segment_policy": audit["start_segment_policy"],
        }
    return ordered, audit


def provisional_segments_from_polygon(
    points: Sequence[Any],
    *,
    xz_to_pixel_affine: Any = None,
    source: str,
    method: str,
    role: str,
    trust_level: str = "candidate",
    may_be_used_as_final_wall: bool = False,
) -> list[dict[str, Any]]:
    """Return candidate wall segments with stable provisional ids."""
    ordered, _ = canonical_screen_clockwise_segments_from_polygon(
        points,
        xz_to_pixel_affine=xz_to_pixel_affine,
        source=source,
        method=method,
        role=role,
        trust_level=trust_level,
        may_be_used_as_final_wall=may_be_used_as_final_wall,
    )
    return ordered

def final_wall_naming_audit(walls: Sequence[Any]) -> dict[str, Any]:
    clean_walls = [wall for wall in walls if isinstance(wall, dict)]
    points: list[list[float]] = []
    for wall in clean_walls:
        first = wall.get("first_point")
        if isinstance(first, list) and len(first) == 2 and _number(first[0]) is not None and _number(first[1]) is not None:
            points.append([float(first[0]), float(first[1])])
    center = _centroid(points) if points else None
    bounds = _bounds(points) if len(points) >= 2 else None
    items: list[dict[str, Any]] = []
    missing_audit_fields: list[dict[str, Any]] = []
    for idx, wall in enumerate(clean_walls):
        item = {
            "id": wall.get("id"),
            "clockwise_order_index": idx + 1,
            "original_visible_wall_count": wall.get("original_visible_wall_count") or wall.get("canonical_naming", {}).get("original_visible_wall_count"),
            "visible_role": wall.get("visible_role") or wall.get("canonical_naming", {}).get("visible_role"),
            "original_image_visibility": wall.get("original_image_visibility") or wall.get("visibility") or wall.get("canonical_naming", {}).get("original_image_visibility"),
            "weak_wall": bool(wall.get("weak_wall")),
        }
        if center and bounds:
            item["geometry_role_fallback"] = classify_segment_geometry_role(wall, center, bounds)
        missing = [field for field in CANONICAL_WALL_NAMING_POLICY["required_final_audit_fields"] if item.get(field) is None and field != "weak_wall"]
        if missing:
            missing_audit_fields.append({"id": wall.get("id"), "missing_fields": missing})
        items.append(item)
    return {
        "policy": canonical_wall_naming_policy(),
        "items": items,
        "missing_final_audit_fields": missing_audit_fields,
        "complete_final_visibility_audit": not missing_audit_fields,
        "note": "Geometry fallback roles are audit hints only; final wall ids are assigned by deterministic rendered top-down screen-clockwise geometry.",
    }
