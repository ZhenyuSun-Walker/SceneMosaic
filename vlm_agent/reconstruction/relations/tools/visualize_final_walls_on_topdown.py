from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.schema_guard import WALL_IDS, validate_wall_geometry, wall_axis_alignment_summary
from relations.agents.state import write_json, write_md


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mapping_bounds(mapping: dict[str, Any]) -> dict[str, float]:
    points: list[list[float]] = []
    for point in mapping.get("grid_points", []) if isinstance(mapping.get("grid_points"), list) else []:
        if not isinstance(point, dict):
            continue
        raw = point.get("point_3d_original_xz")
        if isinstance(raw, list) and len(raw) == 2:
            x, z = _num(raw[0]), _num(raw[1])
            if x is not None and z is not None:
                points.append([x, z])
                continue
        raw3 = point.get("point_3d_original") or point.get("point_3d_normalized")
        if isinstance(raw3, list) and len(raw3) >= 3:
            x, z = _num(raw3[0]), _num(raw3[2])
            if x is not None and z is not None:
                points.append([x, z])
    if not points:
        raise ValueError("grid mapping does not contain usable world x/z grid points")
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    if max(xs) - min(xs) <= 1e-12 or max(zs) - min(zs) <= 1e-12:
        raise ValueError("grid mapping bounds are degenerate")
    return {"x_min": min(xs), "x_max": max(xs), "z_min": min(zs), "z_max": max(zs)}


def _world_to_pixel(point: list[float], bounds: dict[str, float], width: int, height: int) -> list[int]:
    x, z = float(point[0]), float(point[1])
    px = round((x - bounds["x_min"]) / (bounds["x_max"] - bounds["x_min"]) * (width - 1))
    py = round((bounds["z_max"] - z) / (bounds["z_max"] - bounds["z_min"]) * (height - 1))
    return [int(px), int(py)]


def _load_xz_to_pixel_affine(grid_mapping: Path) -> list[list[float]] | None:
    candidates = [grid_mapping.with_name("render_grid_affine.json")]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        matrix = data.get("xz_to_pixel_affine") if isinstance(data, dict) else None
        if (
            isinstance(matrix, list)
            and len(matrix) == 3
            and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        ):
            return [[float(value) for value in row] for row in matrix]
    return None


def _world_to_pixel_exact(point: list[float], affine: list[list[float]]) -> list[int]:
    x, z = float(point[0]), float(point[1])
    px = x * affine[0][0] + z * affine[1][0] + affine[2][0]
    py = x * affine[0][1] + z * affine[1][1] + affine[2][1]
    return [int(round(px)), int(round(py))]


def _signed_area(points: list[list[float]]) -> float:
    area2 = 0.0
    for idx, p in enumerate(points):
        q = points[(idx + 1) % len(points)]
        area2 += p[0] * q[1] - q[0] * p[1]
    return area2 / 2.0


def _screen_y_down_polygon_order(points: list[list[float]]) -> dict[str, Any]:
    if len(points) < 3:
        return {"signed_area": None, "interpreted_order": "unknown", "screen_clockwise": False}
    signed_area = _signed_area(points)
    if abs(signed_area) <= 1e-9:
        interpreted_order = "degenerate_or_too_small"
    else:
        interpreted_order = "clockwise" if signed_area > 0 else "counter_clockwise"
    return {
        "signed_area": signed_area,
        "interpreted_order": interpreted_order,
        "screen_clockwise": interpreted_order == "clockwise",
    }


def _walls_from_objects(objects_payload: dict[str, Any]) -> list[dict[str, Any]]:
    walls = objects_payload.get("walls")
    if not isinstance(walls, list):
        return []
    by_id = {wall.get("id"): wall for wall in walls if isinstance(wall, dict)}
    return [by_id[wall_id] for wall_id in WALL_IDS if wall_id in by_id]


def _wall_revision_summary(output: Path, candidate_modified: bool | None = None, revision_reason: str | None = None) -> dict[str, Any]:
    wall_audit_path = output.with_name("wall_audit.json")
    summary = {
        "candidate_modified": bool(candidate_modified),
        "revision_used": bool(candidate_modified),
        "revision_reasons": [revision_reason] if revision_reason else [],
        "weak_wall_completion": {"used": False, "weak_wall_ids": [], "completion_methods": [], "completion_reasons": []},
    }
    if not wall_audit_path.exists():
        return summary
    try:
        wall_audit = _load_json(wall_audit_path)
    except Exception:
        return summary
    promotion = wall_audit.get("candidate_wall_promotion") if isinstance(wall_audit, dict) else None
    if isinstance(promotion, dict):
        summary["candidate_modified"] = bool(promotion.get("candidate_modified"))
        summary["revision_used"] = bool(promotion.get("revision_used") or promotion.get("candidate_modified"))
        summary["revision_reasons"] = promotion.get("revision_reasons", [])
        summary["candidate_wall_promotion"] = promotion
    completion = wall_audit.get("weak_wall_completion") if isinstance(wall_audit, dict) else None
    if isinstance(completion, dict):
        summary["weak_wall_completion"] = completion
    return summary


def visualize(topdown_png: Path, grid_mapping: Path, objects_json: Path, output: Path, candidate_modified: bool | None = None, revision_reason: str | None = None) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(topdown_png))
    if image is None:
        raise ValueError(f"failed to read topdown png: {topdown_png}")
    height, width = image.shape[:2]
    mapping = _load_json(grid_mapping)
    bounds = _mapping_bounds(mapping)
    affine = _load_xz_to_pixel_affine(grid_mapping)
    projection_method = "exact_render_grid_affine" if affine else "approximate_grid_bounds"
    objects_payload = _load_json(objects_json)
    walls = _walls_from_objects(objects_payload)
    geometry = validate_wall_geometry(walls)
    axis_summary = wall_axis_alignment_summary(walls)

    colors = {
        "wall_001": (0, 80, 255),
        "wall_002": (0, 200, 0),
        "wall_003": (255, 80, 0),
        "wall_004": (210, 0, 210),
    }
    wall_reports = []
    for wall in walls:
        wall_id = wall.get("id")
        first = wall.get("first_point")
        second = wall.get("second_point")
        if not (isinstance(first, list) and isinstance(second, list) and len(first) == 2 and len(second) == 2):
            wall_reports.append({"id": wall_id, "error": "invalid wall points"})
            continue
        if affine:
            p0 = _world_to_pixel_exact(first, affine)
            p1 = _world_to_pixel_exact(second, affine)
        else:
            p0 = _world_to_pixel(first, bounds, width, height)
            p1 = _world_to_pixel(second, bounds, width, height)
        in_range = all(0 <= value < limit for value, limit in [(p0[0], width), (p1[0], width), (p0[1], height), (p1[1], height)])
        color = colors.get(wall_id, (255, 255, 255))
        cv2.arrowedLine(image, tuple(p0), tuple(p1), color, 4, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(image, tuple(p0), 7, color, -1, cv2.LINE_AA)
        cv2.circle(image, tuple(p1), 5, color, 2, cv2.LINE_AA)
        cv2.putText(image, "first", (p0[0] + 5, p0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(image, "second", (p1[0] + 5, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        mid = [round((p0[0] + p1[0]) / 2), round((p0[1] + p1[1]) / 2)]
        cv2.putText(image, f"{wall_id} first->second", tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        wall_reports.append({
            "id": wall_id,
            "first_point_xz": first,
            "second_point_xz": second,
            "first_point_px": p0,
            "second_point_px": p1,
            "endpoints_in_image_range": in_range,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    screen_vertices = [item.get("first_point_px") for item in wall_reports if isinstance(item.get("first_point_px"), list)]
    screen_order = _screen_y_down_polygon_order(screen_vertices)
    screen_closed = (
        len(wall_reports) == 4
        and all(
            wall_reports[index].get("second_point_px") == wall_reports[(index + 1) % len(wall_reports)].get("first_point_px")
            for index in range(len(wall_reports))
        )
    )
    first_vertices = [wall.get("first_point") for wall in walls if isinstance(wall.get("first_point"), list)]
    signed_area = _signed_area(first_vertices) if len(first_vertices) == 4 else None
    interpreted = "clockwise" if signed_area is not None and signed_area < 0 else "counter_clockwise" if signed_area is not None and signed_area > 0 else "unknown"
    revision_summary = _wall_revision_summary(output, candidate_modified=candidate_modified, revision_reason=revision_reason)
    weak_completion = revision_summary.get("weak_wall_completion", {}) if isinstance(revision_summary.get("weak_wall_completion"), dict) else {}
    report = {
        "topdown_png": str(topdown_png),
        "grid_mapping": str(grid_mapping),
        "objects_json": str(objects_json),
        "output_overlay_png": str(output),
        "coordinate_source": "obb_from_mesh_footprint_min_rect_xz",
        "topdown_usage": "visual_audit_only",
        "fixed_layout_views_usage": "auxiliary_visual_audit_only",
        "candidate_modified": revision_summary.get("candidate_modified", False),
        "revision_used": revision_summary.get("revision_used", False),
        "revision_reasons": revision_summary.get("revision_reasons", []),
        "image_size": [width, height],
        "mapping_bounds_xz": bounds,
        "projection_method": projection_method,
        "render_grid_affine_used": bool(affine),
        "screen_signed_area_y_down": screen_order["signed_area"],
        "screen_interpreted_order": screen_order["interpreted_order"],
        "screen_clockwise": screen_order["screen_clockwise"],
        "screen_closed": screen_closed,
        "walls": wall_reports,
        "all_endpoints_in_image_range": all(item.get("endpoints_in_image_range") for item in wall_reports),
        "geometry": geometry,
        "axis_aligned": axis_summary.get("axis_aligned"),
        "oriented_boundary": axis_summary.get("oriented_boundary"),
        "edge_orientations": axis_summary.get("edge_orientations", []),
        "weak_wall_completion_used": bool(weak_completion.get("used")),
        "weak_wall_ids": weak_completion.get("weak_wall_ids", []),
        "weak_wall_completion": weak_completion,
        "orientation_note": "Final walls are drawn with first->second arrows; coordinates are expected to come from obb_from_mesh_footprint_min_rect after final assembly.",
        "signed_area": signed_area,
        "interpreted_order": interpreted,
        "note": "This tool projects final OBB XZ wall coordinates onto the canonical top-down image for visual audit. It does not make the top-down image the primary coordinate source.",
    }
    if revision_summary.get("candidate_wall_promotion"):
        report["candidate_wall_promotion"] = revision_summary["candidate_wall_promotion"]
    write_json(output.with_name("final_walls_overlay_check.json"), report)
    lines = [
        f"- topdown_png: `{topdown_png}`",
        f"- grid_mapping: `{grid_mapping}`",
        f"- objects_json: `{objects_json}`",
        f"- output_overlay_png: `{output}`",
        "- coordinate_source: `obb_from_mesh_footprint_min_rect_xz`",
        "- topdown_usage: `visual_audit_only`",
        "- fixed_layout_views_usage: `auxiliary_visual_audit_only`",
        f"- candidate_modified: `{report['candidate_modified']}`",
        f"- revision_used: `{report['revision_used']}`",
        f"- revision_reasons: `{report['revision_reasons']}`",
        f"- all_endpoints_in_image_range: `{report['all_endpoints_in_image_range']}`",
        f"- projection_method: `{report['projection_method']}`",
        f"- render_grid_affine_used: `{report['render_grid_affine_used']}`",
        f"- screen_signed_area_y_down: `{report['screen_signed_area_y_down']}`",
        f"- screen_interpreted_order: `{report['screen_interpreted_order']}`",
        f"- screen_clockwise: `{report['screen_clockwise']}`",
        f"- screen_closed: `{report['screen_closed']}`",
        f"- axis_aligned: `{report['axis_aligned']}`",
        f"- oriented_boundary: `{report['oriented_boundary']}`",
        f"- weak_wall_completion_used: `{report['weak_wall_completion_used']}`",
        f"- weak_wall_ids: `{report['weak_wall_ids']}`",
        f"- signed_area: `{signed_area}`",
        f"- interpreted_order: `{interpreted}`",
        "",
        "## Walls",
    ]
    for item in wall_reports:
        lines.append(f"- `{item.get('id')}` xz {item.get('first_point_xz')} -> {item.get('second_point_xz')}; px {item.get('first_point_px')} -> {item.get('second_point_px')}; in_range `{item.get('endpoints_in_image_range')}`")
    if report.get("oriented_boundary"):
        lines.extend(["", "## Oriented Boundary", "- Final walls form an oriented quadrilateral and are not required to be axis-aligned."])
    if report.get("weak_wall_completion_used"):
        lines.extend(["", "## Hidden / Weak Wall Completion", "- Some walls were inferred by room closure and should be visually inspected."])
    if report.get("candidate_modified"):
        lines.extend(["", "## Revision Notice", "- These walls were revised during wall_final_adjudication; inspect visually before accepting."])
    lines.extend(["", "## Note", "- This image projects final OBB XZ wall coordinates onto canonical top-down for human visual audit. Arrows show first_point -> second_point. Top-down is not the primary coordinate source."])
    write_md(output.with_name("final_walls_overlay_check.md"), "Final Walls Overlay Check", lines)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay final wall coordinates on canonical top-down evidence.")
    parser.add_argument("--topdown_png", required=True, type=Path)
    parser.add_argument("--grid_mapping", required=True, type=Path)
    parser.add_argument("--objects_json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate_modified", action="store_true")
    parser.add_argument("--revision_reason", default=None)
    args = parser.parse_args()
    report = visualize(args.topdown_png, args.grid_mapping, args.objects_json, args.output, candidate_modified=args.candidate_modified, revision_reason=args.revision_reason)
    print(json.dumps({"output": report["output_overlay_png"], "all_endpoints_in_image_range": report["all_endpoints_in_image_range"], "axis_aligned": report["axis_aligned"], "oriented_boundary": report["oriented_boundary"], "weak_wall_completion_used": report["weak_wall_completion_used"], "weak_wall_ids": report["weak_wall_ids"], "signed_area": report["signed_area"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
