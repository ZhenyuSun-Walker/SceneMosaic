#!/usr/bin/env python3
"""Generate render-grid wall candidates inside a multi-agent run.

This is the formal OBB render-grid precompute used by relation extraction.  It writes all top-down, mask, affine, candidate, and
overlay assets into the current multi-agent run so later stages consume one
shared grid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
POLY_TOOLS = REPOSITORY_ROOT / "vlm_agent" / "evolution" / "tools"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(POLY_TOOLS) not in sys.path:
    sys.path.insert(0, str(POLY_TOOLS))

from relations.tools.wall_geometry import (  # noqa: E402
    candidate_metrics,
    canonical_clockwise_rect,
    dominant_rect_angle_degrees,
    layout_wall_polygon,
    min_area_rect,
    rect_from_angle,
    xz_to_pixel,
)
from relations.tools.wall_orientation_candidates import (  # noqa: E402
    axis_angle_distance_degrees,
    cluster_object_orientation_axes,
    normalize_axis_angle_degrees,
)
from relations.tools.wall_canonical_naming import canonical_screen_clockwise_segments_from_polygon  # noqa: E402
from render_scene_from_layout import (  # noqa: E402
    build_camera_pose,
    compose_scene,
    denormalize_point,
    generate_2d_grid_on_plane,
    get_camera_matrix,
    project_3d_to_2d,
    render_scene_with_grid,
    resolve_orthographic_camera_frame,
    save_grid_mapping,
)

from vlm_agent.run_paths import ScenePaths  # noqa: E402

Point = tuple[float, float]
# Formal wall coordinates come from posed mesh footprints projected to OpenGL
# XZ and enclosed by the minimum rotated rectangle. Alternative coordinate
# estimators are intentionally excluded from the publication path.
PRIMARY_METHOD = "obb_from_mesh_footprint_min_rect"
SUPPORTING_METHODS: list[str] = []


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save image: {path}")


def pixel_affines(
    mapping: dict[tuple[int, int], Sequence[float]],
    transform_info: dict[str, Any],
    camera_pose: np.ndarray,
    camera_matrix: Any,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    pixel_rows = []
    xz_rows = []
    samples: list[dict[str, Any]] = []
    for key, point_normalized in mapping.items():
        pixel = project_3d_to_2d(point_normalized, camera_pose, camera_matrix, image_size)
        if pixel is None:
            continue
        original = denormalize_point(point_normalized, transform_info)
        u, v = float(pixel[0]), float(pixel[1])
        x, z = float(original[0]), float(original[2])
        pixel_rows.append([u, v, 1.0])
        xz_rows.append([x, z])
        samples.append({
            "grid_i": int(key[0]),
            "grid_j": int(key[1]),
            "pixel": [u, v],
            "point_3d_original": original.tolist(),
            "point_3d_normalized": list(point_normalized),
        })
    if len(samples) < 4:
        raise RuntimeError("not enough projected grid samples for affine fit")
    pixel_to_xz, *_ = np.linalg.lstsq(np.asarray(pixel_rows), np.asarray(xz_rows), rcond=None)
    xz_rows_h = [[x, z, 1.0] for x, z in xz_rows]
    pixel_output = [[row[0], row[1]] for row in pixel_rows]
    xz_to_pixel_matrix, *_ = np.linalg.lstsq(np.asarray(xz_rows_h), np.asarray(pixel_output), rcond=None)
    return pixel_to_xz.astype(np.float64), xz_to_pixel_matrix.astype(np.float64), samples


def foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    border_h = max(4, height // 32)
    border_w = max(4, width // 32)
    corners = np.vstack([
        image_rgb[:border_h, :border_w].reshape(-1, 3),
        image_rgb[-border_h:, :border_w].reshape(-1, 3),
        image_rgb[:border_h, -border_w:].reshape(-1, 3),
        image_rgb[-border_h:, -border_w:].reshape(-1, 3),
    ])
    background = np.median(corners.astype(np.float32), axis=0)
    distance = np.linalg.norm(image_rgb.astype(np.float32) - background.reshape(1, 1, 3), axis=2)
    mask = (distance > 14.0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def draw_candidates(
    image_rgb: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    xz_to_pixel_matrix: np.ndarray,
    path: Path,
) -> None:
    colors = [(226, 35, 54), (30, 153, 139), (46, 134, 171), (255, 169, 77)]
    canvas = image_rgb.copy()
    for index, candidate in enumerate(candidates):
        polygon = [(float(x), float(z)) for x, z in candidate["metrics"]["polygon_xz"]]
        points = np.asarray(xz_to_pixel(polygon, xz_to_pixel_matrix), dtype=np.int32)
        color = colors[index % len(colors)]
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=3)
        x, y = points[0]
        label = str(candidate.get("candidate_id") or f'{index + 1}:{candidate["method"]}')
        cv2.putText(canvas, label, (int(x) + 4, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    save_rgb(path, canvas)


def _grid_to_pixel_affine(projected_samples: Sequence[dict[str, Any]]) -> np.ndarray | None:
    rows = []
    pixels = []
    for sample in projected_samples:
        if not isinstance(sample, dict):
            continue
        pixel = sample.get("pixel")
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        rows.append([float(sample["grid_i"]), float(sample["grid_j"]), 1.0])
        pixels.append([float(pixel[0]), float(pixel[1])])
    if len(rows) < 4:
        return None
    matrix, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(pixels), rcond=None)
    return matrix.astype(np.float64)


def _draw_object_direction_arrows(
    canvas: np.ndarray,
    objects_info: Sequence[dict[str, Any]],
    grid_to_pixel: np.ndarray | None,
    *,
    highlighted_object_ids: set[str] | None = None,
    highlight_color: tuple[int, int, int] = (226, 35, 54),
) -> None:
    if grid_to_pixel is None:
        return
    for item in objects_info:
        if not isinstance(item, dict):
            continue
        center = item.get("center_2d")
        size = item.get("size_2d")
        rotation = item.get("rotation_degrees")
        object_id = str(item.get("object_id") or "")
        if not object_id or not isinstance(center, list) or len(center) != 2:
            continue
        if not isinstance(size, list) or len(size) != 2 or not isinstance(rotation, (int, float)):
            continue
        center_row = np.asarray([float(center[0]), float(center[1]), 1.0]) @ grid_to_pixel
        scale_row = np.asarray([
            float(center[0]) + max(float(size[0]), float(size[1])) * 0.35,
            float(center[1]),
            1.0,
        ]) @ grid_to_pixel
        arrow_length = max(18.0, min(48.0, float(np.linalg.norm(scale_row - center_row))))
        theta = math.radians(float(rotation))
        start = (int(round(center_row[0])), int(round(center_row[1])))
        end = (
            int(round(center_row[0] - arrow_length * math.sin(theta))),
            int(round(center_row[1] + arrow_length * math.cos(theta))),
        )
        highlighted = highlighted_object_ids is not None and object_id in highlighted_object_ids
        color = highlight_color if highlighted else (70, 70, 70)
        thickness = 3 if highlighted else 1
        cv2.arrowedLine(canvas, start, end, color, thickness, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(
            canvas,
            object_id.replace("object_", "o"),
            (start[0] + 4, start[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_orientation_review_overlay(
    image_rgb: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    xz_to_pixel_matrix: np.ndarray,
    objects_info: Sequence[dict[str, Any]],
    projected_samples: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    """Draw each finite room-axis choice in an isolated contact-sheet panel."""
    colors = [(74, 93, 117), (226, 35, 54), (30, 153, 139), (46, 134, 171)]
    grid_to_pixel = _grid_to_pixel_affine(projected_samples)
    panel_width = min(640, image_rgb.shape[1])
    panels: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        canvas = image_rgb.copy()
        color = colors[index % len(colors)]
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index + 1}")
        orientation = candidate.get("orientation", {}) if isinstance(candidate.get("orientation"), dict) else {}
        angle = float(orientation.get("angle_degrees_xz") or 0.0)
        supports = orientation.get("supporting_object_ids", []) if isinstance(orientation.get("supporting_object_ids"), list) else []
        support_ids = {str(item) for item in supports}
        _draw_object_direction_arrows(
            canvas,
            objects_info,
            grid_to_pixel,
            highlighted_object_ids=support_ids if support_ids else None,
            highlight_color=color,
        )
        polygon = [(float(x), float(z)) for x, z in candidate["metrics"]["polygon_xz"]]
        points = np.asarray(xz_to_pixel(polygon, xz_to_pixel_matrix), dtype=np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=5, lineType=cv2.LINE_AA)
        segments, _ = canonical_screen_clockwise_segments_from_polygon(
            polygon,
            xz_to_pixel_affine=xz_to_pixel_matrix.tolist(),
            source="orientation_review_overlay",
            method=str(candidate.get("method") or PRIMARY_METHOD),
            role="wall_orientation_candidate",
        )
        for segment in segments:
            edge_points = xz_to_pixel(
                [tuple(segment["first_point"]), tuple(segment["second_point"])],
                xz_to_pixel_matrix,
            )
            midpoint = (
                (int(edge_points[0][0]) + int(edge_points[1][0])) // 2,
                (int(edge_points[0][1]) + int(edge_points[1][1])) // 2,
            )
            cv2.putText(
                canvas,
                str(segment["id"]),
                (midpoint[0] + 4, midpoint[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )

        scale = panel_width / float(canvas.shape[1])
        content_height = max(1, int(round(canvas.shape[0] * scale)))
        resized = cv2.resize(canvas, (panel_width, content_height), interpolation=cv2.INTER_AREA)
        header_height = 82
        panel = np.full((header_height + content_height, panel_width, 3), 248, dtype=np.uint8)
        panel[header_height:, :, :] = resized
        source = str(orientation.get("source") or "unknown")
        cv2.putText(panel, f"{candidate_id}  angle={angle:+.1f} deg", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"source: {source}", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
        short_supports = ", ".join(str(item).replace("object_", "o") for item in supports[:8])
        support_text = f"orientation supports: {short_supports}" if short_supports else "orientation supports: none"
        cv2.putText(panel, support_text, (12, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), color, 2)
        panels.append(panel)

    if not panels:
        save_rgb(path, image_rgb)
        return

    columns = 1 if len(panels) == 1 else 2
    rows = int(math.ceil(len(panels) / columns))
    gap = 12
    title_height = 44
    panel_height = max(panel.shape[0] for panel in panels)
    sheet_width = columns * panel_width + (columns + 1) * gap
    sheet_height = title_height + rows * panel_height + (rows + 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)
    cv2.putText(
        sheet,
        "Wall orientation candidates - compare isolated panels; same mesh extent",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x0 = gap + column * (panel_width + gap)
        y0 = title_height + gap + row * (panel_height + gap)
        sheet[y0:y0 + panel.shape[0], x0:x0 + panel.shape[1], :] = panel
    save_rgb(path, sheet)

def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _draw_single_candidate(
    image_rgb: np.ndarray,
    candidate: dict[str, Any],
    xz_to_pixel_matrix: np.ndarray,
    output_path: Path,
    label: str,
) -> None:
    canvas = image_rgb.copy()
    polygon = [(float(x), float(z)) for x, z in candidate["metrics"]["polygon_xz"]]
    pts = np.asarray(xz_to_pixel(polygon, xz_to_pixel_matrix), dtype=np.int32)
    color = (226, 35, 54)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=5)
    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
    x0, y0 = pts[0]
    cv2.putText(canvas, label, (int(x0) + 5, int(y0) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    save_rgb(output_path, canvas)


def sample_mesh_footprint_points(
    posed_meshes: Sequence[Any],
    transform_info: dict[str, Any],
    *,
    max_points: int = 20000,
) -> tuple[list[Point], dict[str, Any]]:
    """Sample all posed meshes once for both extent and orientation variants."""
    if not posed_meshes:
        return [], {"error": "no_posed_meshes", "point_count": 0, "object_count": 0}
    pts: list[Point] = []
    per_object_budget = max(64, max_points // max(1, len(posed_meshes)))
    sampled_vertices = 0
    total_vertices = 0
    for mesh in posed_meshes:
        verts = getattr(mesh, "vertices", None)
        if verts is None:
            continue
        verts = np.asarray(verts, dtype=np.float32)
        if verts.size == 0:
            continue
        total_vertices += int(verts.shape[0])
        if verts.shape[0] > per_object_budget:
            step = max(1, verts.shape[0] // per_object_budget)
            verts = verts[::step]
        sampled_vertices += int(verts.shape[0])
        for vertex in verts:
            original = denormalize_point(vertex, transform_info)
            pts.append((float(original[0]), float(original[2])))
    evidence = {
        "source": "posed mesh footprint vertices projected to original OpenGL XZ",
        "coordinate_frame": "OpenGL original XZ; Y is vertical",
        "point_count": len(pts),
        "sampled_vertices": sampled_vertices,
        "total_vertices": total_vertices,
        "object_count": len(posed_meshes),
        "max_points": max_points,
    }
    if len(pts) < 4:
        evidence["error"] = "not_enough_mesh_footprint_points"
    return pts, evidence


def run_precompute(
    *,
    scene_name: str,
    run_root: Path,
    output_dir: Path,
    grid_size: int,
    image_size: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "render_grid_wall_precompute_manifest.json"
    method_candidates_path = output_dir / "method_candidates.json"
    if method_candidates_path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            manifest["reused_existing"] = True
            return manifest

    scene_paths = ScenePaths.from_scene_root(run_root)
    layout_dir = scene_paths.sam3d_output_dir / "layout_params"
    unposed_dir = scene_paths.sam3d_output_dir / "unposed_objects"
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_polygon = None

    _, posed_meshes, layout_params, object_ids, transform_info = compose_scene(
        layout_dir=layout_dir,
        unposed_dir=unposed_dir,
        requested_object_ids=None,
        anchor_id=None,
    )
    camera_direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    grid_2d, mapping, _, plane_right, plane_up = generate_2d_grid_on_plane(
        posed_meshes,
        camera_direction,
        grid_size,
        plane_side="nearest",
    )
    base = render_scene_with_grid(posed_meshes, camera_direction, 0.0, image_size)
    grid = render_scene_with_grid(
        posed_meshes,
        camera_direction,
        0.0,
        image_size,
        grid_2d=grid_2d,
        mapping=mapping,
        draw_grid_lines=True,
    )

    render_topdown_path = output_dir / "render_topdown.png"
    render_topdown_grid_path = output_dir / "render_topdown_grid.png"
    raw_mapping_path = output_dir / "render_topdown_grid_mapping_raw.json"
    canonical_topdown_path = output_dir / "canonical_topdown.png"
    canonical_mapping_path = output_dir / "canonical_topdown_grid_mapping.json"
    save_rgb(render_topdown_path, base)
    save_rgb(render_topdown_grid_path, grid)
    _copy_file(render_topdown_path, canonical_topdown_path)
    save_grid_mapping(
        mapping,
        raw_mapping_path,
        transform_info,
        posed_meshes=posed_meshes,
        object_ids=object_ids,
        axis_right=plane_right,
        axis_up=plane_up,
        camera_direction=camera_direction,
        layout_params=layout_params,
    )
    _copy_file(raw_mapping_path, canonical_mapping_path)

    camera_radius, xmag, ymag = resolve_orthographic_camera_frame(posed_meshes, camera_direction, 0.0)
    camera_pose = build_camera_pose(camera_direction, camera_radius)
    camera_matrix = get_camera_matrix(60.0, image_size, ortho=True, xmag=xmag, ymag=ymag)
    pix_to_xz, xz_to_pix, projected_samples = pixel_affines(
        mapping,
        transform_info,
        camera_pose,
        camera_matrix,
        image_size,
    )
    affine_path = output_dir / "render_grid_affine.json"
    write_json(
        affine_path,
        {
            "grid_size": grid_size,
            "image_size": image_size,
            "same_grid_policy": "current multi-agent run uses this topdown grid and affine for mask, candidates, overlays, and wall visual evidence",
            "pixel_to_xz_affine": pix_to_xz.tolist(),
            "xz_to_pixel_affine": xz_to_pix.tolist(),
            "projected_grid_samples": projected_samples,
        },
    )

    fg = foreground_mask(base)
    mask_path = output_dir / "render_foreground_mask.png"
    cv2.imwrite(str(mask_path), fg)

    mapping_payload = read_json(raw_mapping_path)
    objects_info = mapping_payload.get("objects", []) if isinstance(mapping_payload, dict) else []
    mesh_points, obb_evidence = sample_mesh_footprint_points(posed_meshes, transform_info)
    obb_polygon = None
    if len(mesh_points) >= 4:
        obb_polygon, _, _ = min_area_rect(mesh_points, margin=0.0)
        obb_polygon = canonical_clockwise_rect(obb_polygon)
    orientation_clusters = cluster_object_orientation_axes(
        [item for item in objects_info if isinstance(item, dict)],
        grid_size=grid_size,
    )

    candidates: list[dict[str, Any]] = []
    if obb_polygon and len(obb_polygon) >= 4:
        metrics = candidate_metrics(obb_polygon, layout_polygon, fg, xz_to_pix)
        candidates.append(
            {
                "candidate_id": "mesh_min_area",
                "method": PRIMARY_METHOD,
                "metrics": metrics,
                "orientation": {
                    "source": "mesh_footprint_minimum_area",
                    "angle_degrees_xz": round(
                        normalize_axis_angle_degrees(dominant_rect_angle_degrees(obb_polygon)),
                        6,
                    ),
                    "supporting_object_ids": [],
                },
                "evidence": {
                    **obb_evidence,
                    "note": "Formal baseline OBB. Alternative mask/LSD/layout wall methods are not exposed.",
                },
            }
        )

    baseline_angle = (
        float(candidates[0]["orientation"]["angle_degrees_xz"])
        if candidates
        else None
    )
    for cluster in orientation_clusters.get("clusters", []):
        angle_degrees = float(cluster["angle_degrees_xz"])
        if baseline_angle is not None and axis_angle_distance_degrees(angle_degrees, baseline_angle) <= 1.0:
            continue
        polygon = rect_from_angle(mesh_points, math.radians(angle_degrees), margin=0.0)
        metrics = candidate_metrics(polygon, layout_polygon, fg, xz_to_pix)
        candidates.append(
            {
                "candidate_id": str(cluster["cluster_id"]),
                "method": PRIMARY_METHOD,
                "metrics": metrics,
                "orientation": {
                    "source": "robust_object_rotation_consensus",
                    "angle_degrees_xz": round(angle_degrees, 6),
                    "supporting_object_ids": list(cluster.get("supporting_object_ids", [])),
                    "member_count": cluster.get("member_count"),
                    "total_weight": cluster.get("total_weight"),
                },
                "evidence": {
                    **obb_evidence,
                    "angle_source": "rotation_degrees from the same canonical top-down render grid",
                    "orientation_cluster": cluster,
                    "note": "The complete posed-mesh footprint fixes the extent; only the rectangle axis is constrained by this object-rotation cluster.",
                },
            }
        )

    for idx, candidate in enumerate(candidates, start=1):
        candidate["rank_by_policy"] = idx
        candidate["rank_by_layout_iou_then_coverage"] = idx

    all_overlay_path = output_dir / "candidate_overlay.png"
    draw_candidates(base, candidates, xz_to_pix, all_overlay_path)
    orientation_overlay_path = output_dir / "wall_orientation_candidates_overlay.png"
    draw_orientation_review_overlay(
        base,
        candidates,
        xz_to_pix,
        objects_info,
        projected_samples,
        orientation_overlay_path,
    )
    primary = candidates[0] if candidates else None
    primary_overlay_path = None
    if primary:
        primary_overlay_path = output_dir / "primary_candidate_overlay.png"
        _draw_single_candidate(
            base,
            primary,
            xz_to_pix,
            primary_overlay_path,
            f'primary:{primary.get("candidate_id", PRIMARY_METHOD)}',
        )

    result = {
        "scene": scene_name,
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "method_family": "render_grid_wall_precompute",
        "renderer": {
            "layout_dir": str(layout_dir),
            "unposed_dir": str(unposed_dir),
            "object_count": len(object_ids),
            "camera_direction": camera_direction.tolist(),
            "image_size": image_size,
            "grid_size": grid_size,
            "same_grid_policy": "single current-run render grid reused by topdown, mask, candidate generation, overlays, and downstream wall visual evidence",
        },
        "method_policy": {
            "agent_should_not_choose_method": True,
            "orientation_selected_locally": True,
            "primary_method": PRIMARY_METHOD,
            "supporting_methods": SUPPORTING_METHODS,
            "excluded_methods": [
                "layout_walls_from_06_layout",
                "mesh_vertices_min_area_rect",
                "render_foreground_mask_min_area_rect",
                "opencv_lsd_rect",
                "opencv_hough_linesp_rect",
                "sklearn_ransac_edge_rect",
            ],
            "reason": "Every candidate uses the complete posed-mesh footprint and the formal OBB method family. Local code selects one finite orientation using Agent-identified wall-following objects; the Agent cannot choose methods or invent coordinates.",
        },
        "wall_orientation_review": {
            "schema_version": "formal_wall_orientation_review",
            "local_selection_required": len(candidates) > 1,
            "baseline_candidate_id": "mesh_min_area" if primary else None,
            "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
            "object_orientation_clusters": orientation_clusters,
            "review_asset_layout": "isolated_candidate_contact_sheet",
            "agent_contract": "Identify wall-following objects and their visible orientation evidence from the isolated panels. Local code scores the finite candidate ids. Supporting object ids are orientation evidence only and never imply wall attach.",
        },
        "layout_polygon_available": layout_polygon is not None,
        "layout_polygon_source": None,
        "layout_polygon_note": (
            "Optional upstream layout polygon used only for IoU diagnostics; formal wall coordinates do not depend on it."
            if layout_polygon is not None
            else "Upstream layout polygon unavailable; formal wall coordinates were generated directly from SAM3D mesh footprints."
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "assets": {
            "render_topdown_path": str(render_topdown_path),
            "render_topdown_grid_path": str(render_topdown_grid_path),
            "canonical_topdown_path": str(canonical_topdown_path),
            "canonical_grid_mapping_path": str(canonical_mapping_path),
            "render_topdown_grid_mapping_raw_path": str(raw_mapping_path),
            "render_grid_affine_path": str(affine_path),
            "render_foreground_mask_path": str(mask_path),
            "candidate_overlay_path": str(all_overlay_path),
            "wall_orientation_candidates_overlay_path": str(orientation_overlay_path),
            "primary_candidate_overlay_path": str(primary_overlay_path) if primary_overlay_path else None,
        },
    }
    write_json(method_candidates_path, result)

    manifest = {
        "status": "available" if primary else "missing_primary",
        "scene_name": scene_name,
        "precompute_dir": str(output_dir),
        "method_candidates_path": str(method_candidates_path),
        "manifest_path": str(manifest_path),
        "canonical_topdown_image_path": str(canonical_topdown_path),
        "canonical_mapping_path": str(canonical_mapping_path),
        "render_grid_affine_path": str(affine_path),
        "render_foreground_mask_path": str(mask_path),
        "candidate_overlay_path": str(all_overlay_path),
        "wall_orientation_candidates_overlay_path": str(orientation_overlay_path),
        "primary_candidate_overlay_path": str(primary_overlay_path) if primary_overlay_path else None,
        "primary_method": PRIMARY_METHOD,
        "supporting_methods": SUPPORTING_METHODS,
        "same_grid_policy": result["renderer"]["same_grid_policy"],
        "reused_existing": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate current-run render-grid wall candidates.")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--grid_size", type=int, default=33)
    parser.add_argument("--image_size", type=int, default=768)
    args = parser.parse_args()
    result = run_precompute(
        scene_name=args.scene_name,
        run_root=args.run_root,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
