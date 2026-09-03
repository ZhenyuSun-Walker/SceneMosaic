from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.state import write_json, write_md

CONTACT_SHEET_NAME = "{scene}_wall_evidence_contact_sheet.png"
CANDIDATE_OVERLAY_NAME = "{scene}_candidate_walls_overlay.png"
MANIFEST_NAME = "{scene}_wall_visual_review_manifest.json"
OVERLAY_JSON_NAME = "{scene}_candidate_walls_overlay.json"
OVERLAY_MD_NAME = "{scene}_candidate_walls_overlay.md"
MANIFEST_SCHEMA_VERSION = "wall_visual_review"


def _load_json(path: Path | None) -> Any:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _path(value: Any) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def _path_status(paths: dict[str, Path | None]) -> dict[str, dict[str, Any]]:
    return {key: {"path": str(path) if path else None, "exists": bool(path and path.exists())} for key, path in paths.items()}


def _fixed_view_paths(scene_state: dict[str, Any]) -> dict[str, Path | None]:
    fixed = scene_state.get("evidence", {}).get("fixed_layout_views", {})
    result: dict[str, Path | None] = {
        "top_view_clean": None,
        "top_view_grid": None,
        "front_view": None,
        "back_view": None,
        "left_view": None,
        "right_view": None,
    }
    for item in fixed.get("views", []) if isinstance(fixed.get("views"), list) else []:
        if not isinstance(item, dict):
            continue
        view = item.get("view")
        if view == "top":
            result["top_view_clean"] = _path(item.get("clean_image_path") or item.get("image_path"))
            result["top_view_grid"] = _path(item.get("grid_image_path"))
        elif view in {"front", "back", "left", "right"}:
            result[f"{view}_view"] = _path(item.get("image_path"))
    return result


def _title(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, path: Path | None, font: ImageFont.ImageFont) -> None:
    draw.text((x + 12, y + 10), title, fill=(15, 15, 15), font=font)
    draw.text((x + 12, y + 25), path.name if path else "missing", fill=(80, 80, 80), font=font)


def _make_contact_sheet(image_specs: list[tuple[str, Path | None]], output_path: Path) -> None:
    font = ImageFont.load_default()
    tile_w = 560
    tile_h = 420
    label_h = 44
    cols = 3
    rows = max(1, (len(image_specs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    for idx, (title, path) in enumerate(image_specs):
        x0 = (idx % cols) * tile_w
        y0 = (idx // cols) * tile_h
        draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline=(205, 205, 205), fill=(250, 250, 250))
        _title(draw, x0, y0, title, path, font)
        if not path or not path.exists():
            draw.text((x0 + 12, y0 + 82), "MISSING", fill=(180, 0, 0), font=font)
            continue
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((tile_w - 28, tile_h - label_h - 24), Image.Resampling.LANCZOS)
            ix = x0 + (tile_w - im.width) // 2
            iy = y0 + label_h + (tile_h - label_h - im.height) // 2
            sheet.paste(im, (ix, iy))
        except Exception as exc:
            draw.text((x0 + 12, y0 + 82), f"LOAD FAILED: {exc}", fill=(180, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _candidate_fields(wall_evidence: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidate_walls = wall_evidence.get("candidate_walls") or []
    polygon = wall_evidence.get("candidate_room_polygon_hint")
    if not candidate_walls and isinstance(wall_evidence.get("debug_summary"), dict):
        candidate_walls = wall_evidence["debug_summary"].get("candidate_walls") or []
    if not polygon and isinstance(wall_evidence.get("debug_summary"), dict):
        polygon = wall_evidence["debug_summary"].get("candidate_room_polygon_hint")
    return [w for w in candidate_walls if isinstance(w, dict)], polygon if isinstance(polygon, dict) else None


def _render_grid_assets(wall_evidence: dict[str, Any]) -> dict[str, Path | None]:
    source = wall_evidence.get("render_grid_wall_candidates")
    source = source if isinstance(source, dict) else {}
    assets = source.get("assets") if isinstance(source.get("assets"), dict) else {}
    source_path = _path(source.get("source_path"))
    base = source_path.parent if source_path else None

    def path_from_asset(key: str, default_name: str | None = None) -> Path | None:
        raw = assets.get(key)
        path = _path(raw)
        if path and path.exists():
            return path
        if base and default_name:
            fallback = base / default_name
            if fallback.exists():
                return fallback
        return path if path else None

    return {
        "render_grid_topdown_path": path_from_asset("render_topdown_path", "render_topdown.png"),
        "render_grid_topdown_grid_path": path_from_asset("render_topdown_grid_path", "render_topdown_grid.png"),
        "render_grid_affine_path": path_from_asset("render_grid_affine_path", "render_grid_affine.json"),
        "render_grid_foreground_mask_path": path_from_asset("render_foreground_mask_path", "render_foreground_mask.png"),
        "render_grid_candidate_overlay_path": path_from_asset("candidate_overlay_path", "candidate_overlay.png"),
        "render_grid_wall_orientation_candidates_overlay_path": path_from_asset("wall_orientation_candidates_overlay_path", "wall_orientation_candidates_overlay.png"),
        "render_grid_primary_candidate_overlay_path": path_from_asset("primary_candidate_overlay_path", "primary_candidate_overlay.png"),
        "render_grid_method_candidates_path": source_path,
    }


def _exact_candidate_overlay(
    render_grid_assets: dict[str, Path | None],
    candidate_walls: list[dict[str, Any]],
    output_png: Path,
) -> dict[str, Any] | None:
    source = render_grid_assets.get("render_grid_primary_candidate_overlay_path") or render_grid_assets.get("render_grid_candidate_overlay_path")
    affine = render_grid_assets.get("render_grid_affine_path")
    if not source or not source.exists():
        return None
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_png.write_bytes(source.read_bytes())
    return {
        "overlay_path": str(output_png),
        "generated": True,
        "mapping_possible": True,
        "mapping_method": "exact_render_grid_affine_overlay_reused",
        "mapping_is_approximate": False,
        "source_overlay_path": str(source),
        "render_grid_affine_path": str(affine) if affine else None,
        "reason": "Reused exact current-run render-grid overlay generated with render_grid_affine.json.",
        "source_warning": "Overlay is current-run generated wall candidate evidence for visual review; it is not a final wall set by itself.",
        "candidate_walls_source_trust": [
            {
                "id": wall.get("id"),
                "source": wall.get("source"),
                "trust_level": wall.get("trust_level"),
                "may_be_used_as_final_wall": wall.get("may_be_used_as_final_wall"),
                "topdown_verification_status": wall.get("topdown_verification_status"),
                "first_point": wall.get("first_point"),
                "second_point": wall.get("second_point"),
            }
            for wall in candidate_walls
        ],
    }


def _grid_xz_bounds(grid_mapping: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(grid_mapping, dict):
        return None
    xs: list[float] = []
    zs: list[float] = []
    for pt in grid_mapping.get("grid_points", []) if isinstance(grid_mapping.get("grid_points"), list) else []:
        if not isinstance(pt, dict):
            continue
        raw = pt.get("point_3d_original") or pt.get("point_3d_normalized") or pt.get("point_3d_original_xz")
        if isinstance(raw, list) and len(raw) >= 3:
            xs.append(float(raw[0]))
            zs.append(float(raw[2]))
        elif isinstance(raw, list) and len(raw) == 2:
            xs.append(float(raw[0]))
            zs.append(float(raw[1]))
    if not xs or not zs or max(xs) - min(xs) <= 1e-12 or max(zs) - min(zs) <= 1e-12:
        return None
    return {"x_min": min(xs), "x_max": max(xs), "z_min": min(zs), "z_max": max(zs)}


def _draw_candidate_overlay(
    canonical_topdown: Path | None,
    grid_mapping_path: Path | None,
    candidate_walls: list[dict[str, Any]],
    output_png: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overlay_path": str(output_png),
        "generated": False,
        "mapping_possible": False,
        "mapping_method": None,
        "mapping_is_approximate": True,
        "reason": None,
        "source_warning": "Overlay is for human/agent visual review only and does not represent final walls.",
        "candidate_walls_source_trust": [
            {
                "id": wall.get("id"),
                "source": wall.get("source"),
                "trust_level": wall.get("trust_level"),
                "may_be_used_as_final_wall": wall.get("may_be_used_as_final_wall"),
                "topdown_verification_status": wall.get("topdown_verification_status"),
                "first_point": wall.get("first_point"),
                "second_point": wall.get("second_point"),
            }
            for wall in candidate_walls
        ],
    }
    if not canonical_topdown or not canonical_topdown.exists():
        result["reason"] = "canonical top-down image is missing."
        return result
    if not grid_mapping_path or not grid_mapping_path.exists():
        result["reason"] = "canonical grid mapping JSON is missing."
        return result
    if not candidate_walls:
        result["reason"] = "wall_evidence.json does not contain candidate_walls."
        return result
    try:
        base = Image.open(canonical_topdown).convert("RGBA")
    except Exception as exc:
        result["reason"] = f"canonical top-down could not be opened: {exc}"
        return result
    bounds = _grid_xz_bounds(_load_json(grid_mapping_path))
    if not bounds:
        result["reason"] = "canonical grid mapping does not contain usable XZ bounds."
        return result

    margin = 28
    width, height = base.size

    def xz_to_px(point: list[float]) -> tuple[float, float]:
        x, z = float(point[0]), float(point[1])
        px = margin + (x - bounds["x_min"]) / (bounds["x_max"] - bounds["x_min"]) * (width - 2 * margin)
        py = height - margin - (z - bounds["z_min"]) / (bounds["z_max"] - bounds["z_min"]) * (height - 2 * margin)
        return px, py

    font = ImageFont.load_default()
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    colors = [(255, 40, 40, 255), (40, 120, 255, 255), (0, 170, 80, 255), (255, 150, 0, 255)]
    for idx, wall in enumerate(candidate_walls):
        p1 = wall.get("first_point")
        p2 = wall.get("second_point")
        if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) == 2 and len(p2) == 2):
            continue
        xy1, xy2 = xz_to_px(p1), xz_to_px(p2)
        color = colors[idx % len(colors)]
        draw.line((xy1, xy2), fill=color, width=6)
        for xy in (xy1, xy2):
            draw.ellipse((xy[0] - 6, xy[1] - 6, xy[0] + 6, xy[1] + 6), fill=color)
        mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
        draw.rectangle((mx + 5, my - 8, mx + 80, my + 12), fill=(255, 255, 255, 220))
        draw.text((mx + 8, my - 7), str(wall.get("id")), fill=color, font=font)
    draw.rectangle((8, 8, width - 8, 50), fill=(255, 255, 255, 225))
    draw.text((16, 17), "candidate_walls overlay: approximate grid-extent XZ -> image mapping; visual review only", fill=(180, 0, 0), font=font)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_png)
    result.update(
        {
            "generated": True,
            "mapping_possible": True,
            "mapping_method": "approximate_linear_xz_to_image_using_canonical_grid_extent_and_image_dimensions",
            "mapping_bounds_xz": bounds,
            "image_size": {"width": width, "height": height},
            "image_margin_px": margin,
            "reason": "Generated using grid extent because grid mapping does not store explicit pixel coordinates.",
        }
    )
    return result


def _write_overlay_md(path: Path, overlay_result: dict[str, Any]) -> None:
    lines = [
        f"- overlay_generated: `{overlay_result.get('generated')}`",
        f"- overlay_path: `{overlay_result.get('overlay_path')}`",
        f"- mapping_possible: `{overlay_result.get('mapping_possible')}`",
        f"- mapping_method: `{overlay_result.get('mapping_method')}`",
        f"- mapping_is_approximate: `{overlay_result.get('mapping_is_approximate')}`",
        f"- reason: `{overlay_result.get('reason')}`",
        "",
        "Overlay is for visual review only and does not represent final walls.",
        "",
        "## Candidate Walls",
    ]
    for wall in overlay_result.get("candidate_walls_source_trust", []):
        lines.append(
            f"- {wall.get('id')}: source=`{wall.get('source')}`, trust_level=`{wall.get('trust_level')}`, "
            f"may_be_used_as_final_wall=`{wall.get('may_be_used_as_final_wall')}`, verification=`{wall.get('topdown_verification_status')}`"
        )
    write_md(path, "Candidate Walls Overlay", lines)


def ensure_wall_visual_evidence(scene_state: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    scene_name = str(scene_state.get("scene_name") or "scene")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME.format(scene=scene_name)
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if (
            isinstance(manifest, dict)
            and manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
            and Path(str(manifest.get("contact_sheet_path", ""))).exists()
        ):
            overlay = manifest.get("candidate_walls_overlay_path")
            if not overlay or Path(str(overlay)).exists():
                return manifest

    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    topdown = scene_state.get("topdown_grid_mapping", {}) if isinstance(scene_state.get("topdown_grid_mapping"), dict) else {}
    run_root = Path(scene_state.get("run_root", "."))
    attach_root = output_dir.parents[2] if output_dir.name == "visual_review" and output_dir.parent.name == "walls" else output_dir
    wall_evidence_path = output_dir.parent / "wall_evidence.json" if output_dir.parent.name == "walls" else attach_root / "debug" / "walls" / "wall_evidence.json"

    paths: dict[str, Path | None] = {
        "canonical_topdown_path": _path(topdown.get("canonical_topdown_image_path")),
        "canonical_grid_mapping_path": _path(topdown.get("canonical_mapping_path")),
        "wall_evidence_path": wall_evidence_path,
        "scene_state_path": output_dir.parents[2] / "scene_states" / "scene_state.json" if len(output_dir.parents) >= 3 else None,
        "original_image_path": _path(evidence.get("original_image_path")),
        "mask_overlay_path": _path(evidence.get("mask_overlay_path")),
        "reviewed_overlay_path": _path(evidence.get("reviewed_overlay_path")),
        "object_masks_dir": _path(evidence.get("object_masks_dir")),
    }
    paths.update(_fixed_view_paths(scene_state))

    wall_evidence = _load_json(paths["wall_evidence_path"]) if paths.get("wall_evidence_path") else {}
    wall_evidence = wall_evidence if isinstance(wall_evidence, dict) else {}
    candidate_walls, candidate_polygon = _candidate_fields(wall_evidence)
    render_grid_assets = _render_grid_assets(wall_evidence)
    paths.update(render_grid_assets)

    contact_sheet_path = output_dir / CONTACT_SHEET_NAME.format(scene=scene_name)
    contact_specs = [
        ("render_grid_wall_orientation_candidates", paths.get("render_grid_wall_orientation_candidates_overlay_path")),
        ("render_grid_primary_candidate_exact", paths.get("render_grid_primary_candidate_overlay_path")),
        ("render_grid_candidates_exact", paths.get("render_grid_candidate_overlay_path")),
        ("render_grid_topdown", paths.get("render_grid_topdown_path")),
        ("render_grid_topdown_grid", paths.get("render_grid_topdown_grid_path")),
        ("render_grid_foreground_mask", paths.get("render_grid_foreground_mask_path")),
        ("canonical_topdown", paths.get("canonical_topdown_path")),
        ("top_view_clean", paths.get("top_view_clean")),
        ("top_view_grid", paths.get("top_view_grid")),
        ("front_view", paths.get("front_view")),
        ("back_view", paths.get("back_view")),
        ("left_view", paths.get("left_view")),
        ("right_view", paths.get("right_view")),
        ("original_image", paths.get("original_image_path")),
        ("mask_overlay", paths.get("mask_overlay_path")),
    ]
    if paths.get("reviewed_overlay_path"):
        contact_specs.append(("reviewed_overlay", paths.get("reviewed_overlay_path")))
    _make_contact_sheet(contact_specs, contact_sheet_path)

    overlay_path = output_dir / CANDIDATE_OVERLAY_NAME.format(scene=scene_name)
    overlay_result = _exact_candidate_overlay(render_grid_assets, candidate_walls, overlay_path)
    if overlay_result is None:
        overlay_result = _draw_candidate_overlay(
            paths.get("canonical_topdown_path"),
            paths.get("canonical_grid_mapping_path"),
            candidate_walls,
            overlay_path,
        )
    overlay_json_path = output_dir / OVERLAY_JSON_NAME.format(scene=scene_name)
    write_json(overlay_json_path, overlay_result)
    overlay_md_path = output_dir / OVERLAY_MD_NAME.format(scene=scene_name)
    _write_overlay_md(overlay_md_path, overlay_result)

    sources = sorted({str(w.get("source")) for w in candidate_walls if w.get("source")})
    trust_levels = sorted({str(w.get("trust_level")) for w in candidate_walls if w.get("trust_level")})
    may_final = all(w.get("may_be_used_as_final_wall") is True for w in candidate_walls) if candidate_walls else False
    image_paths_priority = [
        str(contact_sheet_path),
        str(overlay_path) if overlay_result.get("generated") else None,
        str(paths["render_grid_wall_orientation_candidates_overlay_path"]) if paths.get("render_grid_wall_orientation_candidates_overlay_path") and paths["render_grid_wall_orientation_candidates_overlay_path"].exists() else None,
        str(paths["render_grid_primary_candidate_overlay_path"]) if paths.get("render_grid_primary_candidate_overlay_path") and paths["render_grid_primary_candidate_overlay_path"].exists() else None,
        str(paths["render_grid_topdown_path"]) if paths.get("render_grid_topdown_path") and paths["render_grid_topdown_path"].exists() else None,
        str(paths["render_grid_topdown_grid_path"]) if paths.get("render_grid_topdown_grid_path") and paths["render_grid_topdown_grid_path"].exists() else None,
        str(paths["render_grid_foreground_mask_path"]) if paths.get("render_grid_foreground_mask_path") and paths["render_grid_foreground_mask_path"].exists() else None,
        str(paths["canonical_topdown_path"]) if paths.get("canonical_topdown_path") and paths["canonical_topdown_path"].exists() else None,
        str(paths["top_view_clean"]) if paths.get("top_view_clean") and paths["top_view_clean"].exists() else None,
        str(paths["top_view_grid"]) if paths.get("top_view_grid") and paths["top_view_grid"].exists() else None,
        str(paths["original_image_path"]) if paths.get("original_image_path") and paths["original_image_path"].exists() else None,
        str(paths["mask_overlay_path"]) if paths.get("mask_overlay_path") and paths["mask_overlay_path"].exists() else None,
        str(paths["front_view"]) if paths.get("front_view") and paths["front_view"].exists() else None,
        str(paths["back_view"]) if paths.get("back_view") and paths["back_view"].exists() else None,
        str(paths["left_view"]) if paths.get("left_view") and paths["left_view"].exists() else None,
        str(paths["right_view"]) if paths.get("right_view") and paths["right_view"].exists() else None,
        str(paths["reviewed_overlay_path"]) if paths.get("reviewed_overlay_path") and paths["reviewed_overlay_path"].exists() else None,
    ]
    image_paths_priority = [p for p in image_paths_priority if p]
    fallback_image_paths_priority = image_paths_priority[:5]

    manifest = {
        "scene_name": scene_name,
        "review_type": "wall_visual_review",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "contact_sheet_path": str(contact_sheet_path),
        "candidate_walls_overlay_path": str(overlay_path) if overlay_result.get("generated") else None,
        "candidate_walls_overlay_json_path": str(overlay_json_path),
        "candidate_walls_overlay_md_path": str(overlay_md_path),
        "canonical_topdown_path": str(paths.get("canonical_topdown_path")) if paths.get("canonical_topdown_path") else None,
        "canonical_grid_mapping_path": str(paths.get("canonical_grid_mapping_path")) if paths.get("canonical_grid_mapping_path") else None,
        "render_grid_method_candidates_path": str(paths.get("render_grid_method_candidates_path")) if paths.get("render_grid_method_candidates_path") else None,
        "render_grid_affine_path": str(paths.get("render_grid_affine_path")) if paths.get("render_grid_affine_path") else None,
        "render_grid_topdown_path": str(paths.get("render_grid_topdown_path")) if paths.get("render_grid_topdown_path") else None,
        "render_grid_topdown_grid_path": str(paths.get("render_grid_topdown_grid_path")) if paths.get("render_grid_topdown_grid_path") else None,
        "render_grid_foreground_mask_path": str(paths.get("render_grid_foreground_mask_path")) if paths.get("render_grid_foreground_mask_path") else None,
        "render_grid_candidate_overlay_path": str(paths.get("render_grid_candidate_overlay_path")) if paths.get("render_grid_candidate_overlay_path") else None,
        "render_grid_wall_orientation_candidates_overlay_path": str(paths.get("render_grid_wall_orientation_candidates_overlay_path")) if paths.get("render_grid_wall_orientation_candidates_overlay_path") else None,
        "render_grid_primary_candidate_overlay_path": str(paths.get("render_grid_primary_candidate_overlay_path")) if paths.get("render_grid_primary_candidate_overlay_path") else None,
        "top_view_clean_path": str(paths.get("top_view_clean")) if paths.get("top_view_clean") else None,
        "top_view_grid_path": str(paths.get("top_view_grid")) if paths.get("top_view_grid") else None,
        "front_view_path": str(paths.get("front_view")) if paths.get("front_view") else None,
        "back_view_path": str(paths.get("back_view")) if paths.get("back_view") else None,
        "left_view_path": str(paths.get("left_view")) if paths.get("left_view") else None,
        "right_view_path": str(paths.get("right_view")) if paths.get("right_view") else None,
        "original_image_path": str(paths.get("original_image_path")) if paths.get("original_image_path") else None,
        "mask_overlay_path": str(paths.get("mask_overlay_path")) if paths.get("mask_overlay_path") else None,
        "reviewed_overlay_path": str(paths.get("reviewed_overlay_path")) if paths.get("reviewed_overlay_path") else None,
        "object_masks_dir": str(paths.get("object_masks_dir")) if paths.get("object_masks_dir") else None,
        "wall_evidence_path": str(paths.get("wall_evidence_path")) if paths.get("wall_evidence_path") else None,
        "path_status": _path_status(paths),
        "candidate_walls_count": len(candidate_walls),
        "candidate_walls_source": sources,
        "candidate_walls_trust_level": trust_levels,
        "candidate_walls_may_be_final": may_final,
        "candidate_room_polygon_hint": candidate_polygon,
        "candidate_walls": candidate_walls,
        "candidate_overlay": overlay_result,
        "image_paths_for_wall_repair": image_paths_priority,
        "fallback_image_paths_for_wall_repair": fallback_image_paths_priority,
        "failure_summary": [
            "no final-eligible wall set",
            "candidate walls are hint-only",
            "final outputs not generated when final walls are missing",
        ],
        "next_suggestions": [
            "If visual evidence is sufficient, the wall review task should preserve obb_from_mesh_footprint_min_rect walls.",
            "If visual evidence is insufficient, return insufficient_layout_xz_wall_evidence rather than promoting hints.",
            "If automation remains ambiguous, ask a human to provide four layout XZ corners or wall segments.",
        ],
        "api_called": False,
        "final_outputs_modified": False,
    }
    write_json(manifest_path, manifest)
    lines = [
        f"- scene_name: `{scene_name}`",
        f"- contact_sheet_path: `{manifest['contact_sheet_path']}`",
        f"- candidate_walls_overlay_path: `{manifest['candidate_walls_overlay_path']}`",
        f"- canonical_topdown_path: `{manifest['canonical_topdown_path']}`",
        f"- canonical_grid_mapping_path: `{manifest['canonical_grid_mapping_path']}`",
        f"- wall_evidence_path: `{manifest['wall_evidence_path']}`",
        f"- candidate_walls_count: `{manifest['candidate_walls_count']}`",
        f"- candidate_walls_source: `{manifest['candidate_walls_source']}`",
        f"- candidate_walls_trust_level: `{manifest['candidate_walls_trust_level']}`",
        f"- candidate_walls_may_be_final: `{manifest['candidate_walls_may_be_final']}`",
        f"- original_image_path: `{manifest['original_image_path']}`",
        f"- mask_overlay_path: `{manifest['mask_overlay_path']}`",
        f"- image_paths_for_wall_repair: `{manifest['image_paths_for_wall_repair']}`",
        "",
        "## Failure Summary",
    ]
    lines.extend(f"- {item}" for item in manifest["failure_summary"])
    write_md(manifest_path.with_suffix(".md"), "Wall Visual Review Manifest", lines)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wall visual evidence for visual wall inference tasks.")
    parser.add_argument("--scene_state", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    scene_state = json.loads(args.scene_state.read_text(encoding="utf-8"))
    manifest = ensure_wall_visual_evidence(scene_state, args.output_dir)
    print(json.dumps({"manifest_path": str(args.output_dir / MANIFEST_NAME.format(scene=manifest.get("scene_name", "scene"))), "contact_sheet_path": manifest.get("contact_sheet_path"), "candidate_walls_overlay_path": manifest.get("candidate_walls_overlay_path")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
