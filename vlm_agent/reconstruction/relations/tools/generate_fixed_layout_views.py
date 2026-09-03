from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.state import write_json, write_md

DEFAULT_IMAGE_SIZE = 512

VIEW_DIRECTIONS: dict[str, list[float]] = {
    "top": [0.0, -1.0, 0.0],
    "front": [0.0, 0.0, -1.0],
    "back": [0.0, 0.0, 1.0],
    "left": [-1.0, 0.0, 0.0],
    "right": [1.0, 0.0, 0.0],
}

SIDE_VIEWS = {"front", "back", "left", "right"}

GRID_MAPPING_USAGE_NOTE = (
    "canonical top-down [x,z] reference only; side-view pixels should not be interpreted "
    "as independent [x,z] coordinates"
)
TOP_VIEW_USAGE_NOTE = "top grid image may show coordinate overlay; clean image is preferred for visual object inspection"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest_md(manifest_path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"- status: `{manifest.get('status')}`",
        f"- views_dir: `{manifest.get('views_dir')}`",
        f"- canonical_grid_mapping_path: `{manifest.get('canonical_grid_mapping_path')}`",
        f"- render_script: `{manifest.get('render_script')}`",
        f"- render_python: `{manifest.get('render_python')}`",
        f"- grid_mapping_usage_note: `{manifest.get('grid_mapping_usage_note')}`",
        "",
        "## Views",
    ]
    for item in manifest.get("views", []):
        image_path = item.get("clean_image_path") or item.get("image_path")
        lines.extend(
            [
                f"- `{item.get('view')}`: status `{item.get('status')}`, reused `{item.get('reused_existing')}`",
                f"  - image: `{image_path}`",
                f"  - clean_image: `{item.get('clean_image_path')}`",
                f"  - grid_image: `{item.get('grid_image_path')}`",
                f"  - metadata: `{item.get('metadata_path')}`",
                f"  - camera_direction: `{item.get('camera_direction')}`",
                f"  - draw_grid_overlay: `{item.get('draw_grid_overlay')}`",
                f"  - is_clean_view: `{item.get('is_clean_view')}`",
            ]
        )
    write_md(manifest_path.with_suffix(".md"), "Fixed Layout Views Manifest", lines)


def _write_render_log(scene_state: dict[str, Any], output_dir: Path, calls: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    # Runner output is <run>/generated_evidence/views, while standalone
    # fixed-views precompute writes directly to its own precompute dir.
    # Keep logs beside the owner of the generated evidence in both cases.
    if output_dir.parent.name == "generated_evidence" and len(output_dir.parents) > 1:
        log_root = output_dir.parents[1]
    else:
        log_root = output_dir
    log_dir = log_root / "debug" / "rendering"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fixed_views_render_log.json"
    existing = _load_json(log_path)
    log = existing if existing else {"render_calls": []}
    log.setdefault("render_calls", []).extend(calls)
    log["latest_manifest_path"] = str(output_dir / "views_manifest.json")
    log["latest_status"] = manifest.get("status")
    write_json(log_path, log)
    lines = [
        f"- latest_status: `{manifest.get('status')}`",
        f"- manifest: `{output_dir / 'views_manifest.json'}`",
        f"- canonical_grid_mapping_path: `{manifest.get('canonical_grid_mapping_path')}`",
        "",
        "## Latest Calls",
    ]
    for call in calls:
        lines.extend(
            [
                f"- `{call.get('view')}` `{call.get('variant') or 'default'}`: `{call.get('status')}`",
                f"  - output_png: `{call.get('output_png')}`",
                f"  - output_json: `{call.get('output_json')}`",
                f"  - used_grid_mapping: `{call.get('uses_canonical_grid_mapping')}`",
                f"  - passes_grid_mapping_to_renderer: `{call.get('passes_grid_mapping_to_renderer')}`",
                f"  - draw_grid_overlay: `{call.get('draw_grid_overlay')}`",
            ]
        )
    write_md(log_dir / "fixed_views_render_log.md", "Fixed Views Render Log", lines)


def _current_manifest_ok(manifest_path: Path, canonical_mapping_path: str, views: dict[str, list[float]]) -> bool:
    manifest = _load_json(manifest_path)
    if not manifest or manifest.get("canonical_grid_mapping_path") != canonical_mapping_path:
        return False
    items = manifest.get("views")
    if not isinstance(items, list) or {item.get("view") for item in items} != set(views):
        return False
    for item in items:
        view = item.get("view")
        metadata_path = item.get("metadata_path")
        if not metadata_path or not Path(str(metadata_path)).exists():
            return False
        if view == "top":
            if item.get("draw_grid_overlay") is not True:
                return False
            if not item.get("clean_image_path") or not Path(str(item.get("clean_image_path"))).exists():
                return False
            if not item.get("grid_image_path") or not Path(str(item.get("grid_image_path"))).exists():
                return False
        elif view in SIDE_VIEWS:
            if item.get("draw_grid_overlay") is not False or item.get("is_clean_view") is not True:
                return False
            if not item.get("image_path") or not Path(str(item.get("image_path"))).exists():
                return False
        else:
            return False
    return True


def reuse_fixed_layout_views_cache(scene_state: dict[str, Any], cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Reuse a previously generated fixed-views manifest without copying image files.

    The returned manifest is written into the current run's output directory for
    provenance, while each view item keeps its original image/metadata paths.
    This makes cached evidence explicit and avoids accidental regeneration when
    a new run writes to a fresh output directory.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "views_manifest.json"
    cache_dir = cache_dir.resolve()
    cache_manifest_path = cache_dir / "views_manifest.json"
    canonical_mapping = scene_state.get("topdown_grid_mapping", {}).get("canonical_mapping_path")
    if not canonical_mapping:
        return {
            "status": "missing_or_needs_generation",
            "reason": "scene_state.topdown_grid_mapping.canonical_mapping_path is missing",
            "views_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "cache_source_dir": str(cache_dir),
            "views": [],
        }
    canonical_mapping_path = Path(canonical_mapping)
    if not _current_manifest_ok(cache_manifest_path, str(canonical_mapping_path), VIEW_DIRECTIONS):
        return {
            "status": "missing_or_needs_generation",
            "reason": f"fixed views cache is missing, stale, or incomplete: {cache_manifest_path}",
            "views_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "cache_source_dir": str(cache_dir),
            "cache_manifest_path": str(cache_manifest_path),
            "canonical_grid_mapping_path": str(canonical_mapping_path),
            "views": [],
        }

    manifest = _load_json(cache_manifest_path)
    for item in manifest.get("views", []):
        item["reused_existing"] = True
        item["cache_source_dir"] = str(cache_dir)
    manifest["status"] = "available"
    manifest["reused_existing"] = True
    manifest["cache_status"] = "reused_from_fixed_views_cache_dir"
    manifest["cache_source_dir"] = str(cache_dir)
    manifest["cache_manifest_path"] = str(cache_manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest["current_run_views_manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    _write_manifest_md(manifest_path, manifest)
    _write_render_log(scene_state, output_dir, [], manifest)
    return manifest


def _render_python(scene_state: dict[str, Any]) -> str:
    configured = scene_state.get("evidence", {}).get("render_python")
    if configured and Path(configured).exists():
        return str(configured)
    return sys.executable


def _scene_value(scene_state: dict[str, Any], key: str) -> str:
    value = scene_state.get("evidence", {}).get(key)
    if not value:
        raise ValueError(f"scene_state.evidence.{key} is missing")
    return str(value)


def _direction_args(direction: list[float]) -> list[str]:
    return [str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in direction]


def _base_command(render_python: str, render_script: str, layout_dir: str, unposed_dir: str, direction: list[float], output: Path) -> list[str]:
    return [
        render_python,
        render_script,
        "--layout_dir",
        layout_dir,
        "--unposed_dir",
        unposed_dir,
        "--camera_direction",
        *_direction_args(direction),
        "--output",
        str(output),
        "--image_size",
        str(DEFAULT_IMAGE_SIZE),
    ]


def _write_view_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, metadata)


def _run_command(command: list[str], dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "skipped_dry_run"}
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def ensure_fixed_layout_views(scene_state: dict[str, Any], output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "views_manifest.json"
    canonical_mapping = scene_state.get("topdown_grid_mapping", {}).get("canonical_mapping_path")
    if not canonical_mapping:
        return {
            "status": "missing_or_needs_generation",
            "reason": "scene_state.topdown_grid_mapping.canonical_mapping_path is missing",
            "views_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "views": [],
        }
    canonical_mapping_path = Path(canonical_mapping)
    if not canonical_mapping_path.exists():
        return {
            "status": "missing_or_needs_generation",
            "reason": f"canonical grid mapping does not exist: {canonical_mapping_path}",
            "views_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "canonical_grid_mapping_path": str(canonical_mapping_path),
            "views": [],
        }

    render_script = _scene_value(scene_state, "render_script")
    layout_dir = _scene_value(scene_state, "layout_params_dir")
    unposed_dir = _scene_value(scene_state, "unposed_objects_dir")
    render_python = _render_python(scene_state)

    if _current_manifest_ok(manifest_path, str(canonical_mapping_path), VIEW_DIRECTIONS):
        manifest = _load_json(manifest_path)
        for item in manifest.get("views", []):
            item["reused_existing"] = True
        manifest["status"] = "available"
        manifest["reused_existing"] = True
        write_json(manifest_path, manifest)
        _write_manifest_md(manifest_path, manifest)
        _write_render_log(scene_state, output_dir, [], manifest)
        return manifest

    calls: list[dict[str, Any]] = []
    view_items: list[dict[str, Any]] = []
    any_failed = False

    for view, direction in VIEW_DIRECTIONS.items():
        view_dir = output_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = view_dir / f"{view}_view.json"

        if view == "top":
            clean_image_path = view_dir / "top_view_clean.png"
            grid_image_path = view_dir / "top_view_grid.png"
            grid_render_metadata_path = grid_image_path.with_suffix(".json")
            clean_command = _base_command(render_python, render_script, layout_dir, unposed_dir, direction, clean_image_path)
            grid_command = _base_command(render_python, render_script, layout_dir, unposed_dir, direction, grid_image_path)
            grid_command.extend(["--add_grid", "--draw_grid_lines", "--grid_mapping", str(canonical_mapping_path)])

            item = {
                "view": view,
                "image_path": str(clean_image_path),
                "clean_image_path": str(clean_image_path),
                "grid_image_path": str(grid_image_path),
                "metadata_path": str(metadata_path),
                "grid_render_metadata_path": str(grid_render_metadata_path),
                "camera_direction": direction,
                "render_script": render_script,
                "render_python": render_python,
                "canonical_grid_mapping_path": str(canonical_mapping_path),
                "uses_canonical_grid_mapping": True,
                "draw_grid_overlay": True,
                "is_clean_view": False,
                "image_size": DEFAULT_IMAGE_SIZE,
                "grid_mapping_usage_note": TOP_VIEW_USAGE_NOTE,
                "notes": TOP_VIEW_USAGE_NOTE,
                "reused_existing": False,
                "status": "skipped_dry_run" if dry_run else "pending",
            }
            top_calls = [
                {
                    "event": "fixed_layout_view_render",
                    "view": view,
                    "variant": "clean",
                    "camera_direction": direction,
                    "output_png": str(clean_image_path),
                    "output_json": str(metadata_path),
                    "uses_canonical_grid_mapping": True,
                    "passes_grid_mapping_to_renderer": False,
                    "draw_grid_overlay": False,
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "canonical_grid_mapping_path": str(canonical_mapping_path),
                    "grid_mapping_usage_note": TOP_VIEW_USAGE_NOTE,
                    "command": clean_command,
                    "status": "skipped_dry_run" if dry_run else "pending",
                },
                {
                    "event": "fixed_layout_view_render",
                    "view": view,
                    "variant": "grid",
                    "camera_direction": direction,
                    "output_png": str(grid_image_path),
                    "output_json": str(grid_render_metadata_path),
                    "uses_canonical_grid_mapping": True,
                    "passes_grid_mapping_to_renderer": True,
                    "draw_grid_overlay": True,
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "canonical_grid_mapping_path": str(canonical_mapping_path),
                    "grid_mapping_usage_note": TOP_VIEW_USAGE_NOTE,
                    "command": grid_command,
                    "status": "skipped_dry_run" if dry_run else "pending",
                },
            ]
            if not dry_run:
                top_failed = False
                for call in top_calls:
                    result = _run_command(call["command"], dry_run=False)
                    call.update(result)
                    if result.get("status") != "success":
                        top_failed = True
                _write_view_metadata(
                    metadata_path,
                    {
                        "view": view,
                        "camera_direction": direction,
                        "clean_image_path": str(clean_image_path),
                        "grid_image_path": str(grid_image_path),
                        "grid_render_metadata_path": str(grid_render_metadata_path),
                        "canonical_grid_mapping_path": str(canonical_mapping_path),
                        "uses_canonical_grid_mapping": True,
                        "draw_grid_overlay": True,
                        "clean_draw_grid_overlay": False,
                        "grid_draw_grid_overlay": True,
                        "image_size": DEFAULT_IMAGE_SIZE,
                        "notes": TOP_VIEW_USAGE_NOTE,
                    },
                )
                if top_failed or not clean_image_path.exists() or not grid_image_path.exists() or not metadata_path.exists():
                    any_failed = True
                    item["status"] = "failed"
                else:
                    item["status"] = "success"
            calls.extend(top_calls)
            view_items.append(item)
            continue

        image_path = view_dir / f"{view}_view.png"
        command = _base_command(render_python, render_script, layout_dir, unposed_dir, direction, image_path)
        item = {
            "view": view,
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "camera_direction": direction,
            "render_script": render_script,
            "render_python": render_python,
            "canonical_grid_mapping_path": str(canonical_mapping_path),
            "uses_canonical_grid_mapping": True,
            "draw_grid_overlay": False,
            "is_clean_view": True,
            "image_size": DEFAULT_IMAGE_SIZE,
            "grid_mapping_usage_note": GRID_MAPPING_USAGE_NOTE,
            "reused_existing": False,
            "status": "skipped_dry_run" if dry_run else "pending",
        }
        call = {
            "event": "fixed_layout_view_render",
            "view": view,
            "camera_direction": direction,
            "output_png": str(image_path),
            "output_json": str(metadata_path),
            "uses_canonical_grid_mapping": True,
            "passes_grid_mapping_to_renderer": False,
            "draw_grid_overlay": False,
            "image_size": DEFAULT_IMAGE_SIZE,
            "canonical_grid_mapping_path": str(canonical_mapping_path),
            "grid_mapping_usage_note": GRID_MAPPING_USAGE_NOTE,
            "command": command,
            "status": "skipped_dry_run" if dry_run else "pending",
        }
        if not dry_run:
            result = _run_command(command, dry_run=False)
            call.update(result)
            item["status"] = result["status"]
            if result["status"] == "success" and image_path.exists():
                _write_view_metadata(
                    metadata_path,
                    {
                        "view": view,
                        "camera_direction": direction,
                        "image_path": str(image_path),
                        "canonical_grid_mapping_path": str(canonical_mapping_path),
                        "uses_canonical_grid_mapping": True,
                        "draw_grid_overlay": False,
                        "is_clean_view": True,
                        "passes_grid_mapping_to_renderer": False,
                        "grid_mapping_usage_note": GRID_MAPPING_USAGE_NOTE,
                        "image_size": DEFAULT_IMAGE_SIZE,
                    },
                )
            if result["status"] != "success" or not image_path.exists() or not metadata_path.exists():
                any_failed = True
        calls.append(call)
        view_items.append(item)

    manifest = {
        "status": "failed" if any_failed else "available",
        "scene_name": scene_state.get("scene_name"),
        "views_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "canonical_grid_mapping_path": str(canonical_mapping_path),
        "uses_canonical_grid_mapping": True,
        "render_script": render_script,
        "render_python": render_python,
        "view_convention": "front/back/left/right are render camera directions in SceneMosaic world XYZ; canonical grid remains top-down [x,z].",
        "grid_mapping_usage_note": GRID_MAPPING_USAGE_NOTE,
        "side_view_policy": "side views record the canonical grid mapping as shared top-down [x,z] reference metadata only; grid mapping is not passed to the side-view renderer and no grid overlay is drawn.",
        "image_size": DEFAULT_IMAGE_SIZE,
        "views": view_items,
    }
    write_json(manifest_path, manifest)
    _write_manifest_md(manifest_path, manifest)
    _write_render_log(scene_state, output_dir, calls, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed multi-view layout evidence for multi-agent attach reasoning.")
    parser.add_argument("--scene_state", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    scene_state = json.loads(args.scene_state.read_text(encoding="utf-8"))
    manifest = ensure_fixed_layout_views(scene_state, args.output_dir, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
