from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def scan_render_script_capabilities(render_script: Path) -> dict[str, Any]:
    """Describe the SceneMosaic render adapter without executing it."""
    text = render_script.read_text(encoding="utf-8") if render_script.exists() else ""
    arguments = {}
    for name in [
        "layout_dir",
        "unposed_dir",
        "object_ids",
        "anchor_id",
        "camera_direction",
        "radius",
        "output",
        "image_size",
        "add_grid",
        "grid_mapping",
        "grid_size",
        "draw_grid_lines",
        "plane_side",
    ]:
        arguments[name] = f"--{name}" in text
    return {
        "render_script": str(render_script),
        "exists": render_script.exists(),
        "supported_arguments": arguments,
        "capabilities": {
            "select_partial_objects": arguments["object_ids"],
            "topdown_render": arguments["camera_direction"],
            "default_topdown_direction": [0.0, -1.0, 0.0] if "[0.0, -1.0, 0.0]" in text else None,
            "reuse_grid_mapping": arguments["grid_mapping"],
            "output_image": arguments["output"],
            "output_grid_json_when_add_grid": 'with_suffix(".json")' in text or "with_suffix('.json')" in text,
            "preserve_coordinate_grid": arguments["add_grid"] and arguments["draw_grid_lines"],
            "include_walls_flag": False,
            "include_floor_flag": False,
        },
        "notes": [
            "Production relation runs consume canonical top-down and grid mapping from the candidate-local relation layout.",
            "The multi-agent runner does not regenerate top-down evidence.",
        ],
    }


def _mapping_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and (
        isinstance(payload.get("grid_points"), list)
        or isinstance(payload.get("projected_grid_samples"), list)
        or isinstance(payload.get("xz_to_pixel_affine"), list)
    )


def ensure_canonical_topdown_mapping(
    scene_state: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate externally precomputed top-down evidence.

    Regeneration belongs to build_relation_layout_from_sam3d.py.
    """
    mapping = scene_state.get("topdown_grid_mapping", {}) if isinstance(scene_state.get("topdown_grid_mapping"), dict) else {}
    mapping_path = Path(str(mapping.get("canonical_mapping_path"))) if mapping.get("canonical_mapping_path") else None
    image_path = Path(str(mapping.get("canonical_topdown_image_path"))) if mapping.get("canonical_topdown_image_path") else None
    available = bool(mapping_path and _mapping_is_valid(mapping_path) and image_path and image_path.exists())
    return {
        "status": "available" if available else "missing_required_relation_precompute",
        "generated": False,
        "canonical_mapping_path": str(mapping_path) if mapping_path else None,
        "canonical_topdown_image_path": str(image_path) if image_path else None,
        "source": mapping.get("source"),
        "fallback_used": False,
        "dry_run": dry_run,
        "reason": None if available else "Run build_relation_layout_from_sam3d.py before multi-agent relation extraction.",
    }
