from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.state import sorted_object_id_key, write_json, write_md
from vlm_agent.run_paths import ScenePaths


def _load_json_if_exists(path: Path | None) -> Any | None:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Async relation precompute may have created the manifest before its
        # atomic payload is fully visible. Treat that short state as not ready;
        # the runner's normal readiness polling will load it later.
        return None


def _detect_render_python() -> str:
    env_python = os.environ.get("SCENEMOSAIC_RENDER_PYTHON")
    if env_python and Path(env_python).exists():
        return str(Path(env_python))
    return sys.executable


def _list_files(directory: Path | None, suffixes: tuple[str, ...]) -> list[str]:
    if not directory or not directory.exists():
        return []
    return sorted(str(p) for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes)




def _json_has_grid_points(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("grid_points"), list) and bool(data["grid_points"])


def _detect_relation_precompute(run_root: Path) -> dict[str, Any]:
    explicit_dir = os.environ.get("RELATION_LAYOUT_DIR")
    if explicit_dir:
        candidates = [(Path(explicit_dir).expanduser().resolve() / "manifest.json", "explicit_relation_layout")]
    else:
        default_dir = ScenePaths.from_scene_root(run_root).relations_dir / "relation_layout"
        candidates = [(default_dir / "manifest.json", "relation_layout")]
    missing_paths = []
    for manifest_path, source in candidates:
        data = _load_json_if_exists(manifest_path)
        if not isinstance(data, dict):
            missing_paths.append(str(manifest_path))
            continue
        outputs = data.get("outputs") if isinstance(data.get("outputs"), dict) else {}
        return {
            "status": data.get("status"),
            "source": source,
            "manifest_path": str(manifest_path),
            "precompute_dir": data.get("precompute_dir"),
            "formal_wall_method": data.get("formal_wall_method"),
            "outputs": outputs,
            "stages": data.get("stages"),
        }
    return {"status": "missing", "source": None, "manifest_path": missing_paths[0] if missing_paths else None, "checked_manifest_paths": missing_paths}


def _detect_topdown_grid_mapping(run_root: Path, topdown_json: list[str]) -> dict[str, Any]:
    relation_precompute = _detect_relation_precompute(run_root)
    outputs = relation_precompute.get("outputs") if isinstance(relation_precompute, dict) else {}
    if isinstance(outputs, dict):
        mapping = outputs.get("canonical_grid_mapping_path")
        image = outputs.get("canonical_topdown_image_path")
        affine = outputs.get("render_grid_affine_path")
        method_candidates = outputs.get("method_candidates_path")
        if mapping and image and Path(str(mapping)).exists() and Path(str(image)).exists() and _json_has_grid_points(Path(str(mapping))):
            return {
                "canonical_mapping_path": str(mapping),
                "canonical_topdown_image_path": str(image),
                "source": relation_precompute.get("source") or "relation_precompute",
                "fallback_used": False,
                "status": "available",
                "render_grid_affine_path": str(affine) if affine else None,
                "method_candidates_path": str(method_candidates) if method_candidates else None,
                "relation_precompute_manifest_path": relation_precompute.get("manifest_path"),
            }
    return {
        "canonical_mapping_path": None,
        "canonical_topdown_image_path": None,
        "source": None,
        "fallback_used": None,
        "status": "missing_relation_layout_precompute",
    }


def _detect_fixed_layout_views(run_root: Path, topdown_grid_mapping: dict[str, Any]) -> dict[str, Any]:
    relation_precompute = _detect_relation_precompute(run_root)
    outputs = relation_precompute.get("outputs") if isinstance(relation_precompute, dict) else {}
    canonical_mapping = topdown_grid_mapping.get("canonical_mapping_path") if isinstance(topdown_grid_mapping, dict) else None
    if isinstance(outputs, dict):
        relation_manifest = outputs.get("fixed_views_manifest_path")
        relation_views_dir = outputs.get("fixed_views_dir")
        if relation_manifest and Path(str(relation_manifest)).exists():
            manifest = _load_json_if_exists(Path(str(relation_manifest)))
            if isinstance(manifest, dict) and manifest.get("status") == "available":
                manifest = dict(manifest)
                manifest["source"] = relation_precompute.get("source") or "relation_precompute"
                manifest["relation_precompute_manifest_path"] = relation_precompute.get("manifest_path")
                manifest["views_dir"] = manifest.get("views_dir") or relation_views_dir
                manifest["manifest_path"] = str(relation_manifest)
                return manifest
    views_dir = ScenePaths.from_scene_root(run_root).relations_dir / "relation_layout" / "fixed_views"
    return {
        "status": "missing_or_disabled",
        "views_dir": str(views_dir),
        "manifest_path": str(views_dir / "views_manifest.json"),
        "canonical_grid_mapping_path": canonical_mapping,
        "views": [],
    }


def _snake_case(value: str | None) -> str:
    value = (value or "unknown_object").strip()
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "unknown_object"


def _index_reviewed_objects(reviewed_objects_path: Path | None) -> dict[str, dict[str, Any]]:
    data = _load_json_if_exists(reviewed_objects_path)
    if not isinstance(data, dict):
        return {}
    indexed = {}
    for item in data.get("objects", []):
        object_id = item.get("object_id") or item.get("id")
        if object_id:
            indexed[str(object_id)] = item
    return indexed


def _find_mask(mask_dir: Path | None, object_id: str) -> str | None:
    if not mask_dir or not mask_dir.exists():
        return None
    matches = sorted(mask_dir.glob(f"{object_id}_*_mask.png")) + sorted(mask_dir.glob(f"{object_id}.png"))
    return str(matches[0]) if matches else None


def _find_overlay(overlay_dir: Path | None, object_id: str) -> str | None:
    if not overlay_dir or not overlay_dir.exists():
        return None
    matches = sorted(overlay_dir.glob(f"{object_id}_*_overlay.png")) + sorted(overlay_dir.glob(f"{object_id}.png"))
    return str(matches[0]) if matches else None


def build_scene_state(scene_name: str, run_root: Path, render_script: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    scene_paths = ScenePaths.from_scene_root(run_root)
    default_relation_layout = scene_paths.relations_dir / "relation_layout"
    relation_precompute = _detect_relation_precompute(run_root)
    relation_outputs = relation_precompute.get("outputs") if isinstance(relation_precompute.get("outputs"), dict) else {}
    relation_layout_path = relation_outputs.get("layout_for_relation_path")
    layout_for_relation_path = Path(str(relation_layout_path)) if relation_layout_path else (default_relation_layout / "layout_for_relation.json")
    layout_prompt_path = default_relation_layout / "layout_for_relation_prompt.json"
    reviewed_objects_path = scene_paths.perception_dir / "reviewed_objects.json"
    reviewed_masks_dir = scene_paths.perception_dir / "reviewed_masks"
    reviewed_overlays_dir = scene_paths.perception_dir / "reviewed_overlays"
    canonical_topdown_path = relation_outputs.get("canonical_topdown_image_path")
    topdown_dir = Path(str(canonical_topdown_path)).parent if canonical_topdown_path else (default_relation_layout / "render_grid_wall")
    layout_params_dir = scene_paths.sam3d_output_dir / "layout_params"
    unposed_objects_dir = scene_paths.sam3d_output_dir / "unposed_objects"
    sam3d_input_masks_dir = scene_paths.sam3d_input_dir / "masks"

    layout_data = _load_json_if_exists(layout_for_relation_path)
    reviewed_index = _index_reviewed_objects(reviewed_objects_path)

    object_items: list[dict[str, Any]] = []
    layout_objects = layout_data.get("objects", []) if isinstance(layout_data, dict) else []
    for raw in layout_objects:
        object_id = str(raw.get("object_id") or raw.get("id") or "")
        if not object_id:
            continue
        reviewed = reviewed_index.get(object_id, {})
        category = _snake_case(raw.get("category") or reviewed.get("category"))
        description = raw.get("description") or reviewed.get("description") or category.replace("_", " ")
        mask_path = reviewed.get("sam3", {}).get("mask_path") or _find_mask(reviewed_masks_dir, object_id)
        overlay_path = reviewed.get("sam3", {}).get("overlay_path") or _find_overlay(reviewed_overlays_dir, object_id)
        layout_params_path = layout_params_dir / f"{object_id}.json"
        unposed_path = unposed_objects_dir / f"{object_id}.glb"
        object_items.append(
            {
                "object_id": object_id,
                "category": category,
                "description": description,
                "bbox_2d": raw.get("image_bbox") or {"bbox_2d_norm_xyxy": reviewed.get("bbox_2d_norm_xyxy")},
                "bbox_3d": raw.get("bbox_global"),
                "layout_reference": {
                    "orientation_for_agent": raw.get("orientation_for_agent"),
                    "coordinate_system": layout_data.get("coordinate_system") if isinstance(layout_data, dict) else None,
                },
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "layout_params_path": str(layout_params_path) if layout_params_path.exists() else None,
                "unposed_object_path": str(unposed_path) if unposed_path.exists() else None,
                "evidence_status": {
                    "has_mask": bool(mask_path),
                    "has_layout": layout_params_path.exists() and raw.get("bbox_global") is not None,
                    "has_3d_asset": unposed_path.exists(),
                },
            }
        )

    object_items.sort(key=lambda item: sorted_object_id_key(item["object_id"]))
    grounding_items = [
        {
            "object_id": item["object_id"],
            "primary_grounding": "reviewed mask/overlay when available; VLM only for conflict review",
            "mask_path": item.get("mask_path"),
            "overlay_path": item.get("overlay_path"),
            "status": "grounded_by_existing_labels" if item.get("mask_path") else "missing_mask_needs_review",
        }
        for item in object_items
    ]

    topdown_png = _list_files(topdown_dir, (".png",))
    topdown_json = _list_files(topdown_dir, (".json",))
    topdown_grid_mapping = _detect_topdown_grid_mapping(run_root, topdown_json)
    fixed_layout_views = _detect_fixed_layout_views(run_root, topdown_grid_mapping)
    missing_inputs = []
    for label, path in [
        ("layout_for_relation_path", layout_for_relation_path),
        ("reviewed_objects_path", reviewed_objects_path),
        ("layout_params_dir", layout_params_dir),
        ("unposed_objects_dir", unposed_objects_dir),
        ("topdown_dir", topdown_dir),
        ("render_script", render_script),
    ]:
        if not path.exists():
            missing_inputs.append({"name": label, "path": str(path)})

    warnings = []
    if not object_items:
        warnings.append(f"No object candidates were extracted from {layout_for_relation_path}.")
    if not topdown_png:
        warnings.append(f"No top-down PNG evidence found under {topdown_dir}.")

    return {
        "scene_name": scene_name,
        "run_root": str(run_root),
        "evidence": {
            "scene_config_path": str(run_root / "input" / "scene_config.json") if (run_root / "input" / "scene_config.json").exists() else None,
            "original_image_path": str(run_root / "input" / "image.png") if (run_root / "input" / "image.png").exists() else None,
            "mask_overlay_path": str(scene_paths.perception_dir / "reviewed_overlay.png") if (scene_paths.perception_dir / "reviewed_overlay.png").exists() else None,
            "reviewed_objects_path": str(reviewed_objects_path) if reviewed_objects_path.exists() else None,
            "object_masks_dir": str(reviewed_masks_dir) if reviewed_masks_dir.exists() else None,
            "sam3d_input_masks_dir": str(sam3d_input_masks_dir) if sam3d_input_masks_dir.exists() else None,
            "layout_for_relation_path": str(layout_for_relation_path) if layout_for_relation_path.exists() else None,
            "layout_for_relation_prompt_path": str(layout_prompt_path) if layout_prompt_path.exists() else None,
            "layout_params_dir": str(layout_params_dir) if layout_params_dir.exists() else None,
            "unposed_objects_dir": str(unposed_objects_dir) if unposed_objects_dir.exists() else None,
            "topdown_dir": str(topdown_dir) if topdown_dir.exists() else None,
            "topdown_assets": {"png_files": topdown_png, "json_files": topdown_json},
            "render_script": str(render_script),
            "render_python": _detect_render_python(),
            "fixed_layout_views": fixed_layout_views,
            "relation_precompute": relation_precompute,
        },
        "topdown_grid_mapping": topdown_grid_mapping,
        "object_candidates": {
            "source_priority": ["relation_layout", "sam3d_output", "reviewed_masks"],
            "items": object_items,
            "status": "available_needs_validation" if object_items else "missing",
        },
        "object_grounding": {
            "policy": "mask_overlay_and_per_object_masks_are_primary",
            "items": grounding_items,
        },
        "structure_targets": {
            "walls": {
                "status": "to_be_generated",
                "required_ids": ["wall_001", "wall_002", "wall_003", "wall_004"],
                "coordinate_format": "[x, z]",
                "clockwise_required": True,
                "source_evidence": ["topdown_with_coordinates"],
            },
            "floor": {"status": "assumed_present", "id": "floor"},
            "ceiling": {"status": "unknown", "requires_agent_judgement": True},
        },
        "relation_targets": {
            "allowed_relation_types": ["attach", "contain"],
            "attach_schema": {"relation": "attach", "parameters": {"object_anchor": "...", "object_target": "..."}},
            "contain_schema": {"relation": "contain", "parameters": {"object_anchor": "...", "object_target": "..."}},
            "contain_policy": "Contain is same-level with attach. For the same object_anchor/object_target pair, emit contain instead of attach when the target is inside the anchor's storage/display/interior volume.",
        },
        "working_state": {
            "tasks": [],
            "task_results": [],
            "wall_candidates": [],
            "accepted_walls": [],
            "ceiling_decision": None,
            "object_revision_candidates": [],
            "attach_candidates": [],
            "accepted_relations": [],
            "rejected_relations": [],
            "generated_evidence": [],
            "warnings": [],
        },
        "deliverables": {
            "final_outputs_dir": "final_outputs",
            "objects_json_path": "final_outputs/objects.json",
            "relations_json_path": "final_outputs/relations.json",
        },
        "constraints": {
            "do_not_use_old_workflow_outputs_for_inference": True,
            "final_objects_keep_sam3d_object_ids": True,
            "final_objects_match_scene_state_candidate_ids": True,
            "do_not_add_delete_merge_or_renumber_objects": True,
            "object_id_numbering_inherited_from_sam3d_layout": True,
            "object_id_base_may_be_zero_or_one": True,
            "object_id_format": "object_<number>",
            "final_objects_sorted_by_numeric_suffix": True,
            "category_description_changes_require_object_revision_audit": True,
            "final_relations_allow_attach_and_contain": True,
            "ceiling_anchor_requires_positive_judgement": True,
        },
        "missing_inputs": missing_inputs,
        "warnings": warnings,
    }


def write_scene_state(scene_state: dict[str, Any], output_dir: Path) -> None:
    state_path = output_dir / "scene_states" / "scene_state.json"
    summary_path = output_dir / "scene_states" / "scene_state_summary.md"
    write_json(state_path, scene_state)
    candidates = scene_state["object_candidates"]["items"]
    lines = [
        f"- scene_name: `{scene_state['scene_name']}`",
        f"- run_root: `{scene_state['run_root']}`",
        f"- object_candidates: `{len(candidates)}`",
        f"- object_id_range: `{candidates[0]['object_id'] if candidates else 'none'}` to `{candidates[-1]['object_id'] if candidates else 'none'}`",
        f"- missing_inputs: `{len(scene_state['missing_inputs'])}`",
        f"- warnings: `{len(scene_state['warnings'])}`",
        "",
        "## Evidence",
        f"- layout_for_relation: `{scene_state['evidence']['layout_for_relation_path']}`",
        f"- reviewed_objects: `{scene_state['evidence']['reviewed_objects_path']}`",
        f"- topdown_assets_png: `{len(scene_state['evidence']['topdown_assets']['png_files'])}`",
        f"- topdown_assets_json: `{len(scene_state['evidence']['topdown_assets']['json_files'])}`",
        f"- topdown_grid_mapping_status: `{scene_state.get('topdown_grid_mapping', {}).get('status')}`",
        f"- topdown_grid_mapping_path: `{scene_state.get('topdown_grid_mapping', {}).get('canonical_mapping_path')}`",
        f"- fixed_layout_views_status: `{scene_state.get('evidence', {}).get('fixed_layout_views', {}).get('status')}`",
        f"- fixed_layout_views_manifest: `{scene_state.get('evidence', {}).get('fixed_layout_views', {}).get('manifest_path')}`",
        "",
        "## Important Constraint",
        "This scene_state intentionally does not load old final objects.json, old relations.json, old attach pass outputs, or old wall extraction results as inference inputs.",
        "Final objects must preserve exactly the object ids from scene_state.object_candidates; numbering is inherited from SAM3D/layout and may be 0-based or 1-based.",
    ]
    write_md(summary_path, "Scene State Summary", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene_state.json for multi-agent scene delivery.")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--render_script", type=Path, default=REPOSITORY_ROOT / "vlm_agent" / "evolution" / "tools" / "render_scene_from_layout.py")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional direct scene_state.json output path.")
    args = parser.parse_args()
    if args.output_dir is None and args.output is None:
        parser.error("one of --output_dir or --output is required")
    scene_state = build_scene_state(args.scene_name, args.run_root, args.render_script)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, scene_state)
        print(args.output)
    else:
        write_scene_state(scene_state, args.output_dir)
        print(args.output_dir / "scene_states" / "scene_state.json")


if __name__ == "__main__":
    main()
