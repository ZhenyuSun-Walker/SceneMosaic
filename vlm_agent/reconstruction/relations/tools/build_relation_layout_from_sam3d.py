#!/usr/bin/env python3
"""Build the relation-ready layout package directly from SAM3D outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
POLY_TOOLS = REPOSITORY_ROOT / 'vlm_agent' / 'evolution' / 'tools'
if str(POLY_TOOLS) not in sys.path:
    sys.path.insert(0, str(POLY_TOOLS))

from relations.agents.state import sorted_object_id_key, write_json, write_md  # noqa: E402
from relations.tools.build_scene_state import build_scene_state  # noqa: E402
from relations.tools.extract_wall_evidence import build_wall_evidence, write_wall_evidence  # noqa: E402
from relations.tools.generate_fixed_layout_views import ensure_fixed_layout_views  # noqa: E402
from relations.tools.wall_geometry import build_obb_walls  # noqa: E402
from vlm_agent.run_paths import ScenePaths  # noqa: E402

DEFAULT_RENDER_SCRIPT = POLY_TOOLS / 'render_scene_from_layout.py'
PRIMARY_METHOD = 'obb_from_mesh_footprint_min_rect'
FRONT_RENDER_DIRECTION = 'front'
TOP_RENDER_DIRECTION = 'up'
MAX_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR = 16
MAX_HIGH_CONFIDENCE_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR = 32
HIGH_CONFIDENCE_CONTAIN_CANDIDATE_THRESHOLD = 0.95

CONTAINER_KEYWORDS = {
    'basket',
    'bin',
    'bookcase',
    'bookshelf',
    'bunk bed',
    'bowl',
    'box',
    'cabinet',
    'case',
    'closet',
    'crate',
    'cubby',
    'drawer',
    'display cabinet',
    'display case',
    'freezer',
    'fridge',
    'loft bed',
    'multi-tier',
    'open shelf',
    'rack',
    'refrigerator',
    'shelf',
    'shelving',
    'storage',
    'tiered',
    'tray',
    'wardrobe',
}

TOP_VIEW_CONTAINER_KEYWORDS = {
    'basket',
    'bin',
    'bowl',
    'box',
    'crate',
    'drawer',
    'open-top',
    'tray',
}

FRONT_VIEW_CONTAINER_KEYWORDS = {
    'bookcase',
    'bookshelf',
    'bunk bed',
    'cabinet',
    'case',
    'closet',
    'cubby',
    'display',
    'freezer',
    'fridge',
    'loft bed',
    'multi-tier',
    'open shelf',
    'rack',
    'refrigerator',
    'shelf',
    'shelving',
    'tiered',
    'wardrobe',
}


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path)


def load_layout_param_records(layout_params_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(layout_params_dir.glob('object_*.json'), key=lambda p: sorted_object_id_key(p.stem)):
        data = load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        object_id = str(data.get('object_id') or path.stem)
        records.append(
            {
                'object_id': object_id,
                'category': data.get('category') or 'unknown_object',
                'description': data.get('description') or data.get('category') or object_id,
                'source': data.get('source'),
                'layout_params_path': str(path),
                'source_mask_path': data.get('source_mask_path'),
                'vlm_bbox_2d_norm_xyxy': data.get('vlm_bbox_2d_norm_xyxy'),
                'sam3_mask_bbox_abs_xyxy': data.get('sam3_mask_bbox_abs_xyxy'),
                'posed_glb': data.get('posed_glb'),
                'unposed_glb': data.get('unposed_glb'),
                'raw_layout_params': data,
            }
        )
    records.sort(key=lambda item: sorted_object_id_key(item['object_id']))
    return records


def artifact_contract(scene_name: str, run_root: Path, object_records: list[dict[str, Any]], output_dir: Path, include_fixed_views: bool = False) -> dict[str, Any]:
    artifacts = {
        'original_image': {
            'status': 'available' if (run_root / 'input' / 'image.png').exists() else 'missing',
            'description': 'Original input image. The async manager may use this single image with the object list to decide whether a contain-relation task is needed before slower relation artifacts finish.',
            'path': str(run_root / 'input' / 'image.png'),
        },
        'object_list': {
            'status': 'available',
            'description': 'SAM3D object ids, categories, short descriptions, original mask references, and image bboxes. This is enough for early manager DAG planning.',
            'path': str(output_dir / 'objects.json'),
        },
        'object_footprints': {
            'status': 'pending',
            'description': 'Per-object OpenGL XZ footprint polygons and 3D bounding boxes computed from posed SAM3D mesh vertices. Useful for floor/support/object contact and wall proximity.',
            'path': str(output_dir / 'object_footprints.json'),
        },
        'formal_obb_walls': {
            'status': 'pending',
            'description': 'Four screen-clockwise wall segments from the formal obb_from_mesh_footprint_min_rect method. This is the only formal wall coordinate source.',
            'path': str(output_dir / 'walls' / 'obb_walls.json'),
        },
        'canonical_topdown': {
            'status': 'pending',
            'description': 'Top-down render and grid mapping generated from the same render grid used by formal wall candidates and downstream wall visual evidence.',
            'path': str(output_dir / 'render_grid_wall' / 'canonical_topdown.png'),
        },
        'render_grid_affine': {
            'status': 'pending',
            'description': 'Pixel-to-OpenGL-XZ and OpenGL-XZ-to-pixel affine transforms for the canonical top-down image.',
            'path': str(output_dir / 'render_grid_wall' / 'render_grid_affine.json'),
        },
        'wall_evidence': {
            'status': 'pending',
            'description': 'Structured wall evidence package consumed by controlled wall tasks in the multi-agent runner.',
            'path': str(output_dir / 'walls' / 'wall_evidence.json'),
        },
        'contain_candidates': {
            'status': 'pending',
            'description': 'High-recall local hints for possible object-in-container relations. These candidates are not a closed set; the contain agent may add or reject pairs.',
            'path': str(output_dir / 'contain_candidates.json'),
        },
        'contain_visual_evidence': {
            'status': 'pending',
            'description': 'Optional original-image overlay/crops for local contain candidates.',
            'path': str(output_dir / 'contain_visual_evidence' / 'contain_candidates_overlay.png'),
        },
        'fixed_views': {
            'status': 'pending' if include_fixed_views else 'optional_disabled',
            'description': 'Optional fixed top/front/back/left/right scene views. Expensive and disabled by default; tasks must not require them unless include_fixed_views is enabled.',
            'path': str(output_dir / 'fixed_views' / 'views_manifest.json'),
        },
        'scene_state': {
            'status': 'pending',
            'description': 'Relation-ready scene state assembled from this candidate-local layout package.',
            'path': str(output_dir / 'scene_state.json'),
        },
    }
    return {
        'schema_version': 'relation_expected_manifest',
        'status': 'planning_ready',
        'scene_name': scene_name,
        'precompute_dir': str(output_dir),
        'created_by': 'vlm_agent.reconstruction.relations.tools.build_relation_layout_from_sam3d',
        'coordinate_frame': 'OpenGL original XZ; Y is vertical. Top-down PNG coordinates use x right and y downward.',
        'formal_wall_method': PRIMARY_METHOD,
        'object_count': len(object_records),
        'objects': [
            {
                'object_id': item['object_id'],
                'category': item.get('category'),
                'description': item.get('description'),
                'source_mask_path': item.get('source_mask_path'),
                'vlm_bbox_2d_norm_xyxy': item.get('vlm_bbox_2d_norm_xyxy'),
            }
            for item in object_records
        ],
        'artifacts': artifacts,
        'manager_planning_contract': {
            'manager_can_start_before_artifacts_are_available': True,
            'manager_should_plan_a_dag_skeleton_only': True,
            'manager_should_reference_required_artifacts_by_key': True,
            'hydration_step_binds_real_paths_later': True,
            'required_task_fields': ['task_id', 'task_type', 'canonical_task_kind', 'runner_action', 'runtime_class', 'depends_on', 'requires_artifacts'],
        },
    }


def write_expected_manifest(scene_name: str, run_root: Path, output_dir: Path, object_records: list[dict[str, Any]], include_fixed_views: bool = False) -> dict[str, Any]:
    manifest = artifact_contract(scene_name, run_root, object_records, output_dir, include_fixed_views=include_fixed_views)
    write_json(output_dir / 'expected_manifest.json', manifest)
    lines = [
        f"- status: `{manifest['status']}`",
        f"- scene_name: `{scene_name}`",
        f"- object_count: `{manifest['object_count']}`",
        f"- formal_wall_method: `{PRIMARY_METHOD}`",
        '',
        '## Artifact Contract',
    ]
    for key, value in manifest['artifacts'].items():
        lines.append(f"- `{key}`: `{value['status']}` - {value['description']}")
    write_md(output_dir / 'expected_manifest.md', 'Relation Expected Manifest', lines)
    return manifest


def build_scene_state_from_relation_layout(
    scene_name: str,
    run_root: Path,
    render_script: Path,
    relation_layout_dir: Path,
) -> dict[str, Any]:
    previous = os.environ.get("RELATION_LAYOUT_DIR")
    os.environ["RELATION_LAYOUT_DIR"] = str(relation_layout_dir.resolve())
    try:
        return build_scene_state(scene_name, run_root, render_script)
    finally:
        if previous is None:
            os.environ.pop("RELATION_LAYOUT_DIR", None)
        else:
            os.environ["RELATION_LAYOUT_DIR"] = previous


def denormalized_vertices(mesh: Any, transform_info: dict[str, Any]) -> np.ndarray:
    verts = np.asarray(getattr(mesh, 'vertices', []), dtype=np.float32)
    if verts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    from render_scene_from_layout import denormalize_point

    return np.asarray([denormalize_point(vertex, transform_info) for vertex in verts], dtype=np.float32)


def convex_hull_xz(points: np.ndarray) -> list[list[float]]:
    if points.size == 0:
        return []
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) <= 2:
        return [[float(point[0]), float(point[1])] for point in array]
    hull = cv2.convexHull(array.reshape(-1, 1, 2), clockwise=False, returnPoints=True)
    return [[float(point[0]), float(point[1])] for point in hull.reshape(-1, 2)]


def mesh_footprint_payload(vertices: np.ndarray) -> dict[str, Any]:
    if vertices.size == 0:
        return {
            'mesh_footprint_hull_xz': [],
            'mesh_floor_footprint_hull_xz': [],
            'mesh_floor_band_max_y': None,
            'mesh_floor_footprint_source': 'missing_mesh_vertices',
        }
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    height = float(max(mx[1] - mn[1], 0.0))
    floor_band_height = min(height, max(0.012, height * 0.20))
    floor_band_max_y = float(mn[1] + floor_band_height)
    floor_vertices = vertices[vertices[:, 1] <= floor_band_max_y + 1e-8]
    if len(floor_vertices) < 3:
        floor_vertices = vertices
        source = 'all_mesh_vertices_fallback'
    else:
        source = 'bottom_20_percent_mesh_vertices'
    return {
        'mesh_footprint_hull_xz': convex_hull_xz(vertices[:, [0, 2]]),
        'mesh_floor_footprint_hull_xz': convex_hull_xz(floor_vertices[:, [0, 2]]),
        'mesh_floor_band_max_y': floor_band_max_y,
        'mesh_floor_footprint_source': source,
    }


def bbox_payload(vertices: np.ndarray) -> dict[str, Any]:
    if vertices.size == 0:
        return {'available': False, 'reason': 'mesh_vertices_missing'}
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    center = (mn + mx) / 2.0
    extent = mx - mn
    x0, z0 = float(mn[0]), float(mn[2])
    x1, z1 = float(mx[0]), float(mx[2])
    return {
        'available': True,
        'coordinate_frame': 'OpenGL original XZ; Y is vertical',
        'bbox_global': {
            'min': [float(v) for v in mn.tolist()],
            'max': [float(v) for v in mx.tolist()],
            'center': [float(v) for v in center.tolist()],
            'extent': [float(v) for v in extent.tolist()],
        },
        'bbox_xz_min': [min(x0, x1), min(z0, z1)],
        'bbox_xz_max': [max(x0, x1), max(z0, z1)],
        'center_xz': [float(center[0]), float(center[2])],
        'footprint_polygon_xz': [[x0, z0], [x1, z0], [x1, z1], [x0, z1]],
        'extent_xz': [abs(x1 - x0), abs(z1 - z0)],
        'bottom_y': float(min(mn[1], mx[1])),
        'top_y': float(max(mn[1], mx[1])),
        'height_y': float(abs(mx[1] - mn[1])),
        **mesh_footprint_payload(vertices),
    }


def build_objects_from_sam3d(run_root: Path, mapping_objects: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    sam3d_output_dir = ScenePaths.from_scene_root(run_root).sam3d_output_dir
    layout_dir = sam3d_output_dir / 'layout_params'
    unposed_dir = sam3d_output_dir / 'unposed_objects'
    records = {item['object_id']: item for item in load_layout_param_records(layout_dir)}
    mapping_by_id = {str(item.get('object_id')): item for item in mapping_objects or [] if isinstance(item, dict) and item.get('object_id')}
    from render_scene_from_layout import compose_scene

    _, posed_meshes, layout_params, object_ids, transform_info = compose_scene(
        layout_dir=layout_dir,
        unposed_dir=unposed_dir,
        requested_object_ids=None,
        anchor_id=None,
    )
    objects: list[dict[str, Any]] = []
    footprints: list[dict[str, Any]] = []
    for mesh, object_id, layout in zip(posed_meshes, object_ids, layout_params):
        object_id = str(object_id)
        record = records.get(object_id, {'object_id': object_id})
        vertices = denormalized_vertices(mesh, transform_info)
        bbox = bbox_payload(vertices)
        mapping = mapping_by_id.get(object_id, {})
        obj = {
            'object_id': object_id,
            'category': record.get('category') or layout.get('category') or 'unknown_object',
            'description': record.get('description') or layout.get('description') or object_id,
            'source': record.get('source') or layout.get('source'),
            'image_bbox': record.get('vlm_bbox_2d_norm_xyxy'),
            'bbox_global': bbox.get('bbox_global'),
            'orientation_for_agent': {
                'render_direction': None,
                'rotation_degrees_topdown': mapping.get('rotation_degrees'),
                'source': 'render_scene_from_layout.save_grid_mapping.objects.rotation_degrees' if mapping else None,
            },
            'sam3d_paths': {
                'layout_params_path': str(sam3d_output_dir / 'layout_params' / f'{object_id}.json'),
                'posed_glb_path': str(sam3d_output_dir / 'posed_objects' / f'{object_id}.glb'),
                'unposed_glb_path': str(sam3d_output_dir / 'unposed_objects' / f'{object_id}.glb'),
                'source_mask_path': record.get('source_mask_path'),
            },
            'topdown_grid_stats': mapping,
        }
        fp = {
            'object_id': object_id,
            'category': obj['category'],
            'description': obj['description'],
            'source': 'sam3d_output mesh vertices via compose_scene',
            **{k: v for k, v in bbox.items() if k != 'bbox_global'},
            'bbox_global': bbox.get('bbox_global'),
            'topdown_grid_stats': mapping,
        }
        objects.append(obj)
        footprints.append(fp)
    objects.sort(key=lambda item: sorted_object_id_key(item['object_id']))
    footprints.sort(key=lambda item: sorted_object_id_key(item['object_id']))
    layout_for_relation = {
        'schema_version': 'relation_layout_objects',
        'coordinate_system': 'OpenGL original XZ; Y is vertical',
        'source': 'sam3d_output',
        'objects': objects,
        'walls': [],
    }
    object_footprints = {
        'schema_version': 'object_footprints',
        'coordinate_frame': 'OpenGL original XZ; Y is vertical',
        'source': 'sam3d_output mesh vertices via compose_scene',
        'object_count': len(footprints),
        'items': footprints,
    }
    return layout_for_relation, object_footprints



def _text_for_object(obj: dict[str, Any]) -> str:
    return f"{obj.get('category', '')} {obj.get('description', '')}".lower()


def _bbox3(obj: dict[str, Any]) -> dict[str, Any]:
    bbox = obj.get('bbox_global')
    return bbox if isinstance(bbox, dict) else {}


def _bbox_center_inside(inner: dict[str, Any], outer: dict[str, Any], margin_ratio: float = 0.08) -> bool:
    center = inner.get('center')
    mn = outer.get('min')
    mx = outer.get('max')
    extent = outer.get('extent') or []
    if not (isinstance(center, list) and isinstance(mn, list) and isinstance(mx, list) and len(center) >= 3 and len(mn) >= 3 and len(mx) >= 3):
        return False
    margins = []
    for i in range(3):
        if isinstance(extent, list) and len(extent) >= 3:
            margins.append(max(abs(float(extent[i])) * margin_ratio, 0.03))
        else:
            margins.append(0.03)
    return all(float(mn[i]) - margins[i] <= float(center[i]) <= float(mx[i]) + margins[i] for i in range(3))


def _bbox_smaller_than(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    ie = inner.get('extent')
    oe = outer.get('extent')
    if not (isinstance(ie, list) and isinstance(oe, list) and len(ie) >= 3 and len(oe) >= 3):
        return False
    # Containers are often thin in one dimension, so use volume/height plus at least
    # one horizontal axis instead of requiring every axis to be smaller.
    inner_volume = max(float(ie[0]), 0.001) * max(float(ie[1]), 0.001) * max(float(ie[2]), 0.001)
    outer_volume = max(float(oe[0]), 0.001) * max(float(oe[1]), 0.001) * max(float(oe[2]), 0.001)
    horizontal_smaller = float(ie[0]) <= float(oe[0]) * 1.15 or float(ie[2]) <= float(oe[2]) * 1.15
    return inner_volume <= outer_volume * 0.75 and horizontal_smaller


def _norm_image_bbox(obj: dict[str, Any]) -> list[float] | None:
    bbox = obj.get('image_bbox')
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(v) for v in bbox]
    if isinstance(bbox, dict):
        raw = bbox.get('bbox_2d_norm_xyxy') or bbox.get('vlm_bbox_2d_norm_xyxy')
        if isinstance(raw, list) and len(raw) == 4:
            return [float(v) for v in raw]
    return None


def _image_center_inside(inner_obj: dict[str, Any], outer_obj: dict[str, Any], margin: float = 0.035) -> bool:
    inner = _norm_image_bbox(inner_obj)
    outer = _norm_image_bbox(outer_obj)
    if not inner or not outer:
        return False
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] - margin <= cx <= outer[2] + margin and outer[1] - margin <= cy <= outer[3] + margin


def _is_container_like(obj: dict[str, Any]) -> bool:
    text = _text_for_object(obj)
    return any(keyword in text for keyword in CONTAINER_KEYWORDS)


def _contain_render_direction(obj: dict[str, Any]) -> str:
    text = _text_for_object(obj)
    if any(keyword in text for keyword in TOP_VIEW_CONTAINER_KEYWORDS) and not any(keyword in text for keyword in FRONT_VIEW_CONTAINER_KEYWORDS):
        return TOP_RENDER_DIRECTION
    return FRONT_RENDER_DIRECTION


def build_contain_candidates(layout_for_relation: dict[str, Any], object_footprints: dict[str, Any], run_root: Path, output_dir: Path) -> dict[str, Any]:
    objects = [item for item in layout_for_relation.get('objects', []) if isinstance(item, dict)]
    footprint_by_id = {str(item.get('object_id')): item for item in object_footprints.get('items', []) if isinstance(item, dict) and item.get('object_id')}
    containers = []
    pairs = []
    for anchor in objects:
        anchor_id = str(anchor.get('object_id'))
        if not _is_container_like(anchor):
            continue
        direction = _contain_render_direction(anchor)
        anchor_bbox = _bbox3(anchor)
        anchor_fp = footprint_by_id.get(anchor_id, {})
        container_record = {
            'object_id': anchor_id,
            'category': anchor.get('category'),
            'description': anchor.get('description'),
            '渲染方向': direction,
            'render_direction': direction,
            'source': 'local_container_keyword_scan',
            'candidate_is_hint_only': True,
        }
        containers.append(container_record)
        for target in objects:
            target_id = str(target.get('object_id'))
            if target_id == anchor_id:
                continue
            target_bbox = _bbox3(target)
            evidence = []
            score = 0.0
            if _image_center_inside(target, anchor):
                evidence.append('target_image_bbox_center_inside_anchor_image_bbox')
                score += 0.45
            if _bbox_center_inside(target_bbox, anchor_bbox):
                evidence.append('target_3d_bbox_center_inside_anchor_bbox')
                score += 0.35
            if _bbox_smaller_than(target_bbox, anchor_bbox):
                evidence.append('target_bbox_smaller_than_anchor_bbox')
                score += 0.20
            target_text = _text_for_object(target)
            if any(token in target_text for token in ['book', 'box', 'figure', 'figurine', 'toy', 'decor', 'vase', 'bottle', 'plate', 'item', 'object']):
                evidence.append('target_semantics_are_small_or_storable')
                score += 0.10
            # High-recall gate: accept if either image or 3D center suggests inclusion,
            # and the target is not bigger than the container by volume.
            if score < 0.45:
                continue
            pairs.append(
                {
                    'relation': 'contain',
                    'parameters': {'object_anchor': anchor_id, 'object_target': target_id},
                    'decision': 'candidate',
                    'confidence_hint': round(min(score, 0.99), 3),
                    'candidate_is_hint_only': True,
                    'agent_may_add_or_reject': True,
                    'anchor_render_direction': direction,
                    'evidence': evidence,
                    'anchor': {
                        'category': anchor.get('category'),
                        'description': anchor.get('description'),
                        'bbox_global': anchor.get('bbox_global'),
                        'image_bbox': anchor.get('image_bbox'),
                        'footprint_summary': {
                            'center_xz': anchor_fp.get('center_xz'),
                            'extent_xz': anchor_fp.get('extent_xz'),
                        },
                    },
                    'target': {
                        'category': target.get('category'),
                        'description': target.get('description'),
                        'bbox_global': target.get('bbox_global'),
                        'image_bbox': target.get('image_bbox'),
                    },
                }
            )
    # Keep the normal prompt budget bounded, but do not truncate tied
    # high-confidence occupants of one large multi-level anchor.
    raw_pair_count = len(pairs)
    pairs_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        anchor_id = str(pair['parameters']['object_anchor'])
        pairs_by_anchor.setdefault(anchor_id, []).append(pair)
    retained_pairs: list[dict[str, Any]] = []
    truncated_anchor_pairs: dict[str, int] = {}
    high_confidence_overflow_retained: dict[str, int] = {}
    for anchor_id, anchor_pairs in pairs_by_anchor.items():
        ranked_pairs = sorted(
            anchor_pairs,
            key=lambda item: (
                -float(item.get('confidence_hint', 0.0)),
                sorted_object_id_key(item['parameters']['object_target']),
            ),
        )
        retained_for_anchor = ranked_pairs[:MAX_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR]
        overflow_budget = max(
            0,
            MAX_HIGH_CONFIDENCE_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR - len(retained_for_anchor),
        )
        high_confidence_overflow = [
            item
            for item in ranked_pairs[MAX_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR:]
            if float(item.get('confidence_hint', 0.0)) >= HIGH_CONFIDENCE_CONTAIN_CANDIDATE_THRESHOLD
        ][:overflow_budget]
        retained_for_anchor.extend(high_confidence_overflow)
        retained_pairs.extend(retained_for_anchor)
        if high_confidence_overflow:
            high_confidence_overflow_retained[anchor_id] = len(high_confidence_overflow)
        omitted_count = len(ranked_pairs) - len(retained_for_anchor)
        if omitted_count > 0:
            truncated_anchor_pairs[anchor_id] = omitted_count
    pairs = retained_pairs

    overlay = write_contain_candidate_overlay(run_root, output_dir, objects, containers, pairs)
    return {
        'schema_version': 'contain_candidates',
        'status': 'available',
        'source': 'relation_layout local high-recall container scan',
        'policy': {
            'candidate_set_is_closed': False,
            'agent_must_check_for_missing_contain_relations': True,
            'same_anchor_target_pair_attach_and_contain_are_mutually_exclusive': True,
            'soft_max_candidate_pairs_per_anchor': MAX_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR,
            'high_confidence_overflow_threshold': HIGH_CONFIDENCE_CONTAIN_CANDIDATE_THRESHOLD,
            'hard_max_candidate_pairs_per_anchor': MAX_HIGH_CONFIDENCE_CONTAIN_CANDIDATE_PAIRS_PER_ANCHOR,
            'contain_definition': 'Target is inside an ordinary container, or is placed on/within any functional level of multi-level furniture such as a bunk bed, bookcase/shelving unit, multi-tier service cart, or rack. Every multi-level tier including the top is contain. A plain exterior top is attach only for an ordinary single-level anchor.',
        },
        'possible_containers': containers,
        'candidate_pairs': sorted(pairs, key=lambda item: (sorted_object_id_key(item['parameters']['object_anchor']), sorted_object_id_key(item['parameters']['object_target']))),
        'visual_evidence': overlay,
        'summary': {
            'possible_container_count': len(containers),
            'raw_candidate_pair_count': raw_pair_count,
            'candidate_pair_count': len(pairs),
            'high_confidence_overflow_retained': high_confidence_overflow_retained,
            'truncated_anchor_pairs': truncated_anchor_pairs,
        },
    }


def write_contain_candidate_overlay(run_root: Path, output_dir: Path, objects: list[dict[str, Any]], containers: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    image_path = run_root / 'input' / 'image.png'
    visual_dir = output_dir / 'contain_visual_evidence'
    visual_dir.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        return {'status': 'missing_original_image', 'original_image_path': str(image_path)}
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        return {'status': 'skipped', 'reason': f'PIL unavailable: {exc}', 'original_image_path': str(image_path)}
    object_by_id = {str(item.get('object_id')): item for item in objects if item.get('object_id')}
    container_ids = {item['object_id'] for item in containers}
    target_ids = {item['parameters']['object_target'] for item in pairs}
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    def abs_box(obj: dict[str, Any]) -> tuple[int, int, int, int] | None:
        bbox = _norm_image_bbox(obj)
        if not bbox:
            return None
        x0 = int(round(bbox[0] * width))
        y0 = int(round(bbox[1] * height))
        x1 = int(round(bbox[2] * width))
        y1 = int(round(bbox[3] * height))
        return max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)

    for object_id in sorted(container_ids, key=sorted_object_id_key):
        box = abs_box(object_by_id.get(object_id, {}))
        if not box:
            continue
        draw.rectangle(box, outline=(255, 90, 40), width=4)
        draw.text((box[0] + 4, box[1] + 4), f'{object_id} container', fill=(255, 90, 40), font=font)
    for object_id in sorted(target_ids, key=sorted_object_id_key):
        box = abs_box(object_by_id.get(object_id, {}))
        if not box:
            continue
        draw.rectangle(box, outline=(60, 210, 255), width=3)
        draw.text((box[0] + 3, max(0, box[1] - 12)), object_id, fill=(60, 210, 255), font=font)
    overlay_path = visual_dir / 'contain_candidates_overlay.png'
    image.save(overlay_path)
    return {
        'status': 'available',
        'original_image_path': str(image_path),
        'overlay_path': str(overlay_path),
        'container_count': len(container_ids),
        'target_count': len(target_ids),
    }

def load_mapping_objects(mapping_path: Path) -> list[dict[str, Any]]:
    data = load_json_if_exists(mapping_path)
    if isinstance(data, dict) and isinstance(data.get('objects'), list):
        return [item for item in data['objects'] if isinstance(item, dict)]
    return []


def update_expected_status(expected: dict[str, Any], outputs: dict[str, Any], *, fixed_views_status: str) -> dict[str, Any]:
    expected = json.loads(json.dumps(expected))
    artifacts = expected.get('artifacts', {}) if isinstance(expected.get('artifacts'), dict) else {}
    path_by_key = {
        'object_list': outputs.get('layout_for_relation_path'),
        'object_footprints': outputs.get('object_footprints_path'),
        'formal_obb_walls': outputs.get('obb_walls_path'),
        'canonical_topdown': outputs.get('canonical_topdown_image_path'),
        'render_grid_affine': outputs.get('render_grid_affine_path'),
        'wall_evidence': outputs.get('wall_evidence_path'),
        'contain_candidates': outputs.get('contain_candidates_path'),
        'contain_visual_evidence': outputs.get('contain_visual_evidence_path'),
        'fixed_views': outputs.get('fixed_views_manifest_path'),
        'scene_state': outputs.get('scene_state_path'),
    }
    for key, artifact in artifacts.items():
        path = path_by_key.get(key) or artifact.get('path')
        artifact['path'] = path
        if key == 'fixed_views' and fixed_views_status == 'skipped':
            artifact['status'] = 'skipped'
        elif path and Path(str(path)).exists():
            artifact['status'] = 'available'
        else:
            artifact['status'] = 'pending'
    expected['status'] = 'available'
    expected['artifacts'] = artifacts
    return expected


def write_manifest_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"- status: `{manifest.get('status')}`",
        f"- scene_name: `{manifest.get('scene_name')}`",
        f"- precompute_dir: `{manifest.get('precompute_dir')}`",
        f"- formal_wall_method: `{manifest.get('formal_wall_method')}`",
        '',
        '## Stages',
    ]
    for key, value in (manifest.get('stages') or {}).items():
        lines.append(f"- `{key}`: status=`{value.get('status')}`")
    lines.extend(['', '## Outputs'])
    for key, value in sorted((manifest.get('outputs') or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    write_md(path.with_suffix('.md'), 'Relation Layout Manifest', lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a candidate-local relation layout from the formal SAM3D output.')
    parser.add_argument('--scene_name', required=True)
    parser.add_argument('--run_root', required=True, type=Path)
    parser.add_argument('--render_script', type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument('--output_dir', type=Path, default=None, help='Default: <run_root>/reconstruction/relations/relation_layout')
    parser.add_argument('--grid_size', type=int, default=33)
    parser.add_argument('--image_size', type=int, default=768)
    parser.add_argument('--include_fixed_views', action='store_true', help='Generate optional top/front/back/left/right views. Disabled by default.')
    parser.add_argument('--expected_only', action='store_true', help='Write only expected_manifest.json for early manager planning.')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    scene_paths = ScenePaths.from_scene_root(run_root)
    output_dir = (
        args.output_dir or (scene_paths.relations_dir / 'relation_layout')
    ).resolve()
    render_grid_dir = output_dir / 'render_grid_wall'
    walls_dir = output_dir / 'walls'
    fixed_views_dir = output_dir / 'fixed_views'
    for directory in [output_dir, render_grid_dir, walls_dir, fixed_views_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    stages: dict[str, dict[str, Any]] = {}
    layout_records = load_layout_param_records(scene_paths.sam3d_output_dir / 'layout_params')
    expected = write_expected_manifest(args.scene_name, run_root, output_dir, layout_records, include_fixed_views=bool(args.include_fixed_views))
    if args.expected_only:
        print(json.dumps(expected, indent=2, ensure_ascii=False))
        return

    from relations.tools.render_grid_wall_precompute import run_precompute

    render_manifest = run_precompute(
        scene_name=args.scene_name,
        run_root=run_root,
        output_dir=render_grid_dir,
        grid_size=args.grid_size,
        image_size=args.image_size,
    )
    stages['render_grid_wall_precompute'] = {
        'status': render_manifest.get('status'),
        'reused_existing': bool(render_manifest.get('reused_existing')),
    }

    mapping_path = Path(str(render_manifest.get('canonical_mapping_path') or render_grid_dir / 'canonical_topdown_grid_mapping.json'))
    mapping_objects = load_mapping_objects(mapping_path)

    layout_for_relation, object_footprints = build_objects_from_sam3d(run_root, mapping_objects)
    layout_path = output_dir / 'layout_for_relation.json'
    objects_path = output_dir / 'objects.json'
    footprints_path = output_dir / 'object_footprints.json'
    write_json(layout_path, layout_for_relation)
    write_json(objects_path, {'schema_version': 'relation_objects', 'objects': layout_for_relation['objects']})
    write_json(footprints_path, object_footprints)
    stages['object_geometry_from_sam3d'] = {
        'status': 'available',
        'object_count': object_footprints.get('object_count'),
    }

    contain_candidates = build_contain_candidates(layout_for_relation, object_footprints, run_root, output_dir)
    contain_candidates_path = output_dir / 'contain_candidates.json'
    write_json(contain_candidates_path, contain_candidates)
    stages['contain_candidates'] = {
        'status': contain_candidates.get('status'),
        'possible_container_count': contain_candidates.get('summary', {}).get('possible_container_count'),
        'candidate_pair_count': contain_candidates.get('summary', {}).get('candidate_pair_count'),
    }

    method_candidates_path = Path(str(render_manifest.get('method_candidates_path') or render_grid_dir / 'method_candidates.json'))
    affine_path = Path(str(render_manifest.get('render_grid_affine_path') or render_grid_dir / 'render_grid_affine.json'))
    method_candidates = load_json(method_candidates_path)
    render_grid_affine = load_json(affine_path)
    obb_walls = build_obb_walls(method_candidates, render_grid_affine)
    obb_walls_path = walls_dir / 'obb_walls.json'
    write_json(obb_walls_path, obb_walls)
    write_json(walls_dir / 'walls.json', {'schema_version': 'relation_walls', 'walls': obb_walls.get('walls', []), 'source': PRIMARY_METHOD})

    relation_outputs_partial = {
        'layout_for_relation_path': str(layout_path),
        'objects_path': str(objects_path),
        'object_footprints_path': str(footprints_path),
        'obb_walls_path': str(obb_walls_path),
        'method_candidates_path': str(method_candidates_path),
        'render_grid_wall_precompute_dir': str(render_grid_dir),
        'render_grid_affine_path': str(affine_path),
        'canonical_topdown_image_path': render_manifest.get('canonical_topdown_image_path'),
        'canonical_grid_mapping_path': render_manifest.get('canonical_mapping_path'),
        'candidate_overlay_path': render_manifest.get('candidate_overlay_path'),
        'primary_candidate_overlay_path': render_manifest.get('primary_candidate_overlay_path'),
        'fixed_views_dir': str(fixed_views_dir),
        'fixed_views_manifest_path': str(fixed_views_dir / 'views_manifest.json'),
        'contain_candidates_path': str(contain_candidates_path),
        'contain_visual_evidence_path': str(output_dir / 'contain_visual_evidence' / 'contain_candidates_overlay.png'),
    }
    early_manifest = {
        'schema_version': 'relation_layout_manifest',
        'status': 'in_progress',
        'scene_name': args.scene_name,
        'run_root': str(run_root),
        'precompute_dir': str(output_dir),
        'created_by': 'vlm_agent.reconstruction.relations.tools.build_relation_layout_from_sam3d',
        'coordinate_frame': 'OpenGL original XZ; Y is vertical',
        'formal_wall_method': PRIMARY_METHOD,
        'outputs': relation_outputs_partial,
        'stages': stages,
    }
    write_json(output_dir / 'manifest.json', early_manifest)

    scene_state = build_scene_state_from_relation_layout(
        args.scene_name,
        run_root,
        args.render_script.resolve(),
        output_dir,
    )
    fixed_views_result: dict[str, Any]
    if not args.include_fixed_views:
        fixed_views_result = {'status': 'skipped', 'reason': 'fixed views disabled by default', 'views_dir': str(fixed_views_dir), 'manifest_path': str(fixed_views_dir / 'views_manifest.json'), 'views': []}
        stages['fixed_layout_views'] = {'status': 'skipped'}
    else:
        fixed_views_result = ensure_fixed_layout_views(scene_state, fixed_views_dir, dry_run=bool(args.dry_run))
        write_json(fixed_views_dir / 'views_manifest.json', fixed_views_result)
        stages['fixed_layout_views'] = {
            'status': fixed_views_result.get('status'),
            'view_count': len(fixed_views_result.get('views', [])) if isinstance(fixed_views_result.get('views'), list) else 0,
        }
        if fixed_views_result.get('status') == 'available':
            scene_state.setdefault('evidence', {})['fixed_layout_views'] = fixed_views_result

    scene_state = build_scene_state_from_relation_layout(
        args.scene_name,
        run_root,
        args.render_script.resolve(),
        output_dir,
    )
    if isinstance(fixed_views_result, dict) and fixed_views_result.get('status') == 'available':
        scene_state.setdefault('evidence', {})['fixed_layout_views'] = fixed_views_result
    scene_state.setdefault('evidence', {})['render_grid_wall_precompute'] = {
        'status': render_manifest.get('status'),
        'method_candidates_path': str(method_candidates_path),
        'render_grid_affine_path': str(affine_path),
        'canonical_topdown_image_path': render_manifest.get('canonical_topdown_image_path'),
        'canonical_grid_mapping_path': render_manifest.get('canonical_mapping_path'),
        'source': PRIMARY_METHOD,
    }
    wall_evidence = build_wall_evidence(scene_state)
    wall_evidence_path = walls_dir / 'wall_evidence.json'
    write_json(wall_evidence_path, wall_evidence)
    write_wall_evidence(wall_evidence, walls_dir)
    scene_state_path = output_dir / 'scene_state.json'
    write_json(scene_state_path, scene_state)

    outputs = {
        **relation_outputs_partial,
        'wall_evidence_path': str(wall_evidence_path),
        'contain_candidates_path': str(contain_candidates_path),
        'contain_visual_evidence_path': str(output_dir / 'contain_visual_evidence' / 'contain_candidates_overlay.png'),
        'scene_state_path': str(scene_state_path),
        'expected_manifest_path': str(output_dir / 'expected_manifest.json'),
    }
    status = 'available' if render_manifest.get('status') == 'available' and obb_walls.get('status') == 'available' else 'partial'
    if args.include_fixed_views and fixed_views_result.get('status') != 'available':
        status = 'partial'
    manifest = {
        'schema_version': 'relation_layout_manifest',
        'status': status,
        'scene_name': args.scene_name,
        'run_root': str(run_root),
        'precompute_dir': str(output_dir),
        'created_by': 'vlm_agent.reconstruction.relations.tools.build_relation_layout_from_sam3d',
        'coordinate_frame': 'OpenGL original XZ; Y is vertical',
        'formal_wall_method': PRIMARY_METHOD,
        'stages': stages,
        'outputs': outputs,
        'fixed_views_enabled': bool(args.include_fixed_views),
        'dry_run': bool(args.dry_run),
    }
    expected_done = update_expected_status(expected, outputs, fixed_views_status=str(fixed_views_result.get('status')))
    write_json(output_dir / 'expected_manifest.json', expected_done)
    write_json(output_dir / 'manifest.json', manifest)
    write_manifest_md(output_dir / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
