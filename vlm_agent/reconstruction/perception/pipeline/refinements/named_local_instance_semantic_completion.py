"""Bounded safety check after Pass-B names a local carrier candidate."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class NamedLocalSemanticCompletionConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    maximum_targets: int = 24
    maximum_candidates_per_target: int = 4
    minimum_score: float = 0.20
    minimum_existing_coverage: float = 0.65
    minimum_mask_inside_host_bbox: float = 0.90
    maximum_candidate_area_ratio: float = 3.0
    maximum_other_object_candidate_coverage: float = 0.15
    minimum_rank_margin: float = 0.10
    roi_padding_fraction: float = 0.15
    outer_boundary_band_pixels: int = 1
    verify_nonboundary_candidates: bool = False


def _mask(path_value: Any, shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or '')).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.asarray(Image.open(path).convert('L')) > 0
    if value.shape != shape or not value.any():
        raise ValueError(f'invalid mask: {path}')
    return value


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty mask')
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _expand_and_clip(box: Sequence[int], host: Sequence[int], fraction: float) -> list[int]:
    x1, y1, x2, y2 = (int(value) for value in box)
    hx1, hy1, hx2, hy2 = (int(value) for value in host)
    px = max(2, round((x2 - x1) * fraction))
    py = max(2, round((y2 - y1) * fraction))
    return [max(hx1, x1 - px), max(hy1, y1 - py), min(hx2, x2 + px), min(hy2, y2 + py)]


def _inside_bbox(mask: np.ndarray, box: Sequence[int]) -> float:
    x1, y1, x2, y2 = (int(value) for value in box)
    return int(mask[y1:y2, x1:x2].sum()) / max(1, int(mask.sum()))


def _outer_edges(box: Sequence[int], host: Sequence[int], band: int) -> list[str]:
    x1, y1, x2, y2 = (int(value) for value in box)
    hx1, hy1, hx2, hy2 = (int(value) for value in host)
    result = []
    if x1 <= hx1 + band:
        result.append('left')
    if y1 <= hy1 + band:
        result.append('top')
    if x2 >= hx2 - band:
        result.append('right')
    if y2 >= hy2 - band:
        result.append('bottom')
    return result


def _one_component(mask: np.ndarray) -> bool:
    _, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    return int(count) == 1


def _local_candidate_ids(obj: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get('source_id') or '')
        for item in (obj.get('proposal_sources') or [])
        if isinstance(item, Mapping) and str(item.get('pass') or '') == 'local_candidate'
    ]


def _save(mask: np.ndarray, image: np.ndarray, path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    overlay = image.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array((58, 145, 220), dtype=np.uint8)).astype(np.uint8)
    overlay_path = path.parent.parent / 'overlays' / path.name.replace('_mask.png', '_overlay.png')
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_path)
    return str(path), str(overlay_path)


def run_named_local_instance_semantic_completion(
    *,
    session: Any,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    local_candidate_metadata: Sequence[Mapping[str, Any]],
    sam3_a_registry: Mapping[str, Any],
    staging_dir: str | Path,
    work_dir: str | Path,
    infer_candidates_fn: Callable[..., Mapping[str, Any]],
    config: NamedLocalSemanticCompletionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or NamedLocalSemanticCompletionConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError('named local semantic completion is isolated-attempt only')
    out = Path(work_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'named local completion work directory is not empty: {out}')
    out.mkdir(parents=True)
    image_file = Path(image_path).expanduser().resolve()
    image = np.asarray(Image.open(image_file).convert('RGB'))
    shape = image.shape[:2]
    copied = [copy.deepcopy(dict(item)) for item in objects]
    metadata_by_id = {
        str(item.get('candidate_id') or ''): dict(item)
        for item in local_candidate_metadata
        if isinstance(item, Mapping)
    }
    host_by_id = {
        str(item.get('object_id') or ''): dict(item)
        for item in (sam3_a_registry.get('objects') or [])
        if isinstance(item, Mapping)
    }
    masks: dict[str, np.ndarray] = {}
    for obj in copied:
        if (obj.get('sam3') or {}).get('status') != 'success':
            continue
        try:
            masks[str(obj.get('object_id') or '')] = _mask((obj.get('sam3') or {}).get('mask_path'), shape)
        except (OSError, ValueError):
            continue
    targets = []
    for obj in copied:
        local_ids = _local_candidate_ids(obj)
        if len(local_ids) != 1:
            continue
        metadata = metadata_by_id.get(local_ids[0])
        host_id = str((metadata or {}).get('host_source_object_id') or '')
        if not metadata or not host_id or host_id not in host_by_id:
            continue
        if str(obj.get('object_id') or '') not in masks:
            continue
        targets.append((str(obj.get('object_id') or ''), local_ids[0], host_id))
    targets.sort()
    audit: dict[str, Any] = {
        'schema_version': 'scenemosaic_named_local_instance_semantic_completion',
        'config': asdict(cfg),
        'target_count': min(len(targets), cfg.maximum_targets),
        'inference_call_count': 0,
        'completed': [],
        'verified': [],
        'removed_boundary_artifacts': [],
        'unresolved': [],
        'skipped': [],
        'hard_errors': [],
        'additional_vlm_calls': 0,
        'formal_outputs_modified': False,
    }
    retained = []
    target_ids = {item[0] for item in targets[:cfg.maximum_targets]}
    for object_id, local_id, host_id in targets[cfg.maximum_targets:]:
        audit['unresolved'].append({'object_id': object_id, 'candidate_id': local_id, 'host_source_object_id': host_id, 'reason': 'target_budget_exceeded'})
    for obj in copied:
        object_id = str(obj.get('object_id') or '')
        if object_id not in target_ids:
            retained.append(obj)
            continue
        local_id = _local_candidate_ids(obj)[0]
        metadata = metadata_by_id[local_id]
        host_id = str(metadata['host_source_object_id'])
        base = masks[object_id]
        host_mask = _mask((host_by_id[host_id].get('sam3') or {}).get('mask_path'), shape)
        host_bbox = _bbox(host_mask)
        base_bbox = _bbox(base)
        prompt = str(obj.get('sam3_prompt') or '').strip()
        edges = _outer_edges(base_bbox, host_bbox, cfg.outer_boundary_band_pixels)
        if not prompt:
            audit['unresolved'].append({'object_id': object_id, 'candidate_id': local_id, 'host_source_object_id': host_id, 'reason': 'missing_semantic_prompt'})
            retained.append(obj)
            continue
        # A named, interior local candidate is already semantically bound by
        # Pass B. Do not make publication depend on a second SAM3 prompt:
        # small/occluded contents often cannot be recovered by that redundant
        # query even when the original local mask is correct. Boundary-touching
        # candidates remain subject to the stricter probe below.
        if not edges and not cfg.verify_nonboundary_candidates:
            audit['verified'].append({
                'object_id': object_id,
                'candidate_id': local_id,
                'host_source_object_id': host_id,
                'verification': 'pass_b_named_interior_local_candidate',
                'semantic_probe': 'not_required_for_interior_candidate',
            })
            retained.append(obj)
            continue
        positive_bbox = _expand_and_clip(base_bbox, host_bbox, cfg.roi_padding_fraction)
        audit['inference_call_count'] += 1
        target_work = out / object_id
        try:
            response = infer_candidates_fn(
                session, image_file, prompt, target_work,
                score_threshold=cfg.minimum_score,
                positive_bbox_abs_xyxy=positive_bbox,
                maximum_candidates=cfg.maximum_candidates_per_target,
            )
            raw_candidates = (response.get('registry') or {}).get('candidates') or []
        except Exception as exc:
            audit['unresolved'].append({'object_id': object_id, 'candidate_id': local_id, 'host_source_object_id': host_id, 'reason': f'inference_exception:{type(exc).__name__}'})
            retained.append(obj)
            continue
        viable = []
        for raw in raw_candidates:
            try:
                candidate = _mask(raw.get('mask_path'), shape)
            except (OSError, ValueError):
                continue
            score = float(raw.get('score') or 0.0)
            existing_coverage = int(np.logical_and(candidate, base).sum()) / max(1, int(base.sum()))
            inside = _inside_bbox(candidate, host_bbox)
            area_ratio = int(candidate.sum()) / max(1, int(base.sum()))
            if score < cfg.minimum_score or existing_coverage < cfg.minimum_existing_coverage or inside < cfg.minimum_mask_inside_host_bbox or area_ratio > cfg.maximum_candidate_area_ratio:
                continue
            conflicts = []
            for other_id, other_mask in masks.items():
                if other_id == object_id:
                    continue
                # The open carrier itself is expected to overlap a contained
                # object before ownership resolution; do not treat it as a peer.
                other_sources = next((item.get('proposal_sources') or [] for item in copied if str(item.get('object_id') or '') == other_id), [])
                if any(isinstance(row, Mapping) and str(row.get('source_id') or '') == host_id for row in other_sources):
                    continue
                overlap = int(np.logical_and(candidate, other_mask).sum()) / max(1, int(candidate.sum()))
                if overlap > cfg.maximum_other_object_candidate_coverage:
                    conflicts.append({'object_id': other_id, 'candidate_coverage': round(overlap, 6)})
            if conflicts:
                continue
            union = base | candidate
            if not _one_component(union):
                continue
            rank = 2.0 * existing_coverage + min(1.0, area_ratio) + 0.25 * score
            viable.append({'candidate_id': str(raw.get('candidate_id') or ''), 'mask': candidate, 'score': score, 'existing_coverage': existing_coverage, 'inside': inside, 'area_ratio': area_ratio, 'union': union, 'rank': rank})
        viable.sort(key=lambda row: row['rank'], reverse=True)
        if not viable:
            if edges:
                masks.pop(object_id, None)
                audit['removed_boundary_artifacts'].append({'object_id': object_id, 'candidate_id': local_id, 'host_source_object_id': host_id, 'outer_anchor_edges': edges, 'reason': 'no_exact_semantic_candidate_inside_boundary_artifact'})
                continue
            audit['unresolved'].append({'object_id': object_id, 'candidate_id': local_id, 'host_source_object_id': host_id, 'reason': 'no_safe_exact_semantic_candidate'})
            retained.append(obj)
            continue
        if len(viable) > 1 and viable[0]['rank'] - viable[1]['rank'] < cfg.minimum_rank_margin:
            audit['unresolved'].append({'object_id': object_id, 'candidate_id': local_id, 'host_source_object_id': host_id, 'reason': 'ambiguous_semantic_candidate_rank'})
            retained.append(obj)
            continue
        chosen = viable[0]
        union = chosen['union']
        if int(union.sum()) > int(base.sum()) * 1.03:
            output = Path(staging_dir).expanduser().resolve() / 'masks' / f"{object_id}_{str(obj.get('category') or 'object')}_mask.png"
            mask_path, overlay_path = _save(union, image, output)
            obj['sam3'] = copy.deepcopy(dict(obj.get('sam3') or {}))
            obj['sam3'].update({
                'mask_path': mask_path,
                'overlay_path': overlay_path,
                'area_pixels': int(union.sum()),
                'mask_bbox_abs_xyxy': _bbox(union),
                'selection_method': 'named_local_semantic_completion',
                'named_local_semantic_completion': {
                    'candidate_id': chosen['candidate_id'], 'candidate_score': round(chosen['score'], 6),
                    'host_source_object_id': host_id, 'local_candidate_id': local_id,
                    'previous_area_pixels': int(base.sum()), 'completed_area_pixels': int(union.sum()),
                    'existing_coverage': round(chosen['existing_coverage'], 6),
                },
            })
            masks[object_id] = union
            audit['completed'].append({'object_id': object_id, **obj['sam3']['named_local_semantic_completion']})
        else:
            audit['verified'].append({'object_id': object_id, 'candidate_id': chosen['candidate_id'], 'host_source_object_id': host_id, 'existing_coverage': round(chosen['existing_coverage'], 6)})
        retained.append(obj)
    if audit['unresolved']:
        audit['hard_errors'] = [{'type': 'named_local_instance_semantic_completion_unresolved', **item} for item in audit['unresolved']]
    audit['status'] = 'pass' if not audit['hard_errors'] else 'blocked'
    (out / 'named_local_instance_semantic_completion_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'objects': retained, 'audit': audit}


__all__ = ['NamedLocalSemanticCompletionConfig', 'run_named_local_instance_semantic_completion']
