"""Fail-closed completion of an already ambiguous same-prompt SAM3 component.

SAM3-A can deliberately defer a touching candidate when it has negligible IoU
with the assigned main mask.  If both candidates came from the *same exact
semantic prompt*, the deferred candidate is retained locally, lies inside the
same VLM bbox, touches the main mask and does not leak into another final
object, it is safe to union as a completion. The resulting visible mask may
have several disconnected components because of occlusion; requiring a single
pixel-connected component would incorrectly reject real objects such as a
pottery wheel with separated legs. If no candidate satisfies every safety
condition, the existing successful mask remains unchanged and the completion is
recorded as a no-op.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class AmbiguousSamePromptCompletionConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    minimum_candidate_score: float = 0.60
    maximum_candidate_overlap_with_other: float = 0.15
    contact_dilation_pixels: int = 1
    maximum_candidate_area_ratio: float = 1.00
    minimum_existing_owner_coverage: float = 0.99


def _mask(path_value: Any, shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or '')).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result = np.asarray(Image.open(path).convert('L')) > 0
    if result.shape != shape or not result.any():
        raise ValueError(f'invalid mask: {path}')
    return result


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty mask')
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _norm_bbox(raw: Any, width: int, height: int) -> list[int] | None:
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return [round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)]


def _center_in(box: Sequence[int], target: Sequence[int]) -> bool:
    cx = (int(box[0]) + int(box[2])) / 2
    cy = (int(box[1]) + int(box[3])) / 2
    return int(target[0]) <= cx <= int(target[2]) and int(target[1]) <= cy <= int(target[3])


def _touches(left: np.ndarray, right: np.ndarray, pixels: int) -> bool:
    return bool(np.logical_and(
        ndimage.binary_dilation(left, structure=np.ones((3, 3), bool), iterations=max(0, pixels)),
        right,
    ).any())


def _one_component(mask: np.ndarray) -> bool:
    _, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    return int(count) == 1


def _sources(obj: Mapping[str, Any], pass_name: str) -> list[str]:
    return [
        str(row.get('source_id') or '')
        for row in (obj.get('proposal_sources') or [])
        if isinstance(row, Mapping) and str(row.get('pass') or '').upper() == pass_name
    ]


def _save(path: Path, mask: np.ndarray, image: np.ndarray) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    overlay = image.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array((76, 175, 80), dtype=np.uint8)).astype(np.uint8)
    overlay_path = path.parent.parent / 'overlays' / path.name.replace('_mask.png', '_overlay.png')
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_path)
    return str(path), str(overlay_path)


def run_ambiguous_same_prompt_component_completion(
    *,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    sam3_a_registry: Mapping[str, Any],
    staging_dir: str | Path,
    work_dir: str | Path,
    config: AmbiguousSamePromptCompletionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AmbiguousSamePromptCompletionConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError('ambiguous same-prompt completion is isolated-attempt only')
    out = Path(work_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'completion work directory is not empty: {out}')
    out.mkdir(parents=True)
    image = np.asarray(Image.open(Path(image_path).expanduser().resolve()).convert('RGB'))
    shape = image.shape[:2]
    height, width = shape
    copied = [copy.deepcopy(dict(item)) for item in objects]
    source_by_id = {
        str(item.get('object_id') or ''): dict(item)
        for item in (sam3_a_registry.get('objects') or [])
        if isinstance(item, Mapping)
    }
    candidates = {
        str(item.get('candidate_id') or ''): dict(item)
        for item in (sam3_a_registry.get('unassigned_sam3_candidates') or [])
        if isinstance(item, Mapping)
    }
    masks: dict[str, np.ndarray] = {}
    for obj in copied:
        if (obj.get('sam3') or {}).get('status') != 'success':
            continue
        try:
            masks[str(obj.get('object_id') or '')] = _mask((obj.get('sam3') or {}).get('mask_path'), shape)
        except (OSError, ValueError):
            pass

    audit: dict[str, Any] = {
        'schema_version': 'scenemosaic_ambiguous_same_prompt_component_completion',
        'config': asdict(cfg),
        'target_count': 0,
        'completed': [],
        'unresolved': [],
        'skipped': [],
        'hard_errors': [],
        'additional_vlm_calls': 0,
        'formal_outputs_modified': False,
    }
    for obj in copied:
        object_id = str(obj.get('object_id') or '')
        source_ids = [item for item in _sources(obj, 'A') if item in source_by_id]
        if len(source_ids) != 1 or object_id not in masks:
            continue
        source = source_by_id[source_ids[0]]
        source_sam3 = source.get('sam3') or {}
        review = source_sam3.get('merge_review') or {}
        ambiguous_ids = [str(value) for value in (review.get('ambiguous_candidate_ids') or [])]
        # A later review may mark the primary selection as resolved while
        # retaining the original ambiguous IDs. Their geometry is still
        # evidence for this deterministic completion stage, so do not discard
        # them solely because that bookkeeping flag became false.
        if not ambiguous_ids:
            continue
        audit['target_count'] += 1
        target_bbox = _norm_bbox(source.get('bbox_2d_norm_xyxy'), width, height)
        base = masks[object_id]
        if target_bbox is None:
            audit['unresolved'].append({'object_id': object_id, 'reason': 'invalid_source_bbox'})
            continue
        viable: list[dict[str, Any]] = []
        explained_by_existing: list[dict[str, Any]] = []
        unresolved_candidate_seen = False
        for candidate_id in ambiguous_ids:
            raw = candidates.get(candidate_id)
            if raw is None:
                continue
            if str(raw.get('sam3_prompt') or '') != str(source.get('sam3_prompt') or ''):
                continue
            if float(raw.get('score') or 0.0) < cfg.minimum_candidate_score:
                continue
            try:
                candidate = _mask(raw.get('mask_path'), shape)
            except (OSError, ValueError):
                continue
            box = _bbox(candidate)
            existing_owners: list[dict[str, Any]] = []
            for other_id, other_mask in masks.items():
                if other_id == object_id:
                    continue
                coverage = int(np.logical_and(candidate, other_mask).sum()) / max(
                    1, int(candidate.sum())
                )
                if coverage >= cfg.minimum_existing_owner_coverage:
                    existing_owners.append(
                        {
                            "object_id": other_id,
                            "candidate_coverage": round(coverage, 6),
                        }
                    )
            # A deferred same-prompt mask can describe an already-declared
            # neighbouring instance.  It is evidence of neither a missing
            # component nor a semantic merge when an existing mask completely
            # owns it, so preserve both objects and record the explanation.
            if existing_owners:
                explained_by_existing.append(
                    {
                        "candidate_id": candidate_id,
                        "existing_owners": existing_owners,
                    }
                )
                continue
            unresolved_candidate_seen = True
            if not _center_in(box, target_bbox) or not _touches(
                base, candidate, cfg.contact_dilation_pixels
            ):
                continue
            if int(candidate.sum()) > int(base.sum()) * cfg.maximum_candidate_area_ratio:
                continue
            conflicts: list[dict[str, Any]] = []
            for other_id, other_mask in masks.items():
                if other_id == object_id:
                    continue
                overlap = int(np.logical_and(candidate, other_mask).sum())
                if overlap / max(1, int(candidate.sum())) > cfg.maximum_candidate_overlap_with_other:
                    conflicts.append({'object_id': other_id, 'candidate_coverage': round(overlap / int(candidate.sum()), 6)})
            if conflicts:
                continue
            union = base | candidate
            viable.append({'candidate_id': candidate_id, 'mask': candidate, 'score': float(raw.get('score') or 0.0), 'union': union})
        if not viable:
            if explained_by_existing and not unresolved_candidate_seen:
                audit['skipped'].append(
                    {
                        "object_id": object_id,
                        "source_object_id": source_ids[0],
                        "reason": "same_prompt_candidate_already_owned_by_existing_object",
                        "explained_candidates": explained_by_existing,
                    }
                )
                continue
            # This refinement is optional: the lack of a safe extra component
            # is not evidence that the already successful owner mask is invalid.
            # Preserve it and retain the deferred candidates as an auditable no-op.
            audit['skipped'].append(
                {
                    'object_id': object_id,
                    'source_object_id': source_ids[0],
                    'reason': 'no_safe_touching_same_prompt_component',
                    'ambiguous_candidate_ids': ambiguous_ids,
                }
            )
            continue
        selected = max(viable, key=lambda row: (row['score'], int(row['mask'].sum()), row['candidate_id']))
        output = Path(staging_dir).expanduser().resolve() / 'masks' / f"{object_id}_{str(obj.get('category') or 'object')}_mask.png"
        mask_path, overlay_path = _save(output, selected['union'], image)
        obj['sam3'] = copy.deepcopy(dict(obj.get('sam3') or {}))
        obj['sam3'].update({
            'mask_path': mask_path,
            'overlay_path': overlay_path,
            'area_pixels': int(selected['union'].sum()),
            'mask_bbox_abs_xyxy': _bbox(selected['union']),
            'selection_method': 'ambiguous_same_prompt_component_completion',
            'ambiguous_same_prompt_completion': {
                'source_object_id': source_ids[0],
                'candidate_id': selected['candidate_id'],
                'candidate_score': selected['score'],
                'previous_area_pixels': int(base.sum()),
                'completed_area_pixels': int(selected['union'].sum()),
            },
        })
        masks[object_id] = selected['union']
        audit['completed'].append({'object_id': object_id, **obj['sam3']['ambiguous_same_prompt_completion']})
    if audit['unresolved']:
        audit['hard_errors'] = [{'type': 'ambiguous_same_prompt_component_unresolved', **item} for item in audit['unresolved']]
    audit['status'] = 'pass' if not audit['hard_errors'] else 'blocked'
    (out / 'ambiguous_same_prompt_component_completion_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'objects': copied, 'audit': audit}


__all__ = ['AmbiguousSamePromptCompletionConfig', 'run_ambiguous_same_prompt_component_completion']
