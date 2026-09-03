#!/usr/bin/env python3
"""Recover repeated instances when VLM count is lower than delta SAM3 evidence."""

from __future__ import annotations

import copy
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class MultiplicityClosureConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    minimum_peer_count: int = 2
    minimum_candidate_score: float = 0.85
    minimum_single_peer_score: float = 0.88
    minimum_single_peer_area_ratio: float = 0.35
    minimum_single_peer_dimension_ratio: float = 0.35
    minimum_single_peer_suspended_light_score: float = 0.50
    minimum_single_peer_suspended_light_area_ratio: float = 0.15
    minimum_single_peer_suspended_light_dimension_ratio: float = 0.30
    minimum_recessed_ceiling_light_score: float = 0.80
    maximum_recessed_ceiling_light_y_fraction: float = 0.25
    minimum_area_ratio: float = 0.55
    maximum_area_ratio: float = 1.80
    minimum_dimension_ratio: float = 0.50
    maximum_dimension_ratio: float = 2.00
    maximum_existing_iou: float = 0.10
    maximum_existing_candidate_coverage: float = 0.20
    minimum_conflict_pixels: int = 20
    minimum_completion_candidate_score: float = 0.30
    minimum_superset_existing_coverage: float = 0.95
    maximum_superset_area_ratio: float = 3.00
    minimum_touching_candidate_score: float = 0.60
    minimum_touching_candidate_area_ratio: float = 1.05
    maximum_touching_candidate_area_ratio: float = 2.00
    minimum_superset_union_bbox_coverage: float = 0.70
    minimum_completion_union_bbox_coverage: float = 0.90
    maximum_completion_overlap_with_other: float = 0.15
    minimum_adjacent_component_score: float = 0.60
    minimum_adjacent_component_area_ratio: float = 0.20
    maximum_adjacent_component_area_ratio: float = 1.50
    adjacent_component_contact_dilation_pixels: int = 1
    maximum_adjacent_completion_components: int = 4
    minimum_failed_object_candidate_score: float = 0.30
    maximum_additions_per_group: int = 2
    maximum_total_additions: int = 4


HOST_WORDS = {
    "cart",
    "shelf",
    "shelving",
    "rack",
    "cabinet",
    "bookcase",
    "storage",
    "counter",
    "worktable",
    "workbench",
}
OPEN_STORAGE_WORDS = {
    "bookcase",
    "bookshelf",
    "cubby",
    "hutch",
    "rack",
    "shelf",
    "shelving",
    "storage",
}
SINGLE_PEER_REPEAT_WORDS = {"chair", "stool"}
SINGLE_PEER_SUSPENDED_LIGHT_WORDS = {"pendant", "chandelier"}




def normalize(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def sam3_query_key(item: Mapping[str, Any]) -> str:
    """Return the text query that actually produced an item's SAM3 mask.

    Repeated inventory entries can retain descriptive VLM prompts such as
    ``books`` and ``stacked books`` while the segmentation stage shares one
    category-level SAM3 query, ``book``. Candidate closure must compare that
    executed query, not the descriptive inventory wording.
    """

    sam3 = item.get("sam3") or {}
    return normalize(sam3.get("prompt_used") or item.get("sam3_prompt"))


def is_pass_b_addition(obj: Mapping[str, Any]) -> bool:
    return any(
        str(item.get("pass") or "").upper() == "B"
        and str(item.get("action") or "").lower() == "added"
        for item in (obj.get("proposal_sources") or [])
        if isinstance(item, Mapping)
    )


def is_host(obj: Mapping[str, Any]) -> bool:
    text = " ".join(
        normalize(obj.get(key))
        for key in ("category", "sam3_prompt", "instance_name", "description")
    )
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in HOST_WORDS)


def is_open_storage_host(obj: Mapping[str, Any]) -> bool:
    text = " ".join(
        normalize(obj.get(key))
        for key in ("category", "sam3_prompt", "instance_name", "description")
    )
    return any(
        re.search(rf"\b{re.escape(word)}\b", text)
        for word in OPEN_STORAGE_WORDS
    )


def masks_touch(
    left: np.ndarray,
    right: np.ndarray,
    *,
    dilation_pixels: int,
) -> bool:
    if np.logical_and(left, right).any():
        return True
    return bool(
        np.logical_and(
            ndimage.binary_dilation(
                left,
                structure=np.ones((3, 3), dtype=bool),
                iterations=max(1, dilation_pixels),
            ),
            right,
        ).any()
    )


def retain_completion_components_connected_to_base(
    candidate: np.ndarray,
    base: np.ndarray,
    *,
    dilation_pixels: int,
) -> tuple[np.ndarray, int]:
    """Keep only candidate components connected to an existing instance mask.

    SAM3 may return one candidate mask containing a valid whole-instance
    component plus an unrelated, disconnected instance of the same prompt.
    Multiplicity completion must never turn those separate instances into one
    object merely because their enclosing bounding box is target-local.
    """

    labels, component_count = ndimage.label(candidate)
    if component_count <= 1:
        return candidate, 0
    contact_region = ndimage.binary_dilation(
        base,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, dilation_pixels),
    )
    connected_labels = {
        int(label)
        for label in np.unique(labels[contact_region])
        if int(label) > 0
    }
    if not connected_labels:
        return np.zeros_like(candidate, dtype=bool), int(component_count)
    retained = np.isin(labels, list(connected_labels))
    return retained, int(component_count - len(connected_labels))


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def valid_bbox(raw: Any) -> list[int] | None:
    try:
        values = [int(round(float(value))) for value in raw]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not (values[0] < values[2] and values[1] < values[3]):
        return None
    return values


def center(box: Sequence[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def contains(box: Sequence[int], point: Sequence[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(a, b).sum())
    aa, ba = int(a.sum()), int(b.sum())
    union = aa + ba - intersection
    return {
        "intersection_pixels": intersection,
        "iou": intersection / union if union else 0.0,
        "candidate_coverage": intersection / aa if aa else 0.0,
    }


def bbox_coverage(outer: Sequence[int], inner: Sequence[int]) -> float:
    x1 = max(int(outer[0]), int(inner[0]))
    y1 = max(int(outer[1]), int(inner[1]))
    x2 = min(int(outer[2]), int(inner[2]))
    y2 = min(int(outer[3]), int(inner[3]))
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = max(
        1,
        (int(inner[2]) - int(inner[0]))
        * (int(inner[3]) - int(inner[1])),
    )
    return intersection / inner_area


def expanded_object_bbox(obj: Mapping[str, Any]) -> list[int] | None:
    sam3 = obj.get("sam3") or {}
    locality = sam3.get("scale_aware_locality") or {}
    return (
        valid_bbox(locality.get("expanded_vlm_bbox_abs_xyxy"))
        or valid_bbox(sam3.get("vlm_bbox_abs_xyxy"))
    )


def candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("mask_path")
        or candidate.get("candidate_id")
        or ""
    )


def load_mask(path_value: Any, shape: tuple[int, int], root: Path | None = None) -> tuple[np.ndarray, Path]:
    path = Path(str(path_value or "")).expanduser().resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ValueError("candidate path escapes delta unassigned root")
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != shape or not mask.any():
        raise ValueError("invalid mask")
    return mask, path


def next_object_id(objects: Sequence[Mapping[str, Any]]) -> str:
    maximum = 0
    for obj in objects:
        match = re.fullmatch(r"proposal_(\d+)", str(obj.get("object_id") or ""))
        if match:
            maximum = max(maximum, int(match.group(1)))
    return f"proposal_{maximum + 1:03d}"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_post_delta_multiplicity_closure(
    *,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    delta_registry: Mapping[str, Any] | None,
    fallback_registry: Mapping[str, Any] | None = None,
    work_dir: str | Path,
    staging_dir: str | Path,
    config: MultiplicityClosureConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MultiplicityClosureConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("multiplicity closure is isolated-attempt only")
    copied = [copy.deepcopy(dict(item)) for item in objects]
    work = Path(work_dir).expanduser().resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"multiplicity work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": "scenemosaic_post_delta_multiplicity_closure",
        "enabled": cfg.enabled,
        "status": "disabled" if not cfg.enabled else "pending",
        "groups_checked": [],
        "accepted": [],
        "completed": [],
        "rescued_failed": [],
        "rejected": [],
        "filtered_unconnected_components": [],
        "accepted_count": 0,
        "completed_count": 0,
        "rescued_failed_count": 0,
        "additional_vlm_calls": 0,
        "additional_sam3_inference_calls": 0,
        "hard_errors": [],
    }
    audit_path = work / "post_delta_multiplicity_closure_audit.json"
    if not cfg.enabled:
        atomic_json(audit_path, audit)
        return {"objects": copied, "audit": audit, "audit_path": str(audit_path)}

    candidates = []
    seen_candidate_masks = set()
    for registry_value in (delta_registry, fallback_registry):
        for candidate in (registry_value or {}).get("unassigned_sam3_candidates") or []:
            key = str(candidate.get("mask_path") or "")
            if key and key not in seen_candidate_masks:
                candidates.append(candidate)
                seen_candidate_masks.add(key)
    if not candidates:
        audit["status"] = "noop"
        atomic_json(audit_path, audit)
        return {"objects": copied, "audit": audit, "audit_path": str(audit_path)}

    with Image.open(Path(image_path).expanduser().resolve()) as source:
        image_np = np.array(source.convert("RGB"))
    height, width = image_np.shape[:2]
    shape = (height, width)

    existing_masks: dict[str, np.ndarray] = {}
    for obj in copied:
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            continue
        try:
            existing_masks[str(obj.get("object_id") or "")], _ = load_mask(sam3.get("mask_path"), shape)
        except (OSError, ValueError):
            audit["hard_errors"].append(
                {"type": "invalid_existing_success_mask", "object_id": obj.get("object_id")}
            )

    staging = Path(staging_dir).expanduser().resolve()
    candidate_masks: dict[str, np.ndarray] = {}
    used_candidate_keys: set[str] = set()

    def candidate_mask(candidate: Mapping[str, Any]) -> np.ndarray:
        key = candidate_key(candidate)
        if not key:
            raise ValueError("candidate has no stable key")
        if key not in candidate_masks:
            candidate_masks[key], _ = load_mask(candidate.get("mask_path"), shape)
        return candidate_masks[key]

    def write_object_mask(
        obj: Mapping[str, Any],
        mask: np.ndarray,
    ) -> tuple[str, str]:
        object_id = str(obj.get("object_id") or "")
        category_slug = (
            normalize(obj.get("category")).replace(" ", "_")
            or "object"
        )
        mask_path = (
            staging
            / "masks"
            / f"{object_id}_{category_slug}_mask.png"
        )
        overlay_path = (
            staging
            / "overlays"
            / f"{object_id}_{category_slug}_overlay.png"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        overlay_image = image_np.copy()
        overlay_image[mask] = (
            0.55 * overlay_image[mask]
            + 0.45 * np.array((255, 0, 0), dtype=np.uint8)
        ).astype(np.uint8)
        Image.fromarray(overlay_image).save(overlay_path)
        return str(mask_path), str(overlay_path)

    def shared_host(
        points: Sequence[Sequence[float]],
        excluded_object_ids: set[str],
    ) -> tuple[dict[str, Any] | None, list[int] | None]:
        hosts = []
        for host in copied:
            object_id = str(host.get("object_id") or "")
            host_box = valid_bbox(
                (host.get("sam3") or {}).get("vlm_bbox_abs_xyxy")
            )
            if (
                object_id not in excluded_object_ids
                and host_box
                and is_host(host)
                and (host.get("sam3") or {}).get("status") == "success"
                and all(contains(host_box, point) for point in points)
            ):
                area = max(
                    1,
                    (host_box[2] - host_box[0])
                    * (host_box[3] - host_box[1]),
                )
                hosts.append((area, host, host_box))
        if not hosts:
            return None, None
        _, host, host_box = min(hosts, key=lambda item: item[0])
        return host, host_box

    def mask_conflicts(
        mask: np.ndarray,
        *,
        object_id: str,
        exempt_object_ids: set[str],
        maximum_coverage: float,
    ) -> list[dict[str, Any]]:
        area = int(mask.sum())
        conflicts = []
        if not area:
            return conflicts
        for other_id, other_mask in existing_masks.items():
            if other_id == object_id or other_id in exempt_object_ids:
                continue
            intersection = int(np.logical_and(mask, other_mask).sum())
            coverage = intersection / area
            if (
                intersection >= cfg.minimum_conflict_pixels
                and coverage > maximum_coverage
            ):
                conflicts.append(
                    {
                        "object_id": other_id,
                        "intersection_pixels": intersection,
                        "candidate_coverage": round(coverage, 6),
                    }
                )
        return conflicts

    prompt_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in copied:
        prompt = sam3_query_key(obj)
        if prompt and not is_host(obj):
            prompt_members[prompt].append(obj)

    # SAM3 commonly emits a high-confidence part mask beside a lower-score
    # whole-instance mask. Complete an already assigned repeated instance only
    # when the extra candidate is geometrically local, touches or contains the
    # current mask, and contributes no unsafe pixels to another object.
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        prompt = sam3_query_key(obj)
        if (
            sam3.get("status") != "success"
            or len(prompt_members.get(prompt, [])) < cfg.minimum_peer_count
            or object_id not in existing_masks
        ):
            continue
        target_box = valid_bbox(sam3.get("vlm_bbox_abs_xyxy"))
        expanded_box = expanded_object_bbox(obj)
        if target_box is None or expanded_box is None:
            continue
        base = existing_masks[object_id]
        base_area = int(base.sum())
        base_box = valid_bbox(sam3.get("mask_bbox_abs_xyxy"))
        base_score = float(sam3.get("score") or 0.0)
        if not base_area or base_box is None:
            continue
        viable = []
        for candidate in candidates:
            key = candidate_key(candidate)
            if (
                not key
                or key in used_candidate_keys
                or sam3_query_key(candidate) != prompt
            ):
                continue
            score = float(candidate.get("score") or 0.0)
            if score < cfg.minimum_completion_candidate_score:
                continue
            try:
                raw_candidate_value = candidate_mask(candidate)
            except (OSError, ValueError) as exc:
                audit["hard_errors"].append(
                    {
                        "type": "invalid_completion_candidate",
                        "candidate_id": candidate.get("candidate_id"),
                        "error": str(exc),
                    }
                )
                continue
            candidate_value, discarded_component_count = (
                retain_completion_components_connected_to_base(
                    raw_candidate_value,
                    base,
                    dilation_pixels=cfg.adjacent_component_contact_dilation_pixels,
                )
            )
            if discarded_component_count:
                audit["filtered_unconnected_components"].append(
                    {
                        "object_id": object_id,
                        "candidate_id": candidate.get("candidate_id"),
                        "discarded_component_count": discarded_component_count,
                    }
                )
            candidate_box = bbox_from_mask(candidate_value)
            if candidate_box is None:
                continue
            candidate_area = int(candidate_value.sum())
            intersection = int(np.logical_and(base, candidate_value).sum())
            if not candidate_area:
                continue
            existing_coverage = intersection / base_area
            area_ratio = candidate_area / base_area
            host, _ = shared_host(
                [center(base_box), center(candidate_box)],
                {object_id},
            )
            adjacent_same_object_component = (
                is_pass_b_addition(obj)
                and host is not None
                and is_open_storage_host(host)
                and score >= cfg.minimum_adjacent_component_score
                and cfg.minimum_adjacent_component_area_ratio
                <= area_ratio
                <= cfg.maximum_adjacent_component_area_ratio
                and contains(target_box, center(candidate_box))
                and masks_touch(
                    base,
                    candidate_value,
                    dilation_pixels=cfg.adjacent_component_contact_dilation_pixels,
                )
            )
            if not intersection and not adjacent_same_object_component:
                continue
            union_box = [
                min(base_box[0], candidate_box[0]),
                min(base_box[1], candidate_box[1]),
                max(base_box[2], candidate_box[2]),
                max(base_box[3], candidate_box[3]),
            ]
            union_bbox_coverage = bbox_coverage(expanded_box, union_box)
            mode = None
            if (
                existing_coverage
                >= cfg.minimum_superset_existing_coverage
                and 1.05 <= area_ratio <= cfg.maximum_superset_area_ratio
                and contains(expanded_box, center(candidate_box))
                and union_bbox_coverage
                >= cfg.minimum_superset_union_bbox_coverage
            ):
                mode = "safe_superset"
            elif (
                score >= cfg.minimum_touching_candidate_score
                and score > base_score
                and cfg.minimum_touching_candidate_area_ratio
                <= area_ratio
                <= cfg.maximum_touching_candidate_area_ratio
                and contains(target_box, center(candidate_box))
                and union_bbox_coverage
                >= cfg.minimum_completion_union_bbox_coverage
            ):
                mode = "better_local_candidate"
            elif (
                adjacent_same_object_component
                and union_bbox_coverage
                >= cfg.minimum_completion_union_bbox_coverage
            ):
                mode = "adjacent_component_on_open_storage_host"
            if mode is None:
                continue
            extra = np.logical_and(candidate_value, np.logical_not(base))
            extra_area = int(extra.sum())
            if extra_area < max(cfg.minimum_conflict_pixels, int(base_area * 0.05)):
                continue
            exempt_ids = (
                {str(host.get("object_id") or "")}
                if host is not None
                else set()
            )
            conflicts = mask_conflicts(
                extra,
                object_id=object_id,
                exempt_object_ids=exempt_ids,
                maximum_coverage=cfg.maximum_completion_overlap_with_other,
            )
            if conflicts:
                continue
            viable.append(
                {
                    "candidate": candidate,
                    "key": key,
                    "mask": candidate_value,
                    "mode": mode,
                    "score": score,
                    "area_ratio": area_ratio,
                    "previous_area": base_area,
                    "completed_mask": (
                        base | candidate_value
                        if mode == "adjacent_component_on_open_storage_host"
                        else candidate_value
                    ),
                    "completed_area": int(
                        (
                            base | candidate_value
                            if mode == "adjacent_component_on_open_storage_host"
                            else candidate_value
                        ).sum()
                    ),
                    "host_object_id": (
                        str(host.get("object_id") or "")
                        if host is not None
                        else None
                    ),
                    "discarded_unconnected_component_count": (
                        discarded_component_count
                    ),
                }
            )
        if not viable:
            continue
        selected = max(
            viable,
            key=lambda item: (
                item["mode"] == "safe_superset",
                item["completed_area"],
                item["score"],
                str(item["candidate"].get("candidate_id") or ""),
            ),
        )
        completed = selected["completed_mask"]
        selected_keys = {selected["key"]}
        extensions: list[dict[str, Any]] = []

        # A collection can be represented by several adjacent SAM3 components
        # (for example individual book spines).  Once a safe host-local
        # component is selected, extend it only through directly connected,
        # target-local components.  This is deliberately bounded and never
        # crosses into another final object.
        if selected["mode"] == "adjacent_component_on_open_storage_host":
            for _ in range(max(0, cfg.maximum_adjacent_completion_components - 1)):
                current_box = [
                    int(np.nonzero(completed)[1].min()),
                    int(np.nonzero(completed)[0].min()),
                    int(np.nonzero(completed)[1].max()) + 1,
                    int(np.nonzero(completed)[0].max()) + 1,
                ]
                extension_choices = []
                for candidate in candidates:
                    key = candidate_key(candidate)
                    if (
                        not key
                        or key in used_candidate_keys
                        or key in selected_keys
                        or sam3_query_key(candidate) != prompt
                    ):
                        continue
                    score = float(candidate.get("score") or 0.0)
                    candidate_box = valid_bbox(candidate.get("mask_bbox_abs_xyxy"))
                    if (
                        score < cfg.minimum_adjacent_component_score
                        or candidate_box is None
                        or not contains(target_box, center(candidate_box))
                    ):
                        continue
                    try:
                        candidate_value = candidate_mask(candidate)
                    except (OSError, ValueError):
                        continue
                    candidate_area = int(candidate_value.sum())
                    area_ratio = candidate_area / base_area if base_area else 0.0
                    if not (
                        cfg.minimum_adjacent_component_area_ratio
                        <= area_ratio
                        <= cfg.maximum_adjacent_component_area_ratio
                    ):
                        continue
                    if not masks_touch(
                        completed,
                        candidate_value,
                        dilation_pixels=cfg.adjacent_component_contact_dilation_pixels,
                    ):
                        continue
                    host, _ = shared_host(
                        [center(current_box), center(candidate_box)],
                        {object_id},
                    )
                    if host is None or not is_open_storage_host(host):
                        continue
                    extra = np.logical_and(candidate_value, np.logical_not(completed))
                    extra_area = int(extra.sum())
                    if extra_area < max(cfg.minimum_conflict_pixels, int(base_area * 0.05)):
                        continue
                    conflicts = mask_conflicts(
                        extra,
                        object_id=object_id,
                        exempt_object_ids={str(host.get("object_id") or "")},
                        maximum_coverage=cfg.maximum_completion_overlap_with_other,
                    )
                    if conflicts:
                        continue
                    extension_choices.append(
                        {
                            "candidate": candidate,
                            "key": key,
                            "mask": candidate_value,
                            "score": score,
                            "extra_area": extra_area,
                        }
                    )
                if not extension_choices:
                    break
                extension = max(
                    extension_choices,
                    key=lambda item: (
                        item["extra_area"],
                        item["score"],
                        str(item["candidate"].get("candidate_id") or ""),
                    ),
                )
                completed |= extension["mask"]
                selected_keys.add(extension["key"])
                extensions.append(
                    {
                        "candidate_id": extension["candidate"].get("candidate_id"),
                        "candidate_score": extension["score"],
                        "added_area_pixels": extension["extra_area"],
                    }
                )

        completed_box = [
            int(np.nonzero(completed)[1].min()),
            int(np.nonzero(completed)[0].min()),
            int(np.nonzero(completed)[1].max()) + 1,
            int(np.nonzero(completed)[0].max()) + 1,
        ]
        mask_path, overlay_path = write_object_mask(obj, completed)
        prior_sam3 = copy.deepcopy(dict(sam3))
        obj["sam3"] = prior_sam3
        obj["sam3"].update(
            {
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "area_pixels": int(completed.sum()),
                "mask_bbox_abs_xyxy": completed_box,
                "selection_method": "post_delta_same_prompt_mask_completion",
                "multiplicity_completion": {
                    "candidate_id": selected["candidate"].get("candidate_id"),
                    "mode": selected["mode"],
                    "candidate_score": selected["score"],
                    "previous_area_pixels": selected["previous_area"],
                    "completed_area_pixels": int(completed.sum()),
                    "host_object_id": selected["host_object_id"],
                    "extension_candidates": extensions,
                },
            }
        )
        existing_masks[object_id] = completed
        used_candidate_keys.update(selected_keys)
        audit["completed"].append(
            {
                "object_id": object_id,
                **obj["sam3"]["multiplicity_completion"],
                "mask_path": mask_path,
                "overlay_path": overlay_path,
            }
        )

    # A cross-object containment decision may leave a real VLM instance
    # without a mask even though a distinct same-prompt SAM3 candidate remains.
    # Reassign only a peer-sized, target-local candidate on the same verified
    # host and reject any candidate that conflicts with an existing object.
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        prompt = sam3_query_key(obj)
        if (
            sam3.get("status") == "success"
            or len(prompt_members.get(prompt, [])) < cfg.minimum_peer_count
        ):
            continue
        target_box = valid_bbox(sam3.get("vlm_bbox_abs_xyxy"))
        peers = [
            peer
            for peer in prompt_members.get(prompt, [])
            if (peer.get("sam3") or {}).get("status") == "success"
            and str(peer.get("object_id") or "") in existing_masks
        ]
        peer_boxes = [
            valid_bbox((peer.get("sam3") or {}).get("mask_bbox_abs_xyxy"))
            for peer in peers
        ]
        peer_areas = [
            int((peer.get("sam3") or {}).get("area_pixels") or 0)
            for peer in peers
        ]
        if (
            target_box is None
            or len(peers) < cfg.minimum_peer_count
            or any(box is None for box in peer_boxes)
            or not all(peer_areas)
        ):
            continue
        peer_boxes = [box for box in peer_boxes if box is not None]
        median_area = float(np.median(peer_areas))
        median_width = float(
            np.median([box[2] - box[0] for box in peer_boxes])
        )
        median_height = float(
            np.median([box[3] - box[1] for box in peer_boxes])
        )
        prompt_words = set(prompt.split())
        host_not_required = (
            "recessed" in prompt_words
            and bool(prompt_words & {"light", "lamp"})
        )
        viable = []
        for candidate in candidates:
            key = candidate_key(candidate)
            if (
                not key
                or key in used_candidate_keys
                or sam3_query_key(candidate) != prompt
            ):
                continue
            score = float(candidate.get("score") or 0.0)
            candidate_box = valid_bbox(candidate.get("mask_bbox_abs_xyxy"))
            area = int(candidate.get("area_pixels") or 0)
            if (
                score < cfg.minimum_failed_object_candidate_score
                or candidate_box is None
                or not area
                or not contains(target_box, center(candidate_box))
            ):
                continue
            area_ratio = area / median_area if median_area else 0.0
            width_ratio = (
                (candidate_box[2] - candidate_box[0]) / median_width
                if median_width
                else 0.0
            )
            height_ratio = (
                (candidate_box[3] - candidate_box[1]) / median_height
                if median_height
                else 0.0
            )
            if not (
                cfg.minimum_area_ratio
                <= area_ratio
                <= cfg.maximum_area_ratio
                and cfg.minimum_dimension_ratio
                <= width_ratio
                <= cfg.maximum_dimension_ratio
                and cfg.minimum_dimension_ratio
                <= height_ratio
                <= cfg.maximum_dimension_ratio
            ):
                continue
            host, host_box = shared_host(
                [center(box) for box in peer_boxes] + [center(candidate_box)],
                {str(peer.get("object_id") or "") for peer in peers}
                | {object_id},
            )
            if host is None and not host_not_required:
                continue
            try:
                candidate_value = candidate_mask(candidate)
            except (OSError, ValueError) as exc:
                audit["hard_errors"].append(
                    {
                        "type": "invalid_failed_object_candidate",
                        "candidate_id": candidate.get("candidate_id"),
                        "error": str(exc),
                    }
                )
                continue
            exempt_ids = (
                {str(host.get("object_id") or "")}
                if host is not None
                else set()
            )
            conflicts = mask_conflicts(
                candidate_value,
                object_id=object_id,
                exempt_object_ids=exempt_ids,
                maximum_coverage=cfg.maximum_existing_candidate_coverage,
            )
            if conflicts:
                continue
            viable.append(
                {
                    "candidate": candidate,
                    "key": key,
                    "mask": candidate_value,
                    "score": score,
                    "area_ratio": area_ratio,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio,
                    "host": host,
                    "host_box": host_box,
                }
            )
        if not viable:
            continue
        selected = max(
            viable,
            key=lambda item: (
                -abs(1.0 - item["area_ratio"]),
                -abs(1.0 - item["width_ratio"]),
                -abs(1.0 - item["height_ratio"]),
                item["score"],
                str(item["candidate"].get("candidate_id") or ""),
            ),
        )
        rescued = selected["mask"]
        rescued_box = valid_bbox(
            selected["candidate"].get("mask_bbox_abs_xyxy")
        )
        mask_path, overlay_path = write_object_mask(obj, rescued)
        prior_failure = copy.deepcopy(dict(sam3))
        obj["sam3"] = {
            "status": "success",
            "warning": None,
            "prompt_used": selected["candidate"].get("sam3_prompt"),
            "score": selected["score"],
            "area_pixels": int(rescued.sum()),
            "vlm_bbox_abs_xyxy": target_box,
            "mask_bbox_abs_xyxy": rescued_box,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "selection_method": "post_delta_failed_object_reassignment",
            "multiplicity_reassignment": {
                "candidate_id": selected["candidate"].get("candidate_id"),
                "previous_failure": prior_failure,
                "host_object_id": (
                    str(selected["host"].get("object_id") or "")
                    if selected["host"] is not None
                    else None
                ),
                "host_bbox_abs_xyxy": selected["host_box"],
                "area_ratio_to_peer_median": round(
                    selected["area_ratio"], 6
                ),
                "width_ratio_to_peer_median": round(
                    selected["width_ratio"], 6
                ),
                "height_ratio_to_peer_median": round(
                    selected["height_ratio"], 6
                ),
            },
        }
        existing_masks[object_id] = rescued
        used_candidate_keys.add(selected["key"])
        audit["rescued_failed"].append(
            {
                "object_id": object_id,
                "candidate_id": selected["candidate"].get("candidate_id"),
                "prompt": prompt,
                "host_object_id": (
                    str(selected["host"].get("object_id") or "")
                    if selected["host"] is not None
                    else None
                ),
                "mask_path": mask_path,
                "overlay_path": overlay_path,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in copied:
        prompt = sam3_query_key(obj)
        if prompt and not is_host(obj) and (obj.get("sam3") or {}).get("status") == "success":
            groups[prompt].append(obj)

    total_added = 0
    for prompt, peers in sorted(groups.items()):
        prompt_words = set(prompt.split())
        single_peer_repeat = (
            len(peers) == 1
            and bool(prompt_words & SINGLE_PEER_REPEAT_WORDS)
        )
        single_peer_suspended_light = (
            len(peers) == 1
            and bool(prompt_words & SINGLE_PEER_SUSPENDED_LIGHT_WORDS)
            and bool(prompt_words & {"light", "lamp"})
        )
        single_peer_exception = single_peer_repeat or single_peer_suspended_light
        repeated_recessed_ceiling_light = (
            len(peers) >= cfg.minimum_peer_count
            and "recessed" in prompt_words
            and bool(prompt_words & {"light", "lamp"})
        )
        host_not_required = (
            single_peer_exception or repeated_recessed_ceiling_light
        )
        if len(peers) < cfg.minimum_peer_count and not single_peer_exception:
            continue
        peer_boxes = [valid_bbox((peer.get("sam3") or {}).get("mask_bbox_abs_xyxy")) for peer in peers]
        peer_boxes = [box for box in peer_boxes if box]
        peer_areas = [int((peer.get("sam3") or {}).get("area_pixels") or 0) for peer in peers]
        if len(peer_boxes) != len(peers) or not all(peer_areas):
            audit["rejected"].append({"prompt": prompt, "reason": "invalid_peer_geometry"})
            continue
        group_candidates = [
            candidate for candidate in candidates
            if candidate_key(candidate) not in used_candidate_keys
            if sam3_query_key(candidate) == prompt
        ]
        audit["groups_checked"].append(
            {"prompt": prompt, "peer_object_ids": [p.get("object_id") for p in peers], "candidate_count": len(group_candidates)}
        )
        for candidate in sorted(group_candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True):
            if total_added >= cfg.maximum_total_additions:
                audit["rejected"].append({"candidate_id": candidate.get("candidate_id"), "reason": "total_budget_exceeded"})
                continue
            already_for_group = sum(1 for item in audit["accepted"] if item.get("prompt") == prompt)
            if already_for_group >= cfg.maximum_additions_per_group:
                audit["rejected"].append({"candidate_id": candidate.get("candidate_id"), "reason": "group_budget_exceeded"})
                continue
            score = float(candidate.get("score") or 0.0)
            box = valid_bbox(candidate.get("mask_bbox_abs_xyxy"))
            minimum_score = cfg.minimum_candidate_score
            if single_peer_repeat:
                minimum_score = cfg.minimum_single_peer_score
            elif single_peer_suspended_light:
                minimum_score = cfg.minimum_single_peer_suspended_light_score
            elif repeated_recessed_ceiling_light:
                minimum_score = cfg.minimum_recessed_ceiling_light_score
            if score < minimum_score or not box:
                audit["rejected"].append({"candidate_id": candidate.get("candidate_id"), "reason": "score_or_bbox_gate"})
                continue
            area = int(candidate.get("area_pixels") or 0)
            median_area = float(np.median(peer_areas))
            area_ratio = area / median_area if median_area else 0.0
            widths = [b[2] - b[0] for b in peer_boxes]
            heights = [b[3] - b[1] for b in peer_boxes]
            width_ratio = (box[2] - box[0]) / float(np.median(widths))
            height_ratio = (box[3] - box[1]) / float(np.median(heights))
            minimum_area_ratio = cfg.minimum_area_ratio
            minimum_dimension_ratio = cfg.minimum_dimension_ratio
            if single_peer_repeat:
                minimum_area_ratio = cfg.minimum_single_peer_area_ratio
                minimum_dimension_ratio = cfg.minimum_single_peer_dimension_ratio
            elif single_peer_suspended_light:
                minimum_area_ratio = cfg.minimum_single_peer_suspended_light_area_ratio
                minimum_dimension_ratio = cfg.minimum_single_peer_suspended_light_dimension_ratio
            if not (
                minimum_area_ratio <= area_ratio <= cfg.maximum_area_ratio
                and minimum_dimension_ratio <= width_ratio <= cfg.maximum_dimension_ratio
                and minimum_dimension_ratio <= height_ratio <= cfg.maximum_dimension_ratio
            ):
                audit["rejected"].append({"candidate_id": candidate.get("candidate_id"), "reason": "peer_geometry_gate", "area_ratio": area_ratio, "width_ratio": width_ratio, "height_ratio": height_ratio})
                continue
            if (
                repeated_recessed_ceiling_light
                and center(box)[1] / height
                > cfg.maximum_recessed_ceiling_light_y_fraction
            ):
                audit["rejected"].append({
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": "recessed_ceiling_light_outside_top_zone",
                })
                continue

            # Repeated units may live on different shelves. The candidate only
            # needs one verified local host; global peers provide the repeated
            # category geometry and must not all fit into that same host.
            host, host_box = shared_host(
                [center(box)],
                {str(peer.get("object_id") or "") for peer in peers},
            )
            if host is None and not host_not_required:
                audit["rejected"].append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "reason": "no_local_verified_host",
                    }
                )
                continue
            try:
                mask, source_mask = load_mask(candidate.get("mask_path"), shape)
            except (OSError, ValueError) as exc:
                audit["hard_errors"].append({"type": "invalid_unassigned_candidate", "candidate_id": candidate.get("candidate_id"), "error": str(exc)})
                continue
            conflicts = []
            exempt_ids = {str(host.get("object_id") or "")} if host else set()
            for object_id, existing in existing_masks.items():
                if object_id in exempt_ids:
                    continue
                metrics = overlap(mask, existing)
                if (
                    metrics["intersection_pixels"] >= cfg.minimum_conflict_pixels
                    and (
                        metrics["iou"] >= cfg.maximum_existing_iou
                        or metrics["candidate_coverage"] >= cfg.maximum_existing_candidate_coverage
                    )
                ):
                    conflicts.append({"object_id": object_id, **metrics})
            if conflicts:
                audit["rejected"].append({"candidate_id": candidate.get("candidate_id"), "reason": "overlaps_existing_nonhost", "conflicts": conflicts})
                continue

            template = copy.deepcopy(peers[0])
            object_id = next_object_id(copied)
            category_slug = normalize(template.get("category")).replace(" ", "_") or "object"
            staging = Path(staging_dir).expanduser().resolve()
            mask_path = staging / "masks" / f"{object_id}_{category_slug}_mask.png"
            overlay_path = staging / "overlays" / f"{object_id}_{category_slug}_overlay.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            overlay_image = image_np.copy()
            overlay_image[mask] = (0.55 * overlay_image[mask] + 0.45 * np.array((255, 0, 0), dtype=np.uint8)).astype(np.uint8)
            Image.fromarray(overlay_image).save(overlay_path)
            cx, cy = center(box)
            row = max(1, min(10, int(cy / height * 10) + 1))
            col = max(1, min(10, int(cx / width * 10) + 1))
            template.update({
                "object_id": object_id,
                "instance_name": f"{normalize(template.get('category')) or 'object'} recovered r{row}c{col}",
                "bbox_2d_norm_xyxy": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
                "anchor_2d_norm_xy": [cx / width, cy / height],
                "anchor_source": "sam3_candidate_centroid",
                "coarse_location": f"grid_r{row}_c{col}",
                "visual_order": {"row": row, "col": col},
                "proposal_sources": [{
                    "pass": "local",
                    "source_id": candidate.get("candidate_id"),
                    "action": "post_delta_multiplicity_closure",
                    "host_object_id": host.get("object_id") if host else None,
                }],
                "sam3": {
                    "status": "success",
                    "warning": None,
                    "prompt_used": candidate.get("sam3_prompt"),
                    "score": score,
                    "area_pixels": area,
                    "vlm_bbox_abs_xyxy": None,
                    "mask_bbox_abs_xyxy": box,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "selection_method": "post_delta_same_prompt_multiplicity_closure",
                    "multiplicity_closure": {
                        "candidate_id": candidate.get("candidate_id"),
                        "peer_object_ids": [peer.get("object_id") for peer in peers],
                        "host_object_id": host.get("object_id") if host else None,
                        "host_bbox_abs_xyxy": host_box,
                        "area_ratio_to_peer_median": round(area_ratio, 6),
                        "width_ratio_to_peer_median": round(width_ratio, 6),
                        "height_ratio_to_peer_median": round(height_ratio, 6),
                    },
                },
            })
            copied.append(template)
            existing_masks[object_id] = mask
            total_added += 1
            audit["accepted"].append({
                "object_id": object_id,
                "candidate_id": candidate.get("candidate_id"),
                "prompt": prompt,
                "peer_object_ids": [peer.get("object_id") for peer in peers],
                "host_object_id": host.get("object_id") if host else None,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            })

    audit["accepted_count"] = len(audit["accepted"])
    audit["completed_count"] = len(audit["completed"])
    audit["rescued_failed_count"] = len(audit["rescued_failed"])
    changed = bool(
        audit["accepted"]
        or audit["completed"]
        or audit["rescued_failed"]
    )
    audit["status"] = (
        "failed"
        if audit["hard_errors"]
        else ("pass" if changed else "noop")
    )
    atomic_json(audit_path, audit)
    return {"objects": copied, "audit": audit, "audit_path": str(audit_path)}
