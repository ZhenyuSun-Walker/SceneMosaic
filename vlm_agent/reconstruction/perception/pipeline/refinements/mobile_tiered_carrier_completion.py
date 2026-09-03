"""Post-Pass-B completion for small mobile tiered carriers."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MobileTieredCarrierCompletionConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    maximum_hosts: int = 2
    minimum_named_local_children: int = 2
    minimum_dominant_category_count: int = 2
    minimum_dominant_category_fraction: float = 0.66
    maximum_candidates_per_top_probe: int = 6
    minimum_top_candidate_score: float = 0.80
    minimum_candidate_inside_top_roi: float = 0.90
    maximum_top_candidate_host_overlap: float = 0.15
    maximum_other_object_candidate_coverage: float = 0.15
    minimum_candidate_area_pixels: int = 64
    top_roi_horizontal_padding_fraction: float = 0.10
    top_roi_upward_padding_fraction: float = 0.40
    top_roi_downward_fraction: float = 0.18
    top_child_center_fraction: float = 0.18
    minimum_same_band_vertical_overlap: float = 0.50
    maximum_same_band_gap_fraction: float = 0.03
    maximum_supported_tiers: int = 4


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def is_mobile_tiered_host(obj: Mapping[str, Any]) -> bool:
    words: set[str] = set()
    for key in ("category", "description", "instance_name", "sam3_prompt"):
        words |= _tokens(obj.get(key))
    return bool(words & {"cart", "trolley"}) and bool(
        words & {"rolling", "storage", "serving", "utility"}
    )


def _mask(path_value: Any, shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.asarray(Image.open(path).convert("L")) > 0
    if value.shape != shape or not value.any():
        raise ValueError(f"invalid mask: {path}")
    return value


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _claimed_host_box(
    host_record: Mapping[str, Any],
    fallback: Sequence[int],
    shape: tuple[int, int],
) -> list[int]:
    """Prefer the original VLM claim for tier geometry, not an expanded mask."""

    height, width = shape
    raw = (host_record.get("sam3") or {}).get("vlm_bbox_abs_xyxy")
    try:
        box = [int(round(float(value))) for value in raw]
    except (TypeError, ValueError):
        return [int(value) for value in fallback]
    if len(box) != 4 or not (
        0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
    ):
        return [int(value) for value in fallback]
    return box


def _inside(mask: np.ndarray, box: Sequence[int]) -> float:
    x1, y1, x2, y2 = (int(value) for value in box)
    return int(mask[y1:y2, x1:x2].sum()) / max(1, int(mask.sum()))


def _local_ids(obj: Mapping[str, Any]) -> list[str]:
    return [
        str(source.get("source_id") or "")
        for source in (obj.get("proposal_sources") or [])
        if isinstance(source, Mapping)
        and str(source.get("pass") or "") == "local_candidate"
        and str(source.get("source_id") or "")
    ]


def _has_source(obj: Mapping[str, Any], source_id: str) -> bool:
    return any(
        isinstance(source, Mapping)
        and str(source.get("source_id") or "") == source_id
        for source in (obj.get("proposal_sources") or [])
    )


def _same_band(
    left: Sequence[int],
    right: Sequence[int],
    host_height: int,
    cfg: MobileTieredCarrierCompletionConfig,
) -> bool:
    ly1, ly2 = int(left[1]), int(left[3])
    ry1, ry2 = int(right[1]), int(right[3])
    overlap = max(0, min(ly2, ry2) - max(ly1, ry1))
    minimum_height = max(1, min(ly2 - ly1, ry2 - ry1))
    if overlap / minimum_height >= cfg.minimum_same_band_vertical_overlap:
        return True
    gap = max(0, max(ly1, ry1) - min(ly2, ry2))
    return gap <= max(2, round(cfg.maximum_same_band_gap_fraction * host_height))


def _group_bands(
    rows: Sequence[Mapping[str, Any]],
    host_height: int,
    cfg: MobileTieredCarrierCompletionConfig,
) -> list[list[dict[str, Any]]]:
    pending = [dict(row) for row in rows]
    groups: list[list[dict[str, Any]]] = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            retained = []
            for row in pending:
                if any(
                    _same_band(row["bbox"], member["bbox"], host_height, cfg)
                    for member in group
                ):
                    group.append(row)
                    changed = True
                else:
                    retained.append(row)
            pending = retained
        groups.append(group)
    groups.sort(key=lambda group: min(int(row["bbox"][1]) for row in group))
    return groups


def _plural_category(category: str) -> str:
    words = str(category or "object").replace("_", " ").strip().split()
    if not words:
        return "objects"
    last = words[-1]
    if last.endswith("y") and not last.endswith(("ay", "ey", "iy", "oy", "uy")):
        words[-1] = last[:-1] + "ies"
    elif not last.endswith("s"):
        words[-1] = last + "s"
    return " ".join(words)


def _top_roi(
    host_box: Sequence[int],
    shape: tuple[int, int],
    cfg: MobileTieredCarrierCompletionConfig,
) -> list[int]:
    height, width = shape
    x1, y1, x2, y2 = (int(value) for value in host_box)
    host_width = max(1, x2 - x1)
    host_height = max(1, y2 - y1)
    return [
        max(0, x1 - round(cfg.top_roi_horizontal_padding_fraction * host_width)),
        max(0, y1 - round(cfg.top_roi_upward_padding_fraction * host_height)),
        min(width, x2),
        min(height, y1 + round(cfg.top_roi_downward_fraction * host_height)),
    ]


def _save(
    mask: np.ndarray,
    image: np.ndarray,
    mask_path: Path,
) -> tuple[str, str]:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    overlay = image.copy()
    overlay[mask] = (
        0.55 * overlay[mask]
        + 0.45 * np.array((36, 170, 235), dtype=np.uint8)
    ).astype(np.uint8)
    overlay_path = (
        mask_path.parent.parent
        / "overlays"
        / mask_path.name.replace("_mask.png", "_overlay.png")
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_path)
    return str(mask_path), str(overlay_path)


def _tier_names(count: int, has_verified_top: bool) -> list[str]:
    if count <= 0:
        return []
    if not has_verified_top:
        return [f"tier_{index}" for index in range(1, count + 1)]
    if count == 1:
        return ["top"]
    if count == 2:
        return ["top", "bottom"]
    if count == 3:
        return ["top", "middle", "bottom"]
    return ["top"] + [f"tier_{index}" for index in range(2, count)] + ["bottom"]


def run_mobile_tiered_carrier_completion(
    *,
    session: Any,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    local_candidate_metadata: Sequence[Mapping[str, Any]],
    sam3_a_registry: Mapping[str, Any],
    staging_dir: str | Path,
    work_dir: str | Path,
    infer_candidates_fn: Callable[..., Mapping[str, Any]],
    config: MobileTieredCarrierCompletionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MobileTieredCarrierCompletionConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("mobile tiered carrier completion is isolated-attempt only")
    copied = [copy.deepcopy(dict(obj)) for obj in objects]
    work = Path(work_dir).expanduser().resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"mobile tier work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    staging = Path(staging_dir).expanduser().resolve()
    image_file = Path(image_path).expanduser().resolve()
    image = np.asarray(Image.open(image_file).convert("RGB"), dtype=np.uint8)
    shape = image.shape[:2]
    height, width = shape
    audit: dict[str, Any] = {
        "schema_version": "scenemosaic_mobile_tiered_carrier_completion",
        "config": asdict(cfg),
        "status": "disabled" if not cfg.enabled else "pending",
        "target_host_count": 0,
        "processed_host_count": 0,
        "inference_call_count": 0,
        "replaced_atomic_object_ids": [],
        "created_tier_objects": [],
        "safe_abstentions": [],
        "hard_errors": [],
        "additional_vlm_calls": 0,
        "formal_runs_modified": False,
    }
    if not cfg.enabled:
        (work / "mobile_tiered_carrier_completion_audit.json").write_text(
            json.dumps(audit, indent=2)
        )
        return {"objects": copied, "audit": audit}

    metadata_by_id = {
        str(row.get("candidate_id") or ""): dict(row)
        for row in local_candidate_metadata
        if isinstance(row, Mapping) and str(row.get("candidate_id") or "")
    }
    host_registry = {
        str(obj.get("object_id") or ""): dict(obj)
        for obj in (sam3_a_registry.get("objects") or [])
        if isinstance(obj, Mapping) and is_mobile_tiered_host(obj)
    }
    masks: dict[str, np.ndarray] = {}
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        if (obj.get("sam3") or {}).get("status") != "success":
            continue
        try:
            masks[object_id] = _mask((obj.get("sam3") or {}).get("mask_path"), shape)
        except (OSError, ValueError):
            audit["hard_errors"].append({
                "type": "invalid_success_mask",
                "object_id": object_id,
            })

    host_children: dict[str, list[dict[str, Any]]] = {}
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        local_ids = _local_ids(obj)
        if not local_ids or object_id not in masks:
            continue
        host_ids = {
            str((metadata_by_id.get(local_id) or {}).get("host_source_object_id") or "")
            for local_id in local_ids
        }
        host_ids.discard("")
        if len(host_ids) != 1:
            continue
        host_id = next(iter(host_ids))
        if host_id not in host_registry:
            continue
        host_children.setdefault(host_id, []).append({
            "object": obj,
            "object_id": object_id,
            "local_ids": local_ids,
            "mask": masks[object_id],
            "bbox": _bbox(masks[object_id]),
        })

    eligible_hosts = [
        host_id
        for host_id, children in host_children.items()
        if len(children) >= cfg.minimum_named_local_children
    ]
    eligible_hosts.sort()
    audit["target_host_count"] = min(len(eligible_hosts), cfg.maximum_hosts)
    for host_id in eligible_hosts[cfg.maximum_hosts:]:
        audit["safe_abstentions"].append({
            "host_source_object_id": host_id,
            "reason": "host_budget_exceeded",
        })

    replacements: dict[str, list[dict[str, Any]]] = {}
    suppressed: set[str] = set()
    for host_id in eligible_hosts[:cfg.maximum_hosts]:
        children = host_children[host_id]
        categories = [str(row["object"].get("category") or "unknown") for row in children]
        dominant_category, dominant_count = Counter(categories).most_common(1)[0]
        if (
            dominant_count < cfg.minimum_dominant_category_count
            or dominant_count / len(categories) < cfg.minimum_dominant_category_fraction
        ):
            audit["safe_abstentions"].append({
                "host_source_object_id": host_id,
                "reason": "no_dominant_named_child_category",
                "categories": categories,
            })
            continue
        host_record = host_registry[host_id]
        try:
            host_mask = _mask((host_record.get("sam3") or {}).get("mask_path"), shape)
        except (OSError, ValueError):
            audit["hard_errors"].append({
                "type": "invalid_mobile_host_mask",
                "host_source_object_id": host_id,
            })
            continue
        host_mask_box = _bbox(host_mask)
        host_box = _claimed_host_box(host_record, host_mask_box, shape)
        host_height = max(1, host_box[3] - host_box[1])
        host_description = str(
            host_record.get("description")
            or host_record.get("sam3_prompt")
            or host_record.get("category")
            or "mobile storage cart"
        ).replace("_", " ").strip()
        top_prompt = (
            f"all {_plural_category(dominant_category)} on the top tier of the "
            f"{host_description}"
        )
        top_roi = _top_roi(host_box, shape, cfg)
        audit["inference_call_count"] += 1
        try:
            response = infer_candidates_fn(
                session,
                image_file,
                top_prompt,
                work / f"host_{host_id.lower()}" / "top_probe",
                score_threshold=0.20,
                positive_bbox_abs_xyxy=top_roi,
                maximum_candidates=cfg.maximum_candidates_per_top_probe,
            )
            raw_candidates = (response.get("registry") or {}).get("candidates") or []
        except Exception as exc:
            audit["safe_abstentions"].append({
                "host_source_object_id": host_id,
                "reason": f"top_probe_exception:{type(exc).__name__}",
            })
            raw_candidates = []

        child_ids = {row["object_id"] for row in children}
        final_host_ids = {
            str(obj.get("object_id") or "")
            for obj in copied
            if _has_source(obj, host_id)
        }
        viable = []
        for raw in raw_candidates:
            try:
                candidate = _mask(raw.get("mask_path"), shape)
            except (OSError, ValueError):
                continue
            score = float(raw.get("score") or 0.0)
            area = int(candidate.sum())
            inside = _inside(candidate, top_roi)
            host_overlap = int(np.logical_and(candidate, host_mask).sum()) / max(1, area)
            conflicts = []
            for other_id, other_mask in masks.items():
                if other_id in child_ids or other_id in final_host_ids:
                    continue
                coverage = int(np.logical_and(candidate, other_mask).sum()) / max(1, area)
                if coverage > cfg.maximum_other_object_candidate_coverage:
                    conflicts.append({
                        "object_id": other_id,
                        "candidate_coverage": round(coverage, 6),
                    })
            if (
                score < cfg.minimum_top_candidate_score
                or area < cfg.minimum_candidate_area_pixels
                or inside < cfg.minimum_candidate_inside_top_roi
                or host_overlap > cfg.maximum_top_candidate_host_overlap
                or conflicts
            ):
                continue
            viable.append({
                "candidate_id": str(raw.get("candidate_id") or ""),
                "mask": candidate,
                "score": score,
                "inside_top_roi": inside,
                "host_overlap": host_overlap,
                "area_pixels": area,
            })
        viable.sort(
            key=lambda row: (float(row["score"]), -float(row["host_overlap"])),
            reverse=True,
        )
        top_candidate = viable[0] if viable else None
        top_limit = host_box[1] + round(cfg.top_child_center_fraction * host_height)
        top_children = [
            row
            for row in children
            if (int(row["bbox"][1]) + int(row["bbox"][3])) / 2 <= top_limit
        ]
        lower_children = [row for row in children if row not in top_children]
        lower_groups = _group_bands(lower_children, host_height, cfg)
        tier_masks: list[np.ndarray] = []
        tier_members: list[list[dict[str, Any]]] = []
        if top_candidate is not None:
            tier_masks.append(np.asarray(top_candidate["mask"], dtype=bool))
            tier_members.append(top_children)
        elif top_children:
            tier_masks.append(np.logical_or.reduce([row["mask"] for row in top_children]))
            tier_members.append(top_children)
            audit["safe_abstentions"].append({
                "host_source_object_id": host_id,
                "reason": "top_semantic_probe_failed_preserved_named_top_children",
            })
        for group in lower_groups:
            tier_masks.append(np.logical_or.reduce([row["mask"] for row in group]))
            tier_members.append(group)
        if not tier_masks:
            audit["safe_abstentions"].append({
                "host_source_object_id": host_id,
                "reason": "no_tier_masks_after_grouping",
            })
            continue
        if len(tier_masks) > cfg.maximum_supported_tiers:
            audit["hard_errors"].append({
                "type": "mobile_tier_count_exceeds_bound",
                "host_source_object_id": host_id,
                "actual": len(tier_masks),
                "maximum": cfg.maximum_supported_tiers,
            })
            continue

        tier_names = _tier_names(len(tier_masks), top_candidate is not None)
        created = []
        all_child_sources: list[dict[str, Any]] = []
        for row in children:
            all_child_sources.extend(copy.deepcopy(row["object"].get("proposal_sources") or []))
        used_source_keys: set[tuple[str, str]] = set()
        for tier_index, (tier_name, tier_mask, members) in enumerate(
            zip(tier_names, tier_masks, tier_members), 1
        ):
            member_sources: list[dict[str, Any]] = []
            for row in members:
                for source in copy.deepcopy(row["object"].get("proposal_sources") or []):
                    key = (str(source.get("pass") or ""), str(source.get("source_id") or ""))
                    if key not in used_source_keys:
                        member_sources.append(source)
                        used_source_keys.add(key)
            source_id = f"{host_id}:mobile_tier:{tier_name}"
            member_sources.append({
                "pass": "local_tier_completion",
                "source_id": source_id,
                "action": "replace_atomic_children_with_tier",
            })
            category = f"rolling_cart_{tier_name}_tier_contents"
            object_id = min(
                (str(row["object_id"]) for row in members),
                default=f"proposal_mobile_{host_id.lower()}_{tier_index:02d}",
            )
            mask_path, overlay_path = _save(
                tier_mask,
                image,
                staging / "masks" / f"{object_id}_{category}_mask.png",
            )
            box = _bbox(tier_mask)
            base = copy.deepcopy(members[0]["object"]) if members else {}
            base.update({
                "object_id": object_id,
                "category": category,
                "sam3_prompt": top_prompt if tier_name == "top" else dominant_category.replace("_", " "),
                "instance_name": f"{host_description} {tier_name} tier contents",
                "object_type": "small_object_group",
                "bbox_2d_norm_xyxy": [
                    box[0] / width,
                    box[1] / height,
                    box[2] / width,
                    box[3] / height,
                ],
                "anchor_2d_norm_xy": [
                    (box[0] + box[2]) / (2 * width),
                    (box[1] + box[3]) / (2 * height),
                ],
                "description": (
                    f"all visible {dominant_category.replace('_', ' ')} contents "
                    f"on the {tier_name} tier of the {host_description}, grouped "
                    "as one reconstruction object"
                ),
                "fallback_prompts": [],
                "proposal_sources": member_sources,
                "sam3": {
                    "status": "success",
                    "score": (
                        float(top_candidate["score"])
                        if tier_name == "top" and top_candidate is not None
                        else max(
                            float((row["object"].get("sam3") or {}).get("score") or 0.0)
                            for row in members
                        )
                    ),
                    "area_pixels": int(tier_mask.sum()),
                    "mask_bbox_abs_xyxy": box,
                    "vlm_bbox_abs_xyxy": box,
                    "vlm_bbox_iou": 1.0,
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "selection_method": "mobile_tiered_carrier_semantic_top_and_named_band_union",
                    "mobile_tiered_carrier_completion": {
                        "host_source_object_id": host_id,
                        "tier_name": tier_name,
                        "dominant_child_category": dominant_category,
                        "replaced_object_ids": [row["object_id"] for row in members],
                        "top_semantic_candidate_id": (
                            top_candidate["candidate_id"]
                            if tier_name == "top" and top_candidate is not None
                            else None
                        ),
                    },
                },
            })
            created.append(base)
            audit["created_tier_objects"].append({
                "host_source_object_id": host_id,
                "intermediate_object_id": object_id,
                "tier_name": tier_name,
                "category": category,
                "area_pixels": int(tier_mask.sum()),
                "bbox_abs_xyxy": box,
                "replaced_object_ids": [row["object_id"] for row in members],
                "top_semantic_candidate_id": (
                    top_candidate["candidate_id"]
                    if tier_name == "top" and top_candidate is not None
                    else None
                ),
            })
        replacements[host_id] = created
        suppressed |= child_ids
        audit["replaced_atomic_object_ids"].extend(sorted(child_ids))
        audit["processed_host_count"] += 1

    if replacements:
        first_child_index: dict[str, int] = {}
        for index, obj in enumerate(copied):
            object_id = str(obj.get("object_id") or "")
            for host_id, children in host_children.items():
                if any(row["object_id"] == object_id for row in children):
                    first_child_index.setdefault(host_id, index)
        output_objects: list[dict[str, Any]] = []
        for index, obj in enumerate(copied):
            for host_id, insertion_index in first_child_index.items():
                if index == insertion_index and host_id in replacements:
                    output_objects.extend(replacements[host_id])
            if str(obj.get("object_id") or "") not in suppressed:
                output_objects.append(obj)
        copied = output_objects

    audit["replaced_atomic_object_ids"] = sorted(set(audit["replaced_atomic_object_ids"]))
    audit["status"] = (
        "blocked"
        if audit["hard_errors"]
        else "pass"
        if audit["processed_host_count"]
        else "pass_with_safe_abstentions"
        if audit["safe_abstentions"]
        else "noop"
    )
    audit_path = work / "mobile_tiered_carrier_completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    return {"objects": copied, "audit": audit, "audit_path": str(audit_path)}


__all__ = [
    "MobileTieredCarrierCompletionConfig",
    "is_mobile_tiered_host",
    "run_mobile_tiered_carrier_completion",
]
