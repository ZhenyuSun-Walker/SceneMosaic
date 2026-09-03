#!/usr/bin/env python3
"""Bounded functional-assembly normalization and local ROI refinement.

This stage handles two explicit geometry patterns: either a broad rear counter
claim crosses a visible range/stove, or Pass A already separates a sink
assembly and an adjacent counter on opposite sides of the range but their SAM3
prompts still need bounded ROI refinement.  It also completes explicit bed
objects as one physical bed assembly (headboard, frame, mattress, and duvet)
while subtracting independently inventoried pillow masks.  It never publishes
formal outputs or calls a VLM API.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


COMPLETE_BED_PROMPT = (
    "complete bed including frame headboard mattress and duvet excluding pillows"
)
COMPLETE_SINK_VANITY_PROMPT = (
    "complete sink vanity including cabinet countertop basin and fixed faucet"
)
COMPLETE_DOUBLE_SINK_VANITY_PROMPT = (
    "complete double sink vanity including cabinet countertop two basins and fixed faucets"
)
COMPLETE_POTTED_PLANT_PROMPT = (
    "complete potted plant including foliage and pot"
)


@dataclass(frozen=True)
class AssemblyConfig:
    maximum_assemblies: int = 2
    maximum_targets: int = 6
    maximum_candidates_per_target: int = 16
    score_threshold: float = 0.20
    minimum_selected_score: float = 0.45
    minimum_target_bbox_coverage: float = 0.55
    minimum_candidate_bbox_coverage: float = 0.35
    maximum_center_distance_scale: float = 0.75
    ambiguity_quality_gap: float = 0.12
    ambiguity_mask_iou: float = 0.20
    minimum_side_width: float = 0.05
    maximum_sink_vanity_assemblies: int = 2
    maximum_fixed_faucet_unions: int = 2
    minimum_source_mask_coverage: float = 0.85
    maximum_protected_object_removal_fraction: float = 0.35


@dataclass(frozen=True)
class PlantContainerMaskCompletionConfig:
    maximum_assemblies: int = 2
    minimum_candidate_score: float = 0.60
    minimum_candidate_overlap: float = 0.35
    maximum_candidate_overlap: float = 0.92
    minimum_new_pixels: int = 64
    minimum_new_pixel_bbox_fraction: float = 0.75
    minimum_candidate_bbox_fraction: float = 0.70
    minimum_downward_extension_pixels: int = 8
    maximum_other_object_overlap: float = 0.05
    ambiguity_score_gap: float = 0.05


def _tokens(obj: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        str(obj.get(key) or "")
        for key in ("category", "sam3_prompt", "instance_name", "description")
    ).lower()
    return set(re.findall(r"[a-z0-9]+", text))


def _category_tokens(obj: Mapping[str, Any]) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(obj.get("category") or "").lower().replace("_", " "),
        )
    )


def _bbox(obj: Mapping[str, Any]) -> list[float]:
    try:
        box = [float(value) for value in obj.get("bbox_2d_norm_xyxy")]
    except (TypeError, ValueError):
        raise ValueError("invalid bbox") from None
    if (
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or not (0.0 <= box[0] < box[2] <= 1.0)
        or not (0.0 <= box[1] < box[3] <= 1.0)
    ):
        raise ValueError("invalid bbox")
    return box


def _round_box(box: Sequence[float]) -> list[float]:
    return [round(float(value), 6) for value in box]


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _is_sink(obj: Mapping[str, Any]) -> bool:
    return bool(_tokens(obj) & {"sink", "washbasin", "basin"})


def _is_counter_support(obj: Mapping[str, Any]) -> bool:
    tokens = _category_tokens(obj)
    if _is_sink(obj) or tokens & {"island", "pantry", "upper", "wall"}:
        return False
    return bool(tokens & {"counter", "countertop", "worktable"}) or (
        {"base", "cabinet"} <= tokens
    )


def _is_separator(obj: Mapping[str, Any]) -> bool:
    tokens = _category_tokens(obj)
    return bool(tokens & {"range", "stove"}) and not bool(
        tokens & {"hood", "vent"}
    )


def _is_bed(obj: Mapping[str, Any]) -> bool:
    category = _category_tokens(obj)
    return "bed" in category and not bool(
        category & {"bench", "bedside", "headboard"}
    )


def _is_pillow(obj: Mapping[str, Any]) -> bool:
    category = _category_tokens(obj)
    return bool(category & {"pillow", "cushion"}) and not bool(
        category & {"bench", "chair", "sofa"}
    )


def _is_plant(obj: Mapping[str, Any]) -> bool:
    return "plant" in _category_tokens(obj)


def _is_vase_like(obj: Mapping[str, Any]) -> bool:
    return bool(_category_tokens(obj) & {"pot", "planter", "vase"})


def _match_potted_plant_pattern(
    plant: Mapping[str, Any],
    vessel: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Prove a foliage/vessel pair is one potted-plant assembly from boxes."""

    plant_box = _bbox(plant)
    vessel_box = _bbox(vessel)
    vessel_width = vessel_box[2] - vessel_box[0]
    vessel_height = vessel_box[3] - vessel_box[1]
    plant_area = (plant_box[2] - plant_box[0]) * (plant_box[3] - plant_box[1])
    vessel_area = vessel_width * vessel_height
    horizontal_coverage = _overlap_1d(
        plant_box[0], plant_box[2], vessel_box[0], vessel_box[2]
    ) / max(vessel_width, 1e-9)
    vertical_coverage = _overlap_1d(
        plant_box[1], plant_box[3], vessel_box[1], vessel_box[3]
    ) / max(vessel_height, 1e-9)
    if (
        horizontal_coverage < 0.90
        or vertical_coverage < 0.40
        or vessel_area > 0.75 * plant_area
        or _center(vessel_box)[1] < _center(plant_box)[1]
    ):
        return None
    union = [
        min(plant_box[0], vessel_box[0]),
        min(plant_box[1], vessel_box[1]),
        max(plant_box[2], vessel_box[2]),
        max(plant_box[3], vessel_box[3]),
    ]
    return {
        "plant": plant,
        "vessel": vessel,
        "target_bbox_norm_xyxy": _round_box(union),
        "vessel_horizontal_coverage": round(horizontal_coverage, 6),
        "vessel_vertical_coverage": round(vertical_coverage, 6),
        "vessel_to_plant_bbox_area_ratio": round(
            vessel_area / max(plant_area, 1e-9), 6
        ),
    }


def _is_sink_vanity(obj: Mapping[str, Any]) -> bool:
    """A cabinet-mounted sink whose fixed faucet belongs to the same instance."""

    category = _category_tokens(obj)
    return (
        "vanity" in category
        and bool(category & {"sink", "basin"})
        and "kitchen" not in _tokens(obj)
    )


def _is_bathtub(obj: Mapping[str, Any]) -> bool:
    category = _category_tokens(obj)
    return "bathtub" in category and "faucet" not in category


def _is_fixed_bathtub_faucet(obj: Mapping[str, Any]) -> bool:
    category = _category_tokens(obj)
    return "bathtub" in category and "faucet" in category


def _is_fixed_faucet(obj: Mapping[str, Any]) -> bool:
    return "faucet" in _category_tokens(obj)


def _expand_box_upward(box: Sequence[float], fraction: float = 0.18) -> list[float]:
    """Reserve only the shallow band where fixed taps sit above a vanity."""

    height = box[3] - box[1]
    return _round_box(
        [
            box[0],
            max(0.0, box[1] - max(0.04, min(0.12, height * fraction))),
            box[2],
            box[3],
        ]
    )


def _is_absorbable_bed_component(obj: Mapping[str, Any]) -> bool:
    """Keep physical bed parts in the assembly while protecting other objects."""

    category = _category_tokens(obj)
    return bool(
        category
        & {
            "bedframe",
            "bedding",
            "bedspread",
            "blanket",
            "comforter",
            "duvet",
            "frame",
            "headboard",
            "mattress",
            "quilt",
            "sheet",
        }
    )


def _is_absorbable_component(obj: Mapping[str, Any]) -> bool:
    tokens = _tokens(obj)
    return "dishwasher" in tokens or (
        {"under", "counter", "appliance"} <= tokens
    )


def _match_pattern(
    sink: Mapping[str, Any],
    support: Mapping[str, Any],
    separator: Mapping[str, Any],
    cfg: AssemblyConfig,
) -> dict[str, Any] | None:
    sink_box = _bbox(sink)
    support_box = _bbox(support)
    separator_box = _bbox(separator)
    if not (
        support_box[0] + cfg.minimum_side_width < separator_box[0]
        and separator_box[2] + cfg.minimum_side_width < support_box[2]
    ):
        return None
    if _overlap_1d(
        support_box[1], support_box[3], separator_box[1], separator_box[3]
    ) <= 0.0:
        return None
    sink_center_x, sink_center_y = _center(sink_box)
    if not (
        support_box[0] <= sink_center_x <= support_box[2]
        and support_box[1] - 0.12 <= sink_center_y <= support_box[3] + 0.12
    ):
        return None
    separator_width = separator_box[2] - separator_box[0]
    if (
        sink_center_x < separator_box[0]
        and sink_box[2] <= separator_box[0] + 0.25 * separator_width
    ):
        side = "left"
    elif (
        sink_center_x > separator_box[2]
        and sink_box[0] >= separator_box[2] - 0.25 * separator_width
    ):
        side = "right"
    else:
        return None
    return {
        "pattern_mode": "broad_counter_crosses_separator",
        "side": side,
        "distance": min(
            abs(sink_center_x - separator_box[0]),
            abs(sink_center_x - separator_box[2]),
        ),
        "sink_box": sink_box,
        "support_box": support_box,
        "separator_box": separator_box,
        "sink": sink,
        "support": support,
        "separator": separator,
    }


def _match_partitioned_pattern(
    sink: Mapping[str, Any],
    support: Mapping[str, Any],
    separator: Mapping[str, Any],
    cfg: AssemblyConfig,
) -> dict[str, Any] | None:
    "Match an already-correct sink/range/counter left-to-right partition."

    sink_tokens = _tokens(sink)
    if not ({"counter", "assembly", "vanity", "cabinet"} & sink_tokens):
        return None
    sink_box = _bbox(sink)
    support_box = _bbox(support)
    separator_box = _bbox(separator)
    if (
        _overlap_1d(
            sink_box[1], sink_box[3], separator_box[1], separator_box[3]
        )
        <= 0.0
        or _overlap_1d(
            support_box[1],
            support_box[3],
            separator_box[1],
            separator_box[3],
        )
        <= 0.0
    ):
        return None
    sink_center_x, sink_center_y = _center(sink_box)
    support_center_x, support_center_y = _center(support_box)
    separator_center_x, separator_center_y = _center(separator_box)
    vertical_scale = max(
        sink_box[3] - sink_box[1],
        support_box[3] - support_box[1],
        separator_box[3] - separator_box[1],
        0.01,
    )
    if max(
        abs(sink_center_y - separator_center_y),
        abs(support_center_y - separator_center_y),
    ) > max(0.18, 1.25 * vertical_scale):
        return None
    if sink_center_x < separator_center_x < support_center_x:
        side = "left"
        sink_gap = max(0.0, separator_box[0] - sink_box[2])
        support_gap = max(0.0, support_box[0] - separator_box[2])
    elif support_center_x < separator_center_x < sink_center_x:
        side = "right"
        sink_gap = max(0.0, sink_box[0] - separator_box[2])
        support_gap = max(0.0, separator_box[0] - support_box[2])
    else:
        return None
    if max(sink_gap, support_gap) > 0.12:
        return None
    if min(
        sink_box[2] - sink_box[0],
        support_box[2] - support_box[0],
    ) < cfg.minimum_side_width:
        return None
    return {
        "pattern_mode": "already_partitioned_needs_roi",
        "side": side,
        "distance": sink_gap + support_gap,
        "sink_box": sink_box,
        "support_box": support_box,
        "separator_box": separator_box,
        "sink": sink,
        "support": support,
        "separator": separator,
    }


def _blocked_normalization(
    objects: Sequence[Mapping[str, Any]],
    cfg: AssemblyConfig,
    hard_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    copied = [copy.deepcopy(dict(item)) for item in objects]
    return {
        "objects": copied,
        "audit": {
            "schema_version": "scenemosaic_functional_assembly_normalization",
            "status": "blocked",
            "config": asdict(cfg),
            "raw_object_count": len(copied),
            "normalized_object_count": len(copied),
            "record_count": 0,
            "records": [],
            "targets": [],
            "id_remap": {},
            "removed_source_object_ids": [],
            "hard_errors": hard_errors,
            "formal_outputs_modified": False,
        },
    }


def normalize_pass_a_functional_assemblies(
    objects: Sequence[Mapping[str, Any]],
    config: AssemblyConfig | None = None,
) -> dict[str, Any]:
    """Normalize explicit sink/counter and complete-bed assemblies."""

    cfg = config or AssemblyConfig()
    if not 1 <= cfg.maximum_assemblies <= 4:
        raise ValueError("maximum_assemblies must be in [1, 4]")
    if not 0 <= cfg.maximum_sink_vanity_assemblies <= 4:
        raise ValueError("maximum_sink_vanity_assemblies must be in [0, 4]")
    if not 0 <= cfg.maximum_fixed_faucet_unions <= 4:
        raise ValueError("maximum_fixed_faucet_unions must be in [0, 4]")
    if not 0.0 < cfg.minimum_source_mask_coverage <= 1.0:
        raise ValueError("minimum_source_mask_coverage must be in (0, 1]")
    if not 0.0 <= cfg.maximum_protected_object_removal_fraction < 1.0:
        raise ValueError("maximum_protected_object_removal_fraction must be in [0, 1)")
    copied = [copy.deepcopy(dict(item)) for item in objects]
    ids = [str(item.get("object_id") or "") for item in copied]
    errors: list[dict[str, Any]] = []
    if not all(ids) or len(set(ids)) != len(ids):
        errors.append({"type": "invalid_or_duplicate_pass_a_object_id"})
    for item in copied:
        try:
            _bbox(item)
        except ValueError:
            errors.append(
                {
                    "type": "invalid_functional_assembly_bbox",
                    "object_id": str(item.get("object_id") or ""),
                }
            )
    if errors:
        return _blocked_normalization(objects, cfg, errors)

    sinks = [item for item in copied if _is_sink(item)]
    supports = [item for item in copied if _is_counter_support(item)]
    separators = [item for item in copied if _is_separator(item)]
    matches = []
    for sink in sinks:
        candidates = [
            evidence
            for support in supports
            for separator in separators
            if (
                evidence := _match_pattern(
                    sink, support, separator, cfg
                )
            )
            is not None
        ]
        candidates.sort(
            key=lambda item: (
                float(item["distance"]),
                str(item["support"].get("object_id") or ""),
                str(item["separator"].get("object_id") or ""),
            )
        )
        if len(candidates) > 1 and abs(
            float(candidates[0]["distance"])
            - float(candidates[1]["distance"])
        ) <= 0.02:
            errors.append(
                {
                    "type": "ambiguous_functional_assembly_pattern",
                    "sink_object_id": str(sink.get("object_id") or ""),
                    "candidate_count": len(candidates),
                }
            )
        elif candidates:
            matches.append(candidates[0])
    matched_sink_ids = {
        str(item["sink"].get("object_id") or "") for item in matches
    }
    for sink in sinks:
        if str(sink.get("object_id") or "") in matched_sink_ids:
            continue
        candidates = [
            evidence
            for support in supports
            for separator in separators
            if (
                evidence := _match_partitioned_pattern(
                    sink, support, separator, cfg
                )
            )
            is not None
        ]
        candidates.sort(
            key=lambda item: (
                float(item["distance"]),
                str(item["support"].get("object_id") or ""),
                str(item["separator"].get("object_id") or ""),
            )
        )
        if len(candidates) > 1 and abs(
            float(candidates[0]["distance"])
            - float(candidates[1]["distance"])
        ) <= 0.02:
            errors.append(
                {
                    "type": "ambiguous_partitioned_functional_assembly",
                    "sink_object_id": str(sink.get("object_id") or ""),
                    "candidate_count": len(candidates),
                }
            )
        elif candidates:
            matches.append(candidates[0])
    if len(matches) > cfg.maximum_assemblies:
        errors.append(
            {
                "type": "functional_assembly_capacity_overflow",
                "actual": len(matches),
                "maximum": cfg.maximum_assemblies,
            }
        )
    used_ids: set[str] = set()
    for match in matches:
        pair = {
            str(match["sink"]["object_id"]),
            str(match["support"]["object_id"]),
        }
        if used_ids & pair:
            errors.append(
                {
                    "type": "functional_assembly_object_reused",
                    "object_ids": sorted(used_ids & pair),
                }
            )
        used_ids |= pair
    if errors:
        return _blocked_normalization(objects, cfg, errors)

    by_id = {str(item["object_id"]): item for item in copied}
    removed_ids: set[str] = set()
    raw_records = []
    raw_targets = []
    for match in matches:
        sink = by_id[str(match["sink"]["object_id"])]
        support = by_id[str(match["support"]["object_id"])]
        separator = by_id[str(match["separator"]["object_id"])]
        sink_box = list(match["sink_box"])
        support_box = list(match["support_box"])
        separator_box = list(match["separator_box"])
        side = str(match["side"])
        pattern_mode = str(match.get("pattern_mode") or "")
        sep_w = separator_box[2] - separator_box[0]
        sep_h = separator_box[3] - separator_box[1]
        if side == "left":
            edge = separator_box[0] + 0.24 * sep_w
            if pattern_mode == "already_partitioned_needs_roi":
                sink_target = [
                    sink_box[0],
                    max(
                        0.0,
                        min(
                            sink_box[1],
                            separator_box[1] - 0.30 * sep_h,
                        ),
                    ),
                    max(sink_box[2], edge),
                    max(sink_box[3], separator_box[3]),
                ]
            else:
                sink_target = [
                    min(sink_box[0], support_box[0]),
                    min(sink_box[1], support_box[1]),
                    max(sink_box[2], edge),
                    max(sink_box[3], support_box[3]),
                ]
            counter_target = [
                separator_box[2] - 0.075 * sep_w,
                min(support_box[1], separator_box[1] - 0.10 * sep_h),
                support_box[2],
                min(support_box[3], separator_box[3] - 0.20 * sep_h),
            ]
            counter_relation = "right of gas range"
            sink_relation = "left"
            sink_limit = {"maximum_x_norm": round(edge, 6)}
            counter_limit = {
                "minimum_x_norm": round(separator_box[2] - 0.10 * sep_w, 6)
            }
        else:
            edge = separator_box[2] - 0.24 * sep_w
            if pattern_mode == "already_partitioned_needs_roi":
                sink_target = [
                    min(sink_box[0], edge),
                    max(
                        0.0,
                        min(
                            sink_box[1],
                            separator_box[1] - 0.30 * sep_h,
                        ),
                    ),
                    sink_box[2],
                    max(sink_box[3], separator_box[3]),
                ]
            else:
                sink_target = [
                    min(sink_box[0], edge),
                    min(sink_box[1], support_box[1]),
                    max(sink_box[2], support_box[2]),
                    max(sink_box[3], support_box[3]),
                ]
            counter_target = [
                support_box[0],
                min(support_box[1], separator_box[1] - 0.10 * sep_h),
                separator_box[0] + 0.075 * sep_w,
                min(support_box[3], separator_box[3] - 0.20 * sep_h),
            ]
            counter_relation = "left of gas range"
            sink_relation = "right"
            sink_limit = {"minimum_x_norm": round(edge, 6)}
            counter_limit = {
                "maximum_x_norm": round(separator_box[0] + 0.10 * sep_w, 6)
            }
        if (
            counter_target[2] - counter_target[0] < cfg.minimum_side_width
            or counter_target[3] - counter_target[1] < cfg.minimum_side_width
        ):
            errors.append(
                {
                    "type": "functional_assembly_partition_too_small",
                    "sink_object_id": str(sink["object_id"]),
                    "support_object_id": str(support["object_id"]),
                }
            )
            continue

        absorbed = []
        for component in copied:
            component_id = str(component.get("object_id") or "")
            if component_id in {
                str(sink["object_id"]),
                str(support["object_id"]),
                str(separator["object_id"]),
            } or not _is_absorbable_component(component):
                continue
            cx, cy = _center(_bbox(component))
            if (
                sink_target[0] <= cx <= sink_target[2]
                and sink_target[1] <= cy <= sink_target[3]
            ):
                absorbed.append(component_id)
                removed_ids.add(component_id)

        old_sink_id = str(sink["object_id"])
        old_support_id = str(support["object_id"])
        old_separator_id = str(separator["object_id"])
        sink.update(
            {
                "category": "kitchen_sink_counter_assembly",
                "sam3_prompt": (
                    "complete kitchen sink counter assembly including countertop "
                    "base cabinets dishwasher sink and faucet"
                ),
                "instance_name": f"{sink_relation} kitchen sink counter assembly",
                "description": (
                    "complete sink faucet countertop base cabinets and fitted dishwasher"
                ),
                "object_type": "furniture",
                "bbox_2d_norm_xyxy": _round_box(sink_target),
                "anchor_2d_norm_xy": [
                    round((sink_target[0] + sink_target[2]) / 2.0, 6),
                    round((sink_target[1] + sink_target[3]) / 2.0, 6),
                ],
            }
        )
        support.update(
            {
                "category": "kitchen_counter_cabinet",
                "sam3_prompt": (
                    "wooden kitchen base cabinet with white countertop "
                    f"{counter_relation}"
                ),
                "instance_name": f"counter cabinet {counter_relation}",
                "description": f"rear counter cabinet immediately {counter_relation}",
                "object_type": "furniture",
                "bbox_2d_norm_xyxy": _round_box(counter_target),
                "anchor_2d_norm_xy": [
                    round((counter_target[0] + counter_target[2]) / 2.0, 6),
                    round((counter_target[1] + counter_target[3]) / 2.0, 6),
                ],
            }
        )
        raw_targets.extend(
            [
                {
                    "old_source_object_id": old_sink_id,
                    "role": "sink_counter_assembly",
                    "prompt": sink["sam3_prompt"],
                    "bbox_norm_xyxy": list(sink["bbox_2d_norm_xyxy"]),
                    "hard_ownership_limit": sink_limit,
                },
                {
                    "old_source_object_id": old_support_id,
                    "role": "adjacent_counter_cabinet",
                    "prompt": support["sam3_prompt"],
                    "bbox_norm_xyxy": list(support["bbox_2d_norm_xyxy"]),
                    "hard_ownership_limit": counter_limit,
                },
            ]
        )
        raw_records.append(
            {
                "pattern_mode": pattern_mode,
                "sink_source_object_id_before": old_sink_id,
                "support_source_object_id_before": old_support_id,
                "separator_source_object_id_before": old_separator_id,
                "sink_side": side,
                "absorbed_component_ids_before": sorted(absorbed),
                "sink_target_bbox_norm_xyxy": list(sink["bbox_2d_norm_xyxy"]),
                "counter_target_bbox_norm_xyxy": list(
                    support["bbox_2d_norm_xyxy"]
                ),
            }
        )
    if errors:
        return _blocked_normalization(objects, cfg, errors)

    raw_bed_records = []
    raw_bed_targets = []
    for bed in copied:
        old_bed_id = str(bed.get("object_id") or "")
        if old_bed_id in removed_ids or not _is_bed(bed):
            continue
        original_prompt = str(bed.get("sam3_prompt") or "")
        original_description = str(bed.get("description") or "")
        bed.update(
            {
                "sam3_prompt": COMPLETE_BED_PROMPT,
                "description": (
                    "complete physical bed including headboard frame mattress "
                    "and duvet; independent pillows excluded"
                ),
                "object_type": "furniture",
            }
        )
        raw_bed_targets.append(
            {
                "old_source_object_id": old_bed_id,
                "role": "complete_bed_assembly",
                "prompt": COMPLETE_BED_PROMPT,
                "bbox_norm_xyxy": list(bed["bbox_2d_norm_xyxy"]),
                "hard_ownership_limit": {},
            }
        )
        raw_bed_records.append(
            {
                "pattern_mode": "explicit_complete_bed_assembly",
                "bed_source_object_id_before": old_bed_id,
                "original_prompt": original_prompt,
                "original_description": original_description,
                "normalized_prompt": COMPLETE_BED_PROMPT,
            }
        )

    raw_vanity_records = []
    raw_vanity_targets = []
    vanities = [
        item
        for item in copied
        if str(item.get("object_id") or "") not in removed_ids
        and _is_sink_vanity(item)
    ]
    if len(vanities) > cfg.maximum_sink_vanity_assemblies:
        return _blocked_normalization(
            objects,
            cfg,
            [
                {
                    "type": "sink_vanity_assembly_capacity_overflow",
                    "actual": len(vanities),
                    "maximum": cfg.maximum_sink_vanity_assemblies,
                }
            ],
        )
    for vanity in vanities:
        old_vanity_id = str(vanity.get("object_id") or "")
        original_prompt = str(vanity.get("sam3_prompt") or "")
        original_description = str(vanity.get("description") or "")
        prompt = (
            COMPLETE_DOUBLE_SINK_VANITY_PROMPT
            if "double" in _category_tokens(vanity)
            else COMPLETE_SINK_VANITY_PROMPT
        )
        target_box = _expand_box_upward(_bbox(vanity))
        vanity.update(
            {
                "sam3_prompt": prompt,
                "description": (
                    "complete sink vanity including cabinet countertop basin "
                    "and fixed faucet"
                ),
                "object_type": "furniture",
                "bbox_2d_norm_xyxy": target_box,
                "anchor_2d_norm_xy": [
                    round((target_box[0] + target_box[2]) / 2.0, 6),
                    round((target_box[1] + target_box[3]) / 2.0, 6),
                ],
            }
        )
        raw_vanity_targets.append(
            {
                "old_source_object_id": old_vanity_id,
                "role": "complete_sink_vanity_assembly",
                "prompt": prompt,
                "bbox_norm_xyxy": list(target_box),
                "hard_ownership_limit": {},
                "require_source_mask_coverage": cfg.minimum_source_mask_coverage,
                "protect_independent_objects": True,
            }
        )
        raw_vanity_records.append(
            {
                "pattern_mode": "explicit_complete_sink_vanity_assembly",
                "vanity_source_object_id_before": old_vanity_id,
                "original_prompt": original_prompt,
                "original_description": original_description,
                "normalized_prompt": prompt,
                "target_bbox_norm_xyxy": list(target_box),
            }
        )

    raw_botanical_records = []
    plants = [
        item
        for item in copied
        if str(item.get("object_id") or "") not in removed_ids and _is_plant(item)
    ]
    vessels = [
        item
        for item in copied
        if (
            str(item.get("object_id") or "") not in removed_ids
            and _is_vase_like(item)
        )
    ]
    botanical_candidates = [
        match
        for plant in plants
        for vessel in vessels
        if (
            match := _match_potted_plant_pattern(plant, vessel)
        ) is not None
    ]
    candidates_by_plant: dict[str, list[dict[str, Any]]] = {}
    candidates_by_vessel: dict[str, list[dict[str, Any]]] = {}
    for match in botanical_candidates:
        plant_id = str(match["plant"].get("object_id") or "")
        vessel_id = str(match["vessel"].get("object_id") or "")
        candidates_by_plant.setdefault(plant_id, []).append(match)
        candidates_by_vessel.setdefault(vessel_id, []).append(match)
    for match in sorted(
        botanical_candidates,
        key=lambda item: (
            str(item["plant"].get("object_id") or ""),
            str(item["vessel"].get("object_id") or ""),
        ),
    ):
        plant = match["plant"]
        vessel = match["vessel"]
        plant_id = str(plant.get("object_id") or "")
        vessel_id = str(vessel.get("object_id") or "")
        if (
            len(candidates_by_plant[plant_id]) != 1
            or len(candidates_by_vessel[vessel_id]) != 1
        ):
            continue
        target_box = list(match["target_bbox_norm_xyxy"])
        plant.update(
            {
                "category": "potted_plant",
                "sam3_prompt": COMPLETE_POTTED_PLANT_PROMPT,
                "instance_name": "potted plant",
                "description": "complete potted plant including foliage and pot",
                "object_type": "plant",
                "bbox_2d_norm_xyxy": target_box,
                "anchor_2d_norm_xy": [
                    round((target_box[0] + target_box[2]) / 2.0, 6),
                    round((target_box[1] + target_box[3]) / 2.0, 6),
                ],
            }
        )
        removed_ids.add(vessel_id)
        raw_botanical_records.append(
            {
                "pattern_mode": "plant_vessel_potted_plant_assembly",
                "plant_source_object_id_before": plant_id,
                "vessel_source_object_id_before": vessel_id,
                "normalized_prompt": COMPLETE_POTTED_PLANT_PROMPT,
                "target_bbox_norm_xyxy": target_box,
                "vessel_horizontal_coverage": match[
                    "vessel_horizontal_coverage"
                ],
                "vessel_vertical_coverage": match["vessel_vertical_coverage"],
                "vessel_to_plant_bbox_area_ratio": match[
                    "vessel_to_plant_bbox_area_ratio"
                ],
            }
        )

    retained = [
        item for item in copied if str(item["object_id"]) not in removed_ids
    ]
    id_remap = {}
    for index, item in enumerate(retained, start=1):
        old_id = str(item["object_id"])
        new_id = f"A_{index:03d}"
        id_remap[old_id] = new_id
        item["object_id"] = new_id
    records = [
        {
            **raw,
            "sink_source_object_id_after": id_remap[
                raw["sink_source_object_id_before"]
            ],
            "support_source_object_id_after": id_remap[
                raw["support_source_object_id_before"]
            ],
            "separator_source_object_id_after": id_remap[
                raw["separator_source_object_id_before"]
            ],
        }
        for raw in raw_records
    ]
    records.extend(
        {
            **raw,
            "bed_source_object_id_after": id_remap[
                raw["bed_source_object_id_before"]
            ],
        }
        for raw in raw_bed_records
    )
    records.extend(
        {
            **raw,
            "vanity_source_object_id_after": id_remap[
                raw["vanity_source_object_id_before"]
            ],
        }
        for raw in raw_vanity_records
    )
    records.extend(
        {
            **raw,
            "potted_plant_source_object_id_after": id_remap[
                raw["plant_source_object_id_before"]
            ],
        }
        for raw in raw_botanical_records
    )
    targets = [
        {
            **target,
            "source_object_id": id_remap[target["old_source_object_id"]],
        }
        for target in raw_targets
    ]
    targets.extend(
        {
            **target,
            "source_object_id": id_remap[target["old_source_object_id"]],
        }
        for target in raw_bed_targets
    )
    targets.extend(
        {
            **target,
            "source_object_id": id_remap[target["old_source_object_id"]],
        }
        for target in raw_vanity_targets
    )
    audit = {
        "schema_version": "scenemosaic_functional_assembly_normalization",
        "status": "pass" if records else "noop",
        "config": asdict(cfg),
        "raw_object_count": len(objects),
        "normalized_object_count": len(retained),
        "record_count": len(records),
        "records": records,
        "targets": targets,
        "id_remap": id_remap,
        "removed_source_object_ids": sorted(removed_ids),
        "hard_errors": [],
        "formal_outputs_modified": False,
    }
    return {"objects": retained, "audit": audit}


def _load_mask(path_value: Any, shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != shape or not mask.any():
        raise ValueError(f"invalid mask: {path}")
    return mask


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_metrics(
    target: Sequence[int], candidate: Sequence[int]
) -> dict[str, float]:
    ix = max(0, min(target[2], candidate[2]) - max(target[0], candidate[0]))
    iy = max(0, min(target[3], candidate[3]) - max(target[1], candidate[1]))
    intersection = ix * iy
    target_area = max(1, (target[2] - target[0]) * (target[3] - target[1]))
    candidate_area = max(
        1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    )
    target_center = _center(target)
    candidate_center = _center(candidate)
    scale = max(
        math.hypot(target[2] - target[0], target[3] - target[1]),
        math.hypot(candidate[2] - candidate[0], candidate[3] - candidate[1]),
        1.0,
    )
    return {
        "target_bbox_coverage": intersection / target_area,
        "candidate_bbox_coverage": intersection / candidate_area,
        "center_distance_scale": math.dist(target_center, candidate_center) / scale,
    }


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return int(np.logical_and(left, right).sum()) / union if union else 0.0


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_plant_container_assembly(obj: Mapping[str, Any]) -> bool:
    """Return whether one declaration explicitly includes foliage and a vessel."""

    tokens = _tokens(obj)
    has_container = bool(
        tokens & {"pot", "potted", "planter", "vase", "arrangement"}
    )
    has_botanical_content = bool(
        tokens
        & {
            "plant",
            "greenery",
            "foliage",
            "branch",
            "branches",
            "flower",
            "flowers",
            "floral",
        }
    )
    return has_container and has_botanical_content


def _absolute_bbox_from_normalized(
    box: Sequence[float], width: int, height: int
) -> list[int]:
    return [
        max(0, min(width - 1, int(round(float(box[0]) * width)))),
        max(0, min(height - 1, int(round(float(box[1]) * height)))),
        max(1, min(width, int(round(float(box[2]) * width)))),
        max(1, min(height, int(round(float(box[3]) * height)))),
    ]


def _mask_fraction_within_bbox(mask: np.ndarray, box: Sequence[int]) -> float:
    """Return the fraction of one mask whose pixels lie inside box."""

    x1, y1, x2, y2 = [int(value) for value in box]
    return float(mask[y1:y2, x1:x2].sum()) / float(max(1, mask.sum()))


def complete_plant_container_masks(
    *,
    image_path: str | Path,
    registry: Mapping[str, Any],
    output_dir: str | Path,
    config: PlantContainerMaskCompletionConfig | None = None,
) -> dict[str, Any]:
    """Union a proven foliage-plus-vessel candidate into one plant assembly.

    This is intentionally bounded to a pre-existing plant-container declaration.
    It never creates an object, never queries SAM3 or a VLM, and only uses an
    unassigned high-confidence candidate that mostly lies in the declaration's
    bbox while contributing a clear lower extension.  That makes the completion
    suitable for a vase, pot, or planter that SAM3 separated from its foliage.
    """

    cfg = config or PlantContainerMaskCompletionConfig()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"plant-container completion output is not empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    copied = copy.deepcopy(dict(registry))
    objects = copied.get("objects")
    if not isinstance(objects, list):
        raise ValueError("registry must contain objects")

    resolved_image = Path(image_path).expanduser().resolve()
    image = np.array(Image.open(resolved_image).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    shape = (height, width)

    successes: dict[str, tuple[Mapping[str, Any], np.ndarray]] = {}
    errors: list[dict[str, Any]] = []
    for obj in objects:
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            continue
        object_id = str(obj.get("object_id") or "")
        if not object_id:
            errors.append({"type": "missing_successful_object_id"})
            continue
        try:
            successes[object_id] = (obj, _load_mask(sam3.get("mask_path"), shape))
        except Exception as exc:
            errors.append(
                {
                    "type": "invalid_successful_plant_container_mask",
                    "object_id": object_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    candidates = list(copied.get("unassigned_sam3_candidates") or [])
    records: list[dict[str, Any]] = []
    accepted_count = 0
    for object_id, (obj, source_mask) in successes.items():
        if not _is_plant_container_assembly(obj):
            continue
        try:
            target_box = _absolute_bbox_from_normalized(_bbox(obj), width, height)
        except ValueError as exc:
            errors.append(
                {
                    "type": "invalid_plant_container_bbox",
                    "object_id": object_id,
                    "error": str(exc),
                }
            )
            continue
        source_box = _mask_bbox(source_mask)
        eligible: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for raw in candidates:
            candidate_id = str(raw.get("candidate_id") or "")
            prompt = str(
                raw.get("prompt")
                or raw.get("sam3_prompt")
                or raw.get("category_hint")
                or ""
            )
            candidate_tokens = set(re.findall(r"[a-z0-9]+", prompt.lower()))
            if not candidate_tokens & {
                "plant",
                "potted",
                "pot",
                "planter",
                "vase",
                "greenery",
                "foliage",
                "branch",
                "branches",
                "flower",
                "flowers",
                "floral",
            }:
                continue
            try:
                score = float(raw.get("score"))
                candidate_mask = _load_mask(raw.get("mask_path"), shape)
            except (TypeError, ValueError, OSError) as exc:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": f"invalid_candidate:{type(exc).__name__}",
                    }
                )
                continue
            if score < cfg.minimum_candidate_score:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "score_below_threshold",
                        "score": round(score, 6),
                    }
                )
                continue
            intersection = int(np.logical_and(source_mask, candidate_mask).sum())
            candidate_area = int(candidate_mask.sum())
            candidate_overlap = intersection / max(1, candidate_area)
            if not (
                cfg.minimum_candidate_overlap
                <= candidate_overlap
                <= cfg.maximum_candidate_overlap
            ):
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "overlap_outside_completion_band",
                        "candidate_overlap": round(candidate_overlap, 6),
                    }
                )
                continue
            candidate_bbox_fraction = _mask_fraction_within_bbox(
                candidate_mask, target_box
            )
            if candidate_bbox_fraction < cfg.minimum_candidate_bbox_fraction:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "candidate_outside_declared_bbox",
                        "candidate_bbox_fraction": round(candidate_bbox_fraction, 6),
                    }
                )
                continue
            new_pixels = np.logical_and(candidate_mask, ~source_mask)
            new_pixel_count = int(new_pixels.sum())
            if new_pixel_count < cfg.minimum_new_pixels:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "insufficient_new_pixels",
                        "new_pixel_count": new_pixel_count,
                    }
                )
                continue
            new_pixel_bbox_fraction = (
                float(new_pixels[target_box[1]:target_box[3], target_box[0]:target_box[2]].sum())
                / float(max(1, new_pixel_count))
            )
            if new_pixel_bbox_fraction < cfg.minimum_new_pixel_bbox_fraction:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "new_pixels_outside_declared_bbox",
                        "new_pixel_bbox_fraction": round(new_pixel_bbox_fraction, 6),
                    }
                )
                continue
            candidate_box = _mask_bbox(candidate_mask)
            if candidate_box[3] < (
                source_box[3] + cfg.minimum_downward_extension_pixels
            ):
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "no_clear_lower_container_extension",
                        "source_bottom": source_box[3],
                        "candidate_bottom": candidate_box[3],
                    }
                )
                continue
            conflict = None
            for other_id, (_, other_mask) in successes.items():
                if other_id == object_id:
                    continue
                overlap = int(np.logical_and(new_pixels, other_mask).sum()) / max(
                    1, new_pixel_count
                )
                if overlap > cfg.maximum_other_object_overlap:
                    conflict = (other_id, overlap)
                    break
            if conflict is not None:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "new_pixels_overlap_other_object",
                        "other_object_id": conflict[0],
                        "overlap": round(conflict[1], 6),
                    }
                )
                continue
            eligible.append(
                {
                    "candidate_id": candidate_id,
                    "prompt": prompt,
                    "score": score,
                    "mask": candidate_mask,
                    "candidate_overlap": candidate_overlap,
                    "candidate_bbox_fraction": candidate_bbox_fraction,
                    "new_pixel_count": new_pixel_count,
                    "new_pixel_bbox_fraction": new_pixel_bbox_fraction,
                }
            )

        eligible.sort(
            key=lambda item: (
                -float(item["score"]),
                -int(item["new_pixel_count"]),
                str(item["candidate_id"]),
            )
        )
        if len(eligible) > 1 and (
            float(eligible[0]["score"]) - float(eligible[1]["score"])
            <= cfg.ambiguity_score_gap
        ):
            records.append(
                {
                    "object_id": object_id,
                    "status": "ambiguous",
                    "eligible_candidate_ids": [
                        str(item["candidate_id"]) for item in eligible
                    ],
                    "rejected_candidates": rejected,
                }
            )
            continue
        if not eligible:
            records.append(
                {
                    "object_id": object_id,
                    "status": "noop",
                    "rejected_candidates": rejected,
                }
            )
            continue

        winner = eligible[0]
        union_mask = np.logical_or(source_mask, winner["mask"])
        selected_dir = output / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        mask_path = selected_dir / f"{object_id.lower()}_mask.png"
        overlay_path = selected_dir / f"{object_id.lower()}_overlay.png"
        Image.fromarray(union_mask.astype(np.uint8) * 255).save(mask_path)
        canvas = image.astype(np.float32).copy()
        canvas[union_mask] = (
            0.55 * canvas[union_mask]
            + 0.45 * np.array((255, 0, 0), dtype=np.float32)
        )
        Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(
            overlay_path
        )
        union_box = _mask_bbox(union_mask)
        sam3 = dict(obj.get("sam3") or {})
        sam3.update(
            {
                "area_pixels": int(union_mask.sum()),
                "mask_bbox_abs_xyxy": union_box,
                "vlm_bbox_iou": _bbox_metrics(target_box, union_box)[
                    "candidate_bbox_coverage"
                ],
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "selection_method": "plant_container_candidate_union",
                "plant_container_mask_completion": {
                    "candidate_id": str(winner["candidate_id"]),
                    "candidate_prompt": str(winner["prompt"]),
                    "candidate_score": round(float(winner["score"]), 6),
                    "candidate_overlap": round(
                        float(winner["candidate_overlap"]), 6
                    ),
                    "candidate_bbox_fraction": round(
                        float(winner["candidate_bbox_fraction"]), 6
                    ),
                    "new_pixel_count": int(winner["new_pixel_count"]),
                    "new_pixel_bbox_fraction": round(
                        float(winner["new_pixel_bbox_fraction"]), 6
                    ),
                },
            }
        )
        obj["sam3"] = sam3
        accepted_count += 1
        records.append(
            {
                "object_id": object_id,
                "status": "merged",
                "selected_candidate_id": str(winner["candidate_id"]),
                "selected_mask_path": str(mask_path),
                "selected_overlay_path": str(overlay_path),
                "selected_area_pixels": int(union_mask.sum()),
                "selected_mask_bbox_abs_xyxy": union_box,
                "candidate_score": round(float(winner["score"]), 6),
                "candidate_overlap": round(
                    float(winner["candidate_overlap"]), 6
                ),
                "candidate_bbox_fraction": round(
                    float(winner["candidate_bbox_fraction"]), 6
                ),
                "new_pixel_count": int(winner["new_pixel_count"]),
                "new_pixel_bbox_fraction": round(
                    float(winner["new_pixel_bbox_fraction"]), 6
                ),
                "rejected_candidates": rejected,
            }
        )

    audit = {
        "schema_version": "scenemosaic_plant_container_mask_completion",
        "status": "blocked" if errors else ("pass" if accepted_count else "noop"),
        "config": asdict(cfg),
        "candidate_count": len(candidates),
        "accepted_count": accepted_count,
        "records": records,
        "additional_model_load_count": 0,
        "additional_vlm_api_calls": 0,
        "formal_outputs_modified": False,
        "hard_errors": errors,
    }
    audit_path = output / "plant_container_mask_completion_audit.json"
    _atomic_json(audit_path, audit)
    return {"registry": copied, "audit": audit, "audit_path": str(audit_path)}


def run_functional_assembly_roi_refinement(
    *,
    session: Any,
    infer_text_candidates_fn: Any,
    image_path: str | Path,
    registry: Mapping[str, Any],
    normalization_audit: Mapping[str, Any],
    output_dir: str | Path,
    config: AssemblyConfig | None = None,
) -> dict[str, Any]:
    """Refine normalized targets with bounded positive-box probes."""

    cfg = config or AssemblyConfig()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"functional assembly output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copied = copy.deepcopy(dict(registry))
    objects = copied.get("objects")
    if not isinstance(objects, list):
        raise ValueError("registry must contain objects")
    targets = list(normalization_audit.get("targets") or [])
    errors: list[dict[str, Any]] = []
    if len(targets) > cfg.maximum_targets:
        errors.append(
            {
                "type": "functional_assembly_target_capacity_overflow",
                "actual": len(targets),
                "maximum": cfg.maximum_targets,
            }
        )
        targets = targets[: cfg.maximum_targets]
    resolved_image = Path(image_path).expanduser().resolve()
    image = np.array(Image.open(resolved_image).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    shape = (height, width)
    by_id = {str(item.get("object_id") or ""): item for item in objects}
    records = []
    inference_before = int(getattr(session, "inference_run_count", 0))
    pillow_masks = []
    independent_bed_protection_masks = []
    if any(
        str(target.get("role") or "") == "complete_bed_assembly"
        for target in targets
    ):
        for independent in objects:
            sam3 = independent.get("sam3") or {}
            if sam3.get("status") != "success":
                continue
            independent_id = str(independent.get("object_id") or "")
            if _is_absorbable_bed_component(independent):
                continue
            try:
                independent_mask = _load_mask(sam3.get("mask_path"), shape)
            except Exception as exc:
                errors.append(
                    {
                        "type": (
                            "invalid_independent_pillow_mask"
                            if _is_pillow(independent)
                            else "invalid_independent_bed_protection_mask"
                        ),
                        "object_id": independent_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if _is_pillow(independent):
                pillow_masks.append((independent_id, independent_mask))
            else:
                independent_bed_protection_masks.append(
                    (
                        independent_id,
                        str(independent.get("category") or ""),
                        independent_mask,
                    )
                )

    for index, target in enumerate(targets, start=1):
        object_id = str(target.get("source_object_id") or "")
        obj = by_id.get(object_id)
        if obj is None:
            errors.append(
                {"type": "missing_functional_assembly_target", "object_id": object_id}
            )
            continue
        norm_box = [float(value) for value in target["bbox_norm_xyxy"]]
        target_box = [
            max(0, min(width - 1, int(round(norm_box[0] * width)))),
            max(0, min(height - 1, int(round(norm_box[1] * height)))),
            max(1, min(width, int(round(norm_box[2] * width)))),
            max(1, min(height, int(round(norm_box[3] * height)))),
        ]
        span_x = target_box[2] - target_box[0]
        span_y = target_box[3] - target_box[1]
        pad_x = max(16, int(round(span_x * 0.10)))
        pad_y = max(16, int(round(span_y * 0.10)))
        positive_box = [
            max(0, target_box[0] - pad_x),
            max(0, target_box[1] - pad_y),
            min(width, target_box[2] + pad_x),
            min(height, target_box[3] + pad_y),
        ]
        role = str(target.get("role") or "")
        source_mask = None
        required_source_coverage = float(
            target.get("require_source_mask_coverage") or 0.0
        )
        if required_source_coverage:
            try:
                source_mask = _load_mask(
                    (obj.get("sam3") or {}).get("mask_path"), shape
                )
            except Exception as exc:
                errors.append(
                    {
                        "type": "functional_assembly_missing_source_mask",
                        "object_id": object_id,
                        "role": role,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                records.append(
                    {
                        "object_id": object_id,
                        "role": role,
                        "status": "blocked",
                        "reason": "missing_source_mask",
                    }
                )
                continue
        protected_masks = []
        if bool(target.get("protect_independent_objects")):
            for independent in objects:
                independent_id = str(independent.get("object_id") or "")
                if independent_id == object_id or _is_fixed_faucet(independent):
                    continue
                independent_sam3 = independent.get("sam3") or {}
                if independent_sam3.get("status") != "success":
                    continue
                try:
                    independent_mask = _load_mask(
                        independent_sam3.get("mask_path"), shape
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "type": "invalid_protected_independent_mask",
                            "object_id": independent_id,
                            "protected_for_object_id": object_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if independent_mask[
                    target_box[1] : target_box[3],
                    target_box[0] : target_box[2],
                ].any():
                    protected_masks.append(
                        (
                            independent_id,
                            str(independent.get("category") or ""),
                            independent_mask,
                        )
                    )
        raw_dir = (
            output / "raw_candidates" / f"target_{index:02d}_{object_id.lower()}"
        )
        result = infer_text_candidates_fn(
            session,
            resolved_image,
            str(target["prompt"]),
            raw_dir,
            score_threshold=cfg.score_threshold,
            positive_bbox_abs_xyxy=positive_box,
            maximum_candidates=cfg.maximum_candidates_per_target,
            overwrite=False,
        )
        evaluations = []
        for candidate in (result.get("registry") or {}).get("candidates") or []:
            try:
                mask = _load_mask(candidate.get("mask_path"), shape)
                ownership = dict(target.get("hard_ownership_limit") or {})
                if "maximum_x_norm" in ownership:
                    cutoff = int(round(float(ownership["maximum_x_norm"]) * width))
                    mask[:, cutoff:] = False
                if "minimum_x_norm" in ownership:
                    cutoff = int(round(float(ownership["minimum_x_norm"]) * width))
                    mask[:, :cutoff] = False
                candidate_box = _mask_bbox(mask)
            except Exception as exc:
                evaluations.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "eligible": False,
                        "reason": (
                            f"invalid_candidate: {type(exc).__name__}: {exc}"
                        ),
                    }
                )
                continue
            metrics = _bbox_metrics(target_box, candidate_box)
            score = float(candidate.get("score") or 0.0)
            source_mask_coverage = (
                int(np.logical_and(mask, source_mask).sum())
                / max(int(source_mask.sum()), 1)
                if source_mask is not None
                else None
            )
            eligible = (
                score >= cfg.minimum_selected_score
                and metrics["target_bbox_coverage"]
                >= cfg.minimum_target_bbox_coverage
                and metrics["candidate_bbox_coverage"]
                >= cfg.minimum_candidate_bbox_coverage
                and metrics["center_distance_scale"]
                <= cfg.maximum_center_distance_scale
                and (
                    source_mask_coverage is None
                    or source_mask_coverage >= required_source_coverage
                )
            )
            quality = (
                score
                + 0.50 * metrics["target_bbox_coverage"]
                + 0.25 * metrics["candidate_bbox_coverage"]
                + (0.20 * source_mask_coverage if source_mask_coverage is not None else 0.0)
                - 0.20 * metrics["center_distance_scale"]
            )
            evaluations.append(
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "source_mask_path": str(candidate.get("mask_path") or ""),
                    "score": score,
                    "mask": mask,
                    "mask_bbox_abs_xyxy": candidate_box,
                    "target_bbox_coverage": round(
                        metrics["target_bbox_coverage"], 6
                    ),
                    "candidate_bbox_coverage": round(
                        metrics["candidate_bbox_coverage"], 6
                    ),
                    "center_distance_scale": round(
                        metrics["center_distance_scale"], 6
                    ),
                    "source_mask_coverage": (
                        round(source_mask_coverage, 6)
                        if source_mask_coverage is not None
                        else None
                    ),
                    "quality": quality,
                    "eligible": eligible,
                    "reason": None if eligible else "bounded_geometry_or_score_gate",
                }
            )
        eligible = sorted(
            [item for item in evaluations if item.get("eligible")],
            key=lambda item: (
                -float(item["quality"]),
                str(item["candidate_id"]),
            ),
        )
        public_evaluations = [
            {key: value for key, value in item.items() if key != "mask"}
            for item in evaluations
        ]
        if not eligible:
            errors.append(
                {
                    "type": "functional_assembly_no_eligible_roi_candidate",
                    "object_id": object_id,
                }
            )
            records.append(
                {
                    "object_id": object_id,
                    "status": "blocked",
                    "target_bbox_abs_xyxy": target_box,
                    "positive_bbox_abs_xyxy": positive_box,
                    "evaluations": public_evaluations,
                }
            )
            continue
        winner = eligible[0]
        if len(eligible) > 1:
            runner_up = eligible[1]
            gap = float(winner["quality"]) - float(runner_up["quality"])
            overlap = _mask_iou(winner["mask"], runner_up["mask"])
            if (
                gap <= cfg.ambiguity_quality_gap
                and overlap < cfg.ambiguity_mask_iou
            ):
                errors.append(
                    {
                        "type": "functional_assembly_ambiguous_roi_candidates",
                        "object_id": object_id,
                        "candidate_ids": [
                            winner["candidate_id"],
                            runner_up["candidate_id"],
                        ],
                        "quality_gap": round(gap, 6),
                        "mask_iou": round(overlap, 6),
                    }
                )
                continue

        pillow_exclusions = []
        independent_object_exclusions = []
        if str(target.get("role") or "") == "complete_bed_assembly":
            area_before_exclusion = int(winner["mask"].sum())
            for pillow_id, pillow_mask in pillow_masks:
                overlap_pixels = int(
                    np.logical_and(winner["mask"], pillow_mask).sum()
                )
                if overlap_pixels <= 0:
                    continue
                winner["mask"] = np.logical_and(
                    winner["mask"], np.logical_not(pillow_mask)
                )
                pillow_exclusions.append(
                    {
                        "pillow_object_id": pillow_id,
                        "overlap_pixels_removed": overlap_pixels,
                    }
                )
            area_after_pillow_exclusion = int(winner["mask"].sum())
            for (
                independent_id,
                independent_category,
                independent_mask,
            ) in independent_bed_protection_masks:
                if independent_id == object_id:
                    continue
                overlap_pixels = int(
                    np.logical_and(winner["mask"], independent_mask).sum()
                )
                if overlap_pixels <= 0:
                    continue
                winner["mask"] = np.logical_and(
                    winner["mask"], np.logical_not(independent_mask)
                )
                independent_object_exclusions.append(
                    {
                        "object_id": independent_id,
                        "category": independent_category,
                        "overlap_pixels_removed": overlap_pixels,
                    }
                )
            area_after_exclusion = int(winner["mask"].sum())
            pillow_removed_fraction = (
                (area_before_exclusion - area_after_pillow_exclusion)
                / max(area_before_exclusion, 1)
            )
            independent_removed_fraction = (
                (area_after_pillow_exclusion - area_after_exclusion)
                / max(area_before_exclusion, 1)
            )
            removed_fraction = (
                (area_before_exclusion - area_after_exclusion)
                / max(area_before_exclusion, 1)
            )
            if not winner["mask"].any() or removed_fraction > 0.40:
                errors.append(
                    {
                        "type": "complete_bed_protected_object_exclusion_invalid",
                        "object_id": object_id,
                        "area_before_exclusion": area_before_exclusion,
                        "area_after_pillow_exclusion": (
                            area_after_pillow_exclusion
                        ),
                        "area_after_exclusion": area_after_exclusion,
                        "removed_fraction": round(removed_fraction, 6),
                    }
                )
                records.append(
                    {
                        "object_id": object_id,
                        "role": target.get("role"),
                        "status": "blocked",
                        "reason": "independent_object_subtraction_gate",
                        "pillow_exclusions": pillow_exclusions,
                        "independent_object_exclusions": (
                            independent_object_exclusions
                        ),
                    }
                )
                continue
            winner["mask_bbox_abs_xyxy"] = _mask_bbox(winner["mask"])
        elif role == "complete_sink_vanity_assembly":
            area_before_exclusion = int(winner["mask"].sum())
            pillow_exclusions = []
            independent_object_exclusions = []
            for independent_id, independent_category, independent_mask in protected_masks:
                overlap_pixels = int(
                    np.logical_and(winner["mask"], independent_mask).sum()
                )
                if overlap_pixels <= 0:
                    continue
                winner["mask"] = np.logical_and(
                    winner["mask"], np.logical_not(independent_mask)
                )
                independent_object_exclusions.append(
                    {
                        "object_id": independent_id,
                        "category": independent_category,
                        "overlap_pixels_removed": overlap_pixels,
                    }
                )
            area_after_pillow_exclusion = area_before_exclusion
            area_after_exclusion = int(winner["mask"].sum())
            pillow_removed_fraction = 0.0
            independent_removed_fraction = (
                (area_before_exclusion - area_after_exclusion)
                / max(area_before_exclusion, 1)
            )
            removed_fraction = independent_removed_fraction
            final_source_coverage = (
                int(np.logical_and(winner["mask"], source_mask).sum())
                / max(int(source_mask.sum()), 1)
                if source_mask is not None
                else None
            )
            final_metrics = _bbox_metrics(target_box, _mask_bbox(winner["mask"])) if winner["mask"].any() else None
            if (
                not winner["mask"].any()
                or removed_fraction > cfg.maximum_protected_object_removal_fraction
                or (
                    final_source_coverage is not None
                    and final_source_coverage < required_source_coverage
                )
                or final_metrics is None
                or final_metrics["target_bbox_coverage"]
                < cfg.minimum_target_bbox_coverage
            ):
                errors.append(
                    {
                        "type": "complete_sink_vanity_protected_object_exclusion_invalid",
                        "object_id": object_id,
                        "area_before_exclusion": area_before_exclusion,
                        "area_after_exclusion": area_after_exclusion,
                        "removed_fraction": round(removed_fraction, 6),
                        "source_mask_coverage": (
                            round(final_source_coverage, 6)
                            if final_source_coverage is not None
                            else None
                        ),
                    }
                )
                records.append(
                    {
                        "object_id": object_id,
                        "role": role,
                        "status": "blocked",
                        "reason": "protected_object_or_source_coverage_gate",
                        "independent_object_exclusions": independent_object_exclusions,
                    }
                )
                continue
            winner["mask_bbox_abs_xyxy"] = _mask_bbox(winner["mask"])
        else:
            area_before_exclusion = int(winner["mask"].sum())
            area_after_pillow_exclusion = area_before_exclusion
            area_after_exclusion = area_before_exclusion
            pillow_removed_fraction = 0.0
            independent_removed_fraction = 0.0
            removed_fraction = 0.0

        selected_dir = output / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        mask_path = selected_dir / f"{object_id.lower()}_mask.png"
        overlay_path = selected_dir / f"{object_id.lower()}_overlay.png"
        Image.fromarray(winner["mask"].astype(np.uint8) * 255).save(mask_path)
        canvas = image.astype(np.float32).copy()
        canvas[winner["mask"]] = (
            0.55 * canvas[winner["mask"]]
            + 0.45 * np.array((255, 0, 0), dtype=np.float32)
        )
        Image.fromarray(
            np.clip(canvas, 0, 255).astype(np.uint8)
        ).save(overlay_path)
        candidate_box = list(winner["mask_bbox_abs_xyxy"])
        obj["sam3"] = {
            "status": "success",
            "warning": None,
            "prompt_used": str(target["prompt"]),
            "score": float(winner["score"]),
            "area_pixels": int(winner["mask"].sum()),
            "vlm_bbox_abs_xyxy": target_box,
            "mask_bbox_abs_xyxy": candidate_box,
            "vlm_bbox_iou": _bbox_metrics(target_box, candidate_box)[
                "candidate_bbox_coverage"
            ],
            "merged_candidate_indices": [],
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "selection_method": "functional_assembly_roi_refinement",
            "merge_review": {
                "required": False,
                "ambiguous_candidate_indices": [],
                "ambiguous_candidate_ids": [],
                "protected_shared_candidate_indices": [],
                "protected_shared_candidate_ids": [],
                "resolution_stage": None,
            },
            "functional_assembly_roi_refinement": {
                "candidate_id": winner["candidate_id"],
                "positive_bbox_abs_xyxy": positive_box,
                "target_bbox_coverage": winner["target_bbox_coverage"],
                "candidate_bbox_coverage": winner[
                    "candidate_bbox_coverage"
                ],
                "center_distance_scale": winner["center_distance_scale"],
                "hard_ownership_limit": target.get("hard_ownership_limit"),
                "independent_pillow_exclusion": {
                    "applied": bool(pillow_exclusions),
                    "area_before_exclusion": area_before_exclusion,
                    "area_after_exclusion": area_after_pillow_exclusion,
                    "removed_fraction": round(pillow_removed_fraction, 6),
                    "records": pillow_exclusions,
                },
                "independent_object_exclusion": {
                    "applied": bool(independent_object_exclusions),
                    "area_before_exclusion": area_after_pillow_exclusion,
                    "area_after_exclusion": area_after_exclusion,
                    "removed_fraction_of_raw_candidate": round(
                        independent_removed_fraction, 6
                    ),
                    "total_removed_fraction": round(removed_fraction, 6),
                    "records": independent_object_exclusions,
                },
            },
        }
        records.append(
            {
                "object_id": object_id,
                "role": target.get("role"),
                "status": "refined",
                "prompt": target["prompt"],
                "target_bbox_abs_xyxy": target_box,
                "positive_bbox_abs_xyxy": positive_box,
                "selected_candidate_id": winner["candidate_id"],
                "selected_score": winner["score"],
                "selected_mask_path": str(mask_path),
                "selected_area_pixels": int(winner["mask"].sum()),
                "selected_mask_bbox_abs_xyxy": candidate_box,
                "area_before_pillow_exclusion": area_before_exclusion,
                "area_after_pillow_exclusion": area_after_pillow_exclusion,
                "pillow_exclusion_fraction": round(
                    pillow_removed_fraction, 6
                ),
                "pillow_exclusions": pillow_exclusions,
                "area_after_independent_object_exclusion": (
                    area_after_exclusion
                ),
                "independent_object_exclusion_fraction": round(
                    independent_removed_fraction, 6
                ),
                "total_protected_object_exclusion_fraction": round(
                    removed_fraction, 6
                ),
                "independent_object_exclusions": (
                    independent_object_exclusions
                ),
                "evaluations": public_evaluations,
            }
        )

    inference_after = int(getattr(session, "inference_run_count", 0))
    audit = {
        "schema_version": "scenemosaic_functional_assembly_roi_refinement",
        "status": "blocked" if errors else ("pass" if targets else "noop"),
        "config": asdict(cfg),
        "target_count": len(targets),
        "refined_count": sum(
            item.get("status") == "refined" for item in records
        ),
        "records": records,
        "additional_model_load_count": 0,
        "inference_count": inference_after - inference_before,
        "additional_vlm_api_calls": 0,
        "formal_outputs_modified": False,
        "hard_errors": errors,
    }
    audit_path = output / "functional_assembly_roi_refinement_audit.json"
    _atomic_json(audit_path, audit)
    return {
        "registry": copied,
        "audit": audit,
        "audit_path": str(audit_path),
    }


def _pair_is_geometrically_fixed(
    bathtub: Mapping[str, Any], faucet: Mapping[str, Any]
) -> bool:
    """Require the tap to be anchored at the bathtub rim, not merely nearby."""

    bathtub_box = _bbox(bathtub)
    faucet_box = _bbox(faucet)
    bathtub_width = bathtub_box[2] - bathtub_box[0]
    bathtub_height = bathtub_box[3] - bathtub_box[1]
    horizontal_overlap = _overlap_1d(
        bathtub_box[0], bathtub_box[2], faucet_box[0], faucet_box[2]
    )
    if horizontal_overlap < 0.55 * (faucet_box[2] - faucet_box[0]):
        return False
    return bool(
        faucet_box[2] >= bathtub_box[0] - 0.08 * bathtub_width
        and faucet_box[0] <= bathtub_box[2] + 0.08 * bathtub_width
        and faucet_box[1] <= bathtub_box[1] + 0.20 * bathtub_height
        and faucet_box[3] >= bathtub_box[1] - 0.28 * bathtub_height
    )


def union_fixed_bathtub_faucet_masks(
    *,
    image_path: str | Path,
    registry: Mapping[str, Any],
    output_dir: str | Path,
    config: AssemblyConfig | None = None,
) -> dict[str, Any]:
    """Union only an explicitly typed, uniquely matched fixed bathtub faucet.

    The source faucet remains in the temporary Pass-A registry so that Pass B can
    retain its source identity; forced_removals removes that duplicate from
    the final fused proposal deterministically.  No IDs are renumbered here.
    """

    cfg = config or AssemblyConfig()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fixed component output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copied = copy.deepcopy(dict(registry))
    objects = copied.get("objects")
    if not isinstance(objects, list):
        raise ValueError("registry must contain objects")
    image = np.array(
        Image.open(Path(image_path).expanduser().resolve()).convert("RGB"),
        dtype=np.uint8,
    )
    shape = tuple(image.shape[:2])
    bathtubs = [item for item in objects if _is_bathtub(item)]
    faucets = [item for item in objects if _is_fixed_bathtub_faucet(item)]
    records = []
    hard_errors = []
    forced_removals = []
    object_updates = {}
    accepted_count = 0
    for bathtub in bathtubs:
        bathtub_id = str(bathtub.get("object_id") or "")
        matches = [
            faucet
            for faucet in faucets
            if _pair_is_geometrically_fixed(bathtub, faucet)
        ]
        if len(matches) != 1:
            records.append(
                {
                    "bathtub_object_id": bathtub_id,
                    "status": "skipped",
                    "reason": (
                        "no_unique_fixed_bathtub_faucet"
                        if not matches
                        else "ambiguous_fixed_bathtub_faucet"
                    ),
                    "candidate_faucet_object_ids": [
                        str(item.get("object_id") or "") for item in matches
                    ],
                }
            )
            continue
        if accepted_count >= cfg.maximum_fixed_faucet_unions:
            hard_errors.append(
                {
                    "type": "fixed_bathtub_faucet_union_capacity_overflow",
                    "maximum": cfg.maximum_fixed_faucet_unions,
                }
            )
            break
        faucet = matches[0]
        faucet_id = str(faucet.get("object_id") or "")
        try:
            bathtub_mask = _load_mask(
                (bathtub.get("sam3") or {}).get("mask_path"), shape
            )
            faucet_mask = _load_mask(
                (faucet.get("sam3") or {}).get("mask_path"), shape
            )
        except Exception as exc:
            hard_errors.append(
                {
                    "type": "fixed_bathtub_faucet_union_invalid_source_mask",
                    "bathtub_object_id": bathtub_id,
                    "faucet_object_id": faucet_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        conflicting = []
        for other in objects:
            other_id = str(other.get("object_id") or "")
            if other_id in {bathtub_id, faucet_id}:
                continue
            other_sam3 = other.get("sam3") or {}
            if other_sam3.get("status") != "success":
                continue
            try:
                other_mask = _load_mask(other_sam3.get("mask_path"), shape)
            except Exception as exc:
                hard_errors.append(
                    {
                        "type": "fixed_bathtub_faucet_union_invalid_peer_mask",
                        "bathtub_object_id": bathtub_id,
                        "peer_object_id": other_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            overlap_fraction = int(np.logical_and(faucet_mask, other_mask).sum()) / max(
                int(faucet_mask.sum()), 1
            )
            if overlap_fraction > 0.10:
                conflicting.append(
                    {
                        "object_id": other_id,
                        "overlap_fraction_of_faucet": round(overlap_fraction, 6),
                    }
                )
        if conflicting:
            records.append(
                {
                    "bathtub_object_id": bathtub_id,
                    "faucet_object_id": faucet_id,
                    "status": "skipped",
                    "reason": "faucet_overlaps_independent_object",
                    "conflicts": conflicting,
                }
            )
            continue
        union_mask = np.logical_or(bathtub_mask, faucet_mask)
        selected = output / "selected"
        selected.mkdir(parents=True, exist_ok=True)
        mask_path = selected / f"{bathtub_id.lower()}_mask.png"
        overlay_path = selected / f"{bathtub_id.lower()}_overlay.png"
        Image.fromarray(union_mask.astype(np.uint8) * 255).save(mask_path)
        canvas = image.astype(np.float32).copy()
        canvas[union_mask] = (
            0.55 * canvas[union_mask]
            + 0.45 * np.array((255, 0, 0), dtype=np.float32)
        )
        Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(
            overlay_path
        )
        previous = copy.deepcopy(dict(bathtub.get("sam3") or {}))
        bathtub["sam3"] = {
            **previous,
            "status": "success",
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "area_pixels": int(union_mask.sum()),
            "mask_bbox_abs_xyxy": _mask_bbox(union_mask),
            "selection_method": "fixed_bathtub_faucet_mask_union",
            "fixed_component_mask_union": {
                "component_object_id": faucet_id,
                "host_area_pixels_before": int(bathtub_mask.sum()),
                "component_area_pixels": int(faucet_mask.sum()),
                "union_area_pixels": int(union_mask.sum()),
            },
        }
        faucet_sam3 = dict(faucet.get("sam3") or {})
        faucet_sam3["fixed_component_absorbed_by"] = bathtub_id
        faucet["sam3"] = faucet_sam3
        object_updates[bathtub_id] = {
            "sam3_prompt": "complete bathtub including fixed faucet",
            "description": "complete bathtub including its fixed faucet",
        }
        forced_removals.append(faucet_id)
        accepted_count += 1
        records.append(
            {
                "bathtub_object_id": bathtub_id,
                "faucet_object_id": faucet_id,
                "status": "unioned",
                "selected_mask_path": str(mask_path),
                "forced_final_removal": faucet_id,
            }
        )
    audit = {
        "schema_version": "scenemosaic_fixed_bathtub_faucet_union",
        "status": "blocked" if hard_errors else ("pass" if records and accepted_count else "noop"),
        "bathtub_count": len(bathtubs),
        "typed_faucet_count": len(faucets),
        "accepted_count": accepted_count,
        "records": records,
        "forced_removals": forced_removals,
        "object_updates": object_updates,
        "additional_vlm_api_calls": 0,
        "formal_outputs_modified": False,
        "hard_errors": hard_errors,
    }
    audit_path = output / "fixed_bathtub_faucet_union_audit.json"
    _atomic_json(audit_path, audit)
    return {"registry": copied, "audit": audit, "audit_path": str(audit_path)}


def apply_fixed_component_absorption_actions(
    actions: Mapping[str, Any], union_audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Force only deterministic component removals after Pass-B validation."""

    copied = copy.deepcopy(dict(actions))
    forced = sorted({str(item) for item in union_audit.get("forced_removals") or []})
    updates = list(copied.get("updates") or [])
    removed_updates = [
        item for item in updates if str(item.get("pass_a_id") or "") in forced
    ]
    copied["updates"] = [
        item for item in updates if str(item.get("pass_a_id") or "") not in forced
    ]
    original_removals = [str(item) for item in copied.get("removals") or []]
    copied["removals"] = sorted(set(original_removals) | set(forced))
    return {
        "actions": copied,
        "audit": {
            "schema_version": "scenemosaic_fixed_component_action_policy",
            "status": "pass" if forced else "noop",
            "forced_removals": forced,
            "suppressed_component_updates": removed_updates,
            "additional_vlm_api_calls": 0,
            "formal_outputs_modified": False,
            "hard_errors": [],
        },
    }


__all__ = [
    "AssemblyConfig",
    "PlantContainerMaskCompletionConfig",
    "normalize_pass_a_functional_assemblies",
    "complete_plant_container_masks",
    "run_functional_assembly_roi_refinement",
    "union_fixed_bathtub_faucet_masks",
    "apply_fixed_component_absorption_actions",
]
