#!/usr/bin/env python3
"""Audit repeated content prompts inside verified open-storage hosts."""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OpenStorageContentPromptConfig:
    minimum_repeated_instances: int = 2
    minimum_bbox_coverage_by_host: float = 0.80


GENERIC_CATEGORY_WORDS = {"item", "object", "thing", "unknown"}


def _words(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower().replace("_", " "))


def _canonical_category(value: Any) -> str:
    return " ".join(_words(value))


def _valid_bbox(value: Any) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not (0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0):
        return None
    return box


def _bbox_coverage(inner: Sequence[float], outer: Sequence[float]) -> float:
    x1 = max(float(inner[0]), float(outer[0]))
    y1 = max(float(inner[1]), float(outer[1]))
    x2 = min(float(inner[2]), float(outer[2]))
    y2 = min(float(inner[3]), float(outer[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (float(inner[2]) - float(inner[0])) * (
        float(inner[3]) - float(inner[1])
    )
    return intersection / area if area else 0.0


def canonicalize_open_storage_content_prompts(
    *,
    objects: Sequence[Mapping[str, Any]],
    content_hosts: Sequence[Mapping[str, Any]],
    config: OpenStorageContentPromptConfig | None = None,
) -> dict[str, Any]:
    """Record repeated host contents without rewriting instance prompts.

    A repeated category may contain visually distinct objects or deliberately
    grouped units. Its declared sam3_prompt is therefore an input contract
    for segmentation, not a field this audit may normalize.
    """

    cfg = config or OpenStorageContentPromptConfig()
    if cfg.minimum_repeated_instances < 2:
        raise ValueError("minimum_repeated_instances must be at least two")
    if not 0.0 < cfg.minimum_bbox_coverage_by_host <= 1.0:
        raise ValueError("minimum_bbox_coverage_by_host must be in (0, 1]")
    copied = [
        copy.deepcopy(dict(item))
        for item in objects
        if isinstance(item, Mapping)
    ]
    hosts = []
    for raw in content_hosts:
        box = _valid_bbox(raw.get("bbox_norm_xyxy"))
        if str(raw.get("host_mode") or "") != "open_storage" or box is None:
            continue
        hosts.append(
            {
                "source_object_id": str(raw.get("source_object_id") or ""),
                "bbox_norm_xyxy": box,
                "area": (box[2] - box[0]) * (box[3] - box[1]),
            }
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejections = []
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        box = _valid_bbox(obj.get("bbox_2d_norm_xyxy"))
        category = _canonical_category(obj.get("category"))
        category_words = set(_words(category))
        prompt = " ".join(_words(obj.get("sam3_prompt")))
        if (
            not object_id
            or box is None
            or not category
            or category_words <= GENERIC_CATEGORY_WORDS
            or not prompt
        ):
            continue
        matching_hosts = [
            host
            for host in hosts
            if _bbox_coverage(box, host["bbox_norm_xyxy"])
            >= cfg.minimum_bbox_coverage_by_host
        ]
        if not matching_hosts:
            continue
        host = min(matching_hosts, key=lambda item: item["area"])
        if not category_words.issubset(set(_words(prompt))):
            rejections.append(
                {
                    "object_id": object_id,
                    "category": category,
                    "sam3_prompt": prompt,
                    "host_source_object_id": host["source_object_id"],
                    "reason": "category_words_not_preserved_by_prompt",
                }
            )
            continue
        groups[(host["source_object_id"], category)].append(
            {
                "object": obj,
                "object_id": object_id,
                "old_prompt": prompt,
                "bbox_coverage_by_host": _bbox_coverage(
                    box, host["bbox_norm_xyxy"]
                ),
            }
        )

    changes = []
    preserved = []
    checked_groups = []
    for (host_id, category), members in sorted(groups.items()):
        checked_groups.append(
            {
                "host_source_object_id": host_id,
                "category": category,
                "object_ids": [item["object_id"] for item in members],
                "input_prompts": sorted({item["old_prompt"] for item in members}),
            }
        )
        if len(members) < cfg.minimum_repeated_instances:
            continue
        for member in members:
            preserved.append(
                {
                    "object_id": member["object_id"],
                    "host_source_object_id": host_id,
                    "category": category,
                    "sam3_prompt": member["old_prompt"],
                    "bbox_coverage_by_host": round(
                        member["bbox_coverage_by_host"], 6
                    ),
                }
            )

    audit = {
        "schema_version": "scenemosaic_open_storage_content_prompt_audit",
        "status": "noop",
        "config": asdict(cfg),
        "verified_open_storage_host_count": len(hosts),
        "checked_groups": checked_groups,
        "changed_count": len(changes),
        "changes": changes,
        "preserved_count": len(preserved),
        "preserved": preserved,
        "rejections": rejections,
        "semantic_categories_modified": False,
        "object_count_modified": False,
        "additional_sam3_inference_calls": 0,
        "additional_vlm_calls": 0,
        "hard_errors": [],
    }
    return {"objects": copied, "audit": audit}


__all__ = [
    "OpenStorageContentPromptConfig",
    "canonicalize_open_storage_content_prompts",
]
