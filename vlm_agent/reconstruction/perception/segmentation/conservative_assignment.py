#!/usr/bin/env python3
"""Conservative post-assignment recovery for production SAM3 segmentation.

The primary SAM3 assignment remains authoritative.  This module only plans
one-to-one reuse of candidates that production left unassigned.  It never
steals a successful candidate and never performs global category-order
matching.
"""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


CARRIER_ORDER_MAXIMUM_CENTER_OBJECT_SCALE = 1.25
CARRIER_ORDER_MAXIMUM_CENTER_CARRIER_SCALE = 0.20
CARRIER_ORDER_MAXIMUM_ANCHOR_GAP_OBJECT_SCALE = 1.00


def bbox_area(bbox):
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_intersection(left, right):
    if left is None or right is None:
        return 0
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def bbox_iou(left, right):
    intersection = bbox_intersection(left, right)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union else 0.0


def bbox_center(bbox):
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def normalized_bbox_to_abs(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return [
        max(0, min(width, int(round(float(bbox[0]) * width)))),
        max(0, min(height, int(round(float(bbox[1]) * height)))),
        max(0, min(width, int(round(float(bbox[2]) * width)))),
        max(0, min(height, int(round(float(bbox[3]) * height)))),
    ]


def object_point(obj, width, height):
    anchor = obj.get("anchor_2d_norm_xy")
    if (
        isinstance(anchor, (list, tuple))
        and len(anchor) == 2
        and all(isinstance(value, (int, float)) for value in anchor)
    ):
        return (float(anchor[0]) * width, float(anchor[1]) * height)
    return bbox_center(
        normalized_bbox_to_abs(obj.get("bbox_2d_norm_xyxy"), width, height)
    )


def contains_point(bbox, point):
    return bool(
        bbox is not None
        and point is not None
        and bbox[0] <= point[0] <= bbox[2]
        and bbox[1] <= point[1] <= bbox[3]
    )


def point_bbox_distance(point, bbox):
    if point is None or bbox is None:
        return math.inf
    dx = max(float(bbox[0]) - point[0], 0.0, point[0] - float(bbox[2]))
    dy = max(float(bbox[1]) - point[1], 0.0, point[1] - float(bbox[3]))
    return math.hypot(dx, dy)


def carrier_order_geometry_evidence(
    obj,
    candidate_bbox,
    carrier_bbox,
    width,
    height,
):
    target_bbox = normalized_bbox_to_abs(
        obj.get("bbox_2d_norm_xyxy"), width, height
    )
    anchor = object_point(obj, width, height)
    target_center = bbox_center(target_bbox)
    candidate_center = bbox_center(candidate_bbox)
    if (
        target_bbox is None
        or candidate_bbox is None
        or carrier_bbox is None
        or anchor is None
        or target_center is None
        or candidate_center is None
        or bbox_area(target_bbox) <= 0
        or bbox_area(candidate_bbox) <= 0
        or bbox_area(carrier_bbox) <= 0
    ):
        return {
            "status": "reject",
            "reason": "invalid_geometry",
        }
    target_diagonal = math.hypot(
        target_bbox[2] - target_bbox[0],
        target_bbox[3] - target_bbox[1],
    )
    candidate_diagonal = math.hypot(
        candidate_bbox[2] - candidate_bbox[0],
        candidate_bbox[3] - candidate_bbox[1],
    )
    carrier_diagonal = math.hypot(
        carrier_bbox[2] - carrier_bbox[0],
        carrier_bbox[3] - carrier_bbox[1],
    )
    object_scale = max(target_diagonal, candidate_diagonal, 1.0)
    carrier_scale = max(carrier_diagonal, 1.0)
    center_distance = math.dist(target_center, candidate_center)
    center_object_scale = center_distance / object_scale
    center_carrier_scale = center_distance / carrier_scale
    anchor_gap_object_scale = (
        point_bbox_distance(anchor, candidate_bbox) / object_scale
    )
    accepted = (
        center_object_scale <= CARRIER_ORDER_MAXIMUM_CENTER_OBJECT_SCALE
        and center_carrier_scale <= CARRIER_ORDER_MAXIMUM_CENTER_CARRIER_SCALE
        and anchor_gap_object_scale
        <= CARRIER_ORDER_MAXIMUM_ANCHOR_GAP_OBJECT_SCALE
    )
    return {
        "status": "accept" if accepted else "reject",
        "reason": None if accepted else "candidate_object_geometry_infeasible",
        "target_bbox_abs_xyxy": target_bbox,
        "candidate_bbox_abs_xyxy": list(candidate_bbox),
        "carrier_bbox_abs_xyxy": list(carrier_bbox),
        "bbox_iou": round(float(bbox_iou(target_bbox, candidate_bbox)), 6),
        "anchor_in_candidate_bbox": contains_point(candidate_bbox, anchor),
        "center_distance_object_scale": round(center_object_scale, 6),
        "center_distance_carrier_scale": round(center_carrier_scale, 6),
        "anchor_gap_object_scale": round(anchor_gap_object_scale, 6),
        "thresholds": {
            "maximum_center_object_scale": (
                CARRIER_ORDER_MAXIMUM_CENTER_OBJECT_SCALE
            ),
            "maximum_center_carrier_scale": (
                CARRIER_ORDER_MAXIMUM_CENTER_CARRIER_SCALE
            ),
            "maximum_anchor_gap_object_scale": (
                CARRIER_ORDER_MAXIMUM_ANCHOR_GAP_OBJECT_SCALE
            ),
        },
    }


def strictly_contains(outer, inner):
    return bool(
        outer is not None
        and inner is not None
        and bbox_area(outer) > bbox_area(inner)
        and outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def candidate_index(record):
    explicit = record.get("candidate_index")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    match = re.search(r"_(\d{3})$", str(record.get("candidate_id") or ""))
    return int(match.group(1)) if match else None


def unique_minimal_carrier(
    point,
    unit_bbox,
    carrier_objects,
    width,
    height,
    excluded_ids,
):
    """Find one minimal enclosing object, or abstain on overlap ambiguity."""
    if point is None or unit_bbox is None:
        return None, []
    unit_area = bbox_area(unit_bbox)
    records = []
    for obj in carrier_objects:
        object_id = str(obj.get("object_id") or "")
        if not object_id or object_id in excluded_ids:
            continue
        carrier_bbox = normalized_bbox_to_abs(
            obj.get("bbox_2d_norm_xyxy"), width, height
        )
        if (
            carrier_bbox is None
            or not contains_point(carrier_bbox, point)
            or not strictly_contains(carrier_bbox, unit_bbox)
        ):
            continue
        records.append((object_id, carrier_bbox))
    minimal = [
        record
        for record in records
        if not any(
            other[0] != record[0] and strictly_contains(record[1], other[1])
            for other in records
        )
    ]
    if len(minimal) != 1:
        return None, sorted(object_id for object_id, _ in minimal)
    return minimal[0][0], [minimal[0][0]]


def principal_axis_orders(object_points, candidate_points):
    labels = [
        ("object", key, point) for key, point in object_points.items()
    ] + [
        ("candidate", key, point) for key, point in candidate_points.items()
    ]
    if len(labels) < 2 or any(point is None for _, _, point in labels):
        return None
    coordinates = np.asarray([point for _, _, point in labels], dtype=float)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 0:
        return None
    axis = eigenvectors[:, -1]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    projections = coordinates @ axis
    object_projection = {}
    candidate_projection = {}
    for (kind, key, _), projection in zip(labels, projections):
        target = object_projection if kind == "object" else candidate_projection
        target[key] = float(projection)
    if len(set(object_projection.values())) != len(object_projection):
        return None
    if len(set(candidate_projection.values())) != len(candidate_projection):
        return None
    object_order = sorted(object_projection, key=object_projection.get)
    candidate_order = sorted(candidate_projection, key=candidate_projection.get)
    return {
        "axis": [round(float(value), 9) for value in axis],
        "object_order": object_order,
        "candidate_order": candidate_order,
        "object_ranks": {key: rank for rank, key in enumerate(object_order)},
        "candidate_ranks": {
            key: rank for rank, key in enumerate(candidate_order)
        },
    }


def unique_monotone_path(
    object_ranks,
    candidate_ranks,
    object_count,
    candidate_count,
):
    """Return a unique ordinal optimum, independent of image pixels."""
    object_ranks = tuple(object_ranks)
    candidate_ranks = tuple(candidate_ranks)
    if len(candidate_ranks) < len(object_ranks):
        return None
    object_denominator = max(1, object_count - 1)
    candidate_denominator = max(1, candidate_count - 1)

    @lru_cache(maxsize=None)
    def solve(object_offset, candidate_offset):
        remaining_objects = len(object_ranks) - object_offset
        remaining_candidates = len(candidate_ranks) - candidate_offset
        if remaining_objects == 0:
            return (0, 1, ())
        if remaining_candidates < remaining_objects:
            return (None, 0, ())
        options = []
        skipped = solve(object_offset, candidate_offset + 1)
        if skipped[0] is not None:
            options.append(skipped)
        matched = solve(object_offset + 1, candidate_offset + 1)
        if matched[0] is not None:
            ordinal_cost = abs(
                object_ranks[object_offset] * candidate_denominator
                - candidate_ranks[candidate_offset] * object_denominator
            )
            options.append(
                (
                    ordinal_cost + matched[0],
                    matched[1],
                    (candidate_ranks[candidate_offset],) + matched[2],
                )
            )
        if not options:
            return (None, 0, ())
        minimum = min(option[0] for option in options)
        best = [option for option in options if option[0] == minimum]
        ways = min(2, sum(option[1] for option in best))
        return (minimum, ways, best[0][2])

    cost, ways, path = solve(0, 0)
    if cost is None or ways != 1:
        return None
    return {"ordinal_cost": int(cost), "candidate_ranks": list(path)}


def default_mask_loader(path):
    return np.asarray(Image.open(path).convert("L")) > 0


def masks_conflict(left_path, right_path, mask_loader):
    """Reject material overlap while ignoring a few anti-aliased edge pixels."""
    if not left_path or not right_path:
        return True
    try:
        left = np.asarray(mask_loader(left_path), dtype=bool)
        right = np.asarray(mask_loader(right_path), dtype=bool)
    except (OSError, ValueError):
        return True
    if left.shape != right.shape:
        return True
    overlap = int(np.logical_and(left, right).sum())
    if overlap == 0:
        return False
    smaller_area = min(int(left.sum()), int(right.sum()))
    if smaller_area <= 0:
        return True
    return overlap >= 64 or overlap / smaller_area >= 0.01


def _prompt_groups(objects, unassigned):
    groups = defaultdict(lambda: {"objects": [], "unassigned": []})
    for obj in objects:
        sam3 = obj.get("sam3") or {}
        prompt = str(sam3.get("prompt_used") or "").strip()
        if prompt:
            groups[prompt]["objects"].append(obj)
    for candidate in unassigned:
        prompt = str(candidate.get("sam3_prompt") or "").strip()
        if prompt:
            groups[prompt]["unassigned"].append(candidate)
    return groups


def plan_registry_fallbacks(
    registry: Mapping[str, Any],
    *,
    carrier_objects: Sequence[Mapping[str, Any]] | None = None,
    mask_loader: Callable[[str], np.ndarray] = default_mask_loader,
):
    """Plan conservative reuse without mutating the SAM3 registry.

    carrier_objects may contain the full fused registry when registry
    is a prompt-filtered delta run.  Only registry objects can receive a
    fallback; the extra objects are used solely as enclosing carrier evidence.
    """
    objects = copy.deepcopy(list(registry.get("objects") or []))
    carrier_context = copy.deepcopy(
        list(carrier_objects) if carrier_objects is not None else objects
    )
    carrier_by_id = {
        str(item.get("object_id")): item
        for item in carrier_context
        if str(item.get("object_id") or "")
    }
    unassigned = copy.deepcopy(
        list(registry.get("unassigned_sam3_candidates") or [])
    )
    width, height = [int(value) for value in registry.get("image_size") or []]
    if width <= 0 or height <= 0:
        raise ValueError("registry image_size must be [width, height]")

    matches = []
    abstentions = []
    successful_mask_paths = [
        (obj.get("sam3") or {}).get("mask_path")
        for obj in objects
        if (obj.get("sam3") or {}).get("status") == "success"
    ]
    for prompt, group in _prompt_groups(objects, unassigned).items():
        target_objects = group["objects"]
        failed = [
            obj for obj in target_objects
            if (obj.get("sam3") or {}).get("status") != "success"
        ]
        if not failed or not group["unassigned"]:
            continue

        candidates = {}
        locked = {}
        for obj in target_objects:
            sam3 = obj.get("sam3") or {}
            if sam3.get("status") != "success":
                continue
            indices = list(sam3.get("merged_candidate_indices") or [])
            if not indices and isinstance(
                sam3.get("candidate_index_in_prompt_group"), int
            ):
                indices = [sam3["candidate_index_in_prompt_group"]]
            if len(indices) != 1:
                locked[str(obj["object_id"])] = None
                continue
            index = int(indices[0])
            candidates[index] = {
                "candidate_index": index,
                "candidate_id": f"locked_{prompt}_{index:03d}",
                "bbox": list(sam3.get("mask_bbox_abs_xyxy") or []),
                "mask_path": sam3.get("mask_path"),
                "assigned_object_id": str(obj["object_id"]),
                "record": None,
            }
            locked[str(obj["object_id"])] = index
        for record in group["unassigned"]:
            index = candidate_index(record)
            bbox = record.get("mask_bbox_abs_xyxy")
            if index is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if index in candidates:
                abstentions.append(
                    {
                        "prompt": prompt,
                        "candidate_id": record.get("candidate_id"),
                        "reason": "duplicate_candidate_index",
                    }
                )
                continue
            candidates[index] = {
                "candidate_index": index,
                "candidate_id": str(record["candidate_id"]),
                "bbox": list(bbox),
                "mask_path": record.get("mask_path"),
                "assigned_object_id": None,
                "record": record,
            }

        used_indices = {
            index for index in locked.values() if isinstance(index, int)
        }
        available = {
            index: value
            for index, value in candidates.items()
            if index not in used_indices and value["record"] is not None
        }

        # Independent image-edge containment fallback. This is intentionally
        # not generalized to ordinary containment.
        edge_options = defaultdict(list)
        edge_owners = defaultdict(list)
        locked_mask_paths = list(successful_mask_paths)
        for obj in failed:
            object_id = str(obj["object_id"])
            object_bbox = normalized_bbox_to_abs(
                obj.get("bbox_2d_norm_xyxy"), width, height
            )
            if object_bbox is None or bbox_area(object_bbox) <= 0:
                continue
            object_edges = set()
            if object_bbox[0] == 0:
                object_edges.add("left")
            if object_bbox[1] == 0:
                object_edges.add("top")
            if object_bbox[2] == width:
                object_edges.add("right")
            if object_bbox[3] == height:
                object_edges.add("bottom")
            for index, candidate in available.items():
                candidate_bbox = candidate["bbox"]
                candidate_edges = set()
                if candidate_bbox[0] == 0:
                    candidate_edges.add("left")
                if candidate_bbox[1] == 0:
                    candidate_edges.add("top")
                if candidate_bbox[2] == width:
                    candidate_edges.add("right")
                if candidate_bbox[3] == height:
                    candidate_edges.add("bottom")
                shared_edges = object_edges & candidate_edges
                object_area = bbox_area(object_bbox)
                coverage = (
                    bbox_intersection(candidate_bbox, object_bbox) / object_area
                    if object_area
                    else 0.0
                )
                if (
                    not shared_edges
                    or bbox_area(candidate_bbox) < object_area
                    or coverage < 0.90
                ):
                    continue
                if any(
                    masks_conflict(
                        candidate["mask_path"], locked_path, mask_loader
                    )
                    for locked_path in locked_mask_paths
                ):
                    continue
                option = {
                    "candidate_index": index,
                    "candidate_id": candidate["candidate_id"],
                    "shared_image_edges": sorted(shared_edges),
                    "object_bbox_coverage": round(float(coverage), 6),
                }
                edge_options[object_id].append(option)
                edge_owners[index].append(object_id)
        for obj in failed:
            object_id = str(obj["object_id"])
            options = [
                option
                for option in edge_options[object_id]
                if len(edge_owners[option["candidate_index"]]) == 1
            ]
            if len(options) != 1:
                if edge_options[object_id]:
                    abstentions.append(
                        {
                            "object_id": object_id,
                            "prompt": prompt,
                            "method": "shared_image_edge_object_bbox_containment",
                            "reason": "non_unique_edge_containment",
                            "candidate_ids": sorted(
                                option["candidate_id"]
                                for option in edge_options[object_id]
                            ),
                        }
                    )
                continue
            option = options[0]
            matches.append(
                {
                    "object_id": object_id,
                    "candidate_id": option["candidate_id"],
                    "candidate_index": option["candidate_index"],
                    "prompt": prompt,
                    "method": "shared_image_edge_object_bbox_containment",
                    "audit": option,
                }
            )
            used_indices.add(option["candidate_index"])
            available.pop(option["candidate_index"], None)
            locked[object_id] = option["candidate_index"]

        remaining_failed = [
            obj for obj in failed if str(obj["object_id"]) not in locked
        ]
        if not remaining_failed or len(target_objects) < 2:
            continue

        excluded_ids = {str(obj["object_id"]) for obj in target_objects}
        target_carriers = {}
        target_carrier_alternatives = {}
        for obj in target_objects:
            object_id = str(obj["object_id"])
            bbox = normalized_bbox_to_abs(
                obj.get("bbox_2d_norm_xyxy"), width, height
            )
            carrier, alternatives = unique_minimal_carrier(
                object_point(obj, width, height),
                bbox,
                carrier_context,
                width,
                height,
                excluded_ids,
            )
            target_carriers[object_id] = carrier
            target_carrier_alternatives[object_id] = alternatives

        candidate_carriers = {}
        for index, candidate in candidates.items():
            carrier, _ = unique_minimal_carrier(
                bbox_center(candidate["bbox"]),
                candidate["bbox"],
                carrier_context,
                width,
                height,
                excluded_ids,
            )
            candidate_carriers[index] = carrier

        carrier_ids = sorted(
            {
                target_carriers[str(obj["object_id"])]
                for obj in remaining_failed
                if target_carriers[str(obj["object_id"])] is not None
            }
        )
        for obj in remaining_failed:
            object_id = str(obj["object_id"])
            if target_carriers[object_id] is None:
                abstentions.append(
                    {
                        "object_id": object_id,
                        "prompt": prompt,
                        "method": "carrier_order",
                        "reason": "no_unique_carrier",
                        "minimal_carrier_candidates": (
                            target_carrier_alternatives[object_id]
                        ),
                    }
                )

        for carrier_id in carrier_ids:
            carrier_record = carrier_by_id.get(carrier_id)
            carrier_bbox = normalized_bbox_to_abs(
                (carrier_record or {}).get("bbox_2d_norm_xyxy"),
                width,
                height,
            )
            if carrier_bbox is None or bbox_area(carrier_bbox) <= 0:
                abstentions.extend(
                    {
                        "object_id": str(obj["object_id"]),
                        "prompt": prompt,
                        "method": "carrier_order",
                        "reason": "invalid_carrier_geometry",
                        "carrier_object_id": carrier_id,
                    }
                    for obj in remaining_failed
                    if target_carriers[str(obj["object_id"])] == carrier_id
                )
                continue
            carrier_object_ids = [
                str(obj["object_id"])
                for obj in target_objects
                if target_carriers[str(obj["object_id"])] == carrier_id
            ]
            carrier_candidate_indices = [
                index
                for index, candidate_carrier in candidate_carriers.items()
                if candidate_carrier == carrier_id
            ]
            carrier_failed_ids = [
                object_id
                for object_id in carrier_object_ids
                if object_id not in locked
            ]
            if not carrier_failed_ids:
                continue
            carrier_locked = [
                (object_id, locked[object_id])
                for object_id in carrier_object_ids
                if isinstance(locked.get(object_id), int)
            ]
            if not carrier_locked or any(
                candidate_carriers.get(index) != carrier_id
                for _, index in carrier_locked
            ):
                abstentions.extend(
                    {
                        "object_id": object_id,
                        "prompt": prompt,
                        "method": "carrier_order",
                        "reason": "no_consistent_locked_pair",
                        "carrier_object_id": carrier_id,
                    }
                    for object_id in carrier_failed_ids
                )
                continue

            object_by_id = {
                str(obj["object_id"]): obj for obj in target_objects
            }
            order = principal_axis_orders(
                {
                    object_id: object_point(
                        object_by_id[object_id], width, height
                    )
                    for object_id in carrier_object_ids
                },
                {
                    index: bbox_center(candidates[index]["bbox"])
                    for index in carrier_candidate_indices
                },
            )
            if order is None:
                abstentions.extend(
                    {
                        "object_id": object_id,
                        "prompt": prompt,
                        "method": "carrier_order",
                        "reason": "non_unique_principal_axis_order",
                        "carrier_object_id": carrier_id,
                    }
                    for object_id in carrier_failed_ids
                )
                continue
            object_ranks = order["object_ranks"]
            candidate_ranks = order["candidate_ranks"]
            locked_ranks = sorted(
                (
                    object_ranks[object_id],
                    candidate_ranks[index],
                    object_id,
                    index,
                )
                for object_id, index in carrier_locked
            )
            if any(
                left[1] >= right[1]
                for left, right in zip(locked_ranks, locked_ranks[1:])
            ):
                abstentions.extend(
                    {
                        "object_id": object_id,
                        "prompt": prompt,
                        "method": "carrier_order",
                        "reason": "locked_pairs_cross_in_order",
                        "carrier_object_id": carrier_id,
                    }
                    for object_id in carrier_failed_ids
                )
                continue

            object_by_rank = {
                rank: object_id for object_id, rank in object_ranks.items()
            }
            candidate_by_rank = {
                rank: index for index, rank in candidate_ranks.items()
            }
            boundaries = [
                (-1, -1, None, None),
                *locked_ranks,
                (
                    len(carrier_object_ids),
                    len(carrier_candidate_indices),
                    None,
                    None,
                ),
            ]
            for lower, upper in zip(boundaries, boundaries[1:]):
                gap_object_ranks = [
                    rank
                    for rank in range(lower[0] + 1, upper[0])
                    if object_by_rank[rank] not in locked
                ]
                if not gap_object_ranks:
                    continue
                gap_candidate_ranks = [
                    rank
                    for rank in range(lower[1] + 1, upper[1])
                    if candidate_by_rank[rank] not in used_indices
                ]
                alignment = unique_monotone_path(
                    gap_object_ranks,
                    gap_candidate_ranks,
                    len(carrier_object_ids),
                    len(carrier_candidate_indices),
                )
                if alignment is None:
                    abstentions.extend(
                        {
                            "object_id": object_by_rank[rank],
                            "prompt": prompt,
                            "method": "carrier_order",
                            "reason": "no_unique_monotone_optimum",
                            "carrier_object_id": carrier_id,
                            "candidate_rank_options": gap_candidate_ranks,
                        }
                        for rank in gap_object_ranks
                    )
                    continue
                for object_rank, candidate_rank in zip(
                    gap_object_ranks, alignment["candidate_ranks"]
                ):
                    object_id = object_by_rank[object_rank]
                    index = candidate_by_rank[candidate_rank]
                    candidate = candidates[index]
                    geometry = carrier_order_geometry_evidence(
                        object_by_id[object_id],
                        candidate["bbox"],
                        carrier_bbox,
                        width,
                        height,
                    )
                    if geometry["status"] != "accept":
                        abstentions.append(
                            {
                                "object_id": object_id,
                                "prompt": prompt,
                                "method": "carrier_order",
                                "reason": "candidate_object_geometry_infeasible",
                                "candidate_id": candidate["candidate_id"],
                                "candidate_index": index,
                                "carrier_object_id": carrier_id,
                                "geometry": geometry,
                            }
                        )
                        continue
                    matches.append(
                        {
                            "object_id": object_id,
                            "candidate_id": candidate["candidate_id"],
                            "candidate_index": index,
                            "prompt": prompt,
                            "method": (
                                "carrier_partitioned_order_preserving_one_to_one"
                            ),
                            "audit": {
                                "carrier_object_id": carrier_id,
                                "principal_axis": order["axis"],
                                "object_rank": object_rank,
                                "candidate_rank": candidate_rank,
                                "ordinal_cost": alignment["ordinal_cost"],
                                "geometry": geometry,
                                "locked_pairs": [
                                    {
                                        "object_id": locked_object_id,
                                        "candidate_index": locked_index,
                                    }
                                    for _, _, locked_object_id, locked_index
                                    in locked_ranks
                                ],
                            },
                        }
                    )
                    used_indices.add(index)
                    locked[object_id] = index

    matched_objects = [match["object_id"] for match in matches]
    matched_candidates = [match["candidate_id"] for match in matches]
    if len(matched_objects) != len(set(matched_objects)):
        raise RuntimeError("fallback planned one object more than once")
    if len(matched_candidates) != len(set(matched_candidates)):
        raise RuntimeError("fallback planned one candidate more than once")
    return {
        "schema_version": "scenemosaic_conservative_assignment_plan",
        "matches": matches,
        "abstentions": abstentions,
    }
