#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


np = None
torch = None
Image = None
Sam3Model = None
Sam3Processor = None

_DEFAULT_RUNTIME_ROOT = Path(
    os.environ.get(
        "SCENEMOSAIC_RUNTIME_ROOT",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "scenemosaic",
    )
).expanduser().resolve()
DEFAULT_MODEL_DIR = str(
    Path(
        os.environ.get(
            "SCENEMOSAIC_PERCEPTION_MODEL_DIR",
            _DEFAULT_RUNTIME_ROOT / "models" / "sam3",
        )
    ).expanduser().resolve()
)
SCORE_THRESHOLD = 0.30
MASK_THRESHOLD = 0.50
MIN_AREA = 30

LOW_IOU_WARNING_THRESHOLD = 0.01
OFF_BBOX_LOW_CONFIDENCE_SCORE = 0.70
IOU_WEIGHT = 10.0
SCORE_WEIGHT = 1.0
CENTER_WEIGHT = 0.5

# Merge candidates must stay local to one coarse VLM box. These guards are
# intentionally category-independent so prompt text does not grow with every
# new failure case.
MERGE_BBOX_PAD_RATIO = 0.10
MERGE_MIN_PAD_PIXELS = 12
LOCAL_MIN_PAD_IMAGE_RATIO = 0.06
MERGE_MIN_CANDIDATE_MASK_COVERAGE = 0.55
MERGE_MAX_CLUSTER_GAP_RATIO = 0.008
MERGE_MIN_CLUSTER_GAP_PIXELS = 6
MERGE_MAX_CLUSTER_GAP_PIXELS = 18
MERGE_MAX_UNION_BBOX_SPILL = 0.25
MERGE_MAX_CROSS_OBJECT_CONTAINMENT = 0.80

# The adaptive policy never uses a global pixel distance to decide semantic composition.
# Distances are normalized by the current VLM object's own scale. Nearby
# candidates are still not merged unless their masks substantially overlap:
# low-overlap neighbors remain atomic evidence for the second VLM review.
ADAPTIVE_LOCAL_MAX_NORMALIZED_GAP = 0.08
ADAPTIVE_LOCAL_MAX_REPEATED_OBJECT_CENTER_DISTANCE = 1.35
ADAPTIVE_LOCAL_MIN_CANDIDATE_COVERAGE_BY_CURRENT = 0.85
ADAPTIVE_LOCAL_MIN_MASK_IOU_FOR_REDUNDANT_MERGE = 0.50


def assignment_has_minimum_local_evidence(iou, score):
    """Reject a weak text mask that does not touch its VLM localization box."""
    return float(iou) > 0.0 or float(score) >= OFF_BBOX_LOW_CONFIDENCE_SCORE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 segmentation for the no-wall-floor pipeline."
    )
    parser.add_argument(
        "--scene_config",
        help="Path to the scene_config.json for the current SceneMosaic run.",
    )
    parser.add_argument(
        "--dry_run_config",
        action="store_true",
        help="Resolve input/output paths and exit without loading the SAM3 model.",
    )
    parser.set_defaults(assignment_policy="adaptive_local")
    return parser.parse_args()


def load_runtime_modules():
    global np, torch, Image, Sam3Model, Sam3Processor

    if np is not None:
        return

    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from transformers import Sam3Model as _Sam3Model, Sam3Processor as _Sam3Processor

    np = _np
    torch = _torch
    Image = _Image
    Sam3Model = _Sam3Model
    Sam3Processor = _Sam3Processor


def safe_name(s):
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in str(s))


def candidate_output_id(prompt, candidate_index):
    return f"candidate_{safe_name(prompt)}_{candidate_index:03d}"


def has_repeated_pass_b_additions(group_objects):
    # Current fusion writes added; created remains accepted in previously materialized caches.
    return sum(
        any(
            isinstance(source, dict)
            and str(source.get("pass") or "").upper() == "B"
            and str(source.get("action") or "").lower() in {"added", "created"}
            for source in (obj.get("proposal_sources") or [])
        )
        for obj in group_objects
    ) >= 2


def get_object_id(obj):
    return obj.get("object_id") or obj.get("image_object_id")


def canonicalize_objects(objects):
    canonical = []
    seen = set()

    for obj in objects:
        object_id = get_object_id(obj)
        if not object_id:
            raise ValueError(f"Object is missing object_id/image_object_id: {obj}")
        if object_id in seen:
            raise ValueError(f"Duplicate object_id detected in VLM input: {object_id}")
        seen.add(object_id)

        normalized_obj = dict(obj)
        normalized_obj.pop("image_object_id", None)
        normalized_obj["object_id"] = object_id
        canonical.append(normalized_obj)

    return canonical


def xyxy_norm_to_abs(bbox, w, h):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(w, int(round(x1 * w)))),
        max(0, min(h, int(round(y1 * h)))),
        max(0, min(w, int(round(x2 * w)))),
        max(0, min(h, int(round(y2 * h)))),
    ]


def abs_to_norm_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def mask_to_bbox_xyxy(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def retain_target_local_components(mask_bool, target_bbox):
    """Discard significant disconnected components with no support in one claim.

    SAM3 can return one text candidate containing two separate instances of the
    same prompt. A single inventory claim must not inherit a remote component
    when another substantial component actually covers its own VLM bbox. The
    guard is geometry-only: components touching the claim remain intact, so it
    does not impose category-specific assembly rules.
    """
    if mask_bool is None or target_bbox is None or not mask_bool.any():
        return mask_bool, {"status": "noop", "reason": "missing_mask_or_bbox"}
    from scipy import ndimage

    labels, count = ndimage.label(
        mask_bool, structure=np.ones((3, 3), dtype=bool)
    )
    if int(count) <= 1:
        return mask_bool, {"status": "noop", "component_count": int(count)}

    x1, y1, x2, y2 = (int(value) for value in target_bbox)
    total_area = int(mask_bool.sum())
    minimum_significant_area = max(MIN_AREA, int(round(total_area * 0.08)))
    supported_labels = set()
    components = []
    for label in range(1, int(count) + 1):
        component = labels == label
        area = int(component.sum())
        if area == 0:
            continue
        bbox = mask_to_bbox_xyxy(component)
        in_target = int(component[y1:y2, x1:x2].sum())
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        target_centered = x1 <= center_x <= x2 and y1 <= center_y <= y2
        if in_target > 0 or target_centered:
            supported_labels.add(label)
        components.append(
            {
                "label": label,
                "area_pixels": area,
                "bbox_abs_xyxy": bbox,
                "pixels_in_vlm_bbox": in_target,
                "target_centered": target_centered,
            }
        )
    if not supported_labels:
        return mask_bool, {
            "status": "noop",
            "reason": "no_component_supported_by_vlm_bbox",
            "component_count": len(components),
        }
    retained_labels = set(supported_labels)
    excluded = []
    for component in components:
        label = int(component["label"])
        if label in supported_labels:
            continue
        if int(component["area_pixels"]) >= minimum_significant_area:
            excluded.append(component)
            continue
        retained_labels.add(label)
    if not excluded:
        return mask_bool, {
            "status": "noop",
            "component_count": len(components),
            "supported_component_count": len(supported_labels),
        }
    retained = np.isin(labels, list(retained_labels))
    return retained, {
        "status": "filtered",
        "component_count": len(components),
        "supported_component_count": len(supported_labels),
        "minimum_significant_area_pixels": minimum_significant_area,
        "excluded_components": excluded,
    }


def bbox_area(b):
    if b is None:
        return 0
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_intersection(a, b):
    if a is None or b is None:
        return 0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def bbox_iou(a, b):
    inter = bbox_intersection(a, b)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center(b):
    if b is None:
        return None
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_similarity(a, b, w, h):
    ca = bbox_center(a)
    cb = bbox_center(b)
    if ca is None or cb is None:
        return 0.0
    dx = (ca[0] - cb[0]) / w
    dy = (ca[1] - cb[1]) / h
    dist = (dx * dx + dy * dy) ** 0.5
    return max(0.0, 1.0 - dist)


def expand_bbox(bbox, w, h, pad_ratio=MERGE_BBOX_PAD_RATIO):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    # First-pass VLM boxes are coarse hints. A small image-relative floor
    # absorbs ordinary localization error without permitting remote fallback.
    pad_x = max(
        MERGE_MIN_PAD_PIXELS,
        int(round((x2 - x1) * pad_ratio)),
        int(round(w * LOCAL_MIN_PAD_IMAGE_RATIO)),
    )
    pad_y = max(
        MERGE_MIN_PAD_PIXELS,
        int(round((y2 - y1) * pad_ratio)),
        int(round(h * LOCAL_MIN_PAD_IMAGE_RATIO)),
    )
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    ]


def bbox_contains_point(bbox, point):
    if bbox is None or point is None:
        return False
    x1, y1, x2, y2 = bbox
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_coverage(container_bbox, item_bbox):
    area = bbox_area(item_bbox)
    if area <= 0:
        return 0.0
    return bbox_intersection(container_bbox, item_bbox) / area


def bbox_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def bbox_gap(a, b):
    if a is None or b is None:
        return float("inf")
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def mask_coverage_in_bbox(mask_bool, bbox):
    if bbox is None:
        return 0.0
    total = int(mask_bool.sum())
    if total <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    inside = int(mask_bool[y1:y2, x1:x2].sum())
    return inside / total


def mask_overlap_metrics(left_mask, right_mask):
    left_area = int(left_mask.sum())
    right_area = int(right_mask.sum())
    intersection = int(np.logical_and(left_mask, right_mask).sum())
    union = left_area + right_area - intersection
    return {
        "intersection_pixels": intersection,
        "iou": intersection / union if union > 0 else 0.0,
        "left_coverage": intersection / left_area if left_area > 0 else 0.0,
        "right_coverage": intersection / right_area if right_area > 0 else 0.0,
    }


def object_scale_pixels(vlm_bbox):
    """Return a resolution-independent area scale for merge-gap checks."""
    return max(1.0, bbox_area(vlm_bbox) ** 0.5)


def anchor_scale_pixels(vlm_bbox):
    """Use the longer bbox side so thin/clipped instances are not over-penalized."""
    if not vlm_bbox:
        return 1.0
    return max(
        1.0,
        float(vlm_bbox[2] - vlm_bbox[0]),
        float(vlm_bbox[3] - vlm_bbox[1]),
    )


def merge_candidate_metrics(obj, cand, w, h):
    vlm_bbox = xyxy_norm_to_abs(obj.get("bbox_2d_norm_xyxy"), w, h)
    expanded_bbox = expand_bbox(vlm_bbox, w, h)
    cand_bbox = cand["mask_bbox_abs_xyxy"]
    center_inside = bbox_contains_point(expanded_bbox, bbox_center(cand_bbox))
    mask_coverage = mask_coverage_in_bbox(cand["mask_bool"], expanded_bbox)
    return {
        "vlm_bbox_abs_xyxy": vlm_bbox,
        "expanded_vlm_bbox_abs_xyxy": expanded_bbox,
        "candidate_bbox_coverage": bbox_coverage(expanded_bbox, cand_bbox),
        "candidate_mask_coverage": mask_coverage,
        "bbox_iou": bbox_iou(vlm_bbox, cand_bbox),
        "center_similarity": center_similarity(vlm_bbox, cand_bbox, w, h),
        "center_inside_expanded_bbox": center_inside,
        "eligible": bool(
            center_inside
            and mask_coverage >= MERGE_MIN_CANDIDATE_MASK_COVERAGE
        ),
    }


def allows_additional_candidates(obj, prompt_group_size):
    """Only collections or a singleton may absorb extra connected masks."""
    if prompt_group_size == 1:
        return True
    category = str(obj.get("category") or "").lower()
    return category.endswith("_group") or category.endswith("_collection")


def maximum_cardinality_local_anchors(eligible_by_candidate, object_count):
    """Return a maximum-cardinality one-to-one local object/candidate match."""
    edges_by_object = {oi: [] for oi in range(object_count)}
    for ci, eligible in eligible_by_candidate.items():
        for quality, oi, metrics in eligible:
            edges_by_object[oi].append((quality, ci, metrics))
    for edges in edges_by_object.values():
        edges.sort(key=lambda item: item[0], reverse=True)

    candidate_match = {}
    object_match = {}

    def augment(oi, seen_candidates):
        for _, ci, metrics in edges_by_object[oi]:
            if ci in seen_candidates:
                continue
            seen_candidates.add(ci)
            previous = candidate_match.get(ci)
            if previous is None or augment(previous[0], seen_candidates):
                candidate_match[ci] = (oi, metrics)
                object_match[oi] = (ci, metrics)
                return True
        return False

    object_order = sorted(
        range(object_count),
        key=lambda oi: (len(edges_by_object[oi]) or 10**9, oi),
    )
    for oi in object_order:
        if edges_by_object[oi]:
            augment(oi, set())
    return object_match


def assign_scale_aware_candidates_uniquely(target_objs, candidates, w, h):
    """Conservative assignment with object-relative geometry.

    One local anchor is selected for every object whenever possible. Extra
    candidates are automatically unioned only when they are largely redundant
    with the current mask. Merely touching or nearby candidates remain
    unclaimed so Stage 03 can decide whether they are a missing object, an
    extension, a duplicate, or a component.
    """
    assignments = {oi: [] for oi in range(len(target_objs))}
    diagnostics = {}

    for oi, obj in enumerate(target_objs):
        vlm_bbox = xyxy_norm_to_abs(obj.get("bbox_2d_norm_xyxy"), w, h)
        diagnostics[oi] = {
            "vlm_bbox_abs_xyxy": vlm_bbox,
            "expanded_vlm_bbox_abs_xyxy": expand_bbox(vlm_bbox, w, h),
            "object_scale_pixels": round(object_scale_pixels(vlm_bbox), 3),
            "anchor_scale_pixels": round(anchor_scale_pixels(vlm_bbox), 3),
            "max_normalized_gap": ADAPTIVE_LOCAL_MAX_NORMALIZED_GAP,
            "eligible_candidate_indices": [],
            "accepted_candidate_indices": [],
            "ambiguous_candidate_indices": [],
            "protected_shared_candidate_indices": [],
            "rejected_far_candidate_indices": [],
            "rejected_disconnected_candidate_indices": [],
            "candidate_decisions": [],
            "rejected_cross_object_overlap": None,
            "forced_fallback_used": False,
        }

    eligible_by_candidate = {}
    for ci, cand in enumerate(candidates):
        eligible = []
        for oi, obj in enumerate(target_objs):
            metrics = merge_candidate_metrics(obj, cand, w, h)
            if not metrics["eligible"]:
                continue
            vlm_bbox = diagnostics[oi]["vlm_bbox_abs_xyxy"]
            vlm_center = bbox_center(vlm_bbox)
            candidate_center = bbox_center(cand["mask_bbox_abs_xyxy"])
            center_distance = (
                (
                    (vlm_center[0] - candidate_center[0]) ** 2
                    + (vlm_center[1] - candidate_center[1]) ** 2
                ) ** 0.5
            )
            normalized_center_distance = (
                center_distance / anchor_scale_pixels(vlm_bbox)
            )
            metrics["normalized_center_distance"] = normalized_center_distance
            if (
                len(target_objs) > 1
                and normalized_center_distance
                > ADAPTIVE_LOCAL_MAX_REPEATED_OBJECT_CENTER_DISTANCE
            ):
                diagnostics[oi]["rejected_far_candidate_indices"].append(ci)
                continue
            quality = (
                metrics["bbox_iou"],
                metrics["candidate_mask_coverage"],
                metrics["candidate_bbox_coverage"],
                metrics["center_similarity"],
                cand["score"],
            )
            eligible.append((quality, oi, metrics))
            diagnostics[oi]["eligible_candidate_indices"].append(ci)
        if eligible:
            eligible_by_candidate[ci] = eligible

    anchor_matches = maximum_cardinality_local_anchors(
        eligible_by_candidate, len(target_objs)
    )
    anchored_candidates = set()
    owned = {oi: [] for oi in range(len(target_objs))}
    for oi, (ci, metrics) in anchor_matches.items():
        owned[oi].append((ci, metrics))
        anchored_candidates.add(ci)

    # Shared unanchored candidates have uncertain ownership. Do not let the
    # largest or coarsest object bbox consume them before Stage 03 review.
    for ci, eligible in eligible_by_candidate.items():
        if ci in anchored_candidates:
            continue
        eligible_for_addition = [
            item for item in eligible
            if allows_additional_candidates(target_objs[item[1]], len(target_objs))
        ]
        if not eligible_for_addition:
            continue
        if len(eligible_for_addition) > 1:
            for _, oi, _ in eligible_for_addition:
                diagnostics[oi]["protected_shared_candidate_indices"].append(ci)
            continue
        _, oi, metrics = eligible_for_addition[0]
        owned[oi].append((ci, metrics))

    for oi, records in owned.items():
        if not records:
            continue

        vlm_bbox = diagnostics[oi]["vlm_bbox_abs_xyxy"]
        scale = object_scale_pixels(vlm_bbox)
        anchor_ci, _ = max(
            records,
            key=lambda item: (
                bbox_iou(vlm_bbox, candidates[item[0]]["mask_bbox_abs_xyxy"]),
                item[1]["candidate_mask_coverage"],
                item[1]["center_similarity"],
                candidates[item[0]]["score"],
            ),
        )
        accepted = [anchor_ci]
        current_mask = candidates[anchor_ci]["mask_bool"].copy()
        current_bbox = candidates[anchor_ci]["mask_bbox_abs_xyxy"]
        diagnostics[oi]["candidate_decisions"].append({
            "candidate_index": anchor_ci,
            "decision": "accepted_anchor",
        })

        pending = sorted(
            (item for item in records if item[0] != anchor_ci),
            key=lambda item: (
                bbox_gap(current_bbox, candidates[item[0]]["mask_bbox_abs_xyxy"])
                / scale,
                -item[1]["candidate_mask_coverage"],
                -candidates[item[0]]["score"],
            ),
        )
        ambiguous = []
        disconnected = []

        for ci, _ in pending:
            cand = candidates[ci]
            gap_pixels = bbox_gap(current_bbox, cand["mask_bbox_abs_xyxy"])
            normalized_gap = gap_pixels / scale
            overlap = mask_overlap_metrics(current_mask, cand["mask_bool"])
            decision = {
                "candidate_index": ci,
                "gap_pixels": round(float(gap_pixels), 3),
                "normalized_gap": round(float(normalized_gap), 6),
                "mask_iou": round(float(overlap["iou"]), 6),
                "candidate_coverage_by_current": round(
                    float(overlap["right_coverage"]), 6
                ),
                "current_coverage_by_candidate": round(
                    float(overlap["left_coverage"]), 6
                ),
            }

            if normalized_gap > ADAPTIVE_LOCAL_MAX_NORMALIZED_GAP:
                decision["decision"] = "defer_scale_gap"
                disconnected.append(ci)
                diagnostics[oi]["candidate_decisions"].append(decision)
                continue

            redundant_overlap = (
                overlap["right_coverage"]
                >= ADAPTIVE_LOCAL_MIN_CANDIDATE_COVERAGE_BY_CURRENT
                or overlap["iou"] >= ADAPTIVE_LOCAL_MIN_MASK_IOU_FOR_REDUNDANT_MERGE
            )
            if not redundant_overlap:
                decision["decision"] = "defer_ambiguous_low_overlap"
                ambiguous.append(ci)
                diagnostics[oi]["candidate_decisions"].append(decision)
                continue

            proposed_bbox = bbox_union(current_bbox, cand["mask_bbox_abs_xyxy"])
            spill = 1.0 - bbox_coverage(
                diagnostics[oi]["expanded_vlm_bbox_abs_xyxy"], proposed_bbox
            )
            if spill > MERGE_MAX_UNION_BBOX_SPILL:
                decision["decision"] = "defer_union_spill"
                decision["union_bbox_spill"] = round(float(spill), 6)
                ambiguous.append(ci)
                diagnostics[oi]["candidate_decisions"].append(decision)
                continue

            decision["decision"] = "accepted_redundant_overlap"
            diagnostics[oi]["candidate_decisions"].append(decision)
            accepted.append(ci)
            current_mask |= cand["mask_bool"]
            current_bbox = proposed_bbox

        assignments[oi] = sorted(accepted)
        diagnostics[oi]["accepted_candidate_indices"] = sorted(accepted)
        diagnostics[oi]["ambiguous_candidate_indices"] = sorted(ambiguous)
        diagnostics[oi]["rejected_disconnected_candidate_indices"] = sorted(
            disconnected
        )

    # Retain the cross-object containment safety gate for broad SAM3 masks.
    def assignment_mask(indices):
        merged = np.zeros((h, w), dtype=bool)
        for ci in indices:
            merged |= candidates[ci]["mask_bool"]
        return merged

    for left in range(len(target_objs)):
        if not assignments[left]:
            continue
        for right in range(left + 1, len(target_objs)):
            if not assignments[right]:
                continue
            left_mask = assignment_mask(assignments[left])
            right_mask = assignment_mask(assignments[right])
            overlap = mask_overlap_metrics(left_mask, right_mask)
            containment = max(overlap["left_coverage"], overlap["right_coverage"])
            if containment < MERGE_MAX_CROSS_OBJECT_CONTAINMENT:
                continue

            def locality_quality(oi, mask):
                bbox = mask_to_bbox_xyxy(mask)
                vlm_bbox = diagnostics[oi]["vlm_bbox_abs_xyxy"]
                expanded_bbox = diagnostics[oi]["expanded_vlm_bbox_abs_xyxy"]
                return (
                    mask_coverage_in_bbox(mask, expanded_bbox),
                    bbox_iou(vlm_bbox, bbox),
                    center_similarity(vlm_bbox, bbox, w, h),
                    -int(mask.sum()),
                )

            left_quality = locality_quality(left, left_mask)
            right_quality = locality_quality(right, right_mask)
            winner, loser = (
                (left, right) if left_quality >= right_quality else (right, left)
            )
            diagnostics[loser]["rejected_cross_object_overlap"] = {
                "winner_object_id": target_objs[winner]["object_id"],
                "containment": round(float(containment), 6),
            }
            diagnostics[loser]["accepted_candidate_indices"] = []
            assignments[loser] = []

    claimed = {
        ci for indices in assignments.values() for ci in indices
    }
    return assignments, claimed, diagnostics


def tensor_or_array_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def draw_overlay(image_np, mask_bool, color=(255, 0, 0)):
    overlay = image_np.copy()
    overlay[mask_bool] = (
        0.55 * overlay[mask_bool] + 0.45 * np.array(color)
    ).astype(np.uint8)
    return overlay


def save_mask_overlay(mask_bool, image_np, base, mask_dir, overlays_dir, color=(255, 0, 0)):
    mask_path = mask_dir / f"{base}_mask.png"
    overlay_path = overlays_dir / f"{base}_overlay.png"
    Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L").save(mask_path)
    Image.fromarray(draw_overlay(image_np, mask_bool, color=color)).save(overlay_path)
    return str(mask_path), str(overlay_path)


def category_has_token(category, token):
    """Match semantic category tokens without confusing e.g. bed and bedside."""
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(category or "")
    )
    return token.lower() in [part for part in normalized.split("_") if part]


def exclude_independent_pillows_from_beds(objects, image_np):
    """Make separately reconstructed pillows exclusive of their bed masks."""
    load_runtime_modules()
    successful_pillows = [
        obj for obj in objects
        if category_has_token(obj.get("category"), "pillow")
        and obj.get("sam3", {}).get("status") == "success"
        and obj.get("sam3", {}).get("mask_path")
        and Path(obj["sam3"]["mask_path"]).is_file()
    ]
    edits = []
    for bed in objects:
        if (
            not category_has_token(bed.get("category"), "bed")
            # Categories such as bed_pillow name an independent pillow, not a
            # bed.  The more specific pillow role must take precedence.
            or category_has_token(bed.get("category"), "pillow")
            or bed.get("sam3", {}).get("status") != "success"
        ):
            continue
        sam3 = dict(bed["sam3"])
        mask_path = Path(sam3["mask_path"])
        if not mask_path.is_file():
            continue
        before = np.array(Image.open(mask_path).convert("L")) > 0
        after = before.copy()
        excluded = []
        for pillow in successful_pillows:
            if pillow is bed or pillow.get("object_id") == bed.get("object_id"):
                continue
            pillow_mask = np.array(Image.open(pillow["sam3"]["mask_path"]).convert("L")) > 0
            overlap = int(np.logical_and(after, pillow_mask).sum())
            if overlap:
                after[pillow_mask] = False
                excluded.append({"pillow_object_id": pillow["object_id"], "pixels_removed": overlap})
        if not excluded:
            continue
        if not after.any():
            raise RuntimeError(f"pillow exclusion emptied bed mask: {bed['object_id']}")
        Image.fromarray(after.astype(np.uint8) * 255, mode="L").save(mask_path)
        overlay_path = Path(sam3["overlay_path"])
        Image.fromarray(draw_overlay(image_np, after)).save(overlay_path)
        before_area = int(before.sum())
        sam3["area_pixels"] = int(after.sum())
        sam3["mask_bbox_abs_xyxy"] = mask_to_bbox_xyxy(after)
        sam3["manual_mask_edit"] = {
            "action": "exclude_independent_pillow_masks",
            "reason": "Independent pillows must not be reconstructed as part of a bed.",
            "excluded_pillows": excluded,
        }
        bed["sam3"] = sam3
        edits.append({
            "bed_object_id": bed["object_id"],
            "before_area_pixels": before_area,
            "after_area_pixels": int(after.sum()),
            "excluded_pillows": excluded,
        })
    return edits


def rebuild_scene_overlay(objects, image_np):
    """Render the combined overlay from final post-processed mask files."""
    combined = image_np.copy()
    rng = np.random.default_rng(42)
    for obj in objects:
        sam3 = obj.get("sam3", {})
        mask_path = sam3.get("mask_path")
        if sam3.get("status") != "success" or not mask_path or not Path(mask_path).is_file():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        combined[mask] = (0.55 * combined[mask] + 0.45 * color).astype(np.uint8)
    return combined


def candidate_sort_key(cand):
    b = cand["mask_bbox_abs_xyxy"]
    c = bbox_center(b)
    if c is None:
        return (999999, 999999)
    return (c[1], c[0])


def object_sort_key(obj):
    order = obj.get("visual_order") or {}
    return (order.get("row", 999), order.get("col", 999), obj.get("object_id", ""))


def category_level_prompt(category):
    """Convert a repeated registry category into one reusable SAM3 prompt."""
    prompt = str(category or "").strip().replace("_", " ")
    for suffix in (" group", " collection"):
        if prompt.endswith(suffix):
            prompt = prompt[:-len(suffix)].strip()
            break
    return prompt


def build_prompt_groups(objects):
    """Group only identical declared SAM3 queries.

    sam3_prompt is the semantic contract produced by the inventory stage.
    Different prompts under one category can deliberately distinguish a
    collection (for example books) from a compound instance (for example
    stacked books). Collapsing them to a category word changes the object
    requested from SAM3 and can turn a collection into unrelated part masks.
    Exact duplicate prompts still share one call; the category is only a
    fallback for incomplete inventory records.
    """
    groups = defaultdict(list)
    for obj in objects:
        prompt = str(obj.get("sam3_prompt") or "").strip()
        if not prompt:
            prompt = category_level_prompt(obj.get("category")) or "object"
        groups[prompt].append(obj)
    return groups


def load_scene_config(scene_config_path):
    config_path = Path(scene_config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"scene_config not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    scene_name = data.get("scene_name") or config_path.parent.parent.name
    run_root = Path(data["run_root"]).expanduser().resolve()
    input_image = Path(data["input_image"]).expanduser().resolve()
    stages = data.get("stages") or {}
    vlm_first_stage = stages.get("vlm_first", "01_vlm_first")
    sam3_stage = stages.get("sam3", "02_sam3")

    output_dir = run_root / sam3_stage
    return {
        "scene_name": scene_name,
        "run_root": run_root,
        "input_image": input_image,
        "vlm_objects_json": run_root / vlm_first_stage / "vlm_objects.json",
        "vlm_run_info_json": run_root / vlm_first_stage / "vlm_first_run_info.json",
        "output_dir": output_dir,
        "masks_dir": output_dir / "masks",
        "overlays_dir": output_dir / "overlays",
        "unassigned_candidates_dir": output_dir / "unassigned_candidates",
        "debug_dir": output_dir / "debug_prompt_candidates",
        "sam3_objects_json": output_dir / "sam3_objects.json",
        "sam3_overlay": output_dir / "sam3_overlay.png",
        "run_info_json": output_dir / "sam3_run_info.json",
        "scene_config_path": config_path,
    }


def load_runtime_config(args):
    if args.scene_config:
        return load_scene_config(args.scene_config)
    raise ValueError("--scene_config must be provided.")


def ensure_runtime_dirs(config):
    for path in [
        config["output_dir"],
        config["masks_dir"],
        config["overlays_dir"],
        config["unassigned_candidates_dir"],
        config["debug_dir"],
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_run_info(config, object_count_input, object_count_output):
    return {
        "stage": "02_sam3",
        "scene_name": config["scene_name"],
        "run_root": str(config["run_root"]),
        "input_image": str(config["input_image"]),
        "vlm_objects_json": str(config["vlm_objects_json"]),
        "vlm_first_run_info": str(config["vlm_run_info_json"]),
        "output_dir": str(config["output_dir"]),
        "sam3_objects_json": str(config["sam3_objects_json"]),
        "sam3_overlay": str(config["sam3_overlay"]),
        "object_count_input": object_count_input,
        "object_count_output": object_count_output,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def print_dry_run(config):
    print("scene_name:", config["scene_name"])
    print("input_image:", config["input_image"])
    print("vlm_objects_json:", config["vlm_objects_json"])
    print("output_dir:", config["output_dir"])
    print("masks_dir:", config["masks_dir"])
    print("overlays_dir:", config["overlays_dir"])
    print("unassigned_candidates_dir:", config["unassigned_candidates_dir"])
    print("sam3_objects_json:", config["sam3_objects_json"])
    print("sam3_overlay:", config["sam3_overlay"])
    print("run_info_json:", config["run_info_json"])


def main():
    args = parse_args()
    try:
        config = load_runtime_config(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run_config:
        print_dry_run(config)
        return

    load_runtime_modules()
    ensure_runtime_dirs(config)

    image = Image.open(config["input_image"]).convert("RGB")
    img_np = np.array(image).astype(np.uint8)
    h, w = img_np.shape[:2]

    registry = json.loads(config["vlm_objects_json"].read_text(encoding="utf-8"))
    objects = canonicalize_objects(registry.get("objects", []))
    assignment_policy = "adaptive_local"

    print("image:", config["input_image"])
    print("image size:", (w, h))
    print("vlm json:", config["vlm_objects_json"])
    print("num VLM objects:", len(objects))
    print("assignment policy:", assignment_policy)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    if device == "cpu":
        print("WARNING: 当前是 CPU，建议切到有卡模式。")

    processor = Sam3Processor.from_pretrained(DEFAULT_MODEL_DIR)
    model = Sam3Model.from_pretrained(DEFAULT_MODEL_DIR).to(device)
    model.eval()

    groups = build_prompt_groups(objects)

    print("\n=== PROMPT GROUPS ===")
    for prompt, group_objects in groups.items():
        print(f"{prompt!r}: {len(group_objects)} -> {[o['object_id'] for o in group_objects]}")

    updated_by_id = {}
    all_overlay = img_np.copy()
    unassigned_candidates = []
    capture_debug_candidates = bool(config.get("capture_debug_candidates", True))
    capture_unassigned_candidates = bool(
        config.get("capture_unassigned_candidates", True)
    )
    # Bulk raw candidates are expensive.  Ambiguous candidates are a small
    # exception because a later ownership resolver needs their exact masks.
    capture_ambiguous_candidate_evidence = bool(
        config.get("capture_ambiguous_candidate_evidence", True)
    )
    # A same-prompt group with two or more Pass-B additions is the only
    # place where an extra unassigned candidate can close a repeated-instance
    # count (for example a third stack of plates on one cart).
    capture_post_delta_multiplicity_candidates = bool(
        config.get("capture_post_delta_multiplicity_candidates", True)
    )
    rng = np.random.default_rng(42)

    for prompt, group_objs in groups.items():
        print("\n==============================")
        print(f"PROMPT GROUP: {prompt!r}, objects={len(group_objs)}")
        print("==============================")

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=SCORE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])

        print(f"raw SAM3 candidates: masks={len(masks)}, scores={len(scores)}")

        candidates = []
        for mi, mask in enumerate(masks):
            mask_np = tensor_or_array_to_numpy(mask)
            mask_bool = mask_np > 0
            area = int(mask_bool.sum())
            if area < MIN_AREA:
                continue

            score = float(scores[mi].detach().cpu()) if torch.is_tensor(scores[mi]) else float(scores[mi])
            mask_bbox = mask_to_bbox_xyxy(mask_bool)

            candidates.append({
                "candidate_id": mi,
                "mask_bool": mask_bool,
                "score": score,
                "area": area,
                "mask_bbox_abs_xyxy": mask_bbox,
            })

        candidates = sorted(candidates, key=candidate_sort_key)
        print("valid candidates:", len(candidates))

        for ci, cand in enumerate(candidates):
            if capture_debug_candidates:
                dbg_base = f"{safe_name(prompt)}_cand_{ci:03d}"
                save_mask_overlay(
                    cand["mask_bool"],
                    img_np,
                    dbg_base,
                    config["debug_dir"],
                    config["debug_dir"],
                )
            print(
                f"  cand {ci:03d}: score={cand['score']:.4f}, "
                f"area={cand['area']}, bbox={cand['mask_bbox_abs_xyxy']}"
            )

        if len(candidates) == 0:
            for obj in group_objs:
                obj_out = dict(obj)
                obj_out["sam3"] = {
                    "status": "failed",
                    "reason": "no_candidates_from_text_prompt",
                    "prompt_used": prompt,
                    "selection_method": f"sam3_prompt_group_{assignment_policy}",
                }
                updated_by_id[obj["object_id"]] = obj_out
                print(f"  {obj['object_id']} failed: no candidates")
            continue

        instance_objs = []
        merge_objs = sorted(group_objs, key=object_sort_key)
        unit_label = "SCALE-AWARE"
        failure_reason = "no_local_candidate"
        locality_field = "scale_aware_locality"
        selection_method = "scale_aware_conservative_candidate_assignment"
        merge_assignments, assigned_cands, merge_diagnostics = (
            assign_scale_aware_candidates_uniquely(
                merge_objs, candidates, w, h
            )
        )

        ambiguity_evidence_indices = {
            int(candidate_index)
            for diagnostic in merge_diagnostics.values()
            for candidate_index in (
                diagnostic.get("ambiguous_candidate_indices", [])
                or diagnostic.get("protected_shared_candidate_indices", [])
            )
        }
        repeated_pass_b_addition_group = (
            capture_post_delta_multiplicity_candidates
            and has_repeated_pass_b_additions(group_objs)
        )

        for oi, obj in enumerate(merge_objs):
            obj_id = obj["object_id"]
            vlm_bbox_abs = xyxy_norm_to_abs(obj.get("bbox_2d_norm_xyxy"), w, h)
            selected = merge_assignments[oi]

            if not selected:
                obj_out = dict(obj)
                obj_out["sam3"] = {
                    "status": "failed",
                    "reason": failure_reason,
                    "prompt_used": prompt,
                    "vlm_bbox_abs_xyxy": vlm_bbox_abs,
                    locality_field: merge_diagnostics[oi],
                    "selection_method": selection_method,
                }
                updated_by_id[obj_id] = obj_out
                print(f"  {unit_label} {obj_id} failed: no local candidate")
                continue

            merged_mask = np.zeros((h, w), dtype=bool)
            for ci in selected:
                merged_mask |= candidates[ci]["mask_bool"]

            merged_mask, component_locality = retain_target_local_components(
                merged_mask, vlm_bbox_abs
            )
            merged_bbox = mask_to_bbox_xyxy(merged_mask)
            merged_area = int(merged_mask.sum())
            max_score = max(candidates[ci]["score"] for ci in selected)
            iou = bbox_iou(vlm_bbox_abs, merged_bbox)
            if not assignment_has_minimum_local_evidence(iou, max_score):
                for ci in selected:
                    assigned_cands.discard(ci)
                obj_out = dict(obj)
                obj_out["sam3"] = {
                    "status": "failed",
                    "reason": "low_confidence_candidate_outside_vlm_bbox",
                    "prompt_used": prompt,
                    "score": max_score,
                    "vlm_bbox_abs_xyxy": vlm_bbox_abs,
                    "mask_bbox_abs_xyxy": merged_bbox,
                    "vlm_bbox_iou": iou,
                    locality_field: merge_diagnostics[oi],
                    "selection_method": selection_method,
                }
                updated_by_id[obj_id] = obj_out
                print(
                    f"  {unit_label} {obj_id} failed: weak candidate outside VLM bbox "
                    f"(score={max_score:.4f}, IoU={iou:.4f})"
                )
                continue

            base = f"{safe_name(obj_id)}_{safe_name(obj.get('category', 'unknown'))}"
            mask_path, overlay_path = save_mask_overlay(
                merged_mask,
                img_np,
                base,
                config["masks_dir"],
                config["overlays_dir"],
            )

            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            all_overlay[merged_mask] = (
                0.55 * all_overlay[merged_mask] + 0.45 * color
            ).astype(np.uint8)

            obj_out = dict(obj)
            obj_out["sam3"] = {
                "status": "success",
                "warning": None if iou >= LOW_IOU_WARNING_THRESHOLD else "low_iou_with_vlm_bbox",
                "prompt_used": prompt,
                "score": max_score,
                "area_pixels": merged_area,
                "vlm_bbox_abs_xyxy": vlm_bbox_abs,
                "mask_bbox_abs_xyxy": merged_bbox,
                "vlm_bbox_iou": iou,
                "merged_candidate_indices": selected,
                "component_locality": component_locality,
                locality_field: merge_diagnostics[oi],
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "selection_method": selection_method,
            }
            ambiguous = merge_diagnostics[oi].get(
                "ambiguous_candidate_indices", []
            )
            protected = merge_diagnostics[oi].get(
                "protected_shared_candidate_indices", []
            )
            obj_out["sam3"]["merge_review"] = {
                "required": bool(ambiguous or protected),
                "ambiguous_candidate_indices": ambiguous,
                "ambiguous_candidate_ids": [
                    candidate_output_id(prompt, ci) for ci in ambiguous
                ],
                "protected_shared_candidate_indices": protected,
                "protected_shared_candidate_ids": [
                    candidate_output_id(prompt, ci) for ci in protected
                ],
                "resolution_stage": (
                    "03_vlm_check" if ambiguous or protected else None
                ),
            }
            updated_by_id[obj_id] = obj_out

            print(
                f"  {unit_label} {obj_id} -> unique cands {selected}, "
                f"area={merged_area}, IoU={iou:.4f}, bbox={merged_bbox}"
            )

        instance_objs = sorted(instance_objs, key=object_sort_key)
        pairs = []
        for oi, obj in enumerate(instance_objs):
            obj_id = obj["object_id"]
            vlm_bbox_abs = xyxy_norm_to_abs(obj.get("bbox_2d_norm_xyxy"), w, h)

            for ci, cand in enumerate(candidates):
                if ci in assigned_cands:
                    continue

                cand_bbox = cand["mask_bbox_abs_xyxy"]
                iou = bbox_iou(vlm_bbox_abs, cand_bbox)
                center_sim = center_similarity(vlm_bbox_abs, cand_bbox, w, h)
                rank_score = IOU_WEIGHT * iou + CENTER_WEIGHT * center_sim + SCORE_WEIGHT * cand["score"]

                pairs.append({
                    "obj_index": oi,
                    "cand_index": ci,
                    "obj_id": obj_id,
                    "iou": iou,
                    "center_sim": center_sim,
                    "rank_score": rank_score,
                })

        pairs.sort(key=lambda x: x["rank_score"], reverse=True)

        assigned_objs = set()
        assignments = {}

        for pair in pairs:
            oi = pair["obj_index"]
            ci = pair["cand_index"]

            if oi in assigned_objs:
                continue
            if ci in assigned_cands:
                continue

            assigned_objs.add(oi)
            assigned_cands.add(ci)
            assignments[oi] = pair

        for oi, obj in enumerate(instance_objs):
            obj_id = obj["object_id"]
            vlm_bbox_abs = xyxy_norm_to_abs(obj.get("bbox_2d_norm_xyxy"), w, h)
            obj_out = dict(obj)

            if oi not in assignments:
                obj_out["sam3"] = {
                    "status": "failed",
                    "reason": "no_unassigned_candidate_available",
                    "prompt_used": prompt,
                    "vlm_bbox_abs_xyxy": vlm_bbox_abs,
                    "selection_method": "instance_mode_one_to_one_assignment",
                }
                updated_by_id[obj_id] = obj_out
                print(f"  INSTANCE {obj_id} failed: no candidate")
                continue

            assign = assignments[oi]
            ci = assign["cand_index"]
            cand = candidates[ci]

            mask_bool = cand["mask_bool"]
            base = f"{safe_name(obj_id)}_{safe_name(obj.get('category', 'unknown'))}"
            mask_path, overlay_path = save_mask_overlay(
                mask_bool,
                img_np,
                base,
                config["masks_dir"],
                config["overlays_dir"],
            )

            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            all_overlay[mask_bool] = (
                0.55 * all_overlay[mask_bool] + 0.45 * color
            ).astype(np.uint8)

            warning = None if assign["iou"] >= LOW_IOU_WARNING_THRESHOLD else "low_iou_with_vlm_bbox"

            obj_out["sam3"] = {
                "status": "success",
                "warning": warning,
                "prompt_used": prompt,
                "score": cand["score"],
                "area_pixels": cand["area"],
                "vlm_bbox_abs_xyxy": vlm_bbox_abs,
                "mask_bbox_abs_xyxy": cand["mask_bbox_abs_xyxy"],
                "vlm_bbox_iou": assign["iou"],
                "center_similarity": assign["center_sim"],
                "candidate_index_in_prompt_group": ci,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "selection_method": "instance_mode_one_to_one_assignment",
            }
            updated_by_id[obj_id] = obj_out

            print(
                f"  INSTANCE {obj_id} -> cand {ci:03d}, "
                f"score={cand['score']:.4f}, IoU={assign['iou']:.4f}, "
                f"center={assign['center_sim']:.4f}, warning={warning}"
            )

        for ci, cand in enumerate(candidates):
            if ci in assigned_cands:
                continue
            retained_for_ambiguity = (
                capture_ambiguous_candidate_evidence
                and ci in ambiguity_evidence_indices
            )
            retained_for_multiplicity = repeated_pass_b_addition_group
            if (
                not capture_unassigned_candidates
                and not retained_for_ambiguity
                and not retained_for_multiplicity
            ):
                print(f"  UNASSIGNED candidate {ci:03d}: not persisted by storage policy")
                continue

            cand_id = candidate_output_id(prompt, ci)
            base = safe_name(cand_id)
            mask_path, overlay_path = save_mask_overlay(
                cand["mask_bool"],
                img_np,
                base,
                config["unassigned_candidates_dir"],
                config["unassigned_candidates_dir"],
                color=(255, 0, 0),
            )

            candidate_record = {
                "candidate_id": cand_id,
                "source": "sam3_unassigned_candidate",
                "category_hint": prompt.replace(" ", "_"),
                "sam3_prompt": prompt,
                "score": cand["score"],
                "area_pixels": cand["area"],
                "mask_bbox_abs_xyxy": cand["mask_bbox_abs_xyxy"],
                "mask_bbox_norm_xyxy": abs_to_norm_bbox(cand["mask_bbox_abs_xyxy"], w, h),
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "review_status": "pending",
                "retention_reason": (
                    "merge_review_ambiguity"
                    if retained_for_ambiguity and not capture_unassigned_candidates
                    else "post_delta_repeated_instance_closure"
                    if retained_for_multiplicity and not capture_unassigned_candidates
                    else "unassigned_candidate_capture"
                ),
                "note": "SAM3 produced this candidate, but no VLM object was assigned to it. It may be a missed object instance or a duplicate/partial mask.",
            }
            unassigned_candidates.append(candidate_record)

            print(f"  UNASSIGNED {cand_id}: bbox={cand['mask_bbox_abs_xyxy']}, score={cand['score']:.4f}")

    updated_objects = []
    for obj in objects:
        obj_id = obj["object_id"]
        if obj_id in updated_by_id:
            updated_objects.append(updated_by_id[obj_id])
        else:
            obj_out = dict(obj)
            obj_out["sam3"] = {
                "status": "failed",
                "reason": "not_processed",
            }
            updated_objects.append(obj_out)

    bed_pillow_edits = exclude_independent_pillows_from_beds(updated_objects, img_np)

    out_registry = dict(registry)
    out_registry["scene_name"] = config["scene_name"]
    out_registry["image_path"] = str(config["input_image"])
    out_registry["image_size"] = [w, h]
    out_registry["objects"] = updated_objects
    out_registry["unassigned_sam3_candidates"] = unassigned_candidates
    out_registry["manual_bed_pillow_exclusions"] = bed_pillow_edits
    out_registry["sam3_assignment_policy"] = assignment_policy

    write_json(config["sam3_objects_json"], out_registry)
    Image.fromarray(rebuild_scene_overlay(updated_objects, img_np)).save(config["sam3_overlay"])

    run_info = build_run_info(
        config=config,
        object_count_input=len(objects),
        object_count_output=len(updated_objects),
    )
    run_info["assignment_policy"] = assignment_policy
    run_info["registry_version"] = registry.get("image_object_registry_version")
    run_info["manual_bed_pillow_exclusions"] = bed_pillow_edits
    write_json(config["run_info_json"], run_info)

    success_count = sum(1 for o in updated_objects if o.get("sam3", {}).get("status") == "success")
    failed_count = len(updated_objects) - success_count
    low_iou_count = sum(1 for o in updated_objects if o.get("sam3", {}).get("warning") == "low_iou_with_vlm_bbox")

    print("\n=== DONE ===")
    print("success:", success_count)
    print("failed :", failed_count)
    print("low_iou warnings:", low_iou_count)
    print("unassigned candidates:", len(unassigned_candidates))
    print("saved json:", config["sam3_objects_json"])
    print("sam3 overlay:", config["sam3_overlay"])
    print("run info:", config["run_info_json"])
    print("candidate dir:", config["unassigned_candidates_dir"])
    print("debug candidates:", config["debug_dir"])


if __name__ == "__main__":
    main()
