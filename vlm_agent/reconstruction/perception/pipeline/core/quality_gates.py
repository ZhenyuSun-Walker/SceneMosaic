from __future__ import annotations

import copy
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from pipeline.core.inventory_fusion import _object_prompt_map
from pipeline.core.publication import safe_name
from pipeline.core.runtime_io import sha256_file

def _mask_sha256(path: Path) -> str:
    return sha256_file(path)


def build_bounded_candidate_semantic_recall_gate(
    selection_audit: Mapping[str, Any],
    pass_b_actions: Mapping[str, Any],
    final_objects: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless strong bounded recall evidence reaches final output."""

    decisions = {
        str(item.get("candidate_id") or ""): dict(item)
        for item in pass_b_actions.get("local_candidate_decisions") or []
        if isinstance(item, Mapping)
    }
    expected: list[dict[str, Any]] = []
    for raw in selection_audit.get("selected") or []:
        if not isinstance(raw, Mapping):
            continue
        candidate_id = str(raw.get("candidate_id") or "")
        namespace = str(raw.get("source_namespace") or "")
        prompt_terms = {
            token
            for token in re.sub(
                r"[^a-z0-9]+", " ", str(raw.get("prompt") or "").lower()
            ).split()
            if token
        }
        bbox = raw.get("bbox_norm_xyxy")
        try:
            score = float(raw.get("score"))
            top_edge_limit_met = (
                isinstance(bbox, Sequence)
                and not isinstance(bbox, (str, bytes))
                and len(bbox) == 4
                and float(bbox[3]) <= 0.10
            )
        except (TypeError, ValueError):
            score = -1.0
            top_edge_limit_met = False
        reason = None
        if (
            namespace == "top_zone"
            and bool(prompt_terms & {"light", "lighting", "lamp"})
            and score >= 0.65
            and top_edge_limit_met
        ):
            reason = "high_confidence_top_edge_light"
        if reason:
            expected.append(
                {
                    "candidate_id": candidate_id,
                    "source_namespace": namespace,
                    "reason": reason,
                    "score": score,
                    "bbox_norm_xyxy": list(bbox),
                    "source_mask_path": str(raw.get("source_mask_path") or ""),
                }
            )
    accepted_ids = sorted(
        item["candidate_id"]
        for item in expected
        if decisions.get(item["candidate_id"], {}).get("decision")
        == "bind_to_addition"
    )
    rejected_ids = sorted(
        item["candidate_id"]
        for item in expected
        if decisions.get(item["candidate_id"], {}).get("decision")
        == "reject"
    )
    # A bounded top-zone candidate may duplicate an existing Pass-A object
    # whose failed mask was repaired through a Pass-B update.  In that case a
    # `reject` means "already represented", not "omit this light".  Accept it
    # only with direct mask evidence, not bbox proximity: the candidate must
    # be nearly contained by one successful final lighting mask.
    resolved_existing_ids: list[str] = []
    resolved_existing_matches: list[dict[str, Any]] = []
    for item in expected:
        candidate_id = item["candidate_id"]
        decision = decisions.get(candidate_id, {})
        action = str(decision.get("decision") or "")
        if (
            item["reason"] != "high_confidence_top_edge_light"
            or action not in {"reject", "replace_existing"}
            or not final_objects
        ):
            continue
        candidate_path = Path(str(item.get("source_mask_path") or ""))
        if not candidate_path.is_file():
            continue
        try:
            candidate_mask = (
                np.array(Image.open(candidate_path).convert("L")) > 0
            )
        except (OSError, ValueError):
            continue
        candidate_area = int(candidate_mask.sum())
        if candidate_area < 32:
            continue
        for final_obj in final_objects:
            if not isinstance(final_obj, Mapping):
                continue
            sam3 = final_obj.get("sam3") or {}
            final_path = Path(str(sam3.get("mask_path") or ""))
            label = " ".join(
                str(final_obj.get(field) or "")
                for field in (
                    "category",
                    "sam3_prompt",
                    "instance_name",
                    "description",
                    "object_type",
                )
            ).lower()
            if (
                str(sam3.get("status") or "") != "success"
                or not final_path.is_file()
                or not ({"light", "lighting", "lamp"} & set(label.split()))
            ):
                continue
            if action == "replace_existing":
                target_source_id = str(decision.get("target_object_id") or "")
                has_target_source = any(
                    isinstance(source, Mapping)
                    and str(source.get("pass") or "").upper() == "A"
                    and str(source.get("source_id") or "") == target_source_id
                    for source in final_obj.get("proposal_sources") or []
                )
                binding_candidate_ids = {
                    str(candidate.get("candidate_id") or "")
                    for candidate in (
                        (final_obj.get("local_candidate_binding") or {})
                        .get("candidates")
                        or []
                    )
                    if isinstance(candidate, Mapping)
                }
                if (
                    not has_target_source
                    or candidate_id not in binding_candidate_ids
                ):
                    continue
            try:
                final_mask = np.array(Image.open(final_path).convert("L")) > 0
            except (OSError, ValueError):
                continue
            if final_mask.shape != candidate_mask.shape:
                continue
            intersection = int(np.logical_and(candidate_mask, final_mask).sum())
            union = int(np.logical_or(candidate_mask, final_mask).sum())
            candidate_coverage = intersection / candidate_area
            iou = intersection / union if union else 0.0
            if candidate_coverage < 0.98 or iou < 0.50:
                continue
            resolved_existing_ids.append(candidate_id)
            resolved_existing_matches.append(
                {
                    "candidate_id": candidate_id,
                    "final_object_id": str(final_obj.get("object_id") or ""),
                    "candidate_coverage": round(candidate_coverage, 6),
                    "mask_iou": round(iou, 6),
                    "resolution": (
                        "reviewed_local_candidate_bound_to_existing_object"
                        if action == "replace_existing"
                        else "already_represented_by_final_lighting_mask"
                    ),
                }
            )
            break
    resolved_ids = set(accepted_ids) | set(resolved_existing_ids)
    unresolved = [
        {
            **item,
            "actual_decision": decisions.get(item["candidate_id"], {}).get(
                "decision", "missing"
            ),
        }
        for item in expected
        if item["candidate_id"] not in resolved_ids
    ]
    hard_errors = []
    if unresolved:
        hard_errors.append(
            {
                "type": "bounded_candidate_semantic_recall_unresolved",
                "candidate_ids": [
                    item["candidate_id"] for item in unresolved
                ],
                "details": unresolved,
            }
        )
    return {
        "schema_version": "scenemosaic_bounded_candidate_semantic_recall_gate",
        "status": "pass" if not hard_errors else "blocked",
        "expected_accept_count": len(expected),
        "accepted_count": len(accepted_ids),
        "accepted_candidate_ids": accepted_ids,
        "rejected_candidate_ids": rejected_ids,
        "resolved_existing_candidate_ids": sorted(resolved_existing_ids),
        "resolved_existing_matches": resolved_existing_matches,
        "unresolved": unresolved,
        "hard_errors": hard_errors,
    }




def build_container_content_completeness_gate(
    recall_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require every explicit fruit plan to complete its container mask."""

    if recall_audit is None:
        return {
            "schema_version": "scenemosaic_fruit_container_completeness_gate",
            "status": "disabled",
            "plan_count": 0,
            "completed_source_object_ids": [],
            "unresolved": [],
            "hard_errors": [],
        }
    completed_ids = {
        str(value)
        for value in recall_audit.get("completed_source_object_ids") or []
    }
    unresolved = []
    for plan in recall_audit.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        source_id = str(plan.get("target_source_object_id") or "")
        if source_id not in completed_ids:
            unresolved.append(
                {
                    "source_object_id": source_id,
                    "prompt": str(plan.get("prompt") or ""),
                    "resolution": "fruit_not_recalled_or_container_not_completed",
                }
            )
    hard_errors = []
    if unresolved:
        hard_errors.append(
            {
                "type": "explicit_fruit_container_completion_unresolved",
                "source_object_ids": [
                    item["source_object_id"] for item in unresolved
                ],
                "details": unresolved,
            }
        )
    return {
        "schema_version": "scenemosaic_fruit_container_completeness_gate",
        "status": "pass" if not hard_errors else "blocked",
        "plan_count": len(recall_audit.get("plans") or []),
        "completed_source_object_ids": sorted(completed_ids),
        "unresolved": unresolved,
        "hard_errors": hard_errors,
    }

def _record_category_text(record: Mapping[str, Any]) -> str:
    """Return semantic category text, including the old-manifest prompt fallback."""

    return " ".join(
        str(record.get(key) or "")
        for key in ("category", "sam3_prompt")
    )


def _tiny_ceiling_light_extent_tolerance(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Accept coarse VLM localization only for a unique tiny top light."""

    category_tokens = set(
        re.sub(
            r"[^a-z0-9]+", " ", _record_category_text(record).lower()
        ).split()
    )
    if not (
        "ceiling" in category_tokens
        and bool(category_tokens & {"light", "lighting", "downlight"})
    ):
        return None
    decision = decision or {}
    reason = str(decision.get("reason") or "").lower()
    visually_verified_top_edge = (
        str(decision.get("decision") or "") == "pass"
        and "cover" in reason
        and "ceiling light" in reason
        and "top edge" in reason
    )
    if (
        str(record.get("status") or "") != "success"
        or (
            float(record.get("score") or 0.0) < 0.65
            and not visually_verified_top_edge
        )
        or bool(record.get("merge_review_required"))
        or int(record.get("component_count") or 0) != 1
    ):
        return None
    try:
        norm = [float(value) for value in record.get("bbox_norm_xyxy")]
        claimed_box = [
            float(value) for value in record.get("claimed_bbox_abs_xyxy")
        ]
        actual_box = [
            float(value) for value in record.get("actual_mask_bbox_abs_xyxy")
        ]
    except (TypeError, ValueError):
        return None
    if len(norm) != 4 or len(claimed_box) != 4 or len(actual_box) != 4:
        return None
    claimed_width = claimed_box[2] - claimed_box[0]
    claimed_height = claimed_box[3] - claimed_box[1]
    actual_width = actual_box[2] - actual_box[0]
    actual_height = actual_box[3] - actual_box[1]
    if (
        norm[3] > 0.17
        or max(claimed_width, claimed_height) > 64.0
        or max(actual_width, actual_height) > 64.0
        or min(claimed_width, claimed_height, actual_width, actual_height)
        <= 0.0
    ):
        return None
    area_ratio = (actual_width * actual_height) / (
        claimed_width * claimed_height
    )
    claimed_center = (
        (claimed_box[0] + claimed_box[2]) / 2.0,
        (claimed_box[1] + claimed_box[3]) / 2.0,
    )
    actual_center = (
        (actual_box[0] + actual_box[2]) / 2.0,
        (actual_box[1] + actual_box[3]) / 2.0,
    )
    center_scale = math.dist(claimed_center, actual_center) / max(
        math.hypot(claimed_width, claimed_height),
        math.hypot(actual_width, actual_height),
        1.0,
    )
    if not (0.10 <= area_ratio <= 4.0 and center_scale <= 1.25):
        return None
    return {
        "policy": (
            "visually_verified_tiny_top_ceiling_light"
            if visually_verified_top_edge
            else "unique_tiny_top_ceiling_light"
        ),
        "minimum_score": 0.50 if visually_verified_top_edge else 0.65,
        "maximum_bottom_normalized_y": 0.17,
        "maximum_side_pixels": 64,
        "maximum_center_distance_scale": 1.25,
        "area_ratio": round(area_ratio, 6),
        "center_distance_scale": round(center_scale, 6),
    }


def _repeated_recessed_ceiling_light_row_extent_tolerance(
    record: Mapping[str, Any],
    peer_records: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Allow deeper perspective rows only with a matching recessed-light peer."""

    category_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            _record_category_text(record).lower().replace("_", " "),
        )
    )
    reason = str(decision.get("reason") or "").lower()
    if not (
        str(decision.get("decision") or "") == "pass"
        and {"ceiling", "light"} <= category_tokens
        and bool(category_tokens & {"recessed", "downlight"})
        and "light" in reason
        and bool(re.search(r"\b(?:match|matches|matched|cover|covered|visible)\b", reason))
        and not re.search(
            r"\b(?:missing|misses|incomplete|partial|incorrect|extra|leak)\w*\b",
            reason,
        )
        and str(record.get("status") or "") == "success"
        and float(record.get("score") or 0.0) >= 0.75
        and not bool(record.get("merge_review_required"))
        and int(record.get("component_count") or 0) == 1
    ):
        return None

    def geometry(item: Mapping[str, Any]):
        try:
            norm = [float(value) for value in item["bbox_norm_xyxy"]]
            claimed = [
                float(value) for value in item["claimed_bbox_abs_xyxy"]
            ]
            actual = [
                float(value) for value in item["actual_mask_bbox_abs_xyxy"]
            ]
        except (KeyError, TypeError, ValueError):
            return None
        if not (
            len(norm) == 4
            and len(claimed) == 4
            and len(actual) == 4
            and 0.0 <= norm[0] < norm[2] <= 1.0
            and 0.0 <= norm[1] < norm[3] <= 1.0
            and claimed[0] < claimed[2]
            and claimed[1] < claimed[3]
            and actual[0] < actual[2]
            and actual[1] < actual[3]
        ):
            return None
        claimed_width = claimed[2] - claimed[0]
        claimed_height = claimed[3] - claimed[1]
        actual_width = actual[2] - actual[0]
        actual_height = actual[3] - actual[1]
        if (
            norm[3] > 0.25
            or max(
                claimed_width,
                claimed_height,
                actual_width,
                actual_height,
            )
            > 64.0
        ):
            return None
        claimed_center = (
            (claimed[0] + claimed[2]) / 2.0,
            (claimed[1] + claimed[3]) / 2.0,
        )
        actual_center = (
            (actual[0] + actual[2]) / 2.0,
            (actual[1] + actual[3]) / 2.0,
        )
        scale = max(
            math.hypot(claimed_width, claimed_height),
            math.hypot(actual_width, actual_height),
            1.0,
        )
        area_ratio = (actual_width * actual_height) / (
            claimed_width * claimed_height
        )
        center_scale = math.dist(claimed_center, actual_center) / scale
        if not (0.25 <= area_ratio <= 1.50 and center_scale <= 0.35):
            return None
        return {
            "norm_center_x": (norm[0] + norm[2]) / 2.0,
            "norm_center_y": (norm[1] + norm[3]) / 2.0,
            "actual_width": actual_width,
            "actual_height": actual_height,
            "area_ratio": area_ratio,
            "center_distance_scale": center_scale,
        }

    current = geometry(record)
    if current is None:
        return None
    peer_ids = []
    for peer in peer_records:
        if str(peer.get("evidence_id") or "") == str(
            record.get("evidence_id") or ""
        ):
            continue
        peer_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(peer.get("category") or "").lower().replace("_", " "),
            )
        )
        if not (
            {"ceiling", "light"} <= peer_tokens
            and bool(peer_tokens & {"recessed", "downlight"})
            and str(peer.get("status") or "") == "success"
            and float(peer.get("score") or 0.0) >= 0.75
            and not bool(peer.get("merge_review_required"))
            and int(peer.get("component_count") or 0) == 1
        ):
            continue
        peer_geometry = geometry(peer)
        if peer_geometry is None:
            continue
        width_similarity = min(
            current["actual_width"], peer_geometry["actual_width"]
        ) / max(current["actual_width"], peer_geometry["actual_width"])
        height_similarity = min(
            current["actual_height"], peer_geometry["actual_height"]
        ) / max(current["actual_height"], peer_geometry["actual_height"])
        if not (
            abs(
                current["norm_center_y"]
                - peer_geometry["norm_center_y"]
            )
            <= 0.015
            and abs(
                current["norm_center_x"]
                - peer_geometry["norm_center_x"]
            )
            >= 0.10
            and width_similarity >= 0.65
            and height_similarity >= 0.65
        ):
            continue
        peer_ids.append(str(peer.get("evidence_id") or ""))

    if not peer_ids:
        return None
    return {
        "policy": "repeated_recessed_ceiling_light_row",
        "minimum_score": 0.75,
        "maximum_bottom_normalized_y": 0.25,
        "maximum_vertical_center_delta_normalized": 0.015,
        "minimum_horizontal_center_separation_normalized": 0.10,
        "minimum_dimension_similarity": 0.65,
        "maximum_side_pixels": 64,
        "maximum_center_distance_scale": 0.35,
        "area_ratio": round(current["area_ratio"], 6),
        "center_distance_scale": round(
            current["center_distance_scale"], 6
        ),
        "peer_evidence_ids": sorted(peer_ids),
    }


def _small_flat_media_extent_tolerance(
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Accept a centered, unique thin book/notebook mask inside a coarse box."""

    category_tokens = set(
        re.sub(
            r"[^a-z0-9]+", " ", _record_category_text(record).lower()
        ).split()
    )
    if not bool(
        category_tokens & {"book", "notebook", "paper", "document", "magazine"}
    ):
        return None
    if (
        str(record.get("status") or "") != "success"
        or float(record.get("score") or 0.0) < 0.70
        or bool(record.get("merge_review_required"))
        or int(record.get("component_count") or 0) != 1
    ):
        return None
    try:
        claimed = [
            float(value) for value in record.get("claimed_bbox_abs_xyxy")
        ]
        actual = [
            float(value) for value in record.get("actual_mask_bbox_abs_xyxy")
        ]
    except (TypeError, ValueError):
        return None
    if len(claimed) != 4 or len(actual) != 4:
        return None
    claimed_width = claimed[2] - claimed[0]
    claimed_height = claimed[3] - claimed[1]
    actual_width = actual[2] - actual[0]
    actual_height = actual[3] - actual[1]
    dimensions = (claimed_width, claimed_height, actual_width, actual_height)
    if (
        min(dimensions) <= 0.0
        or max(claimed_width, actual_width) > 192.0
        or max(claimed_height, actual_height) > 96.0
        or min(claimed_width, claimed_height) > 48.0
        or min(actual_width, actual_height) > 48.0
    ):
        return None
    width_ratio = actual_width / claimed_width
    height_ratio = actual_height / claimed_height
    area_ratio = width_ratio * height_ratio
    claimed_center = (
        (claimed[0] + claimed[2]) / 2.0,
        (claimed[1] + claimed[3]) / 2.0,
    )
    actual_center = (
        (actual[0] + actual[2]) / 2.0,
        (actual[1] + actual[3]) / 2.0,
    )
    center_scale = math.dist(claimed_center, actual_center) / max(
        math.hypot(claimed_width, claimed_height),
        math.hypot(actual_width, actual_height),
        1.0,
    )
    if not (
        0.40 <= area_ratio <= 2.50
        and 0.50 <= width_ratio <= 2.00
        and 0.50 <= height_ratio <= 2.00
        and center_scale <= 0.20
    ):
        return None
    return {
        "policy": "unique_small_flat_media_localization",
        "minimum_score": 0.70,
        "maximum_width_pixels": 192,
        "maximum_height_pixels": 96,
        "maximum_thin_side_pixels": 48,
        "maximum_center_distance_scale": 0.20,
        "width_ratio": round(width_ratio, 6),
        "height_ratio": round(height_ratio, 6),
        "area_ratio": round(area_ratio, 6),
        "center_distance_scale": round(center_scale, 6),
    }


def _thin_attached_shelf_extent_tolerance(
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Allow a thin shelf board mask only when an attached light was suppressed."""

    category_tokens = set(
        re.findall(r"[a-z0-9]+", _record_category_text(record).lower())
    )
    if "shelf" not in category_tokens:
        return None
    try:
        claimed = [float(value) for value in record["claimed_bbox_abs_xyxy"]]
        actual = [float(value) for value in record["actual_mask_bbox_abs_xyxy"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(claimed) != 4 or len(actual) != 4:
        return None
    claimed_width = claimed[2] - claimed[0]
    claimed_height = claimed[3] - claimed[1]
    actual_width = actual[2] - actual[0]
    actual_height = actual[3] - actual[1]
    if min(claimed_width, claimed_height, actual_width, actual_height) <= 0:
        return None
    width_ratio = actual_width / claimed_width
    height_ratio = actual_height / claimed_height
    center_x_delta_scale = abs(
        (actual[0] + actual[2] - claimed[0] - claimed[2]) / 2.0
    ) / claimed_width
    if not (
        claimed_width >= 160
        and claimed_height <= 64
        and claimed_width / claimed_height >= 6.0
        and 0.90 <= width_ratio <= 1.10
        and 0.65 <= height_ratio <= 1.10
        and center_x_delta_scale <= 0.05
    ):
        return None
    return {
        "policy": "thin_shelf_with_locally_suppressed_attached_light",
        "minimum_height_ratio": 0.65,
        "minimum_width_ratio": 0.90,
        "maximum_width_ratio": 1.10,
        "minimum_claimed_width_pixels": 160,
        "maximum_claimed_height_pixels": 64,
        "minimum_aspect_ratio": 6.0,
        "width_ratio": round(width_ratio, 6),
        "height_ratio": round(height_ratio, 6),
        "center_x_delta_scale": round(center_x_delta_scale, 6),
    }


def _thin_wall_shelf_span_extent_tolerance(
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Accept a verified, very thin wall-shelf span despite a loose VLM height.

    This is deliberately limited to a one-piece wall shelf whose SAM3 mask
    spans the claimed width.  It does not relax cabinets, open shelving, or
    shelves with contents.
    """

    if (
        str(decision.get("decision") or "") != "pass"
        or str(record.get("status") or "") != "success"
        or bool(record.get("merge_review_required"))
    ):
        return None
    category_tokens = set(
        re.findall(r"[a-z0-9]+", _record_category_text(record).lower())
    )
    if not {"wall", "shelf"} <= category_tokens:
        return None
    try:
        score = float(record.get("score") or 0.0)
        component_count = int(record.get("component_count"))
        claimed = [float(value) for value in record["claimed_bbox_abs_xyxy"]]
        actual = [float(value) for value in record["actual_mask_bbox_abs_xyxy"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(claimed) != 4 or len(actual) != 4 or component_count != 1:
        return None
    claimed_width = claimed[2] - claimed[0]
    claimed_height = claimed[3] - claimed[1]
    actual_width = actual[2] - actual[0]
    actual_height = actual[3] - actual[1]
    if min(claimed_width, claimed_height, actual_width, actual_height) <= 0:
        return None
    width_ratio = actual_width / claimed_width
    center_x_delta_scale = abs(
        (actual[0] + actual[2] - claimed[0] - claimed[2]) / 2.0
    ) / claimed_width
    decision_reason = str(decision.get("reason") or "").lower()
    if not (
        score >= 0.80
        and claimed_width >= 160
        and claimed_height <= 96
        and claimed_width / claimed_height >= 4.0
        and actual_height <= min(32.0, 0.35 * claimed_height)
        and 0.90 <= width_ratio <= 1.10
        and center_x_delta_scale <= 0.05
        and ("shelf" in decision_reason or "span" in decision_reason)
    ):
        return None
    return {
        "policy": "verified_thin_wall_shelf_span",
        "minimum_score": 0.80,
        "minimum_claimed_width_pixels": 160,
        "maximum_claimed_height_pixels": 96,
        "maximum_actual_height_pixels": 32,
        "minimum_width_ratio": 0.90,
        "maximum_width_ratio": 1.10,
        "width_ratio": round(width_ratio, 6),
        "center_x_delta_scale": round(center_x_delta_scale, 6),
    }


def _simple_visible_object_extent_tolerance(
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Allow a loose claim box only for explicitly verified simple objects."""

    if (
        str(decision.get("decision") or "") != "pass"
        or str(record.get("status") or "") != "success"
    ):
        return None
    try:
        coverage = float(record["claimed_bbox_coverage_by_mask_bbox"])
        component_count = int(record["component_count"])
        claimed = [float(value) for value in record["claimed_bbox_abs_xyxy"]]
        actual = [float(value) for value in record["actual_mask_bbox_abs_xyxy"]]
    except (KeyError, TypeError, ValueError):
        return None
    if coverage < 0.35 or component_count != 1:
        return None
    if not (
        len(claimed) == 4
        and len(actual) == 4
        and claimed[0] < claimed[2]
        and claimed[1] < claimed[3]
        and actual[0] < actual[2]
        and actual[1] < actual[3]
    ):
        return None
    claimed_width = claimed[2] - claimed[0]
    claimed_height = claimed[3] - claimed[1]
    actual_width = actual[2] - actual[0]
    actual_height = actual[3] - actual[1]
    actual_center = (
        (actual[0] + actual[2]) / 2.0,
        (actual[1] + actual[3]) / 2.0,
    )
    pad_x = max(4.0, 0.05 * claimed_width)
    pad_y = max(4.0, 0.05 * claimed_height)
    if not (
        claimed[0] - pad_x <= actual_center[0] <= claimed[2] + pad_x
        and claimed[1] - pad_y <= actual_center[1] <= claimed[3] + pad_y
        and actual_width <= 1.15 * claimed_width
        and actual_height <= 1.15 * claimed_height
    ):
        return None

    category_tokens = set(
        re.findall(r"[a-z0-9]+", _record_category_text(record).lower())
    )
    reason = str(decision.get("reason") or "").lower()
    if re.search(
        r"\b(?:missing|misses|omits?|omitted|incomplete|partial|partially|"
        r"incorrect|leaks?|leaking|extra)\b|"
        r"\bnot\s+(?:fully\s+)?covered\b",
        reason,
    ):
        return None
    merge_review_required = bool(record.get("merge_review_required"))
    verified_separate_pillow = (
        bool(category_tokens & {"pillow", "cushion"})
        and float(record.get("score") or 0.0) >= 0.80
        and coverage >= 0.55
        and "mask cover" in reason
        and bool(
            re.search(
                r"(?:one|visible|separate|individual)",
                reason,
            )
        )
        and bool(re.search(r"(?:pillow|cushion)", reason))
    )
    if merge_review_required and not verified_separate_pillow:
        return None
    policy = None
    if verified_separate_pillow:
        policy = "visually_verified_separate_pillow_mask"
    elif (
        category_tokens & {"pillow", "cushion"}
        and reason
        == (
            "local assembly policy keeps a pillow excluded from the bed "
            "mask as an independent inventory instance"
        )
        and float(record.get("score") or 0.0) >= 0.90
        and coverage >= 0.55
    ):
        policy = "high_confidence_independent_pillow_preserved_from_bed_assembly"
    elif (
        category_tokens & {"pillow", "cushion"}
        and re.search(r"(?:pillow|cushion).{0,32}cover", reason)
    ):
        policy = "simple_pillow_explicitly_covered"
    elif (
        "lamp" in category_tokens
        and all(token in reason for token in ("base", "shade", "cover"))
    ):
        policy = "lamp_base_and_shade_explicitly_covered"
    elif (
        {"soap", "dispenser"} <= category_tokens
        and reason == "mask matches visible soap dispenser"
        and float(record.get("score") or 0.0) >= 0.90
        and coverage >= 0.65
    ):
        policy = "high_confidence_visible_soap_dispenser"
    elif (
        ({"bedside", "table"} <= category_tokens or "nightstand" in category_tokens)
        and "visible" in reason
        and "cover" in reason
        and bool({"top", "front", "body"} & set(re.findall(r"[a-z0-9]+", reason)))
    ):
        policy = "occluded_bedside_table_visible_surfaces_explicitly_covered"
    if policy is None:
        return None
    return {
        "policy": f"{policy}",
        "minimum_coverage": 0.35,
        "component_count": component_count,
        "actual_to_claimed_width_ratio": round(actual_width / claimed_width, 6),
        "actual_to_claimed_height_ratio": round(actual_height / claimed_height, 6),
    }


def _verified_lighting_fixture_extent_tolerance(
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bounded tolerance for a verified sconce or floor lamp with a loose VLM box.

    This permits only the two transparent, one-piece lighting cases observed in
    the blind-scene regression.  It remains stricter than the ordinary
    simple-object tolerance: category, score, coverage, component count, and
    the Pass-B visual justification must all agree.  It is not a general
    relaxation for lights, assemblies, or incomplete masks.
    """

    if (
        str(decision.get("decision") or "") != "pass"
        or str(record.get("status") or "") != "success"
        or bool(record.get("merge_review_required"))
    ):
        return None
    try:
        score = float(record.get("score") or 0.0)
        coverage = float(record["claimed_bbox_coverage_by_mask_bbox"])
        component_count = int(record["component_count"])
        claimed = [float(value) for value in record["claimed_bbox_abs_xyxy"]]
        actual = [float(value) for value in record["actual_mask_bbox_abs_xyxy"]]
    except (KeyError, TypeError, ValueError):
        return None
    if not (score >= 0.90 and 0.60 <= coverage < 0.75 and component_count == 1):
        return None
    if not (
        len(claimed) == 4
        and len(actual) == 4
        and claimed[0] < claimed[2]
        and claimed[1] < claimed[3]
        and actual[0] < actual[2]
        and actual[1] < actual[3]
    ):
        return None
    claimed_width = claimed[2] - claimed[0]
    claimed_height = claimed[3] - claimed[1]
    actual_width = actual[2] - actual[0]
    actual_height = actual[3] - actual[1]
    actual_center = (
        (actual[0] + actual[2]) / 2.0,
        (actual[1] + actual[3]) / 2.0,
    )
    pad_x = max(4.0, 0.05 * claimed_width)
    pad_y = max(4.0, 0.05 * claimed_height)
    if not (
        claimed[0] - pad_x <= actual_center[0] <= claimed[2] + pad_x
        and claimed[1] - pad_y <= actual_center[1] <= claimed[3] + pad_y
        and actual_width <= 1.15 * claimed_width
        and actual_height <= 1.15 * claimed_height
    ):
        return None
    category_tokens = set(
        re.findall(r"[a-z0-9]+", _record_category_text(record).lower())
    )
    reason = str(decision.get("reason") or "").lower()
    if re.search(
        r"\b(?:missing|misses|omits?|omitted|incomplete|partial|partially|"
        r"incorrect|leaks?|leaking|extra)\b|"
        r"\bnot\s+(?:fully\s+)?covered\b",
        reason,
    ):
        return None
    policy = None
    if {"wall", "sconce"} <= category_tokens and (
        "wall sconce" in reason
        and "body" in reason
        and ("match" in reason or "covered" in reason)
    ):
        policy = "high_confidence_wall_sconce_body_verified"
    elif {"floor", "lamp"} <= category_tokens and all(
        token in reason for token in ("floor lamp", "shade", "legs", "cover")
    ):
        policy = "high_confidence_floor_lamp_shade_and_legs_verified"
    if policy is None:
        return None
    return {
        "policy": f"{policy}",
        "minimum_score": 0.90,
        "minimum_coverage": 0.60,
        "component_count": component_count,
        "actual_to_claimed_width_ratio": round(actual_width / claimed_width, 6),
        "actual_to_claimed_height_ratio": round(actual_height / claimed_height, 6),
    }


def build_assigned_evidence_extent_gate(
    assigned_manifest: Mapping[str, Any],
    evidence_decisions: Sequence[Mapping[str, Any]],
    *,
    minimum_claimed_bbox_coverage: float = 0.75,
    completed_container_content_source_ids: Sequence[str] = (),
    completed_dedicated_attachment_source_ids: Sequence[str] = (),
    attached_lighting_host_source_ids: Sequence[str] = (),
    downstream_spatial_rescue_source_ids: Sequence[str] = (),
    completed_fixed_component_source_ids: Sequence[str] = (),
    locally_rebound_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Do not accept a mask whose extent materially misses its own claim."""

    if not 0.0 < minimum_claimed_bbox_coverage <= 1.0:
        raise ValueError("minimum claimed-bbox coverage must be in (0, 1]")
    decisions = {
        str(item.get("evidence_id") or ""): dict(item)
        for item in evidence_decisions
        if isinstance(item, Mapping)
    }
    records = [
        item
        for item in assigned_manifest.get("records") or []
        if isinstance(item, Mapping)
    ]
    low_coverage = []
    tolerated_tiny_localization = []
    tolerated_repeated_ceiling_light_row_localization = []
    tolerated_small_flat_media_localization = []
    tolerated_thin_attached_shelf_localization = []
    tolerated_simple_visible_object_localization = []
    tolerated_verified_lighting_fixture_localization = []
    downstream_spatial_rescues = []
    completed_container_assemblies = []
    completed_dedicated_attachment_assemblies = []
    completed_fixed_component_assemblies = []
    locally_rebound_masks = []
    completed_container_sources = {
        str(value) for value in completed_container_content_source_ids
    }
    completed_dedicated_attachment_sources = {
        str(value) for value in completed_dedicated_attachment_source_ids
    }
    downstream_spatial_rescue_sources = {
        str(value) for value in downstream_spatial_rescue_source_ids
    }
    completed_fixed_component_sources = {
        str(value) for value in completed_fixed_component_source_ids
    }
    locally_rebound_sources = {
        str(value) for value in locally_rebound_source_ids
    }
    attached_lighting_hosts = {
        str(value) for value in attached_lighting_host_source_ids
    }
    unresolved = []
    for record in records:
        value = record.get("claimed_bbox_coverage_by_mask_bbox")
        if value is None:
            continue
        coverage = float(value)
        if coverage >= minimum_claimed_bbox_coverage:
            continue
        evidence_id = str(record.get("evidence_id") or "")
        decision_record = decisions.get(evidence_id, {})
        item = {
            "evidence_id": evidence_id,
            "source_object_id": str(record.get("source_object_id") or ""),
            "category": str(record.get("category") or record.get("sam3_prompt") or ""),
            "coverage": coverage,
            "decision": str(decision_record.get("decision") or "missing"),
        }
        tiny_tolerance = _tiny_ceiling_light_extent_tolerance(
            record, decision_record
        )
        repeated_ceiling_row_tolerance = (
            _repeated_recessed_ceiling_light_row_extent_tolerance(
                record, records, decision_record
            )
        )
        flat_media_tolerance = _small_flat_media_extent_tolerance(record)
        attached_shelf_tolerance = (
            _thin_attached_shelf_extent_tolerance(record)
            if item["source_object_id"] in attached_lighting_hosts
            else None
        )
        simple_visible_tolerance = _simple_visible_object_extent_tolerance(
            record, decision_record
        )
        verified_lighting_fixture_tolerance = (
            _verified_lighting_fixture_extent_tolerance(record, decision_record)
        )
        thin_wall_shelf_tolerance = _thin_wall_shelf_span_extent_tolerance(
            record, decision_record
        )
        if tiny_tolerance is not None:
            item["tiny_localization_tolerance"] = tiny_tolerance
        if repeated_ceiling_row_tolerance is not None:
            item["repeated_ceiling_light_row_tolerance"] = (
                repeated_ceiling_row_tolerance
            )
        if flat_media_tolerance is not None:
            item["small_flat_media_localization_tolerance"] = (
                flat_media_tolerance
            )
        if attached_shelf_tolerance is not None:
            item["thin_attached_shelf_localization_tolerance"] = (
                attached_shelf_tolerance
            )
        if simple_visible_tolerance is not None:
            item["simple_visible_object_localization_tolerance"] = (
                simple_visible_tolerance
            )
        if thin_wall_shelf_tolerance is not None:
            item["thin_wall_shelf_span_tolerance"] = thin_wall_shelf_tolerance
        if verified_lighting_fixture_tolerance is not None:
            item["verified_lighting_fixture_localization_tolerance"] = (
                verified_lighting_fixture_tolerance
            )
        low_coverage.append(item)
        if item["decision"] == "pass":
            if item["source_object_id"] in downstream_spatial_rescue_sources:
                item["downstream_resolution"] = (
                    "strict_bbox_positive_sam3_spatial_rescue"
                )
                downstream_spatial_rescues.append(item)
            elif item["source_object_id"] in locally_rebound_sources:
                item["local_candidate_rebinding_resolution"] = (
                    "verified_final_local_candidate_replace"
                )
                locally_rebound_masks.append(item)
            elif item["source_object_id"] in completed_fixed_component_sources:
                item["fixed_component_resolution"] = (
                    "fixed_component_absorbed_into_functional_host_assembly"
                )
                completed_fixed_component_assemblies.append(item)
            elif item["source_object_id"] in completed_container_sources:
                item["container_content_resolution"] = (
                    "fruit_merged_into_completed_container_mask"
                )
                completed_container_assemblies.append(item)
            elif item["source_object_id"] in completed_dedicated_attachment_sources:
                item["dedicated_attachment_resolution"] = (
                    "attached_content_absorbed_into_explicit_host_assembly"
                )
                completed_dedicated_attachment_assemblies.append(item)
            elif attached_shelf_tolerance is not None:
                tolerated_thin_attached_shelf_localization.append(item)
            elif tiny_tolerance is not None:
                tolerated_tiny_localization.append(item)
            elif repeated_ceiling_row_tolerance is not None:
                tolerated_repeated_ceiling_light_row_localization.append(
                    item
                )
            elif flat_media_tolerance is not None:
                tolerated_small_flat_media_localization.append(item)
            elif thin_wall_shelf_tolerance is not None:
                tolerated_thin_attached_shelf_localization.append(item)
            elif verified_lighting_fixture_tolerance is not None:
                tolerated_verified_lighting_fixture_localization.append(item)
            elif simple_visible_tolerance is not None:
                tolerated_simple_visible_object_localization.append(item)
            else:
                unresolved.append(item)
    hard_errors = []
    if unresolved:
        hard_errors.append(
            {
                "type": "assigned_mask_claimed_extent_mismatch_passed",
                "minimum_claimed_bbox_coverage": minimum_claimed_bbox_coverage,
                "evidence_ids": [
                    item["evidence_id"] for item in unresolved
                ],
                "details": unresolved,
            }
        )
    return {
        "schema_version": "scenemosaic_assigned_evidence_extent_gate",
        "status": "pass" if not hard_errors else "blocked",
        "minimum_claimed_bbox_coverage": minimum_claimed_bbox_coverage,
        "low_coverage": low_coverage,
        "downstream_spatial_rescues": downstream_spatial_rescues,
        "locally_rebound_masks": locally_rebound_masks,
        "tolerated_tiny_localization": tolerated_tiny_localization,
        "tolerated_repeated_ceiling_light_row_localization": (
            tolerated_repeated_ceiling_light_row_localization
        ),
        "tolerated_small_flat_media_localization": (
            tolerated_small_flat_media_localization
        ),
        "tolerated_thin_attached_shelf_localization": (
            tolerated_thin_attached_shelf_localization
        ),
        "tolerated_localization": (
            tolerated_tiny_localization
            + tolerated_repeated_ceiling_light_row_localization
            + tolerated_small_flat_media_localization
            + tolerated_thin_attached_shelf_localization
            + tolerated_verified_lighting_fixture_localization
            + tolerated_simple_visible_object_localization
            + downstream_spatial_rescues
            + locally_rebound_masks
        ),
        "tolerated_simple_visible_object_localization": (
            tolerated_simple_visible_object_localization
        ),
        "tolerated_verified_lighting_fixture_localization": (
            tolerated_verified_lighting_fixture_localization
        ),
        "completed_container_content_assemblies": (
            completed_container_assemblies
        ),
        "completed_fixed_component_assemblies": (
            completed_fixed_component_assemblies
        ),
        "completed_dedicated_attachment_assemblies": (
            completed_dedicated_attachment_assemblies
        ),
        "unresolved_pass_count": len(unresolved),
        "hard_errors": hard_errors,
    }


def build_locally_validated_assigned_evidence_overflow(
    assigned_manifest: Mapping[str, Any],
    final_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve only high-confidence, source-traceable evidence-panel overflow.

    A panel capacity limit must not reject an object that is already locally
    validated by SAM3 with a tight VLM-box match.  The bypass is intentionally
    unavailable for low-score, loose, merged, or missing-mask objects.
    A 0.80 bbox IoU remains conservative here because the object must also
    have score >= 0.90, a present mask, and no merge-review ambiguity.
    """

    suspicious = [
        str(value)
        for value in assigned_manifest.get("suspicious_overflow_object_ids") or []
        if str(value)
    ]
    by_source: dict[str, Mapping[str, Any]] = {}
    for obj in final_objects:
        for source in obj.get("proposal_sources") or []:
            if (
                isinstance(source, Mapping)
                and str(source.get("pass") or "").upper() == "A"
                and str(source.get("source_id") or "")
            ):
                by_source[str(source["source_id"])] = obj
    accepted = []
    unresolved = []
    for source_id in suspicious:
        obj = by_source.get(source_id)
        sam3 = dict((obj or {}).get("sam3") or {})
        merge_review = sam3.get("merge_review") or {}
        try:
            score = float(sam3.get("score") or 0.0)
            bbox_iou = float(sam3.get("vlm_bbox_iou") or 0.0)
        except (TypeError, ValueError):
            score, bbox_iou = 0.0, 0.0
        mask_path = Path(str(sam3.get("mask_path") or ""))
        valid = (
            obj is not None
            and sam3.get("status") == "success"
            and score >= 0.90
            and bbox_iou >= 0.80
            and not bool(merge_review.get("required"))
            and mask_path.is_file()
        )
        record = {
            "source_object_id": source_id,
            "proposal_id": str((obj or {}).get("object_id") or "") or None,
            "score": round(score, 6),
            "vlm_bbox_iou": round(bbox_iou, 6),
            "merge_review_required": bool(merge_review.get("required")),
            "mask_present": mask_path.is_file(),
        }
        if valid:
            record["resolution"] = "local_sam3_high_confidence_bbox_match"
            accepted.append(record)
        else:
            unresolved.append(record)
    return {
        "schema_version": "scenemosaic_assigned_evidence_overflow_local_validation",
        "status": "pass" if not unresolved else "blocked",
        "minimum_score": 0.90,
        "minimum_vlm_bbox_iou": 0.80,
        "accepted": accepted,
        "unresolved": unresolved,
        "hard_errors": ([] if not unresolved else [{
            "type": "assigned_sam3_evidence_budget_overflow",
            "object_ids": [item["source_object_id"] for item in unresolved],
        }]),
    }


def resolve_assigned_evidence_semantic_gate_after_spatial_rescue(
    assigned_manifest: Mapping[str, Any],
    semantic_gate: Mapping[str, Any] | None,
    final_objects: Sequence[Mapping[str, Any]],
    downstream_spatial_rescue_source_ids: Sequence[str],
    excluded_unmaterialized_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Clear an earlier uncertain evidence item only after a final spatial rescue.

    The Pass-B decision remains in the audit trail.  This is deliberately
    limited to source objects whose later bbox-first rescue actually produced
    a final SAM3 mask; it never converts an arbitrary uncertain decision into
    a pass.
    """

    unresolved_ids = [
        str(value)
        for value in ((semantic_gate or {}).get("unresolved_evidence_ids") or [])
        if str(value)
    ]
    evidence_source = {
        str(item.get("evidence_id") or ""): str(item.get("source_object_id") or "")
        for item in assigned_manifest.get("selected") or []
        if isinstance(item, Mapping)
    }
    rescued_sources = {str(value) for value in downstream_spatial_rescue_source_ids if str(value)}
    excluded_sources = {
        str(value) for value in excluded_unmaterialized_source_ids if str(value)
    }
    final_by_source: dict[str, Mapping[str, Any]] = {}
    for obj in final_objects:
        for source in obj.get("proposal_sources") or []:
            if (
                isinstance(source, Mapping)
                and str(source.get("pass") or "").upper() == "A"
                and str(source.get("source_id") or "")
            ):
                final_by_source[str(source["source_id"])] = obj
    resolved = []
    still_unresolved = []
    for evidence_id in unresolved_ids:
        source_id = evidence_source.get(evidence_id, "")
        obj = final_by_source.get(source_id)
        sam3 = (obj or {}).get("sam3") or {}
        mask_path = Path(str(sam3.get("mask_path") or ""))
        if source_id in excluded_sources:
            resolved.append({
                "evidence_id": evidence_id,
                "source_object_id": source_id,
                "proposal_id": None,
                "resolution": "unmaterialized_claim_excluded_after_failed_recovery",
            })
        elif (
            source_id in rescued_sources
            and sam3.get("status") == "success"
            and mask_path.is_file()
        ):
            resolved.append({
                "evidence_id": evidence_id,
                "source_object_id": source_id,
                "proposal_id": str((obj or {}).get("object_id") or ""),
                "resolution": "downstream_bbox_spatial_rescue",
            })
        else:
            still_unresolved.append(evidence_id)
    return {
        "schema_version": "scenemosaic_assigned_evidence_spatial_recovery",
        "status": "pass" if not still_unresolved else "blocked",
        "resolved": resolved,
        "unresolved_evidence_ids": still_unresolved,
    }




def build_functional_assembly_gate(
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Detect sink/faucet or fitted-dishwasher pieces split from support."""

    standalone_categories = {
        "sink",
        "kitchen_sink",
        "vessel_sink",
        "washbasin",
        "basin",
        "faucet",
        "kitchen_faucet",
        "bathroom_faucet",
    }
    support_terms = {"counter", "countertop", "vanity", "cabinet", "worktable"}
    live = [
        dict(item)
        for item in objects
        if (item.get("sam3") or {}).get("status") == "success"
    ]
    components = [
        item
        for item in live
        if str(item.get("category") or "").lower() in standalone_categories
    ]
    supports = [
        item
        for item in live
        if (
            {
                token
                for token in re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    str(item.get("category") or "").lower(),
                ).split()
                if token
            }
            & support_terms
        )
    ]
    dishwashers = [
        item
        for item in live
        if str(item.get("category") or "").lower()
        in {"dishwasher", "built_in_dishwasher", "fitted_dishwasher"}
    ]
    sink_supports = [
        item
        for item in supports
        if "sink"
        in {
            token
            for token in re.sub(
                r"[^a-z0-9]+",
                " ",
                " ".join(
                    str(item.get(key) or "").lower()
                    for key in ("category", "sam3_prompt", "instance_name")
                ),
            ).split()
            if token
        }
    ]

    def related(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_box = _bbox(left)
        right_box = _bbox(right)
        ix = max(
            0.0,
            min(left_box[2], right_box[2])
            - max(left_box[0], right_box[0]),
        )
        iy = max(
            0.0,
            min(left_box[3], right_box[3])
            - max(left_box[1], right_box[1]),
        )
        if ix > 0.0 and iy > 0.0:
            return True
        return math.dist(_anchor(left), _anchor(right)) <= 0.12

    unresolved = []
    seen_pairs = set()
    for component in components:
        neighbors = [
            other
            for other in [*components, *supports]
            if other.get("object_id") != component.get("object_id")
            and related(component, other)
        ]
        for other in neighbors:
            pair = tuple(
                sorted(
                    (
                        str(component.get("object_id") or ""),
                        str(other.get("object_id") or ""),
                    )
                )
            )
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            unresolved.append(
                {
                    "component_object_id": str(
                        component.get("object_id") or ""
                    ),
                    "component_category": str(
                        component.get("category") or ""
                    ),
                    "nearby_object_id": str(other.get("object_id") or ""),
                    "nearby_category": str(other.get("category") or ""),
                }
            )
    hard_errors = []
    if unresolved:
        hard_errors.append(
            {
                "type": "functional_sink_assembly_split",
                "pairs": unresolved,
            }
        )
    dishwasher_splits = []
    for dishwasher in dishwashers:
        for support in sink_supports:
            if not related(dishwasher, support):
                continue
            dishwasher_splits.append(
                {
                    "dishwasher_object_id": str(
                        dishwasher.get("object_id") or ""
                    ),
                    "sink_support_object_id": str(
                        support.get("object_id") or ""
                    ),
                    "sink_support_category": str(
                        support.get("category") or ""
                    ),
                }
            )
    if dishwasher_splits:
        hard_errors.append(
            {
                "type": "functional_fitted_dishwasher_split",
                "pairs": dishwasher_splits,
            }
        )
    owner_overlap_pairs = []
    checked_pairs = 0
    if sink_supports:
        import numpy as np
        from PIL import Image

        def semantic_tokens(item: Mapping[str, Any]) -> set[str]:
            return set(
                re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    " ".join(
                        str(item.get(key) or "").lower()
                        for key in (
                            "category",
                            "sam3_prompt",
                            "instance_name",
                        )
                    ),
                ).split()
            )

        def load_mask(item: Mapping[str, Any]):
            mask_path = Path(
                str((item.get("sam3") or {}).get("mask_path") or "")
            )
            if not mask_path.is_file():
                return None
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            return mask if mask.any() else None

        rivals = [
            item
            for item in live
            if not semantic_tokens(item) & {"sink", "faucet", "basin"}
            and (
                bool(semantic_tokens(item) & {"range", "stove"})
                or "counter" in semantic_tokens(item)
            )
        ]
        for sink_support in sink_supports:
            sink_mask = load_mask(sink_support)
            if sink_mask is None:
                continue
            for rival in rivals:
                if rival.get("object_id") == sink_support.get("object_id"):
                    continue
                rival_mask = load_mask(rival)
                if rival_mask is None or rival_mask.shape != sink_mask.shape:
                    continue
                checked_pairs += 1
                intersection = int(np.logical_and(sink_mask, rival_mask).sum())
                if intersection < 30:
                    continue
                rival_area = int(rival_mask.sum())
                sink_area = int(sink_mask.sum())
                rival_coverage = intersection / rival_area
                if rival_coverage < 0.50:
                    continue
                owner_overlap_pairs.append(
                    {
                        "sink_support_object_id": str(
                            sink_support.get("object_id") or ""
                        ),
                        "rival_object_id": str(
                            rival.get("object_id") or ""
                        ),
                        "rival_category": str(rival.get("category") or ""),
                        "intersection_pixels": intersection,
                        "rival_coverage": round(rival_coverage, 6),
                        "sink_coverage": round(
                            intersection / sink_area,
                            6,
                        ),
                    }
                )
        if owner_overlap_pairs:
            hard_errors.append(
                {
                    "type": "functional_sink_assembly_owner_overlap",
                    "minimum_rival_coverage": 0.50,
                    "pairs": owner_overlap_pairs,
                }
            )
    return {
        "schema_version": "scenemosaic_functional_assembly_gate",
        "status": "pass" if not hard_errors else "blocked",
        "checked_standalone_component_count": len(components),
        "checked_dishwasher_count": len(dishwashers),
        "unresolved_pairs": unresolved,
        "dishwasher_splits": dishwasher_splits,
        "checked_owner_overlap_pair_count": checked_pairs,
        "owner_overlap_pairs": owner_overlap_pairs,
        "hard_errors": hard_errors,
    }


# These errors mean a Stage 04 package cannot be trusted or materialized.
# All other quality findings remain visible in best-effort delivery metadata,
# but do not prevent reconstruction from proceeding with its own exclusions.
INTEGRITY_ERROR_TYPES = frozenset(
    {
        "formal_outputs_modified",
        "identity_manifest_mismatch",
        "identity_mismatch",
        "invalid_mask_file",
        "local_candidate_evidence_hash_mismatch",
        "mask_shape_mismatch",
        "missing_local_candidate_evidence",
        "missing_output_file",
    }
)


def apply_delivery_quality_policy(
    quality: Mapping[str, Any], *, strict: bool = False
) -> dict[str, Any]:
    """Classify observed findings for strict or best-effort publication.

    Best-effort delivery retains every observed error in the audit. It blocks
    only broken identity or file-integrity facts; SAM3D delivery separately
    excludes objects without usable masks and labels the package degraded.
    """

    result = dict(quality)
    observed = [
        dict(item)
        for item in result.get("hard_errors") or []
        if isinstance(item, Mapping)
    ]
    blocking = (
        list(observed)
        if strict
        else [
            item
            for item in observed
            if str(item.get("type") or "") in INTEGRITY_ERROR_TYPES
        ]
    )
    blocking_ids = {id(item) for item in blocking}
    advisory = [
        item for item in observed if id(item) not in blocking_ids
    ]
    result["observed_hard_errors"] = observed
    result["hard_errors"] = blocking
    result["blocking_errors"] = blocking
    result["advisory_errors"] = advisory
    result["quality_policy"] = {
        "mode": "strict" if strict else "best_effort",
        "integrity_error_types": sorted(INTEGRITY_ERROR_TYPES),
        "observed_hard_error_count": len(observed),
        "blocking_error_count": len(blocking),
        "advisory_error_count": len(advisory),
    }
    result["status"] = "pass" if not blocking else "fail"
    return result


def build_quality_audit(
    final_objects: Sequence[Mapping[str, Any]],
    final_sam3_objects: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate final identity and mask evidence without semantic guessing."""

    import numpy as np
    from PIL import Image

    if isinstance(final_sam3_objects, Mapping):
        sam3_objects = list(final_sam3_objects.get("objects") or [])
    else:
        sam3_objects = list(final_sam3_objects)
    expected_ids = [str(item["object_id"]) for item in final_objects]
    actual_ids = [str(item.get("object_id")) for item in sam3_objects]
    hard_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if actual_ids != expected_ids:
        hard_errors.append(
            {
                "type": "identity_mismatch",
                "expected": expected_ids,
                "actual": actual_ids,
            }
        )
    mapping_ids = [
        str(item.get("proposal_id")) for item in mapping.get("records") or []
    ]
    if mapping_ids != expected_ids:
        hard_errors.append(
            {
                "type": "identity_manifest_mismatch",
                "expected": expected_ids,
                "actual": mapping_ids,
            }
        )

    mask_hashes: dict[str, list[str]] = defaultdict(list)
    mask_digests: dict[str, str] = {}
    mask_arrays: dict[str, Any] = {}
    success_count = 0
    for obj in sam3_objects:
        object_id = str(obj.get("object_id"))
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            hard_errors.append(
                {
                    "type": "sam3_failed",
                    "object_id": object_id,
                    "reason": sam3.get("reason"),
                }
            )
            continue
        success_count += 1
        for key in ("mask_path", "overlay_path"):
            raw_path = sam3.get(key)
            if not raw_path or not Path(raw_path).is_file():
                hard_errors.append(
                    {
                        "type": "missing_output_file",
                        "object_id": object_id,
                        "field": key,
                        "path": raw_path,
                    }
                )
        mask_path = sam3.get("mask_path")
        if mask_path and Path(mask_path).is_file():
            resolved_mask_path = Path(mask_path)
            digest = _mask_sha256(resolved_mask_path)
            mask_hashes[digest].append(object_id)
            mask_digests[object_id] = digest
            try:
                with Image.open(resolved_mask_path) as source_mask:
                    mask_arrays[object_id] = np.array(
                        source_mask.convert("L")
                    ) > 0
            except Exception as exc:
                hard_errors.append(
                    {
                        "type": "invalid_mask_file",
                        "object_id": object_id,
                        "path": str(resolved_mask_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        local_evidence = sam3.get("local_candidate_evidence") or {}
        for evidence in local_evidence.get("candidates") or []:
            evidence_path = Path(str(evidence.get("evidence_mask_path") or ""))
            if not evidence_path.is_file():
                hard_errors.append(
                    {
                        "type": "missing_local_candidate_evidence",
                        "object_id": object_id,
                        "candidate_id": evidence.get("candidate_id"),
                        "path": str(evidence_path),
                    }
                )
                continue
            expected_digest = str(evidence.get("mask_sha256") or "")
            actual_digest = _mask_sha256(evidence_path)
            if not expected_digest or actual_digest != expected_digest:
                hard_errors.append(
                    {
                        "type": "local_candidate_evidence_hash_mismatch",
                        "object_id": object_id,
                        "candidate_id": evidence.get("candidate_id"),
                        "expected": expected_digest or None,
                        "actual": actual_digest,
                    }
                )
        if sam3.get("warning") == "low_iou_with_vlm_bbox":
            warnings.append(
                {
                    "type": "low_iou_with_vlm_bbox",
                    "object_id": object_id,
                    "iou": sam3.get("vlm_bbox_iou"),
                }
            )
        merge_review = sam3.get("merge_review") or {}
        if merge_review.get("required"):
            hard_errors.append(
                {
                    "type": "unresolved_merge_review",
                    "object_id": object_id,
                    "details": merge_review,
                }
            )
    duplicate_masks = [
        {"sha256": digest, "object_ids": ids}
        for digest, ids in mask_hashes.items()
        if len(ids) > 1
    ]
    for duplicate in duplicate_masks:
        hard_errors.append({"type": "identical_masks", **duplicate})

    overlap_pairs: list[dict[str, Any]] = []
    object_ids = list(mask_arrays)
    for left_index, left_id in enumerate(object_ids):
        left_mask = mask_arrays[left_id]
        for right_id in object_ids[left_index + 1 :]:
            right_mask = mask_arrays[right_id]
            if left_mask.shape != right_mask.shape:
                hard_errors.append(
                    {
                        "type": "mask_shape_mismatch",
                        "left_object_id": left_id,
                        "right_object_id": right_id,
                        "left_shape": list(left_mask.shape),
                        "right_shape": list(right_mask.shape),
                    }
                )
                continue
            intersection = int(np.logical_and(left_mask, right_mask).sum())
            if intersection < 30:
                continue
            left_area = int(left_mask.sum())
            right_area = int(right_mask.sum())
            union = left_area + right_area - intersection
            iou = intersection / union if union else 0.0
            left_coverage = intersection / left_area if left_area else 0.0
            right_coverage = intersection / right_area if right_area else 0.0
            containment = max(left_coverage, right_coverage)
            pair = {
                "left_object_id": left_id,
                "right_object_id": right_id,
                "intersection_pixels": intersection,
                "iou": round(iou, 6),
                "left_coverage": round(left_coverage, 6),
                "right_coverage": round(right_coverage, 6),
            }
            if mask_digests.get(left_id) == mask_digests.get(right_id):
                pair["severity"] = "identical_mask"
            elif iou >= 0.55 or containment >= 0.90:
                pair["severity"] = "hard_error"
                hard_errors.append({"type": "high_mask_overlap", **pair})
            elif iou >= 0.15 or containment >= 0.50:
                pair["severity"] = "warning"
                warnings.append({"type": "moderate_mask_overlap", **pair})
            else:
                pair["severity"] = "informational"
            overlap_pairs.append(pair)
    return {
        "object_count": len(sam3_objects),
        "success_count": success_count,
        "hard_errors": hard_errors,
        "warnings": warnings,
        "duplicate_masks": duplicate_masks,
        "mask_overlap_pairs": overlap_pairs,
        "mask_overlap_thresholds": {
            "hard_iou": 0.55,
            "hard_containment": 0.90,
            "warning_iou": 0.15,
            "warning_containment": 0.50,
            "minimum_intersection_pixels": 30,
        },
    }




def _create_contact_sheet(
    objects: Sequence[Mapping[str, Any]],
    output_path: Path,
    columns: int = 4,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cells = []
    for obj in objects:
        overlay_path = Path(str((obj.get("sam3") or {}).get("overlay_path") or ""))
        if not overlay_path.is_file():
            continue
        with Image.open(overlay_path) as source:
            image = source.convert("RGB")
            image.thumbnail((360, 240), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (380, 290), (245, 245, 245))
            cell.paste(image, ((380 - image.width) // 2, 34))
        draw = ImageDraw.Draw(cell)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        label = f"{obj['object_id']}  {obj.get('category', 'unknown')}"
        draw.text((8, 7), label, fill=(0, 0, 0), font=font)
        cells.append(cell)
    if not cells:
        raise RuntimeError("cannot build an empty SAM3 contact sheet")
    rows = math.ceil(len(cells) / columns)
    sheet = Image.new("RGB", (columns * 380, rows * 290), "white")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 380, (index // columns) * 290))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _copy_final_masks(
    *,
    fused_objects: Sequence[Mapping[str, Any]],
    affected_prompts: set[str],
    pass_a_registry: Mapping[str, Any],
    delta_registry: Mapping[str, Any] | None,
    staging_dir: Path,
    image_path: Path,
) -> list[dict[str, Any]]:
    from PIL import Image
    import numpy as np

    staging_masks = staging_dir / "masks"
    staging_overlays = staging_dir / "overlays"
    candidate_evidence_dir = staging_dir / "evidence" / "local_candidates"
    staging_masks.mkdir(parents=True, exist_ok=True)
    staging_overlays.mkdir(parents=True, exist_ok=True)
    candidate_evidence_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image_np = np.array(source.convert("RGB")).astype(np.uint8)
    a_by_id = {
        str(obj["object_id"]): obj
        for obj in pass_a_registry.get("objects") or []
    }
    delta_by_id = {
        str(obj["object_id"]): obj
        for obj in (delta_registry or {}).get("objects") or []
    }
    final_prompt_by_id = _object_prompt_map(fused_objects)
    final_objects: list[dict[str, Any]] = []

    for raw_obj in fused_objects:
        obj = copy.deepcopy(dict(raw_obj))
        object_id = str(obj["object_id"])
        prompt = final_prompt_by_id[object_id]
        binding = obj.get("local_candidate_binding") or {}
        binding_operation = binding.get("operation")
        source_obj = None
        source_kind = None
        if binding_operation != "replace":
            if prompt in affected_prompts:
                source_obj = delta_by_id.get(object_id)
                source_kind = "sam3_delta_affected_prompt_group"
            else:
                a_sources = [
                    str(source["source_id"])
                    for source in obj.get("proposal_sources") or []
                    if source.get("pass") == "A"
                ]
                source_obj = a_by_id.get(a_sources[0]) if len(a_sources) == 1 else None
                source_kind = "sam3_a_reused_prompt_group"

        mask = None
        if binding_operation == "replace":
            source_kind = "vlm_reviewed_local_candidate_replace"
            source_sam3 = {
                "status": "success",
                "warning": None,
                "selection_method": source_kind,
                "prompt_used": prompt,
                "score": max(
                    (float(item.get("score") or 0.0) for item in binding.get("candidates") or []),
                    default=0.0,
                ),
                "merge_review": {
                    "required": False,
                    "resolution_stage": "pass_b_local_candidate_review",
                },
            }
            mask = np.zeros(image_np.shape[:2], dtype=bool)
        elif source_obj is None:
            obj["sam3"] = {
                "status": "failed",
                "reason": "missing_source_segmentation",
                "selection_method": source_kind,
            }
            final_objects.append(obj)
            continue
        else:
            source_sam3 = copy.deepcopy(dict(source_obj.get("sam3") or {}))
            source_sam3["attempt_reuse_source"] = {
                "kind": source_kind,
                "source_object_id": source_obj.get("object_id"),
                "final_prompt_group": prompt,
            }
            if source_sam3.get("status") == "success":
                source_mask = Path(str(source_sam3.get("mask_path") or ""))
                if source_mask.is_file():
                    mask = np.array(Image.open(source_mask).convert("L")) > 0
                else:
                    source_sam3 = {
                        "status": "failed",
                        "reason": "source_mask_missing",
                        "selection_method": source_kind,
                    }

        if binding_operation in {"replace", "union_existing"}:
            if mask is None:
                source_sam3 = {
                    "status": "failed",
                    "reason": "local_candidate_base_mask_missing",
                    "selection_method": source_kind,
                }
            else:
                candidate_ids = []
                candidate_evidence = []
                for candidate in binding.get("candidates") or []:
                    candidate_id = str(candidate.get("candidate_id"))
                    candidate_path = Path(str(candidate.get("mask_path") or ""))
                    if not candidate_path.is_file():
                        source_sam3 = {
                            "status": "failed",
                            "reason": "local_candidate_mask_missing",
                            "candidate_id": candidate_id,
                            "selection_method": source_kind,
                        }
                        mask = None
                        break
                    candidate_mask = np.array(
                        Image.open(candidate_path).convert("L")
                    ) > 0
                    if candidate_mask.shape != mask.shape:
                        source_sam3 = {
                            "status": "failed",
                            "reason": "local_candidate_mask_shape_mismatch",
                            "candidate_id": candidate_id,
                            "selection_method": source_kind,
                        }
                        mask = None
                        break
                    mask |= candidate_mask
                    candidate_ids.append(candidate_id)
                    evidence_base = safe_name(candidate_id)
                    evidence_mask_path = (
                        candidate_evidence_dir / f"{evidence_base}_mask.png"
                    )
                    shutil.copy2(candidate_path, evidence_mask_path)
                    source_overlay_path = Path(
                        str(candidate.get("overlay_path") or "")
                    )
                    evidence_overlay_path = None
                    if source_overlay_path.is_file():
                        evidence_overlay_path = (
                            candidate_evidence_dir / f"{evidence_base}_overlay.png"
                        )
                        shutil.copy2(source_overlay_path, evidence_overlay_path)
                    candidate_evidence.append(
                        {
                            "candidate_id": candidate_id,
                            "source_mask_path": str(candidate_path),
                            "evidence_mask_path": str(evidence_mask_path),
                            "mask_sha256": _mask_sha256(evidence_mask_path),
                            "source_overlay_path": (
                                str(source_overlay_path)
                                if source_overlay_path.is_file()
                                else None
                            ),
                            "evidence_overlay_path": (
                                str(evidence_overlay_path)
                                if evidence_overlay_path is not None
                                else None
                            ),
                        }
                    )
                if mask is not None:
                    source_sam3["selection_method"] = (
                        "vlm_reviewed_local_candidate_replace"
                        if binding_operation == "replace"
                        else "vlm_reviewed_local_candidate_union"
                    )
                    source_sam3["local_candidate_evidence"] = {
                        "operation": binding_operation,
                        "candidate_ids": candidate_ids,
                        "candidates": candidate_evidence,
                    }
                    source_sam3["merge_review"] = {
                        "required": False,
                        "resolution_stage": "pass_b_local_candidate_review",
                    }

        if source_sam3.get("status") == "success" and mask is not None:
            base = f"{safe_name(object_id)}_{safe_name(obj.get('category', 'unknown'))}"
            mask_path = staging_masks / f"{base}_mask.png"
            overlay_path = staging_overlays / f"{base}_overlay.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            overlay = image_np.copy()
            overlay[mask] = (
                0.55 * overlay[mask]
                + 0.45 * np.array((255, 0, 0), dtype=np.uint8)
            ).astype(np.uint8)
            Image.fromarray(overlay).save(overlay_path)
            source_sam3["mask_path"] = str(mask_path)
            source_sam3["overlay_path"] = str(overlay_path)
            source_sam3["area_pixels"] = int(mask.sum())
            ys, xs = np.nonzero(mask)
            source_sam3["mask_bbox_abs_xyxy"] = [
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            ]
        obj["sam3"] = source_sam3
        final_objects.append(obj)
    return final_objects
