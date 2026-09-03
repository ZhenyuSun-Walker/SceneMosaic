from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from inventory import pipeline as VLM
from pipeline.refinements import top_zone_candidate_recall as TOP_ZONE
from pipeline.core.runtime_io import atomic_json

BOUNDED_POST_SAM3A_REVIEW_CONFIG = {
    "maximum_candidates": 24,
    "minimum_unassigned_score": 0.60,
    "minimum_area_pixels": 32,
    "maximum_area_fraction": 0.04,
    "successful_mask_exclusion_iou": 0.50,
    "successful_mask_exclusion_candidate_coverage": 0.80,
    "duplicate_iou": 0.65,
    "duplicate_containment": 0.85,
    "reserved_unassigned_slots": 4,
}

def select_bounded_post_sam3a_candidates(
    *,
    image_path: str | Path,
    top_candidates: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one bounded mixed review set without creating semantic objects."""

    from PIL import Image

    cfg = dict(BOUNDED_POST_SAM3A_REVIEW_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown bounded candidate config keys: {sorted(unknown)}")
        cfg.update(config)
    maximum = int(cfg["maximum_candidates"])
    if maximum < 1:
        raise ValueError("maximum_candidates must be positive")
    if int(cfg["reserved_unassigned_slots"]) < 0:
        raise ValueError("reserved_unassigned_slots must be non-negative")

    resolved_image = Path(image_path).expanduser().resolve()
    with Image.open(resolved_image) as source_image:
        width, height = source_image.size
    expected_shape = (height, width)
    image_area = width * height
    minimum_area = int(cfg["minimum_area_pixels"])
    maximum_area = int(math.floor(float(cfg["maximum_area_fraction"]) * image_area))
    if minimum_area < 1 or maximum_area < minimum_area:
        raise ValueError("invalid bounded candidate area limits")

    successful_masks: list[tuple[str, Any]] = []
    for obj in sam3_registry.get("objects") or []:
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            continue
        object_id = str(obj.get("object_id") or "")
        try:
            mask = TOP_ZONE.load_mask(sam3["mask_path"], expected_shape)
        except Exception as exc:
            raise ValueError(
                f"cannot validate successful SAM3 mask for {object_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not mask.any():
            raise ValueError(f"successful SAM3 mask is empty: {object_id}")
        successful_masks.append((object_id, mask))

    eligible: list[dict[str, Any]] = []
    semantic_aliases: list[dict[str, Any]] = []
    validation_rejections: list[dict[str, Any]] = []

    def validate_candidate(raw_value: Mapping[str, Any], namespace: str) -> None:
        raw = dict(raw_value)
        review_id = str(raw.get("candidate_id") or "")
        upstream_id = str(raw.get("source_candidate_id") or review_id)
        prompt = str(
            raw.get("prompt")
            or raw.get("sam3_prompt")
            or raw.get("category_hint")
            or ""
        ).strip()
        rejection = {
            "source_namespace": namespace,
            "source_candidate_id": review_id,
            "source_upstream_candidate_id": upstream_id,
            "prompt": prompt,
            "score": None,
            "area_pixels": None,
            "reasons": [],
        }
        try:
            score = float(raw.get("score"))
        except (TypeError, ValueError):
            score = math.nan
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            rejection["reasons"].append("invalid_score")
        else:
            rejection["score"] = score
            if (
                namespace == "sam3_a_unassigned"
                and score < float(cfg["minimum_unassigned_score"])
            ):
                rejection["reasons"].append(
                    "unassigned_score_below_threshold"
                )
        if not review_id:
            rejection["reasons"].append("missing_source_candidate_id")
        if not prompt:
            rejection["reasons"].append("missing_prompt")

        mask_path = Path(str(raw.get("mask_path") or "")).expanduser().resolve()
        overlay_path = Path(str(raw.get("overlay_path") or "")).expanduser().resolve()
        mask = None
        try:
            mask = TOP_ZONE.load_mask(mask_path, expected_shape)
            if not mask.any():
                rejection["reasons"].append("empty_mask")
        except Exception as exc:
            rejection["reasons"].append(
                f"invalid_mask:{type(exc).__name__}"
            )
        try:
            if not overlay_path.is_file():
                raise FileNotFoundError(overlay_path)
            with Image.open(overlay_path) as overlay:
                if overlay.size != (width, height):
                    raise ValueError("shape_mismatch")
        except Exception as exc:
            rejection["reasons"].append(
                f"invalid_overlay:{type(exc).__name__}"
            )

        if mask is None or not mask.any():
            validation_rejections.append(rejection)
            return
        area = int(mask.sum())
        rejection["area_pixels"] = area
        declared_area = raw.get("area_pixels")
        if declared_area is not None and int(declared_area) != area:
            rejection["reasons"].append("declared_area_mismatch")
        if area < minimum_area:
            rejection["reasons"].append("area_too_small")
        if area > maximum_area:
            rejection["reasons"].append("area_too_large")
        bbox = TOP_ZONE.mask_bbox(mask)
        successful_overlaps = []
        for object_id, successful_mask in successful_masks:
            metrics = TOP_ZONE.mask_overlap(mask, successful_mask)
            if (
                metrics["iou"] >= float(cfg["successful_mask_exclusion_iou"])
                or metrics["left_coverage"]
                >= float(cfg["successful_mask_exclusion_candidate_coverage"])
            ):
                successful_overlaps.append(
                    {
                        "object_id": object_id,
                        "iou": round(metrics["iou"], 6),
                        "candidate_coverage": round(metrics["left_coverage"], 6),
                    }
                )
        if successful_overlaps:
            rejection["reasons"].append("overlaps_successful_object")
            rejection["successful_object_overlaps"] = successful_overlaps
        if rejection["reasons"]:
            if rejection["reasons"] == ["unassigned_score_below_threshold"]:
                semantic_aliases.append({
                    "source_namespace": namespace,
                    "source_candidate_id": review_id,
                    "prompt": prompt,
                    "score": score,
                    "area_pixels": area,
                    "bbox_abs_xyxy": bbox,
                    "mask": mask,
                })
            validation_rejections.append(rejection)
            return
        eligible.append(
            {
                "source_namespace": namespace,
                "source_candidate_id": review_id,
                "source_upstream_candidate_id": upstream_id,
                "prompt": prompt,
                "score": score,
                "prompt_evidence": [{
                    "prompt": prompt,
                    "score": score,
                    "source_namespace": namespace,
                    "source_candidate_id": review_id,
                }],
                "area_pixels": area,
                "area_fraction": area / image_area,
                "bbox_abs_xyxy": bbox,
                "bbox_norm_xyxy": [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ],
                "source_mask_path": str(mask_path),
                "source_overlay_path": str(overlay_path),
                "category_hint": str(raw.get("category_hint") or ""),
                "target_object_ids": [
                    str(value) for value in raw.get("target_object_ids") or []
                ],
                "recall_sources": [
                    str(value) for value in raw.get("recall_sources") or []
                ],
                "local_instance_evidence": bool(
                    raw.get("local_instance_evidence")
                ),
                "host_source_object_id": str(
                    raw.get("host_source_object_id") or ""
                ),
                "grouped_component_count": int(
                    raw.get("grouped_component_count") or 0
                ),
                "mask": mask,
            }
        )

    for candidate in top_candidates:
        validate_candidate(candidate, "top_zone")
    for candidate in sam3_registry.get("unassigned_sam3_candidates") or []:
        validate_candidate(candidate, "sam3_a_unassigned")

    eligible.sort(
        key=lambda item: (
            {
                "top_zone": 0,
                "sam3_a_unassigned": 1,
            }.get(item["source_namespace"], 3),
            -float(item["score"]),
            item["bbox_abs_xyxy"][1],
            item["bbox_abs_xyxy"][0],
            item["source_candidate_id"],
        )
    )
    deduplicated: list[dict[str, Any]] = []
    duplicate_rejections: list[dict[str, Any]] = []
    for candidate in eligible:
        duplicate = None
        for existing in deduplicated:
            metrics = TOP_ZONE.mask_overlap(candidate["mask"], existing["mask"])
            if (
                metrics["iou"] >= float(cfg["duplicate_iou"])
                or max(metrics["left_coverage"], metrics["right_coverage"])
                >= float(cfg["duplicate_containment"])
            ):
                duplicate = (existing, metrics)
                break
        if duplicate is None:
            deduplicated.append(candidate)
        else:
            duplicate[0]["prompt_evidence"].extend(candidate["prompt_evidence"])
            duplicate_rejections.append(
                {
                    "source_namespace": candidate["source_namespace"],
                    "source_candidate_id": candidate["source_candidate_id"],
                    "duplicate_of_namespace": duplicate[0]["source_namespace"],
                    "duplicate_of_candidate_id": duplicate[0]["source_candidate_id"],
                    "overlap": {
                        key: round(value, 6) for key, value in duplicate[1].items()
                    },
                }
            )
    for alias in semantic_aliases:
        for existing in deduplicated:
            metrics = TOP_ZONE.mask_overlap(alias["mask"], existing["mask"])
            if not (
                metrics["iou"] >= float(cfg["duplicate_iou"])
                or max(metrics["left_coverage"], metrics["right_coverage"])
                >= float(cfg["duplicate_containment"])
            ):
                continue
            evidence = {
                "prompt": alias["prompt"],
                "score": alias["score"],
                "source_namespace": alias["source_namespace"],
                "source_candidate_id": alias["source_candidate_id"],
            }
            if evidence["prompt"] not in {
                item["prompt"] for item in existing["prompt_evidence"]
            }:
                existing["prompt_evidence"].append(evidence)
            break


    top_pool = [
        item for item in deduplicated if item["source_namespace"] == "top_zone"
    ]
    unassigned_pool = [
        item
        for item in deduplicated
        if item["source_namespace"] == "sam3_a_unassigned"
    ]
    unassigned_reserved = min(
        int(cfg["reserved_unassigned_slots"]),
        len(unassigned_pool),
        maximum,
    )
    top_take = min(
        len(top_pool),
        maximum - unassigned_reserved,
    )
    selected = list(top_pool[:top_take])
    selected.extend(unassigned_pool[:unassigned_reserved])
    for pool, start_index in (
        (unassigned_pool, unassigned_reserved),
        (top_pool, top_take),
    ):
        if len(selected) >= maximum:
            break
        selected.extend(pool[start_index : start_index + maximum - len(selected)])
    selected_keys = {
        (item["source_namespace"], item["source_candidate_id"])
        for item in selected
    }
    overflow_rejections = [
        {
            **{key: value for key, value in item.items() if key != "mask"},
            "reason": "bounded_panel_capacity",
        }
        for item in deduplicated
        if (item["source_namespace"], item["source_candidate_id"])
        not in selected_keys
    ]
    hard_errors: list[dict[str, Any]] = []
    if overflow_rejections:
        hard_errors.append(
            {
                "type": "bounded_candidate_capacity_overflow",
                "count": len(overflow_rejections),
                "source_candidate_ids": [
                    item["source_candidate_id"]
                    for item in overflow_rejections
                ],
            }
        )
    nonblocking_validation_reasons = {
        "invalid_score",
        "unassigned_score_below_threshold",
        "empty_mask",
        "area_too_small",
        "area_too_large",
        "overlaps_successful_object",
    }
    blocking_validation_rejections = []
    for rejection in validation_rejections:
        score_value = rejection.get("score")
        area_value = rejection.get("area_pixels")
        unresolved_reasons = [
            reason
            for reason in rejection["reasons"]
            if reason not in nonblocking_validation_reasons
            and not str(reason).startswith("invalid_mask:")
        ]
        if (
            isinstance(score_value, (int, float))
            and float(score_value) >= float(cfg["minimum_unassigned_score"])
            and isinstance(area_value, int)
            and minimum_area <= area_value <= maximum_area
            and unresolved_reasons
        ):
            blocking_validation_rejections.append(
                {
                    **rejection,
                    "unresolved_reasons": unresolved_reasons,
                }
            )
    if blocking_validation_rejections:
        hard_errors.append(
            {
                "type": "bounded_candidate_validation_unresolved",
                "count": len(blocking_validation_rejections),
                "source_candidate_ids": [
                    item["source_candidate_id"]
                    for item in blocking_validation_rejections
                ],
            }
        )
    for rank, candidate in enumerate(selected, start=1):
        candidate["selection_priority_rank"] = rank
    selected.sort(
        key=lambda item: (
            item["bbox_abs_xyxy"][1],
            item["bbox_abs_xyxy"][0],
            item["selection_priority_rank"],
        )
    )
    for index, candidate in enumerate(selected, start=1):
        candidate["candidate_id"] = f"LC_{index:03d}"

    selected_public = [
        {key: value for key, value in item.items() if key != "mask"}
        for item in selected
    ]
    selected_counts = {
        namespace: sum(
            item["source_namespace"] == namespace for item in selected
        )
        for namespace in (
            "top_zone",
            "sam3_a_unassigned",
        )
    }
    audit = {
        "schema_version": "scenemosaic_bounded_post_sam3a_candidate_selection",
        "status": "candidates_found" if selected else "no_candidates",
        "source_image": str(resolved_image),
        "image_size": [width, height],
        "config": cfg,
        "input_counts": {
            "top_zone": len(top_candidates),
            "sam3_a_unassigned": len(
                sam3_registry.get("unassigned_sam3_candidates") or []
            ),
            "successful_object_masks": len(successful_masks),
        },
        "eligible_before_deduplication": len(eligible),
        "deduplicated_count": len(deduplicated),
        "selected_count": len(selected),
        "selected_counts_by_namespace": selected_counts,
        "unassigned_slot_reserved": unassigned_reserved,
        "all_eligible_unassigned_silently_crowded_out": bool(
            unassigned_pool and selected_counts["sam3_a_unassigned"] == 0
        ),
        "selected": selected_public,
        "validation_rejections": validation_rejections,
        "blocking_validation_rejections": blocking_validation_rejections,
        "duplicate_rejections": duplicate_rejections,
        "overflow_rejections": overflow_rejections,
        "hard_errors": hard_errors,
        "additional_vlm_calls": 0,
        "logical_pipeline_vlm_api_calls": 2,
        "api_calls_unchanged": True,
    }
    if audit["all_eligible_unassigned_silently_crowded_out"]:
        raise AssertionError("eligible SAM3-A candidates were silently crowded out")
    return {
        "filtered": {
            "status": audit["status"],
            "image_size": [width, height],
            "config": cfg,
            "selected": selected,
            "geometric_rejections": validation_rejections,
            "duplicate_rejections": duplicate_rejections,
            "overflow_rejections": overflow_rejections,
        },
        "audit": audit,
    }


def materialize_bounded_post_sam3a_review(
    *,
    image_path: str | Path,
    top_candidates: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selection = select_bounded_post_sam3a_candidates(
        image_path=image_path,
        top_candidates=top_candidates,
        sam3_registry=sam3_registry,
        config=config,
    )
    review = TOP_ZONE.materialize_top_zone_review(
        image_path,
        selection["filtered"],
        output_dir,
        panel_title=(
            "MERGED TOP-ZONE + "
            "SAM3-A UNASSIGNED CANDIDATES"
        ),
        schema_version="scenemosaic_bounded_post_sam3a_candidate_review",
    )
    audit = {
        **selection["audit"],
        "review_manifest": review["manifest_path"],
        "numbered_candidate_panel": review["panel_path"],
    }
    audit_path = Path(output_dir).expanduser().resolve() / "candidate_selection_audit.json"
    atomic_json(audit_path, audit)
    return {
        **review,
        "selection_audit": audit,
        "selection_audit_path": str(audit_path),
    }


LOCAL_MATCH_THRESHOLDS = {
    "minimum_candidate_score": 0.60,
    "minimum_semantic_score": 0.85,
    "minimum_iou": 0.65,
    "minimum_candidate_coverage": 0.80,
    "minimum_addition_coverage": 0.75,
    "minimum_area_ratio": 0.50,
    "maximum_area_ratio": 2.00,
    "maximum_relative_center_distance": 0.25,
    "minimum_total_score": 0.82,
    "minimum_runner_up_margin": 0.10,
}


def _semantic_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _local_candidate_semantic_score(
    candidate: Mapping[str, Any],
    addition: Mapping[str, Any],
) -> float:
    # The SAM3 prompt is the sole instance-level binding label. Category is
    # retained in the formal object schema, but never used for mask binding.
    candidate_key = VLM.category_key(candidate.get("prompt"))
    if not candidate_key:
        return 0.0
    prompt_key = VLM.category_key(addition.get("sam3_prompt"))
    if prompt_key and candidate_key == prompt_key:
        return 1.0

    candidate_tokens = _semantic_tokens(candidate.get("prompt"))
    primary_tokens = _semantic_tokens(addition.get("sam3_prompt"))
    all_tokens = set(primary_tokens)
    for key in ("instance_name", "object_type", "description"):
        all_tokens.update(_semantic_tokens(addition.get(key)))
    if (
        candidate_tokens
        and candidate_tokens.issubset(all_tokens)
        and candidate_tokens.intersection(primary_tokens)
    ):
        return 0.85
    return 0.0


def _local_candidate_pair_metrics(
    candidate: Mapping[str, Any],
    addition: Mapping[str, Any],
) -> dict[str, float]:
    candidate_bbox = [float(value) for value in candidate["bbox_norm_xyxy"]]
    addition_bbox = [float(value) for value in addition["bbox_2d_norm_xyxy"]]
    overlap = VLM.bbox_metrics(candidate_bbox, addition_bbox)
    candidate_area = (
        (candidate_bbox[2] - candidate_bbox[0])
        * (candidate_bbox[3] - candidate_bbox[1])
    )
    addition_area = (
        (addition_bbox[2] - addition_bbox[0])
        * (addition_bbox[3] - addition_bbox[1])
    )
    candidate_center = (
        (candidate_bbox[0] + candidate_bbox[2]) / 2.0,
        (candidate_bbox[1] + candidate_bbox[3]) / 2.0,
    )
    addition_center = (
        (addition_bbox[0] + addition_bbox[2]) / 2.0,
        (addition_bbox[1] + addition_bbox[3]) / 2.0,
    )
    scale = max(
        math.hypot(
            candidate_bbox[2] - candidate_bbox[0],
            candidate_bbox[3] - candidate_bbox[1],
        ),
        math.hypot(
            addition_bbox[2] - addition_bbox[0],
            addition_bbox[3] - addition_bbox[1],
        ),
    )
    relative_center_distance = (
        math.dist(candidate_center, addition_center) / scale if scale else math.inf
    )
    semantic_score = _local_candidate_semantic_score(candidate, addition)
    geometry_score = (
        0.50 * overlap["iou"]
        + 0.25 * overlap["left_coverage"]
        + 0.25 * overlap["right_coverage"]
    )
    total_score = 0.75 * geometry_score + 0.25 * semantic_score
    return {
        "iou": overlap["iou"],
        "candidate_coverage": overlap["left_coverage"],
        "addition_coverage": overlap["right_coverage"],
        "area_ratio": addition_area / candidate_area if candidate_area else math.inf,
        "relative_center_distance": relative_center_distance,
        "semantic_score": semantic_score,
        "geometry_score": geometry_score,
        "total_score": total_score,
    }


def match_local_candidates_to_b_additions(
    fused_objects: Sequence[Mapping[str, Any]],
    local_candidates: Sequence[Mapping[str, Any]],
    *,
    explicitly_rejected_candidate_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Only accepted B additions may receive local evidence. The matcher never
    # creates a new object from an LC candidate.
    thresholds = dict(LOCAL_MATCH_THRESHOLDS)
    additions: list[tuple[str, dict[str, Any]]] = []
    seen_addition_ids: set[str] = set()
    for raw_object in fused_objects:
        obj = dict(raw_object)
        sources = [
            source
            for source in obj.get("proposal_sources") or []
            if source.get("pass") == "B" and source.get("action") == "added"
        ]
        if not sources:
            continue
        if len(sources) != 1:
            raise ValueError("a fused proposal has multiple B addition sources")
        source_id = str(sources[0].get("source_id") or "")
        if not re.fullmatch(r"B_\d{3}", source_id):
            raise ValueError(f"invalid B addition source ID: {source_id!r}")
        if source_id in seen_addition_ids:
            raise ValueError(f"duplicate accepted B addition source: {source_id}")
        seen_addition_ids.add(source_id)
        if not VLM.valid_bbox(obj.get("bbox_2d_norm_xyxy")):
            raise ValueError(f"invalid accepted B addition bbox: {source_id}")
        additions.append((source_id, obj))

    candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for raw_candidate in local_candidates:
        candidate = dict(raw_candidate)
        candidate_id = str(candidate.get("candidate_id") or "")
        if not re.fullmatch(r"LC_\d{3}", candidate_id):
            raise ValueError(f"invalid local candidate ID: {candidate_id!r}")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate local candidate ID: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        if not VLM.valid_bbox(candidate.get("bbox_norm_xyxy")):
            raise ValueError(f"invalid local candidate bbox: {candidate_id}")
        score = float(candidate.get("score") or 0.0)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid local candidate score: {candidate_id}")
        candidates.append(candidate)

    if isinstance(explicitly_rejected_candidate_ids, (str, bytes)):
        raise ValueError("explicit local candidate rejections must be a sequence")
    rejected_candidate_ids = [
        str(value) for value in explicitly_rejected_candidate_ids
    ]
    if len(rejected_candidate_ids) != len(set(rejected_candidate_ids)):
        raise ValueError("duplicate explicit local candidate rejection ID")
    if not all(
        re.fullmatch(r"LC_\d{3}", value) for value in rejected_candidate_ids
    ):
        raise ValueError("invalid explicit local candidate rejection ID")
    unknown_rejections = sorted(set(rejected_candidate_ids) - seen_candidate_ids)
    if unknown_rejections:
        raise ValueError(
            f"unknown explicit local candidate rejection IDs: {unknown_rejections}"
        )
    rejected_candidate_id_set = set(rejected_candidate_ids)

    pair_records: list[dict[str, Any]] = []
    eligible_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_by_addition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_score = float(candidate.get("score") or 0.0)
        for addition_id, addition in additions:
            metrics = _local_candidate_pair_metrics(candidate, addition)
            rejection_reasons = []
            if candidate_score < thresholds["minimum_candidate_score"]:
                rejection_reasons.append("candidate_score")
            if metrics["semantic_score"] < thresholds["minimum_semantic_score"]:
                rejection_reasons.append("semantic_score")
            if metrics["iou"] < thresholds["minimum_iou"]:
                rejection_reasons.append("iou")
            if metrics["candidate_coverage"] < thresholds["minimum_candidate_coverage"]:
                rejection_reasons.append("candidate_coverage")
            if metrics["addition_coverage"] < thresholds["minimum_addition_coverage"]:
                rejection_reasons.append("addition_coverage")
            if not (
                thresholds["minimum_area_ratio"]
                <= metrics["area_ratio"]
                <= thresholds["maximum_area_ratio"]
            ):
                rejection_reasons.append("area_ratio")
            if (
                metrics["relative_center_distance"]
                > thresholds["maximum_relative_center_distance"]
            ):
                rejection_reasons.append("relative_center_distance")
            if metrics["total_score"] < thresholds["minimum_total_score"]:
                rejection_reasons.append("total_score")
            record = {
                "candidate_id": candidate_id,
                "addition_id": addition_id,
                "proposal_id": addition.get("object_id"),
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
                **{key: round(value, 6) for key, value in metrics.items()},
            }
            pair_records.append(record)
            if record["eligible"]:
                eligible_by_candidate[candidate_id].append(record)
                if candidate_id not in rejected_candidate_id_set:
                    eligible_by_addition[addition_id].append(record)

    def ranked(records: Sequence[Mapping[str, Any]], other_key: str) -> list[dict[str, Any]]:
        return sorted(
            (dict(record) for record in records),
            key=lambda record: (-float(record["total_score"]), str(record[other_key])),
        )

    def unique_best(records: Sequence[Mapping[str, Any]], other_key: str) -> dict[str, Any] | None:
        ordered = ranked(records, other_key)
        if not ordered:
            return None
        if len(ordered) > 1 and (
            float(ordered[0]["total_score"])
            - float(ordered[1]["total_score"])
            < thresholds["minimum_runner_up_margin"]
        ):
            return None
        return ordered[0]

    decisions: list[dict[str, Any]] = [
        {
            "candidate_id": candidate_id,
            "decision": "reject",
            "target_object_id": None,
            "reason": "explicit Pass-B not-new-object disposition",
        }
        for candidate_id in rejected_candidate_ids
    ]
    matches: list[dict[str, Any]] = []
    matched_candidate_ids: set[str] = set()
    matched_addition_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in rejected_candidate_id_set:
            continue
        best = unique_best(eligible_by_candidate.get(candidate_id, []), "addition_id")
        if best is None:
            continue
        addition_id = str(best["addition_id"])
        reverse_best = unique_best(
            eligible_by_addition.get(addition_id, []), "candidate_id"
        )
        if reverse_best is None or str(reverse_best["candidate_id"]) != candidate_id:
            continue
        if addition_id in matched_addition_ids:
            raise AssertionError("local candidate matcher violated one-to-one assignment")
        matched_candidate_ids.add(candidate_id)
        matched_addition_ids.add(addition_id)
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": "bind_to_addition",
                "target_object_id": addition_id,
                "reason": "unique local bbox semantic match",
            }
        )
        matches.append(
            {
                key: best[key]
                for key in (
                    "candidate_id",
                    "addition_id",
                    "proposal_id",
                    "iou",
                    "candidate_coverage",
                    "addition_coverage",
                    "area_ratio",
                    "relative_center_distance",
                    "semantic_score",
                    "geometry_score",
                    "total_score",
                )
            }
        )

    hard_errors: list[dict[str, Any]] = []
    conflicting_rejected_candidate_ids = []
    for candidate_id in rejected_candidate_ids:
        eligible = eligible_by_candidate.get(candidate_id, [])
        if not eligible:
            continue
        conflicting_rejected_candidate_ids.append(candidate_id)
        hard_errors.append(
            {
                "type": "conflicting_local_candidate_disposition",
                "candidate_id": candidate_id,
                "eligible_addition_ids": sorted(
                    str(record["addition_id"]) for record in eligible
                ),
                "reason": (
                    "candidate was explicitly rejected but also has a "
                    "high-confidence B addition match"
                ),
            }
        )

    unresolved_candidate_ids = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in rejected_candidate_id_set:
            continue
        if candidate_id in matched_candidate_ids:
            continue
        unresolved_candidate_ids.append(candidate_id)
        eligible = eligible_by_candidate.get(candidate_id, [])
        hard_errors.append(
            {
                "type": (
                    "ambiguous_local_candidate_match"
                    if eligible
                    else "unmatched_local_candidate"
                ),
                "candidate_id": candidate_id,
                "eligible_addition_ids": sorted(
                    str(record["addition_id"]) for record in eligible
                ),
                "reason": (
                    "no unique mutual high-confidence B addition"
                    if eligible
                    else "no high-confidence B addition"
                ),
            }
        )

    if len(decisions) + len(unresolved_candidate_ids) != len(candidates):
        raise AssertionError("local candidate accounting invariant failed")
    partition_sets = [
        set(rejected_candidate_ids),
        set(matched_candidate_ids),
        set(unresolved_candidate_ids),
    ]
    partition_disjoint = all(
        not partition_sets[left] & partition_sets[right]
        for left in range(len(partition_sets))
        for right in range(left + 1, len(partition_sets))
    )
    complete_partition = (
        partition_disjoint
        and set().union(*partition_sets) == seen_candidate_ids
    )
    if not complete_partition:
        raise AssertionError("local candidate partition invariant failed")
    return decisions, {
        "schema_version": "scenemosaic_local_candidate_match",
        "mode": "deterministic_bbox_semantic_one_to_one",
        "thresholds": thresholds,
        "candidate_count": len(candidates),
        "accepted_b_addition_count": len(additions),
        "evaluated_pair_count": len(pair_records),
        "eligible_pair_count": sum(
            1 for record in pair_records if record["eligible"]
        ),
        "explicit_reject_count": len(rejected_candidate_ids),
        "explicit_rejected_candidate_ids": rejected_candidate_ids,
        "conflicting_rejected_candidate_ids": (
            conflicting_rejected_candidate_ids
        ),
        "match_count": len(matches),
        "matched_candidate_ids": sorted(matched_candidate_ids),
        "matches": matches,
        "unresolved_candidate_ids": unresolved_candidate_ids,
        "complete_partition": complete_partition,
        "partition_disjoint": partition_disjoint,
        "pair_evaluations": pair_records,
        "hard_errors": hard_errors,
        "status": "pass" if not hard_errors else "fail",
        "creates_objects_from_local_candidates": False,
    }


def promote_repeated_top_fixture_candidates(
    *,
    pass_a_objects: Sequence[Mapping[str, Any]],
    pass_b_actions: Mapping[str, Any],
    local_candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover corroborated top-edge fixtures rejected by compact review.
    """
    actions = copy.deepcopy(dict(pass_b_actions))
    had_local_candidate_decisions = "local_candidate_decisions" in actions
    additions = list(actions.get("additions") or [])
    removal_ids = {str(value) for value in actions.get("removals") or []}

    def box(item: Mapping[str, Any]) -> list[float] | None:
        raw = item.get("bbox_2d_norm_xyxy") or item.get("bbox_norm_xyxy")
        try:
            values = [float(value) for value in raw]
        except (TypeError, ValueError):
            return None
        if (
            len(values) != 4
            or not (0.0 <= values[0] < values[2] <= 1.0)
            or not (0.0 <= values[1] < values[3] <= 1.0)
        ):
            return None
        return values

    def tokens(item: Mapping[str, Any]) -> set[str]:
        return set(
            re.findall(
                r"[a-z0-9]+",
                " ".join(
                    str(item.get(key) or "")
                    for key in ("category", "sam3_prompt", "description", "prompt")
                ).lower(),
            )
        )

    def is_top_fixture(item: Mapping[str, Any]) -> bool:
        candidate_box = box(item)
        item_tokens = tokens(item)
        return bool(
            candidate_box
            and candidate_box[3] <= 0.18
            and item_tokens & {"light", "lamp", "lighting"}
        )

    def overlap_metrics(
        left: Sequence[float], right: Sequence[float]
    ) -> tuple[float, float, float]:
        inter_left = max(left[0], right[0])
        inter_top = max(left[1], right[1])
        inter_right = min(left[2], right[2])
        inter_bottom = min(left[3], right[3])
        inter_area = max(0.0, inter_right - inter_left) * max(
            0.0, inter_bottom - inter_top
        )
        left_area = max(0.0, left[2] - left[0]) * max(
            0.0, left[3] - left[1]
        )
        right_area = max(0.0, right[2] - right[0]) * max(
            0.0, right[3] - right[1]
        )
        union_area = left_area + right_area - inter_area
        return (
            inter_area / union_area if union_area else 0.0,
            inter_area / left_area if left_area else 0.0,
            inter_area / right_area if right_area else 0.0,
        )

    def matching_addition_id(
        candidate: Mapping[str, Any], candidate_box: Sequence[float]
    ) -> str | None:
        """Find the unique ordinary addition representing an LC candidate.

        A compact response can incorrectly reject an LC while emitting the
        same physical object as an ordinary addition. This is a protocol
        inconsistency, not evidence for a second object.
        """
        candidate_tokens = tokens(candidate)
        matches: list[str] = []
        for index, addition in enumerate(additions, start=1):
            addition_box = box(addition)
            shared_tokens = candidate_tokens & tokens(addition)
            if addition_box is None or not shared_tokens:
                continue
            iou, candidate_coverage, addition_coverage = overlap_metrics(
                candidate_box, addition_box
            )
            if (
                iou >= 0.55
                or candidate_coverage >= 0.80
                or addition_coverage >= 0.80
            ):
                matches.append(f"B_{index:03d}")
        return matches[0] if len(matches) == 1 else None

    repeated_peer_count = sum(
        1 for item in pass_a_objects if is_top_fixture(item)
    )
    source_boxes = [
        (str(item.get("object_id") or ""), box(item))
        for item in pass_a_objects
        if is_top_fixture(item)
    ]
    promoted: list[dict[str, Any]] = []
    retained_decisions: list[dict[str, Any]] = []
    for raw_decision in actions.get("local_candidate_decisions") or []:
        decision = copy.deepcopy(dict(raw_decision))
        candidate_id = str(decision.get("candidate_id") or "")
        candidate = next(
            (
                item
                for item in local_candidates
                if str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        candidate_box = box(candidate or {})
        evidence = list((candidate or {}).get("prompt_evidence") or [])
        evidence_sources = {
            str(item.get("source_namespace") or "")
            for item in evidence
            if isinstance(item, Mapping)
        }
        evidence_scores = [
            float(item.get("score") or 0.0)
            for item in evidence
            if isinstance(item, Mapping)
        ]
        candidate_tokens = tokens(candidate or {})
        candidate_center = (
            ((candidate_box[0] + candidate_box[2]) / 2.0,
             (candidate_box[1] + candidate_box[3]) / 2.0)
            if candidate_box is not None
            else None
        )
        matching_retained_source = False
        if candidate_center is not None:
            matching_retained_source = any(
                source_id not in removal_ids
                and source_box is not None
                and source_box[0] <= candidate_center[0] <= source_box[2]
                and source_box[1] <= candidate_center[1] <= source_box[3]
                for source_id, source_box in source_boxes
            )
        qualifies = (
            decision.get("decision") == "reject"
            and candidate is not None
            and str(candidate.get("source_namespace") or "") == "top_zone"
            and candidate_box is not None
            and candidate_box[3] <= 0.18
            and bool(candidate_tokens & {"light", "lamp", "lighting"})
            and repeated_peer_count >= 2
            and len(evidence_sources) >= 2
            and max(evidence_scores, default=0.0) >= 0.55
            and not matching_retained_source
        )
        if not qualifies:
            retained_decisions.append(decision)
            continue
        existing_addition_id = matching_addition_id(candidate, candidate_box)
        if existing_addition_id is not None:
            decision.update(
                {
                    "decision": "bind_to_addition",
                    "target_object_id": existing_addition_id,
                    "reason": (
                        "compact Pass-B rejected the local candidate but "
                        "emitted the same corroborated fixture as an "
                        "ordinary addition"
                    ),
                }
            )
            retained_decisions.append(decision)
            promoted.append(
                {
                    "candidate_id": candidate_id,
                    "target_object_id": existing_addition_id,
                    "mode": "bound_to_existing_addition",
                }
            )
            continue
        semantic_prompt = next(
            (
                str(item.get("prompt") or "").strip()
                for item in sorted(
                    evidence,
                    key=lambda item: float(item.get("score") or 0.0),
                    reverse=True,
                )
                if "light" in str(item.get("prompt") or "").lower()
            ),
            str(candidate.get("prompt") or "ceiling light").strip(),
        )
        category = "_".join(re.findall(r"[a-z0-9]+", semantic_prompt.lower()))
        if not category:
            retained_decisions.append(decision)
            continue
        anchor = [
            (candidate_box[0] + candidate_box[2]) / 2.0,
            (candidate_box[1] + candidate_box[3]) / 2.0,
        ]
        additions.append(
            VLM._minimal_semantic_record(
                category=category,
                prompt=semantic_prompt,
                bbox=candidate_box,
                anchor=anchor,
            )
        )
        addition_id = f"B_{len(additions):03d}"
        decision.update(
            {
                "decision": "bind_to_addition",
                "target_object_id": addition_id,
                "reason": "corroborated repeated top-zone fixture retained",
            }
        )
        retained_decisions.append(decision)
        promoted.append(
            {
                "candidate_id": candidate_id,
                "target_object_id": addition_id,
                "category": category,
                "prompt": semantic_prompt,
                "evidence_source_count": len(evidence_sources),
                "maximum_evidence_score": max(evidence_scores, default=0.0),
            }
        )
    actions["additions"] = additions
    if had_local_candidate_decisions or retained_decisions:
        actions["local_candidate_decisions"] = retained_decisions
    audit = {
        "schema_version": "scenemosaic_repeated_top_fixture_recall",
        "status": "pass" if promoted else "noop",
        "repeated_top_fixture_peer_count": repeated_peer_count,
        "promoted": promoted,
        "additional_vlm_calls": 0,
        "additional_sam3_inference_calls": 0,
    }
    return actions, audit


FAILED_EXISTING_OBJECT_REBIND_THRESHOLDS = {
    "minimum_candidate_score": 0.60,
    "minimum_semantic_score": 0.85,
    "minimum_iou": 0.55,
    "minimum_candidate_coverage": 0.95,
    "minimum_existing_coverage": 0.50,
    "maximum_relative_center_distance": 0.15,
    "minimum_evidence_source_count": 2,
    "minimum_evidence_score": 0.60,
}


def rebind_verified_local_candidates_to_failed_existing_objects(
    *,
    pass_a_objects: Sequence[Mapping[str, Any]],
    pass_b_actions: Mapping[str, Any],
    local_candidates: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize a verified local mask for a failed existing object.

    Compact Pass B distinguishes only new local-candidate additions from
    rejections. A candidate can therefore be rejected as *not new* while its
    matching Pass-A object has no SAM3 mask. This deterministic recovery is
    deliberately narrower than semantic acceptance: it requires a Pass-B
    semantic pass, a failed SAM3 source, two independent local-evidence
    sources, and a unique high-agreement geometry match. It never creates an
    object or turns an unreviewed candidate into a mask.
    """

    actions = copy.deepcopy(dict(pass_b_actions))
    thresholds = dict(FAILED_EXISTING_OBJECT_REBIND_THRESHOLDS)
    pass_a_by_id = {
        str(item.get("object_id") or ""): dict(item)
        for item in pass_a_objects
        if str(item.get("object_id") or "")
    }
    registry_by_id = {
        str(item.get("object_id") or ""): dict(item)
        for item in sam3_registry.get("objects") or []
        if isinstance(item, Mapping) and str(item.get("object_id") or "")
    }
    candidates_by_id = {
        str(item.get("candidate_id") or ""): dict(item)
        for item in local_candidates
        if str(item.get("candidate_id") or "")
    }
    removed_ids = {str(value) for value in actions.get("removals") or []}
    updated_ids = {
        str(item.get("pass_a_id") or "")
        for item in actions.get("updates") or []
        if isinstance(item, Mapping)
    }

    gate = actions.get("assigned_evidence_semantic_gate")
    assertions_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if isinstance(gate, Mapping):
        for raw_assertion in gate.get(
            "normalized_unmaterialized_repair_assertions"
        ) or []:
            if not isinstance(raw_assertion, Mapping):
                continue
            assertion = dict(raw_assertion)
            if (
                assertion.get("reported_decision") == "pass"
                and assertion.get("effective_decision") == "uncertain"
                and assertion.get("rule")
                == "failed_sam3_evidence_cannot_be_accepted_without_repair"
            ):
                source_id = str(assertion.get("source_object_id") or "")
                if source_id:
                    assertions_by_source[source_id].append(assertion)

    eligible_pairs: list[dict[str, Any]] = []
    for raw_decision in actions.get("local_candidate_decisions") or []:
        if not isinstance(raw_decision, Mapping):
            continue
        decision = dict(raw_decision)
        if decision.get("decision") != "reject":
            continue
        candidate_id = str(decision.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        try:
            candidate_score = float(candidate.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if candidate_score < thresholds["minimum_candidate_score"]:
            continue
        evidence = [
            dict(item)
            for item in candidate.get("prompt_evidence") or []
            if isinstance(item, Mapping)
        ]
        evidence_sources = {
            str(item.get("source_namespace") or "")
            for item in evidence
            if str(item.get("source_namespace") or "")
        }
        try:
            evidence_scores = [
                float(item.get("score") or 0.0) for item in evidence
            ]
        except (TypeError, ValueError):
            continue
        if (
            len(evidence_sources) < thresholds["minimum_evidence_source_count"]
            or max(evidence_scores, default=0.0)
            < thresholds["minimum_evidence_score"]
        ):
            continue
        for source_id, assertions in assertions_by_source.items():
            source = pass_a_by_id.get(source_id)
            registry_item = registry_by_id.get(source_id) or {}
            if (
                source is None
                or source_id in removed_ids
                or source_id in updated_ids
                or (registry_item.get("sam3") or {}).get("status") != "failed"
            ):
                continue
            metrics = _local_candidate_pair_metrics(candidate, source)
            if (
                metrics["semantic_score"] < thresholds["minimum_semantic_score"]
                or metrics["iou"] < thresholds["minimum_iou"]
                or metrics["candidate_coverage"]
                < thresholds["minimum_candidate_coverage"]
                or metrics["addition_coverage"]
                < thresholds["minimum_existing_coverage"]
                or metrics["relative_center_distance"]
                > thresholds["maximum_relative_center_distance"]
            ):
                continue
            eligible_pairs.append(
                {
                    "candidate_id": candidate_id,
                    "source_object_id": source_id,
                    "evidence_ids": sorted(
                        str(item.get("evidence_id") or "")
                        for item in assertions
                        if str(item.get("evidence_id") or "")
                    ),
                    "metrics": {
                        key: round(float(value), 6)
                        for key, value in metrics.items()
                    },
                    "evidence_source_count": len(evidence_sources),
                    "maximum_evidence_score": round(
                        max(evidence_scores, default=0.0), 6
                    ),
                }
            )

    pairs_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in eligible_pairs:
        pairs_by_candidate[pair["candidate_id"]].append(pair)
        pairs_by_source[pair["source_object_id"]].append(pair)
    selected_pairs = [
        pair
        for pair in eligible_pairs
        if len(pairs_by_candidate[pair["candidate_id"]]) == 1
        and len(pairs_by_source[pair["source_object_id"]]) == 1
    ]
    selected_by_candidate = {
        pair["candidate_id"]: pair for pair in selected_pairs
    }
    selected_by_source = {
        pair["source_object_id"]: pair for pair in selected_pairs
    }

    rebound: list[dict[str, Any]] = []
    had_local_candidate_decisions = "local_candidate_decisions" in actions
    rewritten_decisions: list[dict[str, Any]] = []
    for raw_decision in actions.get("local_candidate_decisions") or []:
        decision = copy.deepcopy(dict(raw_decision))
        pair = selected_by_candidate.get(str(decision.get("candidate_id") or ""))
        if pair is not None:
            decision.update(
                {
                    "decision": "replace_existing",
                    "target_object_id": pair["source_object_id"],
                    "reason": (
                        "one-to-one verified local candidate rebound to a "
                        "failed existing object after Pass-B semantic pass"
                    ),
                }
            )
            rebound.append(pair)
        rewritten_decisions.append(decision)
    if had_local_candidate_decisions:
        actions["local_candidate_decisions"] = rewritten_decisions

    recovered_evidence_ids = {
        evidence_id
        for pair in selected_by_source.values()
        for evidence_id in pair["evidence_ids"]
    }
    rewritten_evidence = []
    for raw_decision in actions.get("assigned_evidence_decisions") or []:
        decision = copy.deepcopy(dict(raw_decision))
        if str(decision.get("evidence_id") or "") in recovered_evidence_ids:
            original_reason = str(decision.get("reason") or "")
            decision["decision"] = "pass"
            decision["reason"] = (
                "one-to-one verified local candidate mask was bound to this "
                f"failed object. Original reason: {original_reason}"
            )
        rewritten_evidence.append(decision)
    if "assigned_evidence_decisions" in actions:
        actions["assigned_evidence_decisions"] = rewritten_evidence

    if isinstance(gate, Mapping):
        rewritten_gate = copy.deepcopy(dict(gate))
        unresolved = [
            str(value)
            for value in rewritten_gate.get("unresolved_evidence_ids") or []
            if str(value) not in recovered_evidence_ids
        ]
        rewritten_gate["unresolved_evidence_ids"] = unresolved
        rewritten_gate["status"] = "pass" if not unresolved else "blocked"
        rewritten_gate["local_candidate_rebinding_resolutions"] = rebound
        actions["assigned_evidence_semantic_gate"] = rewritten_gate

    ambiguous_pairs = [
        pair
        for pair in eligible_pairs
        if pair["candidate_id"] not in selected_by_candidate
        or pair["source_object_id"] not in selected_by_source
    ]
    audit = {
        "schema_version": (
            "scenemosaic_failed_existing_object_local_candidate_rebinding"
        ),
        "status": "pass" if rebound else "noop",
        "thresholds": thresholds,
        "eligible_pair_count": len(eligible_pairs),
        "ambiguous_pairs": ambiguous_pairs,
        "rebound": rebound,
        "additional_vlm_calls": 0,
        "additional_sam3_inference_calls": 0,
        "formal_outputs_modified": False,
    }
    return actions, audit


def compose_local_candidate_binding_actions(
    pass_b_actions: Mapping[str, Any],
    local_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert compact rejection output into the binding action shape."""

    binding_actions = copy.deepcopy(dict(pass_b_actions))
    binding_actions.pop("local_candidate_rejections", None)
    binding_actions["local_candidate_decisions"] = [
        copy.deepcopy(dict(item)) for item in local_decisions
    ]
    if "local_candidate_rejections" in binding_actions:
        raise AssertionError("binding actions retain compact rejection protocol")
    return binding_actions


def attach_local_candidate_bindings(
    fused_objects: Sequence[Mapping[str, Any]],
    pass_b_actions: Mapping[str, Any],
    local_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind reviewed local masks to fused A/B object identities."""

    objects = [copy.deepcopy(dict(item)) for item in fused_objects]
    candidates = {str(item["candidate_id"]): dict(item) for item in local_candidates}
    source_to_object: dict[tuple[str, str], dict[str, Any]] = {}
    for obj in objects:
        for source in obj.get("proposal_sources") or []:
            source_to_object[(str(source.get("pass")), str(source.get("source_id")))] = obj

    hard_errors: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = defaultdict(int)
    attached_object_count = 0
    bound_by_object: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for decision in pass_b_actions.get("local_candidate_decisions") or []:
        candidate_id = str(decision.get("candidate_id"))
        action = str(decision.get("decision"))
        decision_counts[action] += 1
        target = decision.get("target_object_id")
        candidate = candidates.get(candidate_id)
        if candidate is None:
            hard_errors.append(
                {"type": "unknown_local_candidate", "candidate_id": candidate_id}
            )
            continue
        if action == "uncertain":
            hard_errors.append(
                {
                    "type": "unresolved_local_candidate",
                    "candidate_id": candidate_id,
                    "reason": decision.get("reason"),
                }
            )
            continue
        if action in {"reject", "duplicate_of_existing", "part_of_existing"}:
            resolved.append(dict(decision))
            continue
        source_pass = "B" if action == "bind_to_addition" else "A"
        target_obj = source_to_object.get((source_pass, str(target)))
        if target_obj is None:
            hard_errors.append(
                {
                    "type": "unbound_local_candidate_target",
                    "candidate_id": candidate_id,
                    "decision": action,
                    "target_object_id": target,
                }
            )
            continue
        bound_by_object[str(target_obj["object_id"])].append((dict(decision), candidate))
        resolved.append(dict(decision))

    for object_id, records in bound_by_object.items():
        target_obj = next(obj for obj in objects if str(obj["object_id"]) == object_id)
        operations = {
            "union_existing" if decision["decision"] == "merge_extend_existing" else "replace"
            for decision, _ in records
        }
        if len(operations) != 1:
            hard_errors.append(
                {
                    "type": "conflicting_local_candidate_operations",
                    "object_id": object_id,
                    "operations": sorted(operations),
                }
            )
            continue
        binding_candidates = []
        for decision, candidate in records:
            candidate_id = str(candidate["candidate_id"])
            binding_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "prompt": candidate.get("prompt"),
                    "score": candidate.get("score"),
                    "bbox_norm_xyxy": candidate.get("bbox_norm_xyxy"),
                    "mask_path": candidate.get("mask_path"),
                    "overlay_path": candidate.get("overlay_path"),
                    "decision": decision["decision"],
                    "reason": decision.get("reason"),
                }
            )
            target_obj.setdefault("proposal_sources", []).append(
                {
                    "pass": "local_candidate",
                    "source_id": candidate_id,
                    "action": decision["decision"],
                }
            )
        target_obj["local_candidate_binding"] = {
            "operation": next(iter(operations)),
            "candidates": binding_candidates,
        }
        attached_object_count += 1

    return objects, {
        "schema_version": "scenemosaic_local_candidate_binding_audit",
        "candidate_count": len(candidates),
        "decision_count": len(pass_b_actions.get("local_candidate_decisions") or []),
        "decision_counts": dict(sorted(decision_counts.items())),
        "bound_object_count": attached_object_count,
        "binding_candidate_count": sum(
            len(
                (item.get("local_candidate_binding") or {}).get("candidates")
                or []
            )
            for item in objects
            if item.get("local_candidate_binding")
        ),
        "resolved_decisions": resolved,
        "hard_errors": hard_errors,
        "status": "pass" if not hard_errors else "fail",
    }
