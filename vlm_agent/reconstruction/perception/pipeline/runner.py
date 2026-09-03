#!/usr/bin/env python3
"""Run a two-pass VLM inventory overlapped with one reusable SAM3 session.

All intermediate writes stay inside an attempt-local staging tree; formal stages are
published only by the adoption layer. Pass A overlaps one local SAM3 top-zone recall probe;
Pass B then reviews the numbered local candidates while SAM3 segments Pass-A
objects with the same loaded session.  After fusion, only complete prompt
groups affected by Pass-B actions are recomputed.  Unaffected groups reuse
Pass-A masks and reviewed local candidates provide direct mask evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
for directory in (CODE_ROOT,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pipeline.core.candidate_review import (
    BOUNDED_POST_SAM3A_REVIEW_CONFIG,
    select_bounded_post_sam3a_candidates,
    materialize_bounded_post_sam3a_review,
    _semantic_tokens,
    _local_candidate_semantic_score,
    _local_candidate_pair_metrics,
    match_local_candidates_to_b_additions,
    compose_local_candidate_binding_actions,
    attach_local_candidate_bindings,
    promote_repeated_top_fixture_candidates,
    rebind_verified_local_candidates_to_failed_existing_objects,
)
from pipeline.core.inventory_fusion import (
    _prompt_groups,
    _object_prompt_map,
    validate_prompt_group_parity,
    compute_affected_prompt_audit,
    make_final_identity,
)
from pipeline.core.quality_gates import (
    _mask_sha256,
    build_bounded_candidate_semantic_recall_gate,
    build_container_content_completeness_gate,
    _tiny_ceiling_light_extent_tolerance,
    _repeated_recessed_ceiling_light_row_extent_tolerance,
    _small_flat_media_extent_tolerance,
    _thin_attached_shelf_extent_tolerance,
    _thin_wall_shelf_span_extent_tolerance,
    _simple_visible_object_extent_tolerance,
    _verified_lighting_fixture_extent_tolerance,
    build_locally_validated_assigned_evidence_overflow,
    resolve_assigned_evidence_semantic_gate_after_spatial_rescue,
    build_functional_assembly_gate,
    build_quality_audit,
    apply_delivery_quality_policy,
    _create_contact_sheet,
    _copy_final_masks,
)
from pipeline.core.publication import (
    atomic_publish,
    safe_name,
    canonicalize_final_object_assets,
)
from pipeline.core.pass_b_policy import (
    validate_b_actions,
    apply_post_pass_b_policy_guard,
)
from pipeline.core.runtime_io import (
    utc_now,
    read_json,
    atomic_json,
    sha256_file,
    snapshot_tree,
    compare_snapshots,
)
from pipeline.refinements import ambiguous_same_prompt_component_completion as AMBIGUOUS_SAME_PROMPT_COMPONENT_COMPLETION
from pipeline.refinements import bbox_failed_mask_rescue as BBOX_FAILED_MASK_RESCUE
from pipeline.refinements import nested_part_mask_suppression as NESTED_PART_MASK_SUPPRESSION
from pipeline.refinements import low_profile_accessory_suppression as LOW_PROFILE_ACCESSORY_SUPPRESSION
from pipeline.refinements import receptacle_content_mask_completion as RECEPTACLE_CONTENT_MASK_COMPLETION
from pipeline.refinements import content_host_review as CONTENT_HOST_REVIEW
from pipeline.refinements import host_part_mask_fusion as HOST_PART_MASK_FUSION
from pipeline.refinements import functional_assembly_refinement as FUNCTIONAL_ASSEMBLY_REFINEMENT
from pipeline.refinements import mask_ownership_resolution as MASK_OWNERSHIP_RESOLUTION
from pipeline.refinements import mobile_tiered_carrier_completion as MOBILE_TIERED_CARRIER_COMPLETION
from pipeline.refinements import named_local_instance_semantic_completion as NAMED_LOCAL_INSTANCE_SEMANTIC_COMPLETION
from pipeline.refinements import open_storage_content_prompt as OPEN_STORAGE_CONTENT_PROMPT
from pipeline.refinements import pass_a_same_mask_conflict as PASS_A_SAME_MASK_CONFLICT
from pipeline.refinements import post_delta_multiplicity_closure as POST_DELTA_MULTIPLICITY_CLOSURE
from pipeline.core import sam3d_delivery_policy as SAM3D_DELIVERY_POLICY
from segmentation import runtime as SAM3_SESSION
from pipeline.core import sam3_review_evidence as SAM3_REVIEW_EVIDENCE
from pipeline.refinements import strict_same_mask_duplicate_collapse as STRICT_SAME_MASK_DUPLICATE_COLLAPSE
from pipeline.refinements import top_zone_candidate_recall as TOP_ZONE

from inventory import pipeline as VLM


_DEFAULT_RUNTIME_ROOT = Path(
    os.environ.get(
        "SCENEMOSAIC_RUNTIME_ROOT",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "scenemosaic",
    )
).expanduser().resolve()
PERCEPTION_ATTEMPTS_ROOT = Path(
    os.environ.get(
        "SCENEMOSAIC_PERCEPTION_ATTEMPTS_ROOT",
        _DEFAULT_RUNTIME_ROOT / "perception_attempts",
    )
).expanduser().resolve()
DEFAULT_OUTPUT = PERCEPTION_ATTEMPTS_ROOT / "perception_run"
FORMAL_STAGE_NAMES = ("01_vlm_first", "02_sam3", "02_sam3_recall")
ASSIGNMENT_POLICY = "adaptive_local"
def resolve_merge_reviews_from_assigned_evidence(
    *,
    objects: Sequence[Mapping[str, Any]],
    assigned_evidence_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close a merge review without inventing a new semantic object.

    Only affirmative decisions (``pass`` or ``update_existing``) on the same
    Pass-A source are accepted, and the successor must have a final SAM3 mask.
    For a Pass-B-only addition, an extra adjacent SAM3 component has no prior
    semantic owner. Keep the declared primary component and defer the extra
    component as audit evidence rather than merging it or materializing an
    undeclared object. This avoids a second VLM call while retaining a
    source-traceable decision for a geometric ambiguity.
    """

    resolved_objects = [copy.deepcopy(dict(item)) for item in objects]
    decisions_by_source = {
        str(item.get("source_object_id") or ""): dict(item)
        for item in assigned_evidence_decisions
        if isinstance(item, Mapping)
        and str(item.get("decision") or "") in {"pass", "update_existing"}
        and str(item.get("source_object_id") or "")
    }
    records = []
    for obj in resolved_objects:
        sam3 = obj.get("sam3") or {}
        review = sam3.get("merge_review") or {}
        if not review.get("required") or sam3.get("status") != "success":
            continue
        mask_path = Path(str(sam3.get("mask_path") or ""))
        if not mask_path.is_file():
            continue
        source_passes = {
            str(source.get("pass") or "").upper()
            for source in obj.get("proposal_sources") or []
            if isinstance(source, Mapping)
        }
        if "B" in source_passes and "A" not in source_passes:
            review["required"] = False
            review["resolution_stage"] = (
                "pass_b_unclaimed_adjacent_component_deferred"
            )
            review["deferred_unclaimed_candidate_ids"] = list(
                review.get("ambiguous_candidate_ids") or []
            )
            sam3["merge_review"] = review
            obj["sam3"] = sam3
            records.append(
                {
                    "proposal_id": str(obj.get("object_id") or ""),
                    "resolution": review["resolution_stage"],
                    "deferred_candidate_ids": review[
                        "deferred_unclaimed_candidate_ids"
                    ],
                }
            )
            continue
        matched_source_id = next(
            (
                str(source.get("source_id") or "")
                for source in obj.get("proposal_sources") or []
                if isinstance(source, Mapping)
                and str(source.get("pass") or "").upper() == "A"
                and str(source.get("source_id") or "") in decisions_by_source
            ),
            "",
        )
        if not matched_source_id:
            continue
        decision = decisions_by_source[matched_source_id]
        review["required"] = False
        review["resolution_stage"] = "assigned_sam3_evidence_semantic_confirmation"
        review["assigned_evidence_resolution"] = {
            "source_object_id": matched_source_id,
            "evidence_id": str(decision.get("evidence_id") or ""),
            "decision": str(decision.get("decision") or ""),
        }
        sam3["merge_review"] = review
        obj["sam3"] = sam3
        records.append({
            "proposal_id": str(obj.get("object_id") or ""),
            **review["assigned_evidence_resolution"],
        })
    return {
        "schema_version": "scenemosaic_assigned_evidence_merge_resolution",
        "status": "pass",
        "resolved_count": len(records),
        "records": records,
        "hard_errors": [],
    }, resolved_objects


def resolve_cross_object_ambiguous_candidate_ownership(
    *,
    objects: Sequence[Mapping[str, Any]],
    unassigned_candidates: Sequence[Mapping[str, Any]],
    candidate_root: Path,
) -> dict[str, Any]:
    """Resolve merge-review candidates already owned by another final object."""

    from PIL import Image
    import numpy as np

    resolved_objects = [copy.deepcopy(dict(item)) for item in objects]
    candidates = {
        str(item.get("candidate_id") or ""): copy.deepcopy(dict(item))
        for item in unassigned_candidates
        if isinstance(item, Mapping)
    }
    root = candidate_root.expanduser().resolve()
    successful = [
        item
        for item in resolved_objects
        if str((item.get("sam3") or {}).get("status") or "") == "success"
    ]
    records: list[dict[str, Any]] = []
    hard_errors: list[dict[str, Any]] = []
    mask_cache: dict[str, np.ndarray] = {}

    def load_mask(path_value: Any, *, require_candidate_root: bool) -> np.ndarray:
        path = Path(str(path_value or "")).expanduser().resolve()
        if require_candidate_root and not path.is_relative_to(root):
            raise ValueError(f"candidate mask escapes root: {path}")
        key = str(path)
        if key not in mask_cache:
            if not path.is_file():
                raise FileNotFoundError(path)
            mask = np.array(Image.open(path).convert("L")) > 0
            if not mask.any():
                raise ValueError(f"empty mask: {path}")
            mask_cache[key] = mask
        return mask_cache[key]

    for obj in resolved_objects:
        sam3 = obj.get("sam3") or {}
        review = sam3.get("merge_review") or {}
        if not review.get("required"):
            continue
        ambiguous_ids = [
            str(value) for value in review.get("ambiguous_candidate_ids") or []
        ]
        ownership_records = []
        all_resolved = bool(ambiguous_ids)
        for candidate_id in ambiguous_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None:
                all_resolved = False
                continue
            try:
                candidate_mask = load_mask(
                    candidate.get("mask_path"), require_candidate_root=True
                )
            except (OSError, ValueError) as exc:
                hard_errors.append(
                    {
                        "type": "invalid_ambiguous_candidate_mask",
                        "candidate_id": candidate_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                all_resolved = False
                continue
            matches = []
            for owner in successful:
                if owner is obj:
                    continue
                try:
                    owner_mask = load_mask(
                        (owner.get("sam3") or {}).get("mask_path"),
                        require_candidate_root=False,
                    )
                except (OSError, ValueError):
                    continue
                if owner_mask.shape != candidate_mask.shape:
                    continue
                intersection = int(np.logical_and(candidate_mask, owner_mask).sum())
                if intersection == 0:
                    continue
                candidate_area = int(candidate_mask.sum())
                owner_area = int(owner_mask.sum())
                union = candidate_area + owner_area - intersection
                metrics = {
                    "iou": intersection / union if union else 0.0,
                    "candidate_coverage_by_owner": intersection / candidate_area,
                    "owner_coverage_by_candidate": intersection / owner_area,
                }
                if (
                    metrics["iou"] >= 0.90
                    and metrics["candidate_coverage_by_owner"] >= 0.95
                    and metrics["owner_coverage_by_candidate"] >= 0.90
                ):
                    matches.append((metrics["iou"], owner, metrics))
            matches.sort(key=lambda item: item[0], reverse=True)
            if not matches or (
                len(matches) > 1 and matches[0][0] - matches[1][0] < 0.02
            ):
                all_resolved = False
                continue
            _, owner, metrics = matches[0]
            ownership = {
                "candidate_id": candidate_id,
                "source_object_id": str(obj.get("object_id") or ""),
                "owned_by_object_id": str(owner.get("object_id") or ""),
                "iou": round(metrics["iou"], 6),
                "candidate_coverage_by_owner": round(
                    metrics["candidate_coverage_by_owner"], 6
                ),
                "owner_coverage_by_candidate": round(
                    metrics["owner_coverage_by_candidate"], 6
                ),
                "thresholds": {
                    "minimum_iou": 0.90,
                    "minimum_candidate_coverage_by_owner": 0.95,
                    "minimum_owner_coverage_by_candidate": 0.90,
                    "minimum_best_match_gap": 0.02,
                },
            }
            ownership_records.append(ownership)
            records.append(ownership)
        if all_resolved and len(ownership_records) == len(ambiguous_ids):
            review["required"] = False
            review["resolution_stage"] = "cross_object_candidate_mask_ownership"
            review["ownership_records"] = ownership_records
            sam3["merge_review"] = review
            obj["sam3"] = sam3

    audit = {
        "schema_version": "scenemosaic_cross_object_candidate_ownership",
        "status": "fail" if hard_errors else "pass",
        "resolved_candidate_count": len(records),
        "resolved_source_object_ids": sorted(
            {item["source_object_id"] for item in records}
        ),
        "records": records,
        "additional_sam3_inference_calls": 0,
        "additional_vlm_calls": 0,
        "formal_outputs_modified": False,
        "hard_errors": hard_errors,
    }
    return {"objects": resolved_objects, "audit": audit}



def _rewrite_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_paths(item, old_root, new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_paths(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        old_prefix = str(old_root)
        if value == old_prefix or value.startswith(old_prefix + os.sep):
            return str(new_root) + value[len(old_prefix):]
    return value


def remap_exact_proposal_id_values(
    value: Any, object_id_map: Mapping[str, str]
) -> Any:
    """Update exact proposal-ID fields while preserving evidence paths."""

    if isinstance(value, str):
        return object_id_map.get(value, value)
    if isinstance(value, Mapping):
        return {
            key: remap_exact_proposal_id_values(item, object_id_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            remap_exact_proposal_id_values(item, object_id_map)
            for item in value
        ]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pass-a-model", default="gpt-5.5")
    parser.add_argument("--pass-b-model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
        help="Pass-A reasoning effort.",
    )
    parser.add_argument(
        "--pass-b-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help=(
            "Pass-B reasoning effort; none is the production default for the "
            "compact structured protocol."
        ),
    )
    parser.add_argument("--max-image-long-edge", type=int, default=1280)
    parser.add_argument("--top-zone-prompt", default="ceiling light")
    parser.add_argument("--top-zone-max-candidates", type=int, default=8)
    parser.add_argument("--top-zone-panel-long-edge", type=int, default=1024)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=SAM3_SESSION.DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--reuse-local-pre-pass-b-cache-root",
        type=Path,
        default=None,
        help=(
            "Reuse one validated failed isolated attempt's canonical local SAM3-A "
            "registry and candidate evidence, then continue at bounded Pass B "
            "in a new empty attempt output. This never reissues Pass A or "
            "recomputes SAM3-A."
        ),
    )
    parser.add_argument("--content-host-max-hosts", type=int, default=8)
    parser.add_argument(
        "--content-host-panel-long-edge",
        type=int,
        default=1280,
    )
    parser.add_argument("--assigned-evidence-max-objects", type=int, default=12)
    parser.add_argument("--assigned-evidence-panel-long-edge", type=int, default=1280)
    parser.add_argument("--allow-third-party-upload", action="store_true")
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help=(
            "Block delivery on every observed quality finding. The default "
            "best_effort policy blocks only identity and file-integrity errors."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.content_host_max_hosts <= 8:
        raise ValueError("--content-host-max-hosts must be in [1, 8]")
    if not 768 <= args.content_host_panel_long_edge <= 1280:
        raise ValueError("--content-host-panel-long-edge must be in [768, 1280]")
    if not 1 <= args.assigned_evidence_max_objects <= 24:
        raise ValueError("--assigned-evidence-max-objects must be in [1, 24]")
    if not 768 <= args.assigned_evidence_panel_long_edge <= 1280:
        raise ValueError("--assigned-evidence-panel-long-edge must be in [768, 1280]")
    if not args.allow_third_party_upload:
        raise PermissionError(
            "--allow-third-party-upload is required for the external VLM calls"
        )
    if not str(args.top_zone_prompt).strip():
        raise ValueError("--top-zone-prompt must be non-empty")
    if not 1 <= args.top_zone_max_candidates <= 8:
        raise ValueError("--top-zone-max-candidates must be in [1, 8]")
    if not 512 <= args.top_zone_panel_long_edge <= 1280:
        raise ValueError("--top-zone-panel-long-edge must be in [512, 1280]")

    target = args.output_dir.expanduser().resolve()
    if (
        target != PERCEPTION_ATTEMPTS_ROOT
        and PERCEPTION_ATTEMPTS_ROOT not in target.parents
    ):
        raise ValueError(
            f"output directory must be below {PERCEPTION_ATTEMPTS_ROOT}"
        )
    if args.reuse_local_pre_pass_b_cache_root is not None:
        source = args.reuse_local_pre_pass_b_cache_root.expanduser().resolve()
        if source == target:
            raise ValueError(
                "cached pre-Pass-B source and output directory must differ"
            )
        if (
            source == PERCEPTION_ATTEMPTS_ROOT
            or PERCEPTION_ATTEMPTS_ROOT not in source.parents
        ):
            raise ValueError(
                "cached pre-Pass-B source must be below the perception attempts root"
            )
        if source.is_symlink() or not source.is_dir():
            raise ValueError(
                "cached pre-Pass-B source must be a real attempt directory"
            )
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set")


def _prepare_output(output_dir: Path, run_root: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    VLM.ensure_attempt_output(output_dir, run_root)
    return output_dir



def _openai_client() -> Any:
    """Construct a VLM client, using OpenAI's default gateway when unset."""

    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "timeout": 300.0,
        "max_retries": 0,
    }
    base_url = (
        os.environ.get("OPEN_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    if base_url and base_url.strip() and base_url.strip() != "None":
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _timed_pass_a(
    *,
    image_data_url: str,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    client = _openai_client()
    objects, call_info = VLM.run_pass_a(
        client=client,
        image_data_url=image_data_url,
        model=model,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
    )
    ended = time.monotonic()
    return {
        "objects": objects,
        "call_info": call_info,
        "start": started,
        "end": ended,
    }


def _valid_norm_bbox(value: Any) -> list[float] | None:
    try:
        box = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if (
        len(box) != 4
        or not all(math.isfinite(component) for component in box)
        or not (0.0 <= box[0] < box[2] <= 1.0)
        or not (0.0 <= box[1] < box[3] <= 1.0)
    ):
        return None
    return box


def _bbox_center_is_inside(candidate_bbox: Any, container_bbox: Any) -> bool:
    candidate = _valid_norm_bbox(candidate_bbox)
    container = _valid_norm_bbox(container_bbox)
    if candidate is None or container is None:
        return False
    center_x = (candidate[0] + candidate[2]) / 2.0
    center_y = (candidate[1] + candidate[3]) / 2.0
    return (
        container[0] <= center_x <= container[2]
        and container[1] <= center_y <= container[3]
    )


def _category_key(value: Any) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _bbox_coverage(inner_bbox: Any, outer_bbox: Any) -> float:
    inner = _valid_norm_bbox(inner_bbox)
    outer = _valid_norm_bbox(outer_bbox)
    if inner is None or outer is None:
        return 0.0
    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[2], outer[2])
    bottom = min(inner[3], outer[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return intersection / inner_area if inner_area else 0.0


def _prompt_declares_group(prompt: Any, category: Any) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(prompt or "").lower()))
    if tokens & {
        "group", "set", "collection", "cluster", "bundle",
        "pair", "row", "stack", "stacked", "pile",
    }:
        return True
    category_words = re.findall(r"[a-z0-9]+", str(category or "").lower())
    return any(f"{word}s" in tokens for word in category_words if word)


def _share_open_storage_host(
    left_bbox: Any,
    right_bbox: Any,
    hosts: Sequence[Mapping[str, Any]],
) -> bool:
    usable_hosts = [
        item for item in hosts
        if _valid_norm_bbox(item.get("bbox_norm_xyxy")) is not None
    ]
    if not usable_hosts:
        return True
    return any(
        _bbox_center_is_inside(left_bbox, host.get("bbox_norm_xyxy"))
        and _bbox_center_is_inside(right_bbox, host.get("bbox_norm_xyxy"))
        for host in usable_hosts
    )


def _filter_content_closure_additions(
    *,
    primary_actions: Mapping[str, Any],
    closure_actions: Mapping[str, Any],
    open_storage_hosts: Sequence[Mapping[str, Any]] = (),
    coverage_threshold: float = 0.80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only independent closure groups missed by main Pass B.

    A closure pass may see individual members inside a group already selected
    by main Pass B. Such members are neither missing nor valid new
    reconstruction units.  The prompt is asked to emit groups, and this gate
    independently rejects a singular closure item that would subdivide a
    same-category main group in the same verified open-storage host. It also
    rejects any closure group substantially covered by a main-pass group.
    """

    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1]")
    primary = [
        dict(item)
        for item in (primary_actions.get("additions") or [])
        if isinstance(item, Mapping)
        and _valid_norm_bbox(item.get("bbox_2d_norm_xyxy")) is not None
    ]
    accepted: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, raw in enumerate(closure_actions.get("additions") or [], start=1):
        if not isinstance(raw, Mapping):
            invalid.append({"closure_addition_index": index, "reason": "not_mapping"})
            continue
        addition = dict(raw)
        bbox = _valid_norm_bbox(addition.get("bbox_2d_norm_xyxy"))
        category = _category_key(addition.get("category"))
        if bbox is None or not category:
            invalid.append(
                {
                    "closure_addition_index": index,
                    "category": category,
                    "reason": "invalid_bbox_or_category",
                }
            )
            continue
        same_host_primary = [
            existing for existing in primary
            if _category_key(existing.get("category")) == category
            and _share_open_storage_host(
                bbox,
                existing.get("bbox_2d_norm_xyxy"),
                open_storage_hosts,
            )
        ]
        group_blockers = [
            existing for existing in same_host_primary
            if _prompt_declares_group(
                existing.get("sam3_prompt"), existing.get("category")
            )
        ]
        if group_blockers and not _prompt_declares_group(
            addition.get("sam3_prompt"), addition.get("category")
        ):
            suppressed.append(
                {
                    "closure_addition_index": index,
                    "category": category,
                    "sam3_prompt": str(addition.get("sam3_prompt") or ""),
                    "reason": "would_subdivide_main_pass_b_group",
                    "blockers": [
                        {
                            "sam3_prompt": str(existing.get("sam3_prompt") or ""),
                            "bbox": existing.get("bbox_2d_norm_xyxy"),
                        }
                        for existing in group_blockers
                    ],
                }
            )
            continue
        blockers = []
        for primary_index, existing in enumerate(primary, start=1):
            if existing not in same_host_primary:
                continue
            coverage = _bbox_coverage(bbox, existing.get("bbox_2d_norm_xyxy"))
            if coverage >= coverage_threshold:
                blockers.append(
                    {
                        "primary_addition_index": primary_index,
                        "sam3_prompt": str(existing.get("sam3_prompt") or ""),
                        "coverage": round(coverage, 6),
                    }
                )
        if blockers:
            suppressed.append(
                {
                    "closure_addition_index": index,
                    "category": category,
                    "sam3_prompt": str(addition.get("sam3_prompt") or ""),
                    "reason": "covered_by_main_pass_b_group",
                    "blockers": blockers,
                }
            )
            continue
        accepted.append(addition)
    audit = {
        "schema_version": "scenemosaic_content_closure_deduplication",
        "status": "pass",
        "coverage_threshold": coverage_threshold,
        "raw_addition_count": len(closure_actions.get("additions") or []),
        "accepted_addition_count": len(accepted),
        "suppressed_addition_count": len(suppressed),
        "suppressed": suppressed,
        "invalid": invalid,
        "hard_errors": [],
    }
    return accepted, audit


def _open_storage_hosts_for_content_closure(
    content_hosts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return every verified open-storage host for a second content audit.

    A crop is only attention context, not the physical interior of its owner.
    Using crop occupancy as a proxy for content lets nearby lamps or furniture
    hide an otherwise empty shelf. Every verified open-storage interior is
    therefore rechecked once against the current inventory. The review prompt
    asks for visible missing units only, so an empty compartment stays empty.
    """
    return [
        dict(item)
        for item in content_hosts
        if isinstance(item, Mapping)
        and str(item.get("host_mode") or "") == "open_storage"
    ]


def _build_content_closure_context(
    *,
    pass_a_objects: Sequence[Mapping[str, Any]],
    actions: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expose the current inventory to a closure-only Pass-B call.

    Compact Pass B uses temporary A-numbered identifiers. First-pass additions
    do not have formal IDs yet, so the closure context assigns temporary,
    attempt-local IDs after the Pass-A range. Closure actions may add objects
    only; update/removal output is discarded and cannot target those IDs.
    """
    current = [copy.deepcopy(dict(item)) for item in pass_a_objects]
    seen_ids = {str(item.get("object_id") or "") for item in current}
    numeric_ids = [
        int(match.group(1))
        for object_id in seen_ids
        if (match := re.fullmatch(r"A_(\d{3})", object_id))
    ]
    next_index = max(numeric_ids, default=0) + 1
    aliases: list[dict[str, Any]] = []
    for addition_index, raw_addition in enumerate(
        actions.get("additions") or [], start=1
    ):
        if not isinstance(raw_addition, Mapping):
            continue
        addition = copy.deepcopy(dict(raw_addition))
        if _valid_norm_bbox(addition.get("bbox_2d_norm_xyxy")) is None:
            continue
        temporary_id = f"A_{next_index:03d}"
        next_index += 1
        addition["object_id"] = temporary_id
        current.append(addition)
        aliases.append(
            {
                "temporary_object_id": temporary_id,
                "initial_pass_b_addition_index": addition_index,
                "category": str(addition.get("category") or ""),
                "bbox_2d_norm_xyxy": list(addition["bbox_2d_norm_xyxy"]),
            }
        )
    return current, aliases


def _renumber_content_hosts(
    hosts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for index, raw_host in enumerate(hosts, start=1):
        host = dict(raw_host)
        host["host_id"] = f"H_{index:03d}"
        result.append(host)
    return result


def _timed_pass_b(
    *,
    image_data_url: str,
    pass_a_objects: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    local_candidate_image_data_url: str | None = None,
    local_candidate_metadata: list[dict[str, Any]] | None = None,
    content_host_metadata: list[dict[str, Any]] | None = None,
    assigned_evidence_image_data_url: str | None = None,
    assigned_evidence_metadata: list[dict[str, Any]] | None = None,
    same_mask_conflicts: list[dict[str, Any]] | None = None,
    content_closure_only: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    client = _openai_client()
    actions, call_info, prompt = VLM.run_pass_b(
        client=client,
        image_data_url=image_data_url,
        pass_a_objects=pass_a_objects,
        model=model,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
        local_candidate_image_data_url=local_candidate_image_data_url,
        local_candidate_metadata=local_candidate_metadata,
        content_host_metadata=content_host_metadata,
        assigned_evidence_image_data_url=assigned_evidence_image_data_url,
        assigned_evidence_metadata=assigned_evidence_metadata,
        same_mask_conflicts=same_mask_conflicts,
        content_closure_only=content_closure_only,
    )
    ended = time.monotonic()
    return {
        "actions": actions,
        "call_info": call_info,
        "prompt": prompt,
        "start": started,
        "end": ended,
    }













def _run_bounded_post_sam3a_pass_b(
    *,
    image_path: Path,
    image_data_url: str,
    pass_a_objects: list[dict[str, Any]],
    sam3_a_registry: Mapping[str, Any],
    top_candidates: Sequence[Mapping[str, Any]],
    same_mask_conflicts: Sequence[Mapping[str, Any]] | None = None,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    review_dir: Path,
    assets_dir: Path,
    panel_long_edge: int,
    session: Any | None = None,
    content_host_max_hosts: int = 8,
    assigned_evidence_max_objects: int = 12,
    assigned_evidence_panel_long_edge: int = 1280,
) -> dict[str, Any]:
    """Materialize one bounded review panel, then call compact Pass B."""

    selection_start = time.monotonic()
    preselected_hosts: list[dict[str, Any]] = []
    container_content_recall = None
    working_sam3_registry = sam3_a_registry
    host_selection = CONTENT_HOST_REVIEW.select_content_hosts(
        pass_a_objects,
        config=CONTENT_HOST_REVIEW.ContentHostConfig(
            maximum_hosts=content_host_max_hosts
        ),
    )
    host_errors = list(host_selection.get("hard_errors") or [])
    if host_errors:
        raise RuntimeError(
            "content-host selection blocked before Pass B: "
            + json.dumps(host_errors, ensure_ascii=False)
        )
    preselected_hosts = list(host_selection.get("selected") or [])
    if session is None:
        raise ValueError(
            "fruit container completion requires the shared SAM3 session"
        )
    container_content_recall = (
        RECEPTACLE_CONTENT_MASK_COMPLETION
        .run_bounded_receptacle_content_mask_completion(
            session=session,
            infer_text_candidates_fn=SAM3_SESSION.infer_text_candidates,
            image_path=image_path,
            pass_a_objects=pass_a_objects,
            sam3_registry=working_sam3_registry,
            output_dir=review_dir.parent / "container_content_recall",
        )
    )
    working_sam3_registry = container_content_recall[
        "completed_registry"
    ]
    review = materialize_bounded_post_sam3a_review(
        image_path=image_path,
        top_candidates=top_candidates,
        sam3_registry=working_sam3_registry,
        output_dir=review_dir,
    )
    candidates = list(review["manifest"].get("candidates") or [])
    host_review = None
    hosts: list[dict[str, Any]] = []
    host_review = CONTENT_HOST_REVIEW.materialize_content_host_review(
        image_path=image_path,
        objects=pass_a_objects,
        output_dir=review_dir / "content_hosts",
        config=CONTENT_HOST_REVIEW.ContentHostConfig(
            maximum_hosts=content_host_max_hosts
        ),
    )
    hosts = list(host_review["manifest"].get("selected") or [])
    preselected_signature = [
        (item.get("source_object_id"), item.get("host_mode"))
        for item in preselected_hosts
    ]
    materialized_signature = [
        (item.get("source_object_id"), item.get("host_mode"))
        for item in hosts
    ]
    if materialized_signature != preselected_signature:
        raise RuntimeError(
            "content-host selection changed during materialization"
        )
    host_errors = list(host_review["manifest"].get("hard_errors") or [])
    if host_errors:
        raise RuntimeError(
            "content-host selection blocked before Pass B: "
            + json.dumps(host_errors, ensure_ascii=False)
        )

    assigned_evidence_review = None
    assigned_evidence_records: list[dict[str, Any]] = []
    assigned_evidence_asset = None
    assigned_evidence_data_url = None
    assigned_evidence_review = (
        SAM3_REVIEW_EVIDENCE.materialize_assigned_sam3_review(
            image_path=image_path,
            pass_a_objects=pass_a_objects,
            sam3_registry=working_sam3_registry,
            output_dir=review_dir / "assigned_sam3_evidence",
            config=SAM3_REVIEW_EVIDENCE.AssignedEvidenceConfig(
                maximum_objects=assigned_evidence_max_objects
            ),
        )
    )
    assigned_manifest = assigned_evidence_review["manifest"]
    assigned_errors = list(assigned_manifest.get("hard_errors") or [])
    if assigned_errors:
        raise RuntimeError(
            "assigned SAM3 evidence blocked before Pass B: "
            + json.dumps(assigned_errors, ensure_ascii=False)
        )
    assigned_evidence_records = list(
        assigned_manifest.get("records") or []
    )
    if assigned_evidence_records:
        assigned_evidence_asset = VLM.create_vlm_jpeg(
            assigned_evidence_review["panel_path"],
            assets_dir / "assigned_sam3_evidence_panel.jpg",
            max_long_edge=assigned_evidence_panel_long_edge,
            quality=84,
            coordinate_grid=False,
        )
        assigned_evidence_data_url = VLM.encode_data_url(
            Path(assigned_evidence_asset["path"]),
            assigned_evidence_asset["mime_type"],
        )

    panel_asset = None
    panel_data_url = None
    combined_panel = None
    panel_source = None
    if hosts:
        combined_panel = CONTENT_HOST_REVIEW.materialize_combined_review_panel(
            image_path=image_path,
            local_candidates=candidates,
            content_hosts=hosts,
            output_path=review_dir / "combined_review_panel.jpg",
        )
        panel_source = combined_panel["panel_path"]
    elif candidates:
        panel_source = review["panel_path"]
    if panel_source is not None:
        panel_asset = VLM.create_vlm_jpeg(
            panel_source,
            assets_dir / "merged_local_candidate_panel.jpg",
            max_long_edge=panel_long_edge,
            quality=84,
            coordinate_grid=False,
        )
        panel_data_url = VLM.encode_data_url(
            Path(panel_asset["path"]), panel_asset["mime_type"]
        )
    selection_end = time.monotonic()
    b_result = _timed_pass_b(
        image_data_url=image_data_url,
        pass_a_objects=pass_a_objects,
        model=model,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
        local_candidate_image_data_url=panel_data_url,
        local_candidate_metadata=candidates if candidates else None,
        content_host_metadata=hosts if hosts else None,
        assigned_evidence_image_data_url=assigned_evidence_data_url,
        assigned_evidence_metadata=(
            assigned_evidence_records if assigned_evidence_records else None
        ),
        same_mask_conflicts=(
            [dict(item) for item in same_mask_conflicts]
            if same_mask_conflicts else None
        ),
    )
    closure_hosts = _renumber_content_hosts(
        _open_storage_hosts_for_content_closure(hosts)
    )
    content_closure = {
        "status": "not_needed",
        "requested_host_source_ids": [],
        "addition_count": 0,
        "discarded_update_count": 0,
        "discarded_removal_count": 0,
        "current_inventory_aliases": [],
        "panel_asset": None,
    }
    if closure_hosts:
        closure_panel = CONTENT_HOST_REVIEW.materialize_combined_review_panel(
            image_path=image_path,
            local_candidates=[],
            content_hosts=closure_hosts,
            output_path=review_dir / "content_closure_panel.jpg",
        )
        closure_asset = VLM.create_vlm_jpeg(
            closure_panel["panel_path"],
            assets_dir / "content_closure_panel.jpg",
            max_long_edge=panel_long_edge,
            quality=84,
            coordinate_grid=False,
        )
        closure_data_url = VLM.encode_data_url(
            Path(closure_asset["path"]), closure_asset["mime_type"]
        )
        closure_context, closure_aliases = _build_content_closure_context(
            pass_a_objects=pass_a_objects,
            actions=b_result["actions"],
        )
        closure_result = _timed_pass_b(
            image_data_url=image_data_url,
            pass_a_objects=closure_context,
            model=model,
            reasoning_effort=reasoning_effort,
            output_dir=output_dir / "content_closure",
            local_candidate_image_data_url=closure_data_url,
            content_host_metadata=closure_hosts,
            content_closure_only=True,
        )
        closure_actions = closure_result["actions"]
        accepted_closure_additions, closure_deduplication_audit = (
            _filter_content_closure_additions(
                primary_actions=b_result["actions"],
                closure_actions=closure_actions,
                open_storage_hosts=closure_hosts,
            )
        )
        atomic_json(
            output_dir / "content_closure_deduplication_audit.json",
            closure_deduplication_audit,
        )
        merged_actions = copy.deepcopy(b_result["actions"])
        merged_actions["additions"].extend(
            copy.deepcopy(accepted_closure_additions)
        )
        b_result["actions"] = merged_actions
        b_result["call_info"]["content_closure"] = closure_result["call_info"]
        content_closure = {
            "status": "completed",
            "requested_host_source_ids": [
                str(host.get("source_object_id") or "")
                for host in closure_hosts
            ],
            "raw_addition_count": len(closure_actions.get("additions") or []),
            "addition_count": len(accepted_closure_additions),
            "discarded_update_count": len(closure_actions.get("updates") or []),
            "discarded_removal_count": len(closure_actions.get("removals") or []),
            "current_inventory_aliases": closure_aliases,
            "panel_asset": closure_asset["path"],
            "panel_manifest": closure_panel,
            "deduplication_audit": closure_deduplication_audit,
        }

    panel_count = 1 if panel_source is not None else 0
    assigned_evidence_image_count = (
        1 if assigned_evidence_data_url is not None else 0
    )
    return {
        "b_result": b_result,
        "review": review,
        "content_host_review": host_review,
        "container_content_recall": container_content_recall,
        "sam3_a_registry": working_sam3_registry,
        "assigned_evidence_review": assigned_evidence_review,
        "assigned_evidence_asset": assigned_evidence_asset,
        "combined_panel": combined_panel,
        "content_host_metadata": hosts,
        "local_candidate_metadata": candidates,
        "local_candidate_panel_asset": panel_asset,
        "selection_interval": {
            "start": selection_start,
            "end": selection_end,
        },
        "api_call_audit": {
            "mode": "bounded_post_sam3a",
            "pass_b_calls_made": 1 + (
                1 if content_closure["status"] == "completed" else 0
            ),
            "inherited_pass_b_calls": 0,
            "current_process_request_image_count": (
                1
                + panel_count
                + assigned_evidence_image_count
                + (2 if content_closure["status"] == "completed" else 0)
            ),
            "additional_vlm_calls": (
                1 if content_closure["status"] == "completed" else 0
            ),
            "logical_pipeline_vlm_api_calls": 2 + (
                1 if content_closure["status"] == "completed" else 0
            ),
            "api_calls_unchanged": content_closure["status"] != "completed",
            "full_scene_image_included": True,
            "total_request_image_count": (
                1
                + panel_count
                + assigned_evidence_image_count
                + (2 if content_closure["status"] == "completed" else 0)
            ),
            "review_panel_count": (
                panel_count
                + assigned_evidence_image_count
                + (1 if content_closure["status"] == "completed" else 0)
            ),
            "candidate_panel_count": panel_count if candidates else 0,
            "local_candidate_count": len(candidates),
            "content_host_tile_count": len(hosts),
            "container_content_completed_count": (
                container_content_recall["audit"].get("completed_count", 0)
                if container_content_recall is not None
                else 0
            ),
            "container_content_additional_sam3_inference_count": (
                container_content_recall["audit"].get("probe_count", 0)
                if container_content_recall is not None
                else 0
            ),
            "assigned_sam3_evidence_count": len(assigned_evidence_records),
            "assigned_sam3_evidence_image_count": assigned_evidence_image_count,
            "assigned_sam3_evidence_manifest": (
                assigned_evidence_review["manifest_path"]
                if assigned_evidence_review is not None else None
            ),
            "content_host_context_only": True,
            "content_closure": content_closure,
            "combined_review_panel": combined_panel,
            "candidate_sources": dict(
                review["selection_audit"].get(
                    "selected_counts_by_namespace", {}
                )
            ),
            "sam3_a_unassigned_included": bool(
                review["selection_audit"]
                .get("selected_counts_by_namespace", {})
                .get("sam3_a_unassigned", 0)
            ),
            "parallel_with_sam3_a": False,
            "original_top_only_panel_sent": False,
        },
    }






def _load_cached_pre_pass_b_bundle(
    *,
    cache_root: Path,
    config: Mapping[str, Any],
    top_zone_prompt: str,
    top_zone_max_candidates: int,
    allow_missing_unassigned_evidence: bool = False,
) -> dict[str, Any]:
    """Validate a stopped attempt cache for a no-SAM3-A continuation.

    Accept either a run stopped before Pass B or a run whose sole Pass-B
    response was unusable because it ended with ``finish_reason=length``.
    Canonical SAM3-A masks and unassigned candidate masks remain in the source
    attempt tree and are read-only evidence for a new, empty continuation tree.
    """

    source = cache_root.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == Path(config["run_root"]).resolve():
        raise ValueError("cached pre-Pass-B source may not be a formal run root")

    manifest_path = source / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "failed":
        raise ValueError("cached pre-Pass-B source must be a failed isolated attempt")
    if manifest.get("scene_name") != config["scene_name"]:
        raise ValueError("cached pre-Pass-B scene mismatch")
    if manifest.get("input_image") != str(Path(config["image_path"]).resolve()):
        raise ValueError("cached pre-Pass-B input image mismatch")
    pass_b_response_path = source / "pass_b" / "response_raw.txt"
    failed_length_pass_b = pass_b_response_path.is_file()
    if failed_length_pass_b:
        source_error = dict(manifest.get("error") or {})
        expected_error = "Pass B response did not finish cleanly: finish_reason='length'"
        if (
            source_error.get("type") != "RuntimeError"
            or source_error.get("message") != expected_error
        ):
            raise ValueError(
                "cached pre-Pass-B source made a non-recoverable Pass-B call"
            )

    required = {
        "asset": source / "assets" / "whole_scene_global_grid.jpg",
        "pass_a_objects": source / "pass_a" / "normalized_objects.json",
        "pass_a_call_info": source / "pass_a" / "call_info.json",
        "pass_a_bbox_review": source / "pass_a" / "bbox_review_assets" / "bbox_review_manifest.json",
        "top_raw": source / "work" / "top_zone_raw_candidates" / "text_candidates.json",
        "top_review": source / "work" / "top_zone_review" / "candidate_manifest.json",
        "sam3_a": source / "work" / "sam3_a" / "sam3_objects.json",
        "unassigned": source / "work" / "sam3_a" / "unassigned_candidates",
    }
    if allow_missing_unassigned_evidence:
        required.pop("unassigned")
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cached pre-Pass-B bundle is incomplete: {missing}")
    if any(path.is_symlink() for path in required.values()):
        raise ValueError("cached pre-Pass-B bundle may not use symlinked roots")

    def checked_evidence_path(raw_value: Any, label: str) -> Path:
        path = Path(str(raw_value or "")).expanduser().resolve()
        if source not in path.parents or not path.is_file():
            raise ValueError(f"cached {label} evidence escapes or is missing: {path}")
        return path

    objects_payload = read_json(required["pass_a_objects"])
    pass_a_objects = list(objects_payload.get("objects") or [])
    if not pass_a_objects:
        raise ValueError("cached pre-Pass-B Pass-A objects are empty")
    pass_a_ids = [str(item.get("object_id") or "") for item in pass_a_objects]
    if any(not item for item in pass_a_ids) or len(set(pass_a_ids)) != len(pass_a_ids):
        raise ValueError("cached pre-Pass-B Pass-A IDs are invalid")

    pass_a_info = read_json(required["pass_a_call_info"])
    raw_registry = read_json(required["top_raw"])
    if raw_registry.get("prompt") != top_zone_prompt:
        raise ValueError("cached pre-Pass-B top-zone prompt mismatch")
    review_manifest = read_json(required["top_review"])
    if review_manifest.get("source_image") != str(Path(config["image_path"]).resolve()):
        raise ValueError("cached pre-Pass-B top-zone image mismatch")
    local_candidates = list(review_manifest.get("candidates") or [])
    if len(local_candidates) > top_zone_max_candidates:
        raise ValueError("cached pre-Pass-B top-zone candidate count exceeds limit")
    expected_lc_ids = [f"LC_{index:03d}" for index in range(1, len(local_candidates) + 1)]
    if [str(item.get("candidate_id") or "") for item in local_candidates] != expected_lc_ids:
        raise ValueError("cached pre-Pass-B top-zone candidate IDs are invalid")
    for candidate in local_candidates:
        checked_evidence_path(candidate.get("mask_path"), "top-zone mask")
        checked_evidence_path(candidate.get("overlay_path"), "top-zone overlay")

    sam3_a_registry = read_json(required["sam3_a"])
    registry_objects = list(sam3_a_registry.get("objects") or [])
    registry_ids = [str(item.get("object_id") or "") for item in registry_objects]
    if set(registry_ids) != set(pass_a_ids):
        raise ValueError("cached pre-Pass-B SAM3-A registry IDs do not match Pass A")
    for item in registry_objects:
        sam3 = dict(item.get("sam3") or {})
        if sam3.get("status") == "success":
            checked_evidence_path(sam3.get("mask_path"), "SAM3-A mask")
            checked_evidence_path(sam3.get("overlay_path"), "SAM3-A overlay")
    for candidate in sam3_a_registry.get("unassigned_sam3_candidates") or []:
        if not allow_missing_unassigned_evidence:
            checked_evidence_path(candidate.get("mask_path"), "unassigned SAM3-A mask")
            checked_evidence_path(candidate.get("overlay_path"), "unassigned SAM3-A overlay")


    asset_path = checked_evidence_path(required["asset"], "whole-scene asset")
    pass_a_assets = read_json(required["pass_a_bbox_review"])
    return {
        "asset": {
            "path": str(asset_path),
            "mime_type": "image/jpeg",
            "reused_from": str(asset_path),
        },
        "pass_a_objects": pass_a_objects,
        "pass_a_info": {**pass_a_info, "inherited": True, "inherited_from": str(source)},
        "pass_a_assets": pass_a_assets,
        "top_raw_result": {"registry": raw_registry, "registry_path": str(required["top_raw"])},
        "top_review": {
            "manifest": review_manifest,
            "manifest_path": str(required["top_review"]),
            "panel_path": str(source / "work" / "top_zone_review" / "numbered_candidate_panel.jpg"),
        },
        "local_candidate_metadata": local_candidates,
        "sam3_a_registry": sam3_a_registry,
        "cache_provenance": {
            "schema_version": "scenemosaic_cached_pre_pass_b_resume",
            "source_root": str(source),
            "source_status": manifest.get("status"),
            "source_failed_pass_b_finish_reason": (
                "length" if failed_length_pass_b else None
            ),
            "reused_local_sam3_a": True,
            "source_sam3_a_object_count": len(registry_objects),
            "source_unassigned_candidate_count": len(sam3_a_registry.get("unassigned_sam3_candidates") or []),
            "formal_outputs_modified": False,
        },
    }






def _select_result_directory_name(quality_status: str) -> str:
    return "final" if quality_status == "pass" else "draft_final"


def main() -> int:
    args = parse_args()
    _validate_args(args)
    pass_b_reasoning_effort = (
        args.pass_b_reasoning_effort or args.reasoning_effort
    )
    cached_pre_pass_b_resume = args.reuse_local_pre_pass_b_cache_root is not None
    cached_pre_pass_b_bundle = None
    config = VLM.load_scene_config(args.scene_config)
    output_dir = _prepare_output(args.output_dir, config["run_root"])
    work_dir = output_dir / "work"
    pass_a_dir = output_dir / "pass_a"
    pass_b_dir = output_dir / "pass_b"
    assets_dir = output_dir / "assets"
    for directory in (work_dir, pass_a_dir, pass_b_dir, assets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    formal_paths = [
        config["run_root"] / name for name in FORMAL_STAGE_NAMES
    ]
    formal_before = snapshot_tree(formal_paths)
    atomic_json(output_dir / "formal_hashes_before.json", formal_before)
    manifest: dict[str, Any] = {
        "schema_version": "scenemosaic_perception_attempt",
        "status": "running",
        "created_at_utc": utc_now(),
        "scene_name": config["scene_name"],
        "scene_config": str(config["config_path"]),
        "input_image": str(config["image_path"]),
        "output_dir": str(output_dir),
        "formal_outputs_modified": False,
        "pass_a_model": args.pass_a_model,
        "pass_b_model": args.pass_b_model,
        "pass_b_response_protocol": VLM.COMPACT_PASS_B_PROTOCOL,
        "reasoning_effort": args.reasoning_effort,
        "pass_a_reasoning_effort": args.reasoning_effort,
        "pass_b_reasoning_effort": pass_b_reasoning_effort,
        "top_zone_recall_config": {
            "prompt": args.top_zone_prompt,
            "maximum_candidates": args.top_zone_max_candidates,
            "panel_long_edge": args.top_zone_panel_long_edge,
            "additional_vlm_calls": 0,
        },
        "bounded_post_sam3a_candidate_review_config": {
            "enabled": True,
            "mode": "bounded_post_sam3a",
            **BOUNDED_POST_SAM3A_REVIEW_CONFIG,
            "additional_vlm_calls": 0,
            "logical_pipeline_vlm_api_calls": 2,
            "api_calls_unchanged": True,
        },
        "fruit_container_completion_config": {
            "enabled": True,
            "default_enabled": True,
            "scope": "explicit_fruit_in_bowl_basket_tray_only",
            "plants_in_scope": False,
            "maximum_targets": 2,
            "resolution": "merge_recalled_fruit_into_new_container_mask",
            "preserve_successful_independent_fruit": True,
            "source_masks_overwritten": False,
            "additional_vlm_calls": 0,
        },
        "assigned_sam3_evidence_config": {
            "enabled": True,
            "maximum_objects": args.assigned_evidence_max_objects,
            "panel_long_edge": args.assigned_evidence_panel_long_edge,
            "additional_vlm_calls": 0,
            "maximum_additional_request_images": 1,
            "formal_outputs_modified": False,
        },
        "content_host_review_config": {
            "enabled": True,
            "maximum_hosts": args.content_host_max_hosts,
            "panel_long_edge": args.content_host_panel_long_edge,
            "context_only": True,
            "additional_vlm_calls": 0,
            "selected_mode": "bounded_post_sam3a",
            "candidate_source_policy": (
                "top_zone_plus_bounded_sam3a_unassigned"
            ),
            "supported_modes": ["bounded_post_sam3a"],
        },
        "open_storage_ownership_resolution_config": {
            "enabled": True,
            "default_enabled": True,
            "scope": "verified_open_storage_only",
            "source_masks_overwritten": False,
            "additional_sam3_inference_calls": 0,
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        },
        "ambiguous_same_prompt_component_completion_config": {
            "enabled": True,
            "scope": "saved_same_prompt_touching_ambiguous_sam3_components_only",
            "additional_sam3_inference_calls": 0,
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        },
        "named_local_instance_semantic_completion_config": {
            "enabled": True,
            "scope": "pass_b_named_carrier_local_instances_only",
            "maximum_targets": 24,
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        },
        "bbox_failed_mask_rescue_config": {
            "enabled": True,
            "default_enabled": True,
            "mode": "isolated_attempt_bbox_positive_prompt",
            "maximum_targets": (
                BBOX_FAILED_MASK_RESCUE.BboxFailedMaskRescueConfig.maximum_targets
            ),
            "maximum_candidates_per_target": (
                BBOX_FAILED_MASK_RESCUE.BboxFailedMaskRescueConfig
                .maximum_candidates_per_target
            ),
            "maximum_recessed_candidates_per_target": (
                BBOX_FAILED_MASK_RESCUE.BboxFailedMaskRescueConfig
                .maximum_recessed_candidates_per_target
            ),
            "semantic_label_status_after_rescue": "provisional",
            "maximum_additional_sam3_inference_calls": (
                BBOX_FAILED_MASK_RESCUE.BboxFailedMaskRescueConfig.maximum_targets
            ),
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        },
        "post_delta_multiplicity_closure_config": {
            "enabled": True,
            "default_enabled": True,
            "minimum_pass_b_peer_count": 2,
            "minimum_candidate_score": 0.85,
            "requires_shared_verified_host": True,
            "maximum_total_additions": 4,
            "additional_sam3_inference_calls": 0,
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        },
        "assignment_policy": ASSIGNMENT_POLICY,
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "manual_object_additions": 0,
    }
    atomic_json(output_dir / "run_manifest.json", manifest)

    try:
        if cached_pre_pass_b_resume:
            cached_pre_pass_b_bundle = _load_cached_pre_pass_b_bundle(
                cache_root=args.reuse_local_pre_pass_b_cache_root,
                config=config,
                top_zone_prompt=args.top_zone_prompt,
                top_zone_max_candidates=args.top_zone_max_candidates,
                allow_missing_unassigned_evidence=False,
            )
            asset = cached_pre_pass_b_bundle["asset"]
            pass_a_objects = cached_pre_pass_b_bundle["pass_a_objects"]
            pass_a_info = cached_pre_pass_b_bundle["pass_a_info"]
            pass_a_assets = cached_pre_pass_b_bundle["pass_a_assets"]
            top_raw_result = cached_pre_pass_b_bundle["top_raw_result"]
            top_review = cached_pre_pass_b_bundle["top_review"]
            local_candidate_metadata = cached_pre_pass_b_bundle[
                "local_candidate_metadata"
            ]
            local_candidate_panel_asset = None
            image_data_url = VLM.encode_data_url(
                Path(asset["path"]), asset["mime_type"]
            )
            local_candidate_image_data_url = None
            manifest["upload_asset"] = asset
            manifest["cached_pre_pass_b_provenance"] = cached_pre_pass_b_bundle[
                "cache_provenance"
            ]
            intervals = {}
            model_load_start = time.monotonic()
            session = SAM3_SESSION.load_session(
                model_dir=args.model_dir,
                device="cuda",
            )
            model_load_end = time.monotonic()
            intervals["model_load"] = {
                "start": model_load_start,
                "end": model_load_end,
            }
            atomic_json(
                output_dir / "top_zone_recall.json",
                {
                    "schema_version": "scenemosaic_top_zone_recall_orchestration",
                    "mode": "cached_pre_pass_b_resume",
                    "prompt": args.top_zone_prompt,
                    "raw_candidate_count": top_raw_result["registry"].get(
                        "candidate_count", 0
                    ),
                    "review_candidate_count": len(local_candidate_metadata),
                    "raw_registry": top_raw_result["registry_path"],
                    "review_manifest": top_review["manifest_path"],
                    "candidate_panel_asset": None,
                    "current_process_model_load_count": session.model_load_count,
                    "current_process_top_zone_inference_count": 0,
                    "current_process_sam3_a_inference_count": 0,
                    "vlm_api_calls_added_before_pass_b": 0,
                },
            )
            atomic_json(output_dir / "run_manifest.json", manifest)
        else:
            asset = VLM.create_vlm_jpeg(
                config["image_path"],
                assets_dir / "whole_scene_global_grid.jpg",
                max_long_edge=args.max_image_long_edge,
                quality=85,
                coordinate_grid=True,
                grid_divisions=10,
            )
            image_data_url = VLM.encode_data_url(
                Path(asset["path"]), asset["mime_type"]
            )
            manifest["upload_asset"] = asset

            intervals: dict[str, dict[str, float]] = {}
            with ThreadPoolExecutor(max_workers=1) as executor:
                a_future = executor.submit(
                    _timed_pass_a,
                    image_data_url=image_data_url,
                    model=args.pass_a_model,
                    reasoning_effort=args.reasoning_effort,
                    output_dir=pass_a_dir,
                )
                local_recall_start = time.monotonic()
                model_load_start = time.monotonic()
                session = SAM3_SESSION.load_session(
                    model_dir=args.model_dir,
                    device="cuda",
                )
                model_load_end = time.monotonic()
                top_raw_start = time.monotonic()
                top_raw_result = SAM3_SESSION.infer_text_candidates(
                    session,
                    config["image_path"],
                    args.top_zone_prompt,
                    work_dir / "top_zone_raw_candidates",
                )
                top_raw_end = time.monotonic()
                top_filtered = TOP_ZONE.filter_top_zone_candidates(
                    top_raw_result["registry"].get("candidates") or [],
                    image_size=tuple(top_raw_result["registry"]["image_size"]),
                    config=TOP_ZONE.TopZoneConfig(
                        maximum_candidates=args.top_zone_max_candidates
                    ),
                )
                top_review = TOP_ZONE.materialize_top_zone_review(
                    config["image_path"],
                    top_filtered,
                    work_dir / "top_zone_review",
                )
                local_candidate_metadata = list(
                    top_review["manifest"].get("candidates") or []
                )
                local_candidate_panel_asset = None
                local_candidate_image_data_url = None
                local_recall_end = time.monotonic()
                a_result = a_future.result()

            pass_a_objects = a_result["objects"]
            pass_a_info = a_result["call_info"]
            pass_a_start = a_result["start"]
            pass_a_end = a_result["end"]
            atomic_json(
                pass_a_dir / "objects.json",
                {
                    "schema_version": "scenemosaic_pipelined_pass_a",
                    "objects": pass_a_objects,
                },
            )
            pass_a_assets = VLM.render_bbox_review_assets(
                config["image_path"],
                pass_a_objects,
                pass_a_dir / "bbox_review_assets",
            )
            atomic_json(pass_a_dir / "call_info.json", pass_a_info)
            atomic_json(
                output_dir / "top_zone_recall.json",
                {
                    "schema_version": "scenemosaic_top_zone_recall_orchestration",
                    "mode": "fresh_parallel_run",
                    "prompt": args.top_zone_prompt,
                    "raw_candidate_count": top_raw_result["registry"].get(
                        "candidate_count", 0
                    ),
                    "review_candidate_count": len(local_candidate_metadata),
                    "raw_registry": top_raw_result["registry_path"],
                    "review_manifest": top_review["manifest_path"],
                    "candidate_panel_asset": local_candidate_panel_asset,
                    "model_load_count": session.model_load_count,
                    "vlm_api_calls_added": 0,
                },
            )
            intervals["pass_a"] = {"start": pass_a_start, "end": pass_a_end}
            intervals["model_load"] = {
                "start": model_load_start,
                "end": model_load_end,
            }
            intervals["top_zone_raw_sam3"] = {
                "start": top_raw_start,
                "end": top_raw_end,
            }
            intervals["top_zone_local_recall"] = {
                "start": local_recall_start,
                "end": local_recall_end,
            }

        normalization_result = (
            FUNCTIONAL_ASSEMBLY_REFINEMENT.normalize_pass_a_functional_assemblies(
                pass_a_objects
            )
        )
        functional_assembly_normalization = normalization_result["audit"]
        pass_a_objects = normalization_result["objects"]
        if functional_assembly_normalization.get("hard_errors"):
            atomic_json(
                output_dir / "functional_assembly_normalization_audit.json",
                functional_assembly_normalization,
            )
            raise RuntimeError(
                "functional assembly normalization failed closed before Pass B"
            )
        atomic_json(
            output_dir / "functional_assembly_normalization_audit.json",
            functional_assembly_normalization,
        )
        atomic_json(
            pass_a_dir / "normalized_objects.json",
            {
                "schema_version": "scenemosaic_normalized_pass_a",
                "objects": pass_a_objects,
            },
        )
        manifest["functional_assembly_normalization"] = {
            "status": functional_assembly_normalization.get("status"),
            "raw_object_count": functional_assembly_normalization.get(
                "raw_object_count", len(pass_a_objects)
            ),
            "normalized_object_count": len(pass_a_objects),
            "target_count": len(
                functional_assembly_normalization.get("targets") or []
            ),
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        }
        validate_prompt_group_parity(pass_a_objects)
        bounded_candidate_review = None
        bounded_candidate_api_audit = None
        pass_b_review_api_audit = None
        content_host_review = None
        container_content_candidate_recall = None
        functional_assembly_refinement = None
        fixed_bathtub_faucet_union = None
        fixed_component_action_policy = None
        pass_a_same_mask_conflict_detection = None
        pass_a_same_mask_conflict_resolution = None
        assigned_evidence_review = None
        assigned_evidence_asset = None
        pass_b_review_panel_asset = local_candidate_panel_asset
        pass_b_review_image_data_url = local_candidate_image_data_url
        local_branch_start = None
        local_branch_start = time.monotonic()
        top_candidate_metadata = list(local_candidate_metadata)
        sam3_a_start = time.monotonic()
        if cached_pre_pass_b_resume:
            if cached_pre_pass_b_bundle is None:
                raise AssertionError("cached pre-Pass-B bundle was not loaded")
            sam3_a_result = {
                "registry": dict(cached_pre_pass_b_bundle["sam3_a_registry"])
            }
            intervals["cached_pre_pass_b_sam3_a"] = {
                "start": sam3_a_start,
                "end": sam3_a_start,
                "source_root": cached_pre_pass_b_bundle["cache_provenance"][
                    "source_root"
                ],
                "source_object_count": cached_pre_pass_b_bundle[
                    "cache_provenance"
                ]["source_sam3_a_object_count"],
                "current_process_inference_count": 0,
                "formal_outputs_modified": False,
            }
        else:
            sam3_a_result = SAM3_SESSION.segment_registry(
                session,
                config["image_path"],
                pass_a_objects,
                work_dir / "sam3_a",
                assignment_policy=ASSIGNMENT_POLICY,
                scene_name=config["scene_name"],
                registry_metadata={
                    "source": "pipelined_pass_a_before_bounded_pass_b",
                    "image_object_registry_version": "pipelined_attempt_a",
                },
            )
        sam3_a_end = time.monotonic()
        local_branch_end = sam3_a_end
        functional_assembly_refinement_start = time.monotonic()
        functional_assembly_refinement = (
            FUNCTIONAL_ASSEMBLY_REFINEMENT.run_functional_assembly_roi_refinement(
                session=session,
                infer_text_candidates_fn=SAM3_SESSION.infer_text_candidates,
                image_path=config["image_path"],
                registry=sam3_a_result["registry"],
                normalization_audit=functional_assembly_normalization,
                output_dir=work_dir / "functional_assembly_roi_refinement",
            )
        )
        functional_assembly_refinement_end = time.monotonic()
        intervals["functional_assembly_roi_refinement"] = {
            "start": functional_assembly_refinement_start,
            "end": functional_assembly_refinement_end,
        }
        sam3_a_result["registry"] = functional_assembly_refinement[
            "registry"
        ]
        atomic_json(
            output_dir / "functional_assembly_roi_refinement_audit.json",
            functional_assembly_refinement["audit"],
        )
        if functional_assembly_refinement["audit"].get("hard_errors"):
            raise RuntimeError(
                "functional assembly ROI refinement failed closed before Pass B"
            )
        manifest["functional_assembly_roi_refinement"] = {
            "status": functional_assembly_refinement["audit"].get("status"),
            "target_count": functional_assembly_refinement["audit"].get(
                "target_count", 0
            ),
            "refined_count": functional_assembly_refinement["audit"].get(
                "refined_count", 0
            ),
            "inference_count": functional_assembly_refinement["audit"].get(
                "inference_count", 0
            ),
            "additional_model_load_count": 0,
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        }
        fixed_union_start = time.monotonic()
        fixed_bathtub_faucet_union = (
            FUNCTIONAL_ASSEMBLY_REFINEMENT.union_fixed_bathtub_faucet_masks(
                image_path=config["image_path"],
                registry=sam3_a_result["registry"],
                output_dir=work_dir / "fixed_bathtub_faucet_union",
            )
        )
        fixed_union_end = time.monotonic()
        intervals["fixed_bathtub_faucet_union"] = {
            "start": fixed_union_start,
            "end": fixed_union_end,
        }
        sam3_a_result["registry"] = fixed_bathtub_faucet_union["registry"]
        atomic_json(
            output_dir / "fixed_bathtub_faucet_union_audit.json",
            fixed_bathtub_faucet_union["audit"],
        )
        if fixed_bathtub_faucet_union["audit"].get("hard_errors"):
            raise RuntimeError(
                "fixed bathtub faucet union failed closed before Pass B"
            )
        for pass_a_object in pass_a_objects:
            object_update = (
                fixed_bathtub_faucet_union["audit"]
                .get("object_updates", {})
                .get(str(pass_a_object.get("object_id") or ""))
            )
            if object_update:
                pass_a_object.update(dict(object_update))
        validate_prompt_group_parity(pass_a_objects)
        manifest["fixed_bathtub_faucet_union"] = {
            "status": fixed_bathtub_faucet_union["audit"].get("status"),
            "accepted_count": fixed_bathtub_faucet_union["audit"].get(
                "accepted_count", 0
            ),
            "forced_removals": fixed_bathtub_faucet_union["audit"].get(
                "forced_removals", []
            ),
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        }

        plant_container_completion_start = time.monotonic()
        plant_container_mask_completion = (
            FUNCTIONAL_ASSEMBLY_REFINEMENT.complete_plant_container_masks(
                image_path=config["image_path"],
                registry=sam3_a_result["registry"],
                output_dir=work_dir / "plant_container_mask_completion",
            )
        )
        plant_container_completion_end = time.monotonic()
        intervals["plant_container_mask_completion"] = {
            "start": plant_container_completion_start,
            "end": plant_container_completion_end,
        }
        sam3_a_result["registry"] = plant_container_mask_completion["registry"]
        atomic_json(
            output_dir / "plant_container_mask_completion_audit.json",
            plant_container_mask_completion["audit"],
        )
        if plant_container_mask_completion["audit"].get("hard_errors"):
            raise RuntimeError(
                "plant-container mask completion failed closed before Pass B"
            )
        manifest["plant_container_mask_completion"] = {
            "status": plant_container_mask_completion["audit"].get("status"),
            "accepted_count": plant_container_mask_completion["audit"].get(
                "accepted_count", 0
            ),
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        }

        same_mask_detection_start = time.monotonic()
        pass_a_same_mask_conflict_detection = (
            PASS_A_SAME_MASK_CONFLICT.detect_pass_a_same_mask_conflicts(
                image_path=config["image_path"],
                registry=sam3_a_result["registry"],
                output_dir=work_dir / "pass_a_same_mask_conflicts",
            )
        )
        same_mask_detection_end = time.monotonic()
        intervals["pass_a_same_mask_conflict_detection"] = {
            "start": same_mask_detection_start,
            "end": same_mask_detection_end,
        }
        atomic_json(
            output_dir / "pass_a_same_mask_conflicts_audit.json",
            pass_a_same_mask_conflict_detection["audit"],
        )
        if pass_a_same_mask_conflict_detection["audit"].get("hard_errors"):
            raise RuntimeError(
                "Pass-A same-mask conflict detection failed closed before Pass B"
            )
        manifest["pass_a_same_mask_conflict_detection"] = {
            "status": pass_a_same_mask_conflict_detection["audit"].get(
                "status"
            ),
            "conflict_count": pass_a_same_mask_conflict_detection["audit"].get(
                "conflict_count", 0
            ),
            "additional_vlm_calls": 0,
            "additional_sam3_inference_count": 0,
            "formal_outputs_modified": False,
        }
        bounded_result = _run_bounded_post_sam3a_pass_b(
            image_path=config["image_path"],
            image_data_url=image_data_url,
            pass_a_objects=pass_a_objects,
            sam3_a_registry=sam3_a_result["registry"],
            top_candidates=top_candidate_metadata,
            same_mask_conflicts=(
                pass_a_same_mask_conflict_detection["conflicts"]
                if pass_a_same_mask_conflict_detection is not None
                else None
            ),
            model=args.pass_b_model,
            reasoning_effort=pass_b_reasoning_effort,
            output_dir=pass_b_dir,
            review_dir=work_dir / "bounded_post_sam3a_review",
            assets_dir=assets_dir,
            panel_long_edge=args.content_host_panel_long_edge,
            session=session,
            content_host_max_hosts=args.content_host_max_hosts,
            assigned_evidence_max_objects=(
                args.assigned_evidence_max_objects
            ),
            assigned_evidence_panel_long_edge=(
                args.assigned_evidence_panel_long_edge
            ),
        )
        bounded_candidate_review = bounded_result["review"]
        content_host_review = bounded_result["content_host_review"]
        container_content_candidate_recall = bounded_result[
            "container_content_recall"
        ]
        sam3_a_result["registry"] = bounded_result["sam3_a_registry"]
        if container_content_candidate_recall is not None:
            atomic_json(
                output_dir
                / "container_content_candidate_recall_audit.json",
                container_content_candidate_recall["audit"],
            )
        assigned_evidence_review = bounded_result[
            "assigned_evidence_review"
        ]
        assigned_evidence_asset = bounded_result[
            "assigned_evidence_asset"
        ]
        bounded_candidate_api_audit = bounded_result["api_call_audit"]
        pass_b_review_api_audit = bounded_candidate_api_audit
        local_candidate_metadata = bounded_result["local_candidate_metadata"]
        local_candidate_panel_asset = bounded_result[
            "local_candidate_panel_asset"
        ]
        pass_b_review_panel_asset = local_candidate_panel_asset
        intervals["bounded_candidate_selection"] = bounded_result[
            "selection_interval"
        ]
        b_result = bounded_result["b_result"]
        intervals["pass_b"] = {
            "start": b_result["start"],
            "end": b_result["end"],
        }
        pass_b_actions = b_result["actions"]
        pass_b_info = b_result["call_info"]
        atomic_json(pass_b_dir / "actions.json", pass_b_actions)
        atomic_json(
            output_dir / "bounded_candidate_api_call_audit.json",
            bounded_candidate_api_audit,
        )
        atomic_json(
            output_dir / "bounded_candidate_selection_audit.json",
            bounded_candidate_review["selection_audit"],
        )
        if content_host_review is not None:
            atomic_json(
                output_dir / "content_host_selection_audit.json",
                content_host_review["manifest"],
            )
        if assigned_evidence_review is not None:
            atomic_json(
                output_dir / "assigned_sam3_evidence_audit.json",
                assigned_evidence_review["manifest"],
            )

        intervals["model_load"] = {
            "start": model_load_start,
            "end": model_load_end,
        }
        intervals["sam3_a"] = {"start": sam3_a_start, "end": sam3_a_end}
        intervals["sam3_local_branch"] = {
            "start": local_branch_start,
            "end": local_branch_end,
        }
        validate_b_actions(
            pass_a_objects,
            pass_b_actions,
            expected_local_candidate_ids=[
                str(item["candidate_id"])
                for item in local_candidate_metadata
            ],
        )
        if fixed_bathtub_faucet_union is not None:
            fixed_component_action_policy = (
                FUNCTIONAL_ASSEMBLY_REFINEMENT.apply_fixed_component_absorption_actions(
                    pass_b_actions, fixed_bathtub_faucet_union["audit"]
                )
            )
            pass_b_actions = fixed_component_action_policy["actions"]
            atomic_json(
                output_dir / "fixed_component_action_policy_audit.json",
                fixed_component_action_policy["audit"],
            )
            validate_b_actions(
                pass_a_objects,
                pass_b_actions,
                expected_local_candidate_ids=[
                    str(item["candidate_id"])
                    for item in local_candidate_metadata
                ],
            )
        content_hosts = (
            list(
                (content_host_review.get("manifest") or {}).get("selected")
                or []
            )
            if content_host_review is not None
            else []
        )
        pass_b_action_guard_result = apply_post_pass_b_policy_guard(
            pass_a_objects,
            pass_b_actions,
            sam3_a_result["registry"],
            content_hosts=content_hosts,
        )
        pass_b_actions = pass_b_action_guard_result["actions"]
        pass_b_action_guard_audit = pass_b_action_guard_result["audit"]
        pass_b_actions, top_fixture_recall_audit = (
            promote_repeated_top_fixture_candidates(
                pass_a_objects=pass_a_objects,
                pass_b_actions=pass_b_actions,
                local_candidates=local_candidate_metadata,
            )
        )
        (
            pass_b_actions,
            failed_existing_object_rebinding_audit,
        ) = rebind_verified_local_candidates_to_failed_existing_objects(
            pass_a_objects=pass_a_objects,
            pass_b_actions=pass_b_actions,
            local_candidates=local_candidate_metadata,
            sam3_registry=sam3_a_result["registry"],
        )
        validate_b_actions(
            pass_a_objects,
            pass_b_actions,
            expected_local_candidate_ids=[
                str(item["candidate_id"])
                for item in local_candidate_metadata
            ],
        )
        atomic_json(
            output_dir / "pass_b_action_guard_audit.json",
            pass_b_action_guard_audit,
        )
        atomic_json(
            output_dir / "repeated_top_fixture_recall_audit.json",
            top_fixture_recall_audit,
        )
        atomic_json(
            output_dir / "failed_existing_object_local_candidate_rebinding_audit.json",
            failed_existing_object_rebinding_audit,
        )
        atomic_json(pass_b_dir / "effective_actions.json", pass_b_actions)
        if pass_a_same_mask_conflict_detection is not None:
            same_mask_resolution_start = time.monotonic()
            pass_a_same_mask_conflict_resolution = (
                PASS_A_SAME_MASK_CONFLICT.resolve_pass_a_same_mask_conflicts(
                    session=session,
                    infer_text_candidates_fn=SAM3_SESSION.infer_text_candidates,
                    image_path=config["image_path"],
                    registry=sam3_a_result["registry"],
                    actions=pass_b_actions,
                    conflicts=pass_a_same_mask_conflict_detection["conflicts"],
                    output_dir=work_dir / "pass_a_same_mask_conflict_resolution",
                )
            )
            same_mask_resolution_end = time.monotonic()
            intervals["pass_a_same_mask_conflict_resolution"] = {
                "start": same_mask_resolution_start,
                "end": same_mask_resolution_end,
            }
            sam3_a_result["registry"] = pass_a_same_mask_conflict_resolution[
                "registry"
            ]
            atomic_json(
                output_dir / "pass_a_same_mask_conflict_resolution_audit.json",
                pass_a_same_mask_conflict_resolution["audit"],
            )

        fused_objects, fusion_audit = VLM.fuse_passes(
            pass_a_objects, pass_b_actions
        )
        binding_actions = pass_b_actions
        local_match_audit = None
        if "local_candidate_decisions" not in pass_b_actions:
            local_decisions, local_match_audit = (
                match_local_candidates_to_b_additions(
                    fused_objects,
                    local_candidate_metadata,
                    explicitly_rejected_candidate_ids=(
                        pass_b_actions.get("local_candidate_rejections") or []
                    ),
                )
            )
            binding_actions = compose_local_candidate_binding_actions(
                pass_b_actions,
                local_decisions,
            )
        fused_objects, local_binding_audit = attach_local_candidate_bindings(
            fused_objects,
            binding_actions,
            local_candidate_metadata,
        )
        if local_match_audit is not None:
            local_binding_audit["local_candidate_match"] = local_match_audit
            local_binding_audit["hard_errors"].extend(
                local_match_audit["hard_errors"]
            )
            if local_match_audit["hard_errors"]:
                local_binding_audit["status"] = "fail"
        prompt_normalization_result = (
            OPEN_STORAGE_CONTENT_PROMPT
            .canonicalize_open_storage_content_prompts(
                objects=fused_objects,
                content_hosts=content_hosts,
            )
        )
        fused_objects = prompt_normalization_result["objects"]
        open_storage_content_prompt_audit = (
            prompt_normalization_result["audit"]
        )
        atomic_json(
            output_dir
            / "open_storage_content_prompt_normalization_audit.json",
            open_storage_content_prompt_audit,
        )
        fused_objects, identity = make_final_identity(fused_objects)
        affected_audit = compute_affected_prompt_audit(
            pass_a_objects,
            pass_b_actions,
            fused_objects,
        )
        affected_prompts = set(affected_audit["new_prompts"])
        validate_prompt_group_parity(fused_objects)
        atomic_json(output_dir / "fusion_audit.json", fusion_audit)
        if local_match_audit is not None:
            atomic_json(
                output_dir / "local_candidate_match_audit.json",
                local_match_audit,
            )
        atomic_json(
            output_dir / "local_candidate_binding_audit.json",
            local_binding_audit,
        )
        atomic_json(output_dir / "identity_manifest_provisional.json", identity)
        atomic_json(output_dir / "affected_prompt_groups.json", affected_audit)
        atomic_json(
            output_dir / "fused_proposals.json",
            {
                "schema_version": "scenemosaic_pipelined_proposals",
                "objects": fused_objects,
            },
        )
        fused_assets = VLM.render_bbox_review_assets(
            config["image_path"],
            fused_objects,
            output_dir / "fused_bbox_review_assets",
        )

        delta_result = None
        delta_start = time.monotonic()
        if affected_prompts:
            delta_result = SAM3_SESSION.segment_registry(
                session,
                config["image_path"],
                fused_objects,
                work_dir / "sam3_delta",
                assignment_policy=ASSIGNMENT_POLICY,
                prompt_filter=affected_prompts,
                scene_name=config["scene_name"],
                registry_metadata={
                    "source": "pipelined_pass_b_affected_groups",
                    "image_object_registry_version": "pipelined_attempt_delta",
                    "raw_candidate_evidence_storage": (
                        "ambiguity_and_repeated_instance_delta_candidates"
                    ),
                },
                capture_debug_candidates=False,
                capture_unassigned_candidates=False,
                capture_ambiguous_candidate_evidence=True,
                capture_post_delta_multiplicity_candidates=True,
            )
        delta_end = time.monotonic()
        intervals["sam3_delta"] = {"start": delta_start, "end": delta_end}

        staging_prefix = ".final_staging"
        staging_dir = output_dir / f"{staging_prefix}_{os.getpid()}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        finalize_start = time.monotonic()
        final_objects = _copy_final_masks(
            fused_objects=fused_objects,
            affected_prompts=affected_prompts,
            pass_a_registry=sam3_a_result["registry"],
            delta_registry=delta_result["registry"] if delta_result else None,
            staging_dir=staging_dir,
            image_path=config["image_path"],
        )
        with SAM3_SESSION._load_production_module().Image.open(
            config["image_path"]
        ) as source:
            image_np = SAM3_SESSION._load_production_module().np.array(
                source.convert("RGB")
            ).astype(SAM3_SESSION._load_production_module().np.uint8)
        multiplicity_start = time.monotonic()
        multiplicity_result = (
            POST_DELTA_MULTIPLICITY_CLOSURE.run_post_delta_multiplicity_closure(
                image_path=config["image_path"],
                objects=final_objects,
                delta_registry=delta_result["registry"] if delta_result else None,
                fallback_registry=sam3_a_result["registry"],
                work_dir=work_dir / "post_delta_multiplicity_closure",
                staging_dir=staging_dir,
                config=POST_DELTA_MULTIPLICITY_CLOSURE.MultiplicityClosureConfig(
                    enabled=True,
                    isolated_attempt_only=True,
                ),
            )
        )
        final_objects = multiplicity_result["objects"]
        multiplicity_audit = multiplicity_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        multiplicity_end = time.monotonic()
        intervals["post_delta_multiplicity_closure"] = {
            "start": multiplicity_start,
            "end": multiplicity_end,
        }
        multiplicity_audit_path = (
            output_dir / "post_delta_multiplicity_closure_audit.json"
        )
        atomic_json(multiplicity_audit_path, multiplicity_audit)
        contained_component_start = time.monotonic()
        contained_component_result = (
            NESTED_PART_MASK_SUPPRESSION.run_nested_part_mask_suppression(
                objects=final_objects,
                image_shape=tuple(image_np.shape[:2]),
                output_dir=work_dir / "contained_component_suppression",
                config=(
                    NESTED_PART_MASK_SUPPRESSION
                    .NestedPartMaskSuppressionConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = contained_component_result["objects"]
        contained_component_audit = contained_component_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        contained_component_end = time.monotonic()
        intervals["contained_component_suppression"] = {
            "start": contained_component_start,
            "end": contained_component_end,
        }
        contained_component_audit_path = (
            output_dir / "contained_component_suppression_audit.json"
        )
        atomic_json(contained_component_audit_path, contained_component_audit)

        low_profile_accessory_start = time.monotonic()
        low_profile_accessory_result = (
            LOW_PROFILE_ACCESSORY_SUPPRESSION
            .run_low_profile_accessory_suppression(
                objects=final_objects,
                image_shape=tuple(image_np.shape[:2]),
                output_dir=work_dir / "low_profile_accessory_suppression",
                config=(
                    LOW_PROFILE_ACCESSORY_SUPPRESSION
                    .LowProfileAccessoryConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = low_profile_accessory_result["objects"]
        low_profile_accessory_audit = low_profile_accessory_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        low_profile_accessory_end = time.monotonic()
        intervals["low_profile_accessory_suppression"] = {
            "start": low_profile_accessory_start,
            "end": low_profile_accessory_end,
        }
        atomic_json(
            output_dir / "low_profile_accessory_suppression_audit.json",
            low_profile_accessory_audit,
        )

        strict_duplicate_start = time.monotonic()
        strict_duplicate_result = (
            STRICT_SAME_MASK_DUPLICATE_COLLAPSE.run_strict_same_mask_duplicate_collapse(
                objects=final_objects,
                retry_registry=None,
                retry_audit=None,
                image_shape=tuple(image_np.shape[:2]),
                output_dir=(
                    work_dir / "strict_same_mask_duplicate_collapse"
                ),
                config=(
                    STRICT_SAME_MASK_DUPLICATE_COLLAPSE
                    .StrictSameMaskDuplicateCollapseConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = strict_duplicate_result["objects"]
        strict_duplicate_audit = strict_duplicate_result["audit"]

        strict_duplicate_end = time.monotonic()
        intervals["strict_same_mask_duplicate_collapse"] = {
            "start": strict_duplicate_start,
            "end": strict_duplicate_end,
        }
        atomic_json(
            output_dir / "strict_same_mask_duplicate_collapse_audit.json",
            strict_duplicate_audit,
        )
        bbox_rescue_start = time.monotonic()
        bbox_rescue_result = BBOX_FAILED_MASK_RESCUE.run_bbox_failed_mask_rescue(
            session=session,
            image_path=config["image_path"],
            objects=final_objects,
            work_dir=work_dir / "bbox_failed_mask_rescue",
            staging_dir=staging_dir,
            infer_candidates_fn=SAM3_SESSION.infer_text_candidates,
            config=BBOX_FAILED_MASK_RESCUE.BboxFailedMaskRescueConfig(
                enabled=True,
                isolated_attempt_only=True,
                priority_source_object_ids=frozenset(
                    str(item.get("source_object_id") or "")
                    for item in (
                        pass_b_actions.get("assigned_evidence_decisions") or []
                    )
                    if isinstance(item, Mapping)
                    and str(item.get("source_object_id") or "")
                ),
            ),
            unresolved_source_object_ids=frozenset(
                str(item.get("source_object_id") or "")
                for item in (
                    pass_b_actions.get("assigned_evidence_decisions") or []
                )
                if isinstance(item, Mapping)
                and item.get("decision") == "uncertain"
                and str(item.get("source_object_id") or "")
            ),
        )
        final_objects = bbox_rescue_result["objects"]
        bbox_rescue_audit = bbox_rescue_result["audit"]

        bbox_rescue_end = time.monotonic()
        intervals["bbox_failed_mask_rescue"] = {
            "start": bbox_rescue_start,
            "end": bbox_rescue_end,
        }
        bbox_rescue_audit_path = output_dir / "bbox_failed_mask_rescue_audit.json"
        atomic_json(bbox_rescue_audit_path, bbox_rescue_audit)

        # Text-mask failures are repaired only here, after the pre-Pass-B
        # completion has already run.  Re-run the same deterministic
        # plant-container check against the rescued objects so that this route
        # cannot bypass a declared foliage-plus-vessel assembly.
        post_bbox_plant_container_start = time.monotonic()
        post_bbox_plant_container_registry = copy.deepcopy(
            sam3_a_result["registry"]
        )
        post_bbox_plant_container_registry["objects"] = copy.deepcopy(
            final_objects
        )
        post_bbox_plant_container_result = (
            FUNCTIONAL_ASSEMBLY_REFINEMENT.complete_plant_container_masks(
                image_path=config["image_path"],
                registry=post_bbox_plant_container_registry,
                output_dir=(
                    work_dir / "plant_container_mask_completion_after_bbox_rescue"
                ),
            )
        )
        post_bbox_plant_container_end = time.monotonic()
        intervals["plant_container_mask_completion_after_bbox_rescue"] = {
            "start": post_bbox_plant_container_start,
            "end": post_bbox_plant_container_end,
        }
        final_objects = post_bbox_plant_container_result["registry"]["objects"]
        post_bbox_plant_container_audit = post_bbox_plant_container_result[
            "audit"
        ]
        post_bbox_plant_container_audit_path = (
            output_dir / "plant_container_mask_completion_after_bbox_rescue_audit.json"
        )
        atomic_json(
            post_bbox_plant_container_audit_path,
            post_bbox_plant_container_audit,
        )
        if post_bbox_plant_container_audit.get("hard_errors"):
            raise RuntimeError(
                "plant-container mask completion failed closed after bbox rescue"
            )
        manifest["plant_container_mask_completion_after_bbox_rescue"] = {
            "status": post_bbox_plant_container_audit.get("status"),
            "accepted_count": post_bbox_plant_container_audit.get(
                "accepted_count", 0
            ),
            "additional_vlm_calls": 0,
            "formal_outputs_modified": False,
        }
        ambiguous_completion_start = time.monotonic()
        ambiguous_completion_result = (
            AMBIGUOUS_SAME_PROMPT_COMPONENT_COMPLETION
            .run_ambiguous_same_prompt_component_completion(
                image_path=config["image_path"],
                objects=final_objects,
                sam3_a_registry=sam3_a_result["registry"],
                staging_dir=staging_dir,
                work_dir=work_dir / "ambiguous_same_prompt_component_completion",
                config=(
                    AMBIGUOUS_SAME_PROMPT_COMPONENT_COMPLETION
                    .AmbiguousSamePromptCompletionConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = ambiguous_completion_result["objects"]
        ambiguous_completion_audit = ambiguous_completion_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        ambiguous_completion_end = time.monotonic()
        intervals["ambiguous_same_prompt_component_completion"] = {
            "start": ambiguous_completion_start,
            "end": ambiguous_completion_end,
        }
        atomic_json(
            output_dir / "ambiguous_same_prompt_component_completion_audit.json",
            ambiguous_completion_audit,
        )
        named_local_completion_start = time.monotonic()
        named_local_completion_result = (
            NAMED_LOCAL_INSTANCE_SEMANTIC_COMPLETION
            .run_named_local_instance_semantic_completion(
                session=session,
                image_path=config["image_path"],
                objects=final_objects,
                local_candidate_metadata=local_candidate_metadata,
                sam3_a_registry=sam3_a_result["registry"],
                staging_dir=staging_dir,
                work_dir=work_dir / "named_local_instance_semantic_completion",
                infer_candidates_fn=SAM3_SESSION.infer_text_candidates,
                config=(
                    NAMED_LOCAL_INSTANCE_SEMANTIC_COMPLETION
                    .NamedLocalSemanticCompletionConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = named_local_completion_result["objects"]
        named_local_completion_audit = named_local_completion_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        named_local_completion_end = time.monotonic()
        intervals["named_local_instance_semantic_completion"] = {
            "start": named_local_completion_start,
            "end": named_local_completion_end,
        }
        atomic_json(
            output_dir / "named_local_instance_semantic_completion_audit.json",
            named_local_completion_audit,
        )
        mobile_tier_start = time.monotonic()
        mobile_tier_result = (
            MOBILE_TIERED_CARRIER_COMPLETION
            .run_mobile_tiered_carrier_completion(
                session=session,
                image_path=config["image_path"],
                objects=final_objects,
                local_candidate_metadata=local_candidate_metadata,
                sam3_a_registry=sam3_a_result["registry"],
                staging_dir=staging_dir,
                work_dir=work_dir / "mobile_tiered_carrier_completion",
                infer_candidates_fn=SAM3_SESSION.infer_text_candidates,
                config=(
                    MOBILE_TIERED_CARRIER_COMPLETION
                    .MobileTieredCarrierCompletionConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = mobile_tier_result["objects"]
        mobile_tier_audit = mobile_tier_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        mobile_tier_end = time.monotonic()
        intervals["mobile_tiered_carrier_completion"] = {
            "start": mobile_tier_start,
            "end": mobile_tier_end,
        }
        atomic_json(
            output_dir / "mobile_tiered_carrier_completion_audit.json",
            mobile_tier_audit,
        )
        production = SAM3_SESSION._load_production_module()
        bed_pillow_edits = production.exclude_independent_pillows_from_beds(
            final_objects, image_np
        )
        ownership_start = time.monotonic()
        ownership_resolution_audit = {
            "schema_version": "scenemosaic_open_storage_ownership_resolution",
            "status": "disabled",
            "changed_host_count": 0,
            "changed_host_object_ids": [],
            "formal_outputs_modified": False,
            "hard_errors": [],
        }
        ownership_result = (
            MASK_OWNERSHIP_RESOLUTION.resolve_open_storage_ownership(
                image_path=config["image_path"],
                objects=final_objects,
                output_dir=work_dir / "open_storage_ownership_resolution",
            )
        )
        final_objects = ownership_result["objects"]
        ownership_resolution_audit = ownership_result["audit"]
        ownership_end = time.monotonic()
        intervals["open_storage_ownership_resolution"] = {
            "start": ownership_start,
            "end": ownership_end,
        }
        atomic_json(
            output_dir / "open_storage_ownership_resolution_audit.json",
            ownership_resolution_audit,
        )
        assigned_evidence_merge_resolution, final_objects = (
            resolve_merge_reviews_from_assigned_evidence(
                objects=final_objects,
                assigned_evidence_decisions=(
                    pass_b_actions.get("assigned_evidence_decisions") or []
                ),
            )
        )
        atomic_json(
            output_dir / "assigned_evidence_merge_resolution_audit.json",
            assigned_evidence_merge_resolution,
        )
        cross_object_candidate_ownership = (
            resolve_cross_object_ambiguous_candidate_ownership(
                objects=final_objects,
                unassigned_candidates=(
                    sam3_a_result["registry"].get(
                        "unassigned_sam3_candidates", []
                    )
                ),
                candidate_root=work_dir / "sam3_a" / "unassigned_candidates",
            )
        )
        final_objects = cross_object_candidate_ownership["objects"]
        cross_object_candidate_ownership_audit = (
            cross_object_candidate_ownership["audit"]
        )
        atomic_json(
            output_dir / "cross_object_candidate_ownership_audit.json",
            cross_object_candidate_ownership_audit,
        )
        dedicated_attachment_start = time.monotonic()
        dedicated_attachment_result = (
            HOST_PART_MASK_FUSION.run_host_part_mask_fusion(
                image_np=image_np,
                image_path=config["image_path"],
                session=session,
                infer_candidates_fn=SAM3_SESSION.infer_text_candidates,
                objects=final_objects,
                staging_dir=staging_dir,
                work_dir=work_dir / "dedicated_attachment_absorption",
                config=(
                    HOST_PART_MASK_FUSION
                    .HostPartMaskFusionConfig(
                        enabled=True,
                        isolated_attempt_only=True,
                    )
                ),
            )
        )
        final_objects = dedicated_attachment_result["objects"]
        dedicated_attachment_audit = dedicated_attachment_result["audit"]
        final_objects, identity = make_final_identity(final_objects)
        (
            final_objects,
            canonical_final_asset_publication_audit,
        ) = canonicalize_final_object_assets(
            objects=final_objects,
            staging_dir=staging_dir,
            image_np=image_np,
        )
        atomic_json(
            output_dir / "canonical_final_asset_publication_audit.json",
            canonical_final_asset_publication_audit,
        )
        dedicated_attachment_end = time.monotonic()
        intervals["dedicated_attachment_absorption"] = {
            "start": dedicated_attachment_start,
            "end": dedicated_attachment_end,
        }
        atomic_json(
            output_dir / "dedicated_attachment_absorption_audit.json",
            dedicated_attachment_audit,
        )
        production.Image.fromarray(
            production.rebuild_scene_overlay(final_objects, image_np)
        ).save(staging_dir / "sam3_overlay.png")
        _create_contact_sheet(
            final_objects,
            staging_dir / "sam3_overlay_contact_sheet.jpg",
        )

        quality = build_quality_audit(final_objects, final_objects, identity)
        quality["canonical_final_asset_publication"] = (
            canonical_final_asset_publication_audit
        )
        quality["hard_errors"].extend(
            canonical_final_asset_publication_audit.get("hard_errors") or []
        )
        quality["assigned_evidence_merge_resolution"] = (
            assigned_evidence_merge_resolution
        )
        quality["local_candidate_binding_audit"] = local_binding_audit
        quality["hard_errors"].extend(local_binding_audit["hard_errors"])
        quality["failed_existing_object_local_candidate_rebinding"] = (
            failed_existing_object_rebinding_audit
        )
        quality["pass_b_action_guard"] = pass_b_action_guard_audit
        quality["hard_errors"].extend(
            pass_b_action_guard_audit.get("hard_errors") or []
        )
        quality["cross_object_candidate_ownership"] = (
            cross_object_candidate_ownership_audit
        )
        quality["hard_errors"].extend(
            cross_object_candidate_ownership_audit.get("hard_errors") or []
        )
        quality["dedicated_attachment_absorption"] = (
            dedicated_attachment_audit
        )
        quality["hard_errors"].extend(
            dedicated_attachment_audit.get("hard_errors") or []
        )
        quality["functional_assembly_normalization"] = (
            functional_assembly_normalization
        )
        quality["hard_errors"].extend(
            functional_assembly_normalization.get("hard_errors") or []
        )
        quality["open_storage_content_prompt_normalization"] = (
            open_storage_content_prompt_audit
        )
        quality["hard_errors"].extend(
            open_storage_content_prompt_audit.get("hard_errors") or []
        )
        if functional_assembly_refinement is not None:
            quality["functional_assembly_roi_refinement"] = (
                functional_assembly_refinement["audit"]
            )
            quality["hard_errors"].extend(
                functional_assembly_refinement["audit"].get("hard_errors") or []
            )
        else:
            quality["functional_assembly_roi_refinement"] = {
                "status": "disabled_or_not_applicable",
                "hard_errors": [],
            }

        quality["strict_same_mask_duplicate_collapse"] = strict_duplicate_audit
        quality["hard_errors"].extend(
            strict_duplicate_audit.get("hard_errors") or []
        )
        quality["contained_component_suppression"] = contained_component_audit
        quality["hard_errors"].extend(
            contained_component_audit.get("hard_errors") or []
        )
        quality["low_profile_accessory_suppression"] = (
            low_profile_accessory_audit
        )
        quality["hard_errors"].extend(
            low_profile_accessory_audit.get("hard_errors") or []
        )
        quality["bbox_failed_mask_rescue"] = bbox_rescue_audit
        quality["hard_errors"].extend(
            bbox_rescue_audit.get("hard_errors") or []
        )
        quality["plant_container_mask_completion_after_bbox_rescue"] = (
            post_bbox_plant_container_audit
        )
        quality["hard_errors"].extend(
            post_bbox_plant_container_audit.get("hard_errors") or []
        )
        quality["ambiguous_same_prompt_component_completion"] = (
            ambiguous_completion_audit
        )
        quality["hard_errors"].extend(
            ambiguous_completion_audit.get("hard_errors") or []
        )
        quality["named_local_instance_semantic_completion"] = (
            named_local_completion_audit
        )
        quality["hard_errors"].extend(
            named_local_completion_audit.get("hard_errors") or []
        )
        quality["mobile_tiered_carrier_completion"] = mobile_tier_audit
        quality["hard_errors"].extend(
            mobile_tier_audit.get("hard_errors") or []
        )
        quality["post_delta_multiplicity_closure"] = multiplicity_audit
        quality["hard_errors"].extend(
            multiplicity_audit.get("hard_errors") or []
        )
        if pass_a_same_mask_conflict_detection is not None:
            quality["pass_a_same_mask_conflict_detection"] = (
                pass_a_same_mask_conflict_detection["audit"]
            )
            quality["hard_errors"].extend(
                pass_a_same_mask_conflict_detection["audit"].get("hard_errors")
                or []
            )
        else:
            quality["pass_a_same_mask_conflict_detection"] = {
                "status": "disabled_or_not_bounded_review",
                "hard_errors": [],
            }
        if pass_a_same_mask_conflict_resolution is not None:
            quality["pass_a_same_mask_conflict_resolution"] = (
                pass_a_same_mask_conflict_resolution["audit"]
            )
            quality["hard_errors"].extend(
                pass_a_same_mask_conflict_resolution["audit"].get("hard_errors")
                or []
            )
        else:
            quality["pass_a_same_mask_conflict_resolution"] = {
                "status": "not_applicable",
                "hard_errors": [],
            }
        quality["open_storage_ownership_resolution"] = (
            ownership_resolution_audit
        )
        quality["hard_errors"].extend(
            ownership_resolution_audit.get("hard_errors") or []
        )
        if container_content_candidate_recall is not None:
            container_content_gate = build_container_content_completeness_gate(
                container_content_candidate_recall["audit"]
            )
            quality["container_content_candidate_recall"] = (
                container_content_candidate_recall["audit"]
            )
        else:
            container_content_gate = (
                build_container_content_completeness_gate(None)
            )
            quality["container_content_candidate_recall"] = {
                "status": "disabled",
                "hard_errors": [],
            }
        quality["container_content_completeness_gate"] = container_content_gate
        quality["hard_errors"].extend(
            container_content_gate.get("hard_errors") or []
        )
        if assigned_evidence_review is None:
            quality["hard_errors"].append(
                {"type": "missing_assigned_sam3_evidence_review"}
            )
            quality["assigned_sam3_evidence"] = {"status": "missing"}
        else:
            assigned_manifest = assigned_evidence_review["manifest"]
            quality["assigned_sam3_evidence"] = {
                "status": assigned_manifest.get("status"),
                "selected_count": assigned_manifest.get("selected_count", 0),
                "overflow_count": assigned_manifest.get("overflow_count", 0),
                "suspicious_overflow_count": assigned_manifest.get(
                    "suspicious_overflow_count", 0
                ),
                "audit_path": str(
                    output_dir / "assigned_sam3_evidence_audit.json"
                ),
                "additional_vlm_calls": 0,
            }
            quality["hard_errors"].extend(
                assigned_manifest.get("hard_errors") or []
            )
            semantic_gate = pass_b_actions.get(
                "assigned_evidence_semantic_gate"
            )
            evidence_decisions = list(
                pass_b_actions.get("assigned_evidence_decisions") or []
            )
            quality["assigned_sam3_evidence"][
                "semantic_gate"
            ] = semantic_gate
            quality["assigned_sam3_evidence"][
                "decision_count"
            ] = len(evidence_decisions)
            expected_evidence_count = int(
                assigned_manifest.get("selected_count", 0)
            )
            if (
                not isinstance(semantic_gate, Mapping)
                or len(evidence_decisions) != expected_evidence_count
                or not semantic_gate.get("complete_partition")
            ):
                quality["hard_errors"].append(
                    {
                        "type": "assigned_sam3_semantic_gate_incomplete",
                        "expected_count": expected_evidence_count,
                        "actual_count": len(evidence_decisions),
                    }
                )
            downstream_spatial_rescue_source_ids = sorted(
                {
                    str(source.get("source_id") or "")
                    for obj in final_objects
                    if (
                        (obj.get("sam3") or {}).get("selection_method")
                        == "bbox_first_spatial_rescue"
                    )
                    for source in (obj.get("proposal_sources") or [])
                    if isinstance(source, Mapping)
                    and str(source.get("pass") or "").upper() == "A"
                    and str(source.get("source_id") or "")
                }
            )
            semantic_gate_recovery = (
                resolve_assigned_evidence_semantic_gate_after_spatial_rescue(
                    assigned_manifest,
                    semantic_gate if isinstance(semantic_gate, Mapping) else None,
                    final_objects,
                    downstream_spatial_rescue_source_ids,
                    bbox_rescue_audit.get(
                        "excluded_unmaterialized_source_object_ids", []
                    ),
                )
            )
            quality["assigned_sam3_evidence"][
                "semantic_gate_recovery"
            ] = semantic_gate_recovery
            unresolved_evidence_ids = list(
                semantic_gate_recovery.get("unresolved_evidence_ids") or []
            )
            if unresolved_evidence_ids:
                quality["hard_errors"].append(
                    {
                        "type": "assigned_sam3_semantic_gate_unresolved",
                        "evidence_ids": unresolved_evidence_ids,
                    }
                )
            locally_rebound_source_ids = sorted(
                {
                    str(source.get("source_id") or "")
                    for obj in final_objects
                    if (
                        (obj.get("sam3") or {}).get("status") == "success"
                        and (obj.get("sam3") or {}).get("selection_method")
                        == "vlm_reviewed_local_candidate_replace"
                    )
                    for source in (obj.get("proposal_sources") or [])
                    if isinstance(source, Mapping)
                    and str(source.get("pass") or "").upper() == "A"
                    and str(source.get("source_id") or "")
                }
            )
            overflow_local_validation = (
                build_locally_validated_assigned_evidence_overflow(
                    assigned_manifest, final_objects
                )
            )
            quality["assigned_sam3_evidence"][
                "overflow_local_validation"
            ] = overflow_local_validation
            quality["hard_errors"].extend(
                overflow_local_validation.get("hard_errors") or []
            )
            extent_gate = {
                "status": "not_enforced",
                "reason": (
                    "assigned-mask claimed-bbox extent is recorded by SAM3 "
                    "but is not a delivery gate"
                ),
            }
            quality["assigned_sam3_evidence_extent_gate"] = extent_gate
            quality["assigned_sam3_evidence"]["extent_gate"] = extent_gate
        if "assigned_sam3_evidence_extent_gate" not in quality:
            quality["assigned_sam3_evidence_extent_gate"] = {
                "status": "disabled_or_missing"
            }

        if bounded_candidate_review is not None:
            candidate_semantic_gate = (
                build_bounded_candidate_semantic_recall_gate(
                    bounded_candidate_review["selection_audit"],
                    pass_b_actions,
                    final_objects,
                )
            )
        else:
            candidate_semantic_gate = {
                "status": "disabled",
                "hard_errors": [],
            }
        quality["bounded_candidate_semantic_recall_gate"] = (
            candidate_semantic_gate
        )
        quality["hard_errors"].extend(
            candidate_semantic_gate.get("hard_errors") or []
        )
        functional_assembly_gate = build_functional_assembly_gate(
            final_objects
        )
        quality["functional_assembly_gate"] = functional_assembly_gate
        quality["hard_errors"].extend(
            functional_assembly_gate.get("hard_errors") or []
        )
        if fixed_bathtub_faucet_union is not None:
            quality["fixed_bathtub_faucet_union"] = (
                fixed_bathtub_faucet_union["audit"]
            )
            quality["hard_errors"].extend(
                fixed_bathtub_faucet_union["audit"].get("hard_errors") or []
            )
        else:
            quality["fixed_bathtub_faucet_union"] = {
                "status": "disabled_or_not_applicable",
                "hard_errors": [],
            }
        if fixed_component_action_policy is not None:
            quality["fixed_component_action_policy"] = (
                fixed_component_action_policy["audit"]
            )


        formal_after = snapshot_tree(formal_paths)
        atomic_json(output_dir / "formal_hashes_after.json", formal_after)
        formal_comparison = compare_snapshots(formal_before, formal_after)
        if not formal_comparison["unchanged"]:
            quality["hard_errors"].append(
                {
                    "type": "formal_outputs_modified",
                    "details": formal_comparison,
                }
            )
        if session.model_load_count != 1:
            quality["hard_errors"].append(
                {
                    "type": "model_loaded_more_than_once",
                    "count": session.model_load_count,
                }
            )

        review_mode = "bounded_post_sam3a"
        quality["content_host_review"] = (
            {
                "status": content_host_review["manifest"]["status"],
                "mode": review_mode,
                "selected_host_count": content_host_review["manifest"][
                    "selected_count"
                ],
                "context_only": True,
                "selection_audit": str(
                    output_dir / "content_host_selection_audit.json"
                ),
                "additional_vlm_calls": 0,
                "api_call_audit": pass_b_review_api_audit,
            }
            if content_host_review is not None
            else {
                "status": "disabled",
                "mode": review_mode,
                "selected_host_count": 0,
                "context_only": True,
                "additional_vlm_calls": 0,
                "api_call_audit": pass_b_review_api_audit,
            }
        )
        if pass_b_review_api_audit is None:
            quality["hard_errors"].append(
                {"type": "missing_pass_b_review_api_audit"}
            )
        else:
            expected_mode = review_mode
            if pass_b_review_api_audit.get("mode") != expected_mode:
                quality["hard_errors"].append(
                    {
                        "type": "pass_b_review_mode_mismatch",
                        "expected": expected_mode,
                        "actual": pass_b_review_api_audit.get("mode"),
                    }
                )
            closure = pass_b_review_api_audit.get("content_closure") or {}
            expected_current_calls = 1 + (
                1 if closure.get("status") == "completed" else 0
            )
            if (
                pass_b_review_api_audit.get("pass_b_calls_made")
                != expected_current_calls
            ):
                quality["hard_errors"].append(
                    {"type": "pass_b_review_call_count_mismatch"}
                )
        quality = apply_delivery_quality_policy(
            quality,
            strict=bool(args.strict_quality),
        )
        quality["hard_error_count"] = len(quality["hard_errors"])
        quality["warning_count"] = len(quality["warnings"])
        publish_name = _select_result_directory_name(quality["status"])
        publish_dir = output_dir / publish_name
        final_objects = _rewrite_paths(final_objects, staging_dir, publish_dir)
        quality = _rewrite_paths(quality, staging_dir, publish_dir)
        atomic_json(bbox_rescue_audit_path, bbox_rescue_audit)
        atomic_json(multiplicity_audit_path, multiplicity_audit)
        bbox_rescue_summary = {
            "enabled": bool(bbox_rescue_audit.get("enabled")),
            "status": bbox_rescue_audit.get("status"),
            "target_count": int(bbox_rescue_audit.get("target_count") or 0),
            "accepted_count": int(bbox_rescue_audit.get("accepted_count") or 0),
            "inference_call_count": int(
                bbox_rescue_audit.get("inference_call_count") or 0
            ),
            "semantic_label_status": "provisional",
            "additional_vlm_calls": 0,
            "audit_path": str(bbox_rescue_audit_path),
        }
        multiplicity_summary = {
            "enabled": bool(multiplicity_audit.get("enabled")),
            "status": multiplicity_audit.get("status"),
            "groups_checked": len(multiplicity_audit.get("groups_checked") or []),
            "accepted_count": int(multiplicity_audit.get("accepted_count") or 0),
            "additional_sam3_inference_calls": 0,
            "additional_vlm_calls": 0,
            "audit_path": str(multiplicity_audit_path),
        }
        open_storage_content_prompt_summary = {
            "status": open_storage_content_prompt_audit.get("status"),
            "verified_open_storage_host_count": int(
                open_storage_content_prompt_audit.get(
                    "verified_open_storage_host_count"
                ) or 0
            ),
            "changed_count": int(
                open_storage_content_prompt_audit.get("changed_count") or 0
            ),
            "audit_path": str(
                output_dir
                / "open_storage_content_prompt_normalization_audit.json"
            ),
        }
        mobile_tier_summary = {
            "status": mobile_tier_audit.get("status"),
            "target_host_count": int(mobile_tier_audit.get("target_host_count") or 0),
            "processed_host_count": int(mobile_tier_audit.get("processed_host_count") or 0),
            "inference_call_count": int(mobile_tier_audit.get("inference_call_count") or 0),
            "created_tier_object_count": len(
                mobile_tier_audit.get("created_tier_objects") or []
            ),
            "additional_vlm_calls": 0,
            "audit_path": str(output_dir / "mobile_tiered_carrier_completion_audit.json"),
        }

        identity["records"] = [
            {
                **record,
                "binding_status": (
                    "sam3_mask_bound"
                    if any(
                        obj["object_id"] == record["proposal_id"]
                        and (obj.get("sam3") or {}).get("status") == "success"
                        for obj in final_objects
                    )
                    else "sam3_failed"
                ),
            }
            for record in identity["records"]
        ]
        top_review_candidate_count = len(
            top_review["manifest"].get("candidates") or []
        )
        bounded_candidate_summary = {
            "enabled": True,
            "status": (
                bounded_candidate_review["selection_audit"]["status"]
                if bounded_candidate_review is not None
                else "disabled"
            ),
            "selected_candidate_count": (
                bounded_candidate_review["selection_audit"]["selected_count"]
                if bounded_candidate_review is not None
                else 0
            ),
            "selected_counts_by_namespace": (
                bounded_candidate_review["selection_audit"][
                    "selected_counts_by_namespace"
                ]
                if bounded_candidate_review is not None
                else {}
            ),
            "candidate_panel": (
                local_candidate_panel_asset["path"]
                if bounded_candidate_review is not None
                and local_candidate_panel_asset is not None
                else None
            ),
            "selection_audit": (
                str(output_dir / "bounded_candidate_selection_audit.json")
                if bounded_candidate_review is not None
                else None
            ),
            "api_call_audit": bounded_candidate_api_audit,
            "additional_vlm_calls": int(
                (bounded_candidate_api_audit or {}).get("additional_vlm_calls")
                or 0
            ),
            "logical_pipeline_vlm_api_calls": int(
                (bounded_candidate_api_audit or {}).get(
                    "logical_pipeline_vlm_api_calls"
                ) or 2
            ),
            "api_calls_unchanged": bool(
                (bounded_candidate_api_audit or {}).get(
                    "api_calls_unchanged", True
                )
            ),
        }
        content_host_summary = {
            "enabled": True,
            "status": (
                content_host_review["manifest"]["status"]
                if content_host_review is not None
                else "disabled"
            ),
            "selected_host_count": (
                content_host_review["manifest"]["selected_count"]
                if content_host_review is not None
                else 0
            ),
            "content_host_panel": (
                content_host_review["panel_path"]
                if content_host_review is not None
                else None
            ),
            "mode": (pass_b_review_api_audit or {}).get("mode"),
            "combined_review_panel": (pass_b_review_api_audit or {}).get(
                "combined_review_panel"
            ),
            "pass_b_review_panel": (
                pass_b_review_panel_asset["path"]
                if pass_b_review_panel_asset is not None
                else None
            ),
            "candidate_sources": (pass_b_review_api_audit or {}).get(
                "candidate_sources", {}
            ),
            "pass_b_request_image_count": (pass_b_review_api_audit or {}).get(
                "total_request_image_count", 1
            ),
            "parallel_with_sam3_a": (pass_b_review_api_audit or {}).get(
                "parallel_with_sam3_a", False
            ),
            "selection_audit": (
                str(output_dir / "content_host_selection_audit.json")
                if content_host_review is not None
                else None
            ),
            "context_only": True,
            "content_closure": (pass_b_review_api_audit or {}).get(
                "content_closure"
            ),
            "additional_vlm_calls": int(
                (pass_b_review_api_audit or {}).get("additional_vlm_calls")
                or 0
            ),
            "logical_pipeline_vlm_api_calls": int(
                (pass_b_review_api_audit or {}).get(
                    "logical_pipeline_vlm_api_calls"
                ) or 2
            ),
            "api_calls_unchanged": bool(
                (pass_b_review_api_audit or {}).get(
                    "api_calls_unchanged", True
                )
            ),
        }
        post_bbox_plant_container_summary = {
            "status": post_bbox_plant_container_audit.get("status"),
            "accepted_count": post_bbox_plant_container_audit.get(
                "accepted_count", 0
            ),
            "additional_vlm_calls": 0,
            "audit_path": str(post_bbox_plant_container_audit_path),
        }
        final_registry = {
            "schema_version": "scenemosaic_perception_registry",
            "scene_name": config["scene_name"],
            "image_path": str(config["image_path"]),
            "objects": final_objects,
            "canonical_final_asset_publication": (
                canonical_final_asset_publication_audit
            ),
            "manual_bed_pillow_exclusions": bed_pillow_edits,
            "sam3_assignment_policy": ASSIGNMENT_POLICY,
            "affected_prompt_groups": affected_audit,
            "open_storage_content_prompt_normalization": (
                open_storage_content_prompt_summary
            ),
            "bbox_failed_mask_rescue": bbox_rescue_summary,
            "plant_container_mask_completion_after_bbox_rescue": (
                post_bbox_plant_container_summary
            ),
            "post_delta_multiplicity_closure": multiplicity_summary,
            "mobile_tiered_carrier_completion": mobile_tier_summary,

            "top_zone_recall": {
                "prompt": args.top_zone_prompt,
                "raw_candidate_count": top_raw_result["registry"].get(
                    "candidate_count", 0
                ),
                "review_candidate_count": top_review_candidate_count,
                "binding_status": local_binding_audit["status"],
                "binding_audit": str(
                    output_dir / "local_candidate_binding_audit.json"
                ),
                "decision_counts": local_binding_audit["decision_counts"],
                "raw_registry": top_raw_result["registry_path"],
                "review_manifest": top_review["manifest_path"],
            },
            "bounded_post_sam3a_candidate_review": bounded_candidate_summary,
            "content_host_review": content_host_summary,
            "model_load_count": session.model_load_count,
            "inference_run_count": session.inference_run_count,
        }
        atomic_json(staging_dir / "sam3_objects.json", final_registry)
        atomic_json(staging_dir / "identity_manifest.json", identity)
        atomic_json(staging_dir / "quality_report.json", quality)

        atomic_json(output_dir / "quality_report.json", quality)
        atomic_json(output_dir / "formal_hash_comparison.json", formal_comparison)

        manifest.update(
            {
                "status": (
                    "success" if quality["status"] == "pass" else "quality_failed"
                ),
                "finished_at_utc": utc_now(),
                "pass_a": {
                    "object_count": len(pass_a_objects),
                    "call": pass_a_info,
                    "bbox_review_assets": pass_a_assets,
                },
                "pass_b": {
                    "addition_count": len(pass_b_actions["additions"]),
                    "update_count": len(pass_b_actions["updates"]),
                    "removal_count": len(pass_b_actions["removals"]),
                    "local_candidate_decision_count": len(
                        pass_b_actions.get("local_candidate_decisions") or []
                    ),
                    "local_candidate_rejection_count": len(
                        pass_b_actions.get("local_candidate_rejections") or []
                    ),
                    "local_candidate_match_count": (
                        local_match_audit.get("match_count", 0)
                        if local_match_audit is not None
                        else 0
                    ),
                    "local_candidate_unresolved_count": (
                        len(local_match_audit.get("unresolved_candidate_ids") or [])
                        if local_match_audit is not None
                        else 0
                    ),
                    "call": pass_b_info,
                },
                "pass_b_action_guard": {
                    "status": pass_b_action_guard_audit.get("status"),
                    "partition_count": pass_b_action_guard_audit.get(
                        "partition_count", 0
                    ),
                    "suppressed_update_count": len(
                        pass_b_action_guard_audit.get("suppressed_updates") or []
                    ),
                    "suppressed_addition_count": len(
                        pass_b_action_guard_audit.get("suppressed_additions") or []
                    ),
                    "audit_path": str(
                        output_dir / "pass_b_action_guard_audit.json"
                    ),
                },
                "top_zone_recall": {
                    "prompt": args.top_zone_prompt,
                    "raw_candidate_count": top_raw_result["registry"].get(
                        "candidate_count", 0
                    ),
                    "review_candidate_count": top_review_candidate_count,
                    "candidate_panel": top_review["panel_path"],
                    "binding_status": local_binding_audit["status"],
                    "bound_object_count": local_binding_audit["bound_object_count"],
                    "decision_counts": local_binding_audit["decision_counts"],
                    "raw_registry": top_raw_result["registry_path"],
                    "review_manifest": top_review["manifest_path"],
                    "binding_audit": str(
                        output_dir / "local_candidate_binding_audit.json"
                    ),
                },
                "bounded_post_sam3a_candidate_review": bounded_candidate_summary,
                "content_host_review": content_host_summary,
                "open_storage_content_prompt_normalization": (
                    open_storage_content_prompt_summary
                ),
                "fruit_container_completion": (
                    {
                        "enabled": True,
                        "status": container_content_candidate_recall["audit"].get(
                            "status"
                        ),
                        "probe_count": container_content_candidate_recall[
                            "audit"
                        ].get("probe_count", 0),
                        "completed_count": container_content_candidate_recall[
                            "audit"
                        ].get("completed_count", 0),
                        "completed_source_object_ids": (
                            container_content_candidate_recall["audit"].get(
                                "completed_source_object_ids", []
                            )
                        ),
                        "plants_in_scope": False,
                        "additional_vlm_calls": 0,
                        "audit_path": str(
                            output_dir
                            / "container_content_candidate_recall_audit.json"
                        ),
                    }
                    if container_content_candidate_recall is not None
                    else {
                        "enabled": False,
                        "status": "disabled",
                        "probe_count": 0,
                        "completed_count": 0,
                        "completed_source_object_ids": [],
                        "plants_in_scope": False,
                        "additional_vlm_calls": 0,
                    }
                ),

                "bbox_failed_mask_rescue": bbox_rescue_summary,
                "post_delta_multiplicity_closure": multiplicity_summary,
                "mobile_tiered_carrier_completion": mobile_tier_summary,
                "fused_object_count": len(fused_objects),
                "final_object_count": len(final_objects),
                "fused_bbox_review_assets": fused_assets,
                "affected_prompt_count": len(affected_prompts),
                "sam3_model_load_count": session.model_load_count,
                "sam3_inference_run_count": session.inference_run_count,
                "published_result": str(publish_dir),
                "quality_status": quality["status"],
                "quality_policy": quality["quality_policy"],
                "formal_outputs_modified": not formal_comparison["unchanged"],
                "pass_a_inherited": cached_pre_pass_b_resume,
                "pass_b_inherited": False,
                "eligible_for_stage04": quality["status"] == "pass",
                "sam3_model_load_scope": (
                    "one_in_cache_continuation_process; cached top-zone evidence "
                    "came from one earlier source-process load"
                    if cached_pre_pass_b_resume
                    else "one_in_fresh_process"
                ),
            }
        )
        atomic_json(output_dir / "run_manifest.json", manifest)
        atomic_publish(staging_dir, publish_dir)

        try:
            sam3d_delivery = (
                SAM3D_DELIVERY_POLICY.materialize_delivery_package(
                    publish_dir=publish_dir,
                )
            )
        except Exception as delivery_exc:
            sam3d_delivery = {
                "status": "infrastructure_failed",
                "sam3d_execution_allowed": False,
                "object_count": 0,
                "error": {
                    "type": type(delivery_exc).__name__,
                    "message": str(delivery_exc),
                },
            }
            atomic_json(
                output_dir / "sam3d_delivery_failure.json",
                sam3d_delivery,
            )
        manifest["sam3d_delivery"] = sam3d_delivery
        source_quality_passed = quality["status"] == "pass"
        manifest["eligible_for_stage04"] = bool(
            source_quality_passed
            and sam3d_delivery.get("sam3d_execution_allowed")
        )
        if not source_quality_passed:
            manifest["status"] = "quality_failed"
        elif sam3d_delivery.get("status") == "sam3d_ready_degraded":
            manifest["status"] = "success_degraded"
        elif sam3d_delivery.get("status") == "sam3d_ready":
            manifest["status"] = "success"
        else:
            manifest["status"] = "infrastructure_failed"
        atomic_json(output_dir / "run_manifest.json", manifest)

        print("status:", manifest["status"])
        print("scene_name:", config["scene_name"])
        print("pass_a_objects:", len(pass_a_objects))
        print("pass_b_additions:", len(pass_b_actions["additions"]))
        print("top_zone_raw_candidates:", top_raw_result["registry"].get("candidate_count", 0))
        print("top_zone_review_candidates:", len(local_candidate_metadata))
        print("top_zone_bound_objects:", local_binding_audit["bound_object_count"])
        print("fused_objects:", len(fused_objects))
        print("affected_prompts:", len(affected_prompts))
        print("sam3_model_load_count:", session.model_load_count)
        print("sam3_inference_run_count:", session.inference_run_count)
        print(
            "bbox_failed_mask_rescue:",
            bbox_rescue_summary["status"],
            f"accepted={bbox_rescue_summary['accepted_count']}",
            f"inferences={bbox_rescue_summary['inference_call_count']}",
        )
        print(
            "post_delta_multiplicity_closure:",
            multiplicity_summary["status"],
            f"accepted={multiplicity_summary['accepted_count']}",
            "additional_inferences=0",
        )
        print("quality_status:", quality["status"])
        print("formal_outputs_modified:", not formal_comparison["unchanged"])
        print("result_dir:", publish_dir)
        print("sam3d_delivery_status:", sam3d_delivery.get("status"))
        print(
            "sam3d_execution_allowed:",
            sam3d_delivery.get("sam3d_execution_allowed"),
        )
        print(
            "contact_sheet:",
            publish_dir / "sam3_overlay_contact_sheet.jpg",
        )
        return 0 if sam3d_delivery.get("sam3d_execution_allowed") else 3
    except Exception as exc:
        formal_after = snapshot_tree(formal_paths)
        formal_comparison = compare_snapshots(formal_before, formal_after)
        atomic_json(output_dir / "formal_hashes_after.json", formal_after)
        atomic_json(output_dir / "formal_hash_comparison.json", formal_comparison)
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "formal_outputs_modified": not formal_comparison["unchanged"],
            }
        )
        atomic_json(output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
