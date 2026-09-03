#!/usr/bin/env python3
"""Reusable, in-process SAM3 runtime for production perception attempts.

The runtime delegates segmentation and output generation to :mod:`.engine`.
It adds lifecycle management around that implementation: one model load,
isolated attempt configuration, complete-prompt-group filtering, and a stable
Python API.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import conservative_assignment as ASSIGNMENT_FALLBACK
from . import engine as SEGMENTATION_ENGINE


_DEFAULT_RUNTIME_ROOT = Path(
    os.environ.get(
        "SCENEMOSAIC_RUNTIME_ROOT",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "scenemosaic",
    )
).expanduser().resolve()
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "SCENEMOSAIC_PERCEPTION_MODEL_DIR",
        _DEFAULT_RUNTIME_ROOT / "models" / "sam3",
    )
).expanduser().resolve()
ASSIGNMENT_POLICIES = {"adaptive_local"}

_production_module = None
_production_module_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_production_module():
    """Return the production segmentation engine exactly once."""
    global _production_module
    if _production_module is not None:
        return _production_module

    with _production_module_lock:
        if _production_module is not None:
            return _production_module
        _production_module = SEGMENTATION_ENGINE
        return _production_module


def _registry_from_input(
    objects: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    registry_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(objects, Mapping):
        registry = copy.deepcopy(dict(objects))
        if "objects" not in registry:
            raise ValueError("Registry input must contain an 'objects' list.")
    else:
        registry = {"objects": copy.deepcopy(list(objects))}

    if registry_metadata:
        for key, value in registry_metadata.items():
            if key == "objects":
                continue
            registry[key] = copy.deepcopy(value)
    if not isinstance(registry.get("objects"), list):
        raise TypeError("Registry 'objects' must be a list.")
    return registry


def prompt_groups(
    objects: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact prompt grouping used by the production SAM3 stage.

    The returned dictionaries are copies.  Callers may use the keys to choose
    affected groups without mutating their source registry.
    """
    production = _load_production_module()
    registry = _registry_from_input(objects, registry_metadata=None)
    canonical = production.canonicalize_objects(registry["objects"])
    groups = production.build_prompt_groups(canonical)
    return {
        str(prompt): copy.deepcopy(list(group_objects))
        for prompt, group_objects in groups.items()
    }


def _select_complete_prompt_groups(
    production,
    registry: Mapping[str, Any],
    prompt_filter: Iterable[str] | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    source_registry = copy.deepcopy(dict(registry))
    canonical = production.canonicalize_objects(
        source_registry.get("objects", [])
    )
    groups = production.build_prompt_groups(canonical)
    available_prompts = list(groups)

    if prompt_filter is None:
        selected_prompts = available_prompts
    else:
        requested = []
        seen = set()
        for raw_prompt in prompt_filter:
            prompt = str(raw_prompt)
            if prompt not in seen:
                requested.append(prompt)
                seen.add(prompt)
        unknown = [prompt for prompt in requested if prompt not in groups]
        if unknown:
            raise ValueError(
                "Unknown SAM3 prompt group(s): "
                + ", ".join(repr(prompt) for prompt in unknown)
                + ". Available groups: "
                + ", ".join(repr(prompt) for prompt in available_prompts)
            )
        requested_set = set(requested)
        # Preserve production group order, not caller set order.
        selected_prompts = [
            prompt for prompt in available_prompts
            if prompt in requested_set
        ]

    selected_ids = {
        obj["object_id"]
        for prompt in selected_prompts
        for obj in groups[prompt]
    }
    source_registry["objects"] = [
        obj for obj in canonical if obj["object_id"] in selected_ids
    ]
    source_registry["sam3_session_selection"] = {
        "source_object_count": len(canonical),
        "selected_object_count": len(source_registry["objects"]),
        "available_prompts": available_prompts,
        "selected_prompts": selected_prompts,
        "complete_prompt_groups_only": True,
    }
    return source_registry, available_prompts, selected_prompts


@dataclass
class Sam3Session:
    """One loaded SAM3 processor/model pair.

    ``run_lock`` protects the temporary segmentation-engine bindings used by
    ``segment_registry``.  A session is intentionally serial; orchestration may
    overlap a remote VLM request with this local GPU work, but must not issue
    two GPU segmentation calls through the same session simultaneously.
    """

    processor: Any
    model: Any
    device: str
    model_dir: Path
    production: Any = field(repr=False)
    loaded_at: str = field(default_factory=_utc_now)
    model_load_count: int = 1
    inference_run_count: int = 0
    run_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )


def load_session(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    device: str | None = None,
) -> Sam3Session:
    """Load SAM3 once and return a reusable in-process session."""
    production = _load_production_module()
    production.load_runtime_modules()

    resolved_model_dir = Path(model_dir).expanduser().resolve()
    if not resolved_model_dir.exists():
        raise FileNotFoundError(f"SAM3 model directory not found: {resolved_model_dir}")

    auto_device = "cuda" if production.torch.cuda.is_available() else "cpu"
    resolved_device = str(device or auto_device)
    if resolved_device.startswith("cuda") and not production.torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but torch.cuda.is_available() is false: {resolved_device}"
        )

    processor = production.Sam3Processor.from_pretrained(
        str(resolved_model_dir)
    )
    model = production.Sam3Model.from_pretrained(
        str(resolved_model_dir)
    ).to(resolved_device)
    model.eval()
    return Sam3Session(
        processor=processor,
        model=model,
        device=resolved_device,
        model_dir=resolved_model_dir,
        production=production,
    )


class _CachedProcessorProvider:
    def __init__(self, session: Sam3Session):
        self._session = session

    def from_pretrained(self, _model_dir):
        return self._session.processor


class _CachedModelHandle:
    def __init__(self, session: Sam3Session):
        self._session = session

    def to(self, requested_device):
        requested = str(requested_device)
        actual = str(self._session.device)
        if requested.split(":", 1)[0] != actual.split(":", 1)[0]:
            raise RuntimeError(
                "Production SAM3 device selection differs from loaded session: "
                f"production={requested!r}, session={actual!r}"
            )
        return self._session.model


class _CachedModelProvider:
    def __init__(self, session: Sam3Session):
        self._session = session

    def from_pretrained(self, _model_dir):
        return _CachedModelHandle(self._session)


def _build_runtime_config(
    image_path: Path,
    output_dir: Path,
    input_registry_path: Path,
    scene_name: str,
) -> dict[str, Any]:
    return {
        "scene_name": scene_name,
        "run_root": output_dir.parent,
        "input_image": image_path,
        "vlm_objects_json": input_registry_path,
        "vlm_run_info_json": input_registry_path.parent / "vlm_first_run_info.json",
        "output_dir": output_dir,
        "masks_dir": output_dir / "masks",
        "overlays_dir": output_dir / "overlays",
        "unassigned_candidates_dir": output_dir / "unassigned_candidates",
        "debug_dir": output_dir / "debug_prompt_candidates",
        "sam3_objects_json": output_dir / "sam3_objects.json",
        "sam3_overlay": output_dir / "sam3_overlay.png",
        "run_info_json": output_dir / "sam3_run_info.json",
        "scene_config_path": None,
    }


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"SAM3 session output directory is not empty: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _apply_post_assignment_fallbacks(
    production,
    output_registry: dict[str, Any],
    carrier_objects: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    assignment_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize only conservative adaptive fallback matches."""
    if assignment_policy != "adaptive_local":
        return output_registry, {
            "schema_version": "scenemosaic_conservative_assignment_plan",
            "status": "disabled_for_assignment_policy",
            "matches": [],
            "abstentions": [],
        }
    if not output_registry.get("unassigned_sam3_candidates"):
        return output_registry, {
            "schema_version": "scenemosaic_conservative_assignment_plan",
            "status": "no_unassigned_candidates",
            "matches": [],
            "abstentions": [],
        }

    audit = ASSIGNMENT_FALLBACK.plan_registry_fallbacks(
        output_registry,
        carrier_objects=carrier_objects,
    )
    audit["status"] = "applied" if audit["matches"] else "abstained"
    if not audit["matches"]:
        output_registry["post_assignment_fallback_audit"] = audit
        temporary = config["sam3_objects_json"].with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(output_registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, config["sam3_objects_json"])
        return output_registry, audit

    production.load_runtime_modules()
    with production.Image.open(config["input_image"]) as source:
        image_np = production.np.asarray(source.convert("RGB")).astype(
            production.np.uint8
        )
    objects_by_id = {
        str(obj["object_id"]): obj
        for obj in output_registry.get("objects") or []
    }
    candidates_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in output_registry.get("unassigned_sam3_candidates") or []
    }
    applied_candidate_ids = set()
    for match in audit["matches"]:
        object_id = str(match["object_id"])
        candidate_id = str(match["candidate_id"])
        obj = objects_by_id.get(object_id)
        candidate = candidates_by_id.get(candidate_id)
        if obj is None or candidate is None:
            raise RuntimeError(
                f"fallback plan references missing pair: {object_id}, {candidate_id}"
            )
        prior_sam3 = copy.deepcopy(dict(obj.get("sam3") or {}))
        if prior_sam3.get("status") == "success":
            raise RuntimeError(f"fallback attempted to replace success: {object_id}")
        candidate_path = Path(str(candidate.get("mask_path") or ""))
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"fallback candidate mask missing: {candidate_path}"
            )
        mask = production.np.asarray(
            production.Image.open(candidate_path).convert("L")
        ) > 0
        if mask.shape != image_np.shape[:2] or not bool(mask.any()):
            raise RuntimeError(
                f"fallback candidate mask invalid for {candidate_id}: {mask.shape}"
            )
        base = (
            f"{production.safe_name(object_id)}_"
            f"{production.safe_name(obj.get('category', 'unknown'))}"
        )
        mask_path, overlay_path = production.save_mask_overlay(
            mask,
            image_np,
            base,
            config["masks_dir"],
            config["overlays_dir"],
        )
        vlm_bbox = prior_sam3.get("vlm_bbox_abs_xyxy")
        if not isinstance(vlm_bbox, list) or len(vlm_bbox) != 4:
            vlm_bbox = production.xyxy_norm_to_abs(
                obj.get("bbox_2d_norm_xyxy"),
                image_np.shape[1],
                image_np.shape[0],
            )
        mask_bbox = production.mask_to_bbox_xyxy(mask)
        iou = production.bbox_iou(vlm_bbox, mask_bbox)
        locality = copy.deepcopy(
            prior_sam3.get("scale_aware_locality") or {}
        )
        locality["accepted_candidate_indices"] = [
            int(match["candidate_index"])
        ]
        locality["forced_fallback_used"] = True
        locality["post_assignment_fallback"] = copy.deepcopy(match)
        obj["sam3"] = {
            "status": "success",
            "warning": (
                None
                if iou >= production.LOW_IOU_WARNING_THRESHOLD
                else "low_iou_with_vlm_bbox"
            ),
            "prompt_used": str(match["prompt"]),
            "score": float(candidate.get("score") or 0.0),
            "area_pixels": int(mask.sum()),
            "vlm_bbox_abs_xyxy": vlm_bbox,
            "mask_bbox_abs_xyxy": mask_bbox,
            "vlm_bbox_iou": float(iou),
            "merged_candidate_indices": [int(match["candidate_index"])],
            "scale_aware_locality": locality,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "selection_method": str(match["method"]),
            "post_assignment_fallback": copy.deepcopy(match),
            "merge_review": {
                "required": False,
                "resolution_stage": "post_assignment_conservative_fallback",
            },
        }
        applied_candidate_ids.add(candidate_id)

    output_registry["unassigned_sam3_candidates"] = [
        candidate
        for candidate in output_registry.get("unassigned_sam3_candidates") or []
        if str(candidate.get("candidate_id")) not in applied_candidate_ids
    ]
    output_registry["post_assignment_fallback_audit"] = audit
    temporary = config["sam3_objects_json"].with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output_registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, config["sam3_objects_json"])
    production.Image.fromarray(
        production.rebuild_scene_overlay(
            output_registry.get("objects") or [], image_np
        )
    ).save(config["sam3_overlay"])
    return output_registry, audit


def segment_registry(
    session: Sam3Session,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    output_dir: str | Path,
    assignment_policy: str = "adaptive_local",
    prompt_filter: Iterable[str] | None = None,
    *,
    scene_name: str | None = None,
    registry_metadata: Mapping[str, Any] | None = None,
    capture_debug_candidates: bool = True,
    capture_unassigned_candidates: bool = True,
    capture_ambiguous_candidate_evidence: bool = True,
    capture_post_delta_multiplicity_candidates: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Segment one registry using an already-loaded SAM3 session.

    ``prompt_filter`` contains exact keys returned by :func:`prompt_groups`.
    Filtering happens only after production grouping, so all objects sharing a
    selected prompt are always processed together.
    """
    if not isinstance(session, Sam3Session):
        raise TypeError("session must be a Sam3Session returned by load_session().")
    if assignment_policy not in ASSIGNMENT_POLICIES:
        raise ValueError(
            f"Unsupported assignment_policy={assignment_policy!r}; "
            f"expected one of {sorted(ASSIGNMENT_POLICIES)}"
        )

    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.is_file():
        raise FileNotFoundError(f"SAM3 input image not found: {resolved_image}")
    resolved_output = Path(output_dir).expanduser().resolve()
    inferred_scene_name = scene_name or resolved_image.stem

    registry = _registry_from_input(objects, registry_metadata)
    filtered_registry, available_prompts, selected_prompts = (
        _select_complete_prompt_groups(
            session.production, registry, prompt_filter
        )
    )
    if not filtered_registry["objects"]:
        raise ValueError("No objects remain after applying prompt_filter.")

    with session.run_lock:
        _prepare_output_dir(resolved_output, overwrite=overwrite)
        input_dir = resolved_output / "_session_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_registry_path = input_dir / "vlm_objects.json"
        input_registry_path.write_text(
            json.dumps(filtered_registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (input_dir / "vlm_first_run_info.json").write_text(
            json.dumps(
                {
                    "source": "sam3_session",
                    "scene_name": inferred_scene_name,
                    "created_at": _utc_now(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        config = _build_runtime_config(
            resolved_image,
            resolved_output,
            input_registry_path,
            inferred_scene_name,
        )
        config["capture_debug_candidates"] = bool(capture_debug_candidates)
        config["capture_unassigned_candidates"] = bool(
            capture_unassigned_candidates
        )
        # Keep only masks that are required to resolve an adaptive-local
        # ambiguity when bulk unassigned-candidate capture is disabled.
        config["capture_ambiguous_candidate_evidence"] = bool(
            capture_ambiguous_candidate_evidence
        )
        config["capture_post_delta_multiplicity_candidates"] = bool(
            capture_post_delta_multiplicity_candidates
        )

        production = session.production
        original_parse_args = production.parse_args
        original_load_runtime_config = production.load_runtime_config
        original_processor_class = production.Sam3Processor
        original_model_class = production.Sam3Model
        run_index = session.inference_run_count + 1

        try:
            production.parse_args = lambda: argparse.Namespace(
                scene_config=None,
                dry_run_config=False,
                assignment_policy=assignment_policy,
            )
            production.load_runtime_config = lambda _args: config
            production.Sam3Processor = _CachedProcessorProvider(session)
            production.Sam3Model = _CachedModelProvider(session)
            production.main()
        finally:
            production.parse_args = original_parse_args
            production.load_runtime_config = original_load_runtime_config
            production.Sam3Processor = original_processor_class
            production.Sam3Model = original_model_class

        session.inference_run_count = run_index

        output_registry = json.loads(
            config["sam3_objects_json"].read_text(encoding="utf-8")
        )
        output_registry, fallback_audit = _apply_post_assignment_fallbacks(
            production,
            output_registry,
            registry.get("objects") or [],
            config,
            assignment_policy,
        )
        run_info = json.loads(
            config["run_info_json"].read_text(encoding="utf-8")
        )
        session_info = {
            "api": "sam3_session",
            "model_dir": str(session.model_dir),
            "device": session.device,
            "loaded_at": session.loaded_at,
            "model_load_count": session.model_load_count,
            "inference_run_index": run_index,
            "available_prompts": available_prompts,
            "selected_prompts": selected_prompts,
            "complete_prompt_groups_only": True,
        }
        run_info["sam3_session"] = session_info
        run_info["post_assignment_fallback_audit"] = fallback_audit
        config["run_info_json"].write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "registry": output_registry,
        "run_info": run_info,
        "output_dir": str(resolved_output),
        "sam3_objects_json": str(config["sam3_objects_json"]),
        "sam3_overlay": str(config["sam3_overlay"]),
        "run_info_json": str(config["run_info_json"]),
        "available_prompts": available_prompts,
        "selected_prompts": selected_prompts,
        "model_load_count": session.model_load_count,
        "inference_run_index": run_index,
    }



def infer_text_candidates(
    session: Sam3Session,
    image_path: str | Path,
    prompt: str,
    output_dir: str | Path,
    *,
    score_threshold: float | None = None,
    mask_threshold: float | None = None,
    min_area: int | None = None,
    positive_bbox_abs_xyxy: Sequence[float] | None = None,
    maximum_candidates: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return all instance masks for one text concept without bbox assignment."""

    if not isinstance(session, Sam3Session):
        raise TypeError("session must be a Sam3Session returned by load_session().")
    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.is_file():
        raise FileNotFoundError(f"SAM3 input image not found: {resolved_image}")
    normalized_prompt = " ".join(str(prompt).split())
    if not normalized_prompt:
        raise ValueError("prompt must be non-empty")
    positive_box = None
    if positive_bbox_abs_xyxy is not None:
        if (
            len(positive_bbox_abs_xyxy) != 4
            or not all(isinstance(value, (int, float)) for value in positive_bbox_abs_xyxy)
        ):
            raise ValueError("positive_bbox_abs_xyxy must contain four numbers")
        positive_box = [float(value) for value in positive_bbox_abs_xyxy]
        if not (
            0 <= positive_box[0] < positive_box[2]
            and 0 <= positive_box[1] < positive_box[3]
        ):
            raise ValueError("positive_bbox_abs_xyxy must be a non-empty XYXY box")
    resolved_output = Path(output_dir).expanduser().resolve()
    production = session.production
    production.load_runtime_modules()
    resolved_score = float(
        production.SCORE_THRESHOLD if score_threshold is None else score_threshold
    )
    resolved_mask = float(
        production.MASK_THRESHOLD if mask_threshold is None else mask_threshold
    )
    resolved_min_area = int(production.MIN_AREA if min_area is None else min_area)
    if not 0 <= resolved_score <= 1 or not 0 <= resolved_mask <= 1:
        raise ValueError("score and mask thresholds must be in [0, 1]")
    if resolved_min_area <= 0:
        raise ValueError("min_area must be positive")
    if maximum_candidates is not None and int(maximum_candidates) <= 0:
        raise ValueError("maximum_candidates must be positive")

    with session.run_lock:
        _prepare_output_dir(resolved_output, overwrite=overwrite)
        masks_dir = resolved_output / "masks"
        overlays_dir = resolved_output / "overlays"
        masks_dir.mkdir()
        overlays_dir.mkdir()

        image = production.Image.open(resolved_image).convert("RGB")
        image_np = production.np.array(image).astype(production.np.uint8)
        height, width = image_np.shape[:2]
        processor_kwargs = {
            "images": image,
            "text": normalized_prompt,
            "return_tensors": "pt",
        }
        if positive_box is not None:
            processor_kwargs["input_boxes"] = [[positive_box]]
            processor_kwargs["input_boxes_labels"] = [[1]]
        inputs = session.processor(**processor_kwargs)
        inputs = {
            key: value.to(session.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with production.torch.no_grad():
            outputs = session.model(**inputs)
        result = session.processor.post_process_instance_segmentation(
            outputs,
            threshold=resolved_score,
            mask_threshold=resolved_mask,
            target_sizes=[(height, width)],
        )[0]
        masks = result.get("masks", [])
        scores = result.get("scores", [])
        candidates = []
        for raw_index, raw_mask in enumerate(masks):
            mask_np = production.tensor_or_array_to_numpy(raw_mask)
            mask_bool = mask_np > 0
            area = int(mask_bool.sum())
            if area < resolved_min_area:
                continue
            raw_score = scores[raw_index]
            score = (
                float(raw_score.detach().cpu())
                if production.torch.is_tensor(raw_score)
                else float(raw_score)
            )
            candidates.append(
                {
                    "raw_candidate_index": raw_index,
                    "mask_bool": mask_bool,
                    "score": score,
                    "area": area,
                    "mask_bbox_abs_xyxy": production.mask_to_bbox_xyxy(mask_bool),
                }
            )
        raw_candidate_count = len(candidates)
        if positive_box is not None:
            def spatial_rank(candidate):
                box = candidate["mask_bbox_abs_xyxy"]
                ix1 = max(float(positive_box[0]), float(box[0]))
                iy1 = max(float(positive_box[1]), float(box[1]))
                ix2 = min(float(positive_box[2]), float(box[2]))
                iy2 = min(float(positive_box[3]), float(box[3]))
                intersection = max(0.0, ix2 - ix1 + 1) * max(0.0, iy2 - iy1 + 1)
                target_area = max(1.0, (positive_box[2] - positive_box[0] + 1) * (positive_box[3] - positive_box[1] + 1))
                candidate_area = max(1.0, (box[2] - box[0] + 1) * (box[3] - box[1] + 1))
                union = target_area + candidate_area - intersection
                iou = intersection / union if union else 0.0
                target_coverage = intersection / target_area
                target_center = ((positive_box[0] + positive_box[2]) / 2, (positive_box[1] + positive_box[3]) / 2)
                candidate_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                scale = max(((positive_box[2] - positive_box[0]) ** 2 + (positive_box[3] - positive_box[1]) ** 2) ** 0.5, 1.0)
                center_distance = ((target_center[0] - candidate_center[0]) ** 2 + (target_center[1] - candidate_center[1]) ** 2) ** 0.5 / scale
                return (
                    2.0 * iou
                    + 2.0 * target_coverage
                    + max(0.0, 1.0 - center_distance)
                    + 0.25 * float(candidate["score"])
                )
            candidates = sorted(candidates, key=spatial_rank, reverse=True)
        else:
            candidates = sorted(candidates, key=production.candidate_sort_key)
        candidate_count_before_limit = len(candidates)
        if maximum_candidates is not None:
            candidates = candidates[: int(maximum_candidates)]

        records = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"local_candidate_{candidate_index:03d}"
            base = f"{candidate_id}_{production.safe_name(normalized_prompt)}"
            mask_path, overlay_path = production.save_mask_overlay(
                candidate["mask_bool"],
                image_np,
                base,
                masks_dir,
                overlays_dir,
                color=(255, 0, 0),
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "prompt": normalized_prompt,
                    "score": candidate["score"],
                    "area_pixels": candidate["area"],
                    "area_fraction": candidate["area"] / (height * width),
                    "mask_bbox_abs_xyxy": candidate["mask_bbox_abs_xyxy"],
                    "mask_bbox_norm_xyxy": production.abs_to_norm_bbox(
                        candidate["mask_bbox_abs_xyxy"], width, height
                    ),
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "raw_candidate_index": candidate["raw_candidate_index"],
                }
            )

        run_index = session.inference_run_count + 1
        session.inference_run_count = run_index
        registry = {
            "schema_version": "scenemosaic_sam3_text_candidates",
            "image_path": str(resolved_image),
            "image_size": [width, height],
            "prompt": normalized_prompt,
            "positive_bbox_abs_xyxy": positive_box,
            "thresholds": {
                "score": resolved_score,
                "mask": resolved_mask,
                "min_area": resolved_min_area,
            },
            "candidate_count": len(records),
            "raw_candidate_count": raw_candidate_count,
            "candidate_count_before_limit": candidate_count_before_limit,
            "maximum_candidates": maximum_candidates,
            "candidates": records,
            "sam3_session": {
                "model_dir": str(session.model_dir),
                "device": session.device,
                "model_load_count": session.model_load_count,
                "inference_run_index": run_index,
            },
        }
        registry_path = resolved_output / "text_candidates.json"
        temporary = registry_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, registry_path)
    return {
        "registry": registry,
        "output_dir": str(resolved_output),
        "registry_path": str(registry_path),
        "model_load_count": session.model_load_count,
        "inference_run_index": run_index,
    }


__all__ = [
    "ASSIGNMENT_POLICIES",
    "DEFAULT_MODEL_DIR",
    "Sam3Session",
    "load_session",
    "infer_text_candidates",
    "prompt_groups",
    "segment_registry",
]
