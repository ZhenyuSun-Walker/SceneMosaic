#!/usr/bin/env python3
"""Fuse explicitly claimed component masks into one host assembly.

This is deliberately narrower than generic overlap removal.  It only joins a
Pass-B addition to a successful Pass-A attachment host when the host's own
Pass-A wording explicitly claims that kind of attached content (for example,
``wall coat rack robes``).  A second failed duplicate addition may be removed
only when the accepted sibling mask already covers its declared box.  The
stage is isolated-attempt only, has no VLM calls, and materializes a fresh union mask in
the final staging directory.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class HostPartMaskFusionConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    minimum_child_box_coverage_by_host: float = 0.75
    minimum_failed_sibling_box_coverage: float = 0.75
    host_bbox_expansion_fraction: float = 0.06
    maximum_positive_bbox_duplicate_confirmations: int = 1
    minimum_confirmation_score: float = 0.85
    minimum_confirmation_target_coverage: float = 0.25
    minimum_confirmation_covered_by_accepted_mask: float = 0.85
    maximum_complete_host_probes: int = 2
    maximum_candidates_per_complete_host_probe: int = 8
    minimum_complete_host_score: float = 0.45
    minimum_complete_host_original_coverage: float = 0.85
    minimum_complete_host_added_fraction: float = 0.02
    maximum_complete_host_area_ratio: float = 3.0
    minimum_idempotent_anchor_area_ratio: float = 0.90
    maximum_idempotent_anchor_area_ratio: float = 1.10
    minimum_supplemental_attachment_score: float = 0.20
    maximum_supplemental_attachment_area_ratio: float = 0.75
    minimum_supplemental_added_fraction: float = 0.02
    supplemental_bbox_expansion_fraction: float = 0.20
    enable_semantic_component_absorption: bool = True
    minimum_semantic_component_mask_containment: float = 0.90
    maximum_semantic_component_area_ratio: float = 0.50


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tokens(value: Any) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", str(value or "").lower())
    normalized = set(raw)
    for token in raw:
        if len(token) > 3 and token.endswith("ies"):
            normalized.add(token[:-3] + "y")
        elif len(token) > 3 and token.endswith("s"):
            normalized.add(token[:-1])
    return normalized


def _object_tokens(obj: Mapping[str, Any]) -> set[str]:
    return _tokens(
        " ".join(
            str(obj.get(key) or "")
            for key in ("category", "sam3_prompt", "instance_name", "description")
        )
    )


SEMANTIC_COMPONENT_TOKENS = frozenset(
    {
        "arm",
        "armrest",
        "back",
        "backrest",
        "base",
        "cushion",
        "door",
        "drawer",
        "footboard",
        "frame",
        "handle",
        "headboard",
        "leg",
        "panel",
        "seat",
        "shelf",
    }
)

SEMANTIC_ANCHOR_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "black",
        "blue",
        "brown",
        "dark",
        "decorative",
        "gray",
        "green",
        "large",
        "light",
        "metal",
        "orange",
        "red",
        "small",
        "the",
        "white",
        "wood",
        "wooden",
    }
)


def _semantic_component_anchor_tokens(
    host: Mapping[str, Any], child: Mapping[str, Any]
) -> list[str]:
    """Return a meaningful host name shared by an intrinsic component."""
    child_tokens = _object_tokens(child)
    if not (child_tokens & SEMANTIC_COMPONENT_TOKENS):
        return []
    # Anchor on the host category, not shared colour/material adjectives in
    # free-form prompts.  A child may mention its host only in its prompt
    # (for example, category="drawer", prompt="sideboard drawer").
    host_name_tokens = (
        _tokens(host.get("category"))
        - SEMANTIC_COMPONENT_TOKENS
        - SEMANTIC_ANCHOR_STOP_TOKENS
    )
    shared = host_name_tokens & child_tokens
    return sorted(shared)


def _has_pass_source(obj: Mapping[str, Any], pass_name: str) -> bool:
    return any(
        str(source.get("pass") or "").upper() == pass_name
        for source in (obj.get("proposal_sources") or [])
        if isinstance(source, Mapping)
    )


def _pass_a_source_ids(obj: Mapping[str, Any]) -> list[str]:
    return [
        str(source.get("source_id") or "")
        for source in (obj.get("proposal_sources") or [])
        if isinstance(source, Mapping)
        and str(source.get("pass") or "").upper() == "A"
        and str(source.get("source_id") or "")
    ]


def _bbox(obj: Mapping[str, Any]) -> list[float] | None:
    try:
        value = [float(item) for item in obj.get("bbox_2d_norm_xyxy")]
    except (TypeError, ValueError):
        return None
    if len(value) != 4 or not (0.0 <= value[0] < value[2] <= 1.0 and 0.0 <= value[1] < value[3] <= 1.0):
        return None
    return value


def _coverage_by_expanded_host(child: Sequence[float], host: Sequence[float], expansion: float) -> float:
    hx1, hy1, hx2, hy2 = [float(value) for value in host]
    pad_x = (hx2 - hx1) * expansion
    pad_y = (hy2 - hy1) * expansion
    hx1, hy1 = max(0.0, hx1 - pad_x), max(0.0, hy1 - pad_y)
    hx2, hy2 = min(1.0, hx2 + pad_x), min(1.0, hy2 + pad_y)
    cx1, cy1, cx2, cy2 = [float(value) for value in child]
    intersection = max(0.0, min(hx2, cx2) - max(hx1, cx1)) * max(0.0, min(hy2, cy2) - max(hy1, cy1))
    return intersection / max(1e-9, (cx2 - cx1) * (cy2 - cy1))


def _mask_bbox_coverage(mask: np.ndarray, bbox_norm: Sequence[float]) -> float:
    height, width = mask.shape
    x1 = max(0, min(width - 1, int(round(float(bbox_norm[0]) * width))))
    y1 = max(0, min(height - 1, int(round(float(bbox_norm[1]) * height))))
    x2 = max(x1 + 1, min(width, int(round(float(bbox_norm[2]) * width))))
    y2 = max(y1 + 1, min(height, int(round(float(bbox_norm[3]) * height))))
    return int(mask[y1:y2, x1:x2].sum()) / max(1, (x2 - x1) * (y2 - y1))


def _confirm_failed_duplicate_with_positive_bbox(
    *,
    child: Mapping[str, Any],
    accepted_mask: np.ndarray,
    image_path: str | Path | None,
    session: Any,
    infer_candidates_fn: Callable[..., Mapping[str, Any]] | None,
    work_dir: Path,
    cfg: HostPartMaskFusionConfig,
) -> dict[str, Any]:
    """Confirm that a failed sibling is already covered by one SAM3 component.

    This is not a generic failed-mask bypass.  It is available only to the
    explicit host-assembly caller and uses the failed VLM box as a positive
    SAM3 input.
    """

    box = _bbox(child)
    if box is None or image_path is None or session is None or infer_candidates_fn is None:
        return {"status": "not_run", "reason": "missing_local_probe_inputs"}
    height, width = accepted_mask.shape
    target = [
        max(0, min(width - 1, int(round(box[0] * width)))),
        max(0, min(height - 1, int(round(box[1] * height)))),
        max(1, min(width, int(round(box[2] * width)))),
        max(1, min(height, int(round(box[3] * height)))),
    ]
    target[2] = max(target[0] + 1, target[2])
    target[3] = max(target[1] + 1, target[3])
    prompt = str((child.get("sam3") or {}).get("prompt_used") or child.get("sam3_prompt") or "").strip()
    if not prompt:
        return {"status": "not_run", "reason": "missing_child_prompt"}
    try:
        result = infer_candidates_fn(
            session, image_path, prompt, work_dir, score_threshold=0.20,
            positive_bbox_abs_xyxy=target, maximum_candidates=4,
        )
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    records = list((result.get("registry") or {}).get("candidates") or [])
    viable = []
    for record in records:
        try:
            score = float(record.get("score") or 0.0)
            path = Path(str(record.get("mask_path") or "")).expanduser().resolve()
            if not path.is_file() or not path.is_relative_to(work_dir.resolve()):
                continue
            candidate = np.array(Image.open(path).convert("L")) > 0
            if candidate.shape != accepted_mask.shape or not bool(candidate.any()):
                continue
            target_coverage = _mask_bbox_coverage(candidate, box)
            covered_by_accepted = int(np.logical_and(candidate, accepted_mask).sum()) / max(1, int(candidate.sum()))
        except (OSError, ValueError):
            continue
        if (
            score >= cfg.minimum_confirmation_score
            and target_coverage >= cfg.minimum_confirmation_target_coverage
            and covered_by_accepted >= cfg.minimum_confirmation_covered_by_accepted_mask
        ):
            viable.append({
                "candidate_id": str(record.get("candidate_id") or ""),
                "score": round(score, 6),
                "target_coverage": round(target_coverage, 6),
                "covered_by_accepted_mask": round(covered_by_accepted, 6),
            })
    viable.sort(key=lambda item: (item["score"], item["target_coverage"], item["covered_by_accepted_mask"]), reverse=True)
    if not viable:
        return {"status": "not_confirmed", "positive_bbox_abs_xyxy": target, "candidate_count": len(records)}
    return {
        "status": "confirmed",
        "positive_bbox_abs_xyxy": target,
        "candidate_count": len(records),
        "selected": viable[0],
    }


def _host_role(obj: Mapping[str, Any]) -> str | None:
    tokens = _object_tokens(obj)
    if {"coat", "rack"} <= tokens:
        return "coat_rack"
    if "pegboard" in tokens and ("rack" in tokens or "tool" in tokens):
        return "pegboard"
    if tokens & {"crate", "box", "caddy"} and tokens & {"tool", "utensil"}:
        return "tool_crate"
    if "towel" in tokens and ("ladder" in tokens or "rack" in tokens):
        return "towel_support"
    if "drying" in tokens and "rack" in tokens:
        return "drying_rack"
    if "roll" in tokens and ("holder" in tokens or "rack" in tokens):
        return "roll_holder"
    return None


def _child_family(obj: Mapping[str, Any]) -> str | None:
    tokens = _object_tokens(obj)
    if tokens & {"bathrobe", "robe", "coat", "apron", "garment", "clothes", "clothing"}:
        return "garment"
    if "towel" in tokens:
        return "towel"
    if tokens & {"paper", "roll", "tube"}:
        return "roll"
    if tokens & {"tool", "brush", "hammer", "chisel", "scissors", "spatula", "wrench"}:
        return "tool"
    return None


def _explicitly_claims(host: Mapping[str, Any], family: str) -> bool:
    tokens = _object_tokens(host)
    expected = {
        "garment": {"bathrobe", "robe", "apron", "garment", "clothes", "clothing"},
        "towel": {"towel"},
        "roll": {"roll", "tube", "paper"},
        "tool": {"tool", "brush", "hammer", "chisel", "scissors", "spatula", "wrench"},
    }[family]
    return bool(tokens & expected)


def _role_allows(role: str, family: str) -> bool:
    return {
        "coat_rack": {"garment"},
        "pegboard": {"garment", "tool"},
        "tool_crate": {"tool"},
        "towel_support": {"towel"},
        "drying_rack": {"towel", "garment"},
        "roll_holder": {"roll"},
    }.get(role, set()) >= {family}


def _complete_host_prompts(
    role: str,
    host: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    # These are physical assemblies, not open-storage interiors: their
    # attached/contained pixels belong to the host reconstruction object.
    if role == "pegboard":
        return (
            "complete pegboard tool rack with every attached hanging garment and hand tool",
            "complete pegboard including all visible hanging tools and clothing",
        )
    if role == "tool_crate":
        container_name = (
            "tool caddy"
            if "caddy" in _object_tokens(host or {})
            else "tool crate"
        )
        return (
            f"complete {container_name} together with every visible tool contained in or resting on it",
            f"cluster of all visible hand tools held inside and protruding above the {container_name}",
        )
    return ()


def _probe_complete_functional_host(
    *,
    role: str,
    host: Mapping[str, Any],
    host_mask: np.ndarray,
    image_path: str | Path | None,
    session: Any,
    infer_candidates_fn: Callable[..., Mapping[str, Any]] | None,
    work_dir: Path,
    cfg: HostPartMaskFusionConfig,
) -> dict[str, Any]:
    prompts = _complete_host_prompts(role, host)
    if not prompts:
        return {"status": "not_applicable", "inference_count": 0}
    box = _bbox(host)
    if box is None or image_path is None or session is None or infer_candidates_fn is None:
        return {
            "status": "not_run",
            "reason": "missing_local_probe_inputs",
            "inference_count": 0,
        }
    height, width = host_mask.shape
    positive_box = [
        max(0, min(width - 1, int(round(box[0] * width)))),
        max(0, min(height - 1, int(round(box[1] * height)))),
        max(1, min(width, int(round(box[2] * width)))),
        max(1, min(height, int(round(box[3] * height)))),
    ]
    positive_box[2] = max(positive_box[0] + 1, positive_box[2])
    positive_box[3] = max(positive_box[1] + 1, positive_box[3])
    if role == "tool_crate":
        # VLM boxes for caddies/crates commonly stop at the container rim and
        # omit tools protruding above it.  Expand only this functional role's
        # local SAM3 support box upward; generic host boxes remain unchanged.
        box_width = positive_box[2] - positive_box[0]
        box_height = positive_box[3] - positive_box[1]
        pad_x = max(2, int(round(box_width * 0.08)))
        positive_box = [
            max(0, positive_box[0] - pad_x),
            max(0, positive_box[1] - box_height),
            min(width, positive_box[2] + pad_x),
            min(height, positive_box[3] + max(1, int(round(box_height * 0.10)))),
        ]
    host_area = max(1, int(host_mask.sum()))
    viable: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts[: cfg.maximum_complete_host_probes], 1):
        probe_dir = work_dir / "complete_functional_host" / str(host.get("object_id") or "host") / f"probe_{index:02d}"
        try:
            result = infer_candidates_fn(
                session,
                image_path,
                prompt,
                probe_dir,
                score_threshold=0.20,
                positive_bbox_abs_xyxy=positive_box,
                maximum_candidates=cfg.maximum_candidates_per_complete_host_probe,
            )
        except Exception as exc:
            records.append({"prompt": prompt, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        candidate_records = list((result.get("registry") or {}).get("candidates") or [])
        records.append({"prompt": prompt, "status": "completed", "candidate_count": len(candidate_records)})
        for candidate in candidate_records:
            try:
                score = float(candidate.get("score") or 0.0)
                path = Path(str(candidate.get("mask_path") or "")).expanduser().resolve()
                mask = np.array(Image.open(path).convert("L")) > 0
            except (OSError, ValueError):
                continue
            if mask.shape != host_mask.shape or not bool(mask.any()):
                continue
            host_coverage = int(np.logical_and(mask, host_mask).sum()) / host_area
            area_ratio = int(mask.sum()) / host_area
            added_fraction = int(np.logical_and(mask, ~host_mask).sum()) / host_area
            all_candidates.append({
                "mask": mask,
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "prompt": prompt,
                "score": score,
                "host_coverage": host_coverage,
                "added_fraction": added_fraction,
                "area_ratio": area_ratio,
            })
            if (
                score < cfg.minimum_complete_host_score
                or host_coverage < cfg.minimum_complete_host_original_coverage
                or (
                    added_fraction < cfg.minimum_complete_host_added_fraction
                    and not cfg.minimum_idempotent_anchor_area_ratio
                    <= area_ratio <= cfg.maximum_idempotent_anchor_area_ratio
                )
                or area_ratio > cfg.maximum_complete_host_area_ratio
            ):
                continue
            viable.append({
                "mask": mask,
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "prompt": prompt,
                "score": score,
                "host_coverage": host_coverage,
                "added_fraction": added_fraction,
                "area_ratio": area_ratio,
            })
    if not viable:
        return {
            "status": "not_confirmed",
            "inference_count": min(len(prompts), cfg.maximum_complete_host_probes),
            "records": records,
        }
    selected = sorted(
        viable,
        key=lambda item: (item["score"], item["host_coverage"], item["added_fraction"], -item["area_ratio"]),
        reverse=True,
    )[0]
    combined = selected["mask"].copy()
    supplements: list[dict[str, Any]] = []
    host_bbox = [float(value) for value in box]
    for item in sorted(
        all_candidates,
        key=lambda value: (value["score"], value["added_fraction"]),
        reverse=True,
    ):
        if item is selected or np.array_equal(item["mask"], selected["mask"]):
            continue
        candidate_mask = item["mask"]
        ys, xs = np.nonzero(candidate_mask)
        candidate_box = [
            float(xs.min()) / width,
            float(ys.min()) / height,
            float(xs.max() + 1) / width,
            float(ys.max() + 1) / height,
        ]
        bbox_coverage = _coverage_by_expanded_host(
            candidate_box, host_bbox, cfg.supplemental_bbox_expansion_fraction
        )
        new_pixels = int(np.logical_and(candidate_mask, ~combined).sum())
        new_fraction = new_pixels / host_area
        if not (
            item["score"] >= cfg.minimum_supplemental_attachment_score
            and item["area_ratio"] <= cfg.maximum_supplemental_attachment_area_ratio
            and new_fraction >= cfg.minimum_supplemental_added_fraction
            and bbox_coverage >= 0.70
        ):
            continue
        combined |= candidate_mask
        supplements.append({
            "candidate_id": item["candidate_id"],
            "prompt": item["prompt"],
            "score": round(float(item["score"]), 6),
            "bbox_coverage_by_expanded_host": round(bbox_coverage, 6),
            "new_pixels": new_pixels,
            "new_fraction": round(new_fraction, 6),
        })
    if (
        not supplements
        and selected["added_fraction"] < cfg.minimum_complete_host_added_fraction
    ):
        return {
            "status": "not_confirmed",
            "reason": "idempotent_anchor_without_supplement",
            "inference_count": min(len(prompts), cfg.maximum_complete_host_probes),
            "records": records,
            "selected": {key: value for key, value in selected.items() if key != "mask"},
        }
    return {
        "status": "confirmed",
        "inference_count": min(len(prompts), cfg.maximum_complete_host_probes),
        "records": records,
        "selected": {key: value for key, value in selected.items() if key != "mask"},
        "supplemental_attachments": supplements,
        "mask": combined,
    }


def _success_mask(obj: Mapping[str, Any], shape: tuple[int, int], staging: Path) -> np.ndarray | None:
    sam3 = obj.get("sam3") or {}
    if sam3.get("status") != "success":
        return None
    path = Path(str(sam3.get("mask_path") or "")).expanduser().resolve()
    if not path.is_file() or not path.is_relative_to(staging):
        return None
    mask = np.array(Image.open(path).convert("L")) > 0
    return mask if mask.shape == shape and bool(mask.any()) else None


def _write_mask_and_overlay(
    obj: dict[str, Any],
    mask: np.ndarray,
    image_np: np.ndarray,
    staging: Path,
    *,
    selection_method: str = "dedicated_attachment_host_union",
) -> None:
    object_id = str(obj.get("object_id") or "object")
    category = re.sub(r"[^a-z0-9]+", "_", str(obj.get("category") or "object").lower()).strip("_") or "object"
    mask_path = staging / "masks" / f"{object_id}_{category}_mask.png"
    overlay_path = staging / "overlays" / f"{object_id}_{category}_overlay.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    overlay = image_np.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array((255, 0, 0), dtype=np.uint8)).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)
    ys, xs = np.nonzero(mask)
    sam3 = copy.deepcopy(dict(obj.get("sam3") or {}))
    sam3.update({
        "mask_path": str(mask_path), "overlay_path": str(overlay_path),
        "area_pixels": int(mask.sum()),
        "mask_bbox_abs_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "selection_method": selection_method,
    })
    obj["sam3"] = sam3


def _renumber(objects: list[dict[str, Any]], staging: Path) -> dict[str, str]:
    remap: dict[str, str] = {}
    for index, obj in enumerate(objects, start=1):
        old_id = str(obj.get("object_id") or "")
        new_id = f"proposal_{index:03d}"
        remap[old_id] = new_id
        obj["object_id"] = new_id
    for obj in objects:
        old_id = next(key for key, value in remap.items() if value == obj["object_id"])
        if old_id == obj["object_id"]:
            continue
        sam3 = obj.get("sam3") or {}
        for key in ("mask_path", "overlay_path"):
            path = Path(str(sam3.get(key) or "")).expanduser().resolve()
            if not path.is_file() or not path.is_relative_to(staging) or not path.name.startswith(old_id + "_"):
                continue
            target = path.with_name(obj["object_id"] + path.name[len(old_id):])
            if target.exists():
                target.unlink()
            path.rename(target)
            sam3[key] = str(target)
    return remap


def run_host_part_mask_fusion(
    *, image_np: np.ndarray, objects: Sequence[Mapping[str, Any]], staging_dir: str | Path,
    work_dir: str | Path, config: HostPartMaskFusionConfig | None = None,
    image_path: str | Path | None = None, session: Any = None,
    infer_candidates_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = config or HostPartMaskFusionConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("host part mask fusion is isolated-attempt only")
    copied = [copy.deepcopy(dict(item)) for item in objects]
    staging = Path(staging_dir).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": "scenemosaic_dedicated_attachment_absorption",
        "enabled": cfg.enabled, "status": "disabled" if not cfg.enabled else "noop",
        "absorbed_hosts": [], "removed_duplicate_additions": [],
        "removed_object_ids": [], "completed_host_source_ids": [],
        "positive_bbox_duplicate_confirmations": [],
        "functional_host_completions": [],
        "semantic_component_absorptions": [],
        "semantic_component_rejections": [],
        "id_remap": {}, "hard_errors": [],
        "additional_sam3_inference_calls": 0, "additional_vlm_calls": 0,
    }
    if not cfg.enabled:
        _atomic_json(work / "dedicated_attachment_absorption_audit.json", audit)
        return {"objects": copied, "audit": audit}
    shape = tuple(int(value) for value in image_np.shape[:2])
    remove_ids: set[str] = set()
    positive_confirmation_count = 0
    for host in copied:
        role = _host_role(host)
        host_box = _bbox(host)
        host_mask = _success_mask(host, shape, staging)
        if role is None or host_box is None or host_mask is None or not _has_pass_source(host, "A"):
            continue
        completion = _probe_complete_functional_host(
            role=role,
            host=host,
            host_mask=host_mask,
            image_path=image_path,
            session=session,
            infer_candidates_fn=infer_candidates_fn,
            work_dir=work,
            cfg=cfg,
        )
        completion_mask = completion.pop("mask", None)
        audit["additional_sam3_inference_calls"] += int(
            completion.get("inference_count") or 0
        )
        audit["functional_host_completions"].append(
            {
                "host_object_id": str(host.get("object_id") or ""),
                "host_role": role,
                **completion,
            }
        )
        completed_by_probe = completion_mask is not None
        accepted: list[tuple[dict[str, Any], np.ndarray, str, float]] = []
        for child in copied:
            if child is host or not _has_pass_source(child, "B"):
                continue
            family = _child_family(child)
            child_box = _bbox(child)
            child_mask = _success_mask(child, shape, staging)
            if family is None or child_box is None or child_mask is None:
                continue
            coverage = _coverage_by_expanded_host(
                child_box, host_box, cfg.host_bbox_expansion_fraction
            )
            explicit_or_intrinsic = _explicitly_claims(host, family) or (
                role == "pegboard" and family in {"garment", "tool"}
            ) or (role == "tool_crate" and family == "tool")
            if (
                _role_allows(role, family)
                and explicit_or_intrinsic
                and coverage >= cfg.minimum_child_box_coverage_by_host
            ):
                accepted.append((child, child_mask, family, coverage))
        if not accepted and not completed_by_probe:
            continue
        merged = (
            np.asarray(completion_mask, dtype=bool).copy()
            if completed_by_probe
            else host_mask.copy()
        )
        host_remove_ids: set[str] = set()
        for child, child_mask, _, _ in accepted:
            merged |= child_mask
            host_remove_ids.add(str(child.get("object_id") or ""))
        removed_failed = []
        for child in copied:
            child_id = str(child.get("object_id") or "")
            if (
                child is host
                or child_id in host_remove_ids
                or not _has_pass_source(child, "B")
            ):
                continue
            family = _child_family(child)
            child_box = _bbox(child)
            status = str((child.get("sam3") or {}).get("status") or "")
            if family is None or child_box is None or status != "failed":
                continue
            explicit_or_intrinsic = _explicitly_claims(host, family) or (
                role == "pegboard" and family in {"garment", "tool"}
            ) or (role == "tool_crate" and family == "tool")
            if not (_role_allows(role, family) and explicit_or_intrinsic):
                continue
            host_coverage = _coverage_by_expanded_host(
                child_box, host_box, cfg.host_bbox_expansion_fraction
            )
            mask_coverage = _mask_bbox_coverage(merged, child_box)
            if host_coverage < cfg.minimum_child_box_coverage_by_host:
                continue
            if mask_coverage >= cfg.minimum_failed_sibling_box_coverage:
                host_remove_ids.add(child_id)
                removed_failed.append({
                    "object_id": child_id, "family": family,
                    "accepted_mask_bbox_coverage": round(mask_coverage, 6),
                    "resolution": "direct_accepted_mask_coverage",
                })
                continue
            if positive_confirmation_count >= cfg.maximum_positive_bbox_duplicate_confirmations:
                continue
            confirmation = _confirm_failed_duplicate_with_positive_bbox(
                child=child, accepted_mask=merged, image_path=image_path,
                session=session, infer_candidates_fn=infer_candidates_fn,
                work_dir=work / "positive_bbox_duplicate_confirmation" / child_id,
                cfg=cfg,
            )
            positive_confirmation_count += 1
            audit["additional_sam3_inference_calls"] += 1
            audit["positive_bbox_duplicate_confirmations"].append({
                "host_object_id": str(host.get("object_id") or ""),
                "failed_child_object_id": child_id, "family": family,
                "accepted_mask_bbox_coverage": round(mask_coverage, 6),
                **confirmation,
            })
            if confirmation.get("status") == "confirmed":
                host_remove_ids.add(child_id)
                removed_failed.append({
                    "object_id": child_id, "family": family,
                    "accepted_mask_bbox_coverage": round(mask_coverage, 6),
                    "resolution": "positive_bbox_duplicate_confirmation",
                    "confirmation": confirmation,
                })
        if not host_remove_ids and not completed_by_probe:
            continue
        remove_ids.update(host_remove_ids)
        _write_mask_and_overlay(
            host,
            merged,
            image_np,
            staging,
            selection_method=(
                "dedicated_attachment_host_completion_union"
                if completed_by_probe
                else "dedicated_attachment_host_union"
            ),
        )
        host_sources = _pass_a_source_ids(host)
        audit["completed_host_source_ids"].extend(host_sources)
        audit["absorbed_hosts"].append({
            "host_object_id": str(host.get("object_id") or ""),
            "host_source_object_ids": host_sources,
            "host_role": role,
            "completion_probe_status": completion.get("status"),
            "accepted_children": [
                {
                    "object_id": str(child.get("object_id") or ""),
                    "family": family,
                    "bbox_coverage_by_expanded_host": round(coverage, 6),
                }
                for child, _, family, coverage in accepted
            ],
            "removed_failed_duplicates": removed_failed,
        })
        audit["removed_duplicate_additions"].extend(removed_failed)

    # A Pass-B item such as "sofa cushion" can be a named intrinsic part of
    # one pre-existing furniture object.  Unlike a loose "throw pillow", the
    # shared host noun makes this ownership explicit.  Absorb only when mask
    # and bbox containment prove a unique host; otherwise leave the quality
    # gate to report the ambiguity rather than guessing.
    if cfg.enable_semantic_component_absorption:
        semantic_children_by_host: dict[str, list[dict[str, Any]]] = {}
        for child in copied:
            child_id = str(child.get("object_id") or "")
            if (
                not child_id
                or child_id in remove_ids
                or not _has_pass_source(child, "B")
            ):
                continue
            child_box = _bbox(child)
            child_mask = _success_mask(child, shape, staging)
            if child_box is None or child_mask is None:
                continue
            child_area = max(1, int(child_mask.sum()))
            matches: list[dict[str, Any]] = []
            for host in copied:
                host_id = str(host.get("object_id") or "")
                if (
                    not host_id
                    or host_id == child_id
                    or host_id in remove_ids
                    or not _has_pass_source(host, "A")
                ):
                    continue
                anchors = _semantic_component_anchor_tokens(host, child)
                if not anchors:
                    continue
                host_box = _bbox(host)
                host_mask = _success_mask(host, shape, staging)
                if host_box is None or host_mask is None:
                    continue
                host_area = max(1, int(host_mask.sum()))
                contained_pixels = int(np.logical_and(child_mask, host_mask).sum())
                child_containment = contained_pixels / child_area
                child_to_host_area_ratio = child_area / host_area
                bbox_coverage = _coverage_by_expanded_host(
                    child_box, host_box, cfg.host_bbox_expansion_fraction
                )
                if (
                    child_containment
                    < cfg.minimum_semantic_component_mask_containment
                    or child_to_host_area_ratio
                    > cfg.maximum_semantic_component_area_ratio
                    or bbox_coverage
                    < cfg.minimum_child_box_coverage_by_host
                ):
                    continue
                matches.append(
                    {
                        "host": host,
                        "host_id": host_id,
                        "child_containment": child_containment,
                        "child_to_host_area_ratio": child_to_host_area_ratio,
                        "bbox_coverage_by_expanded_host": bbox_coverage,
                        "shared_host_tokens": anchors,
                    }
                )
            if len(matches) != 1:
                if len(matches) > 1:
                    audit["semantic_component_rejections"].append(
                        {
                            "child_object_id": child_id,
                            "reason": "ambiguous_semantic_component_host",
                            "candidate_host_object_ids": sorted(
                                str(match["host_id"]) for match in matches
                            ),
                        }
                    )
                continue
            match = matches[0]
            semantic_children_by_host.setdefault(
                str(match["host_id"]), []
            ).append(
                {
                    "child": child,
                    "child_mask": child_mask,
                    **{key: value for key, value in match.items() if key != "host"},
                }
            )

        for host_id, records in semantic_children_by_host.items():
            host = next(
                (
                    item
                    for item in copied
                    if str(item.get("object_id") or "") == host_id
                ),
                None,
            )
            if host is None:
                continue
            host_mask = _success_mask(host, shape, staging)
            if host_mask is None:
                continue
            merged = host_mask.copy()
            child_records: list[dict[str, Any]] = []
            for record in records:
                child = record["child"]
                child_id = str(child.get("object_id") or "")
                added_pixels = int(
                    np.logical_and(record["child_mask"], ~merged).sum()
                )
                merged |= record["child_mask"]
                remove_ids.add(child_id)
                child_records.append(
                    {
                        "object_id": child_id,
                        "shared_host_tokens": record["shared_host_tokens"],
                        "child_mask_containment": round(
                            float(record["child_containment"]), 6
                        ),
                        "child_to_host_area_ratio": round(
                            float(record["child_to_host_area_ratio"]), 6
                        ),
                        "bbox_coverage_by_expanded_host": round(
                            float(record["bbox_coverage_by_expanded_host"]), 6
                        ),
                        "new_host_pixels": added_pixels,
                    }
                )
            if not np.array_equal(merged, host_mask):
                _write_mask_and_overlay(
                    host,
                    merged,
                    image_np,
                    staging,
                    selection_method="semantic_host_component_absorption_union",
                )
            audit["semantic_component_absorptions"].append(
                {
                    "host_object_id": host_id,
                    "host_source_object_ids": _pass_a_source_ids(host),
                    "resolution": "unique_named_intrinsic_component",
                    "absorbed_children": child_records,
                }
            )
    if remove_ids or audit["completed_host_source_ids"]:
        for item in copied:
            if str(item.get("object_id") or "") not in remove_ids:
                continue
            for key in ("mask_path", "overlay_path"):
                path = Path(str((item.get("sam3") or {}).get(key) or "")).expanduser().resolve()
                if path.is_file() and path.is_relative_to(staging):
                    path.unlink()
        copied = [item for item in copied if str(item.get("object_id") or "") not in remove_ids]
        audit["removed_object_ids"] = sorted(remove_ids)
        if remove_ids:
            audit["id_remap"] = _renumber(copied, staging)
        audit["completed_host_source_ids"] = sorted(set(audit["completed_host_source_ids"]))
        audit["status"] = "pass"
    _atomic_json(work / "dedicated_attachment_absorption_audit.json", audit)
    return {"objects": copied, "audit": audit}
