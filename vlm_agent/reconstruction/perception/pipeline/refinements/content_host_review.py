#!/usr/bin/env python3
"""Build a bounded high-resolution review panel for content-bearing objects.

This module is intentionally semantic-light.  It selects only objects whose
existing Pass-A names consistently describe an exposed support surface or open
storage carrier, then renders local crops for the existing Pass-B call.  The
crops are attention aids, not segmentation evidence and never create objects.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .repair_scope_router import classify_object_role


@dataclass(frozen=True)
class ContentHostConfig:
    maximum_hosts: int = 8
    minimum_score: float = 0.68
    minimum_bbox_area_fraction: float = 0.0015
    maximum_bbox_area_fraction: float = 0.35
    duplicate_iou: float = 0.65
    duplicate_containment: float = 0.85
    maximum_crop_iou: float = 0.55
    fail_on_budget_overflow: bool = True


OPEN_STORAGE_HEADS = {
    "bookcase",
    "bookshelf",
    "cubby",
    "hutch",
    "rack",
    "shelf",
    "shelving",
}
SUPPORT_SURFACE_HEADS = {
    "cabinet",
    "cart",
    "console",
    "counter",
    "countertop",
    "desk",
    "dresser",
    "island",
    "nightstand",
    "sideboard",
    "table",
    "vanity",
}
GENERIC_HOST_SUFFIXES = {"fixture", "furniture", "storage", "system", "unit"}
SUPPORTED_TYPES = {"furniture", "structural_surface"}
CONFLICT_TYPES = {
    "appliance",
    "lighting",
    "opening",
    "plant",
    "small_object",
    "textile",
}


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _field_host_mode(value: Any) -> str | None:
    tokens = _tokens(value)
    while len(tokens) > 1 and tokens[-1] in GENERIC_HOST_SUFFIXES:
        tokens.pop()
    if not tokens:
        return None
    head = tokens[-1]
    if head in OPEN_STORAGE_HEADS:
        return "open_storage"
    if head in SUPPORT_SURFACE_HEADS:
        return "support_surface"
    return None


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        and 0 <= float(value[0]) < float(value[2]) <= 1
        and 0 <= float(value[1]) < float(value[3]) <= 1
    )


def _bbox_metrics(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left[2]) - float(left[0])) * (
        float(left[3]) - float(left[1])
    )
    right_area = (float(right[2]) - float(right[0])) * (
        float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "left_coverage": intersection / left_area if left_area else 0.0,
        "right_coverage": intersection / right_area if right_area else 0.0,
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _expand_interval(
    start: float,
    end: float,
    minimum_size: float,
) -> tuple[float, float]:
    size = end - start
    if size >= minimum_size:
        return max(0.0, start), min(1.0, end)
    center = (start + end) / 2
    start = center - minimum_size / 2
    end = center + minimum_size / 2
    if start < 0:
        end -= start
        start = 0.0
    if end > 1:
        start -= end - 1
        end = 1.0
    return max(0.0, start), min(1.0, end)


def _initial_crop(host: Mapping[str, Any]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in host["bbox_norm_xyxy"]]
    width = x2 - x1
    height = y2 - y1
    if host["host_mode"] == "open_storage":
        x_pad = min(0.04, max(0.015, 0.06 * width))
        y_pad = min(0.04, max(0.015, 0.06 * height))
        crop = [x1 - x_pad, y1 - y_pad, x2 + x_pad, y2 + y_pad]
        min_width, min_height = 0.14, 0.20
    else:
        x_pad = _clamp(0.15 * width, 0.02, 0.06)
        text = " ".join(
            str(host.get(key) or "")
            for key in ("category", "sam3_prompt", "instance_name", "description")
        ).lower()
        partial_at_edge = (
            any(token in text for token in ("partial", "cropped", "truncated"))
            and (x1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99)
        )
        top_pad = (
            _clamp(1.75 * height, 0.12, 0.30)
            if partial_at_edge
            else _clamp(0.75 * height, 0.05, 0.18)
        )
        crop = [x1 - x_pad, y1 - top_pad, x2 + x_pad, y1 + 0.55 * height]
        min_width, min_height = 0.14, 0.18
    crop[0], crop[2] = _expand_interval(crop[0], crop[2], min_width)
    crop[1], crop[3] = _expand_interval(crop[1], crop[3], min_height)
    return [round(_clamp(value, 0.0, 1.0), 6) for value in crop]


def _adjust_neighbor_padding(hosts: list[dict[str, Any]]) -> None:
    """Shrink only padding between disjoint neighboring host bboxes."""

    for left_index in range(len(hosts)):
        for right_index in range(left_index + 1, len(hosts)):
            left = hosts[left_index]
            right = hosts[right_index]
            lb = left["bbox_norm_xyxy"]
            rb = right["bbox_norm_xyxy"]
            lc = left["crop_norm_xyxy"]
            rc = right["crop_norm_xyxy"]
            vertical_overlap = min(lb[3], rb[3]) - max(lb[1], rb[1])
            horizontal_overlap = min(lb[2], rb[2]) - max(lb[0], rb[0])
            if lb[2] <= rb[0] and vertical_overlap > 0:
                split = (lb[2] + rb[0]) / 2
                if lc[2] > split and rc[0] < split:
                    lc[2] = split
                    rc[0] = split
            elif rb[2] <= lb[0] and vertical_overlap > 0:
                split = (rb[2] + lb[0]) / 2
                if rc[2] > split and lc[0] < split:
                    rc[2] = split
                    lc[0] = split
            if lb[3] <= rb[1] and horizontal_overlap > 0:
                split = (lb[3] + rb[1]) / 2
                if lc[3] > split and rc[1] < split:
                    lc[3] = split
                    rc[1] = split
            elif rb[3] <= lb[1] and horizontal_overlap > 0:
                split = (rb[3] + lb[1]) / 2
                if rc[3] > split and lc[1] < split:
                    rc[3] = split
                    lc[1] = split
    for host in hosts:
        host["crop_norm_xyxy"] = [
            round(float(value), 6) for value in host["crop_norm_xyxy"]
        ]


def select_content_hosts(
    objects: Sequence[Mapping[str, Any]],
    *,
    config: ContentHostConfig | None = None,
) -> dict[str, Any]:
    """Select a bounded set of content-bearing regions from Pass-A objects."""

    cfg = config or ContentHostConfig()
    if cfg.maximum_hosts < 1:
        raise ValueError("maximum_hosts must be positive")
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    hard_errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in objects:
        item = dict(raw)
        object_id = str(item.get("object_id") or "")
        if not object_id or object_id in seen_ids:
            hard_errors.append(
                {"type": "invalid_or_duplicate_object_id", "object_id": object_id}
            )
            continue
        seen_ids.add(object_id)
        bbox = item.get("bbox_2d_norm_xyxy")
        if not _valid_bbox(bbox):
            modes = [
                _field_host_mode(item.get(key))
                for key in ("category", "sam3_prompt", "instance_name")
            ]
            if any(modes):
                hard_errors.append(
                    {"type": "strong_host_invalid_bbox", "object_id": object_id}
                )
            else:
                rejections.append(
                    {"object_id": object_id, "reason": "invalid_bbox_nonhost"}
                )
            continue

        modes = [
            _field_host_mode(item.get(key))
            for key in ("category", "sam3_prompt", "instance_name")
        ]
        mode_counts = {
            mode: modes.count(mode)
            for mode in ("open_storage", "support_surface")
        }
        host_mode = max(mode_counts, key=mode_counts.get)
        votes = mode_counts[host_mode]
        scope_audit = classify_object_role(item)
        scope_router_override = False
        object_type = str(item.get("object_type") or "unknown")
        if (
            object_type not in CONFLICT_TYPES
            and scope_audit.get("host_mode") != host_mode
        ):
            routed_mode = scope_audit.get("host_mode")
            confidence = float(scope_audit.get("confidence") or 0.0)
            if (
                routed_mode in {"open_storage", "support_surface"}
                and confidence >= 0.90
            ):
                # Explicit category semantics outrank generic end words.
                host_mode = str(routed_mode)
                votes = max(2, mode_counts[host_mode])
                scope_router_override = True
            else:
                rejections.append(
                    {
                        "object_id": object_id,
                        "reason": "scope_router_disagreement",
                        "vote_host_mode": host_mode,
                        "scope_router": scope_audit,
                    }
                )
                continue
        if votes < 2:
            rejections.append(
                {
                    "object_id": object_id,
                    "reason": "insufficient_host_head_votes",
                    "field_modes": modes,
                }
            )
            continue

        object_type = str(item.get("object_type") or "unknown")
        if object_type in CONFLICT_TYPES:
            rejections.append(
                {
                    "object_id": object_id,
                    "reason": "semantic_type_conflict",
                    "host_mode": host_mode,
                    "object_type": object_type,
                }
            )
            continue
        if object_type in SUPPORTED_TYPES:
            type_support = 1.0
        elif object_type == "unknown":
            type_support = 0.5 if votes == 3 else 0.0
        elif object_type == "decor" and host_mode == "open_storage":
            type_support = 0.5
        else:
            rejections.append(
                {
                    "object_id": object_id,
                    "reason": "unsupported_host_type",
                    "host_mode": host_mode,
                    "object_type": object_type,
                }
            )
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]
        area = (x2 - x1) * (y2 - y1)
        if not cfg.minimum_bbox_area_fraction <= area <= cfg.maximum_bbox_area_fraction:
            rejections.append(
                {
                    "object_id": object_id,
                    "reason": "host_bbox_area_out_of_range",
                    "host_mode": host_mode,
                    "area_fraction": round(area, 8),
                }
            )
            continue
        field_agreement = votes / 3
        affordance = 1.0 if host_mode == "open_storage" else 0.8
        score = (
            0.40 * min(votes / 2, 1.0)
            + 0.20 * field_agreement
            + 0.20 * type_support
            + 0.10
            + 0.10 * affordance
        )
        if score < cfg.minimum_score:
            rejections.append(
                {
                    "object_id": object_id,
                    "reason": "host_score_below_threshold",
                    "score": round(score, 6),
                }
            )
            continue
        accepted.append(
            {
                "source_object_id": object_id,
                "category": str(item.get("category") or ""),
                "sam3_prompt": str(item.get("sam3_prompt") or ""),
                "instance_name": str(item.get("instance_name") or ""),
                "description": str(item.get("description") or ""),
                "object_type": object_type,
                "host_mode": host_mode,
                "head_votes": votes,
                "scope_router_override": scope_router_override,
                "scope_router_audit": scope_audit,
                "score": round(score, 6),
                "bbox_norm_xyxy": [round(float(value), 6) for value in bbox],
                "bbox_area_fraction": round(area, 8),
            }
        )

    accepted.sort(
        key=lambda item: (
            -item["score"],
            -item["head_votes"],
            0 if item["host_mode"] == "open_storage" else 1,
            item["bbox_norm_xyxy"][1],
            item["bbox_norm_xyxy"][0],
            item["source_object_id"],
        )
    )
    deduplicated: list[dict[str, Any]] = []
    duplicate_rejections: list[dict[str, Any]] = []
    for candidate in accepted:
        duplicate = None
        for existing in deduplicated:
            metrics = _bbox_metrics(
                candidate["bbox_norm_xyxy"], existing["bbox_norm_xyxy"]
            )
            if (
                candidate["host_mode"] == existing["host_mode"]
                and (
                    metrics["iou"] >= cfg.duplicate_iou
                    or max(metrics["left_coverage"], metrics["right_coverage"])
                    >= cfg.duplicate_containment
                )
            ):
                duplicate = (existing, metrics)
                break
        if duplicate is None:
            deduplicated.append(candidate)
        else:
            duplicate_rejections.append(
                {
                    "source_object_id": candidate["source_object_id"],
                    "duplicate_of_object_id": duplicate[0]["source_object_id"],
                    "overlap": {
                        key: round(value, 6) for key, value in duplicate[1].items()
                    },
                }
            )

    overflow = deduplicated[cfg.maximum_hosts :]
    selected = [dict(item) for item in deduplicated[: cfg.maximum_hosts]]
    if overflow and cfg.fail_on_budget_overflow:
        hard_errors.append(
            {
                "type": "content_host_budget_overflow",
                "count": len(overflow),
                "source_object_ids": [
                    item["source_object_id"] for item in overflow
                ],
            }
        )
    for host in selected:
        host["crop_norm_xyxy"] = _initial_crop(host)
    _adjust_neighbor_padding(selected)
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            metrics = _bbox_metrics(
                left["crop_norm_xyxy"], right["crop_norm_xyxy"]
            )
            if metrics["iou"] > cfg.maximum_crop_iou:
                hard_errors.append(
                    {
                        "type": "content_host_crop_ambiguity",
                        "left_object_id": left["source_object_id"],
                        "right_object_id": right["source_object_id"],
                        "crop_iou": round(metrics["iou"], 6),
                    }
                )

    selected.sort(
        key=lambda item: (
            item["bbox_norm_xyxy"][1],
            item["bbox_norm_xyxy"][0],
            item["source_object_id"],
        )
    )
    for index, host in enumerate(selected, start=1):
        host["host_id"] = f"H_{index:03d}"
    return {
        "schema_version": "scenemosaic_content_host_selection",
        "status": "pass" if not hard_errors else "blocked",
        "config": asdict(cfg),
        "input_object_count": len(objects),
        "eligible_before_deduplication": len(accepted),
        "selected_count": len(selected),
        "selected": selected,
        "rejections": rejections,
        "duplicate_rejections": duplicate_rejections,
        "overflow_rejections": overflow,
        "hard_errors": hard_errors,
        "additional_vlm_calls": 0,
    }


def _norm_to_abs(
    bbox: Sequence[float], width: int, height: int
) -> list[int]:
    x1 = max(0, min(width - 1, int(math.floor(float(bbox[0]) * width))))
    y1 = max(0, min(height - 1, int(math.floor(float(bbox[1]) * height))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox[2]) * width))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox[3]) * height))))
    return [x1, y1, x2, y2]


def _draw_global_grid(
    crop: Image.Image,
    crop_norm: Sequence[float],
) -> Image.Image:
    result = crop.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    font = _font(max(13, min(20, width // 24)))
    x1, y1, x2, y2 = [float(value) for value in crop_norm]
    for step in range(1, 4):
        fraction = step / 4
        x = int(round(fraction * width))
        y = int(round(fraction * height))
        gx = x1 + fraction * (x2 - x1)
        gy = y1 + fraction * (y2 - y1)
        draw.line((x, 0, x, height), fill=(0, 210, 230), width=1)
        draw.line((0, y, width, y), fill=(0, 210, 230), width=1)
        draw.text((x + 2, 2), f"x={gx:.3f}", fill=(0, 255, 255), font=font)
        draw.text((2, y + 2), f"y={gy:.3f}", fill=(0, 255, 255), font=font)
    return result


def materialize_content_host_review(
    *,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    config: ContentHostConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render local crops and one numbered content-host panel."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    crops_dir = output / "crops"
    crops_dir.mkdir()
    selection = select_content_hosts(objects, config=config)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    records = []
    for host in selection["selected"]:
        record = dict(host)
        crop_abs = _norm_to_abs(record["crop_norm_xyxy"], width, height)
        crop = image.crop(tuple(crop_abs))
        crop = _draw_global_grid(crop, record["crop_norm_xyxy"])
        crop_path = crops_dir / f"{record['host_id']}_{record['source_object_id']}.jpg"
        crop.save(crop_path, quality=92)
        record.update(
            {
                "crop_abs_xyxy": crop_abs,
                "crop_path": str(crop_path),
                "crop_to_global": {
                    "x_offset_norm": record["crop_norm_xyxy"][0],
                    "y_offset_norm": record["crop_norm_xyxy"][1],
                    "x_scale_norm": round(
                        record["crop_norm_xyxy"][2] - record["crop_norm_xyxy"][0],
                        6,
                    ),
                    "y_scale_norm": round(
                        record["crop_norm_xyxy"][3] - record["crop_norm_xyxy"][1],
                        6,
                    ),
                },
            }
        )
        records.append(record)

    columns = 2
    cell_width = 640
    rows = max(1, math.ceil(len(records) / columns))
    cell_height = max(250, min(360, 1240 // rows))
    header_height = 40
    panel = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows * cell_height),
        "white",
    )
    ImageDraw.Draw(panel).text(
        (10, 8),
        "CONTENT-HOST HIGH-RES REVIEW (H IDs are context only)",
        fill="black",
        font=_font(22),
    )
    for index, record in enumerate(records):
        crop = Image.open(record["crop_path"]).convert("RGB")
        crop.thumbnail(
            (cell_width - 16, cell_height - 62), Image.Resampling.LANCZOS
        )
        x0 = (index % columns) * cell_width
        y0 = header_height + (index // columns) * cell_height
        panel.paste(
            crop,
            (
                x0 + (cell_width - crop.width) // 2,
                y0 + 50 + (cell_height - 58 - crop.height) // 2,
            ),
        )
        label = (
            f"{record['host_id']}  {record['source_object_id']}  "
            f"{record['category']}  g={record['crop_norm_xyxy']}"
        )
        ImageDraw.Draw(panel).text(
            (x0 + 8, y0 + 8), label, fill="black", font=_font(17)
        )
    panel_path = output / "content_host_panel.jpg"
    panel.save(panel_path, quality=90)
    manifest = {
        **selection,
        "source_image": str(Path(image_path).expanduser().resolve()),
        "image_size": [width, height],
        "selected": records,
        "content_host_panel": str(panel_path),
    }
    manifest_path = output / "content_host_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "panel_path": str(panel_path),
    }


def _expanded_abs_crop(
    bbox: Sequence[int],
    *,
    image_size: tuple[int, int],
    padding_fraction: float = 0.75,
) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = [int(value) for value in bbox]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    target_width = max(
        int(math.ceil(box_width * (1 + 2 * padding_fraction))),
        int(math.ceil(width * 0.10)),
    )
    target_height = max(
        int(math.ceil(box_height * (1 + 2 * padding_fraction))),
        int(math.ceil(height * 0.10)),
    )
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = int(math.floor(center_x - target_width / 2))
    top = int(math.floor(center_y - target_height / 2))
    right = left + target_width
    bottom = top + target_height
    if left < 0:
        right -= left
        left = 0
    if right > width:
        left -= right - width
        right = width
    if top < 0:
        bottom -= top
        top = 0
    if bottom > height:
        top -= bottom - height
        bottom = height
    return [
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    ]


def materialize_combined_review_panel(
    *,
    image_path: str | Path,
    local_candidates: Sequence[Mapping[str, Any]],
    content_hosts: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Pack LC evidence and H attention crops into one bounded JPEG.

    H tiles use two grid slots because they must retain enough resolution for
    small shelf or tabletop contents. LC tiles use one slot and remain the
    only entries eligible for the local-candidate accept/reject protocol.
    """

    candidates = [dict(item) for item in local_candidates]
    hosts = [dict(item) for item in content_hosts]
    if not candidates and not hosts:
        raise ValueError("combined review panel requires candidates or hosts")
    if len(candidates) > 24:
        raise ValueError("combined review panel supports at most 24 LC candidates")
    if len(hosts) > 8:
        raise ValueError("combined review panel supports at most 8 content hosts")
    expected_lc_ids = [
        f"LC_{index:03d}" for index in range(1, len(candidates) + 1)
    ]
    actual_lc_ids = [
        str(item.get("candidate_id") or "") for item in candidates
    ]
    if actual_lc_ids != expected_lc_ids:
        raise ValueError("LC IDs must be continuous and visually ordered")
    expected_h_ids = [
        f"H_{index:03d}" for index in range(1, len(hosts) + 1)
    ]
    actual_h_ids = [str(item.get("host_id") or "") for item in hosts]
    if actual_h_ids != expected_h_ids:
        raise ValueError("content-host IDs must be continuous and visually ordered")

    source = Image.open(image_path).convert("RGB")
    source_size = source.size
    items: list[dict[str, Any]] = []
    for host in hosts:
        crop_path = Path(str(host.get("crop_path") or "")).expanduser().resolve()
        if not crop_path.is_file():
            raise FileNotFoundError(crop_path)
        items.append(
            {
                "kind": "host",
                "span": (
                    5 if host.get("host_mode") == "open_storage" else 2
                ),
                "host_mode": host.get("host_mode"),
                "id": host["host_id"],
                "label": (
                    f"{host['host_id']} {host['source_object_id']} "
                    f"{host['category']} g={host['crop_norm_xyxy']}"
                ),
                "image": Image.open(crop_path).convert("RGB"),
            }
        )
    for candidate in candidates:
        overlay_path = Path(
            str(candidate.get("overlay_path") or "")
        ).expanduser().resolve()
        if not overlay_path.is_file():
            raise FileNotFoundError(overlay_path)
        overlay = Image.open(overlay_path).convert("RGB")
        if overlay.size != source_size:
            raise ValueError(
                f"candidate overlay size mismatch: {candidate['candidate_id']}"
            )
        bbox = candidate.get("bbox_abs_xyxy")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) for value in bbox)
        ):
            raise ValueError(
                f"candidate bbox must be integer xyxy: {candidate['candidate_id']}"
            )
        crop_abs = _expanded_abs_crop(bbox, image_size=source_size)
        prompt = str(
            candidate.get("prompt")
            or candidate.get("sam3_prompt")
            or candidate.get("category_hint")
            or "object"
        ).strip()
        items.append(
            {
                "kind": "candidate",
                "span": 1,
                "id": candidate["candidate_id"],
                "label": (
                    f"{candidate['candidate_id']} {prompt} "
                    f"s={float(candidate.get('score') or 0):.3f}"
                ),
                "image": overlay.crop(tuple(crop_abs)),
            }
        )

    rows: list[list[dict[str, Any]]] = []
    remaining_hosts = [item for item in items if item["kind"] == "host"]
    remaining_candidates = [
        item for item in items if item["kind"] == "candidate"
    ]
    while remaining_hosts or remaining_candidates:
        row: list[dict[str, Any]] = []
        capacity = 5
        while remaining_candidates and capacity >= 1:
            row.append(remaining_candidates.pop(0))
            capacity -= 1
        while remaining_hosts:
            fitting_index = next(
                (
                    index
                    for index, item in enumerate(remaining_hosts)
                    if int(item["span"]) <= capacity
                ),
                None,
            )
            if fitting_index is None:
                break
            item = remaining_hosts.pop(fitting_index)
            row.append(item)
            capacity -= int(item["span"])
        rows.append(row)

    panel_width = 1280
    header_height = 40
    slot_width = panel_width // 5
    row_weights = [
        (
            4.0
            if any(
                item.get("kind") == "host"
                and item.get("host_mode") == "open_storage"
                for item in row
            )
            else (
                1.5
                if any(item.get("kind") == "candidate" for item in row)
                else 1.0
            )
        )
        for row in rows
    ]
    height_unit = 1240.0 / sum(row_weights)
    row_heights = [max(72, int(round(height_unit * weight))) for weight in row_weights]
    row_heights[-1] += 1240 - sum(row_heights)
    panel = Image.new(
        "RGB",
        (panel_width, header_height + sum(row_heights)),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (10, 8),
        "PASS-B REVIEW: H=context only; LC=SAM3 mask evidence",
        fill="black",
        font=_font(21),
    )
    y_offsets = []
    running_y = header_height
    for row_height in row_heights:
        y_offsets.append(running_y)
        running_y += row_height
    for row_index, row in enumerate(rows):
        row_height = row_heights[row_index]
        slot_index = 0
        for item in row:
            cell_width = slot_width * int(item["span"])
            x0 = slot_index * slot_width
            y0 = y_offsets[row_index]
            draw.rectangle(
                (x0, y0, x0 + cell_width - 1, y0 + row_height - 1),
                outline=(205, 205, 205),
                width=1,
            )
            draw.text(
                (x0 + 5, y0 + 5),
                item["label"],
                fill="black",
                font=_font(14 if item["span"] == 1 else 15),
            )
            tile = item["image"].copy()
            tile.thumbnail(
                (cell_width - 10, row_height - 42),
                Image.Resampling.LANCZOS,
            )
            panel.paste(
                tile,
                (
                    x0 + (cell_width - tile.width) // 2,
                    y0 + 36 + (row_height - 38 - tile.height) // 2,
                ),
            )
            slot_index += int(item["span"])

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.save(target, quality=88)
    return {
        "schema_version": "scenemosaic_combined_pass_b_review_panel",
        "panel_path": str(target),
        "panel_size": list(panel.size),
        "row_count": len(rows),
        "local_candidate_count": len(candidates),
        "content_host_count": len(hosts),
        "tile_priority": "local_candidates_then_full_width_open_storage",
        "row_heights": row_heights,
        "open_storage_full_width_count": sum(
            item.get("kind") == "host"
            and item.get("host_mode") == "open_storage"
            and int(item.get("span") or 0) == 5
            for row in rows
            for item in row
        ),
        "request_additional_image_count": 1,
    }


__all__ = [
    "ContentHostConfig",
    "materialize_combined_review_panel",
    "materialize_content_host_review",
    "select_content_hosts",
]
