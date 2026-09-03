"""Human-review assets for normalized object inventories."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _safe_asset_component(value: Any) -> str:
    value = str(value or "object").strip().lower()
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in value)
    return cleaned.strip("_") or "object"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_bbox_review_assets(
    image_path: str | Path,
    objects: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render one context crop per VLM bbox and a compact contact sheet."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_w, image_h = image.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    records = []
    thumbnails = []
    for obj in objects:
        object_id = str(obj["object_id"])
        category = _safe_asset_component(obj.get("category"))
        x1, y1, x2, y2 = obj["bbox_2d_norm_xyxy"]
        bx1, by1, bx2, by2 = (
            round(x1 * image_w), round(y1 * image_h),
            round(x2 * image_w), round(y2 * image_h),
        )
        box_w = max(1, bx2 - bx1)
        box_h = max(1, by2 - by1)
        pad = max(16, round(max(box_w, box_h) * 0.35))
        crop_x1 = max(0, bx1 - pad)
        crop_y1 = max(0, by1 - pad)
        crop_x2 = min(image_w, bx2 + pad)
        crop_y2 = min(image_h, by2 + pad)
        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).copy()
        draw = ImageDraw.Draw(crop, "RGBA")
        local_bbox = (bx1 - crop_x1, by1 - crop_y1, bx2 - crop_x1, by2 - crop_y1)
        line_width = max(2, round(max(crop.size) / 220))
        draw.rectangle(local_bbox, outline=(255, 40, 40, 255), width=line_width)
        anchor = obj.get("anchor_2d_norm_xy")
        if anchor:
            ax = round(anchor[0] * image_w) - crop_x1
            ay = round(anchor[1] * image_h) - crop_y1
            radius = max(4, line_width * 2)
            draw.ellipse(
                (ax - radius, ay - radius, ax + radius, ay + radius),
                fill=(255, 40, 40, 255), outline=(255, 255, 255, 255), width=1,
            )
        label = f"{object_id} | {category}"
        label_box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle(
            (0, 0, label_box[2] + 12, label_box[3] + 10),
            fill=(0, 0, 0, 185),
        )
        draw.text((6, 4), label, font=font, fill=(255, 255, 255, 255))
        crop.thumbnail((640, 640), Image.Resampling.LANCZOS)
        filename = f"{object_id}_{category}_bbox_review.jpg"
        asset_path = output_dir / filename
        crop.save(asset_path, format="JPEG", quality=88, optimize=True)
        records.append(
            {
                "object_id": object_id,
                "category": obj.get("category"),
                "bbox_2d_norm_xyxy": obj["bbox_2d_norm_xyxy"],
                "anchor_2d_norm_xy": anchor,
                "crop_xyxy_px": [crop_x1, crop_y1, crop_x2, crop_y2],
                "image": str(asset_path),
            }
        )
        thumbnail = crop.copy()
        thumbnail.thumbnail((300, 230), Image.Resampling.LANCZOS)
        thumbnails.append((label, thumbnail))

    cols = 4
    cell_w, cell_h, label_h = 320, 270, 28
    rows = max(1, math.ceil(len(thumbnails) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (28, 28, 28))
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (label, thumbnail) in enumerate(thumbnails):
        col, row = index % cols, index // cols
        left, top = col * cell_w, row * cell_h
        thumb_x = left + (cell_w - thumbnail.width) // 2
        thumb_y = top + label_h + (cell_h - label_h - thumbnail.height) // 2
        sheet.paste(thumbnail, (thumb_x, thumb_y))
        sheet_draw.text((left + 6, top + 5), label, font=small_font, fill=(255, 255, 255))
    contact_sheet = output_dir / "bbox_review_contact_sheet.jpg"
    sheet.save(contact_sheet, format="JPEG", quality=86, optimize=True)
    manifest_path = output_dir / "bbox_review_manifest.json"
    _write_json(
        manifest_path,
        {
            "source_image": str(image_path),
            "source_size": [image_w, image_h],
            "contact_sheet": str(contact_sheet),
            "objects": records,
        },
    )
    return {
        "directory": str(output_dir),
        "manifest_json": str(manifest_path),
        "contact_sheet": str(contact_sheet),
        "object_count": len(records),
    }
