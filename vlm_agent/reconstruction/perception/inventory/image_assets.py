"""Small, reproducible image assets used only for VLM API requests.

The reconstruction pipeline keeps its original images and PNG masks unchanged.
These helpers create smaller JPEG derivatives so the same full-resolution image
is not uploaded repeatedly during object review.
"""

import base64
from pathlib import Path


def create_vlm_jpeg(
    source_path,
    output_path,
    max_long_edge=1280,
    quality=85,
    coordinate_grid=False,
    grid_divisions=10,
):
    """Create a bounded RGB JPEG, optionally with a lightweight coordinate ruler."""
    from PIL import Image

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_long_edge:
            scale = max_long_edge / longest
            image = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.LANCZOS,
            )
        if coordinate_grid:
            from PIL import ImageDraw, ImageFont

            if grid_divisions < 2:
                raise ValueError("grid_divisions must be at least 2")
            width_out, height_out = image.size
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", max(12, width_out // 100))
            except OSError:
                font = ImageFont.load_default()
            for index in range(grid_divisions + 1):
                x = round(index * (width_out - 1) / grid_divisions)
                y = round(index * (height_out - 1) / grid_divisions)
                alpha = 100 if index % 2 == 0 else 55
                draw.line((x, 0, x, height_out - 1), fill=(0, 205, 255, alpha), width=1)
                draw.line((0, y, width_out - 1, y), fill=(0, 205, 255, alpha), width=1)
                if index < grid_divisions:
                    label = f"{index / grid_divisions:.1f}"
                    top_box = draw.textbbox((x + 3, 1), label, font=font)
                    left_box = draw.textbbox((3, y + 1), label, font=font)
                    draw.rectangle(top_box, fill=(0, 0, 0, 145))
                    draw.rectangle(left_box, fill=(0, 0, 0, 145))
                    draw.text((x + 3, 1), label, fill=(255, 255, 255, 235), font=font)
                    draw.text((3, y + 1), label, fill=(255, 255, 255, 235), font=font)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        image.save(output_path, format="JPEG", quality=quality, optimize=True)

    return {
        "path": str(output_path),
        "mime_type": "image/jpeg",
        "source_path": str(source_path),
        "source_bytes": source_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "source_size": [width, height],
        "output_size": list(image.size),
        "max_long_edge": max_long_edge,
        "quality": quality,
        "coordinate_grid": {
            "enabled": bool(coordinate_grid),
            "divisions": int(grid_divisions) if coordinate_grid else None,
        },
    }


def encode_data_url(path, mime_type="image/jpeg"):
    with open(path, "rb") as handle:
        image_b64 = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"
