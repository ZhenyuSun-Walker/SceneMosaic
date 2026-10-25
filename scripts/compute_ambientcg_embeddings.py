#!/usr/bin/env python3
"""Compute CLIP image embeddings for locally downloaded ambientCG materials."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Hugging Face reads this setting during import; preserve an explicit override.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATERIALS_DIR = PROJECT_ROOT / "data" / "materials"
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "vlm_agent" / "evolution" / "cache" / "open_clip"
MODEL_NAME = "ViT-H-14-378-quickgelu"
PRETRAINED = "dfn5b"
EXPECTED_DIMENSION = 1024
DEFAULT_TEXTURE_RESOLUTION = "2K"
DEFAULT_TEXTURE_FORMAT = "PNG"
TEXTURE_RESOLUTIONS = ("1K", "2K", "4K", "8K", "12K", "16K")
TEXTURE_FORMATS = ("PNG", "JPG")


def env_texture_resolution() -> str:
    """Resolve the texture resolution from the environment (default 2K)."""
    value = os.environ.get("SCENEMOSAIC_MATERIAL_RESOLUTION")
    return value if value in TEXTURE_RESOLUTIONS else DEFAULT_TEXTURE_RESOLUTION


def env_texture_format() -> str:
    """Resolve the texture format from the environment (default PNG)."""
    value = os.environ.get("SCENEMOSAIC_MATERIAL_FORMAT")
    return value if value in TEXTURE_FORMATS else DEFAULT_TEXTURE_FORMAT


def texture_path(
    material_dir: Path,
    channel: str,
    *,
    resolution: str | None = None,
    file_format: str | None = None,
) -> Path:
    """Return the path of one ambientCG texture of ``material_dir``.

    ambientCG names textures ``<material>_<res>-<fmt>_<channel>.<ext>`` where
    the filename label uses the format in uppercase (e.g. ``2K-PNG_Color.png``
    for PNG, ``2K-JPG_Color.jpg`` for JPG). ``channel`` is the map suffix such
    as ``Color``, ``NormalGL`` or ``Roughness``. Unset arguments fall back to
    the environment overrides, then to 2K PNG.
    """
    resolution = resolution or env_texture_resolution()
    file_format = file_format or env_texture_format()
    extension = file_format.lower()
    return material_dir / (
        f"{material_dir.name}_{resolution}-{file_format}_{channel}.{extension}"
    )


def discover_materials(
    materials_dir: Path,
    *,
    resolution: str | None = None,
    file_format: str | None = None,
) -> list[tuple[str, Path]]:
    """Return deterministic material IDs and their base-color textures."""

    resolution = resolution or env_texture_resolution()
    file_format = file_format or env_texture_format()
    discovered: list[tuple[str, Path]] = []
    if not materials_dir.is_dir():
        raise FileNotFoundError(f"Material directory not found: {materials_dir}")

    for material_dir in sorted(materials_dir.iterdir(), key=lambda path: path.name):
        if not material_dir.is_dir() or material_dir.is_symlink():
            continue
        color_path = texture_path(
            material_dir, "Color", resolution=resolution, file_format=file_format
        )
        if color_path.is_file() and not color_path.is_symlink():
            discovered.append((material_dir.name, color_path))

    if not discovered:
        raise FileNotFoundError(
            f"No ambientCG {resolution}-{file_format} Color textures found below {materials_dir}"
        )
    return discovered


def encode_materials(
    materials: list[tuple[str, Path]],
    *,
    device: str,
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    """Encode base-color textures with the retrieval model used at runtime."""

    import open_clip
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
        cache_dir=str(cache_dir),
    )
    model.eval()

    encoded_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(materials), batch_size),
            desc="Computing ambientCG embeddings",
        ):
            batch_items = materials[start : start + batch_size]
            tensors = []
            for _, color_path in batch_items:
                with Image.open(color_path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            image_batch = torch.stack(tensors).to(device)
            features = model.encode_image(image_batch)
            features = features / features.norm(dim=-1, keepdim=True)
            encoded_batches.append(
                features.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    embeddings = np.concatenate(encoded_batches, axis=0)
    if embeddings.shape != (len(materials), EXPECTED_DIMENSION):
        raise ValueError(
            "Unexpected embedding shape: "
            f"received {embeddings.shape}, expected "
            f"({len(materials)}, {EXPECTED_DIMENSION})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Computed embeddings contain non-finite values")
    return embeddings


def write_outputs(
    output_dir: Path,
    material_ids: list[str],
    embeddings: np.ndarray,
    *,
    overwrite: bool,
) -> None:
    """Write the matrix and row index without exposing partial final files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "clip_embeddings.npy"
    index_path = output_dir / "embedding_index.yaml"
    existing = [path for path in (embeddings_path, index_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}; pass --overwrite")

    embeddings_tmp = output_dir / ".clip_embeddings.npy.tmp"
    index_tmp = output_dir / ".embedding_index.yaml.tmp"
    with embeddings_tmp.open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
    index_tmp.write_text(
        yaml.safe_dump(material_ids, sort_keys=False),
        encoding="utf-8",
    )
    embeddings_tmp.replace(embeddings_path)
    index_tmp.replace(index_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute CLIP embeddings for downloaded ambientCG materials"
    )
    parser.add_argument(
        "--materials-dir",
        type=Path,
        default=DEFAULT_MATERIALS_DIR,
        help="ambientCG material directory (default: <project>/data/materials)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="embedding output directory (default: <project>/data/embeddings)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "OpenCLIP model cache "
            "(default: <project>/vlm_agent/evolution/cache/open_clip)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device; defaults to CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default=env_texture_resolution(),
        choices=list(TEXTURE_RESOLUTIONS),
        help=(
            "Texture resolution (default: $SCENEMOSAIC_MATERIAL_RESOLUTION "
            f"or {DEFAULT_TEXTURE_RESOLUTION})"
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default=env_texture_format(),
        choices=list(TEXTURE_FORMATS),
        help=(
            "Texture file format (default: $SCENEMOSAIC_MATERIAL_FORMAT "
            f"or {DEFAULT_TEXTURE_FORMAT})"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    materials_dir = args.materials_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    materials = discover_materials(
        materials_dir,
        resolution=args.resolution,
        file_format=args.format,
    )
    embeddings = encode_materials(
        materials,
        device=device,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
    )
    write_outputs(
        output_dir,
        [material_id for material_id, _ in materials],
        embeddings,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {len(materials)} {EXPECTED_DIMENSION}D embeddings to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
