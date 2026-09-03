#!/usr/bin/env python3
"""Resolve room surface descriptions to local PBR material directories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml

SUMMARY_PREFIX = "ROOM_MATERIAL_RETRIEVAL_SUMMARY:"
SURFACE_KEYS = ("walls", "floor", "ceiling")
REQUIRED_CHANNELS = ("Color", "NormalGL", "Roughness")
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

_cached_model = None
_cached_tokenizer = None
_cached_device = None


def print_summary(payload: dict) -> None:
    print(f"{SUMMARY_PREFIX} {json.dumps(payload, ensure_ascii=False)}")


def load_room_material_queries(objects_json: Path) -> dict[str, str] | None:
    with objects_json.open("r", encoding="utf-8") as f:
        objects_data = json.load(f)

    room_materials = objects_data.get("room_materials")
    if room_materials is None:
        return None
    if not isinstance(room_materials, dict):
        raise ValueError("room_materials must be an object.")

    queries = {}
    for key in SURFACE_KEYS:
        query = room_materials.get(key)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"room_materials.{key} must be a non-empty string.")
        queries[key] = query.strip()
    return queries


def load_material_index(embeddings_dir: Path) -> tuple[np.ndarray, list[str]]:
    embeddings_path = embeddings_dir / "clip_embeddings.npy"
    index_path = embeddings_dir / "embedding_index.yaml"
    embeddings = np.load(embeddings_path)
    with index_path.open("r", encoding="utf-8") as f:
        material_ids = yaml.safe_load(f)

    if not isinstance(material_ids, list) or not all(isinstance(item, str) for item in material_ids):
        raise ValueError("embedding_index.yaml must contain a list of material IDs.")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(material_ids):
        raise ValueError("Embedding rows do not match the number of material IDs.")
    if not len(material_ids):
        raise ValueError("Material embedding index is empty.")
    return embeddings.astype(np.float32, copy=False), material_ids


def get_clip_model(device: str | None) -> tuple[object, object, str]:
    global _cached_model, _cached_tokenizer, _cached_device

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _cached_model is None or _cached_device != target_device:
        cache_dir = Path(__file__).resolve().parents[1] / "cache" / "open_clip"
        _cached_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14-378-quickgelu",
            pretrained="dfn5b",
            device=target_device,
            cache_dir=cache_dir,
        )
        _cached_model.eval()
        _cached_tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
        _cached_device = target_device
    return _cached_model, _cached_tokenizer, target_device


def retrieve_material_id(query: str, embeddings: np.ndarray, material_ids: list[str], device: str | None) -> str:
    model, tokenizer, target_device = get_clip_model(device)
    text_tokens = tokenizer([query]).to(target_device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_embedding = text_features.detach().cpu().numpy()[0].astype(np.float32, copy=False)

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm <= 0 or np.any(embedding_norms <= 0):
        raise ValueError("Encountered a zero-norm CLIP embedding.")
    similarities = (embeddings / embedding_norms[:, None]) @ (query_embedding / query_norm)
    return material_ids[int(np.argmax(similarities))]


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


def validate_material_dir(
    materials_dir: Path,
    material_id: str,
    *,
    resolution: str | None = None,
    file_format: str | None = None,
) -> Path:
    material_dir = materials_dir / material_id
    required_paths = [
        texture_path(
            material_dir, channel, resolution=resolution, file_format=file_format
        )
        for channel in REQUIRED_CHANNELS
    ]
    if not material_dir.is_dir():
        raise FileNotFoundError(f"Material directory not found: {material_dir}")
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Material textures not found: {', '.join(missing_paths)}")
    return material_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve PBR materials for room surfaces.")
    parser.add_argument("--objects_json", type=Path, required=True)
    parser.add_argument("--materials_dir", type=Path, required=True)
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default=env_texture_resolution(),
        choices=list(TEXTURE_RESOLUTIONS),
        help=(
            "Material texture resolution (default: "
            f"$SCENEMOSAIC_MATERIAL_RESOLUTION or {DEFAULT_TEXTURE_RESOLUTION})"
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default=env_texture_format(),
        choices=list(TEXTURE_FORMATS),
        help=(
            "Material texture file format (default: "
            f"$SCENEMOSAIC_MATERIAL_FORMAT or {DEFAULT_TEXTURE_FORMAT})"
        ),
    )
    args = parser.parse_args()

    try:
        queries = load_room_material_queries(args.objects_json)
        if queries is None:
            print_summary({"status": "legacy", "reason": "room_materials_absent"})
            return 0

        embeddings, material_ids = load_material_index(args.embeddings_dir)
        selected_ids = {
            surface: retrieve_material_id(query, embeddings, material_ids, args.device)
            for surface, query in queries.items()
        }
        selected_dirs = {
            surface: validate_material_dir(
                args.materials_dir,
                material_id,
                resolution=args.resolution,
                file_format=args.format,
            )
            for surface, material_id in selected_ids.items()
        }
        print_summary(
            {
                "status": "resolved",
                "wall_material_dir": str(selected_dirs["walls"]),
                "floor_material_dir": str(selected_dirs["floor"]),
                "ceiling_material_dir": str(selected_dirs["ceiling"]),
                "material_ids": selected_ids,
            }
        )
        return 0
    except Exception as exc:
        message = str(exc)
        print(f"Room material retrieval failed: {message}", file=sys.stderr)
        print_summary({"status": "failed", "reason": type(exc).__name__, "error": message})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
