from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import trimesh

from ..config import RuntimeConfig
from ..run_paths import ScenePaths
from .sam3d_runner import (
    SEED,
    Sam3DInputRecord,
    Sam3DRunnerError,
    _checkpoint_asset_paths,
    _load_records,
    _load_scene_contract,
    _resolve_dino_cache,
    dry_run_payload,
    load_job,
)

ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SECRET_NAME_RE = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


class Sam3DStageError(RuntimeError):
    """Raised when Stage 05 cannot be safely run, reused, or published."""


@dataclass(frozen=True)
class Sam3DStageArtifacts:
    stage05_dir: Path
    object_index_map_path: Path
    run_info_path: Path
    scene_glb_path: Path
    layout_params_dir: Path
    posed_objects_dir: Path
    unposed_objects_dir: Path
    objects_dir: Path
    object_ids: tuple[str, ...]
    source_manifest_sha256: str
    artifact_hashes: dict[str, str]
    attempt_id: str | None = None
    audit_path: Path | None = None
    reused: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sam3DStageError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise Sam3DStageError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _real_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise Sam3DStageError(f"{label} must be a real non-empty file: {path}")


def _flatten_numbers(value: Any, *, label: str) -> list[float]:
    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            if not item:
                raise Sam3DStageError(f"{label} contains an empty list")
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Sam3DStageError(f"{label} must contain only finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise Sam3DStageError(f"{label} must contain only finite numbers")
        flattened.append(number)

    if not isinstance(value, list) or not value:
        raise Sam3DStageError(f"{label} must be a non-empty list")
    visit(value)
    return flattened


def _validate_glb(path: Path, label: str) -> str:
    _real_nonempty_file(path, label)
    if path.stat().st_size < 12:
        raise Sam3DStageError(f"{label} is shorter than a GLB header: {path}")
    with path.open("rb") as handle:
        header = handle.read(12)
    magic, version, declared_length = struct.unpack("<4sII", header)
    if magic != b"glTF" or version != 2 or declared_length != path.stat().st_size:
        raise Sam3DStageError(f"{label} has an invalid GLB 2.0 header: {path}")
    return _sha256(path)


def _directory_files(directory: Path, suffix: str, label: str) -> dict[str, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise Sam3DStageError(f"{label} must be a real directory: {directory}")
    result: dict[str, Path] = {}
    for item in directory.iterdir():
        if item.is_symlink() or not item.is_file() or item.suffix != suffix:
            raise Sam3DStageError(f"unexpected entry in {label}: {item}")
        if item.stem in result:
            raise Sam3DStageError(f"duplicate object artifact in {label}: {item.stem}")
        result[item.stem] = item
    return result


def _validate_relative_symlink(link: Path, expected_target: Path, label: str) -> None:
    if not link.is_symlink():
        raise Sam3DStageError(f"{label} must be a relative symlink: {link}")
    raw_target = Path(os.readlink(link))
    if raw_target.is_absolute():
        raise Sam3DStageError(f"{label} must not use an absolute symlink: {link}")
    actual = (link.parent / raw_target).resolve()
    if actual != expected_target.resolve() or not actual.is_file():
        raise Sam3DStageError(f"{label} target mismatch: {link} -> {raw_target}")

def _validate_asset_records(
    records: Any,
    expected_paths: tuple[Path, ...],
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise Sam3DStageError(f"{label} asset closure mismatch")
    hashes: dict[str, str] = {}
    for record, expected_path in zip(records, expected_paths):
        if not isinstance(record, dict):
            raise Sam3DStageError(f"{label} asset record is invalid")
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if path != expected_path.resolve():
            raise Sam3DStageError(f"{label} asset path mismatch: {path}")
        _real_nonempty_file(path, f"{label} asset")
        size = path.stat().st_size
        digest = _sha256(path)
        if record.get("size_bytes") != size or record.get("sha256") != digest:
            raise Sam3DStageError(f"{label} asset hash mismatch: {path}")
        hashes[str(path)] = digest
    return hashes


def _stage04_records(
    scene_config_path: Path,
) -> tuple[str, Path, Path, tuple[Sam3DInputRecord, ...]]:
    try:
        scene_name, run_root, input_image, manifest_path = _load_scene_contract(
            Path(scene_config_path)
        )
        records = _load_records(
            manifest_path,
            scene_name=scene_name,
            input_image=input_image,
        )
    except Sam3DRunnerError as exc:
        raise Sam3DStageError(str(exc)) from exc
    return scene_name, run_root, manifest_path, records


def validate_stage05(
    scene_config_path: Path,
    stage05_dir: Path | None = None,
    *,
    published_output_dir: Path | None = None,
    project_root: Path | None = None,
) -> Sam3DStageArtifacts:
    """Strictly validate a formal Stage 05 or an unpublished attempt.

    ``published_output_dir`` is only different from ``stage05_dir`` while an
    attempt is being checked before its atomic promotion. Paths recorded in
    Stage 05 always point at the eventual formal publication.
    """

    scene_name, run_root, manifest_path, records = _stage04_records(
        Path(scene_config_path)
    )
    paths = ScenePaths.from_scene_config(Path(scene_config_path))
    root = Path(stage05_dir or paths.sam3d_output_dir).expanduser().resolve()
    published = Path(
        published_output_dir or paths.sam3d_output_dir
    ).expanduser().resolve()
    expected_published = paths.sam3d_output_dir
    if published != expected_published:
        raise Sam3DStageError(
            f"published Stage 05 path mismatch: expected {expected_published}, got {published}"
        )
    if not root.is_dir() or root.is_symlink():
        raise Sam3DStageError(f"Stage 05 must be a real directory: {root}")

    index_path = root / "object_index_map.json"
    run_info_path = root / "sam3d_run_info.json"
    scene_path = root / "scene.glb"
    _real_nonempty_file(index_path, "Stage 05 object index")
    _real_nonempty_file(run_info_path, "Stage 05 run info")
    scene_sha = _validate_glb(scene_path, "Stage 05 scene")

    index = _read_json(index_path, "Stage 05 object index")
    run_info = _read_json(run_info_path, "Stage 05 run info")
    manifest_sha = _sha256(manifest_path)
    expected_ids = tuple(record.object_id for record in records)
    invariant_pairs = (
        (index.get("stage"), "sam3d_output", "index.stage"),
        (run_info.get("stage"), "sam3d_output", "run_info.stage"),
        (index.get("scene_name"), scene_name, "index.scene_name"),
        (run_info.get("scene_name"), scene_name, "run_info.scene_name"),
        (index.get("source_manifest"), str(manifest_path), "index.source_manifest"),
        (run_info.get("source_manifest"), str(manifest_path), "run_info.source_manifest"),
        (index.get("source_manifest_sha256"), manifest_sha, "index source hash"),
        (run_info.get("source_manifest_sha256"), manifest_sha, "run_info source hash"),
        (index.get("output_dir"), str(published), "index.output_dir"),
        (run_info.get("output_dir"), str(published), "run_info.output_dir"),
        (index.get("seed"), SEED, "index.seed"),
        (run_info.get("seed"), SEED, "run_info.seed"),
        (index.get("appearance_mode"), "vertex_color", "index appearance"),
        (run_info.get("appearance_mode"), "vertex_color", "run_info appearance"),
    )
    for actual, expected, label in invariant_pairs:
        if actual != expected:
            raise Sam3DStageError(
                f"Stage 05 {label} mismatch: expected {expected!r}, got {actual!r}"
            )
    if run_info.get("status") != "success":
        raise Sam3DStageError(f"Stage 05 is not successful: {run_info.get('status')}")
    for payload, label in ((index, "index"), (run_info, "run_info")):
        if (
            payload.get("with_texture_baking") is not False
            or payload.get("texture_verified") is not False
            or payload.get("vertex_color_verified") is not True
        ):
            raise Sam3DStageError(
                f"Stage 05 {label} does not prove vertex-color/no-texture mode"
            )
    if run_info.get("object_count") != len(expected_ids):
        raise Sam3DStageError("Stage 05 run_info object_count mismatch")
    if run_info.get("object_index_map") != str(published / index_path.name):
        raise Sam3DStageError("Stage 05 run_info object_index_map path mismatch")
    if run_info.get("object_index_map_sha256") != _sha256(index_path):
        raise Sam3DStageError("Stage 05 object index hash mismatch")
    if index.get("scene_glb_sha256") != scene_sha:
        raise Sam3DStageError("Stage 05 scene GLB hash mismatch")

    local_project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    source = index.get("sam3d_source")
    if not isinstance(source, dict):
        raise Sam3DStageError("Stage 05 has no SAM3D source provenance")
    expected_source_root = local_project / "sam-3d-objects"
    expected_inference = expected_source_root / "notebook" / "inference.py"
    if (
        source.get("root") != str(expected_source_root)
        or source.get("inference_path") != str(expected_inference)
        or not expected_inference.is_file()
    ):
        raise Sam3DStageError(
            "Stage 05 was not produced by the current repository-local sam-3d-objects"
        )
    checkpoint = index.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise Sam3DStageError("Stage 05 has no checkpoint provenance")
    checkpoint_path = Path(
        str(checkpoint.get("path") or "")
    ).expanduser().resolve()
    _real_nonempty_file(checkpoint_path, "SAM3D checkpoint configuration")
    if checkpoint.get("sha256") != _sha256(checkpoint_path):
        raise Sam3DStageError("Stage 05 checkpoint hash mismatch")
    try:
        checkpoint_assets = _checkpoint_asset_paths(checkpoint_path)
    except Sam3DRunnerError as exc:
        raise Sam3DStageError(str(exc)) from exc
    _validate_asset_records(
        checkpoint.get("assets"),
        checkpoint_assets,
        label="SAM3D checkpoint",
    )

    dino = index.get("dino")
    if not isinstance(dino, dict) or dino.get("offline") is not True:
        raise Sam3DStageError("Stage 05 does not prove offline DINO loading")
    raw_hub_dir = str(dino.get("hub_dir") or "")
    if not raw_hub_dir:
        raise Sam3DStageError("Stage 05 has no DINO hub directory")
    try:
        dino_hub, dino_repository, dino_checkpoints = _resolve_dino_cache(
            Path(raw_hub_dir)
        )
    except Sam3DRunnerError as exc:
        raise Sam3DStageError(str(exc)) from exc
    if (
        dino.get("hub_dir") != str(dino_hub)
        or dino.get("repository") != str(dino_repository)
    ):
        raise Sam3DStageError("Stage 05 DINO path provenance mismatch")
    _validate_asset_records(
        dino.get("checkpoints"),
        dino_checkpoints,
        label="DINO checkpoint",
    )

    index_objects = index.get("objects")
    if not isinstance(index_objects, list) or len(index_objects) != len(records):
        raise Sam3DStageError("Stage 05 object index count mismatch")
    index_ids = tuple(
        str(item.get("object_id") or "")
        for item in index_objects
        if isinstance(item, dict)
    )
    if index_ids != expected_ids or len(index_ids) != len(index_objects):
        raise Sam3DStageError("Stage 04 and Stage 05 object ID closure failed")

    layout_dir = root / "layout_params"
    posed_dir = root / "posed_objects"
    unposed_dir = root / "unposed_objects"
    objects_dir = root / "objects"
    layout_files = _directory_files(layout_dir, ".json", "layout_params")
    posed_files = _directory_files(posed_dir, ".glb", "posed_objects")
    unposed_files = _directory_files(unposed_dir, ".glb", "unposed_objects")
    for label, actual in (
        ("layout_params", tuple(sorted(layout_files))),
        ("posed_objects", tuple(sorted(posed_files))),
        ("unposed_objects", tuple(sorted(unposed_files))),
    ):
        if actual != expected_ids:
            raise Sam3DStageError(
                f"Stage 05 {label} ID closure mismatch: {actual} != {expected_ids}"
            )
    if not objects_dir.is_dir() or objects_dir.is_symlink():
        raise Sam3DStageError(f"Stage 05 objects must be a real directory: {objects_dir}")
    object_entries = tuple(sorted(item.name for item in objects_dir.iterdir()))
    if object_entries != expected_ids:
        raise Sam3DStageError("Stage 05 object compatibility directory closure failed")

    artifact_hashes: dict[str, str] = {
        "object_index_map.json": _sha256(index_path),
        "sam3d_run_info.json": _sha256(run_info_path),
        "scene.glb": scene_sha,
    }
    for record, entry in zip(records, index_objects):
        if not isinstance(entry, dict):
            raise Sam3DStageError(f"Stage 05 index record is invalid for {record.object_id}")
        object_id = record.object_id
        identity_checks = (
            (entry.get("order_index"), record.order_index, "order_index"),
            (entry.get("sam3d_index"), record.order_index, "sam3d_index"),
            (entry.get("sam3d_internal_name"), object_id, "sam3d_internal_name"),
            (entry.get("mask_path"), str(record.mask_path), "mask_path"),
            (entry.get("mask_sha256"), record.mask_sha256, "mask_sha256"),
        )
        for actual, expected, label in identity_checks:
            if actual != expected:
                raise Sam3DStageError(
                    f"Stage 05 {label} mismatch for {object_id}: {actual!r}"
                )

        posed_path = posed_files[object_id]
        unposed_path = unposed_files[object_id]
        layout_path = layout_files[object_id]
        posed_sha = _validate_glb(posed_path, f"posed GLB for {object_id}")
        unposed_sha = _validate_glb(unposed_path, f"unposed GLB for {object_id}")
        layout = _read_json(layout_path, f"layout for {object_id}")
        if (
            layout.get("object_id") != object_id
            or layout.get("sam3d_internal_name") != object_id
            or layout.get("sam3d_index") != record.order_index
            or layout.get("source_mask_path") != str(record.mask_path)
            or layout.get("source_mask_sha256") != record.mask_sha256
        ):
            raise Sam3DStageError(f"Stage 05 layout identity mismatch for {object_id}")
        expected_lengths = {"translation": 3, "rotation": 4, "scale": 3}
        for key, length in expected_lengths.items():
            values = _flatten_numbers(layout.get(key), label=f"{object_id}.{key}")
            if len(values) != length:
                raise Sam3DStageError(
                    f"{object_id}.{key} must contain exactly {length} numbers"
                )
        expected_relative = {
            "posed_glb": os.path.join("posed_objects", f"{object_id}.glb"),
            "unposed_glb": os.path.join("unposed_objects", f"{object_id}.glb"),
            "layout_json": os.path.join("layout_params", f"{object_id}.json"),
        }
        for key, expected in expected_relative.items():
            if layout.get(key) != expected:
                raise Sam3DStageError(f"Stage 05 layout path mismatch: {object_id}.{key}")

        expected_hashes = {
            "posed_glb_sha256": posed_sha,
            "unposed_glb_sha256": unposed_sha,
            "layout_sha256": _sha256(layout_path),
        }
        for key, expected in expected_hashes.items():
            if entry.get(key) != expected:
                raise Sam3DStageError(f"Stage 05 {key} mismatch for {object_id}")

        final_object_dir = published / "objects" / object_id
        expected_paths = {
            "output_dir": final_object_dir,
            "local_mesh_path": final_object_dir / "local_mesh.glb",
            "camera_mesh_path": final_object_dir / "camera_mesh.glb",
            "full_json_path": final_object_dir / f"{object_id}_full.json",
        }
        for key, expected in expected_paths.items():
            if entry.get(key) != str(expected):
                raise Sam3DStageError(f"Stage 05 {key} mismatch for {object_id}")

        object_dir = objects_dir / object_id
        if not object_dir.is_dir() or object_dir.is_symlink():
            raise Sam3DStageError(f"invalid compatibility directory for {object_id}")
        expected_link_names = {
            "local_mesh.glb",
            "camera_mesh.glb",
            f"{object_id}_full.json",
        }
        if {path.name for path in object_dir.iterdir()} != expected_link_names:
            raise Sam3DStageError(f"compatibility link closure failed for {object_id}")
        _validate_relative_symlink(
            object_dir / "local_mesh.glb", unposed_path, f"{object_id} local mesh"
        )
        _validate_relative_symlink(
            object_dir / "camera_mesh.glb", posed_path, f"{object_id} camera mesh"
        )
        _validate_relative_symlink(
            object_dir / f"{object_id}_full.json",
            layout_path,
            f"{object_id} full layout",
        )
        artifact_hashes[f"posed_objects/{object_id}.glb"] = posed_sha
        artifact_hashes[f"unposed_objects/{object_id}.glb"] = unposed_sha
        artifact_hashes[f"layout_params/{object_id}.json"] = _sha256(layout_path)

    audit_path = root / "sam3d_stage_audit.json"
    attempt_id = None
    if audit_path.is_file() and not audit_path.is_symlink():
        audit = _read_json(audit_path, "SAM3D stage audit")
        attempt_id = str(audit.get("attempt_id") or "") or None
        artifact_hashes[audit_path.name] = _sha256(audit_path)

    return Sam3DStageArtifacts(
        stage05_dir=root,
        object_index_map_path=index_path,
        run_info_path=run_info_path,
        scene_glb_path=scene_path,
        layout_params_dir=layout_dir,
        posed_objects_dir=posed_dir,
        unposed_objects_dir=unposed_dir,
        objects_dir=objects_dir,
        object_ids=expected_ids,
        source_manifest_sha256=manifest_sha,
        artifact_hashes=artifact_hashes,
        attempt_id=attempt_id,
        audit_path=audit_path if audit_path.is_file() else None,
    )


def _equivalence_mesh(
    path: Path,
    label: str,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    _validate_glb(path, label)
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise Sam3DStageError(f"cannot parse {label}: {path}") from exc
    if not isinstance(mesh, trimesh.Trimesh):
        raise Sam3DStageError(f"{label} must contain one mesh: {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = np.asarray(
        getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    )
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or vertices.shape[0] == 0
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or faces.shape[0] == 0
        or not np.isfinite(vertices).all()
        or getattr(mesh.visual, "kind", None) != "vertex"
        or colors.ndim != 2
        or colors.shape[0] != vertices.shape[0]
        or colors.shape[1] not in {3, 4}
    ):
        raise Sam3DStageError(
            f"{label} must contain finite triangle geometry and vertex colors"
        )
    return mesh, vertices, faces, colors


def _compare_equivalence_meshes(
    reference_path: Path,
    candidate_path: Path,
    *,
    label: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference, reference_vertices, reference_faces, reference_colors = (
        _equivalence_mesh(reference_path, f"reference {label}")
    )
    candidate, candidate_vertices, candidate_faces, candidate_colors = (
        _equivalence_mesh(candidate_path, f"candidate {label}")
    )
    if (
        reference_vertices.shape != candidate_vertices.shape
        or reference_faces.shape != candidate_faces.shape
        or not np.array_equal(reference_faces, candidate_faces)
    ):
        raise Sam3DStageError(f"{label} mesh topology mismatch")
    if not np.allclose(
        reference_vertices,
        candidate_vertices,
        rtol=rtol,
        atol=atol,
    ):
        difference = float(
            np.max(np.abs(reference_vertices - candidate_vertices))
        )
        raise Sam3DStageError(
            f"{label} vertex mismatch; max_abs_difference={difference}"
        )
    if (
        reference_colors.shape != candidate_colors.shape
        or not np.array_equal(reference_colors, candidate_colors)
    ):
        raise Sam3DStageError(f"{label} vertex color mismatch")
    reference_bounds = np.asarray(reference.bounds, dtype=np.float64)
    candidate_bounds = np.asarray(candidate.bounds, dtype=np.float64)
    if not np.allclose(
        reference_bounds,
        candidate_bounds,
        rtol=rtol,
        atol=atol,
    ):
        raise Sam3DStageError(f"{label} bounding box mismatch")
    reference_sha256 = _sha256(reference_path)
    candidate_sha256 = _sha256(candidate_path)
    return {
        "vertex_count": int(reference_vertices.shape[0]),
        "face_count": int(reference_faces.shape[0]),
        "vertex_colors_exact": True,
        "bbox_close": True,
        "max_abs_vertex_difference": float(
            np.max(np.abs(reference_vertices - candidate_vertices))
        ),
        "reference_sha256": reference_sha256,
        "candidate_sha256": candidate_sha256,
        "glb_sha256_equal": reference_sha256 == candidate_sha256,
    }


def _equivalence_index_ids(stage_dir: Path, label: str) -> tuple[str, ...]:
    index = _read_json(
        stage_dir / "object_index_map.json",
        f"{label} object index",
    )
    entries = index.get("objects")
    if not isinstance(entries, list) or not entries:
        raise Sam3DStageError(f"{label} object index is empty or invalid")
    identifiers = tuple(
        str(item.get("object_id") or "")
        for item in entries
        if isinstance(item, dict)
    )
    if (
        len(identifiers) != len(entries)
        or any(not identifier for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise Sam3DStageError(f"{label} object index IDs are invalid")
    return identifiers


def compare_stage05_equivalence(
    scene_config_path: Path,
    *,
    reference_dir: Path,
    candidate_dir: Path,
    project_root: Path | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Gate a repository-local attempt against an existing formal Stage 05."""

    if (
        not math.isfinite(rtol)
        or not math.isfinite(atol)
        or rtol < 0
        or atol < 0
    ):
        raise Sam3DStageError("SAM3D equivalence tolerances must be finite")
    _, run_root, _, records = _stage04_records(Path(scene_config_path))
    reference = Path(reference_dir).expanduser().resolve()
    candidate = Path(candidate_dir).expanduser().resolve()
    paths = ScenePaths.from_scene_config(Path(scene_config_path))
    formal = paths.sam3d_output_dir.resolve()
    attempts = paths.sam3d_attempts_dir.resolve()
    if reference != formal:
        raise Sam3DStageError("equivalence reference must be formal Stage 05")
    if candidate.parent != attempts or candidate == reference:
        raise Sam3DStageError(
            "equivalence candidate must be one isolated Stage 05 attempt"
        )
    candidate_artifacts = validate_stage05(
        scene_config_path,
        candidate,
        published_output_dir=reference,
        project_root=project_root,
    )
    expected_ids = tuple(record.object_id for record in records)
    reference_ids = _equivalence_index_ids(reference, "reference")
    candidate_ids = _equivalence_index_ids(candidate, "candidate")
    if not (
        expected_ids
        == reference_ids
        == candidate_ids
        == candidate_artifacts.object_ids
    ):
        raise Sam3DStageError(
            "SAM3D equivalence object ID/order/count closure failed"
        )

    _validate_glb(reference / "scene.glb", "reference scene GLB")
    _validate_glb(candidate / "scene.glb", "candidate scene GLB")
    objects: dict[str, Any] = {}
    for object_id in expected_ids:
        reference_layout = _read_json(
            reference / "layout_params" / f"{object_id}.json",
            f"reference layout {object_id}",
        )
        candidate_layout = _read_json(
            candidate / "layout_params" / f"{object_id}.json",
            f"candidate layout {object_id}",
        )
        layout_differences: dict[str, float] = {}
        for key in ("translation", "rotation", "scale"):
            reference_values = np.asarray(
                _flatten_numbers(
                    reference_layout.get(key),
                    label=f"reference {object_id}.{key}",
                ),
                dtype=np.float64,
            )
            candidate_values = np.asarray(
                _flatten_numbers(
                    candidate_layout.get(key),
                    label=f"candidate {object_id}.{key}",
                ),
                dtype=np.float64,
            )
            if (
                reference_values.shape != candidate_values.shape
                or not np.allclose(
                    reference_values,
                    candidate_values,
                    rtol=rtol,
                    atol=atol,
                )
            ):
                raise Sam3DStageError(
                    f"{object_id}.{key} layout equivalence mismatch"
                )
            layout_differences[key] = float(
                np.max(np.abs(reference_values - candidate_values))
            )

        mesh_records = {
            kind: _compare_equivalence_meshes(
                reference / f"{kind}_objects" / f"{object_id}.glb",
                candidate / f"{kind}_objects" / f"{object_id}.glb",
                label=f"{kind} {object_id}",
                rtol=rtol,
                atol=atol,
            )
            for kind in ("posed", "unposed")
        }
        objects[object_id] = {
            "layout_close": True,
            "layout_max_abs_difference": layout_differences,
            "meshes": mesh_records,
        }

    reference_scene_sha256 = _sha256(reference / "scene.glb")
    candidate_scene_sha256 = _sha256(candidate / "scene.glb")
    return {
        "schema_version": "scenemosaic_sam3d_equivalence",
        "valid": True,
        "scene_config": str(Path(scene_config_path).expanduser().resolve()),
        "reference_dir": str(reference),
        "candidate_dir": str(candidate),
        "object_ids": list(expected_ids),
        "object_count": len(expected_ids),
        "rtol": rtol,
        "atol": atol,
        "topology_exact": True,
        "vertices_close": True,
        "vertex_colors_exact": True,
        "bbox_close": True,
        "objects": objects,
        "scene_glb": {
            "reference_sha256": reference_scene_sha256,
            "candidate_sha256": candidate_scene_sha256,
            "sha256_equal": reference_scene_sha256 == candidate_scene_sha256,
            "hash_is_audit_only": True,
        },
        "api_calls": 0,
    }


def _runtime_lookup(runtime: Any, *names: str) -> Any:
    if runtime is None:
        return None
    containers: list[Any] = [runtime]
    if isinstance(runtime, Mapping):
        nested = runtime.get("sam3d")
    else:
        nested = getattr(runtime, "sam3d", None)
    if nested is not None:
        containers.append(nested)
    for container in containers:
        for name in names:
            if isinstance(container, Mapping):
                value = container.get(name)
            else:
                value = getattr(container, name, None)
            if value is not None and str(value).strip():
                return value
    return None


def _snapshot_empty_directory(path: Path) -> tuple[bool, tuple[Path, ...]]:
    if not path.exists() and not path.is_symlink():
        return False, ()
    if path.is_symlink() or not path.is_dir():
        raise Sam3DStageError(f"Stage 05 target is not a real directory: {path}")
    directories: list[Path] = []
    for item in path.rglob("*"):
        if item.is_symlink() or not item.is_dir():
            raise Sam3DStageError(
                f"Stage 05 target contains data and cannot be replaced: {item}"
            )
        directories.append(item.relative_to(path))
    return True, tuple(sorted(directories, key=lambda value: (len(value.parts), str(value))))


def _remove_empty_directory(path: Path) -> None:
    if not path.exists():
        return
    directories = sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    path.rmdir()


def _restore_empty_directory(path: Path, snapshot: tuple[bool, tuple[Path, ...]]) -> None:
    existed, directories = snapshot
    if not existed:
        return
    path.mkdir(parents=True, exist_ok=True)
    for relative in directories:
        (path / relative).mkdir(parents=True, exist_ok=True)


def _redact_and_limit(text: str | None, limit: int = 2 * 1024 * 1024) -> str:
    result = text or ""
    for name, value in os.environ.items():
        if SECRET_NAME_RE.search(name) and len(value) >= 4:
            result = result.replace(value, "<redacted>")
    if len(result) > limit:
        result = result[:limit] + "\n<truncated>\n"
    return result


class Sam3DStage:
    """Run repository-local SAM3D in an isolated attempt and publish Stage 05."""

    def __init__(
        self,
        scene_config_path: Path,
        runtime_config: Any | None = None,
        *,
        project_root: Path | None = None,
        sam3d_python: Path | None = None,
        checkpoint_path: Path | None = None,
        dino_hub_cache: Path | None = None,
        timeout_seconds: float | None = None,
        gpu_preflight: Callable[[], Mapping[str, Any]] | None = None,
        with_texture_baking: bool = False,
    ) -> None:
        self.scene_config_path = Path(scene_config_path).expanduser().resolve()
        self.runtime_config = runtime_config
        self.project_root = Path(
            project_root
            or _runtime_lookup(runtime_config, "project_root", "scenemosaic_root")
            or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()
        raw_python = sam3d_python or _runtime_lookup(
            runtime_config,
            "sam3d_python",
            "sam3d_python_path",
            "sam3d_interpreter",
            "python",
        )
        raw_checkpoint = checkpoint_path or _runtime_lookup(
            runtime_config,
            "sam3d_checkpoint_path",
            "sam3d_checkpoint",
            "sam3d_config_path",
            "sam3d_config",
            "checkpoint_path",
            "checkpoint",
        )
        raw_dino = dino_hub_cache or _runtime_lookup(
            runtime_config,
            "dino_hub_cache",
            "sam3d_dino_hub_cache",
            "torch_hub_cache",
        )
        self.sam3d_python = (
            Path(raw_python).expanduser().resolve() if raw_python is not None else None
        )
        self.checkpoint_path = (
            Path(raw_checkpoint).expanduser().resolve()
            if raw_checkpoint is not None
            else None
        )
        self.dino_hub_cache = (
            Path(raw_dino).expanduser().resolve() if raw_dino is not None else None
        )
        raw_timeout = timeout_seconds
        if raw_timeout is None:
            raw_timeout = _runtime_lookup(
                runtime_config, "sam3d_timeout_seconds", "timeout_seconds"
            )
        self.timeout_seconds = float(raw_timeout) if raw_timeout is not None else None
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise Sam3DStageError("SAM3D timeout_seconds must be positive")
        self.with_texture_baking = bool(with_texture_baking)
        self._gpu_preflight = gpu_preflight or RuntimeConfig.verify_gpu
        self._gpu_preflight_report: dict[str, Any] | None = None

        _, self.run_root, _, _ = _stage04_records(self.scene_config_path)
        paths = ScenePaths.from_scene_config(self.scene_config_path)
        self.stage05_dir = paths.sam3d_output_dir
        self.attempts_dir = paths.sam3d_attempts_dir
        self.runner_path = (
            self.project_root
            / "vlm_agent"
            / "reconstruction"
            / "sam3d_runner.py"
        )

    def _require_runtime(self) -> None:
        if self.with_texture_baking:
            raise Sam3DStageError(
                "texture baking is unsupported by SceneMosaic's repository-local "
                "sam-3d-objects; Stage 05 is fixed to vertex colors"
            )
        missing = [
            name
            for name, value in (
                ("sam3d_python", self.sam3d_python),
                ("checkpoint_path", self.checkpoint_path),
                ("dino_hub_cache", self.dino_hub_cache),
            )
            if value is None
        ]
        if missing:
            raise Sam3DStageError(
                "SAM3D runtime profile is missing: " + ", ".join(missing)
            )
        assert self.sam3d_python is not None
        _real_nonempty_file(self.sam3d_python, "SAM3D Python interpreter")
        if not os.access(self.sam3d_python, os.X_OK):
            raise Sam3DStageError(
                f"SAM3D Python interpreter is not executable: {self.sam3d_python}"
            )
        _real_nonempty_file(self.runner_path, "repository-local SAM3D runner")

    def _attempt_path(self, attempt_id: str) -> Path:
        if (
            not ATTEMPT_ID_RE.fullmatch(attempt_id)
            or attempt_id in {".", ".."}
            or ".." in attempt_id
        ):
            raise Sam3DStageError(f"invalid SAM3D attempt ID: {attempt_id!r}")
        path = (self.attempts_dir / attempt_id).resolve()
        if not _is_within(path, self.attempts_dir.resolve()):
            raise Sam3DStageError(f"SAM3D attempt escapes attempts directory: {path}")
        return path

    def _job(self, attempt_dir: Path):
        self._require_runtime()
        assert self.checkpoint_path is not None
        assert self.dino_hub_cache is not None
        try:
            return load_job(
                scene_config_path=self.scene_config_path,
                output_dir=attempt_dir,
                published_output_dir=self.stage05_dir,
                project_root=self.project_root,
                checkpoint_path=self.checkpoint_path,
                dino_hub_cache=self.dino_hub_cache,
            )
        except Sam3DRunnerError as exc:
            raise Sam3DStageError(str(exc)) from exc

    def dry_run(self) -> dict[str, Any]:
        """Validate all inputs and runtime paths without imports, GPU, or writes."""

        existing: Sam3DStageArtifacts | None = None
        if self.stage05_dir.exists() or self.stage05_dir.is_symlink():
            try:
                snapshot = _snapshot_empty_directory(self.stage05_dir)
            except Sam3DStageError:
                existing = validate_stage05(
                    self.scene_config_path,
                    self.stage05_dir,
                    project_root=self.project_root,
                )
            else:
                # Existing directories-only placeholders are safe. They are
                # removed with rmdir immediately before atomic publication.
                del snapshot

        hypothetical = self._attempt_path("dry-run")
        job = self._job(hypothetical)
        payload = dry_run_payload(job)
        payload.update(
            {
                "sam3d_python": str(self.sam3d_python),
                "runner_path": str(self.runner_path),
                "stage05_state": "reusable" if existing is not None else "available",
                "reused": existing is not None,
            }
        )
        return payload

    def _build_command(self, attempt_dir: Path) -> list[str]:
        assert self.sam3d_python is not None
        assert self.checkpoint_path is not None
        assert self.dino_hub_cache is not None
        return [
            str(self.sam3d_python),
            str(self.runner_path),
            "--scene-config",
            str(self.scene_config_path),
            "--output-dir",
            str(attempt_dir),
            "--published-output-dir",
            str(self.stage05_dir),
            "--project-root",
            str(self.project_root),
            "--checkpoint-path",
            str(self.checkpoint_path),
            "--dino-hub-cache",
            str(self.dino_hub_cache),
        ]

    def _write_audit(
        self,
        attempt_dir: Path,
        *,
        attempt_id: str,
        status: str,
        command: list[str],
        started_at_utc: str,
        returncode: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> Path:
        audit = {
            "schema_version": "scenemosaic_sam3d_stage_audit",
            "attempt_id": attempt_id,
            "status": status,
            "scene_config": str(self.scene_config_path),
            "attempt_dir": str(attempt_dir),
            "published_output_dir": str(self.stage05_dir),
            "started_at_utc": started_at_utc,
            "updated_at_utc": _utc_now(),
            "command": command,
            "shell": False,
            "approve_gpu": True,
            "gpu_started": True,
            "gpu_preflight": self._gpu_preflight_report,
            "seed": SEED,
            "appearance_mode": "vertex_color",
            "with_texture_baking": False,
            "api_calls": 0,
            "returncode": returncode,
        }
        if error_type:
            audit["error_type"] = error_type
        if error_message:
            audit["error_message"] = _redact_and_limit(error_message, limit=32768)
        path = attempt_dir / "sam3d_stage_audit.json"
        _write_json(path, audit)
        return path

    def run(
        self,
        *,
        approve_gpu: bool = False,
        attempt_id: str | None = None,
    ) -> Sam3DStageArtifacts:
        """Run SAM3D once, retaining failures and atomically publishing success."""

        if self.stage05_dir.exists() or self.stage05_dir.is_symlink():
            try:
                existing = validate_stage05(
                    self.scene_config_path,
                    self.stage05_dir,
                    project_root=self.project_root,
                )
            except Sam3DStageError:
                _snapshot_empty_directory(self.stage05_dir)
            else:
                return replace(existing, reused=True)

        self.dry_run()
        if not approve_gpu:
            raise Sam3DStageError(
                "SAM3D GPU execution requires explicit approve_gpu=True for this invocation"
            )
        gpu_preflight = dict(self._gpu_preflight())
        if not gpu_preflight.get("valid"):
            raise Sam3DStageError(
                "SAM3D GPU execution was not started; GPU preflight failed"
            )
        self._gpu_preflight_report = gpu_preflight

        selected_id = attempt_id or (
            datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S_")
            + uuid.uuid4().hex[:12]
        )
        attempt_dir = self._attempt_path(selected_id)
        if self.attempts_dir.exists() and (
            self.attempts_dir.is_symlink() or not self.attempts_dir.is_dir()
        ):
            raise Sam3DStageError(
                f"SAM3D attempts root must be a real directory: {self.attempts_dir}"
            )
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir.mkdir(exist_ok=False)
        command = self._build_command(attempt_dir)
        started_at = _utc_now()
        environment = {
            name: value
            for name, value in os.environ.items()
            if not SECRET_NAME_RE.search(name)
        }
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"

        try:
            kwargs: dict[str, Any] = {
                "cwd": self.project_root,
                "env": environment,
                "text": True,
                "capture_output": True,
                "check": False,
            }
            if self.timeout_seconds is not None:
                kwargs["timeout"] = self.timeout_seconds
            completed = subprocess.run(command, **kwargs)
            stdout = _redact_and_limit(completed.stdout)
            stderr = _redact_and_limit(completed.stderr)
            _write_text(attempt_dir / "runner.stdout.log", stdout)
            _write_text(attempt_dir / "runner.stderr.log", stderr)
            if completed.returncode != 0:
                self._write_audit(
                    attempt_dir,
                    attempt_id=selected_id,
                    status="failed",
                    command=command,
                    started_at_utc=started_at,
                    returncode=completed.returncode,
                    error_type="Sam3DRunnerProcessError",
                    error_message=stderr or stdout or "SAM3D runner failed",
                )
                raise Sam3DStageError(
                    f"SAM3D runner failed with exit code {completed.returncode}; "
                    f"attempt retained at {attempt_dir}"
                )

            validate_stage05(
                self.scene_config_path,
                attempt_dir,
                published_output_dir=self.stage05_dir,
                project_root=self.project_root,
            )
            self._write_audit(
                attempt_dir,
                attempt_id=selected_id,
                status="validated_for_publication",
                command=command,
                started_at_utc=started_at,
                returncode=completed.returncode,
            )
        except Sam3DStageError as exc:
            audit_path = attempt_dir / "sam3d_stage_audit.json"
            if not audit_path.exists():
                try:
                    self._write_audit(
                        attempt_dir,
                        attempt_id=selected_id,
                        status="failed_validation",
                        command=command,
                        started_at_utc=started_at,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                except Exception:
                    pass
            raise
        except Exception as exc:
            try:
                self._write_audit(
                    attempt_dir,
                    attempt_id=selected_id,
                    status="failed",
                    command=command,
                    started_at_utc=started_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception:
                pass
            raise Sam3DStageError(
                f"SAM3D execution failed; attempt retained at {attempt_dir}"
            ) from exc

        placeholder = _snapshot_empty_directory(self.stage05_dir)
        _remove_empty_directory(self.stage05_dir)
        published = False
        try:
            os.replace(attempt_dir, self.stage05_dir)
            published = True
            artifacts = validate_stage05(
                self.scene_config_path,
                self.stage05_dir,
                project_root=self.project_root,
            )
        except Exception as exc:
            rollback_error: Exception | None = None
            if published and self.stage05_dir.exists():
                try:
                    os.replace(self.stage05_dir, attempt_dir)
                except Exception as rollback_exc:
                    rollback_error = rollback_exc
            try:
                _restore_empty_directory(self.stage05_dir, placeholder)
            except Exception as restore_exc:
                rollback_error = rollback_error or restore_exc
            detail = (
                f"; rollback also failed: {rollback_error}"
                if rollback_error is not None
                else ""
            )
            raise Sam3DStageError(
                "atomic Stage 05 publication failed; successful attempt was retained"
                + detail
            ) from exc

        try:
            self.attempts_dir.rmdir()
        except OSError:
            # Failed or retained attempts are audit evidence and keep the
            # directory non-empty. Only a newly empty container is removed.
            pass

        return replace(
            artifacts,
            attempt_id=selected_id,
            audit_path=self.stage05_dir / "sam3d_stage_audit.json",
            reused=False,
        )


__all__ = [
    "Sam3DStage",
    "Sam3DStageArtifacts",
    "Sam3DStageError",
    "validate_stage05",
]
