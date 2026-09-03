from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_agent.run_paths import ScenePathError, ScenePaths

READY_STATUSES = frozenset({"sam3d_ready", "sam3d_ready_degraded"})
SEED = 42
SECRET_NAME_RE = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


class Sam3DRunnerError(RuntimeError):
    """Raised when the repository-local SAM3D runner contract is invalid."""


@dataclass(frozen=True)
class Sam3DInputRecord:
    order_index: int
    object_id: str
    sam3d_internal_name: str
    category: str | None
    description: str | None
    source: Any
    mask_path: Path
    mask_sha256: str
    bbox_2d_norm_xyxy: Any
    mask_bbox_abs_xyxy: Any
    sam3: dict[str, Any]


@dataclass(frozen=True)
class Sam3DRunnerJob:
    scene_name: str
    run_root: Path
    scene_config_path: Path
    stage04_dir: Path
    input_image: Path
    source_manifest: Path
    source_manifest_sha256: str
    output_dir: Path
    published_output_dir: Path
    project_root: Path
    sam3d_source_root: Path
    inference_path: Path
    checkpoint_path: Path
    checkpoint_assets: tuple[Path, ...]
    dino_hub_dir: Path
    dino_repository: Path
    dino_checkpoints: tuple[Path, ...]
    records: tuple[Sam3DInputRecord, ...]


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
        raise Sam3DRunnerError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise Sam3DRunnerError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _redact_secrets(message: str) -> str:
    result = message
    for name, value in os.environ.items():
        if SECRET_NAME_RE.search(name) and len(value) >= 4:
            result = result.replace(value, "<redacted>")
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _real_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise Sam3DRunnerError(f"{label} must be a real non-empty file: {path}")


CHECKPOINT_ASSET_SUFFIXES = frozenset(
    {".yaml", ".yml", ".ckpt", ".pt", ".pth", ".safetensors"}
)


def _checkpoint_asset_paths(config_path: Path) -> tuple[Path, ...]:
    """Resolve every local model/config file reachable from a pipeline YAML."""

    root = Path(config_path).expanduser().resolve()
    pending = [root]
    assets: set[Path] = set()
    parsed_yaml: set[Path] = set()

    def visit(value: Any, *, relative_to: Path) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item, relative_to=relative_to)
            return
        if isinstance(value, list):
            for item in value:
                visit(item, relative_to=relative_to)
            return
        if not isinstance(value, str):
            return
        candidate = Path(value).expanduser()
        if candidate.suffix.lower() not in CHECKPOINT_ASSET_SUFFIXES:
            return
        if not candidate.is_absolute():
            candidate = relative_to / candidate
        candidate = candidate.resolve()
        _real_nonempty_file(candidate, "SAM3D checkpoint dependency")
        assets.add(candidate)
        if (
            candidate.suffix.lower() in {".yaml", ".yml"}
            and candidate not in parsed_yaml
        ):
            pending.append(candidate)

    while pending:
        yaml_path = pending.pop()
        if yaml_path in parsed_yaml:
            continue
        _real_nonempty_file(yaml_path, "SAM3D checkpoint YAML")
        parsed_yaml.add(yaml_path)
        assets.add(yaml_path)
        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise Sam3DRunnerError(
                f"cannot parse SAM3D checkpoint YAML: {yaml_path}"
            ) from exc
        visit(payload, relative_to=yaml_path.parent)
    return tuple(sorted(assets))


def _asset_records(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _resolve_dino_cache(raw_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    candidate = Path(raw_path).expanduser().resolve()
    if not candidate.is_dir() or candidate.is_symlink():
        raise Sam3DRunnerError(
            f"DINO hub cache must be a real directory: {candidate}"
        )

    if (candidate / "hubconf.py").is_file():
        repository = candidate
        hub_dir = candidate.parent
    else:
        repository = candidate / "facebookresearch_dinov2_main"
        hub_dir = candidate
    _real_nonempty_file(repository / "hubconf.py", "cached DINO hubconf")

    checkpoint_dirs = (hub_dir / "checkpoints", hub_dir / "hub" / "checkpoints")
    checkpoints = tuple(
        sorted(
            path.resolve()
            for directory in checkpoint_dirs
            if directory.is_dir()
            for path in directory.glob("*.pth")
            if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        )
    )
    if not checkpoints:
        raise Sam3DRunnerError(
            f"DINO hub cache contains no local .pth checkpoint: {hub_dir}"
        )
    return hub_dir, repository.resolve(), checkpoints


def _load_scene_contract(scene_config_path: Path) -> tuple[str, Path, Path, Path]:
    config_path = Path(scene_config_path).expanduser().resolve()
    _real_nonempty_file(config_path, "scene_config")
    config = _read_json(config_path, "scene_config")
    scene_name = str(config.get("scene_name") or "").strip()
    if not scene_name:
        raise Sam3DRunnerError("scene_config.scene_name must not be empty")

    raw_run_root = config.get("run_root")
    if not isinstance(raw_run_root, str) or not raw_run_root.strip():
        raise Sam3DRunnerError("scene_config.run_root must be a non-empty path")
    run_root = Path(raw_run_root).expanduser().resolve()
    expected_config = run_root / "input" / "scene_config.json"
    if config_path != expected_config:
        raise Sam3DRunnerError(
            f"scene_config must be the formal run config: {expected_config}"
        )

    try:
        paths = ScenePaths.from_scene_config(config_path)
    except ScenePathError as exc:
        raise Sam3DRunnerError(str(exc)) from exc
    stage04_dir = paths.sam3d_input_dir
    input_image = stage04_dir / "image.png"
    manifest_path = stage04_dir / "sam3d_input_manifest.json"
    _real_nonempty_file(input_image, "Stage 04 image")
    _real_nonempty_file(manifest_path, "Stage 04 manifest")
    return scene_name, run_root, input_image, manifest_path


def _load_records(
    manifest_path: Path,
    *,
    scene_name: str,
    input_image: Path,
) -> tuple[Sam3DInputRecord, ...]:
    manifest = _read_json(manifest_path, "Stage 04 manifest")
    if manifest.get("delivery_status") not in READY_STATUSES:
        raise Sam3DRunnerError(
            "Stage 04 delivery status is not sam3d_ready or sam3d_ready_degraded"
        )
    if manifest.get("sam3d_execution_allowed") is not True:
        raise Sam3DRunnerError("Stage 04 does not allow SAM3D execution")
    # Adoption preserves the protected delivery's source scene name. A formal
    # SceneMosaic run may intentionally use a different run/scene identifier.
    if not str(manifest.get("scene_name") or "").strip():
        raise Sam3DRunnerError("Stage 04 manifest.scene_name must not be empty")

    raw_image = manifest.get("image_path")
    if not isinstance(raw_image, str) or Path(raw_image).expanduser().resolve() != input_image:
        raise Sam3DRunnerError("Stage 04 manifest image_path is not the formal image")

    objects = manifest.get("objects")
    object_count = manifest.get("object_count")
    if (
        not isinstance(objects, list)
        or not objects
        or not isinstance(object_count, int)
        or object_count != len(objects)
    ):
        raise Sam3DRunnerError("Stage 04 object_count must match a non-empty objects list")

    masks_root = manifest_path.parent / "masks"
    records: list[Sam3DInputRecord] = []
    seen_masks: set[Path] = set()
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise Sam3DRunnerError(f"Stage 04 object {index} is not a JSON object")
        expected_id = f"object_{index + 1:03d}"
        object_id = str(item.get("object_id") or "")
        internal_name = str(item.get("sam3d_internal_name") or "")
        if object_id != expected_id or internal_name != expected_id:
            raise Sam3DRunnerError(
                "Stage 04 IDs must be continuous object_001 through object_N and "
                "sam3d_internal_name must equal object_id"
            )
        if item.get("order_index") != index:
            raise Sam3DRunnerError(f"Stage 04 order_index mismatch for {object_id}")

        raw_mask = item.get("mask_path")
        if not isinstance(raw_mask, str) or not raw_mask.strip():
            raise Sam3DRunnerError(f"Stage 04 mask_path is missing for {object_id}")
        mask_path = Path(raw_mask).expanduser().resolve()
        if not _is_within(mask_path, masks_root.resolve()) or mask_path.parent != masks_root.resolve():
            raise Sam3DRunnerError(
                f"Stage 04 mask must be a direct child of the formal masks directory: {mask_path}"
            )
        _real_nonempty_file(mask_path, f"Stage 04 mask for {object_id}")
        if mask_path in seen_masks:
            raise Sam3DRunnerError(f"duplicate Stage 04 mask: {mask_path}")
        seen_masks.add(mask_path)
        mask_sha256 = str(item.get("mask_sha256") or "")
        if len(mask_sha256) != 64 or _sha256(mask_path) != mask_sha256:
            raise Sam3DRunnerError(f"Stage 04 mask hash mismatch for {object_id}")

        records.append(
            Sam3DInputRecord(
                order_index=index,
                object_id=object_id,
                sam3d_internal_name=internal_name,
                category=(
                    str(item["category"])
                    if item.get("category") is not None
                    else None
                ),
                description=(
                    str(item["description"])
                    if item.get("description") is not None
                    else None
                ),
                source=item.get("source"),
                mask_path=mask_path,
                mask_sha256=mask_sha256,
                bbox_2d_norm_xyxy=item.get("bbox_2d_norm_xyxy"),
                mask_bbox_abs_xyxy=item.get("mask_bbox_abs_xyxy"),
                sam3=item.get("sam3") if isinstance(item.get("sam3"), dict) else {},
            )
        )
    return tuple(records)


def load_job(
    *,
    scene_config_path: Path,
    output_dir: Path,
    published_output_dir: Path,
    project_root: Path,
    checkpoint_path: Path,
    dino_hub_cache: Path,
) -> Sam3DRunnerJob:
    project = Path(project_root).expanduser().resolve()
    expected_project = Path(__file__).resolve().parents[2]
    if project != expected_project:
        raise Sam3DRunnerError(
            "SAM3D project_root must be the SceneMosaic repository containing this runner: "
            f"{expected_project}"
        )

    source_root = project / "sam-3d-objects"
    inference_path = source_root / "notebook" / "inference.py"
    if not source_root.is_dir() or source_root.is_symlink():
        raise Sam3DRunnerError(
            f"repository-local sam-3d-objects is missing: {source_root}"
        )
    _real_nonempty_file(inference_path, "repository-local SAM3D inference module")

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    _real_nonempty_file(checkpoint, "SAM3D checkpoint configuration")
    if checkpoint.suffix.lower() not in {".yaml", ".yml"}:
        raise Sam3DRunnerError("SAM3D checkpoint configuration must be YAML")
    checkpoint_assets = _checkpoint_asset_paths(checkpoint)
    dino_hub_dir, dino_repository, dino_checkpoints = _resolve_dino_cache(
        Path(dino_hub_cache)
    )

    scene_name, run_root, input_image, manifest_path = _load_scene_contract(
        Path(scene_config_path)
    )
    records = _load_records(
        manifest_path,
        scene_name=scene_name,
        input_image=input_image,
    )

    output = Path(output_dir).expanduser().resolve()
    published = Path(published_output_dir).expanduser().resolve()
    expected_published = ScenePaths.from_scene_config(
        Path(scene_config_path)
    ).sam3d_output_dir
    if published != expected_published:
        raise Sam3DRunnerError(
            f"published output must be the formal Stage 05 directory: {expected_published}"
        )
    if not _is_within(output, run_root) or output == published:
        raise Sam3DRunnerError(
            "runner output must be an isolated attempt directory inside the formal run root"
        )

    return Sam3DRunnerJob(
        scene_name=scene_name,
        run_root=run_root,
        scene_config_path=Path(scene_config_path).expanduser().resolve(),
        stage04_dir=manifest_path.parent,
        input_image=input_image,
        source_manifest=manifest_path,
        source_manifest_sha256=_sha256(manifest_path),
        output_dir=output,
        published_output_dir=published,
        project_root=project,
        sam3d_source_root=source_root,
        inference_path=inference_path,
        checkpoint_path=checkpoint,
        dino_hub_dir=dino_hub_dir,
        checkpoint_assets=checkpoint_assets,
        dino_repository=dino_repository,
        dino_checkpoints=dino_checkpoints,
        records=records,
    )


def dry_run_payload(job: Sam3DRunnerJob) -> dict[str, Any]:
    return {
        "status": "dry_run_ok",
        "scene_name": job.scene_name,
        "scene_config": str(job.scene_config_path),
        "input_image": str(job.input_image),
        "source_manifest": str(job.source_manifest),
        "source_manifest_sha256": job.source_manifest_sha256,
        "attempt_output_dir": str(job.output_dir),
        "published_output_dir": str(job.published_output_dir),
        "project_root": str(job.project_root),
        "sam3d_source_root": str(job.sam3d_source_root),
        "checkpoint_path": str(job.checkpoint_path),
        "checkpoint_sha256": _sha256(job.checkpoint_path),
        "checkpoint_assets": _asset_records(job.checkpoint_assets),
        "dino_hub_dir": str(job.dino_hub_dir),
        "dino_repository": str(job.dino_repository),
        "dino_checkpoint_count": len(job.dino_checkpoints),
        "dino_checkpoints": _asset_records(job.dino_checkpoints),
        "object_count": len(job.records),
        "object_ids": [record.object_id for record in job.records],
        "seed": SEED,
        "appearance_mode": "vertex_color",
        "with_texture_baking": False,
        "gpu_started": False,
        "api_calls": 0,
    }


def _ensure_empty_attempt(path: Path) -> None:
    if path.is_symlink():
        raise Sam3DRunnerError(f"attempt output cannot be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise Sam3DRunnerError(f"attempt output is not a directory: {path}")
        if any(path.iterdir()):
            raise Sam3DRunnerError(f"attempt output is not empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _install_offline_torch_hub(torch_module: Any, job: Sam3DRunnerJob) -> None:
    torch_module.hub.set_dir(str(job.dino_hub_dir))
    original_load: Callable[..., Any] = torch_module.hub.load

    def offline_load(
        repo_or_dir: str,
        model: str,
        *args: Any,
        source: str = "github",
        **kwargs: Any,
    ) -> Any:
        if str(repo_or_dir) != "facebookresearch/dinov2":
            raise Sam3DRunnerError(
                f"network-backed torch.hub repository is disabled: {repo_or_dir}"
            )
        return original_load(
            str(job.dino_repository),
            model,
            *args,
            source="local",
            **kwargs,
        )

    def block_download(url: str, *args: Any, **kwargs: Any) -> None:
        raise Sam3DRunnerError(
            "offline DINO checkpoint is missing; network download is disabled"
        )

    torch_module.hub.load = offline_load
    torch_module.hub.download_url_to_file = block_download


def _load_runtime_modules(job: Sam3DRunnerJob) -> dict[str, Any]:
    conda_prefix = Path(sys.executable).resolve().parent.parent
    os.environ.setdefault("CONDA_PREFIX", str(conda_prefix))
    os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
    os.environ["LIDRA_SKIP_INIT"] = "true"
    os.environ["TORCH_HOME"] = str(job.dino_hub_dir)

    source_root = str(job.sam3d_source_root)
    notebook_root = str(job.sam3d_source_root / "notebook")
    for entry in (source_root, notebook_root, str(job.project_root)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

    import torch

    _install_offline_torch_hub(torch, job)
    inference_module = importlib.import_module("inference")
    loaded_inference = Path(inference_module.__file__).resolve()
    if loaded_inference != job.inference_path.resolve():
        raise Sam3DRunnerError(
            f"SAM3D inference resolved outside SceneMosaic: {loaded_inference}"
        )
    sam3d_package = importlib.import_module("sam3d_objects")
    package_path = Path(sam3d_package.__file__).resolve()
    if not _is_within(package_path, job.sam3d_source_root):
        raise Sam3DRunnerError(
            f"sam3d_objects package resolved outside SceneMosaic: {package_path}"
        )

    utilities = importlib.import_module("vlm_agent.evolution.sam3d_utils")
    return {
        "torch": torch,
        "Inference": inference_module.Inference,
        "load_image": inference_module.load_image,
        "load_mask": inference_module.load_mask,
        "make_scene": utilities.make_scene_untextured_mesh,
        "normalize_scene": utilities.normalize_scene_to_canonical_cube,
    }


def _mesh_has_vertex_colors(mesh: Any) -> bool:
    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "vertex_colors", None)
    return (
        getattr(visual, "kind", None) == "vertex"
        and colors is not None
        and len(colors) == len(mesh.vertices)
        and len(colors) > 0
    )


def _relative_symlink(target: Path, link_path: Path) -> None:
    relative = os.path.relpath(target, link_path.parent)
    link_path.symlink_to(relative)


def _relative_from_output(job: Sam3DRunnerJob, path: Path) -> str:
    return os.path.relpath(path, job.output_dir)


def _export_outputs(
    job: Sam3DRunnerJob,
    outputs: list[dict[str, Any]],
    modules: dict[str, Any],
) -> None:
    import trimesh

    if len(outputs) != len(job.records):
        raise Sam3DRunnerError(
            f"SAM3D returned {len(outputs)} outputs for {len(job.records)} objects"
        )
    missing_meshes = [
        job.records[index].object_id
        for index, output in enumerate(outputs)
        if not isinstance(output, dict) or output.get("glb") is None
    ]
    if missing_meshes:
        raise Sam3DRunnerError(
            "SAM3D returned no mesh for: " + ", ".join(missing_meshes)
        )

    scene_mesh, posed_meshes, unposed_meshes, layout_params = modules[
        "make_scene"
    ](*outputs, return_all=True)
    scene_mesh, posed_meshes, layout_params = modules["normalize_scene"](
        scene_mesh,
        posed_meshes,
        layout_params,
        cube_size=1.0,
        in_place=True,
    )
    expected_count = len(job.records)
    actual_counts = {
        "posed": len(posed_meshes),
        "unposed": len(unposed_meshes),
        "layout": len(layout_params),
    }
    if scene_mesh is None or any(value != expected_count for value in actual_counts.values()):
        raise Sam3DRunnerError(
            f"SAM3D output closure failed: expected {expected_count}, got {actual_counts}"
        )
    missing_colors = [
        record.object_id
        for record, posed, unposed in zip(job.records, posed_meshes, unposed_meshes)
        if not _mesh_has_vertex_colors(posed) or not _mesh_has_vertex_colors(unposed)
    ]
    if missing_colors:
        raise Sam3DRunnerError(
            "vertex color verification failed for: " + ", ".join(missing_colors)
        )

    posed_dir = job.output_dir / "posed_objects"
    unposed_dir = job.output_dir / "unposed_objects"
    layout_dir = job.output_dir / "layout_params"
    objects_dir = job.output_dir / "objects"
    for directory in (posed_dir, unposed_dir, layout_dir, objects_dir):
        directory.mkdir(parents=True, exist_ok=False)

    index_entries: list[dict[str, Any]] = []
    for index, record in enumerate(job.records):
        posed_path = posed_dir / f"{record.object_id}.glb"
        unposed_path = unposed_dir / f"{record.object_id}.glb"
        layout_path = layout_dir / f"{record.object_id}.json"
        object_dir = objects_dir / record.object_id
        object_dir.mkdir()

        posed_meshes[index].export(posed_path)
        unposed_meshes[index].export(unposed_path)
        layout = dict(layout_params[index])
        layout.update(
            {
                "sam3d_index": record.order_index,
                "sam3d_internal_name": record.sam3d_internal_name,
                "object_id": record.object_id,
                "category": record.category,
                "description": record.description,
                "source": record.source,
                "source_mask_path": str(record.mask_path),
                "source_mask_sha256": record.mask_sha256,
                "vlm_bbox_2d_norm_xyxy": record.bbox_2d_norm_xyxy,
                "sam3_mask_bbox_abs_xyxy": record.mask_bbox_abs_xyxy,
                "posed_glb": _relative_from_output(job, posed_path),
                "unposed_glb": _relative_from_output(job, unposed_path),
                "layout_json": _relative_from_output(job, layout_path),
            }
        )
        _write_json(layout_path, layout)

        local_mesh = object_dir / "local_mesh.glb"
        camera_mesh = object_dir / "camera_mesh.glb"
        full_json = object_dir / f"{record.object_id}_full.json"
        _relative_symlink(unposed_path, local_mesh)
        _relative_symlink(posed_path, camera_mesh)
        _relative_symlink(layout_path, full_json)

        final_object_dir = job.published_output_dir / "objects" / record.object_id
        index_entries.append(
            {
                "order_index": record.order_index,
                "sam3d_index": record.order_index,
                "sam3d_internal_name": record.sam3d_internal_name,
                "object_id": record.object_id,
                "category": record.category,
                "description": record.description,
                "mask_path": str(record.mask_path),
                "mask_sha256": record.mask_sha256,
                "output_dir": str(final_object_dir),
                "local_mesh_path": str(final_object_dir / "local_mesh.glb"),
                "camera_mesh_path": str(final_object_dir / "camera_mesh.glb"),
                "full_json_path": str(final_object_dir / f"{record.object_id}_full.json"),
                "posed_glb_sha256": _sha256(posed_path),
                "unposed_glb_sha256": _sha256(unposed_path),
                "layout_sha256": _sha256(layout_path),
            }
        )

    export_scene = trimesh.Scene()
    for record, mesh in zip(job.records, posed_meshes):
        export_scene.add_geometry(
            mesh,
            geom_name=record.object_id,
            node_name=record.object_id,
        )
    scene_path = job.output_dir / "scene.glb"
    export_scene.export(scene_path)

    index_path = job.output_dir / "object_index_map.json"
    index_payload = {
        "schema_version": "scenemosaic_sam3d_object_index",
        "scene_name": job.scene_name,
        "run_root": str(job.run_root),
        "stage": "sam3d_output",
        "source_manifest": str(job.source_manifest),
        "source_manifest_sha256": job.source_manifest_sha256,
        "output_dir": str(job.published_output_dir),
        "seed": SEED,
        "with_texture_baking": False,
        "texture_verified": False,
        "vertex_color_verified": True,
        "appearance_mode": "vertex_color",
        "scene_glb_sha256": _sha256(scene_path),
        "sam3d_source": {
            "root": str(job.sam3d_source_root),
            "inference_path": str(job.inference_path),
        },
        "checkpoint": {
            "assets": _asset_records(job.checkpoint_assets),
            "path": str(job.checkpoint_path),
            "sha256": _sha256(job.checkpoint_path),
        },
        "dino": {
            "hub_dir": str(job.dino_hub_dir),
            "checkpoints": _asset_records(job.dino_checkpoints),
            "repository": str(job.dino_repository),
            "offline": True,
        },
        "objects": index_entries,
    }
    _write_json(index_path, index_payload)
    _write_json(
        job.output_dir / "sam3d_run_info.json",
        {
            "schema_version": "scenemosaic_sam3d_run_info",
            "stage": "sam3d_output",
            "scene_name": job.scene_name,
            "run_root": str(job.run_root),
            "input_image": str(job.input_image),
            "source_manifest": str(job.source_manifest),
            "source_manifest_sha256": job.source_manifest_sha256,
            "output_dir": str(job.published_output_dir),
            "object_index_map": str(job.published_output_dir / index_path.name),
            "object_index_map_sha256": _sha256(index_path),
            "object_count": len(job.records),
            "created_at_utc": _utc_now(),
            "status": "success",
            "seed": SEED,
            "with_texture_baking": False,
            "texture_verified": False,
            "vertex_color_verified": True,
            "appearance_mode": "vertex_color",
            "api_calls": 0,
            "source_is_repository_local": True,
            "dino_offline": True,
        },
    )


def execute(job: Sam3DRunnerJob) -> dict[str, Any]:
    _ensure_empty_attempt(job.output_dir)
    modules = _load_runtime_modules(job)
    inference = modules["Inference"](str(job.checkpoint_path), compile=False)
    image = modules["load_image"](str(job.input_image))
    outputs: list[dict[str, Any]] = []
    for record in job.records:
        print(
            f"[{record.order_index:03d}] reconstruct "
            f"{record.sam3d_internal_name} | {record.category}",
            flush=True,
        )
        mask = modules["load_mask"](str(record.mask_path))
        outputs.append(inference(image, mask, seed=SEED))
    _export_outputs(job, outputs, modules)
    return {
        "status": "success",
        "scene_name": job.scene_name,
        "object_count": len(job.records),
        "object_ids": [record.object_id for record in job.records],
        "output_dir": str(job.output_dir),
        "published_output_dir": str(job.published_output_dir),
        "seed": SEED,
        "appearance_mode": "vertex_color",
        "api_calls": 0,
    }


def _failed_run_info(job: Sam3DRunnerJob, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": "scenemosaic_sam3d_run_info",
        "stage": "sam3d_output",
        "scene_name": job.scene_name,
        "run_root": str(job.run_root),
        "source_manifest": str(job.source_manifest),
        "source_manifest_sha256": job.source_manifest_sha256,
        "output_dir": str(job.published_output_dir),
        "object_count": len(job.records),
        "created_at_utc": _utc_now(),
        "status": "failed",
        "seed": SEED,
        "appearance_mode": "vertex_color",
        "with_texture_baking": False,
        "api_calls": 0,
        "error_type": type(exc).__name__,
        "error_message": _redact_secrets(str(exc)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repository-local sam-3d-objects for formal SceneMosaic Stage 05."
    )
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--published-output-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dino-hub-cache", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    job: Sam3DRunnerJob | None = None
    try:
        job = load_job(
            scene_config_path=Path(args.scene_config),
            output_dir=Path(args.output_dir),
            published_output_dir=Path(args.published_output_dir),
            project_root=Path(args.project_root),
            checkpoint_path=Path(args.checkpoint_path),
            dino_hub_cache=Path(args.dino_hub_cache),
        )
        result = dry_run_payload(job) if args.dry_run else execute(job)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        if job is not None and not args.dry_run:
            try:
                _write_json(job.output_dir / "sam3d_run_info.json", _failed_run_info(job, exc))
            except Exception:
                pass
        print(f"ERROR: {_redact_secrets(str(exc))}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
