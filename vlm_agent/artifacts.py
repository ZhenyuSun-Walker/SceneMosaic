from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .run_paths import SEMANTIC_STAGES, ScenePathError, ScenePaths
from .state import RunState


class RunContractError(ValueError):
    """Raised when a formal scene run does not satisfy the path contract."""


@dataclass(frozen=True)
class RunContract:
    """Resolved, read-only paths from a formal ``input/scene_config.json``."""

    scene_name: str
    scene_config_path: Path
    run_root: Path
    input_image: Path
    config: dict[str, Any]

    @classmethod
    def from_scene_config(cls, scene_config_path: Path) -> "RunContract":
        config_path = Path(scene_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise RunContractError(f"scene_config is not a file: {config_path}")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunContractError(f"cannot parse scene_config: {config_path}") from exc
        if not isinstance(config, dict):
            raise RunContractError("scene_config must contain a JSON object")

        scene_name = str(config.get("scene_name") or "").strip()
        if not scene_name:
            raise RunContractError("scene_config.scene_name must not be empty")

        raw_run_root = config.get("run_root")
        if not isinstance(raw_run_root, str) or not raw_run_root.strip():
            raise RunContractError("scene_config.run_root must be a non-empty path")
        run_root = Path(raw_run_root).expanduser().resolve()
        expected_config = run_root / "input" / "scene_config.json"
        if config_path != expected_config:
            raise RunContractError(
                f"scene_config must be the formal run config: expected {expected_config}, got {config_path}"
            )

        raw_input_image = config.get("input_image")
        if not isinstance(raw_input_image, str) or not raw_input_image.strip():
            raise RunContractError("scene_config.input_image must be a non-empty path")
        input_image = Path(raw_input_image).expanduser().resolve()
        expected_image = run_root / "input" / "image.png"
        if input_image != expected_image:
            raise RunContractError(
                f"scene_config.input_image must be the formal input image: expected {expected_image}, got {input_image}"
            )
        if not input_image.is_file():
            raise RunContractError(f"formal input image is missing: {input_image}")
        try:
            ScenePaths.from_scene_config(config_path)
        except ScenePathError as exc:
            raise RunContractError(str(exc)) from exc

        return cls(
            scene_name=scene_name,
            scene_config_path=config_path,
            run_root=run_root,
            input_image=input_image,
            config=config,
        )

    @property
    def stage03_dir(self) -> Path:
        return self.output_paths.perception_dir

    @property
    def stage04_dir(self) -> Path:
        return self.output_paths.sam3d_input_dir

    @property
    def stage05_dir(self) -> Path:
        return self.output_paths.sam3d_output_dir

    @property
    def sam3d_attempts_dir(self) -> Path:
        return self.output_paths.sam3d_attempts_dir

    @property
    def relations_dir(self) -> Path:
        return self.output_paths.relations_dir

    @property
    def relation_candidates_dir(self) -> Path:
        return self.output_paths.relation_candidates_dir

    @property
    def evolution_dir(self) -> Path:
        return self.output_paths.evolution_dir

    @property
    def output_paths(self) -> ScenePaths:
        try:
            return ScenePaths.from_scene_config(self.scene_config_path)
        except ScenePathError as exc:
            raise RunContractError(str(exc)) from exc


SCENE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
STAGES = dict(SEMANTIC_STAGES)
COORDINATE_SYSTEM = {
    "name": "OpenGL_glTF_global",
    "up_axis": "+Y",
    "down_axis": "-Y",
    "front_axis": "-Z",
    "back_axis": "+Z",
    "left_axis": "-X",
    "right_axis": "+X",
    "ground_plane": "XZ",
}


class SceneRunError(RuntimeError):
    """Raised when a formal scene run cannot be initialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class SceneRun:
    contract: RunContract
    state: RunState
    input_sha256: str


def initialize_scene_run(
    runtime: RuntimeConfig,
    *,
    scene_name: str,
    input_image: Path,
    run_root: Path | None = None,
) -> SceneRun:
    name = str(scene_name or "").strip()
    if not SCENE_NAME_RE.fullmatch(name):
        raise SceneRunError(
            "scene_name may contain only letters, numbers, underscores and hyphens"
        )
    source = Path(input_image).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise SceneRunError(f"input_image must be a regular file: {source}")

    workspace_runs = (runtime.workspace_root / "runs").resolve()
    final_root = (
        Path(run_root).expanduser().resolve()
        if run_root is not None
        else workspace_runs / name
    )
    if not _is_within(final_root, workspace_runs):
        raise SceneRunError(
            f"run_root must stay below the configured workspace runs root: {workspace_runs}"
        )
    if final_root.is_symlink() or (final_root.exists() and not final_root.is_dir()):
        raise SceneRunError(f"run_root is not a regular directory: {final_root}")
    if final_root.exists() and any(final_root.iterdir()):
        raise SceneRunError(f"run_root already exists and is non-empty: {final_root}")

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_root.parent / f".{final_root.name}.init-{uuid.uuid4().hex}.tmp"
    if temporary.exists():
        raise SceneRunError(f"temporary initialization path already exists: {temporary}")

    input_hash = _sha256(source)
    try:
        (temporary / "input").mkdir(parents=True, exist_ok=False)
        (temporary / ".pipeline.lock").touch(exist_ok=False)
        copied_image = temporary / "input" / "image.png"
        shutil.copy2(source, copied_image)

        final_image = final_root / "input" / "image.png"
        config: dict[str, Any] = {
            "schema_version": "scenemosaic_scene_config",
            "scene_name": name,
            "run_root": str(final_root),
            "input_image": str(final_image),
            "source_input_image": str(source),
            "source_input_sha256": input_hash,
            "workspace_root": str(runtime.workspace_root),
            "vlm_model": runtime.vlm_model,
            "object_id_base": 1,
            "object_id_format": "object_{index:03d}",
            "wall_id_format": "wall_{index:03d}",
            "coordinate_system": COORDINATE_SYSTEM,
            "conda_envs": {
                "sam3": str(runtime.perception_python.parent.parent),
                "sam3d": str(runtime.sam3d_python.parent.parent),
            },
            "stage_envs": {
                "perception": "sam3",
                "sam3d_input": "sam3",
                "sam3d_output": "sam3d",
                "relations": "relation",
                "evolution": "scenemosaic",
            },
            "stages": STAGES,
            "keep_virtual_surfaces": ["virtual_floor", "virtual_ceiling"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        config_path = temporary / "input" / "scene_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if final_root.exists():
            final_root.rmdir()
        os.replace(temporary, final_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    contract = RunContract.from_scene_config(final_root / "input" / "scene_config.json")
    state = RunState.create(final_root, scene_name=name)
    state.record_stage(
        "input",
        status="published",
        successful_state="initialized",
        facts={
            "scene_config": str(contract.scene_config_path),
            "input_image": str(contract.input_image),
            "input_sha256": input_hash,
        },
    )
    return SceneRun(contract=contract, state=state, input_sha256=input_hash)


def load_scene_run(scene_root: Path) -> SceneRun:
    root = Path(scene_root).expanduser().resolve()
    contract = RunContract.from_scene_config(root / "input" / "scene_config.json")
    state = RunState.load(root)
    input_hash = _sha256(contract.input_image)
    expected = contract.config.get("source_input_sha256")
    if expected and input_hash != expected:
        raise SceneRunError("formal input image hash no longer matches scene_config")
    return SceneRun(contract=contract, state=state, input_sha256=input_hash)
