"""Resolve the versioned on-disk contract for one SceneMosaic run.

The scene configuration is the authority for output locations. Keeping the
contract here prevents individual stages from guessing paths or inspecting
directory names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_STAGES: dict[str, str] = {
    "input": "input",
    "perception": "reconstruction/perception",
    "sam3d_input": "reconstruction/sam3d/input",
    "sam3d_attempts": "reconstruction/sam3d/attempts",
    "sam3d_output": "reconstruction/sam3d/output",
    "relations": "reconstruction/relations",
}


class ScenePathError(ValueError):
    """Raised when a scene configuration declares an unsafe output path."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenePathError(f"cannot parse scene_config: {path}") from exc
    if not isinstance(payload, dict):
        raise ScenePathError("scene_config must contain a JSON object")
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_relative(run_root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ScenePathError(f"scene_config.stages.{label} must be a non-empty path")
    relative = Path(raw)
    if relative.is_absolute():
        raise ScenePathError(f"scene_config.stages.{label} must be relative")
    resolved = (run_root / relative).resolve()
    if resolved == run_root or not _is_within(resolved, run_root):
        raise ScenePathError(f"scene_config.stages.{label} escapes the scene root")
    return resolved


@dataclass(frozen=True)
class ScenePaths:
    """Resolved formal paths for a scene run."""

    run_root: Path
    scene_config_path: Path
    config: Mapping[str, Any]
    input_dir: Path
    perception_dir: Path
    sam3d_input_dir: Path
    sam3d_attempts_dir: Path
    sam3d_output_dir: Path
    relations_dir: Path
    evolution_dir: Path

    @classmethod
    def from_scene_root(cls, scene_root: Path) -> "ScenePaths":
        root = Path(scene_root).expanduser().resolve()
        return cls.from_scene_config(root / "input" / "scene_config.json")

    @classmethod
    def from_scene_config(cls, scene_config_path: Path) -> "ScenePaths":
        config_path = Path(scene_config_path).expanduser().resolve()
        config = _read_config(config_path)
        raw_root = config.get("run_root")
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ScenePathError("scene_config.run_root must be a non-empty path")
        run_root = Path(raw_root).expanduser().resolve()
        expected_config = run_root / "input" / "scene_config.json"
        if config_path != expected_config:
            raise ScenePathError(
                f"scene_config must be the formal run config: {expected_config}"
            )

        if config.get("schema_version") != "scenemosaic_scene_config":
            raise ScenePathError("unsupported scene_config schema; create a new run")
        stages = config.get("stages")
        if not isinstance(stages, dict):
            raise ScenePathError("scene_config.stages must contain a JSON object")

        resolved: dict[str, Path] = {}
        for logical_name, raw in SEMANTIC_STAGES.items():
            if stages.get(logical_name) != raw:
                raise ScenePathError(
                    f"scene_config.stages.{logical_name} must equal {raw!r}"
                )
            resolved[logical_name] = _resolve_relative(
                run_root, raw, label=logical_name
            )

        if resolved["input"] != run_root / "input":
            raise ScenePathError("scene_config input stage must resolve to <run_root>/input")
        values = list(resolved.values())
        if len(set(values)) != len(values):
            raise ScenePathError("scene_config stage paths must be distinct")

        return cls(
            run_root=run_root,
            scene_config_path=config_path,
            config=config,
            input_dir=resolved["input"],
            perception_dir=resolved["perception"],
            sam3d_input_dir=resolved["sam3d_input"],
            sam3d_attempts_dir=resolved["sam3d_attempts"],
            sam3d_output_dir=resolved["sam3d_output"],
            relations_dir=resolved["relations"],
            # The original evolution CLI writes its native ``outputs/`` tree
            # relative to the project root. For an integrated scene, that
            # project root is the scene root; Stage 08 bookkeeping stays
            # in hidden siblings rather than a visible ``evolution/`` wrapper.
            evolution_dir=run_root,
        )

    @property
    def relation_candidates_dir(self) -> Path:
        return self.relations_dir / "candidates"

    @property
    def relation_publications_dir(self) -> Path:
        return self.relations_dir / "publications"

    @property
    def evolution_runs_dir(self) -> Path:
        return self.evolution_dir / ".stage08_attempts"

    @property
    def evolution_publications_dir(self) -> Path:
        return self.evolution_dir / ".stage08_publications"

