from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    output_dir: Path | None = None

    @property
    def tools_dir(self) -> Path:
        return self.root / "tools"

    @property
    def outputs_dir(self) -> Path:
        return self.root / self.output_dir if self.output_dir is not None else self.root / "outputs"

    @property
    def layout_params_dir(self) -> Path:
        return self.outputs_dir / "layout_params"

    @property
    def unposed_objects_dir(self) -> Path:
        return self.outputs_dir / "unposed_objects"

    @property
    def collision_hulls_dir(self) -> Path:
        return self.outputs_dir / "collision_hulls"

    @property
    def composed_scene_path(self) -> Path:
        return self.outputs_dir / "composed_scene.glb"

    @property
    def simulated_layout_dir(self) -> Path:
        return self.outputs_dir / "layout_params_simulated"

    @property
    def optimization_log_path(self) -> Path:
        return self.outputs_dir / "vlm_agent_log.json"

    @property
    def representative_scenes_dir(self) -> Path:
        return self.outputs_dir / "representative_scenes"
