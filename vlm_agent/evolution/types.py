from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SceneArtifacts:
    unposed_objects_dir: Path | None = None
    layout_params_dir: Path | None = None
    collision_hulls_dir: Path | None = None
    composed_scene_mesh: Path | None = None
    simulated_layout_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationAction:
    action_type: Literal[
        "compose_scene",
        "build_collision_hulls",
        "simulate_scene_physics",
        "adjust_layout",
        "request_visual_feedback",
        "export_artifacts",
    ]
    target_object_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True)
class OptimizationReport:
    round_index: int
    prompt: str
    model_response: str
    actions: list[OptimizationAction]
    tool_outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineReport:
    """Overall report for the active pipeline."""
    stage: Literal["optimization", "generation", "composition"]
    artifacts: SceneArtifacts
    vlm_prompts: list[str] = field(default_factory=list)
    vlm_responses: list[str] = field(default_factory=list)
    tool_outputs: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
