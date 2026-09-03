from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .reconstruction.sam3_runtime import (
    verify_perception_dependencies,
    verify_sam3_runtime_provider,
)


RUNTIME_CONFIG_ENV = "SCENEMOSAIC_RUNTIME_CONFIG"
DEFAULT_VLM_MODEL = "gpt-5.5"
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|BASE_URL)", re.I
)
RENDERER_FUNCTIONS: Mapping[str, tuple[str, ...]] = {
    "compose_scene": ("layout_dir", "unposed_dir"),
    "denormalize_point": ("point_normalized", "transform_info"),
    "generate_2d_grid_on_plane": ("posed_meshes", "camera_direction", "grid_size"),
    "render_scene_with_grid": ("meshes", "camera_direction", "radius", "image_size"),
    "save_grid_mapping": ("mapping", "output_path", "transform_info"),
    "resolve_orthographic_camera_frame": ("meshes", "camera_direction", "radius"),
    "build_camera_pose": ("camera_direction", "radius"),
    "get_camera_matrix": ("yfov_deg", "image_size"),
    "project_3d_to_2d": ("point_3d", "camera_pose", "camera_matrix", "image_size"),
}


class RuntimeConfigError(ValueError):
    """Raised when the unified runtime profile is unsafe or incomplete."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeConfigError(f"{label} must contain a JSON object: {path}")
    return payload


def _path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _env_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not ENV_NAME_RE.fullmatch(value):
        raise RuntimeConfigError(f"{label} must be an environment variable name")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """One secret-free deployment profile for the complete SceneMosaic pipeline."""

    project_root: Path
    profile_path: Path
    workspace_root: Path
    perception_attempts_root: Path
    perception_python: Path
    perception_model_dir: Path
    sam3d_python: Path
    sam3d_checkpoint: Path
    dino_hub_cache: Path
    relation_precompute_python: Path
    relation_runner_python: Path
    scenemosaic_python: Path
    perception_api_key_env: str = "OPENAI_API_KEY"
    perception_api_base_env: str = "OPENAI_BASE_URL"
    relation_api_key_env: str = "OPENAI_API_KEY"
    relation_api_base_env: str = "OPENAI_BASE_URL"
    scenemosaic_api_key_env: str = "OPENAI_API_KEY"
    scenemosaic_api_base_env: str = "OPENAI_BASE_URL"
    vlm_model: str = DEFAULT_VLM_MODEL

    @classmethod
    def load(
        cls,
        profile_path: Path | str | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        project_root: Path | None = None,
    ) -> "RuntimeConfig":
        env = os.environ if environ is None else environ
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        selected = profile_path or env.get(RUNTIME_CONFIG_ENV)
        profile = (
            Path(selected).expanduser().resolve()
            if selected
            else root / "configs" / "runtime.local.json"
        )
        payload = _read_json(profile, "runtime profile")

        forbidden = {
            "api_key",
            "perception_api_key",
            "relation_api_key",
            "scenemosaic_api_key",
            "authorization",
        }
        leaked = sorted(forbidden.intersection(payload))
        if leaked:
            raise RuntimeConfigError(
                "runtime profile may contain environment variable names only; "
                f"forbidden secret fields: {leaked}"
            )

        return cls(
            project_root=root,
            profile_path=profile,
            workspace_root=_path(
                payload.get("workspace_root"),
                base=profile.parent,
                label="workspace_root",
            ),
            perception_attempts_root=_path(
                payload.get("perception_attempts_root"),
                base=profile.parent,
                label="perception_attempts_root",
            ),
            perception_python=_path(
                payload.get("perception_python"),
                base=profile.parent,
                label="perception_python",
            ),
            perception_model_dir=_path(
                payload.get("perception_model_dir"),
                base=profile.parent,
                label="perception_model_dir",
            ),
            sam3d_python=_path(
                payload.get("sam3d_python"), base=profile.parent, label="sam3d_python"
            ),
            sam3d_checkpoint=_path(
                payload.get("sam3d_checkpoint"),
                base=profile.parent,
                label="sam3d_checkpoint",
            ),
            dino_hub_cache=_path(
                payload.get("dino_hub_cache"),
                base=profile.parent,
                label="dino_hub_cache",
            ),
            relation_precompute_python=_path(
                payload.get("relation_precompute_python"),
                base=profile.parent,
                label="relation_precompute_python",
            ),
            relation_runner_python=_path(
                payload.get("relation_runner_python"),
                base=profile.parent,
                label="relation_runner_python",
            ),
            scenemosaic_python=_path(
                payload.get("scenemosaic_python", sys.executable),
                base=profile.parent,
                label="scenemosaic_python",
            ),
            perception_api_key_env=_env_name(
                payload.get("perception_api_key_env", "OPENAI_API_KEY"),
                label="perception_api_key_env",
            ),
            perception_api_base_env=_env_name(
                payload.get("perception_api_base_env", "OPENAI_BASE_URL"),
                label="perception_api_base_env",
            ),
            relation_api_key_env=_env_name(
                payload.get("relation_api_key_env", "OPENAI_API_KEY"),
                label="relation_api_key_env",
            ),
            relation_api_base_env=_env_name(
                payload.get("relation_api_base_env", "OPENAI_BASE_URL"),
                label="relation_api_base_env",
            ),
            scenemosaic_api_key_env=_env_name(
                payload.get("scenemosaic_api_key_env", "OPENAI_API_KEY"),
                label="scenemosaic_api_key_env",
            ),
            scenemosaic_api_base_env=_env_name(
                payload.get("scenemosaic_api_base_env", "OPENAI_BASE_URL"),
                label="scenemosaic_api_base_env",
            ),
            vlm_model=str(payload.get("vlm_model") or DEFAULT_VLM_MODEL),
        )

    @property
    def agent_root(self) -> Path:
        return self.project_root / "vlm_agent"

    @property
    def reconstruction_root(self) -> Path:
        return self.agent_root / "reconstruction"

    @property
    def perception_root(self) -> Path:
        return self.reconstruction_root / "perception"

    @property
    def perception_entrypoint(self) -> Path:
        return self.perception_root / "pipeline" / "runner.py"

    @property
    def relation_package_root(self) -> Path:
        return self.reconstruction_root / "relations"

    @property
    def relation_wrapper(self) -> Path:
        return (
            self.relation_package_root
            / "tools"
            / "benchmark_relation_async_planning.py"
        )

    @property
    def sam3_source_root(self) -> Path:
        return self.project_root / "sam3"

    @property
    def sam3d_source_root(self) -> Path:
        return self.project_root / "sam-3d-objects"

    @property
    def renderer_path(self) -> Path:
        return self.agent_root / "evolution" / "tools" / "render_scene_from_layout.py"

    def child_environment(
        self,
        stage: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a child environment without ever serializing secret values."""

        source = os.environ if environ is None else environ
        if stage == "perception":
            key_name = self.perception_api_key_env
            base_name = self.perception_api_base_env
        elif stage == "relation":
            key_name = self.relation_api_key_env
            base_name = self.relation_api_base_env
        elif stage == "scenemosaic":
            key_name = self.scenemosaic_api_key_env
            base_name = self.scenemosaic_api_base_env
        else:
            raise RuntimeConfigError(f"unknown API stage: {stage}")
        child = {
            name: value
            for name, value in source.items()
            if not SENSITIVE_ENV_NAME_RE.search(name)
        }
        key_value = source.get(key_name)
        base_value = source.get(base_name)
        if key_value:
            child[key_name] = key_value
            child["OPENAI_API_KEY"] = key_value
        if base_value:
            child[base_name] = base_value
            child["OPEN_BASE_URL"] = base_value
        existing_pythonpath = child.get("PYTHONPATH")
        python_paths: list[str] = []
        if stage == "perception":
            python_paths.append(str(self.sam3_source_root))
            child.update(
                {
                    "SCENEMOSAIC_RUNTIME_ROOT": str(self.workspace_root),
                    "SCENEMOSAIC_PERCEPTION_ATTEMPTS_ROOT": str(
                        self.perception_attempts_root
                    ),
                    "SCENEMOSAIC_PERCEPTION_MODEL_DIR": str(
                        self.perception_model_dir
                    ),
                    "SCENEMOSAIC_API_METERING_ROOT": str(
                        self.workspace_root / "api_metering"
                    ),
                    "SCENEMOSAIC_SAM3_ENV": (
                        self.perception_python.parent.parent.name or "sam3"
                    ),
                }
            )
        elif stage == "relation":
            # Relation subprocesses intentionally use their standalone
            # relations.* imports without importing SceneMosaic's heavy
            # package initializer.  The reconstruction package root supplies
            # that stable top-level namespace in the isolated environment.
            python_paths.append(str(self.reconstruction_root))
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        if python_paths:
            child["PYTHONPATH"] = os.pathsep.join(python_paths)
        child["PYTHONDONTWRITEBYTECODE"] = "1"
        child["PYTHONNOUSERSITE"] = "1"
        return child

    def has_api_credentials(
        self,
        stage: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        source = os.environ if environ is None else environ
        names = {
            "perception": self.perception_api_key_env,
            "relation": self.relation_api_key_env,
            "scenemosaic": self.scenemosaic_api_key_env,
        }
        try:
            return bool(source.get(names[stage]))
        except KeyError as exc:
            raise RuntimeConfigError(f"unknown API stage: {stage}") from exc

    def has_api_base(
        self,
        stage: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        source = os.environ if environ is None else environ
        names = {
            "perception": self.perception_api_base_env,
            "relation": self.relation_api_base_env,
            "scenemosaic": self.scenemosaic_api_base_env,
        }
        try:
            value = source.get(names[stage])
        except KeyError as exc:
            raise RuntimeConfigError(f"unknown API stage: {stage}") from exc
        return bool(value and value.strip() and value.strip() != "None")

    @staticmethod
    def verify_gpu() -> dict[str, Any]:
        """Verify GPU visibility without importing CUDA or starting a model."""

        base = {
            "schema_version": "scenemosaic_gpu_preflight",
            "gpu_used": False,
            "api_calls": 0,
        }
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip().lower() in {
            "",
            "-1",
            "none",
            "void",
        }:
            return {
                **base,
                "valid": False,
                "devices": [],
                "reason": "CUDA_VISIBLE_DEVICES disables all GPUs",
            }
        executable = shutil.which("nvidia-smi")
        if not executable:
            return {
                **base,
                "valid": False,
                "devices": [],
                "reason": "nvidia-smi not found",
            }
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,name,uuid,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                **base,
                "valid": False,
                "devices": [],
                "reason": type(exc).__name__,
            }
        devices: list[dict[str, Any]] = []
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                fields = [item.strip() for item in line.split(",", 3)]
                if len(fields) != 4:
                    continue
                try:
                    memory_mib = int(fields[3])
                except ValueError:
                    continue
                devices.append(
                    {
                        "index": fields[0],
                        "name": fields[1],
                        "uuid": fields[2],
                        "memory_total_mib": memory_mib,
                    }
                )
        return {
            **base,
            "valid": completed.returncode == 0 and bool(devices),
            "devices": devices,
            "reason": None if devices else "no visible NVIDIA GPU",
        }

    def verify(self, *, require_gpu: bool = False) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def record(
            name: str,
            valid: bool,
            detail: Any = None,
            *,
            required: bool = True,
        ) -> None:
            checks.append(
                {
                    "name": name,
                    "valid": bool(valid),
                    "required": required,
                    "detail": detail,
                }
            )

        record("project_root", self.project_root.is_dir(), str(self.project_root))
        # The VLM model name is intentionally not validated here: the profile
        # accepts any non-empty model id so that OpenAI-compatible gateways
        # whose id differs from DEFAULT_VLM_MODEL (e.g. casing) can be used.
        # DEFAULT_VLM_MODEL is still the fallback when the profile omits it.
        expected_attempts_root = (self.workspace_root / "perception_attempts").resolve()
        record(
            "perception_attempts_root_contract",
            self.perception_attempts_root == expected_attempts_root,
            {
                "actual": str(self.perception_attempts_root),
                "expected": str(expected_attempts_root),
            },
        )
        for name, path, kind in (
            ("workspace_root", self.workspace_root, "dir"),
            ("perception_attempts_root", self.perception_attempts_root, "dir"),
            ("perception_python", self.perception_python, "file"),
            ("perception_model_dir", self.perception_model_dir, "dir"),
            ("sam3d_python", self.sam3d_python, "file"),
            ("sam3d_checkpoint", self.sam3d_checkpoint, "file"),
            ("dino_hub_cache", self.dino_hub_cache, "dir"),
            ("relation_precompute_python", self.relation_precompute_python, "file"),
            ("relation_runner_python", self.relation_runner_python, "file"),
            ("scenemosaic_python", self.scenemosaic_python, "file"),
            ("sam3_source_root", self.sam3_source_root, "dir"),
            ("sam3d_source_root", self.sam3d_source_root, "dir"),
            ("renderer", self.renderer_path, "file"),
        ):
            valid = path.is_dir() if kind == "dir" else path.is_file()
            record(name, valid, str(path))

        perception = verify_perception_runtime(self.perception_root)
        record("perception_runtime", perception["valid"], perception)
        relation = verify_relation_runtime(self.relation_package_root)
        record("relation_runtime", relation["valid"], relation)
        sam3_provider = verify_sam3_runtime_provider(
            self.perception_python,
            self.sam3_source_root,
        )
        record(
            "sam3_runtime_provider",
            sam3_provider["valid"],
            sam3_provider,
        )
        perception_dependencies = verify_perception_dependencies(
            self.perception_python
        )
        record(
            "perception_python_dependencies",
            perception_dependencies["valid"],
            perception_dependencies,
        )
        sam3d = verify_sam3d_interface(self.sam3d_source_root)
        record("sam3d_runtime_interface", sam3d["valid"], sam3d)
        renderer_runtime = verify_headless_renderer_runtime(
            self.relation_precompute_python
        )
        record(
            "renderer_egl_runtime",
            renderer_runtime["valid"],
            renderer_runtime,
        )
        renderer = verify_renderer_contract(self.renderer_path)
        record("renderer_nine_functions", renderer["valid"], renderer)

        if require_gpu:
            gpu = self.verify_gpu()
            record("gpu", bool(gpu.get("valid")), gpu)

        return {
            "schema_version": "scenemosaic_runtime_verification",
            "profile_path": str(self.profile_path),
            "project_root": str(self.project_root),
            "api_calls": 0,
            "gpu_used": False,
            "valid": all(item["valid"] for item in checks if item["required"]),
            "scenemosaic_ready": all(item["valid"] for item in checks),
            "checks": checks,
        }


def _verify_required_files(
    package_root: Path,
    *,
    component: str,
    release_id: str,
    required_files: tuple[str, ...],
) -> dict[str, Any]:
    root = Path(package_root).expanduser().resolve()
    failures: list[dict[str, str]] = []
    checked = 0
    for relative in required_files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            failures.append({"path": relative, "reason": "missing_or_symlink"})
        else:
            checked += 1
    return {
        "component": component,
        "release": release_id,
        "release_id": release_id,
        "package_root": str(root),
        "required_file_count": len(required_files),
        "checked_file_count": checked,
        "valid": root.is_dir() and not root.is_symlink() and not failures,
        "failures": failures,
    }


def verify_perception_runtime(package_root: Path) -> dict[str, Any]:
    return _verify_required_files(
        package_root,
        component="perception",
        release_id="scenemosaic-perception",
        required_files=(
            "adoption.py",
            "stage.py",
            "inventory/pipeline.py",
            "pipeline/runner.py",
            "segmentation/engine.py",
            "segmentation/runtime.py",
        ),
    )


def verify_relation_runtime(package_root: Path) -> dict[str, Any]:
    return _verify_required_files(
        package_root,
        component="relations",
        release_id="scenemosaic-relations",
        required_files=(
            "stage.py",
            "graph.py",
            "agents/run_scene_delivery_multi_agent.py",
            "agents/final_assembler.py",
            "agents/schema_guard.py",
            "object_physics/run_sidecar.py",
            "policies/multi_agent_policy.md",
            "prompts/multi_agent/delivery_schema_guard_system.txt",
            "prompts/multi_agent/dynamic_task_agent_compact_system.txt",
            "prompts/multi_agent/dynamic_task_agent_system.txt",
            "prompts/multi_agent/execution_supervisor_system.txt",
            "prompts/multi_agent/scene_delivery_manager_compact_system.txt",
            "prompts/multi_agent/wall_final_adjudication_task.txt",
            "tools/benchmark_relation_async_planning.py",
        ),
    )


def verify_sam3d_interface(source_root: Path) -> dict[str, Any]:
    root = Path(source_root).expanduser().resolve()
    required = (
        "notebook/inference.py",
        "sam3d_objects/__init__.py",
        "sam3d_objects/pipeline/__init__.py",
    )
    report = _verify_required_files(
        root,
        component="sam3d",
        release_id="repository-local-sam-3d-objects",
        required_files=required,
    )
    entrypoint = root / "notebook" / "inference.py"
    interface_valid = False
    if entrypoint.is_file() and not entrypoint.is_symlink():
        try:
            tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
            interface_valid = any(
                isinstance(node, ast.ClassDef) and node.name == "Inference"
                for node in tree.body
            )
        except (OSError, SyntaxError, UnicodeError):
            interface_valid = False
    if not interface_valid:
        report["failures"].append(
            {"path": "notebook/inference.py", "reason": "Inference interface"}
        )
    report["interface"] = "notebook.inference.Inference"
    report["valid"] = bool(report["valid"] and interface_valid)
    return report


def verify_headless_renderer_runtime(python_path: Path) -> dict[str, Any]:
    """Check the native EGL loader used by pyrender without creating a GPU context."""

    executable = Path(python_path).expanduser().resolve()
    report: dict[str, Any] = {
        "valid": False,
        "python": str(executable),
        "platform": "egl",
        "gpu_used": False,
        "api_calls": 0,
    }
    if not executable.is_file():
        report["reason"] = "Python interpreter is missing"
        return report

    probe = """
import ctypes
import ctypes.util
import json

library = ctypes.util.find_library("EGL")
if not library:
    raise RuntimeError("the system dynamic linker cannot find EGL")
ctypes.CDLL(library)
import pyrender
print(json.dumps({"egl_library": library, "pyrender": pyrender.__version__}))
"""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not SENSITIVE_ENV_NAME_RE.search(name)
    }
    environment.update(
        {
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report["reason"] = type(exc).__name__
        return report
    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        report["reason"] = stderr[-1] if stderr else "EGL import probe failed"
        return report
    try:
        detail = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        report["reason"] = "EGL import probe returned invalid output"
        return report
    report.update({"valid": True, **detail})
    return report


def verify_renderer_contract(renderer_path: Path) -> dict[str, Any]:
    if not renderer_path.is_file():
        return {"valid": False, "missing": list(RENDERER_FUNCTIONS)}
    try:
        tree = ast.parse(renderer_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {"valid": False, "missing": list(RENDERER_FUNCTIONS)}
    signatures = {
        node.name: tuple(
            argument.arg for argument in (*node.args.posonlyargs, *node.args.args)
        )
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [
        name
        for name, expected in RENDERER_FUNCTIONS.items()
        if name not in signatures or signatures[name][: len(expected)] != expected
    ]
    return {
        "valid": not missing,
        "missing_or_incompatible": missing,
        "required_signatures": {
            name: list(arguments) for name, arguments in RENDERER_FUNCTIONS.items()
        },
        "signatures": {name: list(arguments) for name, arguments in signatures.items()},
    }
