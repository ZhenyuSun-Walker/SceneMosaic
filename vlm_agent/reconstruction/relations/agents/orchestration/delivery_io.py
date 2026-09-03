from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from relations.agents.state import write_audit_pair, write_json, write_md

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_agent.run_paths import ScenePaths

RELATIONS_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = RELATIONS_ROOT / "prompts" / "multi_agent"
POLICY_PATH = RELATIONS_ROOT / "policies" / "multi_agent_policy.md"
RENDER_GRID_WALL_PRECOMPUTE_PATH_ENV = "RENDER_GRID_WALL_PRECOMPUTE_PATH"
RELATION_LAYOUT_DIR_ENV = "RELATION_LAYOUT_DIR"


def configure_wall_candidate_environment(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(getattr(args, "wall_candidate_mode", "render_grid_only") or "render_grid_only").strip().lower()
    if mode != "render_grid_only":
        raise ValueError(f"unsupported wall candidate mode for the formal pipeline: {mode}")
    return {
        "wall_candidate_mode": mode,
        "render_grid_wall_precompute_path_env": RENDER_GRID_WALL_PRECOMPUTE_PATH_ENV,
    }

def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def write_manager_plan(output_dir: Path, plan: dict[str, Any]) -> None:
    write_json(output_dir / "manager_plans" / "manager_plan_000.json", plan)
    lines = [
        f"- dry_run: `{plan.get('dry_run', False)}`",
        f"- objective: {plan.get('objective', '')}",
        f"- task_count: `{len(plan.get('tasks', []))}`",
        "",
        "## Tasks",
    ]
    for task in plan.get("tasks", []):
        lines.append(f"- `{task.get('task_id')}`: {task.get('task_type')} ({task.get('agent_role')})")
    write_md(output_dir / "manager_plans" / "manager_plan_000.md", "Manager Plan 000", lines)


def write_task_output(output_dir: Path, task: dict[str, Any], response: dict[str, Any]) -> None:
    task_dir = output_dir / "agent_outputs" / task.get("task_id", "task_0001")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "prompt.txt").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_json(task_dir / "response.json", response)
    lines = [
        f"- task_id: `{response.get('task_id', task.get('task_id'))}`",
        f"- status: `{response.get('status')}`",
        f"- dry_run: `{response.get('dry_run', False)}`",
        "",
        "## Findings",
    ]
    for item in response.get("findings", []):
        lines.append(f"- {item}")
    write_md(task_dir / "response.md", "Dynamic Task Response", lines)


def load_static_policy() -> dict[str, Any]:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"production multi-agent policy is missing: {POLICY_PATH}")
    return {
        "status": "static_production_policy",
        "output_path": str(POLICY_PATH),
        "source": "relations/policies/multi_agent_policy.md",
    }


def copy_policy_to_run(output_dir: Path) -> None:
    target = output_dir / "policy" / "multi_agent_policy.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(POLICY_PATH, target)


def write_dry_run_final_marker(output_dir: Path) -> None:
    marker = output_dir / "final_outputs" / "DRY_RUN_NO_FINAL_OUTPUTS.md"
    marker.write_text(
        "# Dry Run Final Outputs\n\n"
        "No real `objects.json` or `relations.json` was generated because `--dry_run` was enabled.\n"
        "Remove `--dry_run` and provide a valid API key environment variable to request real agent output.\n"
        "Final real outputs must preserve exactly the object ids from scene_state.object_candidates; object numbering is inherited from SAM3D/layout and may be 0-based or 1-based.\n",
        encoding="utf-8",
    )


def update_agent_audits(output_dir: Path, plan: dict[str, Any], task_results: list[dict[str, Any]], supervisor_result: dict[str, Any]) -> None:
    write_audit_pair(
        output_dir,
        "agents",
        "agent_task_log",
        {
            "dry_run": plan.get("dry_run", False),
            "manager_plan_id": plan.get("plan_id"),
            "tasks": plan.get("tasks", []),
            "task_results": task_results,
        },
        "Agent Task Log",
    )
    write_audit_pair(output_dir, "supervisor", "supervisor_log", supervisor_result, "Supervisor Log")


def write_render_audit(output_dir: Path, render_capabilities: dict[str, Any], dry_run: bool) -> None:
    write_audit_pair(
        output_dir,
        "rendering",
        "topdown_render_log",
        {
            "dry_run": dry_run,
            "status": "capability_scan_completed",
            "render_capabilities": render_capabilities,
            "render_calls": [],
        },
        "Topdown Render Log",
    )


def _load_json_file(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def relation_layout_manifest_candidates(run_root: Path | None) -> list[Path]:
    explicit_dir = os.environ.get(RELATION_LAYOUT_DIR_ENV)
    if explicit_dir:
        return [Path(explicit_dir).expanduser().resolve() / "manifest.json"]
    if not run_root:
        return []
    default_dir = ScenePaths.from_scene_root(run_root).relations_dir / "relation_layout"
    return [default_dir / "manifest.json"]


def relation_layout_expected_manifest_path(run_root: Path | None) -> Path | None:
    explicit_dir = os.environ.get(RELATION_LAYOUT_DIR_ENV)
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve() / "expected_manifest.json"
    if not run_root:
        return None
    default_dir = ScenePaths.from_scene_root(run_root).relations_dir / "relation_layout"
    return default_dir / "expected_manifest.json"


def load_relation_layout_manifest(run_root: Path | None) -> dict[str, Any] | None:
    for manifest_path in relation_layout_manifest_candidates(run_root):
        manifest = _load_json_file(manifest_path)
        if isinstance(manifest, dict):
            manifest = dict(manifest)
            manifest["manifest_path"] = str(manifest_path)
            manifest["manifest_source"] = "relation_layout"
            return manifest
    return None


def _relation_precompute_outputs_from_run_root(run_root: Path | None) -> dict[str, Any]:
    manifest = load_relation_layout_manifest(run_root)
    if not isinstance(manifest, dict):
        return {}
    outputs = manifest.get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def wait_for_relation_precompute(args: argparse.Namespace, *, poll_seconds: float = 1.0) -> dict[str, Any] | None:
    wait_seconds = float(getattr(args, "relation_precompute_wait_seconds", 0) or 0)
    if wait_seconds <= 0:
        return load_render_grid_wall_precompute(args)
    deadline = time.time() + wait_seconds
    latest: dict[str, Any] | None = None
    while time.time() <= deadline:
        latest = load_render_grid_wall_precompute(args)
        if isinstance(latest, dict) and latest.get("status") == "available":
            return latest
        time.sleep(max(0.1, poll_seconds))
    return latest


def load_render_grid_wall_precompute(args: argparse.Namespace) -> dict[str, Any] | None:
    raw_path = getattr(args, "render_grid_wall_precompute_path", None)
    raw_dir = getattr(args, "render_grid_wall_precompute_dir", None)
    run_root = Path(getattr(args, "run_root", "")).resolve() if getattr(args, "run_root", None) else None
    relation_outputs = _relation_precompute_outputs_from_run_root(run_root)
    precompute_dir: Path | None = None
    method_candidates_path: Path | None = None
    if raw_path:
        path = Path(raw_path).resolve()
        if path.is_dir():
            precompute_dir = path
            method_candidates_path = path / "method_candidates.json"
        else:
            method_candidates_path = path
            precompute_dir = path.parent
    elif raw_dir:
        precompute_dir = Path(raw_dir).resolve()
        method_candidates_path = precompute_dir / "method_candidates.json"
    elif relation_outputs.get("method_candidates_path"):
        method_candidates_path = Path(str(relation_outputs["method_candidates_path"])).resolve()
        precompute_dir = Path(str(relation_outputs.get("render_grid_wall_precompute_dir") or method_candidates_path.parent)).resolve()
    if not method_candidates_path:
        return None

    manifest_path = precompute_dir / "render_grid_wall_precompute_manifest.json" if precompute_dir else None
    manifest = _load_json_file(manifest_path) or {}
    method_data = _load_json_file(method_candidates_path)
    if not isinstance(method_data, dict) or not isinstance(method_data.get("candidates"), list):
        return {
            "status": "missing",
            "reason": f"method_candidates.json missing or invalid: {method_candidates_path}",
            "precompute_dir": str(precompute_dir) if precompute_dir else None,
            "method_candidates_path": str(method_candidates_path),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "loaded_from_external_precompute": True,
        }

    def pick(key: str, filename: str | None = None) -> str | None:
        value = manifest.get(key)
        if value:
            return str(value)
        if precompute_dir and filename:
            path = precompute_dir / filename
            return str(path) if path.exists() else str(path)
        return None

    return {
        "status": "available",
        "scene_name": manifest.get("scene_name") or method_data.get("scene"),
        "precompute_dir": str(precompute_dir) if precompute_dir else None,
        "method_candidates_path": str(method_candidates_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "canonical_topdown_image_path": pick("canonical_topdown_image_path", "canonical_topdown.png"),
        "canonical_mapping_path": pick("canonical_mapping_path", "canonical_topdown_grid_mapping.json"),
        "render_grid_affine_path": pick("render_grid_affine_path", "render_grid_affine.json"),
        "render_foreground_mask_path": pick("render_foreground_mask_path", "render_foreground_mask.png"),
        "candidate_overlay_path": pick("candidate_overlay_path", "candidate_overlay.png"),
        "wall_orientation_candidates_overlay_path": pick("wall_orientation_candidates_overlay_path", "wall_orientation_candidates_overlay.png"),
        "primary_candidate_overlay_path": pick("primary_candidate_overlay_path", "primary_candidate_overlay.png"),
        "primary_method": manifest.get("primary_method") or method_data.get("method_policy", {}).get("primary_method"),
        "supporting_methods": manifest.get("supporting_methods") or method_data.get("method_policy", {}).get("supporting_methods"),
        "same_grid_policy": manifest.get("same_grid_policy") or method_data.get("renderer", {}).get("same_grid_policy"),
        "loaded_from_external_precompute": True,
        "reused_existing": True,
        "relation_precompute_manifest_path": (load_relation_layout_manifest(run_root) or {}).get("manifest_path") if run_root else None,
        "loaded_from_relation_precompute": bool(relation_outputs),
    }


def apply_render_grid_precompute_to_scene_state(scene_state: dict[str, Any], precompute: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(precompute, dict) or precompute.get("status") not in {"available", "missing_primary"}:
        return scene_state
    method_candidates_path = precompute.get("method_candidates_path")
    canonical_mapping = precompute.get("canonical_mapping_path")
    canonical_topdown = precompute.get("canonical_topdown_image_path")
    if method_candidates_path:
        os.environ[RENDER_GRID_WALL_PRECOMPUTE_PATH_ENV] = str(method_candidates_path)
    if canonical_mapping:
        scene_state["topdown_grid_mapping"] = {
            "canonical_mapping_path": str(canonical_mapping),
            "canonical_topdown_image_path": str(canonical_topdown) if canonical_topdown else None,
            "source": "render_grid_wall_precompute",
            "fallback_used": False,
            "status": "available",
            "render_grid_affine_path": precompute.get("render_grid_affine_path"),
            "method_candidates_path": method_candidates_path,
        }
    scene_state.setdefault("evidence", {})["render_grid_wall_precompute"] = precompute
    return scene_state


def fixed_views_precompute_dir_arg(args: argparse.Namespace) -> Path | None:
    explicit = getattr(args, "fixed_views_precompute_dir", None)
    if explicit:
        return Path(explicit).resolve()
    run_root = Path(getattr(args, "run_root", "")).resolve() if getattr(args, "run_root", None) else None
    relation_outputs = _relation_precompute_outputs_from_run_root(run_root)
    fixed_dir = relation_outputs.get("fixed_views_dir") if isinstance(relation_outputs, dict) else None
    fixed_manifest = relation_outputs.get("fixed_views_manifest_path") if isinstance(relation_outputs, dict) else None
    if fixed_dir and fixed_manifest and Path(str(fixed_manifest)).exists():
        return Path(str(fixed_dir)).resolve()
    return None




def wait_for_fixed_views_precompute_dir(args: argparse.Namespace, *, poll_seconds: float = 1.0) -> Path | None:
    initial = fixed_views_precompute_dir_arg(args)
    if initial:
        return initial
    wait_seconds = float(getattr(args, "fixed_views_precompute_wait_seconds", getattr(args, "relation_precompute_wait_seconds", 0)) or 0)
    if wait_seconds <= 0:
        return None
    deadline = time.time() + wait_seconds
    while time.time() <= deadline:
        current = fixed_views_precompute_dir_arg(args)
        if current:
            return current
        time.sleep(max(0.1, poll_seconds))
    return fixed_views_precompute_dir_arg(args)
