#!/usr/bin/env python3
"""Run independent object-physics enrichment beside or after relation extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from relations.agents.api_client import APIClientError, APIConfig, SceneDeliveryAPIClient
from relations.object_physics.core import resolve_object_physics, validate_object_physics, validate_object_upright_priors


PROMPT_PATH = Path(__file__).with_name("object_physics_agent_system.txt")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def scene_objects(scene_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = scene_state.get("object_candidates") if isinstance(scene_state, dict) else None
    items = candidates.get("items", []) if isinstance(candidates, dict) else []
    return {
        str(item["object_id"]): item
        for item in items
        if isinstance(item, dict) and item.get("object_id")
    }


def original_image_path(
    run_root: Path,
    scene_state: dict[str, Any],
    expected_manifest: dict[str, Any] | None = None,
) -> Path | None:
    evidence = scene_state.get("evidence") if isinstance(scene_state, dict) else None
    candidate = evidence.get("original_image_path") if isinstance(evidence, dict) else None
    paths = [Path(candidate)] if candidate else []
    artifacts = expected_manifest.get("artifacts", {}) if isinstance(expected_manifest, dict) else {}
    original = artifacts.get("original_image") if isinstance(artifacts, dict) else None
    manifest_candidate = original.get("path") if isinstance(original, dict) else original
    if manifest_candidate:
        paths.append(Path(manifest_candidate))
    paths.extend(run_root / "input" / name for name in ("image.png", "image.jpg", "image.jpeg"))
    return next((path for path in paths if path.exists()), None)


def objects_from_expected_manifest(expected_manifest: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for item in expected_manifest.get("objects", []):
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id") or item.get("id")
        if not object_id:
            continue
        obj = {
            "id": str(object_id),
            "category": item.get("category") or "unknown_object",
            "description": item.get("description") or item.get("category") or "unknown object",
        }
        bbox_2d = item.get("vlm_bbox_2d_norm_xyxy") or item.get("bbox_2d")
        if isinstance(bbox_2d, list) and len(bbox_2d) >= 4:
            obj["bbox_2d_norm_xyxy"] = bbox_2d[:4]
        objects.append(obj)
    return {"walls": [], "objects": objects}


def build_agent_payload(scene_name: str, objects_payload: dict[str, Any], scene_state: dict[str, Any]) -> dict[str, Any]:
    context = scene_objects(scene_state)
    objects = []
    for obj in objects_payload.get("objects", []):
        object_id = str(obj.get("id"))
        scene_obj = context.get(object_id, {})
        bbox = scene_obj.get("bbox_3d") if isinstance(scene_obj.get("bbox_3d"), dict) else {}
        bbox_2d = obj.get("bbox_2d_norm_xyxy") or scene_obj.get("bbox_2d")
        item = {
            "object_id": object_id,
            "category": obj.get("category"),
            "description": str(obj.get("description") or "")[:240],
            "bbox_extent_xyz": bbox.get("extent"),
        }
        if isinstance(bbox_2d, list) and len(bbox_2d) >= 4:
            try:
                width = abs(float(bbox_2d[2]) - float(bbox_2d[0]))
                height = abs(float(bbox_2d[3]) - float(bbox_2d[1]))
                item["original_image_bbox_xyxy"] = [round(float(value), 6) for value in bbox_2d[:4]]
                item["image_bbox_height_width_ratio"] = round(height / width, 6) if width > 1e-6 else None
            except (TypeError, ValueError):
                pass
        objects.append(item)
    return {
        "schema_version": "object_physics_request_compact",
        "scene_name": scene_name,
        "task": "Estimate independent object simulation-physics priors and one relation-agnostic upright prior. Do not reason about attach, contain, walls, or object identity.",
        "objects": objects,
        "required_fields": ["upright", "mass", "linear_damping", "angular_damping", "static_friction", "dynamic_friction"],
        "response_policy": "Return object_id, one boolean upright field, and exactly the five required numeric fields for every supplied object. Return no reasons, confidence, prose, or relation data.",
    }


def sidecar_publish_eligible(validation_errors: list[str], upright_validation_errors: list[str]) -> bool:
    """Return whether this sidecar is complete enough for downstream publication."""
    return not validation_errors and not upright_validation_errors


def write_report(output_dir: Path, runtime: dict[str, Any], audit: dict[str, Any], errors: list[str]) -> None:
    lines = [
        "# Object Physics Sidecar Report",
        "",
        f"- status: `{runtime['status']}`",
        f"- model: `{runtime['model']}`",
        f"- relation_critical_path_dependency: `false`",
        f"- object_count: `{audit.get('object_count', 0)}`",
        f"- fallback_only_object_count: `{audit.get('fallback_only_object_count', 0)}`",
        f"- validation_errors: `{len(errors)}`",
        f"- upright_validation_errors: `{len(runtime.get('upright_validation_errors', []))}`",
        f"- publish_eligible: `{str(bool(runtime.get('publish_eligible'))).lower()}`",
        "",
        "The sidecar subprocess never writes relation final outputs. Final assembly may consume its ready upright/physics priors by object id without waiting; otherwise local fallbacks are used.",
    ]
    (output_dir / "scene_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--agent_run_dir", type=Path)
    parser.add_argument("--expected_manifest", type=Path)
    parser.add_argument("--sidecar_id", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--vision_model", default="gpt-5.5")
    parser.add_argument("--api_base")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--vision_image_detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--include_original", action="store_true")
    parser.add_argument("--decision_json", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.agent_run_dir and not args.expected_manifest:
        raise SystemExit("Provide either --agent_run_dir or --expected_manifest")
    if args.output_dir:
        output_dir = args.output_dir
    elif args.agent_run_dir:
        output_dir = args.agent_run_dir / "physics_sidecar" / args.sidecar_id
    else:
        raise SystemExit("--output_dir is required with --expected_manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_manifest = read_json(args.expected_manifest) if args.expected_manifest else None
    if args.expected_manifest and not isinstance(expected_manifest, dict):
        raise SystemExit("--expected_manifest must contain a JSON object")
    if args.expected_manifest:
        objects_path = args.expected_manifest
        relations_path = None
        scene_state_path = None
        objects_payload = objects_from_expected_manifest(expected_manifest)
        scene_state = {}
        input_mode = "expected_manifest_async"
    else:
        objects_path = args.agent_run_dir / "final_outputs" / "objects.json"
        relations_path = args.agent_run_dir / "final_outputs" / "relations.json"
        scene_state_path = args.agent_run_dir / "scene_states" / "scene_state.json"
        if not objects_path.exists() or not relations_path.exists():
            raise SystemExit("Physics sidecar requires completed relation final_outputs/objects.json and relations.json")
        objects_payload = read_json(objects_path)
        scene_state = read_json(scene_state_path) if scene_state_path.exists() else {}
        input_mode = "completed_relation_outputs"
    if not objects_payload.get("objects"):
        raise SystemExit("Physics sidecar input contains no objects")
    image_path = original_image_path(args.run_root, scene_state, expected_manifest)

    agent_payload: dict[str, Any] = {}
    status = "fallback_only"
    api_error = None
    retry_events: list[dict[str, Any]] = []
    if args.decision_json:
        agent_payload = read_json(args.decision_json)
        status = "decision_json"
    elif not args.dry_run:
        client = SceneDeliveryAPIClient(APIConfig(
            model=args.model,
            vision_model=args.vision_model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            dry_run=False,
            image_detail=args.vision_image_detail,
        ))
        api_request_payload = build_agent_payload(args.scene_name, objects_payload, scene_state)
        try:
            images = [str(image_path)] if args.include_original and image_path else None
            agent_payload = client.call_json(
                role_name="ObjectPhysicsEstimator",
                system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
                user_payload=api_request_payload,
                use_vision_model=bool(images),
                image_paths=images,
            )
            status = "agent_completed"
        except APIClientError as exc:
            status = "fallback_after_api_error"
            api_error = str(exc)
            retry_events = exc.retry_events

    sidecar, enriched, audit = resolve_object_physics(objects_payload, agent_payload)
    expected_ids = [str(obj.get("id")) for obj in objects_payload.get("objects", [])]
    validation_errors = validate_object_physics(sidecar, expected_ids)
    upright_validation_errors = validate_object_upright_priors(sidecar, expected_ids)
    upright_validation_errors.extend(
        f"{object_id} agent upright must be a JSON boolean"
        for object_id in audit.get("invalid_agent_upright_object_ids", [])
        if f"{object_id} agent upright must be a JSON boolean" not in upright_validation_errors
    )
    publish_eligible = sidecar_publish_eligible(validation_errors, upright_validation_errors)
    if validation_errors:
        status = "validation_failed"
    elif upright_validation_errors and status in {"agent_completed", "decision_json"}:
        status = f"{status}_with_incomplete_upright"

    write_json(output_dir / "agent_response.json", agent_payload)
    write_json(output_dir / "object_physics.json", sidecar)
    write_json(output_dir / "objects_with_physics.json", enriched)
    write_json(output_dir / "physics_audit.json", audit)
    runtime = {
        "status": status,
        "scene_name": args.scene_name,
        "sidecar_id": args.sidecar_id,
        "model": args.vision_model if args.include_original else args.model,
        "include_original": bool(args.include_original and image_path),
        "relation_critical_path_dependency": False,
        "original_relation_outputs_modified": False,
        "input_mode": input_mode,
        "input_objects_path": str(objects_path),
        "input_relations_path": str(relations_path) if relations_path else None,
        "original_image_path": str(image_path) if image_path else None,
        "api_error": api_error,
        "retry_events": retry_events,
        "validation_errors": validation_errors,
        "upright_validation_errors": upright_validation_errors,
        "publish_eligible": publish_eligible,
    }
    write_json(output_dir / "runtime.json", runtime)
    write_report(output_dir, runtime, audit, validation_errors)
    print(json.dumps({
        "status": status,
        "output_dir": str(output_dir),
        "object_count": audit.get("object_count"),
        "fallback_only_object_count": audit.get("fallback_only_object_count"),
        "validation_errors": validation_errors,
        "upright_validation_errors": upright_validation_errors,
        "publish_eligible": publish_eligible,
    }, ensure_ascii=False, indent=2))
    if not publish_eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

