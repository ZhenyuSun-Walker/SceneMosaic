#!/usr/bin/env python3
"""Run one TD_mansion image through SceneMosaic and export a matching dataset."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def wait_for_gpu(index: int, required_gib: float, state_path: Path, state: dict) -> None:
    while True:
        result = subprocess.run(["nvidia-smi", f"--id={index}",
                                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        available = float(result.stdout.strip()) / 1024
        if available >= required_gib:
            return
        state.update(status="waiting_for_gpu", free_gpu_gib=available,
                     required_gpu_gib=required_gib, gpu_index=index)
        write_json(state_path, state)
        print(f"GPU {index}: {available:.1f} GiB free; waiting for {required_gib:.1f} GiB.", flush=True)
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/austin_sun/Pi3_early_fusion/data/TD_mansion"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/austin_sun/Pi3_early_fusion/data/TD_MANSION_mosaic"))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs/runtime.local.json")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--required-gpu-gib", type=float, default=40.)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.scene):
        parser.error("Scene identifier contains invalid characters")
    source = args.source_root.resolve() / args.scene
    image = source / "rgb_td/rgb_td.png"
    if not image.is_file():
        parser.error(f"Input not found: {image}")
    output = args.output_root.resolve() / args.scene
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".generation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "render_manifest.json").is_file():
        print(f"Complete dataset already exists: {output}")
        return 0
    workspace = ROOT / "workspace/td_mansion_mosaic"
    config_path = workspace / "runtime.json"
    original_path = args.runtime_config.resolve()
    config = json.loads(original_path.read_text())
    for name, value in list(config.items()):
        if isinstance(value, str) and value.startswith("../"):
            config[name] = str((original_path.parent / value).resolve())
    config.update(workspace_root=str(workspace),
                  perception_attempts_root=str(workspace / "perception_attempts"))
    write_json(config_path, config)
    (workspace / "perception_attempts").mkdir(parents=True, exist_ok=True)
    (output / "source_input").mkdir(exist_ok=True)
    shutil.copy2(image, output / "source_input/rgb_td.png")
    state_path = output / "generation_status.json"
    state = {"scene": args.scene, "source_image": str(image), "output_scene": str(output),
             "runtime_config": str(config_path), "vlm_model": config["vlm_model"],
             "run_root": str(workspace / "runs" / args.scene), "status": "prepared"}
    write_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)
    if args.prepare_only:
        return 0
    if not os.environ.get(config.get("perception_api_key_env", "OPENAI_API_KEY")):
        state.update(status="blocked", reason="OPENAI_API_KEY is unset in this shell")
        write_json(state_path, state)
        raise RuntimeError(state["reason"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        # Publish perception first; the full orchestrator reuses its validated artifacts.
        wait_for_gpu(args.gpu, 8., state_path, state)
        from vlm_agent.config import RuntimeConfig
        from vlm_agent.artifacts import initialize_scene_run, load_scene_run
        from vlm_agent.reconstruction.perception.stage import PerceptionStage
        runtime = RuntimeConfig.load(config_path)
        run_root = Path(state["run_root"])
        run = (load_scene_run(run_root) if (run_root / "pipeline_state.json").is_file()
               else initialize_scene_run(runtime, scene_name=args.scene, input_image=image))
        state.update(status="perception_running")
        write_json(state_path, state)
        perception = PerceptionStage(runtime, run.contract.scene_config_path)
        perception.run(approve_gpu=True, approve_api=True, approve_third_party_upload=True)
        state.update(status="perception_complete")
        write_json(state_path, state)
        wait_for_gpu(args.gpu, args.required_gpu_gib, state_path, state)
        state.update(status="scene_generation_running")
        write_json(state_path, state)
        subprocess.run([sys.executable, "-m", "vlm_agent.cli", "--runtime-config", str(config_path),
                        "run", "--scene-root", str(run_root)], cwd=ROOT, check=True)
        glb = run_root / "outputs/representative_scenes/scene_01.glb"
        if not glb.is_file():
            raise FileNotFoundError(f"No published scene: {glb}")
        state.update(status="rendering_dataset", generated_glb=str(glb))
        write_json(state_path, state)
        subprocess.run([sys.executable, str(ROOT / "scripts/export_mosaic_dataset.py"),
                        "--glb", str(glb), "--source-scene", str(source),
                        "--output-dir", str(output)], cwd=ROOT, check=True)
        state.update(status="complete")
        write_json(state_path, state)
        print(f"Complete TD_mansion-compatible generated scene: {output}", flush=True)
        return 0
    except Exception as exc:
        # CLI error output is already redacted; do not serialize arbitrary exception text.
        state.update(status="failed", error_type=type(exc).__name__)
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
