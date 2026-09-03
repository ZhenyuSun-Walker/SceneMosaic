#!/usr/bin/env python3
"""Run asynchronous skeleton manager planning with an isolated layout cache."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from vlm_agent.run_paths import ScenePaths

RELATIONS_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTE_SCRIPT = RELATIONS_ROOT / 'tools' / 'build_relation_layout_from_sam3d.py'
RUNNER_SCRIPT = RELATIONS_ROOT / 'agents' / 'run_scene_delivery_multi_agent.py'
DEFAULT_RENDER_SCRIPT = CODE_ROOT / 'tools' / 'render_scene_from_layout.py'
DEFAULT_PRECOMPUTE_PYTHON = Path(os.environ.get('SCENEMOSAIC_RELATION_PRECOMPUTE_PYTHON', sys.executable))
DEFAULT_BASE_MODEL = 'gpt-5.5'
DEFAULT_WALL_SPECIALIST_MODEL = 'gpt-5.5'
DEFAULT_WALL_SPECIALIST_TASK_KINDS = [
    'wall_alignment_object_review',
    'provisional_wall_visibility_review',
    'wall_visibility_semantics_review',
    'wall_final_adjudication',
    'wall_attach_candidate_recall',
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')


def read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def run_logged(name: str, cmd: list[str], log_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f'{name}.stdout.log'
    stderr_path = log_dir / f'{name}.stderr.log'
    cmd_path = log_dir / f'{name}.cmd.txt'
    cmd_path.write_text(shlex.join(cmd) + '\n', encoding='utf-8')
    with stdout_path.open('w', encoding='utf-8') as out, stderr_path.open('w', encoding='utf-8') as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env)
    return {
        'name': name,
        'cmd': cmd,
        'cmd_text': shlex.join(cmd),
        'returncode': proc.returncode,
        'stdout_path': str(stdout_path),
        'stderr_path': str(stderr_path),
        'cmd_path': str(cmd_path),
    }


def start_logged(name: str, cmd: list[str], log_dir: Path, env: dict[str, str] | None = None) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f'{name}.stdout.log'
    stderr_path = log_dir / f'{name}.stderr.log'
    cmd_path = log_dir / f'{name}.cmd.txt'
    cmd_path.write_text(shlex.join(cmd) + '\n', encoding='utf-8')
    out = stdout_path.open('w', encoding='utf-8')
    err = stderr_path.open('w', encoding='utf-8')
    proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, env=env)
    meta = {
        'name': name,
        'cmd': cmd,
        'cmd_text': shlex.join(cmd),
        'stdout_path': str(stdout_path),
        'stderr_path': str(stderr_path),
        'cmd_path': str(cmd_path),
        '_stdout_handle': out,
        '_stderr_handle': err,
    }
    return proc, meta


def finish_logged(proc: subprocess.Popen[str], meta: dict[str, Any]) -> dict[str, Any]:
    returncode = proc.wait()
    meta['_stdout_handle'].close()
    meta['_stderr_handle'].close()
    result = {k: v for k, v in meta.items() if not k.startswith('_')}
    result.update({'returncode': returncode})
    return result


def precompute_cmd(args: argparse.Namespace, output_dir: Path, *, expected_only: bool = False) -> list[str]:
    cmd = [
        str(args.precompute_python),
        str(PRECOMPUTE_SCRIPT),
        '--scene_name', args.scene_name,
        '--run_root', str(args.run_root),
        '--output_dir', str(output_dir),
        '--grid_size', str(args.grid_size),
        '--image_size', str(args.image_size),
    ]
    if expected_only:
        cmd.append('--expected_only')
    return cmd


def runner_cmd(args: argparse.Namespace, output_dir: Path, relation_layout_dir: Path) -> list[str]:
    cmd = [
        str(args.runner_python),
        str(RUNNER_SCRIPT),
        '--scene_name', args.scene_name,
        '--run_root', str(args.run_root),
        '--model', args.model,
        '--vision_model', args.vision_model,
    ]
    if args.wall_specialist_model:
        cmd.extend(['--wall_specialist_model', args.wall_specialist_model])
    if args.wall_specialist_vision_model:
        cmd.extend(['--wall_specialist_vision_model', args.wall_specialist_vision_model])
    if args.manager_model:
        cmd.extend(["--manager_model", args.manager_model])
    if args.manager_vision_model:
        cmd.extend(["--manager_vision_model", args.manager_vision_model])
    if args.manager_vision_image_detail:
        cmd.extend(["--manager_vision_image_detail", args.manager_vision_image_detail])
    if args.wall_specialist_task_kinds:
        cmd.extend(['--wall_specialist_task_kinds', *args.wall_specialist_task_kinds])
    cmd.extend([
        '--render_script', str(args.render_script),
        '--output_dir', str(output_dir),
        '--relation_layout_dir', str(relation_layout_dir),
        '--wall_candidate_mode', 'render_grid_only',
        '--max_parallel_tasks', str(args.max_parallel_tasks),
        '--relation_precompute_wait_seconds', str(args.relation_precompute_wait_seconds),
        '--fixed_views_precompute_wait_seconds', '0.0',
    ])
    if args.api_base:
        cmd.extend(['--api_base', args.api_base])
    if args.api_key_env:
        cmd.extend(['--api_key_env', args.api_key_env])
    if args.vision_image_detail:
        cmd.extend(['--vision_image_detail', args.vision_image_detail])
    if args.physics_model:
        cmd.extend(['--physics_model', args.physics_model])
    if args.physics_include_original:
        cmd.append('--physics_include_original')
    if args.dry_run:
        cmd.append('--dry_run')
    return cmd


def collect_run_metrics(agent_dir: Path, relation_layout_dir: Path) -> dict[str, Any]:
    runtime = read_json_if_exists(agent_dir / 'debug' / 'agents' / 'runtime_summary.json')
    rel_manifest = read_json_if_exists(relation_layout_dir / 'manifest.json')
    expected = read_json_if_exists(relation_layout_dir / 'expected_manifest.json')
    manager_plan = read_json_if_exists(agent_dir / 'manager_plans' / 'manager_plan_000.json')
    skeleton_plan = read_json_if_exists(agent_dir / 'manager_plans' / 'manager_plan_skeleton_000.json')
    compact_plan_path = agent_dir / 'manager_plans' / 'manager_plan_compact_000.json'
    compilation_audit_path = agent_dir / 'manager_plans' / 'manager_plan_compilation_audit.json'
    return {
        'agent_runtime_summary_path': str(agent_dir / 'debug' / 'agents' / 'runtime_summary.json'),
        'agent_api_calls': runtime.get('api_calls') if isinstance(runtime, dict) else None,
        'relation_layout_manifest_path': str(relation_layout_dir / 'manifest.json'),
        'relation_layout_stages': rel_manifest.get('stages') if isinstance(rel_manifest, dict) else None,
        'expected_manifest_status': expected.get('status') if isinstance(expected, dict) else None,
        'manager_planning_mode': manager_plan.get('manager_planning_mode') if isinstance(manager_plan, dict) else None,
        'manager_plan_format': manager_plan.get('task_model_routing', {}).get('manager_plan_format') if isinstance(manager_plan, dict) and isinstance(manager_plan.get('task_model_routing'), dict) else None,
        'manager_compact_plan_path': str(compact_plan_path),
        'manager_compact_plan_exists': compact_plan_path.exists(),
        'manager_compilation_audit_path': str(compilation_audit_path),
        'manager_compilation_audit_exists': compilation_audit_path.exists(),
        'task_model_routing': manager_plan.get('task_model_routing') if isinstance(manager_plan, dict) else None,
        'manager_task_count': len(manager_plan.get('tasks', [])) if isinstance(manager_plan, dict) and isinstance(manager_plan.get('tasks'), list) else None,
        'skeleton_task_count': len(skeleton_plan.get('tasks', [])) if isinstance(skeleton_plan, dict) and isinstance(skeleton_plan.get('tasks'), list) else None,
        'final_relations_path': str(agent_dir / 'final_outputs' / 'relations.json'),
        'final_relations_exists': (agent_dir / 'final_outputs' / 'relations.json').exists(),
        'final_objects_path': str(agent_dir / 'final_outputs' / 'objects.json'),
        'final_objects_exists': (agent_dir / 'final_outputs' / 'objects.json').exists(),
        'physics_sidecar_dir': str(agent_dir / 'physics_sidecar' / 'integrated_async'),
        'physics_sidecar_runtime': read_json_if_exists(agent_dir / 'physics_sidecar' / 'integrated_async' / 'runtime.json'),
        'physics_sidecar_integration_status': read_json_if_exists(agent_dir / 'physics_sidecar' / 'integrated_async' / 'integration_status.json'),
        'simulation_physics_audit': read_json_if_exists(agent_dir / 'debug' / 'objects' / 'simulation_physics_audit.json'),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    async_ = summary.get('async', {})
    lines = [
        '# Relation Async Planning Benchmark',
        '',
        f"- scene_name: `{summary.get('scene_name')}`",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- dry_run: `{summary.get('dry_run')}`",
        '',
        '## Wall-clock Results',
        '',
        '| Branch | return codes |',
        '|---|---|',
    ]
    commands = async_.get('commands', {})
    rcs = []
    for key, item in commands.items():
        if isinstance(item, dict):
            rcs.append(f"{key}={item.get('returncode')}")
    lines.append(f"| `async` | {'; '.join(rcs)} |")
    lines.extend(['', '## Notes', ''])
    lines.append('Async writes `expected_manifest.json`, starts full preprocessing and the runner concurrently, then hydrates the manager DAG after artifacts appear. The runner also launches Physics Sidecar beside manager planning; relation tasks never wait for it.')
    lines.extend(['', '## Output Directories', ''])
    lines.append(f"- async relation_layout_dir: `{async_.get('relation_layout_dir')}`")
    lines.append(f"- async agent_dir: `{async_.get('agent_dir')}`")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def benchmark_has_failed_commands(summary: dict[str, Any]) -> bool:
    branch = summary.get("async")
    if not isinstance(branch, dict):
        return True
    commands = branch.get("commands", {})
    if not isinstance(commands, dict):
        return True
    for command in commands.values():
        if not isinstance(command, dict) or command.get("returncode") != 0:
            return True
    metrics = branch.get("metrics", {})
    if isinstance(metrics, dict):
        if not metrics.get("final_objects_exists") or not metrics.get("final_relations_exists"):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description='Run skeleton_async manager planning with a clean relation layout directory.')
    parser.add_argument('--scene_name', required=True)
    parser.add_argument('--run_root', required=True, type=Path)
    parser.add_argument('--benchmark_id', required=True)
    parser.add_argument('--model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--vision_model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--manager_model', default=None)
    parser.add_argument('--manager_vision_model', default=None)
    parser.add_argument('--manager_vision_image_detail', choices=['low', 'high', 'auto'], default='low')
    parser.add_argument('--wall_specialist_model', default=DEFAULT_WALL_SPECIALIST_MODEL)
    parser.add_argument('--wall_specialist_vision_model', default=None)
    parser.add_argument(
        '--wall_specialist_task_kinds',
        nargs='+',
        default=DEFAULT_WALL_SPECIALIST_TASK_KINDS,
        help='Canonical task kinds sent to the optional wall specialist; accepts space- or comma-separated values.',
    )
    parser.add_argument('--render_script', type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument('--precompute_python', type=Path, default=DEFAULT_PRECOMPUTE_PYTHON if DEFAULT_PRECOMPUTE_PYTHON.exists() else Path(sys.executable))
    parser.add_argument('--runner_python', type=Path, default=Path(sys.executable))
    parser.add_argument('--max_parallel_tasks', type=int, default=4)
    parser.add_argument('--relation_precompute_wait_seconds', type=float, default=900.0)
    parser.add_argument('--grid_size', type=int, default=33)
    parser.add_argument('--image_size', type=int, default=768)
    parser.add_argument('--api_base', default=None)
    parser.add_argument('--api_key_env', default='OPENAI_API_KEY')
    parser.add_argument('--vision_image_detail', choices=['low', 'high', 'auto'], default=None)
    parser.add_argument('--physics_model', default=None, help='Defaults to --model in the relation runner.')
    parser.add_argument('--physics_include_original', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    args.run_root = args.run_root.resolve()
    args.render_script = args.render_script.resolve()
    scene_paths = ScenePaths.from_scene_root(args.run_root)
    bench_root = scene_paths.relation_candidates_dir / args.benchmark_id
    if bench_root.exists():
        raise SystemExit(f'benchmark directory already exists: {bench_root}. Choose a new --benchmark_id.')
    bench_root.mkdir(parents=True, exist_ok=True)
    log_dir = bench_root / 'logs'

    env = os.environ.copy()

    summary: dict[str, Any] = {
        'schema_version': 'relation_async_benchmark',
        'scene_name': args.scene_name,
        'benchmark_id': args.benchmark_id,
        'run_root': str(args.run_root),
        'benchmark_root': str(bench_root),
        'dry_run': bool(args.dry_run),
        'fixed_views_enabled': False,
    }

    async_layout = bench_root / 'async_relation_layout'
    async_agent = bench_root / 'async_agent_run'
    expected = run_logged('async_expected_only', precompute_cmd(args, async_layout, expected_only=True), log_dir, env)
    async_commands: dict[str, Any] = {'expected_only': expected}
    if expected['returncode'] == 0:
        pre_proc, pre_meta = start_logged('async_precompute', precompute_cmd(args, async_layout), log_dir, env)
        runner_proc, runner_meta = start_logged('async_runner', runner_cmd(args, async_agent, async_layout), log_dir, env)
        # Reap both processes concurrently while preserving independent return codes.
        with ThreadPoolExecutor(max_workers=2) as pool:
            pre_future = pool.submit(finish_logged, pre_proc, pre_meta)
            runner_future = pool.submit(finish_logged, runner_proc, runner_meta)
            pre_result = pre_future.result()
            runner_result = runner_future.result()
        async_commands['precompute'] = pre_result
        async_commands['runner'] = runner_result
    else:
        async_commands['precompute'] = {'name': 'async_precompute', 'returncode': None, 'skipped_reason': 'expected_only_failed'}
        async_commands['runner'] = {'name': 'async_runner', 'returncode': None, 'skipped_reason': 'expected_only_failed'}
    summary['async'] = {
        'relation_layout_dir': str(async_layout),
        'agent_dir': str(async_agent),
        'commands': async_commands,
        'metrics': collect_run_metrics(async_agent, async_layout),
    }
    summary['benchmark_success'] = not benchmark_has_failed_commands(summary)
    write_json(bench_root / 'benchmark_summary.json', summary)
    write_report(bench_root / 'benchmark_report.md', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary['benchmark_success']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
