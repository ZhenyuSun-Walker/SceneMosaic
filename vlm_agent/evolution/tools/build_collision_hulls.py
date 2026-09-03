import argparse
import os
import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager, get_context
from pathlib import Path
from queue import Empty
from typing import Iterable, Optional, Set

import numpy as np
import trimesh
import coacd


TASK_TIMEOUT_SECONDS = 180
R_OPENGL_TO_P3D = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
SUMMARY_PREFIX = "BUILD_COLLISION_HULLS_SUMMARY:"


def load_object_ids(unposed_dir: Path):
    # Return all .glb stems in the directory (support arbitrary filenames)
    return sorted(p.stem for p in unposed_dir.glob("*.glb"))


def resolve_requested_object_ids(requested_object_ids, available_ids: list[str]):
    """Resolve requested ids (which may be numeric like '001' or stems)
    to actual available stems from `available_ids`.

    - If `requested_object_ids` is None -> returns None
    - If a requested id matches an available stem exactly -> use it
    - If a requested id is numeric (all digits), match any available stem that endswith that numeric suffix.
    Raises FileNotFoundError if any requested id cannot be resolved.
    """
    if requested_object_ids is None:
        return None
    if isinstance(requested_object_ids, (str, int)):
        requested_object_ids = [requested_object_ids]

    available_set = set(available_ids)
    resolved = []
    for req in requested_object_ids:
        reqs = str(req)
        # numeric id -> match stems that end with this suffix
        if reqs.isdigit():
            reqs = f"object_{reqs}"
        if reqs in available_set:
            resolved.append(reqs)
            continue
        # try common pattern: strip leading 'object_' and check
        if reqs.startswith("object_"):
            alt = reqs[len("object_"):]
            if alt in available_set:
                resolved.append(alt)
                continue
        raise FileNotFoundError(
            f"Requested object id not found or ambiguous: {reqs}. Available ids: {available_ids}"
        )

    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in resolved:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_split_object_ids(path: Optional[Path], available_ids: list[str]) -> Set[str]:
    """Load split list JSON and resolve entries against available ids.

    The JSON may be a list of ids or a dict of id->bool. Entries may be
    stems or numeric suffixes; resolution uses `resolve_requested_object_ids`.
    """
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Split list not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = []
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = [k for k, v in payload.items() if bool(v)]
    else:
        raise ValueError("Split list JSON must be a list or a dict")

    resolved = resolve_requested_object_ids(entries, available_ids)
    return set(resolved)


def load_requested_object_ids(object_ids: Optional[list[str]]) -> Optional[list[str]]:
    if object_ids is None:
        return None
    normalized = [str(object_id) for object_id in object_ids]
    return sorted(dict.fromkeys(normalized))


def run_coacd_parts(mesh: trimesh.Trimesh, threshold: float, preprocess_mode: str):
    mesh_coacd = coacd.Mesh(mesh.vertices.astype(np.float32), mesh.faces)
    return coacd.run_coacd(
        mesh_coacd,
        threshold=threshold,
        preprocess_mode=preprocess_mode,
    )


def simplify_mesh_faces(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if target_faces <= 0:
        return mesh
    if len(mesh.faces) <= target_faces:
        return mesh

    # Use trimesh decimation when available; fallback to original mesh on failure.
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if simplified is not None and len(simplified.faces) > 0:
            return simplified
    except Exception as e:
        print(f"Error: {e}")
        print(f"Warning: Failed to simplify mesh with {len(mesh.faces)} faces to target {target_faces} faces. Using original mesh.")
        pass
    return mesh


def split_mesh_for_decomposition(mesh: trimesh.Trimesh) -> Iterable[trimesh.Trimesh]:
    center = mesh.bounding_box.centroid
    cut_meshes = [
        mesh.slice_plane(np.array([0, center[1], 0]), np.array([1, 0, 0])).slice_plane(
            np.array([center[0], 0, 0]), np.array([0, 1, 0])
        ),
        mesh.slice_plane(np.array([0, center[1], 0]), np.array([-1, 0, 0])).slice_plane(
            np.array([center[0], 0, 0]), np.array([0, 1, 0])
        ),
        mesh.slice_plane(np.array([0, center[1], 0]), np.array([1, 0, 0])).slice_plane(
            np.array([center[0], 0, 0]), np.array([0, -1, 0])
        ),
        mesh.slice_plane(np.array([0, center[1], 0]), np.array([-1, 0, 0])).slice_plane(
            np.array([center[0], 0, 0]), np.array([0, -1, 0])
        ),
    ]
    return [m for m in cut_meshes if len(m.vertices) > 0 and len(m.faces) > 0]


def scale_parts_to_origin(parts: Iterable[trimesh.Trimesh], origin_mesh: trimesh.Trimesh):
    part_meshes = [m.copy() for m in parts]
    if not part_meshes:
        return []

    combined_collision_mesh = trimesh.util.concatenate(part_meshes)
    collision_size = combined_collision_mesh.bounding_box.extents
    origin_size = origin_mesh.bounding_box.extents

    safe_origin_size = np.where(np.abs(origin_size) < 1e-8, 1.0, origin_size)
    scale_factor = collision_size / safe_origin_size
    safe_scale_factor = np.where(np.abs(scale_factor) < 1e-8, 1.0, scale_factor)

    collision_center = combined_collision_mesh.bounding_box.centroid
    for mesh in part_meshes:
        mesh.apply_translation(-collision_center)
        mesh.apply_scale(1.0 / safe_scale_factor)
        mesh.apply_translation(collision_center)
    return part_meshes


def build_coacd_hull(
    mesh_path: Path,
    split_before_decompose: bool,
    threshold: float,
    preprocess_mode: str,
    target_faces_per_part: int,
):
    mesh = trimesh.load(mesh_path, force="mesh")

    mesh_zup = mesh.copy()
    mesh_zup.vertices = mesh.vertices.astype(np.float32) @ R_OPENGL_TO_P3D

    parts = []
    if split_before_decompose:
        for cut_mesh in split_mesh_for_decomposition(mesh_zup):
            cut_mesh = simplify_mesh_faces(cut_mesh, target_faces_per_part)
            result = run_coacd_parts(cut_mesh, threshold, preprocess_mode)
            parts.extend(result)
    else:
        mesh_zup = simplify_mesh_faces(mesh_zup, target_faces_per_part)
        parts = run_coacd_parts(mesh_zup, threshold, preprocess_mode)

    hulls = [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]
    if not hulls:
        hulls = [mesh_zup.convex_hull]

    scaled_hulls = scale_parts_to_origin(hulls, mesh_zup)
    combined_hull = trimesh.util.concatenate(scaled_hulls) if scaled_hulls else mesh_zup.convex_hull
    return combined_hull, scaled_hulls


def save_part_meshes(part_meshes: Iterable[trimesh.Trimesh], parts_dir: Path):
    parts_dir.mkdir(parents=True, exist_ok=True)
    for idx, part in enumerate(part_meshes):
        part.export(parts_dir / f"collision_part_{idx:03d}.obj")


def process_object(
    object_id: str,
    unposed_dir: Path,
    output_dir: Path,
    split_mode: str,
    split_object_ids: Set[str],
    save_parts: bool,
    threshold: float,
    preprocess_mode: str,
    target_faces_per_part: int,
):
    # object_id is the file stem (without extension). Support arbitrary stems.
    in_path = unposed_dir / f"{object_id}.glb"
    out_path = output_dir / f"{object_id}.obj"

    split_before_decompose = split_mode == "all" or (
        split_mode == "list" and object_id in split_object_ids
    )

    hull, part_meshes = build_coacd_hull(
        in_path,
        split_before_decompose=split_before_decompose,
        threshold=threshold,
        preprocess_mode=preprocess_mode,
        target_faces_per_part=target_faces_per_part,
    )
    if part_meshes:
        scene = trimesh.Scene(part_meshes)
        export_content = scene.export(file_type='obj')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(export_content)
    else:
        hull.export(out_path)

    parts_dir = output_dir / f"object_{object_id}_parts"
    if save_parts:
        save_part_meshes(part_meshes, parts_dir)

    item_key = object_id
    item_value = {
        "source_mesh": str(in_path),
        "collision_hull": str(out_path),
        "split_before_decompose": bool(split_before_decompose),
        "num_parts": int(len(part_meshes)),
        "num_vertices": int(len(hull.vertices)),
        "num_faces": int(len(hull.faces)),
        "volume": float(hull.volume),
        "coacd": {
            "threshold": float(threshold),
            "preprocess_mode": preprocess_mode,
            "target_faces_per_part": int(target_faces_per_part),
        },
    }
    if save_parts:
        item_value["parts_dir"] = str(parts_dir)
    return item_key, item_value, in_path.name, out_path.name


def process_object_with_error_context(
    object_id: str,
    unposed_dir: Path,
    output_dir: Path,
    split_mode: str,
    split_object_ids: Set[str],
    save_parts: bool,
    threshold: float,
    preprocess_mode: str,
    target_faces_per_part: int,
    worker_start_queue=None,
):
    try:
        if worker_start_queue is not None:
            worker_start_queue.put((object_id, os.getpid()))
        return process_object(
            object_id,
            unposed_dir,
            output_dir,
            split_mode,
            split_object_ids,
            save_parts,
            threshold,
            preprocess_mode,
            target_faces_per_part,
        )
    except Exception as exc:
        print(f"[ERROR] Failed while processing {object_id}.glb", flush=True)
        traceback.print_exc()
        raise RuntimeError(f"Failed while processing {object_id}.glb") from exc


def terminate_executor_worker(executor: ProcessPoolExecutor, pid: int):
    process = getattr(executor, "_processes", {}).get(pid)
    if process is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(timeout=1)


def terminate_executor_workers(executor: ProcessPoolExecutor):
    processes = getattr(executor, "_processes", None)
    executor.shutdown(wait=False, cancel_futures=True)
    if not processes:
        return
    for process in processes.values():
        if process.is_alive():
            process.terminate()
    for process in processes.values():
        process.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(
        description="Build convex hulls for object_XXX.glb files as collision meshes."
    )
    parser.add_argument(
        "--unposed_dir",
        type=Path,
        default=Path("outputs/unposed_objects"),
        help="Directory containing object_XXX.glb files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/collision_hulls"),
        help="Directory to save object_XXX.obj convex hulls.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/collision_hulls/manifest.json"),
        help="Path to save hull metadata manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 1) + 4)),
        help="Number of worker threads for parallel hull generation.",
    )
    parser.add_argument(
        "--object_ids",
        nargs="+",
        default=None,
        help="Optional list of object ids to process. When omitted, all objects are processed.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="none",
        choices=["none", "all", "list"],
        help="Whether to split mesh into 4 slices before COACD decomposition.",
    )
    parser.add_argument(
        "--split_list",
        type=Path,
        default=None,
        help="Optional JSON list/dict specifying object ids to split when split_mode=list.",
    )
    parser.add_argument(
        "--save_parts",
        action="store_true",
        help="Save each decomposed collision part as object_XXX_parts/collision_part_YYY.obj.",
    )
    parser.add_argument(
        "--coacd_threshold",
        type=float,
        default=0.1,
        help="COACD threshold.",
    )
    parser.add_argument(
        "--coacd_preprocess_mode",
        type=str,
        default="off",
        help="COACD preprocess mode.",
    )
    parser.add_argument(
        "--target_faces_per_part",
        type=int,
        default=1000,
        help="Target number of faces for each part before COACD. <=0 disables simplification.",
    )
    args = parser.parse_args()

    if not args.unposed_dir.exists():
        raise FileNotFoundError(f"Unposed mesh directory not found: {args.unposed_dir}")

    object_ids = load_object_ids(args.unposed_dir)
    if not object_ids:
        raise FileNotFoundError(f"No .glb files found in: {args.unposed_dir}")

    # Resolve requested ids (may be numeric suffixes or stems)
    requested_object_ids = resolve_requested_object_ids(args.object_ids, object_ids)
    if requested_object_ids is not None:
        requested_object_ids_set = set(requested_object_ids)
        missing_requested = sorted(object_id for object_id in requested_object_ids_set if object_id not in set(object_ids))
        if missing_requested:
            raise FileNotFoundError(
                "Requested object ids were not found in the unposed directory: "
                + ", ".join(missing_requested)
            )
        object_ids = [object_id for object_id in object_ids if object_id in requested_object_ids_set]
        if not object_ids:
            raise FileNotFoundError("No matching object ids remained after applying --object_ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.split_mode == "list" and args.split_list is None:
        raise ValueError("--split_list is required when --split_mode=list")

    split_object_ids = load_split_object_ids(args.split_list, object_ids)

    manifest = {}
    failed_object_ids: list[str] = []
    ctx = get_context("spawn")
    manager = Manager()
    worker_start_queue = manager.Queue()
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
    futures = {
        executor.submit(
            process_object_with_error_context,
            object_id,
            args.unposed_dir,
            args.output_dir,
            args.split_mode,
            split_object_ids,
            args.save_parts,
            args.coacd_threshold,
            args.coacd_preprocess_mode,
            args.target_faces_per_part,
            worker_start_queue,
        ): object_id
        for object_id in object_ids
    }
    submitted_at = {future: time.monotonic() for future in futures}
    pending = set(futures)
    timed_out_futures = set()
    running_worker_pids: dict[str, int] = {}
    try:
        while pending:
            while True:
                try:
                    object_id, pid = worker_start_queue.get_nowait()
                except Empty:
                    break
                running_worker_pids[object_id] = pid

            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                object_id = futures[future]
                if future in timed_out_futures:
                    continue
                try:
                    item_key, item_value, in_name, out_name = future.result()
                except Exception as exc:
                    failed_object_ids.append(object_id)
                    print(f"[FAILED] {object_id}.glb -> {exc}", flush=True)
                    continue
                manifest[item_key] = item_value
                print(f"[OK] {in_name} -> {out_name}")

            now = time.monotonic()
            overdue = [future for future in pending if now - submitted_at[future] > TASK_TIMEOUT_SECONDS]
            if overdue:
                for future in overdue:
                    object_id = futures[future]
                    failed_object_ids.append(object_id)
                    timed_out_futures.add(future)
                    pending.remove(future)
                    print(f"[TIMEOUT] {object_id}.glb", flush=True)
                    pid = running_worker_pids.get(object_id)
                    if pid is not None:
                        terminate_executor_worker(executor, pid)
    finally:
        manager.shutdown()
        executor.shutdown(wait=False, cancel_futures=True)

    # Keep manifest entries stable by object id order.
    manifest = dict(sorted(manifest.items(), key=lambda item: item[0]))

    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved hull manifest to: {args.manifest}")
    print(
        f"{SUMMARY_PREFIX} "
        + json.dumps(
            {
                "failed_object_ids": sorted(dict.fromkeys(failed_object_ids)),
                "succeeded_object_ids": sorted(manifest.keys()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()