import argparse
import json
from pathlib import Path

import numpy as np
import math

try:
    import pybullet as p
except ImportError as e:
    raise ImportError(
        "pybullet is required for collision checking. Install it with: pip install pybullet"
    ) from e


def as_vector(value, expected_sizes):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size not in expected_sizes:
        raise ValueError(f"Expected sizes {expected_sizes}, but got shape {arr.shape}")
    return arr


def parse_scale(value):
    arr = as_vector(value, {1, 3})
    if arr.size == 1:
        return np.repeat(arr, 3)
    return arr


def parse_translation(value):
    return as_vector(value, {3})


def parse_rotation_wxyz(value):
    quat = as_vector(value, {4})
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def quat_wxyz_to_xyzw(value):
    quat = as_vector(value, {4})
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def format_object_id(object_id):
    object_id = str(object_id)
    if object_id and object_id[0].isdigit():
        return f"object_{object_id}"
    return object_id


def resolve_requested_object_ids(requested_object_ids):
    if requested_object_ids is None:
        return None
    if isinstance(requested_object_ids, (str, int)):
        requested_object_ids = [requested_object_ids]
    return [format_object_id(object_id) for object_id in requested_object_ids]


def load_layout(layout_path: Path):
    with layout_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    translation = parse_translation(raw["translation"])
    rotation_wxyz = parse_rotation_wxyz(raw["rotation"])
    scale = parse_scale(raw["scale"])

    return raw, translation, rotation_wxyz, scale


def load_obj_extents(obj_path: Path):
    vertices = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"No vertices found in collision hull: {obj_path}")

    vertices = np.asarray(vertices, dtype=np.float32)
    return vertices.max(axis=0) - vertices.min(axis=0)


def should_skip_thin_object(hull_path: Path, scale, thin_ratio_threshold: float):
    extents = load_obj_extents(hull_path)
    final_sizes = extents * np.asarray(scale, dtype=np.float32).reshape(3)
    smallest_edge = float(np.min(final_sizes))
    if smallest_edge <= 1e-8:
        return True, float("inf"), final_sizes

    ratio = float(np.max(final_sizes) / smallest_edge)
    return ratio > thin_ratio_threshold, ratio, final_sizes


def load_object_ids(layout_dir: Path, hull_dir: Path, requested_object_ids=None):
    layout_ids = {p.stem for p in layout_dir.glob("*.json")}
    hull_ids = {p.stem for p in hull_dir.glob("*.obj")}
    matched_ids = sorted(layout_ids & hull_ids)

    normalized_requested_ids = resolve_requested_object_ids(requested_object_ids)
    if normalized_requested_ids is None:
        return matched_ids

    missing_ids = [object_id for object_id in normalized_requested_ids if object_id not in matched_ids]
    if missing_ids:
        raise FileNotFoundError(
            f"Requested object ids not found in both layout and hull directories: {missing_ids}. "
            f"Matched ids: {matched_ids}"
        )
    return normalized_requested_ids


def extract_contact_bodies(contact):
    if hasattr(contact, "bodyUniqueIdA") and hasattr(contact, "bodyUniqueIdB"):
        return contact.bodyUniqueIdA, contact.bodyUniqueIdB
    if isinstance(contact, (list, tuple)):
        if len(contact) >= 14:
            return contact[1], contact[2]
        if len(contact) >= 2:
            return contact[0], contact[1]
    return None, None


def sanitize_for_json(obj):
    """Recursively replace non-finite floats with None so JSON is valid."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else "inf"
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def run_collision_check(
    layout_dir: Path,
    hull_dir: Path,
    output_path: Path | None,
    object_ids=None,
    steps: int = 1,
    mass: float = 10.0,
    thin_ratio_threshold: float = 15.0,
    anchor_id: str | None = None,
    collision_distance_threshold: float = 6e-3,
):
    object_ids = load_object_ids(layout_dir, hull_dir, requested_object_ids=object_ids)
    if not object_ids:
        raise FileNotFoundError(
            f"No matching object files in {layout_dir} and {hull_dir}."
        )

    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet in DIRECT mode.")

    try:
        p.resetSimulation(physicsClientId=client_id)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=client_id)

        body_to_object_id = {}
        object_sizes = {}
        skipped_objects = []
        flags = 0
        if hasattr(p, "GEOM_FORCE_CONVEX_MESH"):
            flags |= p.GEOM_FORCE_CONVEX_MESH

        for object_id in object_ids:
            layout_path = layout_dir / f"{format_object_id(object_id)}.json"
            hull_path = hull_dir / f"{format_object_id(object_id)}.obj"

            _, translation, rotation_wxyz, scale = load_layout(layout_path)
            should_skip, ratio, final_sizes = should_skip_thin_object(
                hull_path,
                scale,
                thin_ratio_threshold,
            )
            object_sizes[format_object_id(object_id)] = final_sizes.tolist()
            if should_skip:
                skipped_objects.append(
                    {
                        "object_id": format_object_id(object_id),
                        "ratio": ratio,
                    }
                )
                continue

            quat_xyzw = quat_wxyz_to_xyzw(rotation_wxyz).tolist()

            shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=str(hull_path),
                meshScale=scale.tolist(),
                flags=flags,
                physicsClientId=client_id,
            )
            if shape_id < 0:
                raise RuntimeError(f"Failed to create collision shape: {hull_path}")

            body_id = p.createMultiBody(
                baseMass=float(mass),
                baseCollisionShapeIndex=shape_id,
                basePosition=translation.tolist(),
                baseOrientation=quat_xyzw,
                physicsClientId=client_id,
            )
            if body_id < 0:
                raise RuntimeError(f"Failed to create body for: {format_object_id(object_id)}")

            body_to_object_id[body_id] = format_object_id(object_id)

        for _ in range(max(int(steps), 1)):
            p.stepSimulation(physicsClientId=client_id)

        if hasattr(p, "performCollisionDetection"):
            p.performCollisionDetection(physicsClientId=client_id)

        body_ids = list(body_to_object_id.keys())

        pair_stats = {}

        penetration_eps = -collision_distance_threshold

        for i in range(len(body_ids)):
            for j in range(i + 1, len(body_ids)):

                body0 = body_ids[i]
                body1 = body_ids[j]

                obj0 = body_to_object_id[body0]
                obj1 = body_to_object_id[body1]

                if anchor_id and (obj0 == anchor_id or obj1 == anchor_id):
                    continue

                pts = p.getClosestPoints(
                    bodyA=body0,
                    bodyB=body1,
                    distance=collision_distance_threshold,
                    physicsClientId=client_id,
                )

                if not pts:
                    continue

                min_distance = min(pt[8] for pt in pts)

                if min_distance > collision_distance_threshold:
                    continue

                if min_distance >= penetration_eps:
                    continue

                key = tuple(sorted([obj0, obj1]))

                pair_stats[key] = {
                    "a": key[0],
                    "b": key[1],
                    "contacts": len(pts),
                    "min_distance": float(min_distance),
                    "size_a": list(object_sizes.get(obj0, [])),
                    "size_b": list(object_sizes.get(obj1, [])),
                }
    finally:
        p.disconnect(physicsClientId=client_id)

    report = {
        "num_objects": len(object_ids),
        "objects": list(object_ids),
        "num_skipped_objects": len(skipped_objects),
        "skipped_objects": skipped_objects,
        "thin_ratio_threshold": float(thin_ratio_threshold),
        "num_steps": int(max(int(steps), 1)),
        "collision_distance_threshold": float(collision_distance_threshold),
        "num_pairs": len(pair_stats),
        "collisions": list(pair_stats.values()),
    }
    # Sanitize report so JSON doesn't contain Infinity/NaN which is not valid JSON
    sanitized_report = sanitize_for_json(report)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(sanitized_report, f, indent=2)
        print(f"[OK] Saved collision report: {output_path}")

    print(json.dumps(sanitized_report, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check pairwise collisions between objects using collision hulls and layout poses."
        )
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        default=Path("outputs/generated/group_object_020/layout_org/round_1/layout_params"),
        help="Directory containing object_XXX.json layout files.",
    )
    parser.add_argument(
        "--hull_dir",
        type=Path,
        default=Path("outputs/collision_hulls"),
        help="Directory containing object_XXX.obj collision hull files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for collision report.",
    )
    parser.add_argument(
        "--object_ids",
        nargs="+",
        default=None,
        help=(
            "Optional subset of object ids to check (XXX in object_XXX). "
            "Defaults to all matched objects."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of simulation steps to run before checking contacts.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=10.0,
        help="Density proxy for collision shape loading.",
    )
    parser.add_argument(
        "--anchor_id",
        type=str,
        default=None,
        help="Optional anchor object id whose collisions with all other objects are ignored.",
    )
    parser.add_argument(
        "--thin_ratio_threshold",
        type=float,
        default=15.0,
        help="Skip objects whose final size aspect ratio exceeds this value.",
    )
    parser.add_argument(
        "--collision_distance_threshold",
        type=float,
        default=6e-3,
        help="Penetration threshold (positive value, collision when min_distance < -threshold).",
    )

    args = parser.parse_args()

    if not args.layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {args.layout_dir}")
    if not args.hull_dir.exists():
        raise FileNotFoundError(f"Hull directory not found: {args.hull_dir}")
    if args.mass <= 0:
        raise ValueError("--mass must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.thin_ratio_threshold <= 0:
        raise ValueError("--thin_ratio_threshold must be positive.")

    run_collision_check(
        layout_dir=args.layout_dir,
        hull_dir=args.hull_dir,
        output_path=args.output,
        object_ids=args.object_ids,
        steps=args.steps,
        mass=args.mass,
        thin_ratio_threshold=args.thin_ratio_threshold,
        anchor_id=args.anchor_id,
        collision_distance_threshold=args.collision_distance_threshold,
    )


if __name__ == "__main__":
    main()
