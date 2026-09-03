import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import trimesh

from pytorch3d.transforms import quaternion_to_matrix

from render_scene_from_layout import (
    apply_layout_to_unposed_mesh,
    load_object_ids,
    load_layout,
    normalize_object_id
)


def fix_containment_and_save(
    layout_dir: Path,
    unposed_dir: Path,
    output_layout_dir: Path,
    requested_object_ids=None,
    anchor_id=None,
    tolerated_out_of_bounds: float = 0.0,
    detect_only: bool = False,
):
    """
    Main pipeline to fix the containment relationship between the container (anchor)
    and other objects in BEV perspective, then output updated layout JSONs.
    """
    if anchor_id is None:
        raise ValueError("An anchor_id (container object) must be specified for this containment correction.")
    if tolerated_out_of_bounds < 0:
        raise ValueError("tolerated_out_of_bounds must be non-negative.")

    # 1. Load object IDs and filter matched ones
    object_ids = load_object_ids(layout_dir, unposed_dir, requested_object_ids=requested_object_ids)
    normalized_anchor_id = normalize_object_id(anchor_id)

    if normalized_anchor_id not in object_ids:
        raise FileNotFoundError(f"Container anchor_id {normalized_anchor_id} not found in the matched object IDs.")

    # 2. Load all meshes and their initial layout parameters
    posed_meshes = []
    layout_params = []
    anchor_index = -1

    for idx, object_id in enumerate(object_ids):
        if object_id == normalized_anchor_id:
            anchor_index = idx
        layout_path = layout_dir / f"{object_id}.json"
        mesh_path = unposed_dir / f"{object_id}.glb"

        layout, _, _, _ = load_layout(layout_path)
        mesh = trimesh.load(mesh_path, force="mesh")
        posed_mesh = apply_layout_to_unposed_mesh(mesh, layout)

        posed_meshes.append(posed_mesh)
        layout_params.append(deepcopy(layout))

    # 3. Retrieve container (anchor) details in World Space
    anchor_layout = layout_params[anchor_index]
    anchor_rotation = np.asarray(anchor_layout["rotation"], dtype=np.float32).reshape(4)

    # Convert quaternion to row-major matrix to align with the render script's logic
    anchor_rot_mat = (
        quaternion_to_matrix(torch.from_numpy(anchor_rotation))
        .transpose(-1, -2)
        .cpu()
        .numpy()
    )
    # Rotation fix transforms coordinates from World Space into Container Local Space
    rotation_fix = anchor_rot_mat.T.astype(np.float32)

    # 4. Compute Container BEV extents (using X and Z axes as BEV plane)
    # Re-align container mesh to local space to find precise 3D/2D bounding box
    anchor_local_vertices = posed_meshes[anchor_index].vertices.astype(np.float32) @ rotation_fix
    c_min = anchor_local_vertices.min(axis=0)
    c_max = anchor_local_vertices.max(axis=0)

    # We use X and Z dimensions for BEV (Birds-Eye-View) plane bounding box
    # as common in 3D scene representation pipelines
    c_bev_min_x, c_bev_max_x = c_min[0], c_max[0]
    c_bev_min_z, c_bev_max_z = c_min[2], c_max[2]

    container_extent_x = c_bev_max_x - c_bev_min_x
    container_extent_z = c_bev_max_z - c_bev_min_z
    container_local_center_x = (c_bev_min_x + c_bev_max_x) / 2.0
    container_local_center_z = (c_bev_min_z + c_bev_max_z) / 2.0
    out_of_bounds_tolerance_ratio = float(tolerated_out_of_bounds)

    print(
        f"Container BEV local bounds -> X: [{c_bev_min_x:.3f}, {c_bev_max_x:.3f}], "
        f"Z: [{c_bev_min_z:.3f}, {c_bev_max_z:.3f}]"
    )

    # 5. Process each contained object
    output_layout_dir.mkdir(parents=True, exist_ok=True)
    out_of_bounds_object_ids: list[str] = []
    fixed_object_ids: list[str] = []
    written_object_ids: list[str] = []

    for idx, object_id in enumerate(object_ids):
        current_layout = deepcopy(layout_params[idx])

        # Skip correcting the container itself
        if object_id == normalized_anchor_id:
            output_path = output_layout_dir / f"{object_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(current_layout, f, ensure_ascii=False, indent=2)
            written_object_ids.append(object_id)
            continue

        # Transform current object's vertices into Container Local Space
        obj_local_vertices = posed_meshes[idx].vertices.astype(np.float32) @ rotation_fix

        o_min = obj_local_vertices.min(axis=0)
        o_max = obj_local_vertices.max(axis=0)

        o_bev_min_x, o_bev_max_x = o_min[0], o_max[0]
        o_bev_min_z, o_bev_max_z = o_min[2], o_max[2]

        obj_occupied_x = o_bev_max_x - o_bev_min_x
        obj_occupied_z = o_bev_max_z - o_bev_min_z
        out_of_bounds_tolerance_x = out_of_bounds_tolerance_ratio * obj_occupied_x
        out_of_bounds_tolerance_z = out_of_bounds_tolerance_ratio * obj_occupied_z

        # Calculate current object local translation components
        obj_world_trans = np.asarray(current_layout["translation"], dtype=np.float32).reshape(3)
        obj_local_trans = obj_world_trans @ rotation_fix

        fixed_local_x = obj_local_trans[0]
        fixed_local_z = obj_local_trans[2]
        object_out_of_bounds = False

        # --- BEV Dimension X Correction ---
        if obj_occupied_x > container_extent_x + out_of_bounds_tolerance_x * 2:
            # Case 1: Object is larger than container in X axis -> Center it
            object_out_of_bounds = True
            fixed_local_x = container_local_center_x
            print(f"[{object_id}] X-axis size exceeds container. Force Centered.")
        else:
            # Case 2: Standard bounded box constraint check (1-2 borders out of bounds)
            offset_x_max = o_bev_max_x - c_bev_max_x
            offset_x_min = c_bev_min_x - o_bev_min_x

            if offset_x_max > out_of_bounds_tolerance_x:  # Out of right bound -> push left
                object_out_of_bounds = True
                fixed_local_x -= offset_x_max
                print(f"[{object_id}] Right border out of bounds by {offset_x_max:.3f}. Shifted left.")
            if offset_x_min > out_of_bounds_tolerance_x:  # Out of left bound -> push right
                object_out_of_bounds = True
                fixed_local_x += offset_x_min
                print(f"[{object_id}] Left border out of bounds by {offset_x_min:.3f}. Shifted right.")

        # --- BEV Dimension Z Correction ---
        if obj_occupied_z > container_extent_z + out_of_bounds_tolerance_z * 2:
            # Case 1: Object is larger than container in Z axis -> Center it
            object_out_of_bounds = True
            fixed_local_z = container_local_center_z
            print(f"[{object_id}] Z-axis size exceeds container. Force Centered.")
        else:
            # Case 2: Standard bounded box constraint check
            offset_z_max = o_bev_max_z - c_bev_max_z
            offset_z_min = c_bev_min_z - o_bev_min_z

            if offset_z_max > out_of_bounds_tolerance_z:  # Out of top bound -> push bottom
                object_out_of_bounds = True
                fixed_local_z -= offset_z_max
                print(f"[{object_id}] Top border out of bounds by {offset_z_max:.3f}. Shifted down.")
            elif offset_z_min > out_of_bounds_tolerance_z:  # Out of bottom bound -> push top
                object_out_of_bounds = True
                fixed_local_z += offset_z_min
                print(f"[{object_id}] Bottom border out of bounds by {offset_z_min:.3f}. Shifted up.")

        if object_out_of_bounds:
            out_of_bounds_object_ids.append(object_id)
            if detect_only:
                continue

            # 6. Transform the updated local translation back to World Coordinate System
            updated_local_trans = np.array([fixed_local_x, obj_local_trans[1], fixed_local_z], dtype=np.float32)

            # Reverse the rotation fix mapping: World = Local @ rotation_fix.T
            updated_world_trans = updated_local_trans @ rotation_fix.T

            # Update layout parameters
            current_layout["translation"] = [updated_world_trans.tolist()]
            fixed_object_ids.append(object_id)

        # Save layout to target directory
        if not detect_only:
            output_path = output_layout_dir / f"{object_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(current_layout, f, ensure_ascii=False, indent=2)
            written_object_ids.append(object_id)

    report = {
        "anchor_id": normalized_anchor_id,
        "requested_object_ids": list(object_ids),
        "out_of_bounds_object_ids": out_of_bounds_object_ids,
        "fixed_object_ids": fixed_object_ids,
        "written_object_ids": written_object_ids,
        "detect_only": detect_only,
        "output_layout_dir": str(output_layout_dir),
    }
    print(f"FIX_LAYOUT_CONTAINMENT_SUMMARY: {json.dumps(report, ensure_ascii=False)}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Fix object containment relationship by adjusting layout centers in Container BEV coordinates."
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        default=Path("outputs/layout_params"),
        help="Directory containing original object_XXX.json layout files.",
    )
    parser.add_argument(
        "--unposed_dir",
        type=Path,
        default=Path("outputs/unposed_objects"),
        help="Directory containing object_XXX.glb unposed meshes.",
    )
    parser.add_argument(
        "--output_layout_dir",
        type=Path,
        default=Path("outputs/layout_params_test"),
        help="Directory where fixed object layout parameters will be saved.",
    )
    parser.add_argument(
        "--object_ids",
        nargs="+",
        default=None,
        help="Optional subset of object ids to process. Defaults to all matched objects.",
    )
    parser.add_argument(
        "--anchor_id",
        type=str,
        required=True,
        help="The object ID of the container (anchor) that defines the BEV bounds for containment correction.",
    )
    parser.add_argument(
        "--tolerated_out_of_bounds",
        type=float,
        default=0.0,
        help="Tolerance ratio for out-of-bounds checks in BEV units; the anchor BEV extent is not scaled.",
    )
    parser.add_argument(
        "--detect_only",
        action="store_true",
        help="Only detect out-of-bounds objects and do not modify their layouts.",
    )
    args = parser.parse_args()

    if not args.layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {args.layout_dir}")
    if not args.unposed_dir.exists():
        raise FileNotFoundError(f"Unposed mesh directory not found: {args.unposed_dir}")

    fix_containment_and_save(
        layout_dir=args.layout_dir,
        unposed_dir=args.unposed_dir,
        output_layout_dir=args.output_layout_dir,
        requested_object_ids=args.object_ids,
        anchor_id=args.anchor_id,
        tolerated_out_of_bounds=args.tolerated_out_of_bounds,
        detect_only=args.detect_only,
    )


if __name__ == "__main__":
    main()
