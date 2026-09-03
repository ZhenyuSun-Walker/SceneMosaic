import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix

os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

R_OPENGL_TO_P3D = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
R_P3D_TO_OPENGL = R_OPENGL_TO_P3D


def quaternion_to_matrix_numpy(quat):
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got shape {q.shape}.")

    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Quaternion norm is zero.")
    w, x, y, z = q / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def parse_scale(scale_value):
    scale = np.asarray(scale_value, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale, 3)
    if scale.size == 3:
        return scale
    raise ValueError(f"Scale must be scalar or 3D vector, got shape {scale.shape}.")


def apply_layout_to_unposed_mesh(mesh, layout):
    """Apply layout transform to an unposed mesh using the demo.py pipeline logic."""
    translation = torch.tensor(layout["translation"], dtype=torch.float32)
    rotation_q = torch.tensor(layout["rotation"], dtype=torch.float32)
    scale = torch.tensor(layout["scale"], dtype=torch.float32)

    vertices = mesh.vertices.astype(np.float32) @ R_OPENGL_TO_P3D
    vertices_tensor = torch.from_numpy(vertices).float().to(rotation_q.device)
    rotation_m = quaternion_to_matrix(rotation_q).transpose(-1, -2)  # Transpose to apply to row vectors
    l2c_transform = compose_transform(
        scale=scale,
        rotation=rotation_m,
        translation=translation,
    )
    vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
    mesh.vertices = vertices.squeeze(0).cpu().numpy() @ R_P3D_TO_OPENGL
    return mesh


def load_object_ids(layout_dir, unposed_dir):
    # Collect stems of all layout json files and stems of unposed mesh files
    layout_stems = {p.stem for p in layout_dir.glob("*.json")}
    mesh_stems = set()
    for pattern in ("*.glb", "*.gltf", "*.obj", "*.ply"):
        mesh_stems.update({p.stem for p in unposed_dir.glob(pattern)})

    # Return the sorted list of stems that appear in both layout and unposed dirs
    return sorted(layout_stems & mesh_stems)


def compose_scene(layout_dir, unposed_dir):
    object_ids = load_object_ids(layout_dir, unposed_dir)
    if not object_ids:
        raise FileNotFoundError(
            f"No matching object files found in {layout_dir} and {unposed_dir}."
        )

    posed_meshes = []
    for object_id in object_ids:
        # layout files are matched by stem (e.g. "chair_01.json" -> stem "chair_01")
        layout_path = layout_dir / f"{object_id}.json"

        # find a matching mesh file with a supported extension
        mesh_path = None
        for ext in (".glb", ".gltf", ".obj", ".ply"):
            candidate = unposed_dir / f"{object_id}{ext}"
            if candidate.exists():
                mesh_path = candidate
                break
        if mesh_path is None:
            raise FileNotFoundError(f"No unposed mesh found for object '{object_id}' in {unposed_dir}")

        with layout_path.open("r", encoding="utf-8") as f:
            layout = json.load(f)

        mesh = trimesh.load(mesh_path, force="mesh")
        posed_mesh = apply_layout_to_unposed_mesh(mesh, layout)
        posed_meshes.append(posed_mesh)

    if len(posed_meshes) == 1:
        return posed_meshes[0]

    # When multiple posed meshes exist, build a Scene and add each mesh as a separate geometry.
    # Exporting a Scene preserves per-mesh textures and PBR materials; concatenating meshes can lose texture/material
    # information because textures would need to be atlased.
    scene = trimesh.Scene()
    for id, pm in zip(object_ids, posed_meshes):
        scene.add_geometry(pm, node_name=id)
    return scene


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild scene mesh from unposed meshes and layout parameters."
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        default=Path("outputs/layout_params"),
        help="Directory containing object_XXX.json layout files.",
    )
    parser.add_argument(
        "--unposed_dir",
        type=Path,
        default=Path("outputs/unposed_objects"),
        help="Directory containing object_XXX.glb unposed meshes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/composed_scene.glb"),
        help="Output path for the composed scene mesh.",
    )
    args = parser.parse_args()

    layout_dir = args.layout_dir
    unposed_dir = args.unposed_dir
    output_path = args.output

    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {layout_dir}")
    if not unposed_dir.exists():
        raise FileNotFoundError(f"Unposed mesh directory not found: {unposed_dir}")

    scene_mesh = compose_scene(layout_dir, unposed_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(output_path)
    print(f"Saved recomposed scene to: {output_path}")


if __name__ == "__main__":
    main()
