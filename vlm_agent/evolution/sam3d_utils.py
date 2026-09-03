from __future__ import annotations

import importlib
from copy import deepcopy

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T
_R_ZUP_TO_P3D = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
_R_P3D_TO_ZUP = _R_ZUP_TO_P3D
_R_P3D_TO_OPENGL = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
_R_OPENGL_TO_P3D = _R_P3D_TO_OPENGL.T


def make_scene_untextured_mesh(*outputs, in_place=False, return_all=False):
    import trimesh
    compose_transform = importlib.import_module(
        "sam3d_objects.data.dataset.tdfy.transforms_3d"
    ).compose_transform

    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_meshes = []
    all_unposed_meshes = []
    all_layout_params = []
    for output in outputs:
        mesh = output["glb"]
        if mesh is None:
            continue

        R_l2c = quaternion_to_matrix(output["rotation"])
        r_p3d_to_zup = torch.from_numpy(_R_P3D_TO_ZUP).to(device=R_l2c.device, dtype=R_l2c.dtype)
        stored_rotation_matrix = torch.matmul(r_p3d_to_zup, R_l2c)
        stored_rotation = matrix_to_quaternion(stored_rotation_matrix.transpose(-1, -2))

        all_unposed_meshes.append(deepcopy(mesh))

        all_layout_params.append(
            {
                "translation": output["translation"].detach().cpu().tolist(),
                "rotation": stored_rotation.detach().cpu().tolist(),
                "scale": output["scale"].detach().cpu().tolist(),
            }
        )

        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_P3D_TO_OPENGL
        all_meshes.append(mesh)

    if not all_meshes:
        if return_all:
            return None, [], [], []
        return None

    if len(all_meshes) == 1:
        scene_mesh = all_meshes[0]
    else:
        scene_mesh = trimesh.util.concatenate(all_meshes)

    if return_all:
        return scene_mesh, all_meshes, all_unposed_meshes, all_layout_params
    return scene_mesh


def normalize_scene_to_canonical_cube(scene_mesh, posed_meshes, layout_params, cube_size=1.0, in_place=False):
    if scene_mesh is None:
        return scene_mesh, posed_meshes, layout_params

    if not in_place:
        scene_mesh = deepcopy(scene_mesh)
        posed_meshes = [deepcopy(mesh) for mesh in posed_meshes]
        layout_params = deepcopy(layout_params)

    scene_vertices_p3d = scene_mesh.vertices.astype(np.float32) @ _R_OPENGL_TO_P3D
    bbox_min = scene_vertices_p3d.min(axis=0)
    bbox_max = scene_vertices_p3d.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        return scene_mesh, posed_meshes, layout_params

    normalize_scale = float(cube_size) / max_extent

    scene_mesh.vertices = ((scene_vertices_p3d - center) * normalize_scale) @ _R_P3D_TO_OPENGL
    for mesh in posed_meshes:
        vertices_p3d = mesh.vertices.astype(np.float32) @ _R_OPENGL_TO_P3D
        mesh.vertices = ((vertices_p3d - center) * normalize_scale) @ _R_P3D_TO_OPENGL

    for params in layout_params:
        translation = np.asarray(params["translation"], dtype=np.float32)
        scale = np.asarray(params["scale"], dtype=np.float32)
        params["translation"] = ((translation - center) * normalize_scale).tolist()
        params["scale"] = (scale * normalize_scale).tolist()

    return scene_mesh, posed_meshes, layout_params
