import argparse
import math
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", ""))
os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

from simulate_scene_physics import (
    load_layout,
    normalize_object_id
)

R_OPENGL_TO_P3D = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
R_P3D_TO_OPENGL = R_OPENGL_TO_P3D


def load_object_ids(layout_dir, unposed_dir, requested_object_ids=None):
    layout_ids = {p.stem for p in layout_dir.glob("*.json")}
    mesh_ids = {p.stem for p in unposed_dir.glob("*.glb")}
    matched_ids = sorted(layout_ids & mesh_ids)

    normalized_requested_ids = resolve_requested_object_ids(requested_object_ids)
    if normalized_requested_ids is None:
        return matched_ids

    missing_ids = [object_id for object_id in normalized_requested_ids if object_id not in matched_ids]
    if missing_ids:
        normalized_requested_ids = [obj_id for obj_id in normalized_requested_ids if obj_id not in missing_ids]
        if not normalized_requested_ids:
            raise FileNotFoundError(
                f"Requested object ids not found in both layout and unposed directories: {missing_ids}. Matched ids: {matched_ids}"
            )
        else:
            print(f"Warning: Requested object ids not found in both layout and unposed directories: {missing_ids}. Matched ids: {matched_ids}")
    return normalized_requested_ids


def resolve_requested_object_ids(requested_object_ids):
    if requested_object_ids is None:
        return None
    if isinstance(requested_object_ids, (str, int)):
        requested_object_ids = [requested_object_ids]
    return [normalize_object_id(object_id) for object_id in requested_object_ids]


def parse_camera_direction(value):
    direction = np.asarray(value, dtype=np.float32).reshape(-1)
    if direction.size != 3:
        raise ValueError(f"Camera direction must have 3 elements, got shape {direction.shape}.")
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        raise ValueError("Camera direction norm is zero.")
    return direction / norm


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
    rotation_m = quaternion_to_matrix(rotation_q).transpose(-1, -2)
    l2c_transform = compose_transform(
        scale=scale,
        rotation=rotation_m,
        translation=translation,
    )
    vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
    mesh.vertices = vertices.squeeze(0).cpu().numpy()
    # mesh.vertices = vertices.squeeze(0).cpu().numpy() @ R_P3D_TO_OPENGL
    return mesh


def compose_scene(layout_dir, unposed_dir, requested_object_ids=None, anchor_id=None):
    object_ids = load_object_ids(layout_dir, unposed_dir, requested_object_ids=requested_object_ids)
    if not object_ids:
        raise FileNotFoundError(
            f"No matching object files found in {layout_dir} and {unposed_dir}."
        )

    posed_meshes = []
    layout_params = []
    for object_id in object_ids:
        layout_path = layout_dir / f"{object_id}.json"
        mesh_path = unposed_dir / f"{object_id}.glb"

        layout, _, _, _ = load_layout(layout_path)
        mesh = trimesh.load(mesh_path, force="mesh")
        posed_mesh = apply_layout_to_unposed_mesh(mesh, layout)
        posed_meshes.append(posed_mesh)
        layout_params.append(deepcopy(layout))

    if len(posed_meshes) == 1:
        scene_mesh = posed_meshes[0]
    else:
        scene_mesh = trimesh.Scene()
        for idx, posed_mesh in enumerate(posed_meshes):
            scene_mesh.add_geometry(posed_mesh, node_name=f"object_{idx:03d}")

    scene_mesh, posed_meshes, layout_params, transform_info = normalize_scene_to_canonical_cube_with_info(
        scene_mesh,
        posed_meshes,
        layout_params,
        cube_size=1.0,
        in_place=False,
        anchor_id=anchor_id,
        object_ids=object_ids,
    )
    return scene_mesh, posed_meshes, layout_params, object_ids, transform_info


def normalize_scene_to_canonical_cube_with_info(
    scene_mesh,
    posed_meshes,
    layout_params,
    cube_size=1.0,
    in_place=False,
    anchor_id=None,
    object_ids=None,
):
    """Normalize scene to canonical cube and return transform info."""
    if scene_mesh is None:
        return scene_mesh, posed_meshes, layout_params, {}

    if not in_place:
        scene_mesh = deepcopy(scene_mesh)
        posed_meshes = [deepcopy(mesh) for mesh in posed_meshes]
        layout_params = deepcopy(layout_params)

    rotation_fix = None
    if anchor_id is not None:
        if not object_ids:
            raise ValueError("anchor_id requires object_ids to be provided.")
        normalized_anchor_id = normalize_object_id(anchor_id)
        if normalized_anchor_id not in object_ids:
            raise FileNotFoundError(
                f"anchor_id {normalized_anchor_id} not found in object ids: {object_ids}"
            )
        anchor_index = object_ids.index(normalized_anchor_id)
        anchor_rotation = layout_params[anchor_index].get("rotation")
        if anchor_rotation is None:
            raise ValueError(f"anchor_id {normalized_anchor_id} has no rotation in layout params.")

        anchor_rotation = np.asarray(anchor_rotation, dtype=np.float32).reshape(-1)
        if anchor_rotation.size != 4:
            raise ValueError(
                f"anchor_id {normalized_anchor_id} rotation must have 4 elements, got shape {anchor_rotation.shape}."
            )

        # Convert to row-major order for correct application to vertices and translations.
        anchor_rot = (
            quaternion_to_matrix(torch.from_numpy(anchor_rotation))
            .transpose(-1, -2)
            .cpu()
            .numpy()
        )
        rotation_fix = anchor_rot.T.astype(np.float32)

        for mesh in posed_meshes:
            if mesh.vertices.size:
                mesh.vertices = mesh.vertices.astype(np.float32) @ rotation_fix

        if isinstance(scene_mesh, trimesh.Trimesh):
            scene_mesh.vertices = scene_mesh.vertices.astype(np.float32) @ rotation_fix

        rotation_fix_torch = torch.from_numpy(rotation_fix).float()
        for params in layout_params:
            if "translation" in params:
                translation = np.asarray(params["translation"], dtype=np.float32).reshape(3)
                params["translation"] = (translation @ rotation_fix).tolist()
            if "rotation" in params:
                rot_q = np.asarray(params["rotation"], dtype=np.float32).reshape(-1)
                if rot_q.size != 4:
                    raise ValueError(
                        f"rotation must have 4 elements, got shape {rot_q.shape}."
                    )
                rot_mat = quaternion_to_matrix(torch.from_numpy(rot_q)).transpose(-1, -2)
                rot_mat = torch.matmul(rot_mat, rotation_fix_torch)
                rot_q = matrix_to_quaternion(rot_mat.transpose(-1, -2))
                params["rotation"] = rot_q.cpu().numpy().reshape(-1).tolist()

    all_vertices_p3d = []
    for mesh in posed_meshes:
        # all_vertices_p3d.append(mesh.vertices.astype(np.float32) @ R_OPENGL_TO_P3D)
        all_vertices_p3d.append(mesh.vertices.astype(np.float32))
    if not all_vertices_p3d:
        return scene_mesh, posed_meshes, layout_params, {}

    scene_vertices_p3d = np.vstack(all_vertices_p3d)
    bbox_min = scene_vertices_p3d.min(axis=0)
    bbox_max = scene_vertices_p3d.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        return scene_mesh, posed_meshes, layout_params, {}

    normalize_scale = float(cube_size) / max_extent

    for mesh in posed_meshes:
        # vertices_p3d = mesh.vertices.astype(np.float32) @ R_OPENGL_TO_P3D
        # mesh.vertices = ((vertices_p3d - center) * normalize_scale) @ R_P3D_TO_OPENGL
        vertices_p3d = mesh.vertices.astype(np.float32)
        mesh.vertices = ((vertices_p3d - center) * normalize_scale)

    if isinstance(scene_mesh, trimesh.Trimesh):
        # scene_mesh.vertices = ((scene_vertices_p3d - center) * normalize_scale) @ R_P3D_TO_OPENGL
        scene_mesh.vertices = ((scene_vertices_p3d - center) * normalize_scale)

    for params in layout_params:
        translation = np.asarray(params["translation"], dtype=np.float32)
        scale = np.asarray(params["scale"], dtype=np.float32)
        params["translation"] = ((translation - center) * normalize_scale).tolist()
        params["scale"] = (scale * normalize_scale).tolist()

    transform_info = {
        "center": center.tolist(),
        "normalize_scale": float(normalize_scale),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "max_extent": float(max_extent),
    }
    if rotation_fix is not None:
        transform_info["rotation_matrix"] = rotation_fix.tolist()
        transform_info["anchor_id"] = normalize_object_id(anchor_id)
    return scene_mesh, posed_meshes, layout_params, transform_info


def denormalize_point(point_normalized, transform_info):
    """Convert a normalized point back to original 3D coordinates."""
    center = np.asarray(transform_info["center"], dtype=np.float32)
    normalize_scale = transform_info["normalize_scale"]
    point = np.asarray(point_normalized, dtype=np.float32).reshape(3)
    # Reverse the normalization: point_original = (point_normalized / scale + center)
    original_point = (point / normalize_scale) + center
    rotation = transform_info.get("rotation_matrix")
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        original_point = original_point @ rotation.T
    return original_point


def normalize_point(point_original, transform_info):
    """Convert an original point to the normalized canonical-cube coordinates."""
    center = np.asarray(transform_info["center"], dtype=np.float32)
    normalize_scale = float(transform_info["normalize_scale"])
    point = np.asarray(point_original, dtype=np.float32).reshape(3)
    rotation = transform_info.get("rotation_matrix")
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        point = point @ rotation
    return (point - center) * normalize_scale


def estimate_grid_axes_from_mapping(mapping, grid_size):
    """Estimate grid right/up axes from grid mapping points."""
    axis_right = None
    axis_up = None
    for i in range(grid_size):
        for j in range(grid_size):
            key = (i, j)
            if key not in mapping:
                continue
            if axis_right is None and (i + 1, j) in mapping:
                axis_right = np.asarray(mapping[(i + 1, j)], dtype=np.float32) - np.asarray(
                    mapping[key], dtype=np.float32
                )
                if np.linalg.norm(axis_right) <= 1e-8:
                    axis_right = None
                else:
                    axis_right = axis_right / np.linalg.norm(axis_right)
            if axis_up is None and (i, j + 1) in mapping:
                axis_up = np.asarray(mapping[(i, j + 1)], dtype=np.float32) - np.asarray(
                    mapping[key], dtype=np.float32
                )
                if np.linalg.norm(axis_up) <= 1e-8:
                    axis_up = None
                else:
                    axis_up = axis_up / np.linalg.norm(axis_up)
            if axis_right is not None and axis_up is not None:
                return axis_right, axis_up
    return axis_right, axis_up


def load_grid_mapping(mapping_path, transform_info):
    """Load a grid mapping JSON and convert points into current normalized space."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid_points = data.get("grid_points", [])
    if not grid_points:
        raise ValueError(f"Grid mapping file has no grid_points: {mapping_path}")

    mapping = {}
    max_i = -1
    max_j = -1
    for entry in grid_points:
        if "grid_i" not in entry or "grid_j" not in entry:
            raise ValueError("Grid mapping entry is missing grid_i/grid_j.")
        grid_i = int(entry["grid_i"])
        grid_j = int(entry["grid_j"])
        if "point_3d_original" in entry:
            point_normalized = normalize_point(entry["point_3d_original"], transform_info)
        elif "point_3d_normalized" in entry:
            point_normalized = np.asarray(entry["point_3d_normalized"], dtype=np.float32).reshape(3)
        else:
            raise ValueError("Grid mapping entry is missing point_3d_original/point_3d_normalized.")

        mapping[(grid_i, grid_j)] = point_normalized.tolist()
        max_i = max(max_i, grid_i)
        max_j = max(max_j, grid_j)

    grid_size = int(data.get("grid_size", max(max_i, max_j) + 1))
    axis_right, axis_up = estimate_grid_axes_from_mapping(mapping, grid_size)
    if axis_right is not None:
        axis_right = -np.asarray(axis_right, dtype=np.float32)
    return mapping, grid_size, axis_right, axis_up


def get_plane_from_camera(camera_direction, posed_meshes, plane_side="farthest"):
    """
    Find the plane perpendicular to camera direction.
    Use the farthest or nearest point from camera along the camera direction.
    Returns (plane_normal, plane_d) in the form: normal·p + d = 0
    """
    camera_dir = parse_camera_direction(camera_direction)
    
    # Find the farthest point along camera direction
    all_vertices = []
    for mesh in posed_meshes:
        all_vertices.append(mesh.vertices.astype(np.float32))
    
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        distances_along_dir = np.dot(all_vertices, camera_dir)
        if plane_side == "nearest":
            plane_distance = np.min(distances_along_dir)
        else:
            plane_distance = np.max(distances_along_dir)
        # Plane perpendicular to camera direction at the selected extreme point
        plane_normal = camera_dir
        plane_d = -plane_distance
    else:
        plane_normal = camera_dir
        plane_d = 0.0
    
    return plane_normal, plane_d


def generate_2d_grid_on_plane(posed_meshes, camera_direction, grid_size, plane_normal=None, plane_d=None, plane_side="farthest"):
    """
    Generate a 2D grid on the plane perpendicular to camera direction.
    Returns: grid_2d, mapping (dict), plane_center, local_x, local_y
    
    grid_2d: (grid_size, grid_size, 2) array of 2D coordinates
    mapping: dict of {(i, j): original_3d_coords}
    """
    if plane_normal is None or plane_d is None:
        plane_normal, plane_d = get_plane_from_camera(camera_direction, posed_meshes, plane_side=plane_side)
    
    camera_dir = parse_camera_direction(camera_direction)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Find scene center and bounds
    all_vertices = []
    for mesh in posed_meshes:
        all_vertices.append(mesh.vertices.astype(np.float32))
    
    if not all_vertices:
        return None, {}
    
    all_vertices = np.vstack(all_vertices)
    
    # Create local coordinate system on the plane
    # plane normal is the "forward" direction
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up_hint, plane_normal)
    if np.linalg.norm(right) <= 1e-8:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(up_hint, plane_normal)
    right = right / np.linalg.norm(right)
    up = np.cross(plane_normal, right)
    up = up / np.linalg.norm(up)
    
    # Project all vertices onto the plane and get bounds
    projected_on_right = np.dot(all_vertices, right)
    projected_on_up = np.dot(all_vertices, up)
    
    min_right = np.min(projected_on_right)
    max_right = np.max(projected_on_right)
    min_up = np.min(projected_on_up)
    max_up = np.max(projected_on_up)
    
    # Grid origin at the center of the projection bounds
    center_right = (min_right + max_right) / 2.0
    center_up = (min_up + max_up) / 2.0
    
    # Find a point on the plane passing through scene center
    scene_center = np.mean(all_vertices, axis=0)
    # Point on plane closest to scene center
    dist_to_plane = np.dot(scene_center, plane_normal) + plane_d
    plane_center = scene_center - dist_to_plane * plane_normal
    
    # Adjust plane center to use the projection center
    plane_center = plane_center + (center_right - np.dot(plane_center, right)) * right
    plane_center = plane_center + (center_up - np.dot(plane_center, up)) * up
    
    # Generate grid
    grid_2d = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    mapping = {}
    
    extent_right = max_right - min_right
    extent_up = max_up - min_up
    
    # Add padding to grid
    padding = 0.1
    grid_min_right = min_right - padding * extent_right
    grid_max_right = max_right + padding * extent_right
    grid_min_up = min_up - padding * extent_up
    grid_max_up = max_up + padding * extent_up
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Normalized grid coordinates [0, 1]
            u = i / max(grid_size - 1, 1)
            v = j / max(grid_size - 1, 1)

            # Flip the horizontal index so (0, 0) appears at the lower-left in the rendered view.
            display_i = grid_size - 1 - i
            
            # Grid 2D coordinates
            grid_r = grid_min_right + u * (grid_max_right - grid_min_right)
            grid_u = grid_min_up + v * (grid_max_up - grid_min_up)
            
            grid_2d[display_i, j] = [grid_r, grid_u]
            
            # 3D point on plane
            point_3d = plane_center + grid_r * right + grid_u * up
            mapping[(display_i, j)] = point_3d.tolist()
    
    return grid_2d, mapping, plane_center, right, up


def get_camera_basis(camera_direction):
    """Return forward, right, and up axes for the given camera direction."""
    forward = parse_camera_direction(camera_direction)
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)

    if np.linalg.norm(right) <= 1e-8:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_hint)

    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return forward, right, up


def get_camera_axes(camera_direction, axis_right=None, axis_up=None):
    """Resolve orthonormal axes with camera_direction as the last axis."""
    if axis_right is not None and axis_up is not None:
        right = np.asarray(axis_right, dtype=np.float32).reshape(3)
        up = np.asarray(axis_up, dtype=np.float32).reshape(3)
        right_norm = np.linalg.norm(right)
        up_norm = np.linalg.norm(up)
        if right_norm > 1e-8:
            right = right / right_norm
        if up_norm > 1e-8:
            up = up / up_norm
        forward = np.cross(right, up)
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-8:
            forward = forward / forward_norm
        else:
            forward, right, up = get_camera_basis(camera_direction)
    else:
        forward, right, up = get_camera_basis(camera_direction)
    return right, up, forward


def quaternion_to_matrix_np(quat):
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat)
    if norm > 1e-8:
        quat = quat / norm
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def euler_from_matrix_xyz(matrix):
    """Return XYZ Euler angles from rotation matrix (R = Rz @ Ry @ Rx)."""
    r00, r10, r20 = matrix[0, 0], matrix[1, 0], matrix[2, 0]
    r21, r22 = matrix[2, 1], matrix[2, 2]
    # Check for Gimbal Lock: sin(y) around -1 or 1
    if r20 < -0.995:
        x = 0.0
        y = math.pi / 2
        z = math.atan2(matrix[0, 1], matrix[1, 1])
    elif r20 > 0.995:
        x = 0.0
        y = -math.pi / 2
        z = -math.atan2(matrix[0, 1], matrix[1, 1])
    else:
        # Standard non-locked case
        x = math.atan2(r21, r22)
        y = math.asin(-r20)
        z = math.atan2(r10, r00)
    return x, y, z


def extract_layout_quaternion(layout):
    rotation = layout.get("rotation")
    if rotation is None:
        raise ValueError("Layout rotation is missing.")
    rotation = np.asarray(rotation, dtype=np.float32).reshape(-1)
    if rotation.size == 4:
        return rotation
    if rotation.size >= 4:
        return rotation[:4]
    raise ValueError(f"Rotation must have 4 elements, got shape {rotation.shape}.")


def resolve_orthographic_camera_frame(meshes, camera_direction, radius, yfov_deg=60.0, padding=0.35):
    """Resolve orthographic camera distance and half-extent, auto-fitting when radius is 0."""
    if radius is not None and float(radius) > 0:
        yfov_rad = np.deg2rad(yfov_deg)
        ymag = float(radius) * np.tan(yfov_rad / 2.0)
        return float(radius), ymag, ymag

    all_vertices = []
    for mesh in meshes:
        all_vertices.append(mesh.vertices.astype(np.float32))

    if not all_vertices:
        raise ValueError("Cannot auto-fit camera frame without any mesh vertices.")

    vertices = np.vstack(all_vertices)
    forward, right, up = get_camera_basis(camera_direction)

    projected_right = np.dot(vertices, right)
    projected_up = np.dot(vertices, up)
    projected_forward = np.dot(vertices, forward)

    half_width = float((np.max(projected_right) - np.min(projected_right)) / 2.0)
    half_height = float((np.max(projected_up) - np.min(projected_up)) / 2.0)
    half_depth = float((np.max(projected_forward) - np.min(projected_forward)) / 2.0)

    half_extent = max(half_width, half_height)
    if half_extent <= 0:
        half_extent = 1e-6

    xmag = half_extent * (1.0 + padding)
    ymag = xmag
    camera_radius = max(half_depth, half_extent) * (1.0 + padding)
    camera_radius = max(camera_radius, half_depth + 1e-6)
    return camera_radius, xmag, ymag


def compute_grid_bounds_from_mapping(mapping, axis_right, axis_up):
    """Compute grid bounds in the plane using the saved grid points mapping."""
    if not mapping:
        raise ValueError("Grid mapping is empty; cannot compute grid bounds.")

    points = np.asarray(list(mapping.values()), dtype=np.float32)
    projected_right = np.dot(points, axis_right)
    projected_up = np.dot(points, axis_up)

    min_right = float(np.min(projected_right))
    max_right = float(np.max(projected_right))
    min_up = float(np.min(projected_up))
    max_up = float(np.max(projected_up))
    grid_size = max(int(np.sqrt(len(mapping))), 1)
    return grid_size, min_right, max_right, min_up, max_up


def compute_object_2d_stats(mesh, axis_right, axis_up, mapping, rotation_degrees=None, use_aabb_for_object_bboxes=False):
    """Compute 2D center/extent/bbox in grid index coordinates using grid mapping."""
    grid_size, min_right, max_right, min_up, max_up = compute_grid_bounds_from_mapping(
        mapping, axis_right, axis_up
    )
    vertices = mesh.vertices.astype(np.float32)
    projected_right = np.dot(vertices, axis_right)
    projected_up = np.dot(vertices, axis_up)

    denom_right = max(max_right - min_right, 1e-8)
    denom_up = max(max_up - min_up, 1e-8)
    scale = float(max(grid_size - 1, 0))

    grid_i = (max_right - projected_right) / denom_right * scale
    grid_j = (projected_up - min_up) / denom_up * scale

    min_i = float(np.min(grid_i))
    max_i = float(np.max(grid_i))
    min_j = float(np.min(grid_j))
    max_j = float(np.max(grid_j))

    center_i = (min_i + max_i) / 2.0
    center_j = (min_j + max_j) / 2.0
    size_i = max_i - min_i
    size_j = max_j - min_j

    stats = {
        "center_2d": [center_i, center_j],
        "size_2d": [size_i, size_j],
        "bbox_2d": [min_i, min_j, max_i, max_j],
    }

    
    if use_aabb_for_object_bboxes:
        stats["obb_corners_2d"] = [
            [min_i, min_j],
            [max_i, min_j],
            [max_i, max_j],
            [min_i, max_j],
        ]
    elif rotation_degrees is not None:
        theta = np.deg2rad(float(rotation_degrees))

        # 1. Compute object vertices in physical space relative to center
        verts_phys = np.stack([projected_right, projected_up], axis=1).astype(np.float32)
        center_phys = np.array([(np.min(projected_right) + np.max(projected_right)) / 2.0,
                                (np.min(projected_up) + np.max(projected_up)) / 2.0], dtype=np.float32)

        rel_phys = verts_phys - center_phys

        # 2. Compute heading and perpendicular vectors in physical space
        heading = np.array([np.sin(theta), -np.cos(theta)], dtype=np.float32)
        perp = np.array([-np.cos(theta), -np.sin(theta)], dtype=np.float32)

        heading_proj = rel_phys @ heading
        perp_proj = rel_phys @ perp

        min_h, max_h = np.min(heading_proj), np.max(heading_proj)
        min_p, max_p = np.min(perp_proj), np.max(perp_proj)

        # 3. Compute OBB corners in physical space
        corners_phys = [
            center_phys + min_h * heading + min_p * perp,
            center_phys + max_h * heading + min_p * perp,
            center_phys + max_h * heading + max_p * perp,
            center_phys + min_h * heading + max_p * perp,
        ]

        # 4. Convert OBB corners back to grid coordinates
        obb_corners_2d = []
        for pt in corners_phys:
            pt_i = (max_right - pt[0]) / denom_right * scale
            pt_j = (pt[1] - min_up) / denom_up * scale
            obb_corners_2d.append([float(pt_i), float(pt_j)])

        stats["obb_corners_2d"] = obb_corners_2d

    return stats


def estimate_arrow_length_to_bbox(bbox_2d):
    """Estimate arrow length in pixels until it hits the bbox boundary."""
    if bbox_2d is not None and len(bbox_2d) >= 2:
        pts = np.asarray(bbox_2d, dtype=np.float32)
        if pts.ndim >= 2 and len(pts) >= 2:
            return float(np.linalg.norm(pts[1] - pts[0]))
    return None


def build_objects_info(
    posed_meshes,
    object_ids,
    layout_params,
    axis_right,
    axis_up,
    mapping,
    camera_direction=None,
    plane_normal=None,
    plane_d=None,
    transform_info=None,
    draw_object_bboxes=False,
    use_aabb_for_object_bboxes=False,
):
    """Build per-object 2D stats and optional orientation/plane distance info."""
    objects_info = []

    if camera_direction is not None:
        basis_right, basis_up, basis_forward = get_camera_axes(
            camera_direction, axis_right=axis_right, axis_up=axis_up
        )
        basis_matrix = np.stack([basis_right, basis_up, basis_forward], axis=1)
    else:
        basis_matrix = None

    for mesh, object_id, layout in zip(posed_meshes, object_ids, layout_params):
        rotation_degrees = None
        if basis_matrix is not None:
            try:
                rotation_q = extract_layout_quaternion(layout)
                rotation_world = quaternion_to_matrix_np(rotation_q)
                rotation_local = basis_matrix.T @ rotation_world @ basis_matrix
                x, y, angle_z = euler_from_matrix_xyz(rotation_local)
                angle_deg = math.degrees(angle_z)
                rotation_degrees = (angle_deg + 180.0) % 360.0 - 180.0
            except ValueError:
                rotation_degrees = 0.0

        if draw_object_bboxes and rotation_degrees is not None:
            stats = compute_object_2d_stats(
                mesh,
                axis_right,
                axis_up,
                mapping,
                rotation_degrees=rotation_degrees,
                use_aabb_for_object_bboxes=use_aabb_for_object_bboxes
            )
        else:
            stats = compute_object_2d_stats(mesh, axis_right, axis_up, mapping)

        obj_info = {
            "object_id": object_id,
            **stats,
        }

        if rotation_degrees is not None:
            obj_info["rotation_degrees"] = float(rotation_degrees)

        if (
            plane_normal is not None
            and plane_d is not None
            and transform_info is not None
            and "translation" in layout
        ):
            center = np.asarray(layout["translation"], dtype=np.float32).reshape(-1)
            if center.size >= 3:
                center = denormalize_point(center[:3], transform_info)
                signed_distance = float(np.dot(plane_normal, center) + plane_d)
                obj_info["signed_distance_to_plane"] = signed_distance

        objects_info.append(obj_info)

    return objects_info


def save_grid_mapping(mapping, output_path, transform_info, posed_meshes=None, object_ids=None, 
                      axis_right=None, axis_up=None, camera_direction=None, layout_params=None,
                      objects_info=None, use_aabb_for_object_bboxes=False):
    """Save the 2D grid to 3D coordinate mapping."""
    # Convert mapping to a format that's easy to serialize
    # Include both normalized and original (denormalized) coordinates
    
    # Keep column-major rotation matrix for correct application to vertices and translations in other tools.
    rotation_matrix = np.asarray(
        transform_info.get("rotation_matrix", np.eye(3)), dtype=np.float32
    ).reshape(3, 3)
    transform_info["rotation_matrix_col_major"] = rotation_matrix.T.tolist()
    mapping_data = {
        "grid_size": int(np.sqrt(len(mapping))),
        "grid_points": [],
        "transform_info": transform_info,
    }
    
    for (i, j), point_3d_normalized in mapping.items():
        point_original = denormalize_point(point_3d_normalized, transform_info)
        mapping_data["grid_points"].append({
            "grid_i": int(i),
            "grid_j": int(j),
            "point_3d_normalized": point_3d_normalized,
            "point_3d_original": point_original.tolist(),
        })

    plane_normal = None
    plane_d = None
    if axis_right is not None and axis_up is not None and mapping:
        plane_normal = np.cross(axis_right, axis_up)
        norm = np.linalg.norm(plane_normal)
        if norm > 1e-8:
            plane_normal = plane_normal / norm
            rotation = transform_info.get("rotation_matrix")
            if rotation is not None:
                rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
                plane_normal = plane_normal @ rotation.T
                norm = np.linalg.norm(plane_normal)
                if norm > 1e-8:
                    plane_normal = plane_normal / norm
            sample_point = np.asarray(next(iter(mapping.values())), dtype=np.float32)
            sample_point = denormalize_point(sample_point, transform_info)
            plane_d = -float(np.dot(plane_normal, sample_point))
            mapping_data["plane"] = {
                "normal": plane_normal.tolist(),
                "d": plane_d,
            }

    if objects_info is not None:
        objects_info = deepcopy(objects_info)
        for obj in objects_info:
            obj.pop("obb_corners_2d", None)
        mapping_data["objects"] = objects_info
    elif posed_meshes and object_ids and axis_right is not None and axis_up is not None and layout_params:
        if plane_normal is None and camera_direction is not None:
            plane_normal, plane_d = get_plane_from_camera(camera_direction, posed_meshes)
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            sample_point = (-float(plane_d)) * plane_normal
            sample_point = denormalize_point(sample_point, transform_info)
            rotation = transform_info.get("rotation_matrix")
            if rotation is not None:
                rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
                plane_normal = plane_normal @ rotation.T
                norm = np.linalg.norm(plane_normal)
                if norm > 1e-8:
                    plane_normal = plane_normal / norm
            plane_d = -float(np.dot(plane_normal, sample_point))
            mapping_data.setdefault(
                "plane",
                {"normal": plane_normal.tolist(), "d": float(plane_d)},
            )

        mapping_data["objects"] = build_objects_info(
            posed_meshes=posed_meshes,
            object_ids=object_ids,
            layout_params=layout_params,
            axis_right=axis_right,
            axis_up=axis_up,
            mapping=mapping,
            camera_direction=camera_direction,
            plane_normal=plane_normal,
            plane_d=plane_d,
            transform_info=transform_info,
        )
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)


def render_scene_with_grid(
    meshes,
    camera_direction,
    radius,
    image_size,
    grid_2d=None,
    mapping=None,
    draw_grid_lines=False,
    draw_object_directions=False,
    draw_object_bboxes=False,
    object_infos=None,
    hide_object_ids=None,
):
    """Render scene with optional 2D grid overlay on the image."""
    scene = pyrender.Scene(bg_color=[0.60, 0.60, 0.60, 1.0], ambient_light=[1.0, 1.0, 1.0])

    for mesh in meshes:
        scene.add(mesh_to_pyrender_mesh(mesh))

    # Use an orthographic camera that either matches the requested radius or auto-fits the scene.
    yfov_deg = 60.0
    camera_radius, xmag, ymag = resolve_orthographic_camera_frame(meshes, camera_direction, radius, yfov_deg=yfov_deg)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    camera_pose = build_camera_pose(camera_direction, camera_radius)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    if color.shape[-1] == 4:
        alpha = color[:, :, 3:4].astype(np.float32) / 255.0
        rgb = color[:, :, :3].astype(np.float32)
        background = np.full_like(rgb, 235.0)
        color = (rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)
    
    # Overlay grid if provided
    if grid_2d is not None and mapping is not None:
        camera_matrix = get_camera_matrix(yfov_deg, image_size, ortho=True, xmag=xmag, ymag=ymag)
        color = overlay_2d_grid(
            color,
            grid_2d,
            mapping,
            camera_pose,
            image_size,
            camera_matrix=camera_matrix,
            draw_grid_lines=draw_grid_lines,
            draw_object_directions=draw_object_directions,
            draw_object_bboxes=draw_object_bboxes,
            object_infos=object_infos,
            hide_object_ids=hide_object_ids,
        )
    
    return color


def project_3d_to_2d(point_3d, camera_pose, camera_matrix, image_size):
    """
    Project a 3D point to 2D image coordinates.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        camera_pose: 4x4 camera-to-world transformation matrix
        camera_matrix: 3x3 camera intrinsic matrix
        image_size: size of the image (square)
    
    Returns:
        (x, y) in image coordinates, or None if point is behind camera
    """
    point_3d = np.asarray(point_3d, dtype=np.float32).reshape(3, 1)

    # camera_pose is camera_T_world: camera position and axes in world frame
    # We need world_T_camera to transform world points to camera coordinates
    camera_pose_inv = np.linalg.inv(camera_pose.astype(np.float32))

    # Transform from world to camera coordinates
    point_camera = camera_pose_inv[:3, :3] @ point_3d + camera_pose_inv[:3, 3:4]

    # Check if point is in front of camera (not behind)
    # In OpenGL/pyrender: camera looks along -Z, so Z >= 0 means behind camera (not visible)
    if point_camera[2, 0] >= 0:
        return None

    # Support both perspective and orthographic camera_matrix usages.
    # If camera_matrix is a tuple (matrix, meta) where meta contains 'ortho', handle orthographic mapping.
    ortho_meta = None
    if isinstance(camera_matrix, tuple) or isinstance(camera_matrix, list):
        mat = camera_matrix[0]
        ortho_meta = camera_matrix[1]
        camera_matrix = mat

    if ortho_meta is not None and ortho_meta.get("ortho", False):
        # Orthographic projection mapping
        xmag = float(ortho_meta["xmag"])
        ymag = float(ortho_meta["ymag"])

        x_camera = float(point_camera[0, 0])
        y_camera = float(point_camera[1, 0])

        # Map from camera coordinates to image pixels
        u = (x_camera / (2.0 * xmag) + 0.5) * image_size
        v = (0.5 - y_camera / (2.0 * ymag)) * image_size

        if u < 0 or u >= image_size or v < 0 or v >= image_size:
            return None
        return (int(round(u)), int(round(v)))
    else:
        # Perspective projection (existing behavior)
        point_2d_homo = camera_matrix @ point_camera
        x = point_2d_homo[0, 0] / point_2d_homo[2, 0]
        y = point_2d_homo[1, 0] / point_2d_homo[2, 0]

        # Check if point is within image bounds
        if x < 0 or x >= image_size or y < 0 or y >= image_size:
            return None
        return (int(round(x)), int(round(y)))


def get_camera_matrix(yfov_deg, image_size, ortho=False, radius=None, xmag=None, ymag=None):
    """
    Compute camera intrinsic matrix. If ortho=True, return a tuple (matrix, meta)
    where meta contains orthographic params 'ortho', 'xmag', 'ymag'.
    """
    yfov_rad = np.deg2rad(yfov_deg)
    cx = image_size / 2.0
    cy = image_size / 2.0

    if ortho:
        if xmag is None or ymag is None:
            if radius is None:
                raise ValueError("radius or xmag/ymag must be provided for orthographic camera matrix calculation")
            # compute half-height in world units that matches perspective view at distance `radius`
            ymag = float(radius) * np.tan(yfov_rad / 2.0)
            xmag = ymag
        else:
            xmag = float(xmag)
            ymag = float(ymag)
        # scale to pixels: s = image_size / (2 * xmag)
        s = (image_size / (2.0 * xmag)) if xmag != 0 else 1.0
        # Use negative scale on y to map world-up to image-down
        camera_matrix = np.array([
            [s, 0, cx],
            [0, -s, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        meta = {"ortho": True, "xmag": xmag, "ymag": ymag}
        return (camera_matrix, meta)

    # fallback to perspective intrinsics
    f = (image_size / 2.0) / np.tan(yfov_rad / 2.0)
    camera_matrix = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return camera_matrix


def overlay_2d_grid(image, grid_2d, mapping, camera_pose, image_size, grid_line_color=(0, 0, 255),
                     grid_point_color=(0, 0, 255), grid_point_radius=5, radius=None, camera_matrix=None,
                     draw_grid_lines=False, draw_object_directions=False, draw_object_bboxes=False,
                     object_infos=None, hide_object_ids=None):
    """
    Overlay 2D grid points on the rendered image.
    """
    # Convert RGB to BGR for cv2 drawing operations
    image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    # Use a provided camera matrix when available; otherwise fall back to the existing behavior.
    if camera_matrix is None and radius is not None:
        camera_matrix = get_camera_matrix(60.0, image_size, ortho=True, radius=radius)
    elif camera_matrix is None:
        camera_matrix = get_camera_matrix(60.0, image_size)
    
    # Project all grid points to 2D
    grid_size = grid_2d.shape[0]
    projected_points = {}
    
    for (i, j), point_3d in mapping.items():
        proj = project_3d_to_2d(point_3d, camera_pose, camera_matrix, image_size)
        if proj is not None:
            projected_points[(i, j)] = proj
    
    if not projected_points:
        print("[WARN] No grid points were successfully projected!")
        return image

    def grid_coord_to_image(grid_i, grid_j):
        if grid_size <= 1:
            return None
        required = [(0, 0), (grid_size - 1, 0), (0, grid_size - 1)]
        if any(key not in projected_points for key in required):
            return None
        p00 = np.asarray(projected_points[(0, 0)], dtype=np.float32)
        p10 = np.asarray(projected_points[(grid_size - 1, 0)], dtype=np.float32)
        p01 = np.asarray(projected_points[(0, grid_size - 1)], dtype=np.float32)
        u = float(grid_i) / float(grid_size - 1)
        v = float(grid_j) / float(grid_size - 1)
        point = p00 + u * (p10 - p00) + v * (p01 - p00)
        return int(round(float(point[0]))), int(round(float(point[1])))
    
    # Draw grid lines if enabled
    if draw_grid_lines:
        for i in range(grid_size):
            # Horizontal lines (along j direction)
            line_points = []
            for j in range(grid_size):
                if (i, j) in projected_points:
                    line_points.append(projected_points[(i, j)])
            
            # Draw connected line segments
            for k in range(len(line_points) - 1):
                cv2.line(image_bgr, line_points[k], line_points[k + 1], grid_line_color, 1)
        
        for j in range(grid_size):
            # Vertical lines (along i direction)
            line_points = []
            for i in range(grid_size):
                if (i, j) in projected_points:
                    line_points.append(projected_points[(i, j)])
            
            # Draw connected line segments
            for k in range(len(line_points) - 1):
                cv2.line(image_bgr, line_points[k], line_points[k + 1], grid_line_color, 1)
    
    # Draw grid points and coordinates
    for (i, j), (x, y) in projected_points.items():
        cv2.circle(image_bgr, (x, y), grid_point_radius, grid_point_color, -1)
        # Draw coordinate text below the point
        text = f"({i},{j})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + grid_point_radius + 20
        cv2.putText(image_bgr, text, (text_x, text_y), font, font_scale, grid_point_color, font_thickness)

    # Draw per-object overlays in image space.
    if object_infos:
        normalized_hide_object_ids = set(resolve_requested_object_ids(hide_object_ids) or [])
        required = [(0, 0), (grid_size - 1, 0), (0, grid_size - 1)]
        if all(key in projected_points for key in required) and grid_size > 1:
            for obj_info in object_infos:
                object_id = normalize_object_id(obj_info.get("object_id", ""))
                hide_object = object_id in normalized_hide_object_ids

                center_2d = obj_info.get("center_2d")
                size_2d = obj_info.get("size_2d")
                angle_deg = float(obj_info.get("rotation_degrees", 0.0))
                if center_2d is None or size_2d is None:
                    continue

                center_px = grid_coord_to_image(center_2d[0], center_2d[1])
                if center_px is None:
                    continue

                corners = obj_info.get("obb_corners_2d")
                pts = [grid_coord_to_image(pt[0], pt[1]) for pt in corners]
                if draw_object_bboxes and not hide_object:
                    if all(pt is not None for pt in pts):
                        pts = np.asarray(pts, dtype=np.int32)
                        cv2.polylines(image_bgr, [pts], True, (139, 0, 0), 4)

                        label_text = object_id.replace("object", "obj")
                        label_font = cv2.FONT_HERSHEY_SIMPLEX
                        label_font_scale = 0.7
                        label_thickness = 2
                        label_size = cv2.getTextSize(label_text, label_font, label_font_scale, label_thickness)[0]
                        label_margin = 6
                        label_x = int(np.max(pts[:, 0]) - label_size[0])
                        label_y = int(np.min(pts[:, 1]) - label_margin)
                        if label_y < label_size[1] + label_margin:
                            label_y = int(np.max(pts[:, 1]) + label_size[1] + label_margin)
                        label_x = max(0, min(label_x, image_bgr.shape[1] - label_size[0] - 1))
                        label_y = max(label_size[1] + label_margin, min(label_y, image_bgr.shape[0] - label_margin))
                        cv2.putText(image_bgr, label_text, (label_x, label_y), label_font, label_font_scale, (0, 0, 0), label_thickness)

                if not draw_object_directions or object_id.lower().split("_")[0] in ["floor", "wall", "ceiling"]:
                    continue

                arrow_length = estimate_arrow_length_to_bbox(bbox_2d=pts) * 1.25
                if arrow_length is None:
                    continue
                else:
                    arrow_length = min(max(arrow_length, 40.0), 400.0)

                # 0 degrees points downward (-v in image), positive angle rotates clockwise.
                theta = np.deg2rad(angle_deg)
                fixed_head_px = 12.0
                dx = -arrow_length * np.sin(theta)
                dy = arrow_length * np.cos(theta)

                start = (int(round(center_px[0])), int(round(center_px[1])))
                end = (
                    int(round(center_px[0] + dx)),
                    int(round(center_px[1] + dy)),
                )

                cv2.arrowedLine(
                    image_bgr,
                    start,
                    end,
                    (255, 128, 0),
                    4,
                    tipLength=fixed_head_px / arrow_length,
                )
    
    # Convert back to RGB for consistency
    result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return result


def build_camera_pose(camera_direction, radius):
    """Build camera pose matrix for PyRender."""
    forward = parse_camera_direction(camera_direction)
    scene_center = np.zeros(3, dtype=np.float32)
    camera_position = scene_center - forward * float(radius)

    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)  # forward × up_hint (correct order)
    
    if np.linalg.norm(right) <= 1e-8:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_hint)  # same order
    
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)  # right × forward
    true_up = true_up / np.linalg.norm(true_up)

    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = true_up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = camera_position
    
    return camera_pose


def mesh_to_pyrender_mesh(mesh):
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def main():
    parser = argparse.ArgumentParser(
        description="Render a canonical-cube normalized scene from unposed meshes and layout parameters."
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        default=Path("outputs/generated/group_object_001/layout_org/layout_params_simulated"),
        help="Directory containing object_XXX.json layout files.",
    )
    parser.add_argument(
        "--unposed_dir",
        type=Path,
        default=Path("outputs/unposed_objects"),
        help="Directory containing object_XXX.glb unposed meshes.",
    )
    parser.add_argument(
        "--object_ids",
        nargs="+",
        default=None,
        help="Optional subset of object ids to render (XXX in object_XXX). Defaults to all matched objects.",
    )
    parser.add_argument(
        "--anchor_id",
        type=str,
        default=None,
        help="Optional anchor object id to zero its rotation during normalization.",
    )
    parser.add_argument(
        "--hide_object_ids",
        nargs="+",
        default=None,
        help="Optional object ids to hide bbox overlays on the render.",
    )
    parser.add_argument(
        "--camera_direction",
        type=float,
        nargs=3,
        default=[0.0, -1.0, 0.0],
        help="Camera look direction in world XYZ. Default is straight down.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Camera distance from the canonical-cube scene center. Set 0 to auto-fit the scene.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rendered_scene.png"),
        help="Output path for the rendered image.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Square output image size in pixels.",
    )
    parser.add_argument(
        "--add_grid",
        action="store_true",
        help="If set, generate 2D grid points and save mapping to JSON.",
    )
    parser.add_argument(
        "--grid_mapping",
        type=Path,
        default=None,
        help="Optional JSON grid mapping from a previous render to reuse grid points.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=5,
        help="Number of grid points per side (grid_size x grid_size).",
    )
    parser.add_argument(
        "--draw_grid_lines",
        action="store_true",
        help="If set, draw lines connecting grid points.",
    )
    parser.add_argument(
        "--draw_object_directions",
        action="store_true",
        help="If set, draw non-anchor object direction arrows using obj_info center/size/rotation.",
    )
    parser.add_argument(
        "--draw_object_bboxes",
        action="store_true",
        help="If set, draw 2D oriented bounding boxes for objects on the overlay.",
    )
    parser.add_argument(
        "--use_aabb_for_object_bboxes",
        action="store_true",
        help="If set, use axis-aligned bounding boxes instead of oriented bounding boxes for object overlays.",
    )
    parser.add_argument(
        "--plane_side",
        choices=["farthest", "nearest"],
        default="nearest",
        help="Choose whether grid/mapping uses the farthest or nearest plane along the camera direction.",
    )
    args = parser.parse_args()

    layout_dir = args.layout_dir
    unposed_dir = args.unposed_dir
    output_path = args.output

    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {layout_dir}")
    if not unposed_dir.exists():
        raise FileNotFoundError(f"Unposed mesh directory not found: {unposed_dir}")
    if args.radius < 0:
        raise ValueError("--radius must be non-negative.")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    if args.grid_size <= 0:
        raise ValueError("--grid_size must be positive.")
    if args.draw_object_directions and not args.add_grid:
        raise ValueError("--draw_object_directions requires --add_grid.")
    if args.draw_object_bboxes and not args.add_grid:
        raise ValueError("--draw_object_bboxes requires --add_grid.")

    _, posed_meshes, layout_params, object_ids, transform_info = compose_scene(
        layout_dir=layout_dir,
        unposed_dir=unposed_dir,
        requested_object_ids=args.object_ids,
        anchor_id=args.anchor_id,
    )

    # Generate grid if requested
    grid_2d = None
    mapping = None
    plane_right = None
    plane_up = None
    object_infos_for_overlay = None
    if args.add_grid:
        if args.grid_mapping is not None:
            if not args.grid_mapping.exists():
                raise FileNotFoundError(f"Grid mapping file not found: {args.grid_mapping}")
            mapping, grid_size, plane_right, plane_up = load_grid_mapping(
                args.grid_mapping, transform_info
            )
            grid_2d = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
        else:
            grid_2d, mapping, _, plane_right, plane_up = generate_2d_grid_on_plane(
                posed_meshes, args.camera_direction, args.grid_size, plane_side=args.plane_side
            )

    if args.draw_object_directions or args.draw_object_bboxes:
        if mapping is None or plane_right is None or plane_up is None:
            raise ValueError("Object overlays require grid mapping and plane axes.")
        # Compute plane_normal and plane_d in original (denormalized) space so that
        # build_objects_info can populate signed_distance_to_plane.
        computed_plane_normal = np.cross(plane_right, plane_up)
        computed_plane_d = None
        norm = np.linalg.norm(computed_plane_normal)
        if norm > 1e-8:
            computed_plane_normal = computed_plane_normal / norm
            rotation = transform_info.get("rotation_matrix")
            if rotation is not None:
                rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
                computed_plane_normal = computed_plane_normal @ rotation.T
                norm = np.linalg.norm(computed_plane_normal)
                if norm > 1e-8:
                    computed_plane_normal = computed_plane_normal / norm
            sample_point = np.asarray(next(iter(mapping.values())), dtype=np.float32)
            sample_point = denormalize_point(sample_point, transform_info)
            computed_plane_d = -float(np.dot(computed_plane_normal, sample_point))

        object_infos_for_overlay = build_objects_info(
            posed_meshes=posed_meshes,
            object_ids=object_ids,
            layout_params=layout_params,
            axis_right=plane_right,
            axis_up=plane_up,
            mapping=mapping,
            camera_direction=args.camera_direction,
            plane_normal=computed_plane_normal,
            plane_d=computed_plane_d,
            transform_info=transform_info,
            draw_object_bboxes=args.draw_object_bboxes,
            use_aabb_for_object_bboxes=args.use_aabb_for_object_bboxes,
        )

    rendered = render_scene_with_grid(
        meshes=posed_meshes,
        camera_direction=args.camera_direction,
        radius=args.radius,
        image_size=args.image_size,
        grid_2d=grid_2d,
        mapping=mapping,
        draw_grid_lines=args.draw_grid_lines,
        draw_object_directions=args.draw_object_directions,
        draw_object_bboxes=args.draw_object_bboxes,
        object_infos=object_infos_for_overlay,
        hide_object_ids=args.hide_object_ids,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write rendered image to {output_path}")
    print(f"Saved rendered scene to: {output_path} (objects: {object_ids})")

    # Save grid mapping if generated
    if args.add_grid and mapping is not None:
        mapping_output = output_path.with_suffix(".json")
        save_grid_mapping(
            mapping,
            mapping_output,
            transform_info,
            posed_meshes=posed_meshes,
            object_ids=object_ids,
            axis_right=plane_right,
            axis_up=plane_up,
            camera_direction=args.camera_direction,
            layout_params=layout_params,
            objects_info=object_infos_for_overlay,
            use_aabb_for_object_bboxes=args.use_aabb_for_object_bboxes,
        )
        print(f"Saved grid mapping to: {mapping_output}")


if __name__ == "__main__":
    main()