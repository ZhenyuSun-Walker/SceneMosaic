import argparse
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

from trimesh.visual.material import PBRMaterial

R_OPENGL_TO_P3D = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
R_P3D_TO_OPENGL = R_OPENGL_TO_P3D

DEFAULT_TEXTURE_RESOLUTION = "2K"
DEFAULT_TEXTURE_FORMAT = "PNG"
TEXTURE_RESOLUTIONS = ("1K", "2K", "4K", "8K", "12K", "16K")
TEXTURE_FORMATS = ("PNG", "JPG")


def env_texture_resolution() -> str:
    """Resolve the texture resolution from the environment (default 2K)."""
    value = os.environ.get("SCENEMOSAIC_MATERIAL_RESOLUTION")
    return value if value in TEXTURE_RESOLUTIONS else DEFAULT_TEXTURE_RESOLUTION


def env_texture_format() -> str:
    """Resolve the texture format from the environment (default PNG)."""
    value = os.environ.get("SCENEMOSAIC_MATERIAL_FORMAT")
    return value if value in TEXTURE_FORMATS else DEFAULT_TEXTURE_FORMAT


def texture_path(
    material_dir: Path,
    channel: str,
    *,
    resolution: str | None = None,
    file_format: str | None = None,
) -> Path:
    """Return the path of one ambientCG texture of ``material_dir``.

    ambientCG names textures ``<material>_<res>-<fmt>_<channel>.<ext>`` where
    the filename label uses the format in uppercase (e.g. ``2K-PNG_Color.png``
    for PNG, ``2K-JPG_Color.jpg`` for JPG). ``channel`` is the map suffix such
    as ``Color``, ``NormalGL`` or ``Roughness``. Unset arguments fall back to
    the environment overrides, then to 2K PNG.
    """
    resolution = resolution or env_texture_resolution()
    file_format = file_format or env_texture_format()
    extension = file_format.lower()
    return material_dir / (
        f"{material_dir.name}_{resolution}-{file_format}_{channel}.{extension}"
    )


def load_image(path: Path, mode: str = "RGB"):
    with Image.open(path) as image:
        return image.convert(mode)


def quaternion_to_matrix_numpy(quat):
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have 4 elements.")
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
    raise ValueError("Scale must be scalar or 3D vector.")


def yaw_to_quaternion(yaw: float) -> list:
    half = -0.5 * yaw
    return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]


def rotate_points_xz(points: np.ndarray, yaw: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return pts @ rot.T


def _signed_area_xz(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    z = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def normalize_wall_loop_winding(walls: list[dict]) -> list[dict]:
    if len(walls) < 3:
        return list(walls)

    walls.sort(key=lambda w: w.get("id", ""))
    loop_points = np.asarray([wall["first_point"] for wall in walls], dtype=np.float32)
    # Use clockwise winding for a consistent outward-normal convention.
    if _signed_area_xz(loop_points) >= 0.0:
        print("Wall loop is already clockwise, no need to reverse.")
        return list(walls)

    print("Reversing wall loop winding to be clockwise for consistent outward normals.")

    normalized = []
    for wall in reversed(walls):
        normalized.append(
            {
                **wall,
                "first_point": wall["second_point"],
                "second_point": wall["first_point"],
            }
        )
    return normalized


def load_room_material(
    material_dir: Path,
    *,
    resolution: str | None = None,
    file_format: str | None = None,
) -> PBRMaterial:
    base_color_path = texture_path(
        material_dir, "Color", resolution=resolution, file_format=file_format
    )
    normal_path = texture_path(
        material_dir, "NormalGL", resolution=resolution, file_format=file_format
    )
    roughness_path = texture_path(
        material_dir, "Roughness", resolution=resolution, file_format=file_format
    )

    if not base_color_path.exists():
        raise FileNotFoundError(f"Base color texture not found: {base_color_path}")
    if not normal_path.exists():
        raise FileNotFoundError(f"Normal texture not found: {normal_path}")
    if not roughness_path.exists():
        raise FileNotFoundError(f"Roughness texture not found: {roughness_path}")

    base_color = load_image(base_color_path, mode="RGBA")
    normal_texture = load_image(normal_path, mode="RGB")

    roughness_image = load_image(roughness_path, mode="L")
    roughness_array = np.array(roughness_image, dtype=np.uint8)
    metallic_roughness_array = np.zeros((roughness_array.shape[0], roughness_array.shape[1], 3), dtype=np.uint8)
    metallic_roughness_array[..., 1] = roughness_array
    metallic_roughness = Image.fromarray(metallic_roughness_array, mode="RGB")

    return PBRMaterial(
        roughnessFactor=1.0,
        metallicFactor=0.0,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        baseColorTexture=base_color,
        normalTexture=normal_texture,
        metallicRoughnessTexture=metallic_roughness,
        doubleSided=True,
    )


def apply_layout_to_unposed_mesh(mesh: trimesh.Trimesh, layout: dict) -> trimesh.Trimesh:
    translation = np.asarray(layout.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    rotation = np.asarray(layout.get("rotation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
    scale = parse_scale(layout.get("scale", [1.0]))

    vertices = mesh.vertices.astype(np.float32) @ R_OPENGL_TO_P3D
    # apply scale, rotation, translation
    scaled = vertices * scale.reshape(1, 3)
    R = quaternion_to_matrix_numpy(rotation)
    rotated = scaled @ R.T
    translated = rotated + translation.reshape(1, 3)
    mesh.vertices = translated
    return mesh


def make_textured_mesh(mesh: trimesh.Trimesh, uv: np.ndarray, material: PBRMaterial) -> trimesh.Trimesh:
    textured_mesh = mesh.copy()
    textured_mesh.visual = trimesh.visual.TextureVisuals(uv=uv.astype(np.float32), material=material)
    return textured_mesh


def load_posed_meshes(layout_dir: Path, unposed_dir: Path):
    layout_files = sorted(list(layout_dir.glob("object_*.json")))
    posed = []
    for lf in layout_files:
        obj_id = lf.stem.split("object_", 1)[1]
        mesh_path = unposed_dir / f"object_{obj_id}.glb"
        if not mesh_path.exists():
            continue
        with lf.open("r", encoding="utf-8") as f:
            layout = json.load(f)
        mesh = trimesh.load(mesh_path, force="mesh")
        posed_mesh = apply_layout_to_unposed_mesh(mesh, layout)
        posed.append(posed_mesh)
    return posed


def find_floor_ceiling_heights(posed_meshes, objects_json_path: Path):
    all_vertices = []
    for m in posed_meshes:
        all_vertices.append(m.vertices.astype(np.float32))
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        y_min = float(np.min(all_vertices[:, 1]))
        y_max = float(np.max(all_vertices[:, 1]))
    else:
        y_min, y_max = 0.0, 2.5

    ceiling_h = None
    if objects_json_path.exists():
        with objects_json_path.open("r", encoding="utf-8") as f:
            objs = json.load(f)
        if "ceiling" in objs:
            try:
                ceiling_h = float(objs["ceiling"]["height"])
            except Exception:
                ceiling_h = None

    if ceiling_h is None:
        ceiling_h = y_max

    return y_min, ceiling_h


def compute_wall_planes_from_objects(objects_json_path: Path):
    if not objects_json_path.exists():
        raise FileNotFoundError(f"objects.json not found: {objects_json_path}")
    with objects_json_path.open("r", encoding="utf-8") as f:
        objs = json.load(f)
    walls = normalize_wall_loop_winding(objs.get("walls", []))
    # each wall: has first_point [x,z] and second_point [x,z]
    planes = []
    for w in walls:
        p1 = np.asarray(w["first_point"], dtype=np.float32)
        p2 = np.asarray(w["second_point"], dtype=np.float32)
        dir2 = p2 - p1
        if np.linalg.norm(dir2) <= 1e-8:
            continue
        dir2 = dir2 / np.linalg.norm(dir2)
        # normal in xz plane for clockwise winding.
        normal_xz = np.array([dir2[1], -dir2[0]], dtype=np.float32)
        d = float(np.dot(normal_xz, p1))
        planes.append({"normal_xz": normal_xz, "d": d, "p1": p1, "p2": p2, "wall_id": w.get("id")})
    return sorted(planes, key=lambda pl: pl["wall_id"].split("_")[-1])


def choose_wall_plane_extent(planes, posed_meshes, padding=0.00):
    # For each plane, compute candidate plane at farthest vertex along normal
    all_vertices = []
    for m in posed_meshes:
        all_vertices.append(m.vertices.astype(np.float32)[:, [0, 2]])
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
    else:
        all_vertices = np.zeros((0, 2), dtype=np.float32)

    chosen = []
    for pl in planes:
        nx = pl["normal_xz"]
        d_original = pl["d"]
        if all_vertices.size == 0:
            d_use = d_original
        else:
            projections = all_vertices @ nx
            max_proj = float(np.max(projections))
            # choose farther along normal direction
            d_use = max(d_original, max_proj)
        # Apply padding to push wall further out
        d_use = d_use + padding
        chosen.append({**pl, "d_use": d_use})
        print(f"wall {pl.get('wall_id')} normal={nx.tolist()} d_original={d_original:.6f} max_proj={max_proj:.6f} d_use={d_use:.6f}")
    return chosen


def intersect_two_planes_xz(n1, d1, n2, d2):
    A = np.array([n1, n2], dtype=np.float32)
    b = np.array([d1, d2], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) <= 1e-8:
        return None
    sol = np.linalg.solve(A, b)
    return sol


def build_wall_corners(chosen_planes):
    # assume ordered walls; compute 4 intersections
    corners = []
    N = len(chosen_planes)
    for i in range(N):
        pl1 = chosen_planes[i]
        pl2 = chosen_planes[(i + 1) % N]
        p = intersect_two_planes_xz(pl1["normal_xz"], pl1["d_use"], pl2["normal_xz"], pl2["d_use"])
        if p is None:
            # fallback to midpoint of segment points
            point = np.asarray(pl1["p2"], dtype=np.float32)
        else:
            point = np.asarray(p, dtype=np.float32)
        corners.append({"point": point, "wall_id": pl1.get("wall_id")})
        print(f"Chosen plane p1: {pl1['wall_id']} p2: {pl2['wall_id']}: corner={point.tolist()}, d_use={pl1['d_use']:.3f}")
    return corners


def create_wall_mesh(wall_length, wall_height, thickness=0.02):
    # Local wall mesh: length along +X, height along +Y, thickness along +Z, centered at origin.
    half_l = 0.5 * float(wall_length)
    half_h = 0.5 * float(wall_height)
    half_t = 0.5 * float(thickness)

    bl = np.array([-half_l, -half_h, -half_t], dtype=np.float32)
    br = np.array([half_l, -half_h, -half_t], dtype=np.float32)
    tr = np.array([half_l, half_h, -half_t], dtype=np.float32)
    tl = np.array([-half_l, half_h, -half_t], dtype=np.float32)

    bl2 = np.array([-half_l, -half_h, half_t], dtype=np.float32)
    br2 = np.array([half_l, -half_h, half_t], dtype=np.float32)
    tr2 = np.array([half_l, half_h, half_t], dtype=np.float32)
    tl2 = np.array([-half_l, half_h, half_t], dtype=np.float32)

    verts = np.vstack([
        bl, br, tr, tl,
        bl2, br2, tr2, tl2,
        bl, br, br2, bl2,
        tl, tr, tr2, tl2,
        br, br2, tr2, tr,
        bl, tl, tl2, bl2,
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 7, 6], [4, 6, 5],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    uv = np.array([
        [0.0, 0.0], [wall_length, 0.0], [wall_length, wall_height], [0.0, wall_height],
        [0.0, 0.0], [wall_length, 0.0], [wall_length, wall_height], [0.0, wall_height],
        [0.0, 0.0], [wall_length, 0.0], [wall_length, thickness], [0.0, thickness],
        [0.0, 0.0], [wall_length, 0.0], [wall_length, thickness], [0.0, thickness],
        [0.0, 0.0], [thickness, 0.0], [thickness, wall_height], [0.0, wall_height],
        [0.0, 0.0], [thickness, 0.0], [thickness, wall_height], [0.0, wall_height],
    ], dtype=np.float32)
    return mesh, uv


def create_floor_or_ceiling_mesh(corners_xz, y_height):
    # corners_xz is Nx2 in order (convex). Create vertices and fan triangulation from centroid.
    pts3 = [np.array([p[0], y_height, p[1]], dtype=np.float32) for p in corners_xz]
    centroid = np.mean(np.vstack(pts3), axis=0)
    verts = np.vstack(pts3 + [centroid])
    N = len(pts3)
    faces = []
    for i in range(N):
        a = i
        b = (i + 1) % N
        c = N  # centroid
        faces.append([a, b, c])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=False)

    corners_2d = np.asarray(corners_xz, dtype=np.float32)
    uv_origin = corners_2d.min(axis=0)
    uv = np.vstack([
        corners_2d - uv_origin,
        np.mean(corners_2d - uv_origin, axis=0, keepdims=True),
    ])
    return mesh, uv


def recenter_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    # Center mesh at origin and return the translation offset for layout.
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    mesh.vertices = mesh.vertices - center.reshape(1, 3)
    return center.astype(np.float32)


def save_mesh_and_layout(
    mesh: trimesh.Trimesh,
    name: str,
    unposed_dir: Path,
    layout_dir: Path,
    layout: dict | None = None,
):
    unposed_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = unposed_dir / f"{name}.glb"
    # transform mesh back to OPENGL coordinate system for saving
    mesh.vertices = (mesh.vertices.astype(np.float32) @ R_P3D_TO_OPENGL).astype(np.float32)
    mesh.export(mesh_path)
    if layout is None:
        layout = {"translation": [[0.0, 0.0, 0.0]], "rotation": [[1.0, 0.0, 0.0, 0.0]], "scale": [[1.0, 1.0, 1.0]]}
    with (layout_dir / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate simple wall/floor/ceiling meshes from layout and objects.json")
    parser.add_argument("--layout_dir", type=Path, default=Path("outputs/layout_params"))
    parser.add_argument("--unposed_dir", type=Path, default=Path("outputs/unposed_objects"))
    parser.add_argument("--objects_json", type=Path, default=Path("outputs/objects.json"))
    parser.add_argument(
        "--material_dir",
        type=Path,
        default=Path("data/materials/Marble001"),
        help="Legacy fallback material used when a surface-specific material is not supplied.",
    )
    parser.add_argument("--wall_material_dir", type=Path, default=None, help="PBR material directory for walls.")
    parser.add_argument("--floor_material_dir", type=Path, default=None, help="PBR material directory for the floor.")
    parser.add_argument("--ceiling_material_dir", type=Path, default=None, help="PBR material directory for the ceiling.")
    parser.add_argument("--thickness", type=float, default=0.02)
    parser.add_argument("--wall_padding", type=float, default=0.05, help="Padding distance to push walls outward")
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default=env_texture_resolution(),
        choices=list(TEXTURE_RESOLUTIONS),
        help=(
            "Material texture resolution (default: "
            f"$SCENEMOSAIC_MATERIAL_RESOLUTION or {DEFAULT_TEXTURE_RESOLUTION})"
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default=env_texture_format(),
        choices=list(TEXTURE_FORMATS),
        help=(
            "Material texture file format (default: "
            f"$SCENEMOSAIC_MATERIAL_FORMAT or {DEFAULT_TEXTURE_FORMAT})"
        ),
    )
    args = parser.parse_args()

    posed = load_posed_meshes(args.layout_dir, args.unposed_dir)
    floor_y, ceiling_y = find_floor_ceiling_heights(posed, args.objects_json)
    wall_material = load_room_material(
        args.wall_material_dir or args.material_dir,
        resolution=args.resolution,
        file_format=args.format,
    )
    floor_material = load_room_material(
        args.floor_material_dir or args.material_dir,
        resolution=args.resolution,
        file_format=args.format,
    )
    ceiling_material = load_room_material(
        args.ceiling_material_dir or args.material_dir,
        resolution=args.resolution,
        file_format=args.format,
    )

    planes = compute_wall_planes_from_objects(args.objects_json)
    chosen = choose_wall_plane_extent(planes, posed, padding=args.wall_padding)
    corners = build_wall_corners(chosen)
    if len(corners) == 0:
        raise RuntimeError("No wall corners computed; check objects.json walls entries.")

    corner_points = np.array([corner["point"] for corner in corners], dtype=np.float32)

    # Create walls
    N = corner_points.shape[0]
    for i in range(N):
        a = corner_points[(i - 1) % N]
        b = corner_points[i]
        edge_xz = b - a
        wall_length = float(np.linalg.norm(edge_xz))
        if wall_length <= 1e-8:
            wall_length = 1.0
        wall_height = float(abs(ceiling_y - floor_y))
        yaw = float(np.arctan2(edge_xz[1], edge_xz[0]))
        print(f"Wall {corners[i]['wall_id']}: a={a.tolist()[:2]}, b={b.tolist()[:2]}, yaw={np.degrees(yaw):.1f} deg")
        wall_mesh, wall_uv = create_wall_mesh(wall_length, wall_height, thickness=args.thickness)
        wall_mesh = make_textured_mesh(wall_mesh, wall_uv, wall_material)
        wall_center = np.array([
            0.5 * (a[0] + b[0]),
            0.5 * (floor_y + ceiling_y),
            0.5 * (a[1] + b[1]),
        ], dtype=np.float32)
        wall_layout = {
            "translation": [wall_center.tolist()],
            "rotation": [yaw_to_quaternion(yaw)],
            "scale": [[1.0, 1.0, 1.0]],
        }
        name = corners[i].get("wall_id")
        if not name:
            raise RuntimeError(f"Missing wall_id for corner {i}")
        save_mesh_and_layout(wall_mesh, name, args.unposed_dir, args.layout_dir, layout=wall_layout)

    # Floor and ceiling
    edge0 = corner_points[0] - corner_points[-1]
    room_yaw = float(np.arctan2(edge0[1], edge0[0]))
    corners_local = rotate_points_xz(corner_points, -room_yaw)
    floor_mesh, floor_uv = create_floor_or_ceiling_mesh(corners_local, 0.0)
    floor_mesh = make_textured_mesh(floor_mesh, floor_uv, floor_material)
    floor_center = recenter_mesh(floor_mesh)
    floor_center_xz = rotate_points_xz(floor_center[[0, 2]], room_yaw)
    floor_layout = {
        "translation": [[float(floor_center_xz[0]), float(floor_y + floor_center[1]), float(floor_center_xz[1])]],
        "rotation": [yaw_to_quaternion(room_yaw)],
        "scale": [[1.0, 1.0, 1.0]],
    }
    save_mesh_and_layout(floor_mesh, "floor", args.unposed_dir, args.layout_dir, layout=floor_layout)

    ceiling_mesh, ceiling_uv = create_floor_or_ceiling_mesh(corners_local, 0.0)
    ceiling_mesh = make_textured_mesh(ceiling_mesh, ceiling_uv, ceiling_material)
    ceiling_center = recenter_mesh(ceiling_mesh)
    ceiling_center_xz = rotate_points_xz(ceiling_center[[0, 2]], room_yaw)
    ceiling_layout = {
        "translation": [[float(ceiling_center_xz[0]), float(ceiling_y + ceiling_center[1]), float(ceiling_center_xz[1])]],
        "rotation": [yaw_to_quaternion(room_yaw)],
        "scale": [[1.0, 1.0, 1.0]],
    }
    save_mesh_and_layout(ceiling_mesh, "ceiling", args.unposed_dir, args.layout_dir, layout=ceiling_layout)

    print(f"Generated {N} walls, floor and ceiling to {args.unposed_dir} and {args.layout_dir}")


if __name__ == "__main__":
    main()
