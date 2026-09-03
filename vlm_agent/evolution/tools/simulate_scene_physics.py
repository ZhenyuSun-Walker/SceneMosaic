import argparse
import json
from pathlib import Path

import numpy as np

try:
    import sapien
except ImportError as e:
    try:
        import sapien.core as sapien
    except ImportError as inner_e:
        raise ImportError(
            "sapien is required for simulation. Install it with: pip install sapien"
        ) from inner_e

try:
    from sapien.utils.viewer import Viewer as SapienViewer
except Exception:
    SapienViewer = None


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


def parse_gravity_vector(value):
    gravity = as_vector(value, {1, 3})
    if gravity.size == 1:
        gravity = np.array([0.0, -float(gravity[0]), 0.0], dtype=np.float32)
    norm = np.linalg.norm(gravity)
    if norm <= 1e-8:
        raise ValueError("Gravity vector norm is zero.")
    return gravity.astype(np.float32)


def parse_translation(value):
    return as_vector(value, {3})


def parse_rotation_wxyz(value):
    quat = as_vector(value, {4})
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def quat_wxyz_to_rotmat(quat_wxyz):
    q = parse_rotation_wxyz(quat_wxyz)
    w, x, y, z = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_conjugate(quat_wxyz):
    q = parse_rotation_wxyz(quat_wxyz)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_wxyz_multiply(lhs_wxyz, rhs_wxyz):
    lw, lx, ly, lz = parse_rotation_wxyz(lhs_wxyz).tolist()
    rw, rx, ry, rz = parse_rotation_wxyz(rhs_wxyz).tolist()
    out = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )
    return parse_rotation_wxyz(out)


def quat_delta_wxyz(prev_wxyz, curr_wxyz):
    return quat_wxyz_multiply(curr_wxyz, quat_wxyz_conjugate(prev_wxyz))


def quat_delta_angle_deg(prev_wxyz, curr_wxyz):
    delta = quat_delta_wxyz(prev_wxyz, curr_wxyz)
    angle_rad = 2.0 * np.arctan2(np.linalg.norm(delta[1:]), max(float(delta[0]), 1e-8))
    return float(np.degrees(abs(angle_rad)))


def rotmat_to_quat_wxyz(rotmat):
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(rotmat, dtype=np.float32).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return parse_rotation_wxyz([w, x, y, z])


def normalized_vector(value):
    vec = np.asarray(value, dtype=np.float32).reshape(3)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        raise ValueError("Vector norm is zero.")
    return vec / norm


def normalize_object_id(object_id):
    object_id = str(object_id)
    if object_id and object_id[0].isdigit():
        return f"object_{object_id}"
    return object_id


def resolve_requested_object_ids(requested_object_ids):
    if requested_object_ids is None:
        return None
    if isinstance(requested_object_ids, (str, int)):
        requested_object_ids = [requested_object_ids]
    return [normalize_object_id(object_id) for object_id in requested_object_ids]


def layout_pose_to_sapien(translation, rotation_wxyz):
    """Return a SAPIEN pose in the same P3D frame used by layout JSON files."""
    return translation.tolist(), rotation_wxyz.tolist()


def sapien_pose_to_layout(translation_xyz, rotation_wxyz):
    """Convert a SAPIEN pose back to P3D layout JSON fields.

    SAPIEN runs in the same world frame as the layout data here, so fields can
    be written back directly in translation + wxyz format.
    """
    quat = parse_rotation_wxyz(rotation_wxyz)
    return (
        [list(map(float, translation_xyz))],
        [list(map(float, quat))],
    )


def load_object_ids(layout_dir: Path, hull_dir: Path, requested_object_ids=None):
    layout_ids = {p.stem for p in layout_dir.glob("*.json")}
    hull_ids = {p.stem for p in hull_dir.glob("*.obj")}
    matched_ids = sorted(layout_ids & hull_ids)

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


def load_layout(layout_path: Path):
    with layout_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    translation = parse_translation(raw["translation"])
    rotation_wxyz = parse_rotation_wxyz(raw["rotation"])
    scale = parse_scale(raw["scale"])

    return raw, translation, rotation_wxyz, scale


def load_obj_vertices(obj_path: Path):
    vertices = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"No vertex lines found in OBJ: {obj_path}")

    return np.asarray(vertices, dtype=np.float32)


def compute_scene_ground_plane_from_hulls(layout_dir: Path, hull_dir: Path, object_ids, gravity_xyz):
    up_direction = normalized_vector(-np.asarray(gravity_xyz, dtype=np.float32).reshape(3))
    surface_level = None

    for object_id in object_ids:
        layout_path = layout_dir / f"{normalize_object_id(object_id)}.json"
        hull_path = hull_dir / f"{normalize_object_id(object_id)}.obj"

        _, translation, rotation_wxyz, scale = load_layout(layout_path)
        local_vertices = load_obj_vertices(hull_path)

        rotation = quat_wxyz_to_rotmat(rotation_wxyz)
        scaled_vertices = local_vertices * scale.reshape(1, 3)
        world_vertices = scaled_vertices @ rotation.T + translation.reshape(1, 3)

        obj_min_level = float(np.min(world_vertices @ up_direction))
        if surface_level is None or obj_min_level < surface_level:
            surface_level = obj_min_level

    if surface_level is None:
        raise RuntimeError("Failed to compute scene ground plane from hulls.")

    return up_direction, float(surface_level)


def compute_initial_y_offset(layout_dir: Path, hull_dir: Path, object_ids, selected_ground_object_ids):
    max_extent = None
    selected_ground_object_ids = set(selected_ground_object_ids or [])

    for object_id in object_ids:
        if object_id in selected_ground_object_ids:
            continue

        layout_path = layout_dir / f"{normalize_object_id(object_id)}.json"
        hull_path = hull_dir / f"{normalize_object_id(object_id)}.obj"

        _, _, _, scale = load_layout(layout_path)
        local_vertices = load_obj_vertices(hull_path)
        bbox_extents = np.ptp(local_vertices, axis=0)
        scaled_extents = bbox_extents * scale.reshape(3)
        object_max_extent = float(np.max(scaled_extents))
        if max_extent is None or object_max_extent > max_extent:
            max_extent = object_max_extent

    if max_extent is None:
        return 0.0

    return float(min(max_extent / 2.0, 0.4))


def choose_ground_object_id(
    layout_dir: Path,
    object_ids,
    use_ground_object: bool,
    ground_object_ids=None,
):
    if not use_ground_object:
        return set()

    if ground_object_ids:
        normalized_ids = []
        for object_id in ground_object_ids:
            normalized_id = normalize_object_id(object_id)
            if normalized_id not in object_ids:
                raise ValueError(
                    f"Requested --ground_object_id={normalized_id}, "
                    f"but it is not in matched objects: {object_ids}"
                )
            normalized_ids.append(normalized_id)
        return set(normalized_ids)

    min_y = None
    selected_id = None
    for object_id in object_ids:
        layout_path = layout_dir / f"{normalize_object_id(object_id)}.json"
        _, translation, _, _ = load_layout(layout_path)
        y = float(translation[1])
        if min_y is None or y < min_y:
            min_y = y
            selected_id = object_id
    return {selected_id} if selected_id is not None else set()


def create_scene_compatible(engine, scene_config, cpu_only: bool):
    """Create a SAPIEN scene with optional CPU-only mode.

    Newer SAPIEN wrappers build a render system by default when using
    Engine.create_scene(), which fails on machines without a rendering device.
    This helper provides a pure-physics fallback scene.
    """
    if cpu_only:
        if hasattr(sapien, "physx") and hasattr(sapien.physx, "set_scene_config"):
            sapien.physx.set_scene_config(scene_config)
        if hasattr(sapien, "Scene") and hasattr(sapien.physx, "PhysxCpuSystem"):
            return sapien.Scene([sapien.physx.PhysxCpuSystem()])
        raise RuntimeError(
            "--cpu_only was set, but this SAPIEN build does not expose "
            "sapien.Scene([sapien.physx.PhysxCpuSystem()])."
        )

    try:
        return engine.create_scene(scene_config)
    except RuntimeError as e:
        msg = str(e).lower()
        if "render" in msg and "device" in msg:
            print(
                "[WARN] Failed to create scene with renderer (no rendering device). "
                "Falling back to CPU-only PhysX scene."
            )
            if hasattr(sapien, "physx") and hasattr(sapien.physx, "set_scene_config"):
                sapien.physx.set_scene_config(scene_config)
            if hasattr(sapien, "Scene") and hasattr(sapien.physx, "PhysxCpuSystem"):
                return sapien.Scene([sapien.physx.PhysxCpuSystem()])
            raise RuntimeError(
                "SAPIEN cannot create a render scene, and CPU-only scene API is unavailable."
            ) from e
        raise


def create_renderer_compatible():
    """Create a renderer across SAPIEN API variants."""
    if hasattr(sapien, "SapienRenderer"):
        return sapien.SapienRenderer()
    if hasattr(sapien, "VulkanRenderer"):
        return sapien.VulkanRenderer()
    raise RuntimeError(
        "No supported SAPIEN renderer found (SapienRenderer/VulkanRenderer)."
    )


def add_visual_mesh_compatible(builder, mesh_path: Path, scale):
    """Attach visual mesh if the SAPIEN API supports it."""
    if not hasattr(builder, "add_visual_from_file"):
        return False

    try:
        builder.add_visual_from_file(
            filename=str(mesh_path),
            pose=sapien.Pose(),
            scale=scale.tolist(),
        )
        return True
    except TypeError:
        try:
            builder.add_visual_from_file(str(mesh_path))
            return True
        except Exception:
            return False
    except Exception:
        return False


def setup_scene_lighting(scene):
    """Add robust default lights for GUI rendering."""
    if hasattr(scene, "set_ambient_light"):
        scene.set_ambient_light([0.35, 0.35, 0.35])
    if hasattr(scene, "add_directional_light"):
        scene.add_directional_light(
            direction=[0.5, -1.0, -0.3],
            color=[1.0, 1.0, 1.0],
            shadow=True,
        )
    if hasattr(scene, "add_point_light"):
        scene.add_point_light([3.0, 3.0, 3.0], [1.0, 1.0, 1.0], shadow=False)
        scene.add_point_light([-3.0, 3.0, -3.0], [0.7, 0.7, 0.7], shadow=False)


def _poll_viewer_key(window, key_name: str):
    """Best-effort key polling across viewer API variants."""
    candidates = ["key_down", "is_key_down", "key_press", "is_key_pressed"]
    key_variants = [key_name, key_name.lower(), key_name.upper()]
    for attr in candidates:
        if not hasattr(window, attr):
            continue
        fn = getattr(window, attr)
        for key in key_variants:
            try:
                if bool(fn(key)):
                    return True
            except Exception:
                continue
    return False


def quaternion_wxyz_to_rpy(quat_wxyz):
    """Convert a wxyz quaternion to roll/pitch/yaw in radians."""
    w, x, y, z = parse_rotation_wxyz(quat_wxyz).tolist()

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = float(np.sign(sinp) * (np.pi / 2.0))
    else:
        pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def limit_dynamic_actor_velocities(
    actor_to_data,
    selected_ground_object_ids,
    max_abs_velocity_xyz,
    max_abs_angular_velocity_xyz,
):
    """Clamp dynamic object linear/angular velocities component-wise in world XYZ."""
    max_abs_velocity_xyz = np.asarray(max_abs_velocity_xyz, dtype=np.float32).reshape(3)
    max_abs_angular_velocity_xyz = np.asarray(
        max_abs_angular_velocity_xyz, dtype=np.float32
    ).reshape(3)

    def _get_set_velocity_accessors(entity):
        # Old-style Actor API
        linear_getter = None
        linear_setter = None
        angular_getter = None
        angular_setter = None

        if hasattr(entity, "get_velocity") and hasattr(entity, "set_velocity"):
            linear_getter = entity.get_velocity
            linear_setter = entity.set_velocity
        if hasattr(entity, "get_angular_velocity") and hasattr(entity, "set_angular_velocity"):
            angular_getter = entity.get_angular_velocity
            angular_setter = entity.set_angular_velocity

        if (
            linear_getter is not None
            and linear_setter is not None
            and angular_getter is not None
            and angular_setter is not None
        ):
            return linear_getter, linear_setter, angular_getter, angular_setter

        # Newer Entity + PhysX component API
        physx = getattr(sapien, "physx", None)
        dyn_type = getattr(physx, "PhysxRigidDynamicComponent", None)
        if dyn_type is not None and hasattr(entity, "find_component_by_type"):
            try:
                comp = entity.find_component_by_type(dyn_type)
            except Exception:
                comp = None
            if comp is not None:
                if hasattr(comp, "get_linear_velocity") and hasattr(comp, "set_linear_velocity"):
                    linear_getter = comp.get_linear_velocity
                    linear_setter = comp.set_linear_velocity
                elif hasattr(comp, "linear_velocity"):

                    def _linear_getter():
                        return comp.linear_velocity

                    def _linear_setter(v):
                        comp.linear_velocity = v

                    linear_getter = _linear_getter
                    linear_setter = _linear_setter

                if hasattr(comp, "get_angular_velocity") and hasattr(comp, "set_angular_velocity"):
                    angular_getter = comp.get_angular_velocity
                    angular_setter = comp.set_angular_velocity
                elif hasattr(comp, "angular_velocity"):

                    def _angular_getter():
                        return comp.angular_velocity

                    def _angular_setter(v):
                        comp.angular_velocity = v

                    angular_getter = _angular_getter
                    angular_setter = _angular_setter

        return linear_getter, linear_setter, angular_getter, angular_setter

    for actor, data in actor_to_data.items():
        if data["object_id"] in selected_ground_object_ids:
            continue

        (
            velocity_getter,
            velocity_setter,
            angular_velocity_getter,
            angular_velocity_setter,
        ) = _get_set_velocity_accessors(actor)
        if (
            velocity_getter is None
            or velocity_setter is None
            or angular_velocity_getter is None
            or angular_velocity_setter is None
        ):
            continue

        velocity = np.asarray(velocity_getter(), dtype=np.float32)
        clipped_velocity = np.clip(velocity, -max_abs_velocity_xyz, max_abs_velocity_xyz)
        if not np.allclose(velocity, clipped_velocity):
            velocity_setter(clipped_velocity)

        angular_velocity = np.asarray(angular_velocity_getter(), dtype=np.float32)
        clipped_angular_velocity = np.clip(
            angular_velocity,
            -max_abs_angular_velocity_xyz,
            max_abs_angular_velocity_xyz,
        )
        if not np.allclose(angular_velocity, clipped_angular_velocity):
            angular_velocity_setter(clipped_angular_velocity)


def run_simulation(
    layout_dir: Path,
    hull_dir: Path,
    output_layout_dir: Path,
    gravity,
    timestep: float,
    depenetration_steps: int,
    simulation_steps: int,
    mass: float,
    linear_damping: float,
    angular_damping: float,
    solver_iterations: int,
    use_ground_object: bool,
    ground_object_ids,
    ground_plane_mode: str,
    gui: bool,
    gui_manual_control: bool,
    pose_delta_output: Path | None,
    object_ids=None,
    verify_only: bool = False,
    lock_rotation_all: bool = False,
    cpu_only: bool = False,
    init_non_ground_y_offset: bool = False,
):
    gravity_xyz = parse_gravity_vector(gravity)
    requested_object_ids = resolve_requested_object_ids(object_ids)
    object_ids = load_object_ids(layout_dir, hull_dir, requested_object_ids=requested_object_ids)
    if not object_ids:
        raise FileNotFoundError(
            f"No matching object files in {layout_dir} and {hull_dir}."
        )

    if verify_only:
        print("[INFO] Running in verify-only mode (no simulation)")
        print(f"[INFO] Found {len(object_ids)} matched objects: {object_ids}")
        for object_id in object_ids:
            layout_path = layout_dir / f"{normalize_object_id(object_id)}.json"
            hull_path = hull_dir / f"{normalize_object_id(object_id)}.obj"
            raw_layout, translation, rotation_wxyz, scale = load_layout(layout_path)
            print(f"[OK] object_{object_id}:")
            print(f"  Layout file: {layout_path}")
            print(f"  Hull file: {hull_path} (exists: {hull_path.exists()})")
            print(f"  Translation: {translation}")
            print(f"  Rotation (wxyz): {rotation_wxyz}")
            print(f"  Scale: {scale}")
            print("[INFO] All objects loaded successfully!")
        return

    selected_ground_object_ids = choose_ground_object_id(
        layout_dir=layout_dir,
        object_ids=object_ids,
        use_ground_object=use_ground_object,
        ground_object_ids=ground_object_ids,
    )
    if selected_ground_object_ids:
        selected_ground_object_ids = sorted(selected_ground_object_ids)
        print(
            "[INFO] Using static ground objects: "
            + ", ".join(normalize_object_id(object_id) for object_id in selected_ground_object_ids)
        )

    initial_y_offset = 0.0
    if init_non_ground_y_offset:
        initial_y_offset = compute_initial_y_offset(
            layout_dir=layout_dir,
            hull_dir=hull_dir,
            object_ids=object_ids,
            selected_ground_object_ids=selected_ground_object_ids,
        )
        print(f"[INFO] Initial non-ground Y offset enabled: {initial_y_offset:.6f}")

    if cpu_only:
        print("[INFO] CPU-only mode enabled: scene will be created without RenderSystem.")
        if gui:
            print("[WARN] --gui is incompatible with --cpu_only. Falling back to headless mode.")

    engine = sapien.Engine()
    renderer = None
    gui_enabled = bool(gui and (not cpu_only))
    gui_manual_enabled = False

    if not cpu_only:
        try:
            renderer = create_renderer_compatible()
            if hasattr(engine, "set_renderer"):
                engine.set_renderer(renderer)
        except Exception as e:
            print(f"[WARN] Failed to initialize SAPIEN renderer: {e}")
            if gui_enabled:
                print("[WARN] GUI requested, but renderer init failed. Falling back to headless mode.")
            gui_enabled = False

    viewer = None
    if gui_enabled:
        if SapienViewer is None:
            print(
                "[WARN] sapien.utils.viewer.Viewer is unavailable in this environment. "
                "Falling back to headless mode."
            )
            gui_enabled = False
        else:
            try:
                if renderer is not None:
                    viewer = SapienViewer(renderer)
                else:
                    viewer = SapienViewer()
            except TypeError:
                try:
                    viewer = SapienViewer()
                except Exception as e:
                    print(f"[WARN] Failed to initialize Viewer: {e}")
                    gui_enabled = False
            except Exception as e:
                print(f"[WARN] Failed to initialize Viewer: {e}")
                gui_enabled = False

            if viewer is not None:
                print("[INFO] GUI mode enabled.")
                gui_manual_enabled = bool(gui_manual_control)
                if gui_manual_enabled:
                    print(
                        "[INFO] GUI manual mode enabled: starts paused at first frame. "
                        "Controls: [N] single-step, [Space] run/pause, [R] reset pause, [Q/Esc] quit."
                    )

    def bind_viewer_scene(scene):
        if viewer is None:
            return
        if hasattr(viewer, "set_scene"):
            viewer.set_scene(scene)
        camera_translation = [-0.02, 0.05, -0.68]

        if hasattr(viewer, "set_camera_xyz"):
            viewer.set_camera_xyz(*camera_translation)

    if ground_plane_mode == "bbox":
        ground_up_direction, ground_surface_level = compute_scene_ground_plane_from_hulls(
            layout_dir,
            hull_dir,
            object_ids,
            gravity_xyz,
        )
        print(
            f"[INFO] Ground plane mode=bbox, up_direction={ground_up_direction.tolist()}, "
            f"surface_level={ground_surface_level:.6f}"
        )
    else:
        ground_up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ground_surface_level = 0.0
        print("[INFO] Ground plane mode=y0, top surface at y=0.0")

    def build_ground_plane_pose(up_direction, surface_level, half_thickness=0.01):
        up_direction = normalized_vector(up_direction)
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(up_direction, reference))) > 0.9:
            reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = normalized_vector(np.cross(reference, up_direction))
        z_axis = normalized_vector(np.cross(x_axis, up_direction))
        rotation = np.column_stack([x_axis, up_direction, z_axis])
        return up_direction * (float(surface_level) - float(half_thickness)), rotmat_to_quat_wxyz(rotation)

    # Freeze ground plane pose at initialization so it stays fixed for all steps/phases.
    fixed_ground_pose_p, fixed_ground_pose_q = build_ground_plane_pose(
        ground_up_direction,
        ground_surface_level,
    )

    def build_scene_with_actors(gravity_xyz, pose_overrides=None, lock_rotation=False):
        scene_config = sapien.SceneConfig()
        scene_config.gravity = list(map(float, np.asarray(gravity_xyz, dtype=np.float32).tolist()))
        scene = create_scene_compatible(engine=engine, scene_config=scene_config, cpu_only=cpu_only)
        scene.set_timestep(float(timestep))
        if viewer is not None:
            setup_scene_lighting(scene)
        bind_viewer_scene(scene)

        # Keep contacts stable and make the behavior closer to the old backend.
        if hasattr(scene, "set_solver_iterations"):
            scene.set_solver_iterations(int(solver_iterations))

        physical_material = scene.create_physical_material(
            static_friction=0.3,
            dynamic_friction=0.3,
            restitution=0.0,
        )

        # Ground plane aligned with the chosen up direction using a thin kinematic box.
        ground_builder = scene.create_actor_builder()
        ground_builder.add_box_collision(
            half_size=[1000.0, 0.01, 1000.0],
            pose=sapien.Pose(p=fixed_ground_pose_p.tolist(), q=fixed_ground_pose_q.tolist()),
            material=physical_material,
        )
        if viewer is not None and hasattr(ground_builder, "add_box_visual"):
            try:
                ground_builder.add_box_visual(
                    half_size=[1000.0, 0.01, 1000.0],
                    pose=sapien.Pose(p=fixed_ground_pose_p.tolist(), q=fixed_ground_pose_q.tolist()),
                    color=[0.2, 0.24, 0.28],
                )
            except TypeError:
                try:
                    ground_builder.add_box_visual(
                        half_size=[1000.0, 0.01, 1000.0],
                        pose=sapien.Pose(p=fixed_ground_pose_p.tolist(), q=fixed_ground_pose_q.tolist()),
                    )
                except TypeError:
                    ground_builder.add_box_visual([1000.0, 0.01, 1000.0])
        ground_builder.build_kinematic(name="ground_plane")

        actor_to_data = {}
        prev_pose_by_actor = {}
        for object_id in object_ids:
            layout_path = layout_dir / f"{normalize_object_id(object_id)}.json"
            hull_path = hull_dir / f"{normalize_object_id(object_id)}.obj"

            raw_layout, translation, rotation_wxyz, scale = load_layout(layout_path)
            if pose_overrides is not None and object_id in pose_overrides:
                translation = np.asarray(pose_overrides[object_id]["p"], dtype=np.float32)
                rotation_wxyz = parse_rotation_wxyz(pose_overrides[object_id]["q"])

            position_xyz, orientation_wxyz = layout_pose_to_sapien(
                translation=translation,
                rotation_wxyz=rotation_wxyz,
            )
            if init_non_ground_y_offset and object_id not in selected_ground_object_ids:
                position_xyz = np.asarray(position_xyz, dtype=np.float32)
                position_xyz[1] += initial_y_offset

            builder = scene.create_actor_builder()
            if viewer is not None:
                add_visual_mesh_compatible(builder, hull_path, scale)
            if hasattr(builder, "add_multiple_convex_collisions_from_file"):
                builder.add_multiple_convex_collisions_from_file(
                    filename=str(hull_path),
                    pose=sapien.Pose(),
                    scale=scale.tolist(),
                    material=physical_material,
                    density=max(float(mass), 1e-6),
                )
            else:
                raise RuntimeError(
                    "No supported mesh collision API found on ActorBuilder."
                )

            if object_id in selected_ground_object_ids:
                actor = builder.build_kinematic(name=normalize_object_id(object_id))
            else:
                actor = builder.build(name=normalize_object_id(object_id))
                if hasattr(actor, "set_damping"):
                    actor.set_damping(float(linear_damping), float(angular_damping))

            actor.set_pose(sapien.Pose(p=position_xyz, q=orientation_wxyz))
            if lock_rotation:
                physx_component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                if physx_component:
                    physx_component.set_locked_motion_axes([False, False, False, True, True, True])

            actor_to_data[actor] = {
                "object_id": object_id,
                "layout": raw_layout,
                "scale": scale,
            }
            prev_pose_by_actor[actor] = {
                "p": np.asarray(position_xyz, dtype=np.float32),
                "q": parse_rotation_wxyz(orientation_wxyz),
            }

        return scene, actor_to_data, prev_pose_by_actor

    pose_delta_records = []

    def record_pose_deltas(phase: str, step_index: int, actor_to_data, prev_pose_by_actor):
        if pose_delta_output is None:
            return

        frame_records = []
        for actor, data in actor_to_data.items():
            pose = actor.get_pose()
            curr_p = np.asarray(pose.p, dtype=np.float32)
            curr_q = parse_rotation_wxyz(pose.q)
            prev_state = prev_pose_by_actor[actor]

            delta_p = curr_p - prev_state["p"]
            delta_q = quat_delta_wxyz(prev_state["q"], curr_q)
            delta_angle_deg = quat_delta_angle_deg(prev_state["q"], curr_q)

            frame_records.append(
                {
                    "object_id": data["object_id"],
                    "translation_xyz": list(map(float, curr_p.tolist())),
                    "rotation_wxyz": list(map(float, curr_q.tolist())),
                    "delta_translation_xyz": list(map(float, delta_p.tolist())),
                    "delta_rotation_wxyz": list(map(float, delta_q.tolist())),
                    "delta_rotation_angle_deg": delta_angle_deg,
                }
            )

            prev_state["p"] = curr_p
            prev_state["q"] = curr_q

        pose_delta_records.append(
            {
                "phase": phase,
                "step": int(step_index),
                "objects": frame_records,
            }
        )

    zero_gravity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    simulation_gravity = np.asarray(gravity_xyz, dtype=np.float32).reshape(3)
    # Clamp velocity across the whole pipeline. Y is allowed to be faster than X/Z in the legacy default,
    # but the clip is component-wise so custom gravity directions still work.
    max_abs_velocity_xyz = np.array([5e-5, 2e-2, 5e-5], dtype=np.float32)
    # Clamp angular velocity (rad/s) across the whole pipeline.
    max_abs_angular_velocity_xyz = np.array([5e-5, 5e-5, 5e-5], dtype=np.float32)

    actor_to_data = {}
    prev_pose_by_actor = {}
    stop_requested = False
    run_continuous = not gui_manual_enabled
    manual_key_supported = False

    if viewer is not None and gui_manual_enabled and hasattr(viewer, "window"):
        window = viewer.window
        manual_key_supported = any(
            hasattr(window, name)
            for name in ["key_down", "is_key_down", "key_press", "is_key_pressed"]
        )
        if not manual_key_supported:
            print(
                "[WARN] Viewer key polling API is unavailable in this SAPIEN build; "
                "falling back to continuous stepping."
            )
            run_continuous = True

    def after_step_render(scene):
        nonlocal stop_requested
        if viewer is None:
            return
        if hasattr(scene, "update_render"):
            scene.update_render()
        if hasattr(viewer, "render"):
            viewer.render()
        if hasattr(viewer, "closed") and viewer.closed:
            stop_requested = True

    def process_manual_controls():
        nonlocal run_continuous, stop_requested
        if (viewer is None) or (not gui_manual_enabled) or (not manual_key_supported):
            return False

        window = viewer.window
        if _poll_viewer_key(window, "q") or _poll_viewer_key(window, "esc"):
            stop_requested = True
            return False

        if _poll_viewer_key(window, "space"):
            run_continuous = not run_continuous

        if _poll_viewer_key(window, "r"):
            run_continuous = False

        if run_continuous:
            return True

        return _poll_viewer_key(window, "n")

    def step_scene_with_gui(scene, phase: str, steps: int, actor_to_data, prev_pose_by_actor):
        nonlocal stop_requested
        if steps / 50 == 0:
            print(f"[INFO] {phase} step 0/{max(steps, 0)}")
        record_pose_deltas(phase, 0, actor_to_data, prev_pose_by_actor)

        applied_steps = 0
        while applied_steps < max(steps, 0):
            do_step = True
            if gui_manual_enabled:
                do_step = process_manual_controls()
            if stop_requested:
                break

            if do_step:
                scene.step()
                limit_dynamic_actor_velocities(
                    actor_to_data=actor_to_data,
                    selected_ground_object_ids=selected_ground_object_ids,
                    max_abs_velocity_xyz=max_abs_velocity_xyz,
                    max_abs_angular_velocity_xyz=max_abs_angular_velocity_xyz,
                )
                applied_steps += 1
                print(f"[INFO] {phase} step {applied_steps}/{max(steps, 0)}")
                record_pose_deltas(
                    phase,
                    applied_steps,
                    actor_to_data,
                    prev_pose_by_actor,
                )

            after_step_render(scene)

            if stop_requested:
                break

        return applied_steps

    if depenetration_steps > 0:
        print(f"[INFO] Depenetration gravity = {zero_gravity.tolist()}")
        scene, actor_to_data, prev_pose_by_actor = build_scene_with_actors(
            zero_gravity,
            lock_rotation=True,
        )
        step_scene_with_gui(
            scene,
            phase="depenetration",
            steps=depenetration_steps,
            actor_to_data=actor_to_data,
            prev_pose_by_actor=prev_pose_by_actor,
        )
        if stop_requested:
            print("[INFO] Viewer closed by user during depenetration.")

        if (simulation_steps > 0) and (not stop_requested):
            start_poses = {}
            for actor, data in actor_to_data.items():
                pose = actor.get_pose()
                start_poses[data["object_id"]] = {
                    "p": np.asarray(pose.p, dtype=np.float32),
                    "q": parse_rotation_wxyz(pose.q),
                }

            print(f"[INFO] Simulation gravity = {simulation_gravity.tolist()}")
            scene, actor_to_data, prev_pose_by_actor = build_scene_with_actors(
                simulation_gravity,
                pose_overrides=start_poses,
                lock_rotation=lock_rotation_all,
            )
            step_scene_with_gui(
                scene,
                phase="simulation",
                steps=simulation_steps,
                actor_to_data=actor_to_data,
                prev_pose_by_actor=prev_pose_by_actor,
            )
            if stop_requested:
                print("[INFO] Viewer closed by user during simulation.")
    else:
        print(f"[INFO] Simulation gravity = {simulation_gravity.tolist()}")
        scene, actor_to_data, prev_pose_by_actor = build_scene_with_actors(
            simulation_gravity,
            lock_rotation=lock_rotation_all,
        )
        step_scene_with_gui(
            scene,
            phase="simulation",
            steps=simulation_steps,
            actor_to_data=actor_to_data,
            prev_pose_by_actor=prev_pose_by_actor,
        )
        if stop_requested:
            print("[INFO] Viewer closed by user during simulation.")

    output_layout_dir.mkdir(parents=True, exist_ok=True)
    for actor, data in actor_to_data.items():
        pose = actor.get_pose()

        updated_layout = dict(data["layout"])
        translation_p3d, rotation_p3d = sapien_pose_to_layout(
            translation_xyz=pose.p,
            rotation_wxyz=pose.q,
        )
        updated_layout["translation"] = translation_p3d
        if lock_rotation_all:
            updated_layout["rotation"] = data["layout"]["rotation"]
        else:
            updated_layout["rotation"] = rotation_p3d
        updated_layout["scale"] = [list(map(float, data["scale"]))]

        out_path = output_layout_dir / f"{normalize_object_id(data['object_id'])}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(updated_layout, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved simulated pose: {out_path}")

    if pose_delta_output is not None:
        pose_delta_output.parent.mkdir(parents=True, exist_ok=True)
        with pose_delta_output.open("w", encoding="utf-8") as f:
            json.dump(pose_delta_records, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved per-step pose deltas: {pose_delta_output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run simple rigid-body simulation with collision hulls and save updated "
            "object_XXX pose JSON files."
        )
    )
    parser.add_argument(
        "--layout_dir",
        type=Path,
        default=Path("outputs/layout_params"),
        help="Directory containing object_XXX.json layout files.",
    )
    parser.add_argument(
        "--hull_dir",
        type=Path,
        default=Path("outputs/collision_hulls"),
        help="Directory containing object_XXX.obj collision hull files.",
    )
    parser.add_argument(
        "--output_layout_dir",
        type=Path,
        default=Path("outputs/layout_params_simulated"),
        help="Directory to save updated object_XXX.json files.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        nargs="+",
        default=[2.0],
        help=(
            "Gravity magnitude or vector. Pass one value for legacy -Y gravity, "
            "or three values for an arbitrary gravity vector in world XYZ."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0 / 100.0,
        help="Simulation timestep in seconds.",
    )
    parser.add_argument(
        "--depenetration_steps",
        type=int,
        default=200,
        help="Steps with zero gravity to resolve initial penetrations.",
    )
    parser.add_argument(
        "--simulation_steps",
        type=int,
        default=2000,
        help="Steps with gravity enabled.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=10.0,
        help="Density proxy for each object body in SAPIEN collision loading.",
    )
    parser.add_argument(
        "--linear_damping",
        type=float,
        default=5.0,
        help="Linear damping for each object.",
    )
    parser.add_argument(
        "--angular_damping",
        type=float,
        default=5.0,
        help="Angular damping for each object.",
    )
    parser.add_argument(
        "--solver_iterations",
        type=int,
        default=100,
        help="Constraint solver iterations.",
    )
    parser.add_argument(
        "--use_ground_object",
        action="store_true",
        help=(
            "If set, use a static object as ground anchor to reduce global drift. "
            "By default, the lowest-Y object is chosen (PyTorch3D coordinate system)."
        ),
    )
    parser.add_argument(
        "--ground_object_id",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more object ids (XXX in object_XXX) to use as static ground objects. "
            "Only valid when --use_ground_object is enabled."
        ),
    )
    parser.add_argument(
        "--ground_plane_mode",
        type=str,
        choices=["bbox", "y0"],
        default="bbox",
        help=(
            "Ground plane placement mode: bbox aligns the plane to the gravity direction "
            "and uses the transformed collision-hull bbox; y0 keeps the legacy fixed y=0 plane."
        ),
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help=(
            "Create a pure-physics SAPIEN scene without RenderSystem. "
            "Useful on machines without graphics/rendering devices."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Enable SAPIEN GUI viewer when renderer/device is available; "
            "otherwise automatically fall back to headless mode."
        ),
    )
    parser.add_argument(
        "--gui_manual_control",
        action="store_true",
        help=(
            "GUI only: start paused on first frame and manually control stepping "
            "with keyboard (N single-step, Space run/pause, R pause, Q/Esc quit)."
        ),
    )
    parser.add_argument(
        "--pose_delta_output",
        type=Path,
        default=None,
        help=(
            "If set, record per-step translation/rotation changes for each object "
            "and save to this JSON file."
        ),
    )
    parser.add_argument(
        "--lock_rotation_all",
        action="store_true",
        help="Lock rotation for all objects and only update translations in output.",
    )
    parser.add_argument(
        "--object_ids",
        nargs="+",
        default=None,
        help=(
            "Optional subset of object ids to simulate (XXX in object_XXX). "
            "Defaults to all matched objects."
        ),
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify loading of layout and hull files, without running simulation.",
    )
    parser.add_argument(
        "--init_non_ground_y_offset",
        action="store_true",
        help="Shift non-ground objects upward before simulation by a hull-size-based offset.",
    )
    args = parser.parse_args()

    if not args.layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {args.layout_dir}")
    if not args.hull_dir.exists():
        raise FileNotFoundError(f"Hull directory not found: {args.hull_dir}")
    if args.mass <= 0:
        raise ValueError("--mass must be positive.")
    if args.timestep <= 0:
        raise ValueError("--timestep must be positive.")
    if args.ground_object_id is not None and not args.use_ground_object:
        raise ValueError("--ground_object_id requires --use_ground_object.")

    run_simulation(
        layout_dir=args.layout_dir,
        hull_dir=args.hull_dir,
        output_layout_dir=args.output_layout_dir,
        gravity=args.gravity,
        timestep=args.timestep,
        depenetration_steps=args.depenetration_steps,
        simulation_steps=args.simulation_steps,
        mass=args.mass,
        object_ids=args.object_ids,
        linear_damping=args.linear_damping,
        angular_damping=args.angular_damping,
        solver_iterations=args.solver_iterations,
        use_ground_object=args.use_ground_object,
        ground_object_ids=args.ground_object_id,
        ground_plane_mode=args.ground_plane_mode,
        gui=args.gui,
        gui_manual_control=args.gui_manual_control,
        pose_delta_output=args.pose_delta_output,
        verify_only=args.verify_only,
        lock_rotation_all=args.lock_rotation_all,
        cpu_only=args.cpu_only,
        init_non_ground_y_offset=args.init_non_ground_y_offset,
    )


if __name__ == "__main__":
    main()
