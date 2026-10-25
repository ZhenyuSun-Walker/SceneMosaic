#!/usr/bin/env python3
"""Render one generated GLB into the TD_mansion RGB/depth/camera layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import Imath
import numpy as np
import OpenEXR
from PIL import Image
import pyrender
import trimesh


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.])
    if abs(np.dot(forward, up)) > .99:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(right, forward), -forward], axis=1)
    pose[:3, 3] = position
    return pose


def write_depth(path: Path, depth: np.ndarray) -> None:
    height, width = depth.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    image = OpenEXR.OutputFile(str(path), header)
    try:
        image.writePixels({"Y": np.asarray(depth, dtype=np.float32).tobytes()})
    finally:
        image.close()


def load_meshes(path: Path) -> tuple[list[tuple[str, trimesh.Trimesh]], np.ndarray]:
    source = trimesh.load(path, force="scene", process=False)
    meshes = []
    # SceneMosaic exports Y-up OpenGL coordinates; this dataset uses Blender Z-up.
    to_z_up = np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    for node in source.graph.nodes_geometry:
        transform, geometry = source.graph[node]
        mesh = source.geometry[geometry].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh.apply_transform(to_z_up @ transform)
        meshes.append((str(node), mesh))
    if not meshes:
        raise ValueError(f"GLB contains no triangle meshes: {path}")
    bounds = np.array([np.min([m.bounds[0] for _, m in meshes], axis=0),
                       np.max([m.bounds[1] for _, m in meshes], axis=0)])
    offset = np.array([*bounds.mean(axis=0)[:2], bounds[0, 2]])
    for _, mesh in meshes:
        mesh.apply_translation(-offset)
    transform = to_z_up.copy()
    transform[:3, 3] -= offset
    return meshes, transform


def camera_position(name: str, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place cameras in the new scene, using source view names as a recipe only."""
    low, high = bounds
    size = high - low
    center = bounds.mean(axis=0)
    pos = center.copy()
    target = center.copy()
    pos[2] = low[2] + .58 * size[2]
    target[2] = low[2] + .40 * size[2]
    if "high" in name:
        pos[2] = low[2] + .82 * size[2]
    if "center_pan" in name:
        index = int(name.rsplit("_", 1)[-1]) - 1
        angle = 2 * np.pi * index / 9
        target[:2] += .45 * size[:2] * [np.cos(angle), np.sin(angle)]
        target[2] = pos[2]
        return pos, target
    if "south" in name or "_sw_" in name or "_se_" in name:
        pos[1] = low[1] + .12 * size[1]
    if "north" in name or "_nw_" in name or "_ne_" in name:
        pos[1] = high[1] - .12 * size[1]
    if "west" in name or "_sw_" in name or "_nw_" in name:
        pos[0] = low[0] + .12 * size[0]
    if "east" in name or "_se_" in name or "_ne_" in name:
        pos[0] = high[0] - .12 * size[0]
    suffix = name.rsplit("_", 1)[-1]
    if suffix.isdigit() and "corner" not in name:
        fraction = {1: .25, 2: .5, 3: .75}.get(int(suffix), .5)
        if "south" in name or "north" in name:
            pos[0] = low[0] + fraction * size[0]
        elif "west" in name or "east" in name:
            pos[1] = low[1] + fraction * size[1]
    if "_cross" in name:
        target[:2] = low[:2] + high[:2] - pos[:2]
    return pos, target


def make_scene(meshes: list, *, overhead: bool = False, diagram: bool = False) -> tuple[pyrender.Scene, list[str]]:
    scene = pyrender.Scene(bg_color=[1., 1., 1., 1.], ambient_light=[.65, .65, .65])
    bounds = np.array([np.min([m.bounds[0] for _, m in meshes], axis=0),
                       np.max([m.bounds[1] for _, m in meshes], axis=0)])
    size = np.maximum(bounds[1] - bounds[0], 1e-6)
    hidden = []
    for name, mesh in meshes:
        b = mesh.bounds
        extent = b[1] - b[0]
        # Hide broad, thin surfaces at the ceiling only in overhead products.
        ceiling = (extent[2] < .04 * size[2] and
                   extent[0] * extent[1] > .35 * size[0] * size[1] and
                   b.mean(axis=0)[2] > bounds[0, 2] + .9 * size[2])
        if overhead and ceiling:
            hidden.append(name)
            continue
        material = None
        if diagram:
            digest = hashlib.sha256(name.encode()).digest()
            color = [.55 + value / 255 * .35 for value in digest[:3]]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*color, 1.], metallicFactor=0., roughnessFactor=1.)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material))
    return scene, hidden


def export_dataset(glb: Path, source_dir: Path, output_dir: Path, *, image_size: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ["rgb", "depth", "cameras", "rgb_td", "floorplan"]:
        (output_dir / folder).mkdir(exist_ok=True)
    templates = sorted((source_dir / "cameras").glob("*.json"), key=lambda p: int(p.stem))
    if not templates:
        raise ValueError("The source scene has no camera templates")
    meshes, source_to_output = load_meshes(glb)
    final_scene = trimesh.Scene()
    for name, mesh in meshes:
        final_scene.add_geometry(mesh, node_name=name)
    final_scene.export(output_dir / "scene.glb")
    bounds = np.array([np.min([m.bounds[0] for _, m in meshes], axis=0),
                       np.max([m.bounds[1] for _, m in meshes], axis=0)])
    size = np.maximum(bounds[1] - bounds[0], 1e-5)
    clip_near = max(float(size.max()) * .001, .0001)
    clip_far = float(size.max()) * 10
    scene, _ = make_scene(meshes)
    camera_records = []
    for template in templates:
        original = json.loads(template.read_text())
        width, height = original["resolution"]
        k = np.asarray(original["intrinsic_matrix"], dtype=float)
        if image_size is not None:
            k[0] *= image_size / width
            k[1] *= image_size / height
            width = height = image_size
        name = original.get("source_view_name", f"view_{template.stem}")
        pos, target = camera_position(name, bounds)
        pose = look_at(pos, target)
        camera = pyrender.IntrinsicsCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                                          znear=clip_near, zfar=clip_far)
        node = scene.add(camera, pose=pose)
        light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.), pose=pose)
        renderer = pyrender.OffscreenRenderer(width, height)
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
            scene.remove_node(node)
            scene.remove_node(light)
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError(f"Nonfinite or negative depth in view {template.stem}")
        Image.fromarray(color).save(output_dir / "rgb" / f"{template.stem}.png")
        write_depth(output_dir / "depth" / f"{template.stem}.exr", depth)
        w2c = np.linalg.inv(pose)
        cv_w2c = np.diag([1., -1., -1., 1.]) @ w2c
        record = {
            "name": template.stem,
            "image_path": f"{output_dir.name}/rgb/{template.stem}.png",
            "depth_path": f"{output_dir.name}/depth/{template.stem}.exr",
            "resolution": [width, height], "projection_type": "PERSP",
            "intrinsic_matrix": k.tolist(), "camera_to_world_blender": pose.tolist(),
            "world_to_camera_blender": w2c.tolist(), "world_to_camera_cv": cv_w2c.tolist(),
            "camera_location": pos.tolist(), "source_view_name": name,
            "clip": [clip_near, clip_far], "depth_type": "camera_z",
            "depth_background": 0., "depth_units": "generated_scene_units",
            "camera_policy": "new_positions_from_generated_scene_bounds",
            "source_camera_template": str(template),
            "valid_depth_fraction": float(np.mean(depth > 0)),
        }
        if record["valid_depth_fraction"] < .01:
            raise ValueError(f"View {template.stem} contains almost no geometry")
        write_json(output_dir / "cameras" / template.name, record)
        camera_records.append(record)
        print(f"Rendered {template.stem}: {name}, valid depth {record['valid_depth_fraction']:.1%}", flush=True)
    top_size = image_size or 1024
    half_width = float(max(size[:2])) * .55
    top_pos = bounds.mean(axis=0)
    top_pos[2] = bounds[1, 2] + float(size.max())
    top_pose = look_at(top_pos, bounds.mean(axis=0))
    hidden = []
    for diagram, folder, filename in [(False, "rgb_td", "rgb_td.png"),
                                       (True, "floorplan", "floorplan.png")]:
        top_scene, hidden = make_scene(meshes, overhead=True, diagram=diagram)
        top_scene.add(pyrender.OrthographicCamera(xmag=half_width, ymag=half_width,
                                                znear=clip_near, zfar=clip_far), pose=top_pose)
        top_scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.), pose=top_pose)
        renderer = pyrender.OffscreenRenderer(top_size, top_size)
        try:
            flags = pyrender.RenderFlags.RGBA
            if diagram:
                flags |= pyrender.RenderFlags.FLAT
            color, _ = renderer.render(top_scene, flags=flags)
        finally:
            renderer.delete()
        Image.fromarray(color).save(output_dir / folder / filename)
    report = {
        "status": "complete", "source_glb": str(glb), "source_scene": str(source_dir),
        "output_scene": str(output_dir), "view_count": len(camera_records),
        "coordinate_system": "right_handed_blender_z_up",
        "source_glb_to_output": source_to_output.tolist(), "bounds": bounds.tolist(),
        "scale_note": "Reconstruction units; not registered to original TD_mansion metric coordinates.",
        "camera_policy": "Source intrinsics and view labels; new poses in generated scene bounds.",
        "floorplan_type": "flat_color_topdown_mesh_render_not_semantic_annotation",
        "overhead_hidden_ceiling_nodes": hidden,
        "topdown_camera_to_world": top_pose.tolist(), "topdown_ortho_scale": 2 * half_width,
        "minimum_valid_depth_fraction": min(r["valid_depth_fraction"] for r in camera_records),
    }
    write_json(output_dir / "render_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args()
    if args.image_size is not None and args.image_size <= 0:
        parser.error("--image-size must be positive")
    print(json.dumps(export_dataset(args.glb.resolve(), args.source_scene.resolve(),
                                    args.output_dir.resolve(), image_size=args.image_size), indent=2))


if __name__ == "__main__":
    main()
