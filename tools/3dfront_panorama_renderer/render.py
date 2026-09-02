import blenderproc as bproc

import argparse
import json
import math
import re
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


UUID_SUFFIX = re.compile(
    r"_(?=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_[0-9]+$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one directory of per-object GLB files into panoramas."
    )
    parser.add_argument("room_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--camera-height", type=float, default=1.6)
    parser.add_argument("--min-clearance", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args()


def source_label(path: Path) -> str:
    match = UUID_SUFFIX.search(path.stem)
    return path.stem[: match.start()] if match else path.stem


def import_room(room_dir: Path) -> list[bproc.types.MeshObject]:
    if not room_dir.is_dir():
        raise NotADirectoryError(room_dir)
    meshes: list[bproc.types.MeshObject] = []
    for glb_path in sorted(room_dir.glob("*.glb")):
        previous = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
        label = source_label(glb_path)
        for obj in set(bpy.data.objects) - previous:
            if obj.type != "MESH":
                continue
            obj["source_label"] = label
            obj["source_file"] = glb_path.name
            meshes.append(bproc.types.MeshObject(obj))
    if not meshes:
        raise FileNotFoundError(f"No importable GLB meshes in {room_dir}")
    bpy.context.view_layer.update()
    return meshes


def room_bounds(meshes: list[bproc.types.MeshObject]) -> tuple[np.ndarray, np.ndarray]:
    corners = np.concatenate([mesh.get_bound_box() for mesh in meshes], axis=0)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def choose_cameras(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    meshes: list[bproc.types.MeshObject],
    camera_height: float,
    min_clearance: float,
    views: int,
) -> list[np.ndarray]:
    if views < 1:
        raise ValueError("views must be at least 1")
    bvh = bproc.object.create_bvh_tree_multi_objects(meshes)
    z = min(bounds_min[2] + camera_height, bounds_max[2] - 0.2)
    fractions = (0.5, 0.3, 0.7, 0.15, 0.85)
    fallback = np.array(
        [
            (bounds_min[0] + bounds_max[0]) / 2,
            (bounds_min[1] + bounds_max[1]) / 2,
            z,
        ]
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for x_fraction in fractions:
        for y_fraction in fractions:
            candidate = np.array(
                [
                    bounds_min[0] + x_fraction * (bounds_max[0] - bounds_min[0]),
                    bounds_min[1] + y_fraction * (bounds_max[1] - bounds_min[1]),
                    z,
                ]
            )
            nearest = bvh.find_nearest(Vector(candidate))
            clearance = float(nearest[3]) if nearest is not None else float("inf")
            if clearance > min_clearance:
                candidates.append((clearance, candidate))
    if not candidates:
        return [fallback] * views

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [candidates.pop(0)[1]]
    room_diagonal = max(float(np.linalg.norm(bounds_max[:2] - bounds_min[:2])), 1e-6)
    while candidates and len(selected) < views:
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                min(
                    np.linalg.norm(candidates[index][1][:2] - point[:2])
                    for point in selected
                )
                / room_diagonal,
                candidates[index][0],
            ),
        )
        selected.append(candidates.pop(best_index)[1])
    while len(selected) < views:
        selected.append(selected[0].copy())
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bproc.init()
    meshes = import_room(args.room_dir.resolve())
    bounds_min, bounds_max = room_bounds(meshes)
    camera_locations = choose_cameras(
        bounds_min,
        bounds_max,
        meshes,
        args.camera_height,
        args.min_clearance,
        args.views,
    )

    bproc.camera.set_resolution(args.width, args.height)
    camera = bpy.context.scene.camera.data
    camera.type = "PANO"
    if hasattr(camera, "panorama_type"):
        camera.panorama_type = "EQUIRECTANGULAR"
    else:
        camera.cycles.panorama_type = "EQUIRECTANGULAR"
    # Local camera +Y becomes world +Z, which keeps the panorama horizon level.
    rotation = np.array([math.pi / 2.0, 0.0, 0.0])
    for camera_location in camera_locations:
        bproc.camera.add_camera_pose(
            bproc.math.build_transformation_mat(camera_location, rotation)
        )

    center = (bounds_min + bounds_max) / 2
    for location, energy in (
        ([center[0], center[1], bounds_max[2] + 1.5], 1200),
        (
            [
                camera_locations[0][0],
                camera_locations[0][1],
                camera_locations[0][2] + 0.5,
            ],
            500,
        ),
    ):
        light = bproc.types.Light()
        light.set_location(location)
        light.set_energy(energy)
        light.set_radius(1.0)

    bproc.renderer.set_world_background([0.04, 0.04, 0.04], strength=0.8)
    bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX", "CUDA"])
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_noise_threshold(0.05)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "name", "source_label", "source_file"],
        default_values={"source_label": "unknown", "source_file": "unknown"},
    )
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(str(args.output_dir), data)
    (args.output_dir / "render.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": "MIDI-3D/3D-FRONT processed GLB room",
                "projection": "equirectangular",
                "room_dir": str(args.room_dir.resolve()),
                "camera_locations": [location.tolist() for location in camera_locations],
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "mesh_count": len(meshes),
                "resolution": [args.width, args.height],
                "views": args.views,
                "samples": args.samples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
