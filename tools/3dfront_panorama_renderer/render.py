import blenderproc as bproc

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_policy import (
    POLICY_VERSION,
    camera_grid,
    is_structural_file,
    select_cameras,
    validate_camera_locations,
)


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
    parser.add_argument(
        "--allow-relaxed-clearance", action="store_true",
        help="Explicitly allow smaller clearances; recorded in render metadata.",
    )
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
    structural = [
        mesh
        for mesh in meshes
        if is_structural_file(str(mesh.get_cp("source_file")))
    ]
    if not structural:
        raise ValueError("No structural floor/wall/ceil/others GLBs; cannot place cameras safely")
    corners = np.concatenate([mesh.get_bound_box() for mesh in structural], axis=0)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def choose_cameras(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    meshes: list[bproc.types.MeshObject],
    camera_height: float,
    min_clearance: float,
    views: int,
    allow_relaxed: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    bvh = bproc.object.create_bvh_tree_multi_objects(meshes)
    candidates = []
    for candidate in camera_grid(bounds_min, bounds_max, camera_height):
        nearest = bvh.find_nearest(Vector(candidate))
        if nearest is None or nearest[3] is None:
            continue
        candidates.append((float(nearest[3]), candidate))
    selected = select_cameras(candidates, views, min_clearance, allow_relaxed)
    locations = [np.asarray(point) for _, point in selected]
    validate_camera_locations(locations, bounds_min, bounds_max, views)
    return locations, [clearance for clearance, _ in selected]


def main(metadata_filename="render.json") -> None:
    args = parse_args()
    if args.width < 1 or args.height < 1 or args.width != 2 * args.height:
        raise ValueError("Full equirectangular panoramas require positive 2:1 resolution")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory; old renders are not overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bproc.init()
    meshes = import_room(args.room_dir.resolve())
    bounds_min, bounds_max = room_bounds(meshes)
    camera_locations, camera_clearances = choose_cameras(
        bounds_min,
        bounds_max,
        meshes,
        args.camera_height,
        args.min_clearance,
        args.views,
        args.allow_relaxed_clearance,
    )

    bproc.camera.set_resolution(args.width, args.height)
    camera = bpy.context.scene.camera.data
    camera.type = "PANO"
    if hasattr(camera, "panorama_type"):
        camera.panorama_type = "EQUIRECTANGULAR"
    else:
        camera.cycles.panorama_type = "EQUIRECTANGULAR"
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
    (args.output_dir / metadata_filename).write_text(
        json.dumps(
            {
                "format_version": 2,
                "camera_policy_version": POLICY_VERSION,
                "source": "MIDI-3D/3D-FRONT processed GLB room",
                "projection": "equirectangular",
                "room_dir": str(args.room_dir.resolve()),
                "camera_locations": [location.tolist() for location in camera_locations],
                "camera_clearances": camera_clearances,
                "requested_min_clearance": args.min_clearance,
                "allow_relaxed_clearance": args.allow_relaxed_clearance,
                "relaxed_camera_count": sum(c < args.min_clearance for c in camera_clearances),
                "camera_height_requested": args.camera_height,
                "camera_rotation_euler_xyz": rotation.tolist(),
                "coordinate_frame": "Blender world XYZ, Z-up",
                "gltf_to_blender_axes": ["x", "-z", "y"],
                "metric_scale_certified": False,
                "bounds_source": "structural GLBs (exact case-sensitive filenames)",
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
