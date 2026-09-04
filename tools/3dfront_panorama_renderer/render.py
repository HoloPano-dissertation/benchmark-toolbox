import blenderproc as bproc

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from shapely.geometry import Point, shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_policy import (
    POLICY_VERSION,
    candidate_grid,
    choose_from_candidates,
    eye_height,
    poses_separated,
    STANDABLE_HEIGHT_FRACTION,
    camera_clip_planes,
    is_structural_file,
    select_cameras,
    stands_on_furniture,
    validate_camera_locations,
    with_furniture_fallback,
)
from room_layout import recover_layout


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
    parser.add_argument("--camera-height-fraction", type=float, default=0.6,
                        help="Height cap as a fraction of recovered room height (0.2..0.8)")
    parser.add_argument("--min-clearance", type=float, default=0.25)
    parser.add_argument(
        "--allow-relaxed-clearance", action="store_true",
        help="Explicitly allow smaller clearances; recorded in render metadata.",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--layout-json", type=Path, help="Optional precomputed room-layout JSON")
    parser.add_argument("--plan-only", action="store_true", help="Validate cameras and save metadata without rendering")
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


def choose_cameras(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    meshes: list[bproc.types.MeshObject],
    camera_height: float,
    min_clearance: float,
    views: int,
    allow_relaxed: bool = False,
    layout=None,
    height_fraction=0.6,
    standable_fraction=STANDABLE_HEIGHT_FRACTION,
) -> tuple[list[np.ndarray], list[float]]:
    if layout is None:
        raise ValueError("An explicit recovered floor/ceiling envelope is required")
    bvh = bproc.object.create_bvh_tree_multi_objects(meshes)
    structural = [mesh for mesh in meshes if is_structural_file(str(mesh.get_cp("source_file")))]
    structure_bvh = bproc.object.create_bvh_tree_multi_objects(structural)
    polygon = shape(layout["polygon"])
    region = shape(layout["camera_region"])
    floor_z, ceiling_z = layout["floor_z"], layout["ceiling_z"]
    height = ceiling_z-floor_z
    eye_z = eye_height(floor_z, height, camera_height, height_fraction)
    object_boxes = []
    for mesh in meshes:
        if not is_structural_file(str(mesh.get_cp("source_file"))):
            points = np.asarray(mesh.get_bound_box())
            object_boxes.append((points.min(0), points.max(0)))
    candidates = []
    on_furniture = []
    rejected = Counter()
    grid = candidate_grid(region.bounds, eye_z)
    for candidate in grid:
        point = Point(candidate[:2])
        if not region.contains(point) or polygon.boundary.distance(point) < height*0.02:
            rejected["outside_interior"] += 1
            continue
        standing = stands_on_furniture(candidate, object_boxes, floor_z, height,
                                       standable_fraction)
        if standing:
            rejected["on_furniture_footprint"] += 1
        floor_hit = structure_bvh.ray_cast(Vector(candidate), Vector((0, 0, -1)), height*2)
        ceiling_hit = structure_bvh.ray_cast(Vector(candidate), Vector((0, 0, 1)), height*2)
        if floor_hit[0] is None or ceiling_hit[0] is None:
            rejected["missing_floor_or_ceiling_ray_hit"] += 1
            continue
        if abs(floor_hit[0].z-floor_z) > max(height*0.02, 1e-5):
            rejected["different_floor_height"] += 1
            continue
        if ceiling_hit[0].z <= eye_z or floor_hit[0].z >= eye_z:
            rejected["invalid_vertical_order"] += 1
            continue
        nearest = bvh.find_nearest(Vector(candidate))
        if nearest is None or nearest[3] is None:
            continue
        # Keep the camera away from the true contour as well as triangle surfaces.
        accepted = on_furniture if standing else candidates
        accepted.append((min(float(nearest[3]), float(polygon.boundary.distance(point))), candidate))
    # Keep diversity within the reasonably open part of the room: choosing the
    # farthest point from the first camera over the entire grid hugs the walls.
    try:
        selected, used_furniture_fallback = choose_from_candidates(
            candidates, on_furniture, views, min_clearance, allow_relaxed)
    except ValueError as error:
        raise ValueError(f"{error} Candidate diagnostics: {dict(rejected)}, "
                         f"accepted={len(candidates)}, best_clearance="
                         f"{max((c for c, _ in candidates), default=0):.6g}, "
                         f"room_height={height:.6g}") from error
    locations = [np.asarray(point) for _, point in selected]
    validate_camera_locations(locations, bounds_min, bounds_max, views)
    if not poses_separated(locations, height):
        raise ValueError("Camera poses are distinct but insufficiently separated")
    return locations, [clearance for clearance, _ in selected], used_furniture_fallback


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
    if args.layout_json:
        layout = json.loads(args.layout_json.read_text())
        layout = layout.get("layout", layout)
    else:
        layout = recover_layout(args.room_dir.resolve())
    bounds_min, bounds_max = np.asarray(layout["bounds_min"]), np.asarray(layout["bounds_max"])
    camera_locations, camera_clearances, used_furniture_fallback = choose_cameras(
        bounds_min,
        bounds_max,
        meshes,
        args.camera_height,
        args.min_clearance,
        args.views,
        args.allow_relaxed_clearance,
        layout,
        args.camera_height_fraction,
    )

    bproc.camera.set_resolution(args.width, args.height)
    camera = bpy.context.scene.camera.data
    camera.clip_start, camera.clip_end = camera_clip_planes(
        float(bounds_max[2]-bounds_min[2]), min(camera_clearances))
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

    room_height = bounds_max[2]-bounds_min[2]
    light_locations = [[p[0], p[1], p[2]+0.5*(bounds_max[2]-p[2])] for p in camera_locations]
    for location in light_locations:
        light = bproc.types.Light()
        light.set_location(location)
        light.set_energy(100.0*room_height**2 / len(light_locations))
        light.set_radius(0.1*room_height)

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
    if not args.plan_only:
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(str(args.output_dir), data)
    (args.output_dir / metadata_filename).write_text(
        json.dumps(
            {
                "format_version": 3,
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
                "camera_height_fraction": args.camera_height_fraction,
                "camera_clip_planes": [camera.clip_start, camera.clip_end],
                "camera_rotation_euler_xyz": rotation.tolist(),
                "coordinate_frame": "Blender world XYZ, Z-up",
                "gltf_to_blender_axes": ["x", "-z", "y"],
                "metric_scale_certified": False,
                "bounds_source": "recovered room-facing floor and ceiling planes",
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "mesh_count": len(meshes),
                "resolution": [args.width, args.height],
                "views": args.views,
                "standable_height_fraction": STANDABLE_HEIGHT_FRACTION,
                "cameras_on_furniture_fallback": used_furniture_fallback,
                "samples": args.samples,
                "layout": layout,
                "camera_checks": ["inside_floor_and_ceiling_footprints", "floor_and_ceiling_raycast",
                                  "off_furniture_footprints", "surface_and_contour_clearance", "distinct_separated_poses"],
                "light_locations": light_locations,
                "plan_only": args.plan_only,
                "implementation_sha256": {
                    name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in ("render.py", "camera_policy.py", "room_layout.py", "glb_geometry.py")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
