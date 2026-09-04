#!/usr/bin/env python3
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import html
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3dfront_panorama_renderer"))
from _lib.quality import image_checks, make_preview
from camera_policy import POLICY_VERSION
from glb_geometry import glb_triangles, floor_footprint, vertical_hits, glb_bounds
from _lib.layout_targets import polygon_targets


def audit(task):
    record, output_root, report_root = task
    output = Path(output_root) / record["split"] / record["room_id"]
    name = record["split"]+"__"+record["room_id"].replace("/", "__")
    result = {"room_id": record["room_id"], "split": record["split"], "flags": [],
              "views": [], "preview": f"previews/{name}.jpg", "geometry_pass": False}
    thumbs, cameras, footprint = [], [], None
    try:
        metadata = json.loads((output / "render.json").read_text())
        if metadata["camera_policy_version"] != POLICY_VERSION or metadata["plan_only"]:
            raise ValueError("Not a completed %s render" % POLICY_VERSION)
        layout = metadata["layout"]
        polygon = shape(layout["polygon"])
        height = layout["ceiling_z"]-layout["floor_z"]
        sources = [glb_triangles(p) for p in Path(record["room_dir"]).glob("*.glb")
                   if p.stem in {"floor", "ceil", "wall", "others"}]
        triangles = np.concatenate(sources)
        normal_z = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])[:, 2]
        tolerance = max(height*2e-4, 1e-5)
        on_floor = np.max(np.abs(triangles[:, :, 2]-layout["floor_z"]), axis=1) < tolerance
        floor_triangles = triangles[on_floor & (normal_z > 1e-12)]
        full_footprint, _ = floor_footprint(floor_triangles)
        components = list(full_footprint.geoms) if full_footprint.geom_type == "MultiPolygon" else [full_footprint]
        footprint = max(components, key=lambda p: p.intersection(polygon).area)
        iou = footprint.intersection(polygon).area/footprint.union(polygon).area
        result.update(bounds_min=layout["bounds_min"], bounds_max=layout["bounds_max"],
                      target_polygon=layout["polygon"],
                      floor_proxy_iou=iou, floor_oriented_bbox_fill=footprint.area/footprint.minimum_rotated_rectangle.area,
                      layout_sources=layout["sources"])
        if iou < 0.98:
            result["flags"].append("floor_layout_mismatch")
        cameras = np.asarray(metadata["camera_locations"])
        clipping = metadata.get("camera_clip_planes")
        if clipping is None:
            result["flags"].append("camera_clipping_unverified")
        elif clipping[0] >= min(metadata["camera_clearances"]):
            result["flags"].append("near_clip_exceeds_camera_clearance")
        if len(cameras) != metadata["views"] or len(set(map(tuple, np.round(cameras, 7)))) != len(cameras):
            result["flags"].append("camera_count_or_duplicates")
        object_paths = [p for p in Path(record["room_dir"]).glob("*.glb")
                        if p.stem not in {"floor", "ceil", "wall", "others"}]
        boxes = [glb_bounds(p) for p in object_paths]
        room_span = max(np.asarray(layout["bounds_max"])-np.asarray(layout["bounds_min"]))
        result["oversized_source_objects"] = [p.name for p, (lo, hi) in zip(object_paths, boxes)
                                               if max(hi-lo) > 5*room_span]
        if result["oversized_source_objects"]:
            result["flags"].append("oversized_source_object_requires_review")
        for index, camera in enumerate(cameras):
            view = {"index": index, "flags": []}
            point = Point(camera[:2])
            floor_hits = vertical_hits(floor_triangles, camera[:2])
            supported = footprint.covers(point) and np.any(floor_hits < camera[2])
            view["floor_below_camera"] = bool(supported)
            if not supported:
                view["flags"].append("no_floor_below_camera")
            ceiling_hits = vertical_hits(triangles[normal_z < -1e-12], camera[:2])
            above = ceiling_hits[ceiling_hits > camera[2]+1e-5]
            if not len(above):
                view["flags"].append("no_ceiling_above_camera")
            else:
                view["actual_ceiling_above_camera_z"] = float(above.min())
                view["ceiling_target_height_delta"] = float(above.min()-layout["ceiling_z"])
                if abs(view["ceiling_target_height_delta"]) > 0.1*height:
                    view["flags"].append("ceiling_height_requires_review")
            if any(np.all(camera >= low) and np.all(camera <= high) for low, high in boxes):
                view["flags"].append("inside_furniture_bbox")
            metrics, flags, thumb = image_checks(output / f"{index}.hdf5", height)
            view.update(metrics)
            view["flags"].extend(flags)
            target = polygon_targets(layout, camera, *metadata["resolution"])
            draw = ImageDraw.Draw(thumb)
            for boundary in target["boundary"]:
                ys = (boundary/np.pi+0.5)*thumb.height
                draw.line([(float(x*thumb.width/len(ys)), float(y)) for x, y in enumerate(ys)],
                          fill=(255, 210, 0), width=1)
            thumbs.append(thumb)
            result["views"].append(view)
        hashes = [v["rgb_sha256"] for v in result["views"]]
        if len(set(hashes)) != len(hashes):
            result["flags"].append("duplicate_rgb")
        result["flags"].extend(sorted({f for v in result["views"] for f in v["flags"]}))
        hard = {"floor_layout_mismatch", "camera_count_or_duplicates", "no_floor_below_camera",
                "no_ceiling_above_camera", "inside_furniture_bbox", "duplicate_rgb",
                "near_clip_exceeds_camera_clearance"}
        result["geometry_pass"] = not bool(hard.intersection(result["flags"]))
    except Exception as error:
        result["flags"].append("audit_error")
        result["error"] = f"{type(error).__name__}: {error}"
        footprint, cameras = None, []
    make_preview(result, footprint, cameras, thumbs, Path(report_root) / result["preview"])
    (Path(report_root) / "rooms" / f"{name}.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rooms_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.report_root.exists() and any(args.report_root.iterdir()):
        parser.error("Use a fresh report directory")
    (args.report_root / "previews").mkdir(parents=True)
    (args.report_root / "rooms").mkdir()
    records = [json.loads(s) for s in args.rooms_jsonl.read_text().splitlines() if s.strip()]
    if args.limit:
        records = records[:args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(audit, [(r, str(args.output_root), str(args.report_root)) for r in records]))
    summary = {"rooms": len(results), "views": sum(len(r["views"]) for r in results),
               "geometry_pass_rooms": sum(r["geometry_pass"] for r in results),
               "room_flags": dict(Counter(f for r in results for f in set(r["flags"]))),
               "full_manifest": args.limit is None,
               "training_approved": False,
               "note": "Geometry audit only. Target overlays still require review; full training also requires derived-data QA."}
    (args.report_root / "rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results))
    (args.report_root / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    ordered = sorted(results, key=lambda r: (not r["geometry_pass"],
        "oversized_source_object_requires_review" in r["flags"],
        "ceiling_height_requires_review" in r["flags"],
        1-r.get("floor_oriented_bbox_fill", 1)), reverse=True)
    cards = []
    for result in ordered:
        caption = html.escape(result["split"]+"/"+result["room_id"]+": "+", ".join(result["flags"]))
        preview = html.escape(result["preview"], quote=True)
        cards.append(f'<details><summary>{caption}</summary><a href="{preview}">'
                     f'<img loading="lazy" src="{preview}"></a></details>')
    (args.report_root / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Repaired panorama QA</title>'
        '<style>body{font:15px sans-serif;background:#151c28;color:#eee;margin:24px}'
        'summary{padding:8px;cursor:pointer}img{max-width:100%}a{color:#8de}</style>'
        '<h1>Repaired panorama QA</h1><p>Geometry failures first, then source outliers, '
        'ceiling-height warnings and complex contours. Yellow: target floor contour '
        'and extruded-layout boundaries. Flags are not automatic exclusions.</p><pre>'
        +html.escape(json.dumps(summary, indent=2))+'</pre>'+''.join(cards))
    print(json.dumps(summary, indent=2), flush=True)
    if not all(r["geometry_pass"] for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
