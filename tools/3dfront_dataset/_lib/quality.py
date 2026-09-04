import hashlib
import h5py
import numpy as np
from PIL import Image, ImageDraw


def image_checks(hdf5_path, room_height):
    with h5py.File(hdf5_path, "r") as stream:
        colors = stream["colors"][...]
        depth = np.asarray(stream["depth"][...], dtype=float)
    if colors.ndim != 3 or colors.shape[2] != 3 or depth.shape != colors.shape[:2]:
        raise ValueError("Expected aligned HxWx3 RGB and HxW depth")
    gray = colors.astype(float).mean(axis=2) / 255
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e6)
    metrics = {
        "rgb_sha256": hashlib.sha256(colors.tobytes()).hexdigest(),
        "rgb_mean": float(gray.mean()),
        "rgb_std": float(gray.std()),
        "dark_fraction": float((gray < 0.025).mean()),
        "bright_fraction": float((gray > 0.98).mean()),
        "depth_valid_fraction": float(valid.mean()),
        "near_surface_fraction": float((valid & (depth < 0.05 * room_height)).mean()),
    }
    flags = []
    if metrics["dark_fraction"] > 0.80:
        flags.append("mostly_dark")
    if metrics["bright_fraction"] > 0.80:
        flags.append("mostly_clipped")
    if metrics["rgb_std"] < 0.02:
        flags.append("almost_uniform_rgb")
    if metrics["depth_valid_fraction"] < 0.95:
        flags.append("missing_depth")
    if metrics["near_surface_fraction"] > 0.20:
        flags.append("near_surface_dominates")
    image = Image.fromarray(colors)
    image.thumbnail((448, 224))
    return metrics, flags, image


def make_preview(result, footprint, cameras, thumbs, destination):
    canvas = Image.new("RGB", (1220, 610), "#151c28")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), f'{result["split"]} / {result["room_id"]}', fill="white")
    if footprint is not None:
        low, high = np.asarray(result["bounds_min"][:2]), np.asarray(result["bounds_max"][:2])
        low = np.minimum(low, np.asarray(footprint.bounds[:2]))
        high = np.maximum(high, np.asarray(footprint.bounds[2:]))
        scale = 265 / max(high - low)

        def xy(point):
            return (20 + (point[0]-low[0])*scale, 335-(point[1]-low[1])*scale)

        parts = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else [footprint]
        for polygon in parts:
            draw.polygon([xy(p) for p in polygon.exterior.coords], fill="#335748", outline="#83d5a6")
            for ring in polygon.interiors:
                draw.polygon([xy(p) for p in ring.coords], fill="#151c28", outline="#83d5a6")
        lo, hi = result["bounds_min"], result["bounds_max"]
        corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]), (lo[0], lo[1])]
        target_polygon = result.get("target_polygon")
        if target_polygon:
            for ring in target_polygon["coordinates"]:
                draw.line([xy(p) for p in ring], fill="#ffce70", width=2)
        else:
            draw.line([xy(p) for p in corners], fill="#ffce70", width=2)
        for index, camera in enumerate(cameras):
            x, y = xy(camera)
            valid = result["views"][index].get("floor_below_camera", False)
            draw.ellipse((x-5, y-5, x+5, y+5), fill="#76e0ff" if valid else "#ff4c63")
            draw.text((x+6, y-6), str(index), fill="white")
        target_name = "target contour" if target_polygon else "proxy bbox"
        draw.text((15, 350), f"Green: floor; yellow: {target_name}", fill="white")
        draw.text((15, 368), f'Floor/target IoU: {result["floor_proxy_iou"]:.3f}', fill="white")
        draw.text((15, 386), f'Floor rectangularity: {result["floor_oriented_bbox_fill"]:.3f}', fill="white")
    for index, thumb in enumerate(thumbs[:4]):
        left, top = 310+(index % 2)*454, 45+(index//2)*251
        if thumb is not None:
            canvas.paste(thumb, (left, top+20))
        draw.text((left, top), f'Camera {index}', fill="white")
    reasons = ", ".join(result["flags"]) or "No automatic review flags"
    for row, start in enumerate(range(0, min(len(reasons), 360), 140)):
        draw.text((14, 555+row*16), reasons[start:start+140], fill="#ffce70")
    canvas.save(destination, quality=86)
