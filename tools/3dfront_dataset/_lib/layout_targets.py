import math
import numpy as np
from shapely.geometry import Point, shape


def polygon_segments(polygon):
    rings = [polygon.exterior, *polygon.interiors]
    return np.concatenate([np.stack((np.asarray(r.coords)[:-1], np.asarray(r.coords)[1:]), axis=1)
                           for r in rings])


def ray_ranges(polygon, camera_xy, directions):
    segments = polygon_segments(polygon)
    start, edge = segments[:, 0], segments[:, 1]-segments[:, 0]
    delta = start-np.asarray(camera_xy)
    directions = np.asarray(directions)
    det = directions[:, None, 0]*edge[None, :, 1]-directions[:, None, 1]*edge[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (delta[None, :, 0]*edge[None, :, 1]-delta[None, :, 1]*edge[None, :, 0])/det
        fraction = (delta[None, :, 0]*directions[:, None, 1]-delta[None, :, 1]*directions[:, None, 0])/det
    valid = (np.abs(det) > 1e-10) & (distance > 1e-7) & (fraction >= -1e-7) & (fraction <= 1+1e-7)
    return np.min(np.where(valid, distance, np.inf), axis=1)


def polygon_targets(layout, camera, width=1024, height=512):
    polygon = shape(layout["polygon"])
    camera = np.asarray(camera, dtype=float)
    if not polygon.contains(Point(camera[:2])):
        raise ValueError("Camera must be inside the actual polygon, not just its bbox")
    floor_z, ceiling_z = float(layout["floor_z"]), float(layout["ceiling_z"])
    if not floor_z < camera[2] < ceiling_z:
        raise ValueError("Invalid camera height")
    longitude = ((np.arange(width)+0.5)/width-0.5)*2*np.pi
    directions = np.column_stack((np.sin(longitude), np.cos(longitude)))
    ranges = ray_ranges(polygon, camera[:2], directions)
    if not np.isfinite(ranges).all():
        raise ValueError("Some panorama rays do not intersect the room contour")
    boundary = np.stack((-np.arctan2(ceiling_z-camera[2], ranges),
                         np.arctan2(camera[2]-floor_z, ranges)))
    vertices = polygon_segments(polygon)[:, 0]
    delta = vertices-camera[:2]
    distance = np.linalg.norm(delta, axis=1)
    vertex_ranges = ray_ranges(polygon, camera[:2], delta/distance[:, None])
    visible = np.abs(vertex_ranges-distance) <= np.maximum(distance*1e-5, 1e-6)
    pixel_x = ((0.5+np.arctan2(delta[visible, 0], delta[visible, 1])/(2*np.pi)) % 1)*width-0.5
    if not len(pixel_x):
        raise ValueError("No visible room vertices")
    dx = np.abs(np.arange(width)[None, :]-pixel_x[:, None])
    periodic_distance = np.minimum(dx, width-dx).min(0)
    corner = (0.96**periodic_distance)[None, :]
    return {"boundary": boundary.astype(np.float32), "corner": corner.astype(np.float32),
            "ranges": ranges.astype(np.float32), "visible_corner_x": pixel_x.astype(np.float32)}


def native_polygon_corners(layout, camera, width, height):
    polygon = shape(layout["polygon"])
    if polygon.interiors:
        raise ValueError("Native single-ring labels cannot encode floor-plan holes; use dense targets")
    vertices = np.asarray(polygon.exterior.coords)[:-1]
    camera = np.asarray(camera)
    delta = vertices-camera[:2]
    distance = np.linalg.norm(delta, axis=1)
    x = ((0.5+np.arctan2(delta[:, 0], delta[:, 1])/(2*np.pi)) % 1)*width-0.5
    top = (0.5-np.arctan2(layout["ceiling_z"]-camera[2], distance)/np.pi)*height-0.5
    bottom = (0.5+np.arctan2(camera[2]-layout["floor_z"], distance)/np.pi)*height-0.5
    corners = np.stack((np.column_stack((x, top)), np.column_stack((x, bottom))), axis=1).reshape(-1, 2)
    return np.roll(corners, -2*np.argmin(x), axis=0)
