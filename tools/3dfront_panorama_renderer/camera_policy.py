"""Deterministic, Blender-independent camera selection and coordinate checks."""

import math
from pathlib import Path


POLICY_VERSION = "structural-distinct-v4"
STRUCTURAL_STEMS = frozenset({"ceil", "floor", "wall", "others"})


def is_structural_file(filename):
    # Case matters: `others.glb` is structural, `Others_UUID_1.glb` is furniture.
    return Path(filename).stem in STRUCTURAL_STEMS


def validate_bounds(lower, upper):
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("Room bounds must be XYZ triples")
    if not all(math.isfinite(float(v)) for v in (*lower, *upper)):
        raise ValueError("Room bounds must be finite")
    if any(float(hi) - float(lo) <= 1e-6 for lo, hi in zip(lower, upper)):
        raise ValueError("Room must have non-degenerate XYZ extents")


def camera_grid(lower, upper, requested_height):
    validate_bounds(lower, upper)
    if not math.isfinite(requested_height) or requested_height <= 0:
        raise ValueError("camera-height must be finite and positive")
    height = upper[2] - lower[2]
    z = lower[2] + min(requested_height, 0.60 * height)
    z = max(lower[2] + 0.20 * height, min(z, upper[2] - 0.20 * height))
    fractions = (0.5, 0.3, 0.7, 0.15, 0.85, 0.05, 0.95)
    return [
        (lower[0] + x * (upper[0] - lower[0]),
         lower[1] + y * (upper[1] - lower[1]), z)
        for x in fractions for y in fractions
    ]


def select_cameras(candidates, views, min_clearance, allow_relaxed=False):
    """Return (clearance, XYZ) pairs; never pad the result with duplicate poses."""
    if not isinstance(views, int) or views < 1:
        raise ValueError("views must be a positive integer")
    if not math.isfinite(min_clearance) or min_clearance <= 0:
        raise ValueError("min-clearance must be finite and positive")
    unique = {}
    for clearance, point in candidates:
        point = tuple(float(v) for v in point)
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            continue
        if not math.isfinite(clearance) or clearance < 0:
            continue
        key = tuple(round(v, 7) for v in point)
        if key not in unique or clearance > unique[key][0]:
            unique[key] = (float(clearance), point)
    ranked = sorted(unique.values(), key=lambda item: (-item[0], item[1]))
    pool = [item for item in ranked if item[0] >= min_clearance]
    if len(pool) < views:
        if not allow_relaxed:
            raise ValueError(
                f"Only {len(pool)} distinct poses satisfy clearance "
                f"{min_clearance}; requested {views}. Reduce --views or "
                "--min-clearance, or explicitly use --allow-relaxed-clearance."
            )
        # Relax only as much as necessary, choosing the safest remaining points.
        rejected = [item for item in ranked if item[0] < min_clearance]
        pool.extend(rejected[:views - len(pool)])
    if len(pool) < views:
        raise ValueError(f"Only {len(pool)} distinct finite poses for {views} views")
    selected = [pool.pop(0)]
    while len(selected) < views:
        index = max(
            range(len(pool)),
            key=lambda i: (
                min(math.hypot(pool[i][1][0] - p[1][0],
                               pool[i][1][1] - p[1][1]) for p in selected),
                pool[i][0],
            ),
        )
        selected.append(pool.pop(index))
    return selected


def validate_camera_locations(locations, lower, upper, views):
    validate_bounds(lower, upper)
    if len(locations) != views:
        raise ValueError("Unexpected camera count")
    if len({tuple(round(float(v), 7) for v in p) for p in locations}) != views:
        raise ValueError("Duplicate camera positions")
    for point in locations:
        if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("Camera coordinates must be finite XYZ triples")
        if not all(lo - 1e-7 <= v <= hi + 1e-7
                   for v, lo, hi in zip(point, lower, upper)):
            raise ValueError("Camera is outside the structural room bounds")
