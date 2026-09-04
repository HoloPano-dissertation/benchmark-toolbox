import math
from pathlib import Path

POLICY_VERSION = "shell-clipped-off-furniture"
STRUCTURAL_STEMS = frozenset({"ceil", "floor", "wall", "others"})


def camera_clip_planes(room_height, clearance):
    if not all(math.isfinite(v) and v > 0 for v in (room_height, clearance)):
        raise ValueError("Clipping requires positive finite room height and clearance")
    near = min(room_height * 1e-4, clearance * 0.01)
    return near, max(room_height * 100, near * 1000)


def is_structural_file(filename):
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


STANDABLE_HEIGHT_FRACTION = 0.1


def stands_on_furniture(candidate, object_boxes, floor_z, room_height,
                        standable_fraction=STANDABLE_HEIGHT_FRACTION):
    if not math.isfinite(room_height) or room_height <= 0:
        raise ValueError("Room height must be finite and positive")
    ceiling_of_standable = floor_z + room_height * standable_fraction
    x, y = float(candidate[0]), float(candidate[1])
    for lower, upper in object_boxes:
        if upper[2] <= ceiling_of_standable:
            continue
        if lower[0] <= x <= upper[0] and lower[1] <= y <= upper[1]:
            return True
    return False


def with_furniture_fallback(candidates, on_furniture, views):
    if views <= 0:
        raise ValueError("views must be positive")
    if len(candidates) >= views or not on_furniture:
        return candidates, False
    return list(candidates) + list(on_furniture), True


GRID_STEPS = 31
GRID_MARGIN = 0.02
SEPARATION_FRACTION = 0.025


def min_clearance_for(room_height):
    return min(0.1, 0.04*room_height)


def eye_height(floor_z, room_height, camera_height, height_fraction):
    if not 0.2 <= height_fraction <= 0.8:
        raise ValueError("camera-height-fraction must be between 0.2 and 0.8")
    if camera_height <= 0 or not math.isfinite(camera_height):
        raise ValueError("camera-height must be positive and finite")
    return floor_z + min(camera_height, height_fraction*room_height)


def candidate_grid(bounds, eye_z, steps=GRID_STEPS, margin=GRID_MARGIN):
    x0, y0, x1, y1 = bounds
    fractions = [margin + (1-2*margin)*index/(steps-1) for index in range(steps)]
    return [(x0+x*(x1-x0), y0+y*(y1-y0), eye_z) for x in fractions for y in fractions]


def choose_from_candidates(candidates, on_furniture, views, min_clearance,
                           allow_relaxed=False):
    candidates, used_fallback = with_furniture_fallback(candidates, on_furniture, views)
    if candidates:
        preferred_clearance = max(min_clearance, max(c for c, _ in candidates)*0.5)
        preferred = [item for item in candidates if item[0] >= preferred_clearance]
        if len(preferred) >= views:
            candidates = preferred
    return select_cameras(candidates, views, min_clearance, allow_relaxed), used_fallback


def poses_separated(locations, room_height, fraction=SEPARATION_FRACTION):
    for index, first in enumerate(locations):
        for second in locations[index+1:]:
            gap = math.hypot(first[0]-second[0], first[1]-second[1])
            if gap < room_height*fraction:
                return False
    return True
