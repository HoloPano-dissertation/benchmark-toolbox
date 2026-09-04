import json
import math

DEFAULT_REFERENCE_HEIGHT = 2.6

NORMALISED_SHELL_EXTENT = 1.9

RECOVERY_COLLAPSE_RATIO = 0.5
RECOVERY_REVIEW_RATIO = 1.5

PLAUSIBLE_FLOOR_AREA = (1.0, 200.0)
PLAUSIBLE_MAX_EXTENT = (0.8, 30.0)


class ScaleError(ValueError):
    pass


def load_scale_table(path):
    table = json.loads(path.read_text(encoding="utf-8"))
    entries = table.get("scales", table)
    result = {}
    for room_id, value in entries.items():
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0:
            raise ScaleError(f"Table holds a non-positive scale for {room_id}: {value}")
        result[str(room_id)] = scale
    return result


def normalised_height(layout):
    height = float(layout["ceiling_z"]) - float(layout["floor_z"])
    if not math.isfinite(height) or height <= 0:
        raise ScaleError(f"Room height is not positive: {height}")
    return height


def height_scale(layout, reference_height=DEFAULT_REFERENCE_HEIGHT):
    return reference_height / normalised_height(layout)


def shell_extent(layout):
    minimum = layout["bounds_min"]
    maximum = layout["bounds_max"]
    return max(float(maximum[axis]) - float(minimum[axis]) for axis in range(3))


def check_recovery(layout, source_shell_extent=NORMALISED_SHELL_EXTENT):
    if source_shell_extent is None:
        return None, None
    source_shell_extent = float(source_shell_extent)
    if not math.isfinite(source_shell_extent) or source_shell_extent <= 0:
        raise ScaleError(f"Source shell extent is not positive: {source_shell_extent}")
    ratio = shell_extent(layout) / source_shell_extent
    if ratio < RECOVERY_COLLAPSE_RATIO:
        raise ScaleError(
            f"Recovered contour is {ratio:.3f} of the room shell, below "
            f"{RECOVERY_COLLAPSE_RATIO}; the contour collapsed for this room"
        )
    if ratio > RECOVERY_REVIEW_RATIO:
        return ratio, (
            f"Recovered contour is {ratio:.3f} of the room shell, above "
            f"{RECOVERY_REVIEW_RATIO}; it may reach beyond the room"
        )
    return ratio, None


def check_metric_room(layout, scale):
    area = float(layout["area"]) * scale * scale
    low, high = PLAUSIBLE_FLOOR_AREA
    if not low <= area <= high:
        raise ScaleError(f"Floor area {area:.2f} m^2 is outside [{low}, {high}]")
    extent = shell_extent(layout) * scale
    low, high = PLAUSIBLE_MAX_EXTENT
    if not low <= extent <= high:
        raise ScaleError(f"Largest dimension {extent:.2f} m is outside [{low}, {high}]")
    return area


def resolve_scale(room_id, layout, table=None,
                  source_shell_extent=NORMALISED_SHELL_EXTENT,
                  reference_height=DEFAULT_REFERENCE_HEIGHT, strict=True):
    reasons = []
    review = []
    ratio = None
    try:
        ratio, review_reason = check_recovery(layout, source_shell_extent)
        if review_reason:
            review.append(review_reason)
    except ScaleError as error:
        if strict:
            raise
        reasons.append(str(error))

    if table and room_id in table:
        scale, source = table[room_id], "table"
    else:
        scale, source = height_scale(layout, reference_height), "ceiling_height"

    try:
        check_metric_room(layout, scale)
    except ScaleError as error:
        if strict:
            raise
        reasons.append(str(error))

    height = normalised_height(layout)
    report = {
        "room_id": room_id,
        "scale": scale,
        "source": source,
        "normalised_height": height,
        "normalised_shell_extent": shell_extent(layout),
        "source_shell_extent": source_shell_extent,
        "recovery_ratio": ratio,
        "review": review,
        "metric_height": height * scale,
        "metric_floor_area": float(layout["area"]) * scale * scale,
        "rejected": reasons,
    }
    return scale, report


def scale_layout_geometry(layout, scale):
    scaled = dict(layout)
    for key in ("floor_z", "ceiling_z", "footprint_simplification_tolerance"):
        if layout.get(key) is not None:
            scaled[key] = float(layout[key]) * scale
    for key in ("bounds_min", "bounds_max", "furniture_anchor"):
        if layout.get(key) is not None:
            scaled[key] = [float(value) * scale for value in layout[key]]
    if layout.get("area") is not None:
        scaled["area"] = float(layout["area"]) * scale * scale
    for key in ("polygon", "camera_region"):
        if layout.get(key) is not None:
            scaled[key] = scale_geojson(layout[key], scale)
    return scaled


def scale_geojson(geometry, scale):
    if "geometries" in geometry:
        return {
            "type": geometry["type"],
            "geometries": [scale_geojson(part, scale) for part in geometry["geometries"]],
        }
    return {
        "type": geometry["type"],
        "coordinates": _scale_nested(geometry["coordinates"], scale),
    }


def _scale_nested(value, scale):
    if isinstance(value, (int, float)):
        return float(value) * scale
    return [_scale_nested(item, scale) for item in value]
