import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
               / "_lib" / "metric_scale.py")
spec = importlib.util.spec_from_file_location("metric_scale", MODULE_PATH)
metric_scale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric_scale)


def layout(height=1.5, area=3.5, extent=1.9):
    half = extent / 2
    return {
        "floor_z": -height / 2,
        "ceiling_z": height / 2,
        "area": area,
        "bounds_min": [-half, -half, -height / 2],
        "bounds_max": [half, half, height / 2],
        "furniture_anchor": [0.1, 0.2],
        "footprint_simplification_tolerance": 0.001,
        "polygon": {"type": "Polygon",
                    "coordinates": [[[-half, -half], [half, -half],
                                     [half, half], [-half, half], [-half, -half]]]},
        "camera_region": {"type": "GeometryCollection", "geometries": [
            {"type": "Polygon",
             "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]}]},
    }


def test_rooms_of_different_normalisation_become_comparable():
    tall, short = layout(height=1.9), layout(height=0.95)
    tall_height = (tall["ceiling_z"] - tall["floor_z"]) * metric_scale.height_scale(tall)
    short_height = (short["ceiling_z"] - short["floor_z"]) * metric_scale.height_scale(short)
    assert tall_height == pytest.approx(short_height)
    assert tall_height == pytest.approx(metric_scale.DEFAULT_REFERENCE_HEIGHT)


def test_table_takes_precedence_over_the_anchor():
    scale, report = metric_scale.resolve_scale(
        "house/room", layout(), table={"house/room": 1.75})
    assert (scale, report["source"]) == (1.75, "table")


def test_collapsed_contour_is_rejected_although_its_metres_look_plausible():
    collapsed = layout(height=0.076, area=0.007, extent=0.11)
    _, report = metric_scale.resolve_scale("house/room", collapsed, strict=False)
    assert report["rejected"]
    assert 1.0 <= report["metric_floor_area"] <= 200.0
    with pytest.raises(metric_scale.ScaleError):
        metric_scale.resolve_scale("house/room", collapsed)


def test_contour_wider_than_the_shell_is_flagged_but_kept():
    wide = layout(extent=2.0 * metric_scale.NORMALISED_SHELL_EXTENT, area=20.0)
    _, report = metric_scale.resolve_scale("house/room", wide)
    assert report["review"] and report["rejected"] == []


def test_geometry_scaling_multiplies_lengths_squares_area_and_walks_collections():
    scaled = metric_scale.scale_layout_geometry(layout(height=1.4, area=3.5), 2.0)
    assert scaled["ceiling_z"] == pytest.approx(1.4)
    assert scaled["area"] == pytest.approx(14.0)
    assert scaled["polygon"]["coordinates"][0][1][0] == pytest.approx(1.9)
    assert scaled["camera_region"]["geometries"][0]["coordinates"][0][1] == [2.0, 0.0]


def test_scale_table_rejects_a_non_positive_entry(tmp_path):
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"scales": {"house/room": 0.0}}), encoding="utf-8")
    with pytest.raises(metric_scale.ScaleError):
        metric_scale.load_scale_table(path)


def test_a_room_falling_back_to_the_anchor_is_not_certified():
    from_table, _ = metric_scale.resolve_scale(
        "house/a", layout(), table={"house/a": 1.75})
    _, anchored = metric_scale.resolve_scale("house/b", layout(), table={"house/a": 1.75})
    assert anchored["source"] == "ceiling_height"
    assert from_table == 1.75


def test_furniture_dimensions_fall_back_from_size_to_bbox():
    import importlib.util
    path = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
            / "source_metadata.py")
    spec = importlib.util.spec_from_file_location("source_metadata", path)
    import sys
    sys.path.insert(0, str(path.parent / "_lib"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Archive:
        def read(self, _):
            return json.dumps({"furniture": [
                {"jid": "a", "size": [1.0, 2.0, 3.0]},
                {"jid": "b", "bbox": [4.0, 5.0, 6.0]},
                {"jid": "c"}]}).encode()

    sizes, _, _, _ = module.scene_furniture(Archive(), "house")
    assert sizes == {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    assert module.flat_dimensions([[7.0, 8.0, 9.0]]) == [7.0, 8.0, 9.0]
    assert module.flat_dimensions([[1.0, 2.0], [3.0, 4.0]]) is None
    assert module.flat_dimensions(None) is None


def test_the_stretch_of_an_instance_is_applied_before_comparing():
    import importlib.util, sys
    path = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
            / "source_metadata.py")
    sys.path.insert(0, str(path.parent / "_lib"))
    spec = importlib.util.spec_from_file_location("source_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    objects = [{"object_id": "Bed_11111111-1111-1111-1111-111111111111_1",
                "bbox_world": {"size": [0.9, 1.0, 2.0]}}]
    canonical = {"11111111-1111-1111-1111-111111111111": [1.0, 1.0, 2.0]}
    without, _, spread_without = module.room_scale(objects, canonical)
    stretched = {"11111111-1111-1111-1111-111111111111": [[0.9, 1.0, 2.0]]}
    with_scale, _, spread_with = module.room_scale(objects, canonical, stretched)
    assert spread_with < spread_without, "the stretch must reconcile the axes"
    assert spread_with == pytest.approx(0.0)
    assert with_scale == pytest.approx(1.0)
