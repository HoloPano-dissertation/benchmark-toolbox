import importlib.util
import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("shapely")

MODULE_PATH = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
               / "_lib" / "relations.py")
spec = importlib.util.spec_from_file_location("relations", MODULE_PATH)
relations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relations)

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ROOM = {"polygon": {"type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0],
                                     [2.0, 3.0], [2.0, 5.0], [0.0, 5.0], [0.0, 0.0]]]},
        "floor_z": 0.0, "ceiling_z": 2.8}


def obj(object_id, center, size, basis=None):
    return {"object_id": object_id,
            "bbox": {"center": list(center), "size": list(size),
                     "basis": basis or IDENTITY}}


def test_boxes_apart_by_more_than_the_slack_do_not_touch():
    a = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "basis": IDENTITY}
    b = {"center": [2.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "basis": IDENTITY}
    assert not relations.boxes_touch(a, b, expand=0.1)


def test_boxes_within_the_slack_touch():
    a = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "basis": IDENTITY}
    b = {"center": [1.05, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "basis": IDENTITY}
    assert relations.boxes_touch(a, b, expand=0.1)


def test_boxes_overlapping_only_in_plan_do_not_touch():
    low = {"center": [0.0, 0.0, 0.25], "size": [1.0, 1.0, 0.5], "basis": IDENTITY}
    high = {"center": [0.0, 0.0, 2.0], "size": [1.0, 1.0, 0.5], "basis": IDENTITY}
    assert not relations.boxes_touch(low, high, expand=0.1)


def test_rotated_box_touching_is_detected():
    angle = math.radians(45)
    rotated = {"center": [1.2, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
               "basis": [[math.cos(angle), math.sin(angle), 0.0],
                         [-math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]}
    upright = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "basis": IDENTITY}
    assert relations.boxes_touch(rotated, upright, expand=0.1)


def test_a_wall_is_produced_for_every_edge_of_a_non_rectangular_contour():
    walls = relations.wall_boxes(ROOM["polygon"], ROOM["floor_z"], ROOM["ceiling_z"])
    assert len(walls) == 6
    assert all(wall["size"][2] == pytest.approx(2.8) for wall in walls)


def test_object_against_a_wall_and_on_the_floor_is_related_to_both():
    objects = [obj("bed", [0.5, 2.0, 0.3], [0.9, 2.0, 0.6])]
    found, _ = relations.scene_relations(objects, ROOM)
    kinds = {item["relation_type"] for item in found}
    assert "touching_floor" in kinds
    assert "touching_wall" in kinds
    assert "touching_ceiling" not in kinds


def test_two_touching_objects_are_reported_once():
    objects = [obj("a", [2.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
               obj("b", [3.0, 1.0, 0.5], [1.0, 1.0, 1.0])]
    found, _ = relations.scene_relations(objects, ROOM)
    touching = [item for item in found if item["relation_type"] == "touching"]
    assert len(touching) == 1
    assert {touching[0]["source_id"], touching[0]["target_id"]} == {"a", "b"}


def test_object_in_the_middle_of_the_room_touches_no_wall():
    objects = [obj("table", [1.0, 1.5, 1.0], [0.6, 0.6, 0.6])]
    found, _ = relations.scene_relations(objects, ROOM)
    assert all(item["relation_type"] != "touching_wall" for item in found)


def test_layout_is_moved_into_the_camera_frame():
    moved = relations.camera_frame_layout(
        {"polygon": ROOM["polygon"], "floor_z": 0.0, "ceiling_z": 2.8}, [1.0, 2.0, 1.6])
    ring = moved["polygon"]["coordinates"][0]
    assert ring[0] == [-1.0, -2.0]
    assert moved["floor_z"] == pytest.approx(-1.6)
    assert moved["ceiling_z"] == pytest.approx(1.2)
