import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

TRAINING = Path(__file__).resolve().parents[1] / "tools" / "3dfront_training"


@pytest.fixture(scope="module")
def naming():
    spec = importlib.util.spec_from_file_location(
        "export_dpc_scenes_naming", TRAINING / "export_dpc_scenes.py")
    import sys, types
    for name in ("utils", "utils.igibson_utils", "utils.relation_utils",
                 "utils.render_utils", "h5py"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["utils.igibson_utils"].IGScene = type("IGScene", (), {})
    sys.modules["utils.relation_utils"].RelationOptimization = type("R", (), {})
    sys.modules["utils.render_utils"].seg2obj = lambda *a, **k: None
    sys.modules["utils.render_utils"].is_obj_valid = lambda *a, **k: True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        pytest.skip("the exporter is not importable here: %s" % error)
    return module


@pytest.fixture(scope="module")
def scenes():
    spec = importlib.util.spec_from_file_location(
        "scene_layout", TRAINING / "scene_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def square_layout():
    return {"polygon": {"coordinates": [[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0],
                                         [0.0, 3.0], [0.0, 0.0]]]},
            "floor_z": 0.0, "ceiling_z": 2.6}


def test_corners_come_in_ceiling_floor_pairs(scenes):
    corners = scenes.manhattan_world(square_layout())
    assert corners.shape == (8, 3)
    assert (corners[0::2, 2] == 2.6).all()
    assert (corners[1::2, 2] == 0.0).all()
    assert np.allclose(corners[0::2, :2], corners[1::2, :2])


def test_a_pair_stands_in_one_column_after_projection(scenes):
    corners = scenes.manhattan_world(square_layout())
    pixels = np.column_stack((np.repeat([300.0, 100.0, 700.0, 500.0], 2),
                              np.tile([120.0, 400.0], 4)))
    ordered_pix, ordered_world = scenes.sorted_by_column(pixels, corners)
    assert (ordered_pix[0::2, 0] == ordered_pix[1::2, 0]).all()


def test_pairs_are_ordered_left_to_right(scenes):
    corners = scenes.manhattan_world(square_layout())
    pixels = np.column_stack((np.repeat([300.0, 100.0, 700.0, 500.0], 2),
                              np.tile([120.0, 400.0], 4)))
    ordered_pix, _ = scenes.sorted_by_column(pixels, corners)
    assert list(ordered_pix[0::2, 0]) == [100.0, 300.0, 500.0, 700.0]


def test_the_world_corners_follow_the_same_order(scenes):
    corners = scenes.manhattan_world(square_layout())
    pixels = np.column_stack((np.repeat([300.0, 100.0, 700.0, 500.0], 2),
                              np.tile([120.0, 400.0], 4)))
    ordered_pix, ordered_world = scenes.sorted_by_column(pixels, corners)
    assert np.allclose(ordered_world[0::2, :2], ordered_world[1::2, :2])
    assert np.allclose(ordered_world[0], corners[2])


def test_rounding_noise_never_splits_a_pair(scenes):
    """The two corners of a pair are one plan position, so one column."""
    noisy = np.array([[126.056000, 112.3], [126.055992, 431.1],
                      [ 83.200000, 120.0], [ 83.200008, 400.0]])
    snapped = scenes.share_one_column(noisy)
    assert (snapped[0::2, 0] == snapped[1::2, 0]).all()
    assert list(snapped[1::2, 0]) == [126.056, 83.2]
    assert list(snapped[:, 1]) == list(noisy[:, 1])


def test_a_pair_survives_both_snapping_and_ordering(scenes):
    corners = scenes.manhattan_world(square_layout())
    noisy = np.column_stack((np.array([300.0, 300.00001, 100.0, 99.99999,
                                       700.0, 700.00002, 500.0, 499.99998]),
                             np.tile([120.0, 400.0], 4)))
    ordered_pix, _ = scenes.sorted_by_column(noisy, corners)
    assert (ordered_pix[0::2, 0] == ordered_pix[1::2, 0]).all()
    assert list(ordered_pix[0::2, 0]) == [100.0, 300.0, 500.0, 700.0]


def test_a_camera_is_named_by_room_and_view_not_view_alone(naming):
    """The dump writes <house>/<name>, so two rooms of one house must not collide."""
    first = naming.camera_name("Bedroom-1", "0")
    second = naming.camera_name("Kitchen-2", "0")
    assert first != second
    assert "/" not in first and "/" not in second
    assert first == "Bedroom-1__0"
