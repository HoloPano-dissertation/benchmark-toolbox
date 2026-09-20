import importlib.util
from pathlib import Path

import numpy as np
import pytest

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"


@pytest.fixture(scope="module")
def architecture():
    spec = importlib.util.spec_from_file_location(
        "scene_architecture", DATASET / "scene_architecture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_room_is_placed_by_the_furniture_it_shares(architecture, tmp_path, monkeypatch):
    """Scene metres become room units: centred, scaled, and turned from Y-up to Z-up."""
    place = {"scale": 2.0, "cx": 10.0, "cz": -4.0, "cy": 1.0}
    points = np.array([[10.0, 1.0, -4.0],      # the origin of the room frame
                       [12.0, 3.0, -2.0]])
    mapped = architecture.to_room(points, place)
    assert np.allclose(mapped[0], [0.0, 0.0, 0.0])
    assert np.allclose(mapped[1], [1.0, -1.0, 1.0])


def test_a_dropped_ceiling_keeps_only_what_faces_the_room(architecture):
    """Its box sides enclose a cavity no light reaches, and render black."""
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    normals = ([0, -1, 0]*3) + ([0, 1, 0]*3) + ([1, 0, 0]*3)
    kept = architecture.downward_faces(faces, normals, 9)
    assert kept.tolist() == [[0, 1, 2]]


def test_faces_are_left_alone_when_the_scene_records_no_normals(architecture):
    faces = np.array([[0, 1, 2]])
    assert architecture.downward_faces(faces, None, 3).tolist() == faces.tolist()
    assert architecture.downward_faces(faces, [0, 1], 3).tolist() == faces.tolist()


def scene_with(meshes, materials=()):
    return {"mesh": list(meshes), "material": list(materials),
            "scene": {"room": [{"instanceid": "Bedroom-1",
                                "children": [{"ref": m["uid"]} for m in meshes]}]}}


def surface(uid, kind, height=1.0, with_uv=True):
    xyz = [0, 0, 0, 1, 0, 0, 1, height, 0]
    mesh = {"uid": uid, "type": kind, "xyz": xyz, "faces": [0, 1, 2], "material": "mat"}
    if with_uv:
        mesh["uv"] = [0, 0, 1, 0, 1, 1]
    return mesh


def test_texture_coordinates_keep_their_own_numbering(architecture, tmp_path):
    """A surface without them must not shift the ones that follow onto wrong images."""
    meshes = [surface("a", "WallInner"), surface("b", "Floor", with_uv=False),
              surface("c", "Ceiling")]
    scene = scene_with(meshes)
    place = {"scale": 1.0, "cx": 0.0, "cz": 0.0, "cy": 0.0}
    kept = architecture.write_room(scene, scene["scene"]["room"][0], place, {}, None, {},
                                   {"WallInner", "Floor", "Ceiling"}, tmp_path)
    assert kept == {"WallInner": 1, "Floor": 1, "Ceiling": 1}
    lines = (tmp_path / "architecture.obj").read_text().splitlines()
    faces = [line for line in lines if line.startswith("f ")]
    assert faces[0] == "f 1/1 2/2 3/3"
    assert faces[1] == "f 4 5 6"
    assert faces[2] == "f 7/4 8/5 9/6"


def test_every_surface_becomes_its_own_object(architecture, tmp_path):
    scene = scene_with([surface("a", "WallInner"), surface("b", "WallInner")])
    place = {"scale": 1.0, "cx": 0.0, "cz": 0.0, "cy": 0.0}
    architecture.write_room(scene, scene["scene"]["room"][0], place, {}, None, {},
                            {"WallInner"}, tmp_path)
    names = [line[2:] for line in (tmp_path / "architecture.obj").read_text().splitlines()
             if line.startswith("o ")]
    assert names == ["WallInner_1", "WallInner_2"]


def test_a_room_the_scene_does_not_hold_is_named_plainly(architecture):
    with pytest.raises(KeyError, match="Kitchen-9"):
        architecture.room_of(scene_with([surface("a", "WallInner")]), "Kitchen-9")


def test_surfaces_are_wound_to_face_the_room(architecture):
    """A ceiling whose normal points into the slab renders black; it must be flipped."""
    import numpy as np
    centre = np.zeros(3)
    ceiling = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]])
    faces = np.array([[0, 1, 2]])                      # normal points up, away from us
    kept = architecture.face_the_room(ceiling, faces, centre)
    corners = ceiling[kept[0]]
    normal = np.cross(corners[1]-corners[0], corners[2]-corners[0])
    assert normal[2] < 0                               # now it looks down into the room

    floor = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0]])
    kept = architecture.face_the_room(floor, np.array([[0, 2, 1]]), centre)
    corners = floor[kept[0]]
    assert np.cross(corners[1]-corners[0], corners[2]-corners[0])[2] > 0

    wall = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]])
    kept = architecture.face_the_room(wall, np.array([[0, 1, 2]]), centre)
    corners = wall[kept[0]]
    assert np.cross(corners[1]-corners[0], corners[2]-corners[0])[0] < 0


def test_one_ceiling_keeps_each_plane(architecture):
    """Two ceilings in one plane shadow each other and both come out black."""
    import numpy as np
    big = np.array([[-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [2.0, 2.0, 1.0]])
    small = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])
    face = np.array([[0, 1, 2]])
    thinned = architecture.thin_coincident_ceilings(
        [("Ceiling", big, face, {}), ("CustomizedCeiling", small, face, {})])
    assert len(thinned[0][2]) == 1        # the larger one keeps the plane
    assert len(thinned[1][2]) == 0


def test_only_the_faces_sharing_a_plane_are_given_up(architecture):
    """A dropped ceiling is a box: its rim shares the slab, its panel does not."""
    import numpy as np
    slab = np.array([[-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [2.0, 2.0, 1.0]])
    box = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0],
                    [-1.0, -1.0, 0.7], [1.0, -1.0, 0.7], [1.0, 1.0, 0.7]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])        # rim at the slab, panel below it
    thinned = architecture.thin_coincident_ceilings(
        [("Ceiling", slab, np.array([[0, 1, 2]]), {}), ("CustomizedCeiling", box, faces, {})])
    assert thinned[0][2].tolist() == [[0, 1, 2]]
    assert thinned[1][2].tolist() == [[3, 4, 5]]    # the panel survives, the rim goes


def test_a_wall_is_never_thinned(architecture):
    import numpy as np
    wall = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]])
    face = np.array([[0, 1, 2]])
    thinned = architecture.thin_coincident_ceilings([("WallInner", wall, face, {})])
    assert thinned[0][2].tolist() == face.tolist()
