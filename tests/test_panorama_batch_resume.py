import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

RENDERER = Path(__file__).resolve().parents[1] / "tools" / "3dfront_panorama_renderer"


@pytest.fixture(scope="module")
def run_batch():
    spec = importlib.util.spec_from_file_location("run_batch", RENDERER / "run_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def implementation_hashes():
    return {name: hashlib.sha256((RENDERER / name).read_bytes()).hexdigest()
            for name in ("render.py", "camera_policy.py", "room_layout.py", "glb_geometry.py")}


def expectation():
    return {"implementation_sha256": implementation_hashes(), "samples": 32,
            "requested_min_clearance": 0.1, "camera_height_fraction": 0.6}


def finished_room(root, views=4, **overrides):
    metadata = {"implementation_sha256": implementation_hashes(), "samples": 32,
                "views": views, "requested_min_clearance": 0.1,
                "camera_height_fraction": 0.6, "plan_only": False}
    metadata.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "render.json").write_text(json.dumps(metadata), encoding="utf-8")
    for index in range(views):
        (root / f"{index}.hdf5").write_bytes(b"frame")
    (root / ".complete").write_text("done\n", encoding="utf-8")
    return root


def test_a_finished_room_is_kept(run_batch, tmp_path):
    room = finished_room(tmp_path / "room")
    assert run_batch.resume_state(room, expectation(), 4)[0] == "keep"


def test_a_marker_without_metadata_is_rendered_again(run_batch, tmp_path):
    room = finished_room(tmp_path / "room")
    (room / "render.json").write_text("", encoding="utf-8")
    assert run_batch.resume_state(room, expectation(), 4)[0] == "redo"
    (room / "render.json").unlink()
    assert run_batch.resume_state(room, expectation(), 4)[0] == "redo"


def test_other_settings_are_reported_as_stale(run_batch, tmp_path):
    other = finished_room(tmp_path / "other", samples=64)
    assert run_batch.resume_state(other, expectation(), 4)[0] == "stale"


def test_a_missing_or_empty_frame_is_rendered_again(run_batch, tmp_path):
    """A node that died mid-write left frames of zero length; they are not views."""
    short = finished_room(tmp_path / "short")
    (short / "3.hdf5").unlink()
    assert run_batch.resume_state(short, expectation(), 4)[0] == "redo"
    empty = finished_room(tmp_path / "empty")
    (empty / "2.hdf5").write_bytes(b"")
    assert run_batch.resume_state(empty, expectation(), 4)[0] == "redo"


def test_the_leftovers_of_an_unfinished_room_are_cleared(run_batch, tmp_path):
    """The renderer refuses a directory that is not empty, so redoing means clearing."""
    room = finished_room(tmp_path / "room")
    (room / "render.json").write_text("", encoding="utf-8")
    run_batch.discard_room(room)
    assert not room.exists()
    run_batch.discard_room(room)


def test_leftovers_without_any_marker_are_cleared_too(run_batch, tmp_path):
    """A room that never got its marker can still hold files from a killed run."""
    room = finished_room(tmp_path / "room")
    (room / ".complete").unlink()
    (room / "1.hdf5").write_bytes(b"")
    run_batch.discard_room(room)
    assert not room.exists()


def test_the_reason_recorded_is_the_error_the_renderer_raised(run_batch):
    trace = """Traceback (most recent call last):
  File "render.py", line 152, in choose_cameras
    raise ValueError(
ValueError: Only 0 distinct poses satisfy clearance 0.03; requested 4.
Error: script failed, file: 'render.py', exiting.
"""
    assert run_batch.failure_reason(trace).startswith("ValueError: Only 0 distinct poses")
    assert run_batch.failure_reason("") == ""
    assert run_batch.failure_reason("Blender quit\n") == ""


def test_a_temperature_range_is_drawn_from_the_room_name(tmp_path):
    """Same room, same lamps: a set must not change colour when a room is redone."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "render_helpers", Path(__file__).resolve().parents[1]
        / "tools/3dfront_panorama_renderer/run_batch.py")
    source = (Path(__file__).resolve().parents[1]
              / "tools/3dfront_panorama_renderer/render.py").read_text(encoding="utf-8")
    start = source.index("def room_temperature(")
    end = source.index("def blackbody(")
    namespace = {"hashlib": __import__("hashlib")}
    exec(compile(source[start:end], "render.py", "exec"), namespace)
    room_temperature = namespace["room_temperature"]
    assert room_temperature("0", "Bedroom-1") == 0.0
    assert room_temperature("3400", "Bedroom-1") == 3400.0
    first = room_temperature("2700-5000", "Bedroom-1")
    assert first == room_temperature("2700-5000", "Bedroom-1")
    assert 2700 <= first <= 5000
    assert first != room_temperature("2700-5000", "Bedroom-2")
