#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as rc


def run_inference(request: dict, config: str):
    raise NotImplementedError(
        "Running the model for request['input_path'] is not wired up "
        "(see runners/dpc_runner.py for a reference)."
    )


def to_scene_output(native) -> dict:
    objects = []
    for item in native.get("objects", []):
        box = item["bbox3d"]
        objects.append(
            {
                "label": item.get("label", "object"),
                "score": float(item.get("score", 1.0)),
                "centroid": box["centroid"],
                "basis": box["basis"],
                "coeffs": box["coeffs"],
            }
        )
    layout = native.get("layout")
    layout_corners = layout if isinstance(layout, list) else None
    layout_box = None
    if isinstance(layout, dict) and all(
        key in layout for key in ("centroid", "basis", "coeffs")
    ):
        layout_box = rc.oriented_box_to_axis_aligned(
            layout["centroid"], layout["basis"], layout["coeffs"]
        )
    return rc.scene_from_oriented_objects(
        objects, layout_corners=layout_corners, layout_box=layout_box
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="HoloPano runner")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/holopano.yaml")
    args, _unknown = parser.parse_known_args()

    request = rc.read_request(args.request)
    rc.seed_everything(request.get("seed"))
    rc.log(f"[holopano_runner] sample={request.get('sample_id')}")

    native = run_inference(request, args.config)
    scene = to_scene_output(native)
    rc.write_output(
        args.output,
        layout=scene["layout"],
        objects=scene["objects"],
        relations=scene["relations"],
        metadata={"model": "holopano", "sample_id": request.get("sample_id")},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
