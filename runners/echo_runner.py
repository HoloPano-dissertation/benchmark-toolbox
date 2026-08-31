#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as rc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="echo runner")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args, _unknown = parser.parse_known_args()

    requests = rc.read_requests(args.request)
    rc.log(f"[echo_runner] {len(requests)} sample(s)")

    def scene(request: dict) -> dict:
        return {
            "layout": rc.box([0.0, 0.0, 0.0], [4.0, 3.0, 5.0]),
            "objects": [
                rc.scene_object(
                    "obj-1", "chair", rc.box([1.0, 0.0, 1.0], [2.0, 1.0, 2.0]), 0.9
                )
            ],
            "relations": [],
            "metadata": {"model": "echo", "sample_id": request.get("sample_id")},
        }

    if len(requests) == 1:
        one = scene(requests[0])
        rc.write_output(
            args.output,
            layout=one["layout"],
            objects=one["objects"],
            metadata=one["metadata"],
        )
    else:
        rc.write_outputs(args.output, [scene(request) for request in requests])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
