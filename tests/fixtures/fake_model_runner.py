import argparse
import json
import os
import uuid
from pathlib import Path

LAUNCH_ID = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    batch = isinstance(payload, dict) and "samples" in payload
    requests = payload["samples"] if batch else [payload]

    def scene(request):
        return {
            "layout": {
                "min_corner": [0.0, 0.0, 0.0],
                "max_corner": [1.0, 1.0, 1.0],
            },
            "objects": [],
            "relations": [],
            "metadata": {"sample_id": request["sample_id"], "launch": LAUNCH_ID},
        }

    outputs = [scene(request) for request in requests]
    result = {"outputs": outputs} if batch else outputs[0]
    Path(args.output).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
