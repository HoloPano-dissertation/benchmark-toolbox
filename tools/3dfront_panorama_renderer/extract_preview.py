#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def colorize_instances(instance_map: np.ndarray) -> np.ndarray:
    ids = np.asarray(instance_map, dtype=np.uint64)
    red = (ids * 37 + 17) % 255
    green = (ids * 67 + 43) % 255
    blue = (ids * 97 + 89) % 255
    rgb = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    rgb[ids == 0] = 0
    return rgb


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    description: dict[str, object] = {}
    with h5py.File(args.hdf5, "r") as source:
        for key, dataset in source.items():
            description[key] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
            }
        if "colors" in source:
            save_rgb(source["colors"][()], args.output_dir / "colors.png")
        if "normals" in source:
            normals = np.clip(source["normals"][()], 0.0, 1.0)
            save_rgb((normals * 255).astype(np.uint8), args.output_dir / "normals.png")
        if "instance_segmaps" in source:
            instances = source["instance_segmaps"][()]
            save_rgb(colorize_instances(instances), args.output_dir / "instances.png")
        if "depth" in source:
            depth = np.asarray(source["depth"][()], dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0) & (depth < 1e9)
            visual = np.zeros(depth.shape, dtype=np.uint8)
            if np.any(valid):
                near, far = np.percentile(depth[valid], (1, 99))
                if far <= near:
                    far = near + 1.0
                normalized = 1.0 - np.clip((depth - near) / (far - near), 0.0, 1.0)
                visual[valid] = (normalized[valid] * 255).astype(np.uint8)
                description["depth_visualization_range"] = [float(near), float(far)]
            Image.fromarray(visual).save(args.output_dir / "depth.png")
        if "instance_attribute_maps" in source:
            raw = source["instance_attribute_maps"][()]
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            (args.output_dir / "instance_attribute_maps.json").write_text(
                text + ("" if text.endswith("\n") else "\n"), encoding="utf-8"
            )
    (args.output_dir / "datasets.json").write_text(
        json.dumps(description, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
