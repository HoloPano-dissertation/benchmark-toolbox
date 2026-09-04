#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--label-format", choices=("dense", "native"), default="dense")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    previews = root / "validation" / "horizonnet"
    previews.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        split_root = root / "horizonnet" / split
        if args.label_format == "native":
            from external.HorizonNet.dataset import PanoCorBonDataset
            dataset = PanoCorBonDataset(str(split_root), return_cor=True, return_path=True)
        else:
            dataset = sorted((split_root / "img").glob("*.png"))
            manifest = [json.loads(line) for line in (root / "manifests" / f"{split}.jsonl").read_text().splitlines() if line.strip()]
            expected = {str(r["sample_id"]).replace("/", "__") for r in manifest}
            actual = {p.stem for p in dataset}
            targets = {p.stem for p in (split_root / "label_dense").glob("*.npz")}
            if not expected or actual != expected or targets != expected:
                raise ValueError(f"{split}: images/targets do not exactly match the nonempty manifest")
        if not len(dataset):
            raise ValueError(f"Empty {split} split")
        count = 0
        selected = set(np.linspace(0, len(dataset) - 1, 4, dtype=int).tolist())
        for index in range(len(dataset)):
            if args.label_format == "native":
                image, boundary, corner_prob, corners, image_path = dataset[index]
                image, boundary, corner_prob = [t.numpy() for t in (image, boundary, corner_prob)]
            else:
                image_path = dataset[index]
                with Image.open(image_path) as source:
                    image = np.asarray(source.convert("RGB")).transpose(2, 0, 1)
                with np.load(split_root / "label_dense" / (image_path.stem+".npz"), allow_pickle=False) as target:
                    boundary, corner_prob = target["boundary"], target["corner"]
                    assert np.isfinite(target["ranges"]).all() and (target["ranges"] > 0).all(), image_path
                corners = []
            assert tuple(image.shape) == (3, 512, 1024), image_path
            assert tuple(boundary.shape) == (2, 1024), image_path
            assert tuple(corner_prob.shape) == (1, 1024), image_path
            assert all(np.isfinite(t).all() for t in (image, boundary, corner_prob)), image_path
            assert np.all(boundary[0] < 0) and np.all(boundary[1] > 0), image_path
            assert np.all(np.abs(boundary) < np.pi/2), image_path
            assert np.all((corner_prob >= 0) & (corner_prob <= 1)), image_path
            if index in selected:
                preview = Image.open(image_path).convert("RGB")
                draw = ImageDraw.Draw(preview)
                ys = (boundary / np.pi + 0.5) * 512 - 0.5
                for row in ys:
                    draw.line([(x, float(y)) for x, y in enumerate(row)], fill=(255, 210, 0), width=2)
                for x, y in corners:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 40, 40))
                preview.save(previews / f"{split}-{index:04d}.png")
            count += 1
        counts[split] = count
        print(split, count, args.label_format, "samples OK", flush=True)
    status = {"ready": True, "counts": counts,
              "label_format": args.label_format,
              "layout_labels": "floor-contour extrusion; not exact multilevel/ornamental ceiling geometry",
              "training_approved": False,
              "preview_dir": str(previews)}
    (root / "state" / "horizonnet_validation.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
