#!/usr/bin/env python3
"""Train an 8-class Detectron2 Mask R-CNN on rendered MIDI panoramas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from gpu_preflight import check_gpu
from training_gate import require_training_approval


def experiment_classes(root):
    return list(json.loads(
        (root / "state" / "classes.json").read_text(encoding="utf-8"))["classes"])


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, ("bbox", "segm"), False, output_folder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--weights", required=True,
                        help="Local checkpoint, or a URL the checkpointer downloads")
    parser.add_argument(
        "--config",
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    )
    parser.add_argument("--max-iter", type=int, default=14000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--checkpoint-period", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unapproved-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_unapproved_smoke and args.max_iter > 20:
        raise ValueError("Unapproved smoke is limited to 20 iterations")
    require_training_approval(args.coco_root.parent, args.allow_unapproved_smoke)
    classes = experiment_classes(args.coco_root.parent)
    check_gpu(require_detectron=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_json = args.coco_root / "train.json"
    val_json = args.coco_root / "val.json"
    register_coco_instances("midi3d_train", {}, str(train_json), "")
    register_coco_instances("midi3d_val", {}, str(val_json), "")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.DATASETS.TRAIN = ("midi3d_train",)
    cfg.DATASETS.TEST = ("midi3d_val",)
    cfg.DATALOADER.NUM_WORKERS = args.workers
    weights = str(args.weights)
    cfg.MODEL.WEIGHTS = weights if "://" in weights else str(Path(weights).resolve())
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (
        int(args.max_iter * 0.7),
        int(args.max_iter * 0.9),
    )
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = str(args.output_dir.resolve())
    cfg.freeze()

    (args.output_dir / "classes.json").write_text(
        json.dumps(classes, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
