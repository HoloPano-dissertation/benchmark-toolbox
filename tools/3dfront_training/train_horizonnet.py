#!/usr/bin/env python3
"""Train the vendored HorizonNet using package imports and a held-out val split.

Requires the Pano3D repository on PYTHONPATH. The encoder uses ImageNet weights;
no iGibson checkpoint is loaded. Checkpoints use HorizonNet's native format.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from external.HorizonNet.dataset import PanoCorBonDataset
from external.HorizonNet.model import HorizonNet
from gpu_preflight import check_gpu
from training_gate import require_training_approval
from horizon_dense_dataset import DenseLayoutDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--label-format", choices=("dense", "native"), default="dense")
    parser.add_argument("--allow-unapproved-smoke", action="store_true")
    return parser.parse_args()


def save_checkpoint(path, net, optimizer, epoch, val_loss, args):
    state = {
        "kwargs": {"backbone": net.backbone, "use_rnn": net.use_rnn},
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "initialization": "ImageNet ResNet50; random HorizonNet decoder",
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, str(temporary))
    temporary.replace(path)


def epoch_pass(net, loader, device, optimizer=None, max_batches=None):
    train = optimizer is not None
    net.train(train)
    count = 0
    sums = {"boundary_loss": 0.0, "corner_loss": 0.0, "loss": 0.0}
    with torch.set_grad_enabled(train):
        for index, (image, boundary, corners) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            image, boundary, corners = [x.to(device) for x in (image, boundary, corners)]
            pred_boundary, pred_corners = net(image)
            boundary_loss = F.l1_loss(pred_boundary, boundary)
            corner_loss = F.binary_cross_entropy_with_logits(pred_corners, corners)
            loss = boundary_loss + corner_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite HorizonNet loss")
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type=float("inf"))
                optimizer.step()
            batch_size = image.shape[0]
            count += batch_size
            for name, value in (("boundary_loss", boundary_loss),
                                ("corner_loss", corner_loss), ("loss", loss)):
                sums[name] += float(value.detach().cpu()) * batch_size
            if index % 50 == 0:
                print(json.dumps({"phase": "train" if train else "val",
                                  "batch": index, "loss": float(loss.detach().cpu())}), flush=True)
    if count == 0:
        raise ValueError("No samples processed")
    return {**{name: value / count for name, value in sums.items()}, "samples": count}


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.allow_unapproved_smoke and not (args.epochs == 1 and args.max_train_batches
                                          and args.max_train_batches <= 4 and args.max_val_batches
                                          and args.max_val_batches <= 2):
        raise ValueError("Unapproved smoke must be bounded to 1 epoch, 4 train / 2 val batches")
    require_training_approval(args.dataset_root.parent, args.allow_unapproved_smoke)
    check_gpu()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a fresh training output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.label_format == "dense":
        train_set = DenseLayoutDataset(args.dataset_root / "train", augment=True)
        val_set = DenseLayoutDataset(args.dataset_root / "val")
    else:
        train_set = PanoCorBonDataset(str(args.dataset_root / "train"),
                                     flip=True, rotate=True, gamma=True, stretch=True)
        val_set = PanoCorBonDataset(str(args.dataset_root / "val"))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda")
    net = HorizonNet("resnet50", True).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = epoch_pass(net, train_loader, device, optimizer, args.max_train_batches)
        val = epoch_pass(net, val_loader, device, max_batches=args.max_val_batches)
        scheduler.step(val["loss"])
        summary = {"epoch": epoch, "train": train, "val": val}
        print(json.dumps(summary), flush=True)
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        save_checkpoint(args.output_dir / "last.pth", net, optimizer, epoch, val["loss"], args)
        if val["loss"] < best:
            best = val["loss"]
            save_checkpoint(args.output_dir / "best_valid.pth", net, optimizer, epoch, best, args)


if __name__ == "__main__":
    main()
