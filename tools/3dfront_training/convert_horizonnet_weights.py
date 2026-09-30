#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

PREFIX = "layout_estimation.horizon_net.module."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--no-rnn", action="store_true")
    return parser.parse_args()


def rename(state_dict, prefix):
    return {prefix + name: value for name, value in state_dict.items()}


def check_against(renamed, expected):
    missing = sorted(set(renamed) - set(expected))
    unfilled = sorted(set(expected) - set(renamed))
    if missing:
        raise ValueError(
            "%d weights have no place in the target model, so they would be dropped "
            "in silence: %s" % (len(missing), ", ".join(missing[:3])))
    if unfilled:
        raise ValueError(
            "%d weights of the target model stay untrained: %s"
            % (len(unfilled), ", ".join(unfilled[:3])))
    return len(renamed)


def expected_names(backbone, use_rnn, prefix):
    import torch.nn as nn
    from external.HorizonNet.model import HorizonNet

    wrapped = nn.DataParallel(HorizonNet(backbone, use_rnn))
    root = prefix[:-len("module.")] if prefix.endswith("module.") else prefix
    return [root + name for name in wrapped.state_dict()]


def main() -> None:
    args = parse_args()
    import torch

    state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    renamed = rename(state_dict, args.prefix)
    count = check_against(renamed, expected_names(
        args.backbone, not args.no_rnn, args.prefix))
    torch.save(renamed, args.target)
    print(json.dumps({"weights": count, "prefix": args.prefix,
                      "target": str(args.target)}, indent=2))


if __name__ == "__main__":
    main()
