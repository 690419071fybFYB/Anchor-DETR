#!/usr/bin/env python3
"""
Evaluate AnchorDETR checkpoints from data/detr-workdir/r50-dc5 on the
RSOD_small_cocoFormat dataset.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.serialization


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/home/fyb/datasets/RSOD_large_cocoFormat")
DEFAULT_WORKDIR = REPO_ROOT / "data" / "detr-workdir" / "r50-dc5"
MAIN_PY = REPO_ROOT / "AnchorDETR" / "main.py"

# allow checkpoints that store argparse.Namespace objects
torch.serialization.add_safe_globals([argparse.Namespace])


def find_latest_checkpoint(workdir: Path) -> Path:
    checkpoints = sorted(workdir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint*.pth files under {workdir}")
    return checkpoints[0]


def read_checkpoint_args(ckpt_path: Path) -> dict:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: could not load checkpoint metadata ({exc}); continuing without saved args")
        return {}
    args = ckpt.get("args")
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    return vars(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AnchorDETR on RSOD small-object subset")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET,
                        help="Root of RSOD_small_cocoFormat dataset")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                        help="Directory containing checkpoints (default: r50-dc5)")
    parser.add_argument("--eval-set", choices=["val", "test"], default="val",
                        help="Which split to evaluate")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Evaluation batch size")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (cuda/cpu etc.)")
    parser.add_argument("--num-query-pattern", type=int, default=None,
                        help="Override --num_query_pattern; auto-filled from checkpoint when omitted")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER,
                        help="Any additional args appended after '--'")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} not found")

    latest_ckpt = find_latest_checkpoint(args.workdir)
    ckpt_args = read_checkpoint_args(latest_ckpt)

    num_query_pattern = args.num_query_pattern
    if num_query_pattern is None:
        num_query_pattern = ckpt_args.get("num_query_pattern")

    output_dir = args.workdir / "eval" / f"rsod_small_{args.eval_set}"

    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--eval",
        "--dataset_file", "coco",
        "--coco_path", str(dataset_root),
        "--eval_set", args.eval_set,
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--resume", str(latest_ckpt),
        "--output_dir", str(output_dir),
    ]

    if num_query_pattern is not None:
        cmd.extend(["--num_query_pattern", str(num_query_pattern)])

    if args.extra_args:
        cmd.append("--")
        cmd.extend(args.extra_args)

    print("Running command:\n  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
