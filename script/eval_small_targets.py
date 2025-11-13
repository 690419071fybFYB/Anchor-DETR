#!/usr/bin/env python3
"""
Evaluate AnchorDETR on the RSOD_small_cocoFormat dataset using the latest checkpoint.

This script searches data/detr-workdir/dul_recon_att for the most recent
checkpoint_*.pth file, then runs AnchorDETR/main.py in eval mode with that weight
and the small-object-only dataset located under /home/fyb/datasets/RSOD_small_cocoFormat.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.serialization


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/home/fyb/datasets/RSOD_medium_cocoFormat")
DEFAULT_WORKDIR = REPO_ROOT / "data" / "detr-workdir" / "dul_recon_att"
MAIN_PY = REPO_ROOT / "AnchorDETR" / "main.py"

# allowlist argparse.Namespace stored in checkpoints (Torch 2.6 weights_only=True)
torch.serialization.add_safe_globals([argparse.Namespace])


def find_latest_checkpoint(workdir: Path) -> Path:
    """Return the newest checkpoint file in workdir."""
    cand = list(workdir.glob("checkpoint_epoch_*.pth"))
    if not cand:
        fallback = workdir / "checkpoint.pth"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No checkpoint files found in {workdir}")
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AnchorDETR on RSOD small dataset")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET,
                        help="Root of the RSOD_small_cocoFormat dataset")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                        help="Directory containing AnchorDETR checkpoints")
    parser.add_argument("--eval-set", choices=["val", "test"], default="val",
                        help="Which split to evaluate (val uses val2017, test uses test2017)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Evaluation batch size")
    parser.add_argument("--device", default="cuda",
                        help="Device passed to main.py (cuda/cuda:0/cpu)")
    parser.add_argument("--num-query-pattern", type=int, default=None,
                        help="Override AnchorDETR --num_query_pattern (auto-read from checkpoint if omitted)")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER,
                        help="Additional arguments forwarded to AnchorDETR/main.py after '--'")
    return parser.parse_args()


def read_checkpoint_args(ckpt_path: Path) -> dict:
    """Load checkpoint metadata and return saved args if available."""
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to load checkpoint metadata ({exc}); continuing without it")
        return {}
    args = checkpoint.get("args")
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    return vars(args)


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    latest_ckpt = find_latest_checkpoint(args.workdir)

    ckpt_args = read_checkpoint_args(latest_ckpt)

    num_query_pattern = args.num_query_pattern
    if num_query_pattern is None:
        num_query_pattern = ckpt_args.get("num_query_pattern")

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
        "--output_dir", str(args.workdir / "eval" / f"small_{args.eval_set}")
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
