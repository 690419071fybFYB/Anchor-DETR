"""
HIFF-DETR 训练脚本：在 AnchorDETR 原训练流程基础上，强制使用 HIFF 模型构建逻辑。
"""
from __future__ import annotations

import argparse

import main as anchor_main
from models import build_hiff


def get_args_parser() -> argparse.ArgumentParser:
    """
    继承 AnchorDETR 原参数，并额外添加 HIFF 相关超参。
    """
    parser = anchor_main.get_args_parser()
    parser.add_argument('--hiff_hidden_dim', default=512, type=int,
                        help='HIFF 模态专属 MLP 的隐藏层宽度')
    parser.add_argument('--hiff_temperature', default=1.0, type=float,
                        help='HIFF 通道 Softmax 的温度参数')
    return parser


def main(args):
    """
    运行 AnchorDETR 原训练流程，只是将构建函数替换为 HIFF 版本。
    """
    prev_builder = anchor_main.build_model
    anchor_main.build_model = build_hiff
    try:
        anchor_main.main(args)
    finally:
        anchor_main.build_model = prev_builder


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
