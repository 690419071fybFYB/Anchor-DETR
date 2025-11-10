# ------------------------------------------------------------------------
# Copyright (c) 2024.
# ------------------------------------------------------------------------
"""
AnchorDETR variant with HIFF multi-scale feature fusion.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from util.misc import NestedTensor, nested_tensor_from_tensor_list

from .anchor_detr import SetCriterion, PostProcess
from .backbone import build_backbone
from .hiff import HIFF
from .matcher import build_matcher
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .transformer import build_transformer


class AnchorDETRWithHIFF(nn.Module):
    """
    AnchorDETR 前端插入 HIFF 模块：先对 backbone 输出的多尺度特征执行层级融合，再送入 Transformer。
    """

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        fusion_module: HIFF,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.fusion = fusion_module
        self.aux_loss = aux_loss
        hidden_dim = transformer.d_model
        self.num_feature_levels = len(backbone.strides)
        input_proj_list = []
        for idx in range(self.num_feature_levels):
            if idx == 0:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(self.fusion.embed_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            else:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(self.fusion.embed_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor | Tensor, class_frequency: Tensor | None = None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features = self.backbone(samples)
        feats = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            feats.append(src)
            masks.append(mask)
            assert mask is not None

        fused_feats = self.fusion(feats[:self.num_feature_levels], class_frequency=class_frequency)
        proj_feats = []
        for idx, feat in enumerate(fused_feats):
            proj_feats.append(self.input_proj[idx](feat).unsqueeze(1))
        srcs = torch.cat(proj_feats, dim=1)

        outputs_class, outputs_coord = self.transformer(srcs, masks)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


def build_hiff(args):
    """
    构建带 HIFF 融合的 AnchorDETR。
    """
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)
    if len(backbone.num_channels) < 3:
        raise ValueError("HIFF 版本要求 backbone 至少输出三层特征。")

    transformer = build_transformer(args)
    fusion = HIFF(
        in_channels=backbone.num_channels[:3],
        embed_dim=transformer.d_model,
        mlp_hidden_dim=getattr(args, "hiff_hidden_dim", 512),
        temperature=getattr(args, "hiff_temperature", 1.0),
    )

    model = AnchorDETRWithHIFF(
        backbone=backbone,
        transformer=transformer,
        fusion_module=fusion,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
