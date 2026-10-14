# -*- coding: utf-8 -*-
"""I5 主创新：ArcFace 度量学习分类损失（零推理参数/零推理开销，适配 ultralytics 8.4.x）

动机：B0/I4 的分类损失是逐类独立 BCE，类别间在特征空间"不分家"，
混淆矩阵上表现为易混对（炭疽↔褐斑等）互混严重。ArcFace 通过
「归一化 + 角度间隔」迫使同类特征聚拢、异类推开，在特征层面拉开决策边界。

实现（ultralytics 8.4.154 的侵入点比 8.3 更干净：官方分类段已收敛为
bce_loss = self.bce(pred_scores, target_scores) 一行。本类只做两件事）：

1. self.bce → ArcFaceBCE 可调用对象：
   - 特征侧归一化：logits 沿类别维 L2 归一化 → cosθ（等效固定特征范数）；
   - 角度间隔：目标类 cosθ → cos(θ+m)，用恒等式展开（数值稳定、无 acos）：
     cos(θ+m) = cosθ·cos m − sinθ·sin m，sinθ = sqrt(1−cos²θ)；
   - 只对「该 anchor 的目标类」位置施加 margin（target_scores > 0 处），
     其余位置保持 cosθ；
   - 乘 scale 放大回 logit 量级，仍用 BCEWithLogits 监督（兼容 TAL 软标签）。
2. 权重侧归一化：重写 loss()，每步训练前对检测头 cv3 各尺度最后 1×1 conv
   的 weight 沿输入通道维 L2 归一化、bias 置零——分类权重约束在单位球上。

两处归一化合起来 logits = scale·cosθ（特征-权重夹角余弦）。
所有改动只在训练期生效，推理图与普通 v8 完全一致（conv 形状不变），
参数量/推理延迟与基线持平——这是论文的核心卖点。

超参：
    margin  m   角度间隔（弧度）。I5a=0.30（温和）/ I5b=0.50（激进）
    scale   s   余弦放大倍数，默认 16
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss

from loss_i4 import _make_trainer


class ArcFaceBCE:
    """ArcFace 化的 BCE：输入 raw logits (B,N,nc) + 软标签 (B,N,nc)，
    输出与 nn.BCEWithLogitsLoss(reduction='none') 同形状的逐元素损失。
    直接替换 v8DetectionLoss.self.bce，官方分类段其余逻辑零改动。"""

    def __init__(self, margin=0.30, scale=16.0):
        self.margin = float(margin)
        self.scale = float(scale)
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, logits, target):
        t = target.to(logits.dtype)
        # 1) 特征侧归一化：logits/‖logits‖ → cosθ ∈ [-1,1]
        cos = F.normalize(logits, dim=-1, eps=1e-6).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        # 2) ArcFace：目标类 cosθ → cos(θ+m)（恒等式展开，无 acos，AMP 安全）
        cos_m = cos * self._cos_m - (1.0 - cos * cos).clamp(min=0.0).sqrt() * self._sin_m
        # 3) 只在「目标类」位置施加 margin，其余保持原余弦
        cos_adj = torch.where(t > 0, cos_m, cos)
        # 4) 放大回 logit 量级 + BCE 监督
        return self._bce(self.scale * cos_adj, t)


def normalize_head_weights(model):
    """对检测头 cv3 各尺度最后 1×1 conv 做权重归一化（in-place，每步训练前调用）。

    v8 的 cv3 是 ModuleList[Sequential(...→Conv2d(nc))]，找每分支
    out_channels == nc 的最后一个 Conv2d：weight 沿 in_channels 维归一化，
    bias 清零（余弦相似度不带偏置）。等效把分类权重约束在单位球上。
    """
    head = model.model[-1]          # Detect 层
    for branch in head.cv3:
        last = None
        for m in branch.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == head.nc:
                last = m
        if last is None:
            continue
        with torch.no_grad():
            w = last.weight                              # (nc, C_in, 1, 1)
            norm = w.norm(dim=1, keepdim=True).clamp(min=1e-6)
            w.div_(norm)
            if last.bias is not None:
                last.bias.zero_()


class ArcFaceDetectionLoss(v8DetectionLoss):
    """I5 损失类：标签分配/box/DFL 全部官方实现，仅换分类度量 + 权重归一化"""

    def __init__(self, model, margin=0.30, scale=16.0, weight_norm=True):
        super().__init__(model)
        self.net = model            # _CustomModel 实例（权重归一化要用）
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight_norm = bool(weight_norm)
        self.bce = ArcFaceBCE(margin=self.margin, scale=self.scale)
        print(f"[I5] ArcFace 分类损失: margin={self.margin}, scale={self.scale}, "
              f"weight_norm={self.weight_norm}（零推理开销）")

    def loss(self, preds, batch):
        """8.4 的 loss 入口：先做权重归一化，其余全部走官方实现"""
        if self.weight_norm:
            normalize_head_weights(self.net)
        return super().loss(preds, batch)


def make_arcface_trainer(margin=0.30, scale=16.0):
    """I5：返回用 ArcFace 损失的 Trainer 类"""
    return _make_trainer(ArcFaceDetectionLoss, margin=margin, scale=scale)
