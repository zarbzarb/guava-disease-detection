# -*- coding: utf-8 -*-
"""I4 损失优化：I4a 类别加权 BCE / I4b Focal Loss（适配 ultralytics 8.4.x）

原理：v8 检测损失 = box(IoU) + cls(BCE) + dfl 三项。本文件只替换分类项的计算，
box/DFL 及 TAL 标签分配全部沿用官方实现，保证只有损失形式一个变量。

适配说明（ultralytics 8.4.154 实测源码）：
    8.4 的分类段收敛为 bce_loss = self.bce(pred_scores, target_scores) 一行，
    后接 sum()/target_scores_sum。因此：
    - I4a 类别加权：直接用 8.4 官方内建钩子 model.class_weights
      （v8DetectionLoss.__init__ 会 getattr(model, 'class_weights') 并在
      分类段自动相乘）——不需要自定义损失类，只需要自定义 Model 属性；
    - I4b Focal：子类里把 self.bce 换成 FocalBCE 可调用对象（输入 raw logits +
      软标签，输出同形状逐元素损失，reduction='none'），官方分类段其余逻辑零改动。

旧 8.3.x 的 WeightedDetectionLoss/FocalDetectionLoss 子类已废弃移除
（8.4 的 __call__/bbox_decode/bbox_loss 签名都变了，旧写法会崩）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class FocalBCE:
    """I4b：Focal Loss 可调用对象，替换 v8DetectionLoss 的 self.bce。

    输入 raw logits (B, N, nc) + 软标签 target_scores (B, N, nc)，
    输出与 nn.BCEWithLogitsLoss(reduction='none') 同形状的逐元素损失。
    focal = BCE × alpha_t × (1-p_t)^gamma，gamma=2 / alpha=0.25（原论文默认）。
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, logits, target):
        t = target.to(logits.dtype)
        bce = self._bce(logits, t)
        p = logits.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        return bce * alpha_t * (1 - pt).pow(self.gamma)


class FocalDetectionLoss(v8DetectionLoss):
    """I4b 损失类：结构/标签分配与官方一致，仅 self.bce → FocalBCE"""

    def __init__(self, model, gamma=2.0, alpha=0.25):
        super().__init__(model)
        self.bce = FocalBCE(gamma=gamma, alpha=alpha)


def _make_trainer(loss_cls, **loss_kwargs):
    """生成"用自定义损失"的 Trainer 类（8.4 机制不变：init_criterion 惰性构造）。

    做法：子类化 DetectionModel 覆盖 init_criterion()；
    再子类化 DetectionTrainer 覆盖 get_model()（8.4 签名与 8.3 一致，已实测确认）。
    """

    class _CustomModel(DetectionModel):
        def __init__(self, cfg=None, **kw):
            super().__init__(cfg, **kw)
            self._loss_kwargs = loss_kwargs

        def init_criterion(self):
            return loss_cls(self, **self._loss_kwargs)

    class CustomTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = _CustomModel(cfg, nc=self.data["nc"], verbose=verbose)
            if weights:
                model.load(weights)
            return model

    return CustomTrainer


def make_weighted_trainer(class_weights):
    """I4a：官方 class_weights 钩子（8.4 内建支持，v8DetectionLoss 自动读取并相乘）。

    class_weights 来自 common.count_class_freq()，shape (nc,)。
    """
    weights = torch.tensor(class_weights, dtype=torch.float32)

    class _WeightedModel(DetectionModel):
        def __init__(self, cfg=None, **kw):
            super().__init__(cfg, **kw)
            self.class_weights = weights   # v8DetectionLoss.__init__ getattr(model,'class_weights')

    class WeightedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights_=None, verbose=True):
            model = _WeightedModel(cfg, nc=self.data["nc"], verbose=verbose)
            if weights_:
                model.load(weights_)
            return model

    return WeightedTrainer


def make_focal_trainer(gamma=2.0, alpha=0.25):
    """I4b：返回用 Focal 损失的 Trainer 类"""
    return _make_trainer(FocalDetectionLoss, gamma=gamma, alpha=alpha)
