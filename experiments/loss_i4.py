# -*- coding: utf-8 -*-
"""I4 损失优化：I4a 类别加权 BCE / I4b Focal Loss

原理：v8 检测损失 = box(IoU) + cls(BCE) + dfl 三项。本文件只替换分类项的计算，
box/DFL 及 TAL 标签分配全部沿用父类 v8DetectionLoss，保证只有损失形式一个变量。

- I4a 类别加权：BCE 前按 anchor 的真值类别乘权重（权重来自 train 集频率，
  1/sqrt(freq) 归一化），目标 = 提升 B1 体检的三个短板类（果-炭疽 R、叶-点斑 R、
  果-疮痂 P）。
- I4b Focal：BCE × (1-p_t)^gamma，gamma=2（原论文默认），降低易负样本主导。

注意：基于 ultralytics 8.3.x 源码结构编写（v8DetectionLoss 的属性名
self.no/nc/reg_max/preprocess/bbox_decode/assigner/bbox_loss/device、
BboxLoss 的 7 参 forward）。若升级 ultralytics 导致报错，diff 官方
ultralytics/utils/loss.py 的 v8DetectionLoss.__call__ 对齐即可，改动点只在
「分类项」那一小段（有注释标记）。
"""
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class WeightedDetectionLoss(v8DetectionLoss):
    """I4a：类别加权 BCE"""

    def __init__(self, model, class_weights):
        super().__init__(model)
        # (nc,) 每类权重，训练时搬到对应 device
        self.cw = torch.tensor(class_weights, dtype=torch.float32)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, N, nc)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                             batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_labels.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # ────────── 分类项（与父类唯一的差异：乘类别权重）──────────
        raw = self.bce(pred_scores, target_scores.to(dtype))            # (B, N, nc)
        soft = target_scores / target_scores.sum(-1, keepdim=True).clamp(min=1e-6)
        cls_w = (soft * self.cw.to(device=self.device, dtype=dtype)).sum(-1)  # (B, N)
        loss[1] = (raw * cls_w.unsqueeze(-1)).sum() / target_scores_sum
        # ────────────────────────────────────────────────────────────

        loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points,
                                          target_bboxes, target_scores,
                                          target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()


class FocalDetectionLoss(v8DetectionLoss):
    """I4b：Focal Loss（替换分类 BCE），gamma=2 可在 args 调"""

    def __init__(self, model, gamma=2.0, alpha=0.25):
        super().__init__(model)
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, N, nc)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                             batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_labels.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # ────────── 分类项（与父类唯一的差异：BCE → Focal）──────────
        p = pred_scores.sigmoid()
        pt = p * target_scores.to(dtype) + (1 - p) * (1 - target_scores.to(dtype))
        alpha_t = self.alpha * target_scores.to(dtype) + (1 - self.alpha) * (1 - target_scores.to(dtype))
        focal = self.bce(pred_scores, target_scores.to(dtype)) * alpha_t * (1 - pt) ** self.gamma
        loss[1] = focal.sum() / target_scores_sum
        # ────────────────────────────────────────────────────────────

        loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points,
                                          target_bboxes, target_scores,
                                          target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()


def _make_trainer(loss_cls, **loss_kwargs):
    """生成"用自定义损失"的 Trainer 类。

    做法：子类化 DetectionModel，覆盖 init_criterion()（模型训练前用它构造损失）；
    再子类化 DetectionTrainer 覆盖 get_model() 返回自定义模型。
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
    """I4a：返回用类别加权损失的 Trainer 类。class_weights 来自 common.count_class_freq()"""
    return _make_trainer(WeightedDetectionLoss, class_weights=class_weights)


def make_focal_trainer(gamma=2.0, alpha=0.25):
    """I4b：返回用 Focal 损失的 Trainer 类"""
    return _make_trainer(FocalDetectionLoss, gamma=gamma, alpha=alpha)
