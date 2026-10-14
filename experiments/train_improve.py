# -*- coding: utf-8 -*-
"""阶段 3 改进实验（I1 通道压缩 / I2 C2f 精简 / I3 ECA 注意力 / I4 损失优化 / I5 ArcFace）

对照 train_baseline.py：除改进项外，训练配置与 B0 完全一致（100ep/640/batch16/seed42），
保证对比时"只动一个变量"。

用法（见 训练命令.md）：
    python train_improve.py --exp I1a        # 通道压缩 0.5x
    python train_improve.py --exp I2b        # C2f 精简 0.75x
    python train_improve.py --exp I3a        # ECA 插 backbone 三处
    python train_improve.py --exp I4a        # 类别加权损失（权重自动从 train 标签统计）
    python train_improve.py --exp I4b        # Focal 损失
    python train_improve.py --exp I5a        # ArcFace 度量学习分类损失 m=0.30（主创新）
    python train_improve.py --exp I5b        # ArcFace m=0.50（激进档）

实验设计：
    I1a/I1b/I1c  width 倍率 0.5/0.75/1.0（相对 v8n 的 0.25 → 绝对 0.125/0.1875/0.25）
                 注：I1c(1.0x) 与 B0 完全相同，可跳过直接复用 B0 结果
    I2a/I2b/I2c  depth 倍率 0.5/0.75/1.0（相对 v8n 的 0.33 → 绝对 0.165/0.2475/0.33）
    I3a/I3b/I3c  ECA 三种插入位置：backbone 三处 / SPPF 后 / neck 三处（cfgs/ 模板）
    I4a/I4b      类别加权 / Focal
    I5a/I5b      ArcFace 角度间隔 0.30 / 0.50（loss_i5.py，零推理开销）

产出：
    experiments/runs/improve_<exp>/   权重、曲线
    experiments/results.csv           统一记录（note 列写明改进项）
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import (HERE, PROJECT, V8N_WIDTH, V8N_DEPTH, build_cfg,
                    count_class_freq, evaluate_and_record, train_kwargs,
                    resolve_run_dir)
from modules.eca import register_eca
from loss_i4 import make_weighted_trainer, make_focal_trainer
from loss_i5 import make_arcface_trainer

# I1：通道压缩倍率（相对 v8n 的 0.25）
W_MULT = {"I1a": 0.50, "I1b": 0.75, "I1c": 1.00}
# I2：C2f 精简倍率（相对 v8n 的 0.33）
D_MULT = {"I2a": 0.50, "I2b": 0.75, "I2c": 1.00}
# I3：ECA 插入位置 → 结构模板（模板里 width/depth 已写死为 v8n 系数）
ECA_TPL = {
    "I3a": HERE / "cfgs" / "v8n_eca_backbone.yaml",
    "I3b": HERE / "cfgs" / "v8n_eca_sppf.yaml",
    "I3c": HERE / "cfgs" / "v8n_eca_neck.yaml",
}
NOTE = {
    "I1a": "改进-I1a 通道 0.5x", "I1b": "改进-I1b 通道 0.75x", "I1c": "改进-I1c 通道 1.0x(=B0)",
    "I2a": "改进-I2a C2f 0.5x", "I2b": "改进-I2b C2f 0.75x", "I2c": "改进-I2c C2f 1.0x(=B0)",
    "I3a": "改进-I3a ECA-backbone", "I3b": "改进-I3b ECA-SPPF", "I3c": "改进-I3c ECA-neck",
    "I4a": "改进-I4a 类别加权BCE", "I4b": "改进-I4b Focal(g=2,a=0.25)",
    "I5a": "改进-I5a ArcFace(m=0.30,s=16)", "I5b": "改进-I5b ArcFace(m=0.50,s=16)",
}


def parse_args():
    ap = argparse.ArgumentParser(description="阶段 3 改进实验")
    ap.add_argument("--exp", required=True,
                    choices=list(W_MULT) + list(D_MULT) + list(ECA_TPL)
                    + ["I4a", "I4b", "I5a", "I5b"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="显存不足降到 8")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0, help="Windows 稳妥默认 0，提速 --workers 2")
    return ap.parse_args()


def main():
    args = parse_args()
    exp = args.exp
    name = "improve_" + exp.lower()
    register_eca()  # 提前注册 ECA 模块（对 I1/I2/I4 无副作用）

    trainer_cls = None
    if exp in W_MULT:
        # I1：压缩 width。模板用官方 v8 结构，depth 保持 v8n 的 0.33
        w = V8N_WIDTH * W_MULT[exp]
        cfg = build_cfg(width_mult=w, depth_mult=V8N_DEPTH,
                        name=f"i1_{exp.lower()}_w{str(w).replace('.', '')}")
        print(f"[I1] width_multiple = {w:.4f}（v8n 的 {W_MULT[exp]:.2f}x）→ {cfg}")
        model = YOLO(str(cfg))
        model.load("yolov8n.pt")   # COCO 预训练迁移（形状匹配的层自动载入，与 B0 协议一致）
        model_label = f"yolov8n-w{W_MULT[exp]:.2f}"
    elif exp in D_MULT:
        # I2：精简 depth（C2f 重复次数按 round(n*depth) 取整）。width 保持 0.25
        dp = V8N_DEPTH * D_MULT[exp]
        cfg = build_cfg(width_mult=V8N_WIDTH, depth_mult=dp,
                        name=f"i2_{exp.lower()}_d{str(dp).replace('.', '')}")
        print(f"[I2] depth_multiple = {dp:.4f}（v8n 的 {D_MULT[exp]:.2f}x）→ {cfg}")
        model = YOLO(str(cfg))
        model.load("yolov8n.pt")
        model_label = f"yolov8n-d{D_MULT[exp]:.2f}"
    elif exp in ECA_TPL:
        # I3：ECA 插入位置对比（结构与 v8n 等参量级，多了 ECA 的少量参数）
        tpl = ECA_TPL[exp]
        print(f"[I3] 结构模板 = {tpl}")
        model = YOLO(str(tpl))
        model.load("yolov8n.pt")   # ECA 层是新增的（随机初始化），其余层从 COCO 权重迁移
        model_label = f"yolov8n-{exp.lower()}-eca"
    elif exp in ("I4a", "I4b"):
        # I4：标准 v8n 结构 + 自定义损失（结构、数据、超参与 B0 完全一致）
        model = YOLO("yolov8n.pt")
        model_label = f"yolov8n-{exp.lower()}-loss"
        if exp == "I4a":
            weights, freq = count_class_freq()
            trainer_cls = make_weighted_trainer(weights)
        else:
            trainer_cls = make_focal_trainer(gamma=2.0, alpha=0.25)
    else:
        # I5：ArcFace 度量学习分类损失（主创新，零推理开销）
        model = YOLO("yolov8n.pt")
        model_label = f"yolov8n-{exp.lower()}-arcface"
        margin = 0.30 if exp == "I5a" else 0.50
        trainer_cls = make_arcface_trainer(margin=margin, scale=16.0)

    kw = train_kwargs(args, name)
    if trainer_cls is not None:
        model.train(trainer=trainer_cls, **kw)   # I4：自定义损失走自定义 Trainer
    else:
        model.train(**kw)

    best = resolve_run_dir(name) / "weights" / "best.pt"
    evaluate_and_record(best, name, model_label, args.epochs, args.imgsz, args.batch,
                        NOTE[exp], device=args.device)


if __name__ == "__main__":
    main()
