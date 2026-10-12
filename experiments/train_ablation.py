# -*- coding: utf-8 -*-
"""阶段 4 消融实验（A1-A5）：把阶段 3 的单项改进逐级组合，形成 V 型消融表

    A0 基线     = 复用 B0（baseline_yolov8n-2），不重跑
    A1 通道压缩 = I1 最优档
    A2 C2f 精简 = I2 最优档
    A3 压缩+精简 = A1 + A2 组合
    A4 +ECA     = A3 + I3 最优位置
    A5 +损失    = A4 + I4 最优配置   ← 最终模型（阶段 5 对比实验的"本文模型"）

用法（见 训练命令.md）：
    python train_ablation.py --preset A3
    python train_ablation.py --preset A4
    python train_ablation.py --preset A5

⚠️ 跑之前先看脚本里的 PRESETS：各档位默认值是占位，必须按阶段 3 的
   results.csv 结果改成"最优档"（比如 I1 最优是 I1b 就把 width 改 0.75）。

支持自由组合（不想用 preset 时）：
    python train_ablation.py --width 0.5 --depth 0.75 --eca neck --loss weighted
    （--width/--depth 传相对 v8n 的倍率，如 0.5 表示压缩到 0.5x）
"""
import argparse
from pathlib import Path

from common import (HERE, PROJECT, V8N_WIDTH, V8N_DEPTH, build_cfg,
                    count_class_freq, evaluate_and_record, train_kwargs)
from modules.eca import register_eca
from loss_i4 import make_weighted_trainer, make_focal_trainer

ECA_TPL = {
    "backbone": HERE / "cfgs" / "v8n_eca_backbone.yaml",
    "sppf": HERE / "cfgs" / "v8n_eca_sppf.yaml",
    "neck": HERE / "cfgs" / "v8n_eca_neck.yaml",
}

# ══════════════════════════════════════════════════════════════════
# 各消融项的档位（相对 v8n 的倍率）。
# ⚠️ 下面默认值按"I1a/I2a 是最优档"占位——阶段 3 跑完后，把每项改成
#    results.csv 里最优的那档再开跑！eca 取 backbone/sppf/neck，
#    loss 取 weighted/focal。
# ══════════════════════════════════════════════════════════════════
PRESETS = {
    "A1": dict(width=0.50),
    "A2": dict(depth=0.50),
    "A3": dict(width=0.50, depth=0.50),
    "A4": dict(width=0.50, depth=0.50, eca="neck"),
    "A5": dict(width=0.50, depth=0.50, eca="neck", loss="weighted"),
}


def parse_args():
    ap = argparse.ArgumentParser(description="阶段 4 消融实验")
    ap.add_argument("--preset", choices=list(PRESETS), help="A1-A5 组合档")
    # 自由组合覆盖项
    ap.add_argument("--width", type=float, help="通道倍率（相对 v8n 0.25），如 0.5")
    ap.add_argument("--depth", type=float, help="C2f 深度倍率（相对 v8n 0.33），如 0.5")
    ap.add_argument("--eca", choices=list(ECA_TPL), help="ECA 插入位置")
    ap.add_argument("--loss", choices=["weighted", "focal"], help="损失替换")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="显存不足降到 8")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    # preset 为底，命令行覆盖项优先
    cfg_opts = dict(PRESETS[args.preset]) if args.preset else {}
    if args.width is not None: cfg_opts["width"] = args.width
    if args.depth is not None: cfg_opts["depth"] = args.depth
    if args.eca is not None:   cfg_opts["eca"] = args.eca
    if args.loss is not None:  cfg_opts["loss"] = args.loss
    if not cfg_opts:
        ap.error("请指定 --preset 或至少一个组合项（--width/--depth/--eca/--loss）")
    args.opts = cfg_opts
    return args


def main():
    args = parse_args()
    o = args.opts
    register_eca()

    name = args.preset.lower() if args.preset else "ablation_custom"
    parts = []
    if o.get("width"):   parts.append(f"w{o['width']:g}x")
    if o.get("depth"):   parts.append(f"d{o['depth']:g}x")
    if o.get("eca"):     parts.append(f"eca-{o['eca']}")
    if o.get("loss"):    parts.append(o["loss"])
    note = f"消融-{(args.preset or 'custom')}: " + "+".join(parts)

    # ── 1. 结构配置：有无 ECA 决定用哪个模板（ECA 行不含通道参数，改倍率不影响结构）──
    has_compress = bool(o.get("width") or o.get("depth"))
    if o.get("eca"):
        tpl = ECA_TPL[o["eca"]]
        cfg = build_cfg(
            width_mult=V8N_WIDTH * o["width"] if o.get("width") else None,
            depth_mult=V8N_DEPTH * o["depth"] if o.get("depth") else None,
            template=tpl, name=f"{name}_cfg")
    elif has_compress:
        cfg = build_cfg(
            width_mult=V8N_WIDTH * o["width"] if o.get("width") else None,
            depth_mult=V8N_DEPTH * o["depth"] if o.get("depth") else None,
            name=f"{name}_cfg")
    else:
        cfg = None   # 纯损失消融（如 A5 不含压缩时），直接用预训练 v8n

    print(f"[消融] 配置: {note}")
    if cfg:
        print(f"[消融] 结构模板 → {cfg}")

    # ── 2. 训练 ──
    kw = train_kwargs(args, name)
    if o.get("loss"):
        if o["loss"] == "weighted":
            weights_per_cls, _ = count_class_freq()
            trainer_cls = make_weighted_trainer(weights_per_cls)
        else:
            trainer_cls = make_focal_trainer(gamma=2.0, alpha=0.25)
        model = YOLO("yolov8n.pt") if cfg is None else YOLO(str(cfg))
        if cfg is not None:
            model.load("yolov8n.pt")   # 同上：结构变更时按形状迁移 COCO 预训练
        model.train(trainer=trainer_cls, **kw)
    else:
        model = YOLO(str(cfg)) if cfg is not None else YOLO("yolov8n.pt")
        if cfg is not None:
            model.load("yolov8n.pt")
            # COCO 预训练迁移：形状匹配的层自动载入（通道被压缩的层自动跳过、随机初始化），
            # 与 B0（yolov8n.pt 预训练微调）协议一致，保证消融对比公平
        model.train(**kw)

    best = Path(PROJECT) / name / "weights" / "best.pt"
    model_label = "yolov8n-" + "-".join(parts)
    evaluate_and_record(best, name, model_label, args.epochs, args.imgsz, args.batch,
                        note, device=args.device)


if __name__ == "__main__":
    main()
