# -*- coding: utf-8 -*-
"""阶段 5 对比实验（C1/C2）：主流轻量模型在统一协议下的训练

统一协议：同一数据集 / 100ep / 640 / batch16 / seed42（与 B0 完全一致）。

    C1 对比-v5nu  YOLOv5nu（ultralytics 重训版 v5，anchor-free，与 v8 同协议）
    C2 对比-v7tiny YOLOv7-tiny（需自备 yaml，见下）
    C3/C4/C5 复用 B0 / B2 / A5，不重跑（汇总在 eval_pareto.py 里统一评估）

公平性说明（论文要写清楚）：
    - C1 用 yolov5nu.pt（ultralytics 官方 COCO 预训练），与 B0 的 v8n.pt 预训练微调
      协议对齐；如果导师要求纯 v5 anchor-based 结构，改 --model yolov5n.yaml（从头训，
      需在论文注明初始化方式不同）。
    - C2 ultralytics 不内置 v7 结构，需先下载 yolov7-tiny.yaml 放到 cfgs/
      （社区移植版：github.com/wanghao9610/yolov7-ultralytics，取 cfg/training/yolov7-tiny.yaml），
      v7-tiny 无可用的 COCO 兼容权重，从头训，论文注明。

用法（见 训练命令.md）：
    python train_compare.py --exp C1
    python train_compare.py --exp C2
    python train_compare.py --exp C1 --model yolov5n.yaml   # 改用 anchor-based v5 从头训
"""
import argparse
from pathlib import Path

from common import HERE, PROJECT, evaluate_and_record, train_kwargs, resolve_run_dir

EXP_MODEL = {
    "C1": "yolov5nu.pt",     # ultralytics 官方 COCO 预训练（anchor-free，与 v8 同 head 协议）
    "C2": None,              # C2 的结构来自 --cfg（本地 yaml，从头训）
}
NOTE = {"C1": "对比-C1 YOLOv5nu(预训练)", "C2": "对比-C2 YOLOv7-tiny(从头训)"}


def parse_args():
    ap = argparse.ArgumentParser(description="阶段 5 对比实验训练")
    ap.add_argument("--exp", required=True, choices=["C1", "C2"])
    ap.add_argument("--model", default=None, help="覆盖默认权重/结构（如 yolov5n.yaml）")
    ap.add_argument("--cfg", default=None, help="C2 的结构 yaml（默认 cfgs/yolov7-tiny.yaml）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="显存不足降到 8")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    name = "compare_" + args.exp.lower()

    if args.exp == "C1":
        src = args.model or EXP_MODEL["C1"]
        model = YOLO(src)
    else:
        cfg = Path(args.cfg) if args.cfg else HERE / "cfgs" / "yolov7-tiny.yaml"
        if not cfg.exists():
            raise SystemExit(
                f"未找到 {cfg}\n"
                "C2 需要自备 yolov7-tiny.yaml（ultralytics 不内置 v7）：\n"
                "  下载 github.com/wanghao9610/yolov7-ultralytics 的 "
                "cfg/training/yolov7-tiny.yaml，存到 experiments/cfgs/ 后重跑\n"
                "  或用 --cfg 指定你自己的路径")
        model = YOLO(str(cfg))   # 从头训（无可用的 COCO 兼容权重），论文注明

    model.train(**train_kwargs(args, name))

    best = resolve_run_dir(name) / "weights" / "best.pt"
    label = Path(src).stem if args.exp == "C1" else "yolov7-tiny"
    evaluate_and_record(best, name, label, args.epochs, args.imgsz, args.batch,
                        NOTE[args.exp], device=args.device)


if __name__ == "__main__":
    main()
