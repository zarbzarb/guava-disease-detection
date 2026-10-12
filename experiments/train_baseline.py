# -*- coding: utf-8 -*-
"""
train_baseline.py — 基线实验：YOLOv8n/s/m 在番石榴叶果数据集（8 类）上的训练与评估

用法（见 训练命令.md）：
    python train_baseline.py                          # 默认 yolov8n，100 轮
    python train_baseline.py --model yolov8s.pt       # s 尺寸对比
    python train_baseline.py --epochs 150 --batch 8   # 改轮数/批大小

产出：
    experiments/runs/baseline_yolov8n/     训练日志、results.csv、weights/best.pt
    experiments/results.csv                统一实验记录（追加写，论文表格从这里生成）
"""
import os
# 必须在 import torch/ultralytics 之前设置：解决 Windows conda 下 OpenMP 运行时冲突（OMP Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
from datetime import datetime

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset_guava_leaf_fruit", "data.yaml")
PROJECT = os.path.join(HERE, "runs")
RESULTS_CSV = os.path.join(HERE, "results.csv")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 基线训练（番石榴叶果 8 类）")
    ap.add_argument("--model", default="yolov8n.pt",
                    choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                    help="预训练权重（决定模型尺寸）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="显存不足降到 8")
    ap.add_argument("--device", default="0", help="0=GPU，cpu=无显卡")
    ap.add_argument("--workers", type=int, default=0,
                    help="数据加载进程数。Windows 下多进程偶发崩溃，默认 0=主进程加载（稳，稍慢）；"
                         "想提速可 --workers 2")
    ap.add_argument("--name", default=None, help="实验名，默认 baseline_yolov8n/s/m")
    ap.add_argument("--resume", action="store_true", help="断点续训")
    return ap.parse_args()


def append_results_csv(row):
    """统一实验记录：日期, 实验名, 配置, 指标 —— 论文表格从这里生成"""
    header = ["date", "exp", "model", "epochs", "imgsz", "batch",
              "params_M", "GFLOPs", "mAP50", "mAP50-95", "P", "R", "note"]
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def main():
    args = parse_args()
    name = args.name or "baseline_" + args.model.replace(".pt", "")
    run_dir = os.path.join(PROJECT, name)

    # ---------- 1. 训练（支持断点续训） ----------
    if args.resume:
        last = os.path.join(PROJECT, name, "weights", "last.pt")
        if not os.path.exists(last):
            raise SystemExit(f"未找到 {last}，无法续训；请去掉 --resume 直接重跑")
        model = YOLO(last)
        model.train(resume=True)   # 从 last.pt 恢复，沿用上次全部参数
    else:
        model = YOLO(args.model)
        model.train(
            data=DATA,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=42,            # 固定随机种子，保证可复现
            workers=args.workers,
            project=PROJECT,
            name=name,
            exist_ok=args.resume,
            resume=args.resume,
        )

    # ---------- 2. 用 best.pt 在 val 上正式评估 ----------
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    metrics = best.val(data=DATA, imgsz=args.imgsz, device=args.device)
    r = metrics.results_dict

    params_m, gflops = model.info()[1] / 1e6, model.info()[3]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        name, args.model.replace(".pt", ""),
        args.epochs, args.imgsz, args.batch,
        round(params_m, 3), round(gflops, 1),
        round(r["metrics/mAP50(B)"], 4), round(r["metrics/mAP50-95(B)"], 4),
        round(r["metrics/precision(B)"], 4), round(r["metrics/recall(B)"], 4),
        "基线 A0" if "n" in name else "对比/体检",
    ]
    append_results_csv(row)

    # ---------- 3. 控制台摘要 ----------
    print("\n===== 基线结果摘要 =====")
    print(f"实验名      : {name}")
    print(f"参数量      : {params_m:.3f} M   GFLOPs: {gflops:.1f}")
    print(f"mAP50       : {r['metrics/mAP50(B)']:.4f}   (承诺线 0.78 / 保底线 0.75)")
    print(f"mAP50-95    : {r['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision   : {r['metrics/precision(B)']:.4f}   Recall: {r['metrics/recall(B)']:.4f}")
    print(f"结果目录    : {run_dir}")
    print(f"实验记录    : {RESULTS_CSV}")


if __name__ == "__main__":
    main()
