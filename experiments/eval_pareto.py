# -*- coding: utf-8 -*-
"""阶段 5 C6：统一评估 + 参数量-精度 Pareto 图

对一批训好的模型（基线 / 消融各级 / 对比模型）在同一个 val 上重跑 model.val()，
统一收集 Params/GFLOPs/mAP50/P/R → 追加 results.csv（note=C6 统一评估），
并画"参数量-精度"散点图（论文对比章节的主图）。

用法（见 训练命令.md）：
    python eval_pareto.py --runs baseline_yolov8n-2 baseline_yolov8s ^
        improve_i3a ablation_a5 compare_c1 compare_c2

只做纯推理（每次几分钟），不训练。图输出到 experiments/pareto.png。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # 无窗口环境出图
import matplotlib.pyplot as plt

from common import HERE, PROJECT, evaluate_and_record

# 消融链上的关键点（A0→A5 的 V 型消融曲线也从这份数据出，论文可复用）
DEFAULT_RUNS = [
    "baseline_yolov8n-2",   # A0/B0 基线
    "ablation_a1",          # A1
    "ablation_a2",          # A2
    "ablation_a3",          # A3
    "ablation_a4",          # A4
    "ablation_a5",          # A5 = 本文模型
    "compare_c1",           # v5nu
    "compare_c2",           # v7-tiny
]

# 散点图标注名（run 目录名 → 图上显示），没列到的用目录名本身
DISPLAY = {
    "baseline_yolov8n-2": "A0 基线 v8n",
    "ablation_a1": "A1",
    "ablation_a2": "A2",
    "ablation_a3": "A3",
    "ablation_a4": "A4",
    "ablation_a5": "A5 本文模型",
    "compare_c1": "YOLOv5nu",
    "compare_c2": "YOLOv7-tiny",
    "baseline_yolov8s": "B2 v8s",
}


def parse_args():
    ap = argparse.ArgumentParser(description="C6 统一评估 + Pareto 图")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="experiments/runs/ 下的实验目录名（每个目录需有 weights/best.pt）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    return ap.parse_args()


def main():
    args = parse_args()
    points = []
    for run in args.runs:
        best = Path(PROJECT) / run / "weights" / "best.pt"
        if not best.exists():
            print(f"!! 跳过 {run}：未找到 {best}")
            continue
        m = evaluate_and_record(best, f"c6_{run}", run, "-", args.imgsz, "-",
                                note=f"C6 统一评估({run})", device=args.device)
        points.append((DISPLAY.get(run, run), m))

    if not points:
        raise SystemExit("没有任何可用模型，检查 --runs")

    # ── Pareto 散点图：x=参数量(M)，y=mAP50，越靠左上越好 ──
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

    xs = [m["params_M"] for _, m in points]
    ys = [m["mAP50"] * 100 for _, m in points]
    ax.scatter(xs, ys, s=60, color="#4a7c59", zorder=3)
    for (label, m), x, y in zip(points, xs, ys):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 5), fontsize=9, color="#333")
    ax.set_xlabel("参数量 (M)")
    ax.set_ylabel("mAP50 (%)")
    ax.set_title("参数量-精度 Pareto 图（同一 val 集统一评估）")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(0.12)
    fig.tight_layout()

    out = HERE / "pareto.png"
    fig.savefig(out)
    print(f"\nPareto 图已保存: {out}")

    # 控制台汇总表（按 mAP50 降序）
    print("\n===== C6 统一评估汇总（按 mAP50 降序）=====")
    print(f"{'模型':<22}{'Params(M)':>10}{'GFLOPs':>8}{'mAP50':>9}{'P':>8}{'R':>8}")
    for label, m in sorted(points, key=lambda t: -t[1]["mAP50"]):
        print(f"{label:<22}{m['params_M']:>10.3f}{m['GFLOPs']:>8.1f}"
              f"{m['mAP50']:>9.4f}{m['P']:>8.4f}{m['R']:>8.4f}")


if __name__ == "__main__":
    main()
