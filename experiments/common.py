# -*- coding: utf-8 -*-
"""阶段 3-5 实验公共工具：模型配置生成 / results.csv 统一记录 / 统一评估 / 类别频率统计

供 train_improve.py / train_ablation.py / train_compare.py / eval_pareto.py 复用。
train_baseline.py 保持独立不动（已实测跑通过）。
"""
import os
import sys
import io
import csv
import math
import yaml as pyyaml
from collections import Counter
from datetime import datetime
from pathlib import Path

# Windows 控制台 GBK → 统一 UTF-8（含中文打印不炸）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import ultralytics  # noqa: E402  必须在设置环境变量之后 import

HERE = Path(__file__).resolve().parent          # experiments/
ROOT = HERE.parent                              # 项目根
DATA = ROOT / "dataset_guava_leaf_fruit" / "data.yaml"
PROJECT = HERE / "runs"
RESULTS_CSV = HERE / "results.csv"
CFG_DIR = HERE / "cfgs"
PKG_CFG = Path(ultralytics.__file__).resolve().parent / "cfg" / "models"

# v8n 的默认系数——I1/I2 的"倍率"都相对这两个基准
V8N_WIDTH = 0.25
V8N_DEPTH = 0.33

CSV_HEADER = ["date", "exp", "model", "epochs", "imgsz", "batch",
              "params_M", "GFLOPs", "mAP50", "mAP50-95", "P", "R", "note"]


def append_results_csv(row):
    """统一实验记录：追加一行。论文所有表格都从 results.csv 生成。"""
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def build_cfg(width_mult=None, depth_mult=None, template=None, name="custom"):
    """生成自定义模型 yaml 到 cfgs/<name>.yaml 并返回路径。

    - template=None   → 以官方 v8 结构为底（路径自动定位包内 cfg/models/v8/yolov8.yaml）
    - template=路径    → 以本地 yaml 为底（如 ECA 模板，ECA 行不含通道参数，改系数不影响结构）
    - width_mult/depth_mult 传绝对系数（如 0.125），None 则用 v8n 默认（0.25/0.33）

    关键点：必须删掉 scales 字典，否则 parse_model 会按 scale 名覆盖顶层系数。
    """
    CFG_DIR.mkdir(exist_ok=True)
    src = PKG_CFG / "v8" / "yolov8.yaml" if template is None else Path(template)
    d = pyyaml.safe_load(src.read_text(encoding="utf-8"))
    d.pop("scales", None)
    d["width_multiple"] = round(V8N_WIDTH if width_mult is None else width_mult, 4)
    d["depth_multiple"] = round(V8N_DEPTH if depth_mult is None else depth_mult, 4)
    d["max_channels"] = 1024
    out = CFG_DIR / f"{name}.yaml"
    out.write_text(pyyaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def evaluate_and_record(weights, name, model_label, epochs, imgsz, batch, note, device="0"):
    """加载 best.pt 在 val 上正式评估 → 打印摘要 → 追加 results.csv，返回指标 dict。"""
    from ultralytics import YOLO

    best = YOLO(str(weights))
    metrics = best.val(data=str(DATA), split="val", imgsz=imgsz, device=device, workers=0)
    r = metrics.results_dict

    info = best.info()                      # [层数, 参数量, 梯度量, GFLOPs]
    params_m, gflops = info[1] / 1e6, info[3]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        name, model_label, epochs, imgsz, batch,
        round(params_m, 3), round(gflops, 1),
        round(r["metrics/mAP50(B)"], 4), round(r["metrics/mAP50-95(B)"], 4),
        round(r["metrics/precision(B)"], 4), round(r["metrics/recall(B)"], 4),
        note,
    ]
    append_results_csv(row)

    print("\n===== 结果摘要 =====")
    print(f"实验名    : {name}   ({note})")
    print(f"参数量    : {params_m:.3f} M   GFLOPs: {gflops:.1f}")
    print(f"mAP50     : {r['metrics/mAP50(B)']:.4f}   mAP50-95: {r['metrics/mAP50-95(B)']:.4f}")
    print(f"P / R     : {r['metrics/precision(B)']:.4f} / {r['metrics/recall(B)']:.4f}")
    print(f"实验记录  : {RESULTS_CSV}")
    return {"params_M": params_m, "GFLOPs": gflops,
            "mAP50": r["metrics/mAP50(B)"], "mAP50-95": r["metrics/mAP50-95(B)"],
            "P": r["metrics/precision(B)"], "R": r["metrics/recall(B)"]}


def count_class_freq():
    """扫描 train 标签统计每类实例数，返回 (weights, freq)。

    权重 = 1/sqrt(freq)（温和反频率，避免过度补偿），再归一到均值 1。
    I4a 类别加权用的就是它；短板类（果-炭疽/叶-点斑/果-疮痂）权重会更大。
    """
    meta = pyyaml.safe_load(Path(DATA).read_text(encoding="utf-8"))
    nc = meta["nc"]
    base = Path(meta.get("path") or DATA.parent)
    train_img_dir = base / meta["train"]
    # images/… → labels/…（保持相对结构）
    labels_dir = Path(str(train_img_dir).replace(
        f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    if not labels_dir.exists():
        raise SystemExit(f"未找到标签目录 {labels_dir}，请检查 data.yaml 的 path/train 字段")

    cnt = Counter()
    for txt in labels_dir.rglob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                cnt[int(float(parts[0]))] += 1

    freq = [cnt.get(i, 0) for i in range(nc)]
    w = [1.0 / math.sqrt(f) if f else 1.0 for f in freq]
    mean = sum(w) / len(w)
    weights = [round(x / mean, 4) for x in w]
    print(f"[I4a] train 每类实例数: {freq}")
    print(f"[I4a] 类别权重(1/sqrt(freq), 均值=1): {weights}")
    return weights, freq


def train_kwargs(args, name):
    """统一的训练参数（与 B0 基线完全一致，保证变量单一：只动改进项）"""
    return dict(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, seed=42, workers=args.workers,
        project=str(PROJECT), name=name,
    )


def resolve_run_dir(name):
    """训练后定位真实 run 目录。

    坑：同名 run 已存在时 ultralytics 自动加 -2/-3 后缀（如冒烟跑占了
    improve_i3a，正式训练落到 improve_i3a-2）。写死 runs/<name> 会误读
    冒烟跑的旧权重（I3a 已踩坑：100 轮成绩被记成 1 轮的 0.1334）。
    这里按修改时间取最新的匹配目录。
    """
    cands = sorted(PROJECT.glob(name + "*"),
                   key=lambda p: p.stat().st_mtime if p.is_dir() else 0)
    return cands[-1] if cands else PROJECT / name
