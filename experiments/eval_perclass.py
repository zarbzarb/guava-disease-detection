# -*- coding: utf-8 -*-
"""逐类体检：指定 run 的 best.pt 在 val 上的逐类 P/R/mAP50"""
import os, sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset_guava_leaf_fruit", "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline_yolov8n-2", help="runs 下的实验目录名")
    args = ap.parse_args()

    best = os.path.join(HERE, "runs", args.run, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit(f"未找到 {best}")
    m = YOLO(best)
    r = m.val(data=DATA, split="val", device=0, workers=0, verbose=False)

    print("\nmAP50 =", round(r.box.map50, 4), "| P =", round(r.box.mp, 4), "| R =", round(r.box.mr, 4))
    names = r.names
    print(f"{'id':<4}{'class':<26}{'P':>8}{'R':>8}{'mAP50':>8}")
    for c, p_i, r_i, ap_i in zip(r.box.ap_class_index, r.box.p, r.box.r, r.box.ap50):
        print(f"{c:<4}{names[c]:<26}{p_i:>8.4f}{r_i:>8.4f}{ap_i:>8.4f}")


if __name__ == "__main__":
    main()
