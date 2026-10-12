# -*- coding: utf-8 -*-
"""export_onnx.py — 把训练好的 best.pt 导出为 best.onnx（模型冻结后执行一次即可）

用法：
    conda activate py3_11
    cd C:\\Users\\32683\\Desktop\\guava\\experiments
    python export_onnx.py --run runs/baseline_yolov8n-2     # 导出基线模型
    python export_onnx.py --run runs/improved_xxx           # 导出最终改进模型

导出后在同目录 weights/ 下生成 best.onnx，系统检测页自动优先加载（detector.py 已支持）。
依赖：pip install onnxruntime（推理用，导出本身只需 ultralytics）
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs 下的训练目录，如 runs/baseline_yolov8n-2")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    pt = os.path.join(args.run, "weights", "best.pt")
    if not os.path.isfile(pt):
        print(f"[错误] 找不到 {pt}")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(pt)
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                             dynamic=False, simplify=False)
    print(f"[完成] 已导出: {onnx_path}")

    # 冒烟验证：onnx 能否正常推理一张图
    from onnxruntime import InferenceSession
    sess = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[验证] ONNX 加载成功，输入: {sess.get_inputs()[0].shape}, "
          f"输出: {sess.get_outputs()[0].shape}")
    print("[提示] 系统后端重启后自动切换到 ONNX 推理（detector.py 优先找 best.onnx）")


if __name__ == "__main__":
    main()
