# -*- coding: utf-8 -*-
"""bench_inference.py — CPU 推理压测：pt vs onnx 各跑 N 次统计平均耗时（含首帧加载预热剔除）"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

import cv2

RUN = r"C:\Users\32683\Desktop\guava\experiments\runs\baseline_yolov8n-2"
IMG = r"C:\Users\32683\Desktop\guava\system\backend\data\samples\sample_cls1.jpg"
N = 20


def bench(weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    img = cv2.imread(IMG)
    m.predict(source=img, imgsz=640, conf=0.25, device="cpu", verbose=False)  # 预热
    t0 = time.time()
    for _ in range(N):
        m.predict(source=img, imgsz=640, conf=0.25, device="cpu", verbose=False)
    return (time.time() - t0) / N * 1000


pt_ms = bench(os.path.join(RUN, "weights", "best.pt"))
onnx_ms = bench(os.path.join(RUN, "weights", "best.onnx"))
print(f"pt   : {pt_ms:.1f} ms/张 ({N} 次平均, 已预热)")
print(f"onnx : {onnx_ms:.1f} ms/张 ({N} 次平均, 已预热)")
print(f"加速比: {pt_ms / onnx_ms:.2f}x")
