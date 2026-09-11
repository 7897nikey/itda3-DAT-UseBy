# -*- coding: utf-8 -*-
"""학습된 best.pt를 ONNX로 변환 — predict.ipynb(EasyOCR 환경, torch 2.2.2)와
분리된 환경 충돌 없이 쓰기 위함. onnxruntime만 있으면 되고 ultralytics/최신
torch가 필요 없음."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

BASE = r"C:\Users\7897n\Desktop\School\26-2\DAT\ITDA_연합학술대회"
MODEL = os.path.join(BASE, "yolo_train", "runs", "exp_mfg_v1", "weights", "best.pt")

model = YOLO(MODEL)
path = model.export(format="onnx", imgsz=640, simplify=True, opset=12)
print("ONNX 저장 위치:", path)
