# -*- coding: utf-8 -*-
"""2클래스(EXP/MFG) YOLOv8n 파인튜닝. CPU 전용 환경."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

BASE = r"C:\Users\7897n\Desktop\School\26-2\DAT\ITDA_연합학술대회"
DATA_YAML = os.path.join(BASE, "yolo_train", "dataset", "data.yaml")

model = YOLO("yolov8n.pt")  # 사전학습 가중치(공개 체크포인트)에서 파인튜닝

results = model.train(
    data=DATA_YAML,
    epochs=80,
    imgsz=640,
    batch=16,
    device="cpu",
    patience=20,
    project=os.path.join(BASE, "yolo_train", "runs"),
    name="exp_mfg_v1",
    workers=4,
    verbose=True,
)
print("학습 완료. best.pt 위치:", results.save_dir)
