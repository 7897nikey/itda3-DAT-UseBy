# -*- coding: utf-8 -*-
"""val셋 이미지에 best.pt로 예측해서 박스 그려진 결과 저장 (눈으로 확인용)."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

BASE = r"C:\Users\7897n\Desktop\School\26-2\DAT\ITDA_연합학술대회"
MODEL = os.path.join(BASE, "yolo_train", "runs", "exp_mfg_v1", "weights", "best.pt")
VAL_IMG_DIR = os.path.join(BASE, "yolo_train", "dataset", "images", "val")
OUT_DIR = os.path.join(BASE, "yolo_train", "sample_preds")

model = YOLO(MODEL)
results = model.predict(
    source=VAL_IMG_DIR,
    save=True,
    project=os.path.join(BASE, "yolo_train"),
    name="sample_preds",
    exist_ok=True,
    conf=0.25,
)
print(f"예측 완료: {len(results)}장")
for r in results:
    n_boxes = len(r.boxes)
    print(f"{os.path.basename(r.path)}: 박스 {n_boxes}개")
print("결과 폴더:", OUT_DIR)
