# -*- coding: utf-8 -*-
"""bbox_정주희.json(라벨링 툴 결과) -> YOLO 학습 포맷 변환.

핵심 처리:
- EXIF Orientation 보정: 라벨링 툴(브라우저 캔버스)은 EXIF 회전을 자동 적용한
  화면 기준으로 좌표를 기록하는데, PIL로 원본을 그냥 읽으면 회전 전 픽셀
  좌표계가 나온다. ImageOps.exif_transpose로 보정한 픽셀을 실제 학습 이미지로
  저장하고, 박스 좌표는 그 보정된 좌표계 기준으로 그대로 쓴다.
  (실측: 003189.jpg가 회전 미보정 시 박스가 이미지 범위를 벗어나는 것으로 확인됨)
- 클래스: 0=EXP(소비기한), 1=MFG(제조일자)
- train/val 분할: 이미지 단위로 섞은 �025 뒤 90/10 분할, MFG 박스가 있는
  이미지가 양쪽에 고르게 들어가도록 층화(80장뿐이라 val에 아예 안 들어가면
  MFG 검증이 안 됨).
"""
import json, os, random, shutil, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from PIL import Image, ImageOps

BASE = r"C:\Users\7897n\Desktop\School\26-2\DAT\ITDA_연합학술대회"
LABEL_JSON = os.path.join(BASE, "labeling", "bbox_정주희.json")
IMAGE_DIRS = [
    os.path.join(BASE, "labeling", "batch_A_images"),
    os.path.join(BASE, "labeling", "batch_B_images"),
    os.path.join(BASE, "labeling", "batch_C_images"),
]
OUT_DIR = os.path.join(BASE, "yolo_train", "dataset")
CLASSES = ["EXP", "MFG"]
VAL_RATIO = 0.10
SEED = 42

# 이미지 파일명 -> 실제 경로 매핑
image_paths = {}
for d in IMAGE_DIRS:
    for f in os.listdir(d):
        image_paths[f] = os.path.join(d, f)

with open(LABEL_JSON, encoding="utf-8") as f:
    data = json.load(f)
items = data["items"]
print(f"라벨 {len(items)}건 로드 (라벨러: {data.get('labeler')})")

# 층화 분할: MFG 박스 보유 여부로 나눠서 각각 val_ratio만큼 뽑기
random.seed(SEED)
has_mfg = [it for it in items if any(b["cls"] == "MFG" for b in it["boxes"])]
no_mfg = [it for it in items if it not in has_mfg]
random.shuffle(has_mfg)
random.shuffle(no_mfg)


def split(lst, ratio):
    n_val = max(1, round(len(lst) * ratio))
    return lst[n_val:], lst[:n_val]


train_items = []
val_items = []
for group in (has_mfg, no_mfg):
    tr, va = split(group, VAL_RATIO)
    train_items += tr
    val_items += va
random.shuffle(train_items)
random.shuffle(val_items)
print(f"train {len(train_items)}장 (MFG 포함 {sum(1 for it in train_items if it in has_mfg)}장), "
      f"val {len(val_items)}장 (MFG 포함 {sum(1 for it in val_items if it in has_mfg)}장)")

# 출력 폴더 준비
for split_name in ("train", "val"):
    os.makedirs(os.path.join(OUT_DIR, "images", split_name), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split_name), exist_ok=True)

skipped = 0
exif_fixed = 0


def process(it, split_name):
    global skipped, exif_fixed
    fname = it["filename"]
    src = image_paths.get(fname)
    if src is None:
        print("이미지 없음, 스킵:", fname)
        skipped += 1
        return
    try:
        with Image.open(src) as img:
            orientation = img.getexif().get(274)
            fixed = ImageOps.exif_transpose(img)
            if orientation not in (None, 1):
                exif_fixed_local = True
            else:
                exif_fixed_local = False
            w, h = fixed.size
            fixed = fixed.convert("RGB")
            out_img_path = os.path.join(OUT_DIR, "images", split_name, fname)
            fixed.save(out_img_path, quality=95)
    except Exception as e:
        print("이미지 처리 실패, 스킵:", fname, e)
        skipped += 1
        return
    if exif_fixed_local:
        exif_fixed += 1

    lines = []
    for b in it["boxes"]:
        cls_idx = CLASSES.index(b["cls"])
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        # 범위를 벗어나면 이미지 경계로 clip (혹시 모를 잔여 오차 방지)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_name = os.path.splitext(fname)[0] + ".txt"
    with open(os.path.join(OUT_DIR, "labels", split_name, label_name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


for it in train_items:
    process(it, "train")
for it in val_items:
    process(it, "val")

print(f"완료. EXIF 회전 보정 적용된 이미지: {exif_fixed}장, 스킵: {skipped}장")

# data.yaml 작성
yaml_content = f"""path: {OUT_DIR}
train: images/train
val: images/val
names:
  0: EXP
  1: MFG
"""
yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print("data.yaml 작성:", yaml_path)
