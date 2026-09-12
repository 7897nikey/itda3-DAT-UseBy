# -*- coding: utf-8 -*-
"""전처리 변형을 눈으로 비교하는 대조표(contact sheet) 생성. OCR 엔진이 없어도 돌아간다.

사용법:
    python preview_variants.py --images .. --n 6 --out preview.jpg
    python preview_variants.py --images .. --files 000283,001792 --out preview_dot.jpg
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from sweep_ocr import load_image, crop_or_full
from variants import VARIANTS

CELL_W = 420


def fit(img, w=CELL_W, h=150):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    out = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 240, np.uint8)
    y, x = (h - out.shape[0]) // 2, (w - out.shape[1]) // 2
    canvas[y:y + out.shape[0], x:x + out.shape[1]] = out
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="..")
    ap.add_argument("--scan", default="yolo_scan.csv")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--files", default=None, help="image_id 를 쉼표로 (예: 000283,001792)")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--out", default="preview.jpg")
    a = ap.parse_args()

    scan = pd.read_csv(a.scan)
    sp = pd.read_csv(a.splits, dtype=str)
    sp = sp[sp["split"] == "dev"]
    scan["iid"] = scan["image_id"].astype(int)
    sp["iid"] = sp["image_id"].astype(int)
    df = sp.merge(scan, on="iid", suffixes=("", "_s"))
    if a.files:
        want = {int(x) for x in a.files.split(",")}
        df = df[df["iid"].isin(want)]
    else:
        df = df.sample(min(a.n, len(df)), random_state=0)

    names = list(VARIANTS) if a.variants == "all" else [v.strip() for v in a.variants.split(",")]
    cells, labels = [], []
    for row in df.itertuples():
        bgr = load_image(os.path.join(a.images, row.file))
        region, kind = crop_or_full(bgr, row, a.margin)
        for nm in names:
            cells.append(fit(VARIANTS[nm](region)))
            labels.append(f"{row.image_id} [{nm}] {kind} gold={row.final_date}")

    cols = len(names)
    rows = len(cells) // cols
    H, W = 150 + 22, CELL_W
    sheet = np.full((rows * H, cols * W, 3), 255, np.uint8)
    for i, (c, lb) in enumerate(zip(cells, labels)):
        r, q = divmod(i, cols)
        y, x = r * H, q * W
        cv2.putText(sheet, lb[:52], (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)
        sheet[y + 22:y + 22 + 150, x:x + W] = c
    cv2.imwrite(a.out, sheet)
    print(f"{rows}행 x {cols}열 -> {a.out}")


if __name__ == "__main__":
    main()
