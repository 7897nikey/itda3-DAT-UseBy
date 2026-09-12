# -*- coding: utf-8 -*-
"""크롭 이미지의 '상태'를 숫자로 진단한다 (OCR 불필요).

핵심 원칙: 크롭마다 크기가 다르므로, 형태 관련 지표는 **높이 64px로 정규화한 뒤** 잰다.
정규화하지 않으면 큰 사진일수록 흐림 지표가 커져서 비교 자체가 성립하지 않는다.

지표
  blur_n    : 정규화 영상의 라플라시안 분산 / 분산 (대비 영향 제거). 낮을수록 흐림
  contrast  : 원본 밝기 표준편차,  range: 상하위 5% 폭
  lum       : 평균 밝기 (낮으면 어두운 사진)
  glare     : 250 이상 포화 픽셀 비율 (빛 반사)
  gap_gain  : 3x3 close 로 잉크가 몇 % 늘어나는지. **획이 끊겨 있을수록 커진다 = 도트 지표**
  comp_per_w: 폭 대비 연결요소 수. 글자가 조각나 있을수록 큼
  stroke_n  : 정규화 영상의 획 두께 중앙값. 작으면 얇은 획(각인·번짐)
  ink       : 잉크 픽셀 비율
  skew      : 글자 기울기(도). OpenCV 각도 규약을 정규화해서 계산
  box_h/w   : 원본 크롭 크기

사용법:
    python diagnose.py --images .. --split dev --out diag.csv
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from sweep_ocr import load_image, crop_or_full

NORM_H = 64  # 형태 지표를 재는 기준 높이


def _norm_binary(g):
    """높이 64로 맞춘 뒤 '글자=흰색' 이진 영상으로."""
    h, w = g.shape
    s = NORM_H / max(h, 1)
    gn = cv2.resize(g, (max(8, int(w * s)), NORM_H), interpolation=cv2.INTER_AREA)
    gg = 255 - gn if gn.mean() < 110 else gn
    binv = cv2.adaptiveThreshold(gg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 8)
    return gn, binv


def _skew(binv):
    pts = cv2.findNonZero(binv)
    if pts is None or len(pts) < 20:
        return 0.0
    ang = cv2.minAreaRect(pts)[-1]
    while ang < -45:      # OpenCV 버전에 따라 [-90,0) 또는 (0,90] 로 나온다
        ang += 90
    while ang > 45:
        ang -= 90
    return float(ang)


def metrics(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    lo, hi = np.percentile(g, [5, 95])
    gn, binv = _norm_binary(g)

    lap = cv2.Laplacian(gn, cv2.CV_64F)
    blur_n = float(lap.var() / (gn.var() + 1e-6))

    ink = float((binv > 0).mean())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, k)
    gap_gain = float(((closed > 0).mean() - ink) / (ink + 1e-6))

    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
    comps = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 2]
    comp_per_w = float(len(comps) / max(gn.shape[1], 1) * 100)

    dt = cv2.distanceTransform(binv, cv2.DIST_L2, 3)
    stroke_n = float(np.median(dt[dt > 0]) * 2) if (dt > 0).any() else 0.0

    return {"blur_n": round(blur_n, 3), "contrast": round(float(g.std()), 1),
            "range": round(float(hi - lo), 1), "lum": round(float(g.mean()), 1),
            "glare": round(float((g >= 250).mean()), 4),
            "gap_gain": round(gap_gain, 3), "comp_per_w": round(comp_per_w, 2),
            "stroke_n": round(stroke_n, 2), "ink": round(ink, 3),
            "skew": round(_skew(binv), 1), "box_h": h, "box_w": w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="..")
    ap.add_argument("--scan", default="yolo_scan.csv")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--out", default="diag.csv")
    a = ap.parse_args()

    sp = pd.read_csv(a.splits, dtype=str)
    sp = sp[sp["split"] != "excluded"]
    if a.split != "all":
        sp = sp[sp["split"] == a.split]
    scan = pd.read_csv(a.scan)
    sp["iid"] = sp["image_id"].astype(int)
    scan["iid"] = scan["image_id"].astype(int)
    df = sp.merge(scan, on="iid", suffixes=("", "_s"))

    rows = []
    for row in df.itertuples():
        bgr = load_image(os.path.join(a.images, row.file))
        region, kind = crop_or_full(bgr, row, a.margin)
        rows.append({"image_id": row.image_id, "file": row.file, "cat": row.cat,
                     "kind": kind, "exp_score": row.exp_score, **metrics(region)})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    cols = ["blur_n", "contrast", "range", "lum", "glare", "gap_gain", "comp_per_w",
            "stroke_n", "skew", "box_h"]
    print(f"{len(out)}장 -> {a.out}\n")
    print("난이도 범주별 중앙값:")
    print(out.groupby("cat")[cols].median().round(2).to_string())
    print("\n전체 분포:")
    print(out[cols].describe().loc[["25%", "50%", "75%", "max"]].round(2).to_string())
    print("\n기울기 |skew| > 3도:", int((out.skew.abs() > 3).sum()), "장")
    print("포화 픽셀 1% 이상(반사 의심):", int((out.glare > 0.01).sum()), "장")
    print("어두움 lum < 100:", int((out.lum < 100).sum()), "장")
    print("저대비 range < 80:", int((out.range < 80).sum()), "장")


if __name__ == "__main__":
    main()
