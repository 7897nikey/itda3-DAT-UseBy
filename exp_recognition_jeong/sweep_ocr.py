# -*- coding: utf-8 -*-
"""전처리 변형 x 인식 옵션 스윕(sweep). 크롭 단위로 돌려서 인식 축만 따로 잰다.

고정: YOLO 크롭(yolo_scan.csv 의 박스), 날짜 파서(date_parser.py), 평가셋(splits.csv 의 dev)
변수: --variants (variants.py 의 전처리), --margin (크롭 여백)

결과는 sweep/<variant>__m<margin>.jsonl 로 저장되고, 이미 있는 이미지는 건너뛴다.
(중간에 Ctrl+C 해도 다시 실행하면 이어서 진행)

사용법:
    python sweep_ocr.py --images .. --split dev                       # 전체 변형
    python sweep_ocr.py --images .. --split dev --variants none,dotclose
    python sweep_ocr.py --images .. --split dev --variants none --margin 0.30
    python sweep_ocr.py --selftest                                    # 엔진 설치 확인만
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from variants import VARIANTS

CONF = 0.25            # YOLO EXP 채택 임계값 (predict.ipynb 와 동일)
FULL_LONG_EDGE = 900   # 검출 실패 시 전체 이미지 리사이즈 크기


# ── RapidOCR 어댑터 (패키지 버전에 따라 반환 형태가 달라서 흡수) ──────────────
def make_engine(threads=4, rec_path=None):
    """RapidOCR 엔진. rec_path 를 주면 그 인식 모델로 바꿔 끼운다.

    팀원이 준 korean_PP-OCRv5_rec_mobile.onnx 는 문자 사전을 ONNX 메타데이터('character',
    11947자)에 품고 있어서 별도 사전 파일이 필요 없다.
    패키지 버전에 따라 인자 이름이 다르므로 순서대로 시도한다.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        from rapidocr import RapidOCR  # 신버전 패키지명
    attempts = []
    if rec_path:
        attempts += [
            dict(rec_model_path=rec_path, intra_op_num_threads=threads),
            dict(rec_model_path=rec_path),
            dict(params={"Rec.model_path": rec_path, "Global.intra_op_num_threads": threads}),
            dict(params={"Rec.model_path": rec_path}),
        ]
    attempts += [dict(intra_op_num_threads=threads), dict()]
    last = None
    for kw in attempts:
        try:
            eng = RapidOCR(**kw)
            if rec_path and not kw:
                print("[경고] 한국어 모델을 적용하지 못했습니다. 기본 모델로 돌아갑니다.", flush=True)
            return eng
        except Exception as e:      # 인자 이름이 다르면 다음 방식으로
            last = e
    raise SystemExit(f"RapidOCR 초기화 실패: {last}")


def _sort_reading_order(items):
    """OCR 박스를 사람이 읽는 순서(위->아래, 왼->오)로 정렬한다.

    엔진이 주는 순서는 검출 순서일 뿐이라, 정렬하지 않고 이어붙이면
    "2026 08 03"(실제로는 03 08 2026)처럼 날짜 조각의 순서가 뒤바뀐다.
    같은 줄로 볼지 여부는 박스 높이의 60% 를 기준으로 판단한다.
    """
    if not items:
        return items
    hs = [max(1.0, b[2][1] - b[0][1]) for _, _, b in items]
    tol = 0.6 * (sum(hs) / len(hs))
    items = sorted(items, key=lambda it: (it[2][0][1] + it[2][2][1]) / 2)  # y 중심
    lines, cur, cur_y = [], [], None
    for it in items:
        yc = (it[2][0][1] + it[2][2][1]) / 2
        if cur_y is None or abs(yc - cur_y) <= tol:
            cur.append(it)
            cur_y = yc if cur_y is None else (cur_y + yc) / 2
        else:
            lines.append(cur)
            cur, cur_y = [it], yc
    lines.append(cur)
    out = []
    for ln in lines:
        out.extend(sorted(ln, key=lambda it: it[2][0][0]))  # x 왼쪽부터
    return out


def run_ocr(engine, bgr):
    """-> [[text, score, box], ...] (읽는 순서로 정렬됨)"""
    res = engine(bgr)
    if isinstance(res, tuple):
        res = res[0]
    if res is None:
        return []
    items = []
    txts = getattr(res, "txts", None)
    if txts is not None:  # 신버전: 객체 반환
        scores = getattr(res, "scores", None) or [None] * len(txts)
        boxes = getattr(res, "boxes", None)
        for i, t in enumerate(txts):
            b = np.asarray(boxes[i]).tolist() if boxes is not None else [[0, 0], [1, 0], [1, 1], [0, 1]]
            s = scores[i]
            items.append([str(t), float(s) if s is not None else None, b])
    else:                 # 구버전: [box, text, score]
        for r in res:
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                items.append([str(r[1]), float(r[2]), np.asarray(r[0]).tolist()])
    items = _sort_reading_order(items)
    return [[t, s, [[int(x), int(y)] for x, y in b]] for t, s, b in items]


# ── 크롭 ────────────────────────────────────────────────────────────────
def load_image(path):
    with Image.open(path) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def crop_or_full(bgr, row, margin):
    """YOLO EXP 박스를 margin 만큼 넓혀 자른다. 검출 실패면 전체 이미지 축소본."""
    if float(row.exp_score) >= CONF:
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = (float(row.exp_x1), float(row.exp_y1), float(row.exp_x2), float(row.exp_y2))
        mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
        x1, y1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
        x2, y2 = min(W, int(x2 + mw)), min(H, int(y2 + mh))
        if x2 - x1 >= 10 and y2 - y1 >= 10:
            return bgr[y1:y2, x1:x2], "crop"
    h, w = bgr.shape[:2]
    if max(h, w) > FULL_LONG_EDGE:
        s = FULL_LONG_EDGE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr, "full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="..")
    ap.add_argument("--scan", default="yolo_scan.csv")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--split", default="dev", help="dev / holdout / all")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--outdir", default="sweep")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="빠른 확인용: 앞 N장만")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rec", default=None, help="인식 모델 onnx 경로 (예: korean_PP-OCRv5_rec_mobile.onnx)")
    a = ap.parse_args()

    engine = make_engine(a.threads, a.rec)
    if a.rec:
        print(f"인식 모델: {a.rec}", flush=True)
    if a.selftest:
        img = np.full((60, 260, 3), 255, np.uint8)
        cv2.putText(img, "2026.05.29", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        t = time.perf_counter()
        print("인식 결과:", run_ocr(engine, img), f"({time.perf_counter() - t:.2f}초)")
        print("설치 정상. 이제 --selftest 없이 실행하세요.")
        return

    names = list(VARIANTS) if a.variants == "all" else [v.strip() for v in a.variants.split(",")]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"모르는 변형: {unknown}\n사용 가능: {list(VARIANTS)}")

    sp = pd.read_csv(a.splits, dtype=str)
    sp = sp[sp["split"] != "excluded"]
    if a.split != "all":
        sp = sp[sp["split"] == a.split]
    scan = pd.read_csv(a.scan)
    scan["image_id"] = scan["image_id"].astype(str)
    sp["iid"] = sp["image_id"].astype(int)
    scan["iid"] = scan["image_id"].astype(int)
    df = sp.merge(scan, on="iid", suffixes=("", "_s"))
    if a.limit:
        df = df.head(a.limit)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"평가셋 {len(df)}장 x 변형 {len(names)}개 (margin={a.margin})", flush=True)

    for vi, name in enumerate(names, 1):
        out = os.path.join(a.outdir, f"{name}__m{int(a.margin * 100):02d}.jsonl")
        done = set()
        if os.path.exists(out):
            with open(out, encoding="utf-8") as fh:
                done = {json.loads(l)["file"] for l in fh if l.strip()}
        todo = df[~df["file"].isin(done)]
        print(f"[{vi}/{len(names)}] {name}: 남은 {len(todo)}장", flush=True)
        fn = VARIANTS[name]
        t_start = time.perf_counter()
        with open(out, "a", encoding="utf-8") as fh:
            for k, row in enumerate(todo.itertuples(), 1):
                bgr = load_image(os.path.join(a.images, row.file))
                region, kind = crop_or_full(bgr, row, a.margin)
                t0 = time.perf_counter()
                proc = fn(region)
                t1 = time.perf_counter()
                res = run_ocr(engine, proc)
                t2 = time.perf_counter()
                fh.write(json.dumps({
                    "file": row.file, "image_id": row.image_id, "variant": name,
                    "margin": a.margin, "kind": kind, "exp_score": float(row.exp_score),
                    "ocr": res, "t_prep": round(t1 - t0, 4), "t_ocr": round(t2 - t1, 4),
                }, ensure_ascii=False) + "\n")
                fh.flush()
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)} ({(time.perf_counter() - t_start) / 60:.1f}분)", flush=True)
        print(f"  -> {out}", flush=True)
    print("스윕 완료. 이제: python analyze_sweep.py", flush=True)


if __name__ == "__main__":
    main()
