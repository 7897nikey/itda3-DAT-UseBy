# -*- coding: utf-8 -*-
"""제출 파이프라인 공통 코어 (vote / cascade 두 방식이 함께 사용).

측정으로 확정된 구성 (dev 297장, 2026-09-12)
  1) EXIF 회전 보정  — 8.7%(293/3352)가 회전 태그를 갖고 있어 안 하면 크롭 좌표가 어긋난다
  2) YOLO EXP 박스 크롭 (여백 15%), 실패 시 전체 이미지 축소본
  3) 전처리 3경로: combo_all -> adaptive -> illum
     단독 점수 순이 아니라 "서로 다른 이미지를 살리는 순"으로 고른 조합 (0.7598 -> 0.7856)
  4) OCR 결과를 읽는 순서(위->아래, 왼->오)로 정렬  (+1.9%p, 개선 7/악화 1)
  5) 텍스트 정규화 6규칙 후 재파싱  (+3.5%p, 개선 16/악화 3, p=0.0044)
  6) date_parser.extract_expiry_fields 로 날짜 확정

주의
  - torch 불필요 (onnxruntime + RapidOCR). 채점 서버가 오프라인이므로 가중치는 미리 로컬에 있어야 한다.
  - 시간 측정은 warm-up 을 제외한다.
"""
import os
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from date_parser import extract_expiry_fields

# ── 설정 ────────────────────────────────────────────────────────────────
YOLO_ONNX = os.environ.get("ITDA_YOLO", "yolo_exp_mfg.onnx")
REC_ONNX = os.environ.get("ITDA_REC", "")      # 비우면 RapidOCR 기본 인식 모델
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
CROP_MARGIN = 0.15
FULL_LONG_EDGE = 900
MAX_SIDE = 1600
THREADS = int(os.environ.get("ITDA_THREADS", "4"))
FIELDS = ("year", "month", "day")
NONE3 = ("NONE", "NONE", "NONE")


# ── 이미지 ──────────────────────────────────────────────────────────────
def load_image(path):
    """EXIF 회전을 실제 픽셀에 반영해 BGR 로 읽는다."""
    with Image.open(path) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _letterbox(im_rgb, size=YOLO_IMGSZ, color=(114, 114, 114)):
    h, w = im_rgb.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.full((size, size, 3), color, np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(im_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, s, dw, dh


class Detector:
    """YOLO(ONNX) 로 소비기한(EXP) 영역을 찾는다. 팀 검출기가 바뀌면 이 클래스만 갈아끼우면 된다."""

    def __init__(self, onnx_path=YOLO_ONNX, threads=THREADS):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.sess.run(None, {"images": np.zeros((1, 3, YOLO_IMGSZ, YOLO_IMGSZ), np.float32)})  # warm-up

    def crop(self, bgr, margin=CROP_MARGIN):
        """-> (영역 이미지, 'crop' | 'full', EXP 점수)"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, s, dw, dh = _letterbox(rgb)
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        preds = self.sess.run(None, {"images": x})[0][0].T      # (8400, 4+클래스수)
        i = int(np.argmax(preds[:, 4]))                          # 0번 클래스 = EXP
        score = float(preds[i, 4])
        H, W = bgr.shape[:2]
        if score >= YOLO_CONF:
            cx, cy, bw, bh = preds[i, :4]
            x1, y1 = (cx - bw / 2 - dw) / s, (cy - bh / 2 - dh) / s
            x2, y2 = (cx + bw / 2 - dw) / s, (cy + bh / 2 - dh) / s
            mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
            x1, y1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
            x2, y2 = min(W, int(x2 + mw)), min(H, int(y2 + mh))
            if x2 - x1 >= 10 and y2 - y1 >= 10:
                return bgr[y1:y2, x1:x2], "crop", score
        if max(H, W) > FULL_LONG_EDGE:                           # 검출 실패 -> 전체 이미지 축소본
            f = FULL_LONG_EDGE / max(H, W)
            bgr = cv2.resize(bgr, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        return bgr, "full", score


# ── 전처리 3경로 (측정으로 확정) ──────────────────────────────────────────
def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _dark_text(g):
    return 255 - g if g.mean() < 110 else g


def _upscale(img, k=2.0):
    h, w = img.shape[:2]
    if max(h, w) * k > MAX_SIDE:
        k = MAX_SIDE / max(h, w)
    if k <= 1.01:
        return img
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_LANCZOS4)


def _deskew(img):
    """글자 기울기를 추정해 회전 (±20도까지). OpenCV 각도 규약을 정규화해야 한다."""
    g = _dark_text(_gray(img))
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(b)
    if pts is None or len(pts) < 20:
        return img
    ang = cv2.minAreaRect(pts)[-1]
    while ang < -45:
        ang += 90
    while ang > 45:
        ang -= 90
    if abs(ang) < 0.5 or abs(ang) > 20:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _sharpen(img):
    g = _gray(img)
    return _to_bgr(cv2.addWeighted(g, 1.7, cv2.GaussianBlur(g, (0, 0), 1.2), -0.7, 0))


def v_combo_all(img):
    """1순위: 기울기 보정 + 2배 확대 + 대비 강화 + 선명화 (단독 0.7217)"""
    return _sharpen(_clahe(_upscale(_deskew(img), 2.0)))


def v_adaptive(img):
    """2순위: 국소 이진화 (단독 0.6543이지만 combo_all 이 놓친 것을 가장 많이 살림)"""
    g = _dark_text(_gray(img))
    return _to_bgr(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 10))


def v_illum(img):
    """3순위: 조명·반사 정규화 (단독 0.6195, 앞의 둘이 놓친 반사 사진을 살림)"""
    g = _gray(img).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=max(9, min(img.shape[:2]) / 8))
    return _to_bgr(np.clip(g / (bg + 1e-3) * 128.0, 0, 255).astype(np.uint8))


VARIANTS = [("combo_all", v_combo_all), ("adaptive", v_adaptive), ("illum", v_illum)]


# ── OCR ─────────────────────────────────────────────────────────────────
def make_engine(rec_path=REC_ONNX, threads=THREADS):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        from rapidocr import RapidOCR
    attempts = []
    if rec_path:
        attempts += [dict(rec_model_path=rec_path, intra_op_num_threads=threads),
                     dict(rec_model_path=rec_path),
                     dict(params={"Rec.model_path": rec_path})]
    attempts += [dict(intra_op_num_threads=threads), dict()]
    last = None
    for kw in attempts:
        try:
            return RapidOCR(**kw)
        except Exception as e:
            last = e
    raise RuntimeError(f"RapidOCR 초기화 실패: {last}")


def _sort_reading_order(items):
    """검출 순서가 아니라 사람이 읽는 순서로 정렬.

    정렬하지 않으면 '2026 08 03'(실제 03 08 2026)처럼 날짜 조각 순서가 뒤바뀐다.
    같은 줄 판정 기준은 박스 높이의 60%.
    """
    if not items:
        return items
    hs = [max(1.0, b[2][1] - b[0][1]) for _, _, b in items]
    tol = 0.6 * (sum(hs) / len(hs))
    items = sorted(items, key=lambda it: (it[2][0][1] + it[2][2][1]) / 2)
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
        out.extend(sorted(ln, key=lambda it: it[2][0][0]))
    return out


def run_ocr(engine, bgr):
    """-> 읽는 순서로 정렬된 텍스트 한 줄"""
    res = engine(bgr)
    if isinstance(res, tuple):
        res = res[0]
    if res is None:
        return ""
    items = []
    txts = getattr(res, "txts", None)
    if txts is not None:
        boxes = getattr(res, "boxes", None)
        scores = getattr(res, "scores", None) or [None] * len(txts)
        for i, t in enumerate(txts):
            b = np.asarray(boxes[i]).tolist() if boxes is not None else [[0, 0], [1, 0], [1, 1], [0, 1]]
            items.append([str(t), scores[i], b])
    else:
        for r in res:
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                items.append([str(r[1]), r[2], np.asarray(r[0]).tolist()])
    return " ".join(t for t, _, _ in _sort_reading_order(items)).strip()


# ── 텍스트 정규화 (순서 중요) ────────────────────────────────────────────
def _drop_time(t):
    return re.sub(r"\b\d{1,2}\s*[:：]\s*\d{2}(\s*[:：]\s*\d{2})?\b", " ", t)


def _squeeze(t):
    t = re.sub(r"(?<=[\d.\-/])\s+(?=[\d.\-/])", "", t)
    return re.sub(r"([.\-/])[.\-/]+", r"\1", t)


def _clean(t):
    return re.sub(r"[^0-9가-힣A-Za-z.\-/년월일 ]", " ", t)


def _year_prefix(t):
    return re.sub(r"\b\d+(20\d{2})([.\-/])", r"\1\2", t)


def _us_order(t):
    def f(m):
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{y}.{a:02d}.{b:02d}" if a <= 12 and 1 <= b <= 31 else m.group(0)
    return re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", f, t)


def _trim_extra(t):
    return re.sub(r"([.\-/]\d{2})\d\b", r"\1", t)


def normalize_text(t):
    """시각 제거 -> 공백/구분자 정리 -> 잡문자 제거 -> 연도 앞 잡숫자 -> MM/DD/YYYY -> 끝자리 중복.

    공백을 먼저 지우면 '10 16:42' 가 '1016:42' 로 붙어 시각 제거가 실패하므로 순서를 지켜야 한다.
    """
    for fn in (_drop_time, _squeeze, _clean, _year_prefix, _us_order, _trim_extra):
        t = fn(t)
    return t


# ── 파싱 ────────────────────────────────────────────────────────────────
def _parse_raw(text, kind):
    """크롭 텍스트에는 '소비기한' 글자가 없는 경우가 많아 파서의 키워드 로직이 발동하도록 붙여준다."""
    full = ("소비기한 " + text) if (kind == "crop" and text.strip()) else text
    if not full.strip():
        return NONE3
    try:
        p = extract_expiry_fields(full)
    except Exception:
        return NONE3
    return tuple(str(p.get(k, "NONE")) for k in FIELDS)


def n_filled(pred):
    return sum(v != "NONE" for v in pred)


def parse_text(text, kind):
    """원문과 정규화본 중 더 많이 채운 쪽을 택한다 (동점이면 정규화본)."""
    base = _parse_raw(text, kind)
    cand = _parse_raw(normalize_text(text), kind)
    return cand if n_filled(cand) >= n_filled(base) else base


def to_row(image_id, pred):
    y, m, d = pred
    final = "NONE" if (y, m, d) == NONE3 else f"{y}-{m}-{d}"
    return {"image_id": image_id, "year": y, "month": m, "day": d, "final_date": final}


# ── 실행 뼈대 ───────────────────────────────────────────────────────────
def list_images(input_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if os.path.splitext(f)[1].lower() in exts)


def run(strategy, input_dir=None, output_path=None, rec_path=REC_ONNX, limit=0, log_every=50):
    """strategy(engine, region, kind) -> (year, month, day) 를 받아 submission 을 만든다."""
    import pandas as pd

    input_dir = input_dir or os.environ.get("ITDA_INPUT_DIR", "./val_images")
    output_path = output_path or os.environ.get("ITDA_OUTPUT_PATH", "./submission.csv")

    det = Detector()
    engine = make_engine(rec_path)
    run_ocr(engine, np.full((48, 160, 3), 255, np.uint8))   # OCR warm-up (측정에서 제외)

    files = list_images(input_dir)
    if limit:
        files = files[:limit]
    print(f"입력 이미지 {len(files)}장", flush=True)

    rows, t0 = [], time.perf_counter()
    for i, path in enumerate(files, 1):
        image_id = os.path.splitext(os.path.basename(path))[0]
        try:
            bgr = load_image(path)
            region, kind, _ = det.crop(bgr)
            pred = strategy(engine, region, kind)
        except Exception as e:                                # 한 장이 깨져도 전체를 죽이지 않는다
            print(f"[WARN] {image_id}: {e}", flush=True)
            pred = NONE3
        rows.append(to_row(image_id, pred))
        if i % log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(files)}  경과 {el / 60:.1f}분  장당 {el / i:.2f}초", flush=True)

    el = time.perf_counter() - t0
    n = max(1, len(rows))
    print(f"완료 {len(rows)}장 / {el:.1f}초 (장당 {el / n:.3f}초, warm-up 제외)", flush=True)
    df = pd.DataFrame(rows, columns=["image_id", "year", "month", "day", "final_date"])
    df.to_csv(output_path, index=False)
    print(f"저장: {output_path}  |  응답률 {(df.final_date != 'NONE').mean():.1%}", flush=True)
    return df
