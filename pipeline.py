# -*- coding: utf-8 -*-
"""YOLO로 날짜 영역을 찾아 잘라낸 다음 그 조각만 읽는 파이프라인 본체.

순서는 이렇게 됨.
  1) 사진 읽기. 이때 회전을 바로잡음 (학습할 때와 같은 규칙을 써야 함)
  2) YOLO로 네 종류(date/due/code/full) 찾기, 입력 크기 960
  3) full이 있으면 그걸 쓰고 없으면 date와 due를 합친 범위를 잘라냄.
     줄이지 않고 원본 해상도에서 자르기 때문에 글자 크기가 보존됨
  4) 잘라낸 조각만 읽기
  5) 읽은 글자를 날짜로 바꾸기
  6) 실패한 것만 전체 사진으로 다시 시도

전체 사진을 통째로 읽지 않고 잘라서 읽는 이유는 두 가지임. 읽어야 할 픽셀이
줄어서 빨라지고, 포장에 같이 적힌 품목보고번호나 영양성분 숫자가 애초에
안 들어오기 때문에 오답도 줄어듦.
"""
from __future__ import annotations
import os, sys, time, glob, argparse
from pathlib import Path
import numpy as np, cv2, pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from date_parser_plus import extract_expiry_fields
from preprocess import enhance, binarize_dilate

NONE_ROW = {"year":"NONE","month":"NONE","day":"NONE","final_date":"NONE"}
CLS = {0:"date", 1:"due", 2:"code", 3:"full"}
IMAGE_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")


def load_bgr(path):
    """EXIF 회전을 적용해 읽는다. dataset/ 정규화와 동일 규약."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def union_box(boxes):
    a = np.asarray(boxes, dtype=float)
    return [a[:,0].min(), a[:,1].min(), a[:,2].max(), a[:,3].max()]


def pad_clip(box, W, H, pad=0.45, min_pad_px=12):
    x0,y0,x1,y1 = box
    pw = max((x1-x0)*pad, min_pad_px); ph = max((y1-y0)*pad, min_pad_px)
    return [max(0,int(x0-pw)), max(0,int(y0-ph)), min(W,int(x1+pw)), min(H,int(y1+ph))]


def upscale_small(crop, target_h=100):
    """작은 크롭은 OCR 인식률이 급락하므로 확대한다."""
    h, w = crop.shape[:2]
    if h >= target_h or h == 0: return crop
    s = target_h / h
    return cv2.resize(crop, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)


def make_abs_cfg(cfg_path):
    """설정에 적힌 상대경로를 절대경로로 바꾼 임시 설정 파일을 만들어서 그 경로를 돌려줌.

    RapidOCR이 설정 안의 상대경로를 현재 폴더가 아니라 자기 패키지가 설치된
    위치를 기준으로 해석해버림. 그대로 두면 가중치를 못 찾고 죽음.
    predict.ipynb에도 같은 처리가 들어가 있음.
    """
    if not cfg_path or not os.path.exists(cfg_path):
        return None
    import tempfile, re
    base = os.path.dirname(os.path.abspath(cfg_path))
    txt = open(cfg_path, encoding="utf-8").read()

    def _abs(m):
        rel = m.group(1).strip()
        cand = os.path.join(base, rel.lstrip("./"))
        return f"model_path: {cand}" if os.path.exists(cand) else m.group(0)

    txt = re.sub(r"model_path:\s*(\./[^\n]+)", _abs, txt)
    fd, out = tempfile.mkstemp(suffix=".yaml"); os.close(fd)
    open(out, "w", encoding="utf-8").write(txt)
    return out


def _letterbox(img, size, color=(114, 114, 114)):
    """비율 유지하며 정사각형(size x size)으로 패딩 — YOLO 입력 전처리."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, scale, dw, dh


def _nms(boxes, scores, iou_thresh=0.45):
    idxs = np.argsort(-scores)
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        idxs = rest[iou < iou_thresh]
    return keep


class Engine:
    def __init__(self, yolo_weights, conf=0.25, imgsz=960, rapid_cfg=None):
        # region_best.pt(torch)를 그대로 ultralytics.YOLO로 돌리면 이미지 한 장이
        # 유난히 크거나 할 때 전처리 오버헤드가 커서 느림(실측: 6장 평균 배율
        # 7.1배, 큰 사진 한 장은 6.16초→0.41초). onnxruntime으로 직접 돌리면
        # RapidOCR(이미 onnx)과 완전히 같은 실행기라 torch 의존성 자체가
        # 없어지고 훨씬 빠르다. 검출 결과(클래스/신뢰도)는 거의 동일함을 확인함.
        import onnxruntime as ort
        self.det_sess = ort.InferenceSession(yolo_weights, providers=["CPUExecutionProvider"])
        self.conf, self.imgsz = conf, imgsz
        from rapidocr_onnxruntime import RapidOCR
        cfg = make_abs_cfg(rapid_cfg)
        self.ocr = RapidOCR(config_path=cfg) if cfg else RapidOCR()

    def detect(self, img):
        canvas, scale, dw, dh = _letterbox(img, self.imgsz)
        arr = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        arr = arr.transpose(2, 0, 1)[None, ...]
        raw = self.det_sess.run(None, {"images": arr})[0]  # (1, 4+n클래스, N)
        preds = raw[0].T  # (N, 4+n클래스): cx,cy,w,h,score...

        out = {v: [] for v in CLS.values()}
        n_cls = len(CLS)
        cls_scores = preds[:, 4:4 + n_cls]
        cls_id = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)
        mask = cls_conf >= self.conf
        preds, cls_id, cls_conf = preds[mask], cls_id[mask], cls_conf[mask]
        if len(preds) == 0:
            return out

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2 - dw) / scale; y1 = (cy - bh / 2 - dh) / scale
        x2 = (cx + bw / 2 - dw) / scale; y2 = (cy + bh / 2 - dh) / scale
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        for c in np.unique(cls_id):
            m = cls_id == c
            keep = _nms(boxes[m], cls_conf[m])
            name = CLS.get(int(c), "?")
            for idx in keep:
                out[name].append((boxes[m][idx].tolist(), float(cls_conf[m][idx])))
        return out

    def read(self, img):
        res, _ = self.ocr(img)
        return " ".join(x[1] for x in (res or []))

    def run_one(self, path, fallback=True):
        img = load_bgr(path)
        if img is None: return dict(NONE_ROW), {"err":"load_fail"}
        H, W = img.shape[:2]
        det = self.detect(img)
        info = {n: len(v) for n, v in det.items()}

        # 관심영역: full 우선, 없으면 date+due 합집합
        rois = []
        if det["full"]:
            rois = [b for b,_ in sorted(det["full"], key=lambda t:-t[1])[:2]]
        elif det["date"] or det["due"]:
            rois = [union_box([b for b,_ in det["date"]+det["due"]])]

        crops = []
        for box in rois:
            x0,y0,x1,y1 = pad_clip(box, W, H)
            c = img[y0:y1, x0:x1]
            if c.size: crops.append(upscale_small(c))

        # 1차: 원본 크롭
        text = " ".join(self.read(c) for c in crops)
        parsed = extract_expiry_fields(text, from_crop=bool(crops))
        info["route"] = "crop" if crops else "none"
        info["text"] = text[:300]

        # 2차: 실패 시 크롭에만 전처리(CLAHE+morph) 재시도
        #      크롭 영역에만 걸면 배경 대비가 같이 올라가는 낭비가 없고 비용도 싸다
        if parsed["final_date"] == "NONE" and crops:
            t2 = " ".join(self.read(enhance(c)) for c in crops)
            p2 = extract_expiry_fields(t2, from_crop=True)
            if p2["final_date"] != "NONE":
                parsed, info["route"] = p2, "crop+pp"
                info["text"] = t2[:300]

        # 3차: 도트프린터 날짜 전용 보정
        if parsed["final_date"] == "NONE" and crops:
            t2b = " ".join(self.read(binarize_dilate(c)) for c in crops)
            p2b = extract_expiry_fields(t2b, from_crop=True)
            if p2b["final_date"] != "NONE":
                parsed, info["route"] = p2b, "crop+dot"
                info["text"] = t2b[:300]

        # 4차: 그래도 실패 시 전체 이미지
        if parsed["final_date"] == "NONE" and fallback:
            h,w = img.shape[:2]; s = 1600/max(h,w)
            full = cv2.resize(img,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA) if s<1 else img
            t3 = self.read(full)
            p3 = extract_expiry_fields(t3)
            if p3["final_date"] == "NONE":
                t3 = self.read(enhance(full))      # 전체 이미지에도 전처리 재시도
                p3 = extract_expiry_fields(t3)
                if p3["final_date"] != "NONE": info["route"] = "full+pp"
            elif p3["final_date"] != "NONE":
                info["route"] = "fallback"
            if p3["final_date"] != "NONE":
                parsed = p3; info["text"] = t3[:300]
        return parsed, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_path", default="submission.csv")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rapid_cfg", default=None)
    ap.add_argument("--ids", default=None, help="특정 image_id만 (CSV)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no_fallback", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    paths = sorted({p for e in IMAGE_EXTS for p in glob.glob(os.path.join(a.input_dir,e))})
    if a.ids:
        want = set(pd.read_csv(a.ids, dtype=str).image_id)
        paths = [p for p in paths if Path(p).stem in want]
    print(f"대상 {len(paths)}장")

    eng = Engine(a.weights, conf=a.conf, rapid_cfg=a.rapid_cfg)
    rows, dbg, t0 = [], [], time.time()
    for i, p in enumerate(paths, 1):
        parsed, info = eng.run_one(p, fallback=not a.no_fallback)
        rows.append({"image_id": Path(p).stem, **{k:parsed[k] for k in ("year","month","day","final_date")}})
        if a.debug: dbg.append({"image_id": Path(p).stem, **info})
        if i % 50 == 0 or i == len(paths):
            el = time.time()-t0
            print(f"  [{i}/{len(paths)}] {el:.1f}초 (장당 {el/i:.3f}초)")
    df = pd.DataFrame(rows, columns=["image_id","year","month","day","final_date"])
    # 제출 노트북과 같은 형식으로 저장함(BOM 없는 utf-8).
    # utf-8-sig로 저장하면 첫 컬럼 이름 앞에 보이지 않는 문자가 붙어서,
    # 이 파일을 그대로 제출했을 때 채점 쪽에서 image_id 컬럼을 못 찾을 수 있음.
    df.to_csv(a.output_path, index=False)
    el = time.time()-t0
    print(f"\n완료 {el:.1f}초 | 장당 {el/max(1,len(paths)):.3f}초 | 500장 환산 {el/max(1,len(paths))*500:.0f}초")
    print(f"응답률 {(df.final_date!='NONE').mean()*100:.1f}%  → {a.output_path}")
    if a.debug and dbg:
        # 디버그 파일은 엑셀에서 한글이 깨지지 않게 utf-8-sig로 저장함.
        # 이건 제출물이 아니라 눈으로 확인하는 용도라 상관없음.
        pd.DataFrame(dbg).to_csv(str(a.output_path).replace(".csv","_debug.csv"),
                                 index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
