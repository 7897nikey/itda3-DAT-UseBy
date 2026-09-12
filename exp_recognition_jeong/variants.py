# -*- coding: utf-8 -*-
"""크롭 전처리 변형(variant) 모음. 입력·출력 모두 BGR ndarray.

실험 축: 어떤 전처리가 도트 매트릭스/저대비/기울어짐 크롭의 인식률을 올리는가.
새 변형을 추가하려면 함수를 만들고 VARIANTS 에 등록하면 sweep_ocr.py 가 자동으로 돌린다.
"""
import cv2
import numpy as np

MAX_SIDE = 1600  # 업스케일 상한 (속도 보호)


# ── 기본 도구 ────────────────────────────────────────────────────────────
def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _dark_text(g):
    """글자가 밝은 배경 위 어두운 글자가 되도록 필요하면 반전한다."""
    return 255 - g if g.mean() < 110 else g


def upscale(img, k=2.0):
    h, w = img.shape[:2]
    k = min(k, MAX_SIDE / max(h, w)) if max(h, w) * k > MAX_SIDE else k
    if k <= 1.01:
        return img
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_LANCZOS4)


# ── 변형들 ──────────────────────────────────────────────────────────────
def v_none(img):
    return img


def v_up2(img):
    return upscale(img, 2.0)


def v_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def v_illum(img):
    """조명·반사 정규화: 넓게 흐린 영상으로 나눠 배경 밝기 기울기를 없앤다."""
    g = _gray(img).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=max(9, min(img.shape[:2]) / 8))
    out = np.clip(g / (bg + 1e-3) * 128.0, 0, 255).astype(np.uint8)
    return _to_bgr(out)


def v_sharpen(img):
    g = _gray(img)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    return _to_bgr(cv2.addWeighted(g, 1.7, blur, -0.7, 0))


def v_otsu(img):
    g = _dark_text(_gray(img))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _to_bgr(b)


def v_adaptive(img):
    g = _dark_text(_gray(img))
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return _to_bgr(b)


def v_dotclose(img):
    """도트 매트릭스 특화: 2배 확대 → 이진화 → 끊긴 점을 morphological close 로 잇기.

    주의: close 는 '밝은 값'을 부풀린다. 글자가 어두운 상태로 걸면 점을 잇는 게 아니라
    글자가 지워진다. 그래서 글자가 흰색(255)이 되도록 반전한 뒤 close 하고 되돌린다.
    """
    big = upscale(img, 2.0)
    g = _dark_text(_gray(big))
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
    return _to_bgr(255 - closed)  # 다시 '흰 바탕 + 검은 글자'로


def v_dotclose_auto(img):
    """도트 간격을 추정해 close 커널 크기를 맞춘다 (고정 커널보다 잘 붙는지 확인용)."""
    big = upscale(img, 2.0)
    g = _dark_text(_gray(big))
    inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    if n > 3:
        med = float(np.median(np.sort(stats[1:, cv2.CC_STAT_AREA])))
        ks = int(np.clip(round(np.sqrt(max(med, 1.0)) * 1.6), 3, 9)) | 1  # 홀수로
    else:
        ks = 3
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return _to_bgr(255 - cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k, iterations=1))


def v_deskew(img):
    """이진화 후 글자 덩어리의 기울기를 추정해 회전 (±20도까지만)."""
    g = _dark_text(_gray(img))
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(b)
    if pts is None or len(pts) < 20:
        return img
    ang = cv2.minAreaRect(pts)[-1]
    while ang < -45:   # OpenCV 버전에 따라 [-90,0) 또는 (0,90] 로 나온다
        ang += 90
    while ang > 45:
        ang -= 90
    if abs(ang) < 0.5 or abs(ang) > 20:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def v_combo_dot(img):
    return v_dotclose(v_deskew(img))


def v_combo_lowcontrast(img):
    return v_sharpen(v_illum(v_clahe(img)))


def v_combo_all(img):
    return v_sharpen(v_clahe(upscale(v_deskew(img), 2.0)))


# ── 열화 유형별 특화 변형 (2차) ─────────────────────────────────────────
def v_glare_inpaint(img):
    """빛 반사: 포화된(흰색으로 날아간) 영역을 주변 화소로 메운다."""
    g = _gray(img)
    m = g >= 245
    frac = float(m.mean())
    # 반사가 거의 없으면 그대로. 반대로 너무 넓으면(=글자까지 밝은 경우) 메우면 글자가 사라진다.
    # 실측: 003021 은 포화 영역이 83% 라서 inpaint 가 이미지를 통째로 뭉갰다.
    if frac < 0.004 or frac > 0.15:
        return img
    mask = cv2.dilate(m.astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def v_gamma(img, gamma=0.5):
    """어두운 사진: 감마 보정으로 어두운 쪽을 끌어올린다.

    밝은 사진에 걸면 오히려 대비가 죽는다(실측: 000445). 평균 밝기를 보고 조절한다.
    """
    lum = float(_gray(img).mean())
    if lum > 150:          # 이미 밝으면 반대 방향으로 살짝
        gamma = 1.4
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def v_bilateral_sharp(img):
    """번짐·노이즈: 경계를 살리며 잡음만 눌러준 뒤 선명화."""
    sm = cv2.bilateralFilter(img, 7, 60, 60)
    g = _gray(sm)
    blur = cv2.GaussianBlur(g, (0, 0), 1.5)
    return _to_bgr(cv2.addWeighted(g, 1.8, blur, -0.8, 0))


def v_thin(img):
    """번져서 획이 뭉친 경우: 이진화 후 살짝 깎아 글자 사이를 떼어놓는다."""
    big = upscale(img, 2.0)
    g = _dark_text(_gray(big))
    inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return _to_bgr(255 - cv2.erode(inv, k, iterations=1))


def v_lcn(img):
    """구겨짐·불균일 조명: 국소 평균·표준편차로 정규화해 밝기 얼룩을 없앤다."""
    g = _gray(img).astype(np.float32)
    s = max(9, int(min(img.shape[:2]) / 6)) | 1
    mu = cv2.GaussianBlur(g, (s, s), 0)
    sd = cv2.GaussianBlur((g - mu) ** 2, (s, s), 0) ** 0.5
    out = np.clip((g - mu) / (sd + 8.0) * 48 + 128, 0, 255).astype(np.uint8)
    return _to_bgr(out)


def v_retinex(img):
    """어두움 + 반사 동시 대응: 여러 배율의 조명 성분을 로그 영역에서 제거."""
    g = _gray(img).astype(np.float32) + 1.0
    acc = np.zeros_like(g)
    for sigma in (15, 60, 150):
        acc += np.log(g) - np.log(cv2.GaussianBlur(g, (0, 0), sigma) + 1.0)
    acc /= 3.0
    out = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return _to_bgr(out)


def v_up4_sharp(img):
    """글자 자체가 작은 경우: 4배 확대 후 선명화 (상한 1600px)."""
    return v_sharpen(upscale(img, 4.0))


def v_combo_glare(img):
    return v_clahe(v_glare_inpaint(img))


def v_combo_dark(img):
    return v_sharpen(v_clahe(v_gamma(img)))


def v_combo_smear(img):
    return v_thin(v_bilateral_sharp(img))


VARIANTS = {
    "none": v_none,                      # 기준선(baseline)
    "up2": v_up2,
    "clahe": v_clahe,
    "illum": v_illum,
    "sharpen": v_sharpen,
    "otsu": v_otsu,
    "adaptive": v_adaptive,
    "dotclose": v_dotclose,
    "dotclose_auto": v_dotclose_auto,
    "deskew": v_deskew,
    "combo_dot": v_combo_dot,
    "combo_lowcontrast": v_combo_lowcontrast,
    "combo_all": v_combo_all,
    # 2차: 열화 유형별
    "glare_inpaint": v_glare_inpaint,    # 빛 반사
    "gamma": v_gamma,                    # 어두움
    "bilateral_sharp": v_bilateral_sharp,  # 번짐·노이즈
    "thin": v_thin,                      # 획이 뭉친 번짐
    "lcn": v_lcn,                        # 구겨짐·불균일 조명
    "retinex": v_retinex,                # 어두움 + 반사
    "up4_sharp": v_up4_sharp,            # 작은 글자
    "combo_glare": v_combo_glare,
    "combo_dark": v_combo_dark,
    "combo_smear": v_combo_smear,
}
