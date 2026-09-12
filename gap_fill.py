# -*- coding: utf-8 -*-
"""팀 파서(date_parser_base.py)가 못 잡는 사각지대를 메우는 계층.
원본 파서 코드는 건드리지 않고, 텍스트 정규화 + 보조 추출로만 보강한다.
실측 실패 사례(YOLO+전처리 이후에도 무응답이던 12건 중 4건) 기반.
"""
import re

# ── ① 앵커 키워드 별칭 ──────────────────────────────────────
# 팀 파서 KW_CONSUME=['소비기한'] 뿐이라 '사용기한'을 못 알아봄.
# 실측: 001972.jpg "사용기한:2023.03.15" → 앵커 미인식으로 버려짐.
# 같은 의미로 흔히 쓰이는 표기라 안전하게 치환한다.
_ALIAS = [
    (re.compile(r"사용기한"), "소비기한"),
    (re.compile(r"품질유지기한"), "소비기한"),
]

# ── ② '제조'가 들어가지만 제조일과 무관한 복합어 ──────────────
# 실측: 001972.jpg "제조번호:20006" → "제조"가 MFG 키워드로 오매칭되어
# 뒤에 나오는 진짜 소비기한 날짜가 "제조일자"로 잘못 분류됨.
# 날짜와 무관한 '제조+명사' 복합어만 선택적으로 무력화한다.
# (제조일/제조일자/제조년월일/"제조 2020.11.13"처럼 날짜 바로 앞의
#  '제조'는 절대 건드리지 않음 - 팀 파서의 정상 동작 경로이므로)
_MFG_FALSE_POS = re.compile(r"제조(번호|원|회사|국|사|공장|업체)")

def _degrade_mfg_false_positives(text: str) -> str:
    return _MFG_FALSE_POS.sub(lambda m: "　" * len(m.group(0)), text)
    # 전각 공백으로 치환 - 글자수/오프셋을 유지해 다른 매칭에 영향 최소화

def apply_aliases(text: str) -> str:
    t = text
    for pat, rep in _ALIAS:
        t = pat.sub(rep, t)
    t = _degrade_mfg_false_positives(t)
    return t


# ── ③ 월+연도만 있고 일자가 없는 경우 ────────────────────────
# 실측: 002217.jpg "MAY 2023 L0148" → 정답 2023-05-NONE인데
# 팀 파서엔 '일자 없는 월+연도' 패턴 자체가 없음.
MONTH_EN = {m: i+1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
_MONTH_ALT = "|".join(MONTH_EN)

_MY_ENGLISH = re.compile(rf"(?<![A-Za-z])({_MONTH_ALT})\s+(\d{{4}})(?!\d)", re.I)
_MY_NUMERIC = re.compile(r"(?<!\d)(\d{4})[.\-/](\d{1,2})(?!\d)(?!\s*[.\-/]\s*\d)")

_DUE_KW = ["소비기한","유통기한","까지","유효기간","품질유지기한","EXP","BEST","USE BY"]
_BAD_KW = ["전화","고객","상담","번호","로트","LOT","바코드","품목","제조번호"]

def _has(text, kws): return any(k.upper() in text.upper() for k in kws)

def try_month_year_only(text: str, require_anchor: bool = True):
    """(year, month) 또는 None.
    require_anchor=True : 전체이미지 OCR용 - 소비기한 등 앵커가 근처에 있어야 채택
    require_anchor=False: YOLO 크롭 텍스트용 - 크롭 자체가 이미 '날짜 영역'이라는
                          위치 정보를 담보하므로 앵커 텍스트 없이도 채택 가능.
                          단 방해패턴(BAD_KW)은 그대로 차단."""
    for pat, kind in [(_MY_ENGLISH, "en"), (_MY_NUMERIC, "num")]:
        for m in pat.finditer(text):
            s, e = max(0, m.start()-25), min(len(text), m.end()+25)
            ctx = text[s:e]
            if _has(ctx, _BAD_KW):
                continue
            if kind == "en":
                mo, y = MONTH_EN[m.group(1).upper()], int(m.group(2))
            else:
                y, mo = int(m.group(1)), int(m.group(2))
            if not (2015 <= y <= 2035 and 1 <= mo <= 12):
                continue
            if require_anchor and not _has(ctx, _DUE_KW):
                continue
            return y, mo
    return None


# ── ④ 구분자 없이 붙은 '일+영문월+2자리연도' ──────────────────
# 실측: 001309.jpg "BEST 1V 30APR21" → 정답 2021-04-30인데
# 팀 파서 _DMY_ENGLISH는 공백 구분 + 4자리연도만 지원해 매칭 안 됨.
_COMPACT_DMY_EN = re.compile(
    rf"(?<![A-Za-z0-9])(\d{{1,2}})({_MONTH_ALT})(\d{{2}})(?![A-Za-z0-9])", re.I)

def try_compact_dmy(text: str):
    for m in _COMPACT_DMY_EN.finditer(text):
        d, mo, yy = int(m.group(1)), MONTH_EN[m.group(2).upper()], int(m.group(3))
        y = 2000 + yy
        if 1 <= d <= 31 and 2015 <= y <= 2035:
            try:
                from datetime import date; date(y, mo, d)
            except ValueError:
                continue
            return y, mo, d
    return None
