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

# ── ②-b 시각 표기 무력화 ────────────────────────────────────
# 실측: 000833 "HS 19:06 2026:04.27" → 19:06을 날짜로 읽어 2026-06-19가 나왔음.
#       001231 "21:05.05 15:43" → 15:43을 날짜로 읽음.
# 포장에는 날짜 옆에 인쇄 시각이 같이 찍히는 경우가 많다.
#
# 다만 시각과 날짜가 생김새로는 구분이 안 된다. 21:05.05는 2021년 5월 5일이다.
# 그래서 뒤에 아무것도 안 붙은 '맨' 시각만 지운다. 뒤에 .05가 붙어 있으면
# 날짜의 일부이므로 건드리지 않는다. 앞이 숫자면 2026:04 같은 연-월이므로 역시 건드리지 않는다.
# 앞에 마침표·하이픈·슬래시가 오면 날짜의 뒷부분이다(실측: 000980 "2021.06:14").
# 그것까지 지웠다가 맞던 날짜를 NONE으로 만들었으므로 구분자 뒤는 건드리지 않는다.
_TIME_TOKEN = re.compile(r"(?<![\d.\-/:])(\d{1,2}):(\d{2})(?![\d.\-/:])")

def _degrade_times(text: str) -> str:
    return _TIME_TOKEN.sub(lambda m: " " * len(m.group(0)), text)


def apply_aliases(text: str) -> str:
    t = text
    for pat, rep in _ALIAS:
        t = pat.sub(rep, t)
    t = _degrade_mfg_false_positives(t)
    t = _degrade_times(t)
    return t


# ── ③ 월+연도만 있고 일자가 없는 경우 ────────────────────────
# 실측: 002217.jpg "MAY 2023 L0148" → 정답 2023-05-NONE인데
# 팀 파서엔 '일자 없는 월+연도' 패턴 자체가 없음.
MONTH_EN = {m: i+1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
_MONTH_ALT = "|".join(MONTH_EN)

_MY_ENGLISH = re.compile(rf"(?<![A-Za-z])({_MONTH_ALT})\s+(\d{{4}})(?!\d)", re.I)
# 구분자에 콜론을 추가함. OCR이 '2027.10'의 마침표를 콜론으로 자주 흘림
# (실측: cust_0069 '2027:10', cust_0098 '2028:11').
# 앞이 2015~2035 범위의 4자리 연도라 '10:30' 같은 시각과 섞일 일이 없음.
_MY_NUMERIC = re.compile(r"(?<!\d)(\d{4})[.\-/:](\d{1,2})(?!\d)(?!\s*[.\-/:]\s*\d)")
# 월이 앞에 오는 표기(EXP.DATE 09/2026). 뒤가 4자리 연도라 일자와 헷갈릴 일이 없다.
_MY_NUMERIC_REV = re.compile(r"(?<!\d)(\d{1,2})[.\-/](\d{4})(?!\d)")

_DUE_KW = ["소비기한","유통기한","까지","유효기간","품질유지기한","EXP","BEST","USE BY"]
_BAD_KW = ["전화","고객","상담","번호","로트","LOT","바코드","품목","제조번호"]

def _has(text, kws): return any(k.upper() in text.upper() for k in kws)

def try_month_year_only(text: str, require_anchor: bool = True):
    """(year, month) 또는 None.
    require_anchor=True : 전체이미지 OCR용 - 소비기한 등 앵커가 근처에 있어야 채택
    require_anchor=False: YOLO 크롭 텍스트용 - 크롭 자체가 이미 '날짜 영역'이라는
                          위치 정보를 담보하므로 앵커 텍스트 없이도 채택 가능.
                          단 방해패턴(BAD_KW)은 그대로 차단."""
    for pat, kind in [(_MY_ENGLISH, "en"), (_MY_NUMERIC, "num"), (_MY_NUMERIC_REV, "rev")]:
        for m in pat.finditer(text):
            s, e = max(0, m.start()-25), min(len(text), m.end()+25)
            ctx = text[s:e]
            if _has(ctx, _BAD_KW):
                continue
            if kind == "en":
                mo, y = MONTH_EN[m.group(1).upper()], int(m.group(2))
            elif kind == "rev":
                mo, y = int(m.group(1)), int(m.group(2))
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


# ── ⑤ 구분자 없이 붙은 전(全)숫자 날짜 ────────────────────────
# 실측: cust_0076 "U0B01366 20260516 04124150" → 정답 2026-05-16,
#       cust_0082 "#23미디엄 BB8 20290224까지" → 정답 2029-02-24,
#       cust_0093 "261030 95" → 정답 2026-10-30.
# OCR은 숫자를 정확히 읽었는데 파서에 이 형태가 아예 없어서 버려지던 것들임.
#
# 위험한 규칙이다. 제조번호·로트번호도 같은 자릿수라 구분이 안 된다.
# 그래서 세 겹으로 막는다.
#   - 날짜로 성립하지 않는 것은 버린다 (연 2015~2035, 실재하는 월·일)
#   - 방해 키워드가 주변에 있으면 버린다 (제조번호, LOT, 바코드…)
#   - 8자리를 6자리보다 먼저 본다. 6자리는 로트번호와 겹칠 확률이 훨씬 높다
_COMPACT_NUM8 = re.compile(r"(?<!\d)(\d{8})(?!\d)")
_COMPACT_NUM6 = re.compile(r"(?<!\d)(\d{6})(?!\d)")

def _valid_ymd(y, mo, d):
    if not (2015 <= y <= 2035 and 1 <= mo <= 12 and 1 <= d <= 31):
        return False
    try:
        from datetime import date; date(y, mo, d)
    except ValueError:
        return False
    return True

def try_compact_numeric(text: str, require_anchor: bool = True):
    """구분자 없는 YYYYMMDD/YYMMDD에서 (year, month, day)를 뽑는다.

    require_anchor는 다른 보강 단계와 같은 뜻이다. 전체 이미지를 통째로 읽은
    텍스트에는 남의 숫자가 잔뜩 섞이므로 앵커를 요구하고, YOLO 크롭 텍스트에는
    위치가 이미 보증돼 있으므로 요구하지 않는다.
    """
    for pat, width in ((_COMPACT_NUM8, 8), (_COMPACT_NUM6, 6)):
        for m in pat.finditer(text):
            tok = m.group(1)
            s, e = max(0, m.start()-25), min(len(text), m.end()+25)
            ctx = text[s:e]
            if _has(ctx, _BAD_KW):
                continue
            if width == 8:
                y, mo, d = int(tok[:4]), int(tok[4:6]), int(tok[6:8])
            else:
                y, mo, d = 2000 + int(tok[:2]), int(tok[2:4]), int(tok[4:6])
            if not _valid_ymd(y, mo, d):
                continue
            if require_anchor and not _has(ctx, _DUE_KW):
                continue
            return y, mo, d
    return None
