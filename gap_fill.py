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

# 10/1: "OCT.2021"(001954)처럼 점·슬래시로 붙는 표기도 받는다.
_MY_ENGLISH = re.compile(rf"(?<![A-Za-z])({_MONTH_ALT})(?:\s*[.,/\-]\s*|\s+)(\d{{4}})(?!\d)", re.I)
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

# 연-월-일 사이가 전부 공백인 표기 (예: "2020 06 03", "2021 12 02").
# date_parser 쪽에도 _YMD_SPACE_ONLY 가 있지만 그건 기한류 키워드 ±30자
# 안에서만 도는데, 실측 실패 사례는 크롭 텍스트가 날짜 하나뿐이라 앵커로 삼을
# 단어가 아예 없거나(001554 "2020 06 03") OCR이 키워드를 뭉개버린 경우가
# 많았다(001754 "EET EF0FE 2021 12 05" — BEST BEFORE 가 훼손됨).
#
# 그래서 여기서는 크롭 텍스트에 한해(require_anchor=False) 앵커 없이도 받는다.
# YOLO 가 이미 "이 영역이 날짜다"라고 위치를 보증해 준 텍스트이므로,
# 다른 보강 단계(try_compact_numeric 등)와 같은 근거로 완화한다.
#
# 완전 이미지 텍스트(require_anchor=True)에는 앵커를 계속 요구한다 —
# 영양성분표나 주소에 "2021 12 05" 꼴 숫자가 섞여 나올 수 있기 때문이다.
_YMD_SPACE3 = re.compile(r"(?<![\d.\-/:])(\d{4})\s+(\d{1,2})\s+(\d{1,2})(?![\d.\-/:])")


def try_space_ymd(text: str, require_anchor: bool = True):
    """공백만으로 끊긴 연-월-일에서 (year, month, day)를 뽑는다."""
    for m in _YMD_SPACE3.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not _valid_ymd(y, mo, d):
            continue
        s, e = max(0, m.start() - 25), min(len(text), m.end() + 25)
        ctx = text[s:e]
        if _has(ctx, _BAD_KW):
            continue
        if require_anchor and not _has(ctx, _DUE_KW):
            continue
        return y, mo, d
    return None


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


# ── ⑥ 공백만으로 끊긴 2자리 연도 날짜 (10/1) ───────────────────
# 실측(9/23 D+E 실패 분석): 003343 "30 12 23", 001824 "Best Before 07 09 21",
# 002016 "19 102022". 9/17부터 권고만 되고 미구현이던 것.
# 2자리 연도 표기의 연/일 자리 판정은 date_parser 의 기존 규칙(안내문 → 연도 사전)을
# 그대로 따른다. 구두점이 없어 오탐 위험이 크므로 크롭 텍스트에서만 쓰고
# (require_anchor=False 경로), 방해 키워드가 있으면 버린다.
# 세 자리 모두 2자리여야 하고, 앞에 구두점이 (공백 건너) 붙어 있으면 다른 날짜의
# 꼬리다(000260 "04. 01 2 27"에서 "01 2 27"을 잡던 오탐).
_SPACE_2Y = re.compile(r"(?<![\d.\-/:])(?<![\d.\-/:]\s)(\d{2})\s+(\d{2})\s+(\d{2})(?![\d.\-/:%])")
_SPACE_D_MY4 = re.compile(r"(?<![\d.\-/:])(\d{1,2})\s+(\d{2})(\d{4})(?![\d.\-/:])")


def try_space_2y(text: str, require_anchor: bool = True):
    import date_parser as _dp
    hint = _dp._detect_order_hint(text)
    for m in _SPACE_D_MY4.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not _valid_ymd(y, mo, d):
            continue
        s, e = max(0, m.start() - 25), min(len(text), m.end() + 25)
        if _has(text[s:e], _BAD_KW) or (require_anchor and not _has(text[s:e], _DUE_KW)):
            continue
        return y, mo, d
    for m in _SPACE_2Y.finditer(text):
        a, mo, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        s, e = max(0, m.start() - 25), min(len(text), m.end() + 25)
        if _has(text[s:e], _BAD_KW) or (require_anchor and not _has(text[s:e], _DUE_KW)):
            continue
        ymd = (2000 + a, mo, c); dmy = (2000 + c, mo, a)
        ymd_ok, dmy_ok = _valid_ymd(*ymd), _valid_ymd(*dmy)
        if ymd_ok and dmy_ok:
            if hint == "dmy" or (hint != "ymd" and _dp._prefer_dmy_by_year_prior(ymd[0], dmy[0])):
                return dmy
            return ymd
        if ymd_ok:
            return ymd
        if dmy_ok:
            return dmy
    return None


# ── ⑦ 기한 표지 바로 뒤의 월-연 (10/1) ───────────────────────
# 실측: 002063 "LOT 502337-04 EXP 02/2023" — 근처 LOT 때문에 ③에서 버려짐.
#       002753 "EXP:01.2022 G.1.207" — 혼동문자 보정이 "2022 6.1"을 날짜로 만들어 먼저 이김.
#       002054 "Best before end: 0g/r 12 2021" — 공백 구분.
#       003244 "Best before end / ... fin de: 07-21" — "END/FIN" 표지가 있으면 월-연(2자리).
# 표지가 바로 앞(15자 안)에 있을 때만 받으므로 ③의 방해 키워드 검사는 건너뛴다.
_STRONG_DUE = re.compile(r"(?:EXP|BEST\s*BEFORE(?:\s*END)?|BBE|USE\s*BY|소비기한|유통기한)", re.I)
_MY_STRONG = re.compile(r"(?<![\d.\-/])(\d{1,2})(?:\s*[.\-/]\s*|\s+)(\d{4})(?![\d.\-/:])")
_END_HINT = re.compile(r"BEFORE\s*END|\bFIN\b|FIN\s*DE|말일", re.I)
_MY2_END = re.compile(r"(?<![\d.\-/:])(\d{2})\s*[.\-/]\s*(\d{2})(?![\d.\-/:])")


def try_month_year_strong(text: str):
    for m in _MY_STRONG.finditer(text):
        mo, y = int(m.group(1)), int(m.group(2))
        pre = text[max(0, m.start() - 25):m.start()]
        anchors = list(_STRONG_DUE.finditer(pre))
        # 표지와 날짜 사이에 긴 숫자가 끼어 있으면 그 숫자의 표지다
        if 1 <= mo <= 12 and 2015 <= y <= 2035 and anchors and not re.search(r"\d{3}", pre[anchors[-1].end():]):
            return y, mo
    hint = _END_HINT.search(text)
    if hint:
        for m in _MY2_END.finditer(text, hint.end()):
            mo, yy = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 15 <= yy <= 35 and m.start() - hint.end() <= 60:
                return 2000 + yy, mo
    return None
