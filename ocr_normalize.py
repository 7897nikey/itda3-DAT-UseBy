# -*- coding: utf-8 -*-
"""OCR이 헷갈린 글자를 되돌림. 단, 날짜처럼 생긴 토막 안에서만 건드림.
일반 한글/영문 문장은 건드리지 않아야 date_parser의 키워드 판정이 깨지지 않는다.

근거(실측): RapidOCR/EasyOCR이 흔히 내는 오독
  2026.09:21  (점→콜론)   2026 .C9 21 (0→C)   그5.10.고2까지 (2→그/고)
  202B.01.2   (8→B)       25 1O 22    (0→O)
"""
import re, unicodedata

CONFUSE = {
    "O":"0","o":"0","D":"0","Q":"0","C":"0","U":"0","()":"0",
    "l":"1","I":"1","i":"1","|":"1","!":"1","]":"1","[":"1",
    "Z":"2","z":"2","그":"2","고":"2",
    "E":"3","S":"5","s":"5","B":"8","b":"6","G":"6","T":"7","A":"4","q":"9","g":"9",
}
_CH = "".join(re.escape(c) for c in CONFUSE if len(c) == 1)
# 날짜 후보 토막: 숫자/혼동문자 1~4개가 구분자로 2~3번 이어지는 덩어리
_DATEISH = re.compile(
    rf"(?<![A-Za-z가-힣])([0-9{_CH}]{{1,4}}(?:[.\-/:,~\s]+[0-9{_CH}]{{1,4}}){{1,2}})(?![A-Za-z])"
)

def _fix(m):
    return "".join(CONFUSE.get(ch, ch) for ch in m.group(1))

def normalize_ocr_text(t: str) -> str:
    """혼동문자 보정 + 콜론/물결을 점으로 통일."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = _DATEISH.sub(_fix, t)
    # 숫자 사이의 콜론은 날짜 구분자로 오독된 것으로 본다 (시각 표기는 두 자리:두 자리)
    t = re.sub(r"(?<=\d)[:~](?=\d{1,2}(?!\d))", ".", t)
    return t
