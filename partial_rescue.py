# -*- coding: utf-8 -*-
"""완전한 날짜를 못 찾았을 때 월과 일만이라도 건져냄.

채점이 항목별 부분점수이므로, NONE은 확정 0점이고 추측은 기댓값이 양수다.
다만 아무 숫자나 집으면 '연도까지 맞던 케이스'를 망칠 수 있으므로,
팀 파서 주석에 실측으로 기록된 오탐 패턴은 그대로 차단한다.

차단 대상 (팀 파서 주석의 실측 사례)
  - 주소 지번      : "백삼로 9-2"      → 09/02 오탐
  - 보관 온도      : "실온 보관(1-30℃)" → 01/30 오탐 (℃가 C로 오독되는 것 포함)
  - 영양성분 수치  : "2.8g", "47%"
  - 전화/코드      : 긴 숫자열의 일부
"""
import re

# 월.일 후보 - 구분자는 . 과 / 만 (- 는 주소 지번 오탐이 많아 제외: 팀 실측)
_MD = re.compile(
    r"(?<![\d.])(0?[1-9]|1[0-2])[./](0?[1-9]|[12]\d|3[01])(?![\d.])"
    r"(?!\s*(g|kg|mg|ml|L|%|kcal|℃|도|°|[Cc]\)|원|명|인분))"
)
# 이 근처면 신뢰도 상승
GOOD_KW = ["소비기한","유통기한","까지","유효기간","품질유지기한","EXP","BEST","USE BY"]
# 이 근처면 배제
BAD_KW  = ["로 ","길 ","번지","호실","고객","상담","문의","전화","번호","품목보고",
           "영양","열량","나트륨","탄수화물","단백질","지방","당류","내용량","중량",
           "보관","온도","실온","냉장","냉동"]

def _ctx(text, s, e, w=25):
    return text[max(0,s-w):s], text[e:e+w]

def rescue_md(text: str):
    """(month, day) 또는 None. 신뢰도 순으로 후보를 고른다."""
    if not text:
        return None
    best = None
    for m in _MD.finditer(text):
        mo, d = int(m.group(1)), int(m.group(2))
        before, after = _ctx(text, m.start(), m.end())
        ctx = before + after
        if any(b in ctx for b in BAD_KW):
            continue
        score = 0.0
        if any(g in ctx for g in GOOD_KW):
            score += 2.0
        if "까지" in after[:6]:
            score += 1.5
        # 뒤쪽에 등장할수록 소비기한일 확률이 높다 (제조일이 앞)
        score += m.start() / max(1, len(text)) * 0.5
        if best is None or score > best[0]:
            best = (score, mo, d)
    if best is None:
        return None
    return f"{best[1]:02d}", f"{best[2]:02d}"
