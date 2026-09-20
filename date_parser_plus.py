# -*- coding: utf-8 -*-
"""main의 date_parser.py를 그대로 쓰고, 그게 실패한 건에 대해서만 네 겹을 더 시도하는 층.

main 파서를 복사해오지 않고 import로 가져다 씀. 그래야 main 파서가 개선되면
이쪽도 자동으로 따라가고, 두 벌이 따로 놀 일이 없음.

실패한 건에만 다음 단계를 태우는 구조라, 원래 맞던 건은 건드리지 않음.
449장으로 확인했을 때 이 층 전체가 퇴행 0건이었음.

  1) 별칭 정규화 후 main 파서 (사용기한/품질유지기한 -> 소비기한)
  2) 실패하면 OCR 오독 글자를 고쳐서 main 파서 재시도
  3) 실패하면 구분자 없는 압축 표기 (30APR21 같은 것)
  4) 실패하면 구분자 없는 전숫자 날짜 (20260516, 261030)
  5) 실패하면 연-월만이라도 (일자가 아예 안 적힌 포장이 있음)
  6) 그래도 실패하면 월-일만이라도 제출 (채점이 항목별 부분점수라 NONE은 확정 0점)
"""
import os
import sys

# main의 date_parser.py를 가져오기 위한 경로 추가.
# (원래 alt_pipeline_rapidocr/ 하위 폴더에 있을 때는 두 단계 위가 저장소
# 루트였는데, 이 브랜치를 main으로 승격하면서 이 파일 자체가 루트로 옮겨져
# date_parser.py와 같은 폴더에 있게 됨 — 그래서 한 단계만 올라가면 됨)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from date_parser import extract_expiry_fields as _extract, explain as _explain

from ocr_normalize import normalize_ocr_text
from partial_rescue import rescue_md
from gap_fill import (apply_aliases, try_month_year_only, try_compact_dmy,
                      try_compact_numeric, try_space_ymd)

NONE4 = {"year": "NONE", "month": "NONE", "day": "NONE", "final_date": "NONE"}


def _mk(y, mo, d=None):
    if d is None:
        return {"year": f"{y:04d}", "month": f"{mo:02d}", "day": "NONE",
                "final_date": f"{y:04d}-{mo:02d}-NONE"}
    return {"year": f"{y:04d}", "month": f"{mo:02d}", "day": f"{d:02d}",
            "final_date": f"{y:04d}-{mo:02d}-{d:02d}"}


def extract_expiry_fields(text: str, rescue: bool = True, from_crop: bool = False):
    """from_crop은 YOLO가 날짜 영역이라고 이미 위치를 잡아준 텍스트라는 표시임.

    이 경우 4단계에서 "소비기한" 같은 앵커 단어를 요구하지 않음. 위치로 이미
    걸러진 텍스트라 앵커까지 요구하면 놓치는 게 늘어남. 반대로 전체 이미지를
    통째로 읽은 텍스트에는 앵커를 요구함. 포장 전체에는 제조번호나 영양성분
    같은 남의 숫자가 잔뜩 섞여 있어서, 앵커 없이 받으면 오답이 늘어남.
    """
    alias = apply_aliases(text)
    raw = _extract(alias)
    if raw["final_date"] != "NONE":
        return raw

    fixed = normalize_ocr_text(alias)
    if fixed != alias:
        alt = _extract(fixed)
        if alt["final_date"] != "NONE":
            alt["_via"] = "ocr_normalized"
            return alt

    for t in (fixed, alias, text):
        got = try_compact_dmy(t)
        if got:
            y, mo, d = got
            r = _mk(y, mo, d)
            r["_via"] = "gap_fill_compact_dmy"
            return r

    # 구분자 없는 전숫자 날짜(20260516 / 261030). 연-월만 뽑기보다 먼저 시도한다.
    # 온전한 날짜가 나오면 그게 항상 낫기 때문이다.
    for t in (fixed, alias, text):
        got = try_compact_numeric(t, require_anchor=not from_crop)
        if got:
            y, mo, d = got
            r = _mk(y, mo, d)
            r["_via"] = "gap_fill_compact_numeric"
            return r

    # 공백만으로 끊긴 연-월-일(2020 06 03). 연-월만 뽑기 전에 시도한다 —
    # 온전한 날짜가 연-월보다 항상 낫기 때문이다.
    for t in (fixed, alias, text):
        got = try_space_ymd(t, require_anchor=not from_crop)
        if got:
            y, mo, d = got
            r = _mk(y, mo, d)
            r["_via"] = "gap_fill_space_ymd"
            return r

    for t in (fixed, alias, text):
        got = try_month_year_only(t, require_anchor=not from_crop)
        if got:
            y, mo = got
            r = _mk(y, mo)
            r["_via"] = "gap_fill_month_year"
            return r

    if rescue:
        for t in (fixed, alias, text):
            got = rescue_md(t)
            if got:
                mo, d = got
                return {"year": "NONE", "month": mo, "day": d,
                        "final_date": f"NONE-{mo}-{d}", "_via": "partial_rescue"}
    return dict(NONE4)


def explain(text: str):
    return _explain(normalize_ocr_text(apply_aliases(text)))
