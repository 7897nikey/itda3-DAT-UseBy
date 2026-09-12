# -*- coding: utf-8 -*-
"""[A안] 3경로 항목별 투표 (vote)

전처리 3경로를 모두 돌리고 연·월·일을 각각 다수결로 정한다.
dev 297장 측정: 부분점수 0.8058 / 장당 1.93초 (500장 965초 = 제한 2400초의 40%)

왜 항목별 다수결인가
  채점이 연·월·일 항목별 부분점수라서, 세 경로가 "2026-05-18 / 2026-05-NONE / 2026-NONE-18" 처럼
  서로 다르게 실패해도 항목 단위로 합치면 완전한 날짜가 복원된다.
  날짜 통째로 다수결한 경우(0.8272)보다 항목별(0.8350)이 13경로에서도 더 높았다.

동점 처리
  같은 표를 받은 값이 여럿이면 먼저 나온 경로의 값을 쓴다.
  경로 순서가 곧 우선순위이므로 combo_all -> adaptive -> illum 순서를 지킬 것.

사용법
  export ITDA_INPUT_DIR=./val_images ITDA_OUTPUT_PATH=./submission.csv
  python pipeline_vote.py
  # 한국어 인식 모델을 쓰려면: export ITDA_REC=korean_PP-OCRv5_rec_mobile.onnx
"""
import argparse
from collections import Counter

from pipeline_core import FIELDS, VARIANTS, parse_text, run, run_ocr


def vote(engine, region, kind):
    preds = []
    for _, fn in VARIANTS:                      # 3경로 전부 실행
        preds.append(parse_text(run_ocr(engine, fn(region)), kind))

    out = []
    for i in range(len(FIELDS)):                # 연 / 월 / 일 각각 다수결
        vals = [p[i] for p in preds if p[i] != "NONE"]
        out.append(Counter(vals).most_common(1)[0][0] if vals else "NONE")
    return tuple(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rec", default=None, help="인식 모델 onnx (기본: RapidOCR 기본 모델)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pipeline_core
    run(vote, a.input, a.out, a.rec if a.rec is not None else pipeline_core.REC_ONNX, a.limit)
