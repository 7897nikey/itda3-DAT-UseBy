# -*- coding: utf-8 -*-
"""[B안] 3경로 cascade (실패한 이미지만 다음 경로로)

첫 경로에서 날짜가 하나라도 파싱되면 거기서 멈춘다. 실패분에만 추가 비용이 붙는다.
dev 297장 측정: 부분점수 0.7856 / 장당 0.91초 (500장 455초 = 제한 2400초의 19%)

A안(투표, 0.8058 / 1.93초)과의 차이
  점수는 2.0%p 낮고 시간은 절반 이하. 채점 서버가 우리 측정 환경보다 느릴 경우의 보험이다.
  둘은 같은 3경로를 쓰므로 언제든 바꿔 끼울 수 있다.

주의
  '파싱 성공'의 기준을 "세 항목이 전부 채워짐"이 아니라 "하나라도 채워짐"으로 둔 이유는,
  부분점수 체계에서 NONE 은 0점이라 불완전한 답도 빈칸보다 항상 낫기 때문이다.
  다만 이 때문에 첫 경로의 부분 성공이 뒤 경로의 완전 성공을 가릴 수 있다 —
  그 손실까지 포함한 측정치가 0.7856 이다.

사용법
  export ITDA_INPUT_DIR=./val_images ITDA_OUTPUT_PATH=./submission.csv
  python pipeline_cascade.py
"""
import argparse

from pipeline_core import NONE3, VARIANTS, n_filled, parse_text, run, run_ocr


def cascade(engine, region, kind):
    best = NONE3
    for _, fn in VARIANTS:
        pred = parse_text(run_ocr(engine, fn(region)), kind)
        if n_filled(pred) > n_filled(best):
            best = pred
        if n_filled(best) > 0:                  # 하나라도 채워지면 중단
            break
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rec", default=None, help="인식 모델 onnx (기본: RapidOCR 기본 모델)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=None, help="dev / holdout (검증용). 제출 시에는 지정하지 않음")
    a = ap.parse_args()
    import pipeline_core
    run(cascade, a.input, a.out, a.rec if a.rec is not None else pipeline_core.REC_ONNX,
        a.limit, split=a.split)
