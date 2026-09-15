# -*- coding: utf-8 -*-
"""오류를 원인별로 쪼갠다: 검출 실패 / 인식 실패 / 파서 실패.

인식적중(정답 숫자열이 OCR 텍스트에 있음)과 최종 정답 여부를 교차하면 병목이 어디인지 나온다.
  - 텍스트에 정답이 있는데 틀림  -> 파서 문제
  - 텍스트에 정답이 없음        -> 인식 문제 (또는 크롭이 잘못됨)
  - 텍스트 자체가 비어 있음      -> 인식 완전 실패
  - 크롭 실패(kind=full)        -> 검출 문제

사용법:
    python gap_analysis.py --tag clahe__m15
"""
import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

from date_parser import extract_expiry_fields
from analyze_sweep import gold_digit_patterns

FIELDS = ["year", "month", "day"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep")
    ap.add_argument("--tag", default="clahe__m15")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    g = pd.read_csv(a.splits, dtype=str)
    g = g[g["split"] != "excluded"].set_index("file")

    rows = []
    with open(os.path.join(a.sweep, a.tag + ".jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            f = r["file"]
            if f not in g.index:
                continue
            gold = g.loc[f]
            text = " ".join(e[0] for e in r["ocr"]).strip()
            full = ("소비기한 " + text) if (r["kind"] == "crop" and text) else text
            try:
                p = extract_expiry_fields(full) if full else {k: "NONE" for k in FIELDS + ["final_date"]}
            except Exception:
                p = {k: "NONE" for k in FIELDS + ["final_date"]}
            ok = all(str(gold[k]) == str(p[k]) for k in FIELDS)
            digits = re.sub(r"\D", "", text)
            pats = gold_digit_patterns(str(gold["year"]), str(gold["month"]), str(gold["day"]))
            hit = any(x in digits for x in pats) if pats else False

            if r["kind"] == "full":
                cause = "검출실패(크롭 못함)"
            elif not text:
                cause = "인식 완전실패(글자 0개)"
            elif ok:
                cause = "정답"
            elif hit:
                cause = "파서 실패(정답 숫자는 읽힘)"
            else:
                cause = "인식 오류(정답 숫자 없음)"
            rows.append({"file": f, "cat": gold["cat"], "cause": cause, "kind": r["kind"],
                         "gold": gold["final_date"], "pred": p["final_date"],
                         "n_digits": len(digits), "text": text[:90]})

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"[{a.tag}] {n}장\n")
    print("원인별 분해")
    for k, v in df["cause"].value_counts().items():
        print(f"  {k:<28} {v:4d}장 ({v / n:5.1%})")

    print("\n검출 실패(크롭 못함) 중 정답 여부:")
    full = df[df["kind"] == "full"]
    print(f"  {len(full)}장 중 맞힌 것 {int((full['cause'] == '정답').sum())}장")

    par = df[df["cause"] == "파서 실패(정답 숫자는 읽힘)"]
    print(f"\n파서 실패 {len(par)}장 — 어떤 답을 냈나")
    print(f"  NONE 출력: {int((par['pred'] == 'NONE').sum())}장 / 다른 날짜 출력: {int((par['pred'] != 'NONE').sum())}장")
    print("  예시 (정답 | 출력 | OCR 텍스트):")
    for r in par.head(12).itertuples():
        print(f"   {r.gold:<12} | {str(r.pred):<12} | {r.text}")

    rec = df[df["cause"] == "인식 오류(정답 숫자 없음)"]
    print(f"\n인식 오류 {len(rec)}장 — 범주 분포: {dict(Counter(rec['cat']).most_common())}")
    print("  예시 (정답 | OCR 텍스트):")
    for r in rec.head(8).itertuples():
        print(f"   {r.gold:<12} | {r.text}")

    print("\n범주별 원인 분포")
    print(pd.crosstab(df["cat"], df["cause"]).to_string())

    out = a.out or f"gap_{a.tag}.csv"
    df.to_csv(out, index=False)
    print(f"\n전체 -> {out}")


if __name__ == "__main__":
    main()
