# -*- coding: utf-8 -*-
"""스윕 결과 채점·비교. 변형끼리 차이가 통계적으로 의미 있는지까지 판정한다.

지표
  부분점수   : year/month/day 각각 맞으면 1/3점 (운영진 채점이 항목별 부분점수라는 가정)
  완전일치   : 세 항목 모두 일치
  응답률     : final_date 를 NONE 이 아닌 값으로 낸 비율
  인식적중   : 정답 날짜의 숫자열이 OCR 텍스트에 그대로 있는지 (파서 영향을 뺀 인식 단계 지표)
  시간       : 전처리 + 인식 (장당 평균)

비교
  McNemar 검정 : 같은 이미지에 두 변형을 돌린 대응표본이므로 이걸 써야 한다.
                 "몇 장이 서로 갈렸고, 그 쏠림이 우연일 확률은 얼마인가"
  부트스트랩 CI: 부분점수 차이의 95% 신뢰구간

사용법:
    python analyze_sweep.py                          # sweep/ 전체를 표로
    python analyze_sweep.py --base none__m15         # 기준을 지정해 검정
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

from date_parser import extract_expiry_fields

FIELDS = ["year", "month", "day"]


def gold_digit_patterns(y, m, d):
    """정답 날짜가 이미지에 찍혀 있을 만한 숫자열 후보."""
    pats = set()
    if m != "NONE" and d != "NONE":
        pats |= {m + d, d + m}
        if y != "NONE":
            yy = y[2:]
            pats |= {y + m + d, yy + m + d, d + m + yy, m + d + yy, d + m + y, y + m + d}
    elif m != "NONE" and y != "NONE":
        pats |= {y + m, y[2:] + m, m + y[2:]}
    return pats


def score_file(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--base", default=None, help="기준 변형 (기본: none__m15 또는 첫 번째)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--errors", default=None, help="지정하면 그 변형의 틀린 목록을 csv 로 저장")
    a = ap.parse_args()

    gold = pd.read_csv(a.splits, dtype=str)
    gold = gold[gold["split"] != "excluded"].set_index("file")

    per = {}   # 변형 -> DataFrame(index=file)
    for path in sorted(glob.glob(os.path.join(a.sweep, "*.jsonl"))):
        tag = os.path.basename(path)[:-6]
        recs = []
        for r in score_file(path):
            f = r["file"]
            if f not in gold.index:
                continue
            g = gold.loc[f]
            text = " ".join(e[0] for e in r["ocr"]).strip()
            full = ("소비기한 " + text) if (r["kind"] == "crop" and text) else text
            try:
                p = extract_expiry_fields(full) if full else {k: "NONE" for k in FIELDS + ["final_date"]}
            except Exception:
                p = {k: "NONE" for k in FIELDS + ["final_date"]}
            ok = {f"ok_{k}": (str(g[k]) == str(p[k])) for k in FIELDS}
            digits = re.sub(r"\D", "", text)
            pats = gold_digit_patterns(str(g["year"]), str(g["month"]), str(g["day"]))
            recs.append({"file": f, **ok,
                         "item": sum(ok.values()) / 3.0,
                         "exact": all(ok.values()),
                         "answered": p["final_date"] != "NONE",
                         "rec_hit": any(x in digits for x in pats) if pats else False,
                         "t": r["t_prep"] + r["t_ocr"],
                         "cat": g["cat"], "kind": r["kind"],
                         "pred": p["final_date"], "gold": g["final_date"], "text": text})
        if recs:
            per[tag] = pd.DataFrame(recs).set_index("file")

    if not per:
        raise SystemExit(f"{a.sweep}/ 에 결과가 없습니다. 먼저 sweep_ocr.py 를 돌리세요.")

    base = a.base or ("none__m15" if "none__m15" in per else list(per)[0])
    if base not in per:
        raise SystemExit(f"기준 {base} 없음. 있는 것: {list(per)}")
    common = set.intersection(*(set(df.index) for df in per.values()))
    print(f"공통 이미지 {len(common)}장 기준 / 기준선(baseline) = {base}\n")

    rng = np.random.default_rng(0)
    idx = sorted(common)
    rows = []
    b = per[base].loc[idx]
    for tag, df in per.items():
        d = df.loc[idx]
        diff = d["item"].values - b["item"].values
        if a.boot and tag != base:
            bs = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(a.boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ci = f"{lo * 100:+.1f} ~ {hi * 100:+.1f}"
        else:
            ci = "-"
        # McNemar (완전일치 기준)
        bb = int(((~b["exact"]) & d["exact"]).sum())   # 기준 틀림 -> 이 변형 맞음
        cc = int((b["exact"] & (~d["exact"])).sum())   # 그 반대
        n = bb + cc
        if tag == base or n == 0:
            p = float("nan")
        else:
            k = min(bb, cc)
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        rows.append({"변형": tag, "부분점수": d["item"].mean(), "완전일치": d["exact"].mean(),
                     "응답률": d["answered"].mean(), "인식적중": d["rec_hit"].mean(),
                     "장당초": d["t"].mean(), "개선/악화": f"{bb}/{cc}", "p": p, "차이95%CI(%p)": ci})

    t = pd.DataFrame(rows).sort_values("부분점수", ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}" if v < 1.5 else f"{v:.2f}"))
    print("\n※ 개선/악화 = 기준선 대비 새로 맞힌 장수 / 새로 틀린 장수, p 는 McNemar 양측 검정")
    print("  p < 0.05 면 그 차이는 우연으로 보기 어렵다는 뜻입니다.")

    best = t.iloc[0]["변형"]
    d = per[best].loc[idx]
    print(f"\n[{best}] 난이도 범주별 부분점수")
    print(d.groupby("cat")["item"].agg(["mean", "size"]).sort_values("mean").round(3).to_string())
    nores = d[~d["answered"]]
    print(f"\n무응답 {len(nores)}장 중 정답 자체가 NONE 인 것: "
          f"{int((nores['gold'] == 'NONE').sum())}장  (개선 여지가 아닌 것)")
    print("무응답인데 정답이 있는 이미지 (상위 10개):",
          ", ".join(nores[nores["gold"] != "NONE"].index[:10]))

    if a.errors:
        tag = a.errors if a.errors in per else best
        e = per[tag].loc[idx]
        e[~e["exact"]][["cat", "kind", "gold", "pred", "text"]].to_csv(f"errors_{tag}.csv")
        print(f"\n틀린 목록 -> errors_{tag}.csv")


if __name__ == "__main__":
    main()
