# -*- coding: utf-8 -*-
"""submission.csv 채점기.

정답(splits.csv)과 예측(submission.csv)을 image_id 숫자 기준으로 맞춰
- final_date 완전일치 정확도 (주 지표로 가정)
- year / month / day 필드별 정확도
- 오류 유형: 미추출(예측 NONE인데 정답 있음) / 오추출(다른 날짜) / 부분일치
- 난이도 범주(cat)별 정확도
를 출력하고, 틀린 행을 errors.csv 로 저장한다.

사용법:
    python evaluate.py --pred submission.csv --gold splits.csv --split dev
"""
import argparse

import pandas as pd

FIELDS = ["year", "month", "day", "final_date"]


def norm(v):
    v = "NONE" if pd.isna(v) else str(v).strip()
    return v if v else "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="splits.csv")
    ap.add_argument("--split", default="dev", help="dev / holdout / all")
    ap.add_argument("--errors", default="errors.csv")
    a = ap.parse_args()

    gold = pd.read_csv(a.gold, dtype=str)
    gold = gold[gold["split"] != "excluded"]
    if a.split != "all":
        gold = gold[gold["split"] == a.split]
    pred = pd.read_csv(a.pred, dtype=str)
    pred["iid"] = pred["image_id"].astype(int)
    gold["iid"] = gold["image_id"].astype(int)

    m = gold.merge(pred[["iid"] + FIELDS], on="iid", how="left", suffixes=("_g", "_p"))
    n_missing = m["final_date_p"].isna().sum()
    for f in FIELDS:
        m[f + "_g"] = m[f + "_g"].map(norm)
        m[f + "_p"] = m[f + "_p"].map(norm)
        m[f + "_ok"] = m[f + "_g"] == m[f + "_p"]

    n = len(m)
    print(f"채점 대상: {n}장 (split={a.split})" + (f"  ※ 예측 누락 {n_missing}장은 NONE 처리" if n_missing else ""))
    print("-" * 44)
    print(f"final_date 완전일치 : {m.final_date_ok.mean():6.1%}  ({m.final_date_ok.sum()}/{n})")
    for f in ["year", "month", "day"]:
        print(f"{f:<5} 필드 정확도     : {m[f + '_ok'].mean():6.1%}")

    wrong = m[~m.final_date_ok].copy()
    pred_none = wrong["final_date_p"] == "NONE"
    md_ok = wrong["month_ok"] & wrong["day_ok"]
    wrong["error_type"] = "오추출(다른 날짜)"
    wrong.loc[md_ok, "error_type"] = "부분일치(월·일 맞음, 연도 틀림)"
    wrong.loc[pred_none, "error_type"] = "미추출(NONE 출력)"
    print("-" * 44)
    print("오류 유형:")
    for k, v in wrong["error_type"].value_counts().items():
        print(f"  {k:<28} {v:4d}장 ({v / n:5.1%})")

    print("-" * 44)
    print("범주별 완전일치:")
    t = m.groupby("cat")["final_date_ok"].agg(["mean", "size"]).sort_values("mean")
    for c, r in t.iterrows():
        print(f"  {c:<24} {r['mean']:6.1%}  (n={int(r['size'])})")

    cols = ["image_id", "file", "cat", "hint_category", "final_date_g", "final_date_p", "error_type"]
    wrong[cols].sort_values(["error_type", "cat"]).to_csv(a.errors, index=False)
    print(f"\n틀린 {len(wrong)}건 -> {a.errors}")


if __name__ == "__main__":
    main()
