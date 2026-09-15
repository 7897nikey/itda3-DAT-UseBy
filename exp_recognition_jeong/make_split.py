# -*- coding: utf-8 -*-
"""라벨 CSV -> dev / holdout 분할 (splits.csv).

- 같은 제품 연속 촬영(인접 번호) 누수를 막기 위해 번호 차이 <= GROUP_GAP 인 이미지들을 한 그룹으로 묶어
  그룹 단위로 분할한다.
- hint_category 의 대표 범주 비율이 dev/holdout 에서 비슷하도록 층화(stratify)한다.
- 이미지 파일명은 000123.jpg / 3348.jpeg 처럼 제각각이라 숫자(int)로 매칭한다.

사용법:
    python make_split.py --labels itda_labeling_merged.csv --images <사진폴더> --out splits.csv
"""
import argparse
import os
import random
from collections import Counter, defaultdict

import pandas as pd

GROUP_GAP = 2
HOLDOUT_RATIO = 0.25
SEED = 42

# 대표 범주 우선순위: 어려운/희소한 범주를 먼저 잡는다
PRIORITY = ["5b_판독불가_EXP각인추정", "5_판독불가_진짜", "2b_부터까지쌍", "2_다중날짜",
            "6_연도부재", "3_상대표기", "4_참조표기", "1_유통기한만"]


def primary_cat(hint):
    if not isinstance(hint, str) or not hint.strip():
        return "0_일반"
    tags = hint.split(";")
    for p in PRIORITY:
        if p in tags:
            return p
    return tags[0]


def index_images(img_dir):
    m = {}
    for f in os.listdir(img_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".jpg", ".jpeg", ".png") and stem.isdigit():
            m[int(stem)] = f
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="splits.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.labels, dtype=str)
    df["iid"] = df["image_id"].astype(int)
    imgs = index_images(a.images)
    df["file"] = df["iid"].map(imgs)
    missing = df[df["file"].isna()]
    if len(missing):
        print(f"[WARN] 폴더에 없는 라벨 이미지 {len(missing)}건: {missing.image_id.tolist()[:10]}")
    df["cat"] = df["hint_category"].map(primary_cat)

    # 인접 번호 그룹핑
    df = df.sort_values("iid").reset_index(drop=True)
    gid, prev = -1, None
    groups = []
    for i in df["iid"]:
        if prev is None or i - prev > GROUP_GAP:
            gid += 1
        groups.append(gid)
        prev = i
    df["group"] = groups

    # 그룹 대표 범주 = 그룹 내 최빈 범주
    gcat = df.groupby("group")["cat"].agg(lambda s: Counter(s).most_common(1)[0][0])
    gsize = df.groupby("group").size()
    by_cat = defaultdict(list)
    for g, c in gcat.items():
        by_cat[c].append(g)

    rng = random.Random(SEED)
    holdout_groups = set()
    for c, gs in by_cat.items():
        rng.shuffle(gs)
        n_cat = sum(gsize[g] for g in gs)
        target = round(n_cat * HOLDOUT_RATIO)
        got = 0
        for g in gs:
            if got >= target:
                break
            holdout_groups.add(g)
            got += gsize[g]

    df["split"] = df["group"].map(lambda g: "holdout" if g in holdout_groups else "dev")
    df.loc[df["status"] != "완료", "split"] = "excluded"  # 보류 라벨은 채점 제외

    df[["image_id", "file", "split", "group", "cat", "batch", "hint_category",
        "year", "month", "day", "final_date", "status"]].to_csv(a.out, index=False)

    print(df["split"].value_counts().to_string())
    print("\n범주별 dev/holdout:")
    print(pd.crosstab(df["cat"], df["split"]).to_string())
    multi = (gsize > 1).sum()
    print(f"\n그룹 {gsize.size}개 (2장 이상 묶인 그룹 {multi}개) -> {a.out}")


if __name__ == "__main__":
    main()
