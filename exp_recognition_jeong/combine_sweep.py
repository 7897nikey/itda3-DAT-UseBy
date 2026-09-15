# -*- coding: utf-8 -*-
"""여러 전처리를 '함께' 썼을 때 얼마나 오르는지 계산한다 (추가 OCR 실행 없이 sweep 결과만 사용).

답하는 질문
  1) 상한(oracle) : 이미지마다 가장 잘 맞는 전처리를 골랐다면 몇 점인가 → 라우팅/TTA 의 천장
  2) 실현 가능선  : 라우터 없이 '여러 개 돌리고 결과로 고르기'만 해도 몇 점인가
        - cascade : 정해진 순서로 돌리다가 날짜가 파싱되면 멈춤 (실패분에만 비용)
        - vote    : 파싱된 (연,월,일) 중 다수결 (동점이면 OCR 신뢰도 합)
        - vote_field : 연/월/일을 각각 다수결 (부분점수 체계에 유리)
  3) 몇 개면 충분한가 : 상한을 가장 빨리 끌어올리는 조합을 탐욕적으로 고름
  4) 각 변형이 혼자만 살려내는 이미지 수 (고유 기여도)

사용법:
    python combine_sweep.py                       # sweep/ 전체
    python combine_sweep.py --order none__m15,clahe__m15,combo_dot__m15
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from date_parser import extract_expiry_fields
from normalize_retry import RULES as NORM_RULES, n_filled

FIELDS = ["year", "month", "day"]
NONE3 = ("NONE", "NONE", "NONE")


def load(sweep, normalize=False):
    """normalize=True 면 텍스트 정규화 규칙(normalize_retry.RULES)을 적용한 뒤 파싱한다."""
    data = {}
    for path in sorted(glob.glob(os.path.join(sweep, "*.jsonl"))):
        tag = os.path.basename(path)[:-6]
        rec = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = " ".join(e[0] for e in r["ocr"]).strip()
                full = ("소비기한 " + text) if (r["kind"] == "crop" and text) else text
                def _parse(s):
                    try:
                        return extract_expiry_fields(s) if s.strip() else {k: "NONE" for k in FIELDS}
                    except Exception:
                        return {k: "NONE" for k in FIELDS}
                p = _parse(full)
                if normalize:
                    t = text
                    for _, fn in NORM_RULES:
                        t = fn(t)
                    nfull = ("소비기한 " + t) if (r["kind"] == "crop" and t.strip()) else t
                    q = _parse(nfull)
                    if n_filled(tuple(str(q[k]) for k in FIELDS)) >= n_filled(tuple(str(p[k]) for k in FIELDS)):
                        p = q
                confs = [e[1] for e in r["ocr"] if e[1] is not None]
                rec[r["file"]] = {"pred": tuple(str(p[k]) for k in FIELDS),
                                  "conf": float(np.mean(confs)) if confs else 0.0,
                                  "t": r["t_prep"] + r["t_ocr"]}
        if rec:
            data[tag] = rec
    return data


def item(pred, gold):
    return sum(p == g for p, g in zip(pred, gold)) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--normalize", action="store_true", help="텍스트 정규화 규칙을 적용하고 계산")
    ap.add_argument("--order", default=None, help="cascade 순서 (쉼표). 기본: 단독 점수 높은 순")
    a = ap.parse_args()

    data = load(a.sweep, a.normalize)
    if not data:
        raise SystemExit("sweep 결과가 없습니다.")
    g = pd.read_csv(a.splits, dtype=str)
    g = g[g["split"] != "excluded"].set_index("file")
    files = sorted(set.intersection(*(set(d) for d in data.values())) & set(g.index))
    gold = {f: tuple(str(g.loc[f][k]) for k in FIELDS) for f in files}
    print(f"공통 {len(files)}장 / 변형 {len(data)}개\n")

    # 단독 성능
    single = {tag: np.mean([item(d[f]["pred"], gold[f]) for f in files]) for tag, d in data.items()}
    order = ([o.strip() for o in a.order.split(",")] if a.order
             else [t for t, _ in sorted(single.items(), key=lambda kv: -kv[1])])
    tmean = {tag: np.mean([data[tag][f]["t"] for f in files]) for tag in data}

    print("단독 성능 (부분점수 / 장당초)")
    for t in order:
        print(f"  {t:<26} {single[t]:.4f}  {tmean[t]:.3f}s")

    # 상한(oracle)
    orc = np.mean([max(item(data[t][f]["pred"], gold[f]) for t in data) for f in files])
    orc_exact = np.mean([max(float(data[t][f]["pred"] == gold[f]) for t in data) for f in files])
    best_single = max(single.values())
    print(f"\n상한(oracle, 이미지마다 최선 선택): 부분점수 {orc:.4f} / 완전일치 {orc_exact:.4f}")
    print(f"  단독 최고 대비 여유: {(orc - best_single) * 100:+.1f}%p  <- 라우팅·TTA 로 노릴 수 있는 최대치")

    # cascade: 순서대로 돌다가 파싱되면 중단
    def cascade(seq):
        sc, tt = [], []
        for f in files:
            used = 0.0
            pick = NONE3
            for t in seq:
                used += data[t][f]["t"]
                if data[t][f]["pred"] != NONE3:
                    pick = data[t][f]["pred"]
                    break
            sc.append(item(pick, gold[f]))
            tt.append(used)
        return np.mean(sc), np.mean(tt)

    # vote: 전부 돌리고 다수결
    def vote(seq, per_field=False):
        sc = []
        tt = sum(tmean[t] for t in seq)
        for f in files:
            preds = [(data[t][f]["pred"], data[t][f]["conf"]) for t in seq]
            valid = [(p, c) for p, c in preds if p != NONE3]
            if not valid:
                sc.append(item(NONE3, gold[f]))
                continue
            if per_field:
                out = []
                for i in range(3):
                    vals = [p[i] for p, _ in valid if p[i] != "NONE"]
                    out.append(Counter(vals).most_common(1)[0][0] if vals else "NONE")
                sc.append(item(tuple(out), gold[f]))
            else:
                cnt = Counter(p for p, _ in valid)
                top = max(cnt.values())
                tied = [p for p, n in cnt.items() if n == top]
                if len(tied) == 1:
                    sc.append(item(tied[0], gold[f]))
                else:
                    conf = {p: sum(c for q, c in valid if q == p) for p in tied}
                    sc.append(item(max(conf, key=conf.get), gold[f]))
        return np.mean(sc), tt

    print("\n조합 전략")
    s, t = cascade(order)
    print(f"  cascade(전부, 파싱되면 중단)  부분점수 {s:.4f}  장당 {t:.3f}s")
    s3, t3 = cascade(order[:3])
    print(f"  cascade(상위 3개)             부분점수 {s3:.4f}  장당 {t3:.3f}s")
    s, t = vote(order)
    print(f"  vote(전부, 날짜 단위 다수결)   부분점수 {s:.4f}  장당 {t:.3f}s")
    s, t = vote(order, per_field=True)
    print(f"  vote(전부, 항목별 다수결)      부분점수 {s:.4f}  장당 {t:.3f}s")
    s, t = vote(order[:3], per_field=True)
    print(f"  vote(상위 3개, 항목별)         부분점수 {s:.4f}  장당 {t:.3f}s")

    # 탐욕적 조합: 상한을 가장 빨리 올리는 순서
    print("\n상한을 가장 빨리 끌어올리는 조합 (탐욕적)")
    chosen, cur = [], None
    for _ in range(min(5, len(data))):
        best, bs = None, -1
        for t in data:
            if t in chosen:
                continue
            trial = chosen + [t]
            sc = np.mean([max(item(data[x][f]["pred"], gold[f]) for x in trial) for f in files])
            if sc > bs:
                best, bs = t, sc
        chosen.append(best)
        cur = bs
        print(f"  +{best:<26} 누적 상한 {cur:.4f}  (상한의 {cur / orc * 100:.1f}%)")

    # 고유 기여도
    print("\n고유 기여도 (그 변형만 완전일치를 맞힌 이미지 수)")
    uniq = Counter()
    for f in files:
        hit = [t for t in data if data[t][f]["pred"] == gold[f]]
        if len(hit) == 1:
            uniq[hit[0]] += 1
    for t, n in uniq.most_common():
        print(f"  {t:<26} {n}장")
    if not uniq:
        print("  (없음 — 변형끼리 맞히는 이미지가 완전히 겹칩니다)")


if __name__ == "__main__":
    main()
