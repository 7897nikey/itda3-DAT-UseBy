# -*- coding: utf-8 -*-
"""예측 CSV를 홀드아웃 규약대로 갈라 채점한다.

  A~C  오염됨(개발에 노출). 참고용
  D+E  의사결정용. 튜닝 판단은 여기서만
  F    봉인. 제출 직전 1회만

사용: python split_eval.py pred1.csv[,pred2.csv] labels_master.csv [--show-f]
"""
import sys, pandas as pd
from math import comb
from fractions import Fraction

def norm(df):
    d = df.copy()
    d["image_id"] = d.image_id.astype(str).str.strip()
    for c in ("year","month","day"):
        s = d[c].astype(str).str.strip()
        w = {"year":4,"month":2,"day":2}[c]
        d[c] = s.where(~s.str.fullmatch(r"\d+", na=False), s.str.zfill(w))
    return d

def grp(b):
    if b in ("A","B","C"): return "A~C(오염)"
    if b in ("D","E"): return "D+E(판단용)"
    return "F(봉인)"

def sign_p(up, dn):
    up, dn = int(up), int(dn); n = up + dn
    if n == 0: return 1.0
    return float(min(Fraction(2*sum(comb(n,k) for k in range(min(up,dn)+1)), 2**n), 1))

def score(path, gt):
    pr = norm(pd.read_csv(path, dtype=str).fillna("NONE"))
    g = gt[gt.image_id.isin(set(pr.image_id))]
    m = g.merge(pr, on="image_id", how="left", suffixes=("_t","_p")).fillna("NONE")
    m["grp"] = m.batch.map(grp)
    m["s"] = sum((m[f"{c}_t"] == m[f"{c}_p"]).astype(int) for c in ("year","month","day")) / 3
    m["exact"] = (m.year_t==m.year_p)&(m.month_t==m.month_p)&(m.day_t==m.day_p)
    m["coverage"] = (m.year_p!="NONE")|(m.month_p!="NONE")
    return m

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_f = "--show-f" in sys.argv
    gt = norm(pd.read_csv(args[1], dtype=str)); gt = gt[gt.status=="완료"]
    frames = {}
    for path in args[0].split(","):
        m = score(path, gt); frames[path] = m
        print(f"\n■ {path}")
        keys = ["전체","A~C(오염)","D+E(판단용)"] + (["F(봉인)"] if show_f else [])
        for key in keys:
            sub = m if key=="전체" else m[m.grp==key]
            if not len(sub): continue
            print(f"  {key:12s} n={len(sub):4d}  부분점수 {sub.s.mean()*100:6.2f}%"
                  f"  Exact {sub.exact.mean()*100:6.2f}%  응답률 {sub.coverage.mean()*100:5.1f}%")
        if not show_f:
            print("  F(봉인)      규약에 따라 표시하지 않음 (--show-f 로만 개봉)")
    if len(frames) == 2:
        (pa,a),(pb,b) = frames.items()
        j = a.set_index("image_id")[["s","grp"]].join(b.set_index("image_id")["s"], rsuffix="_b")
        print(f"\n■ 짝지은 비교: {pb} - {pa}")
        keys = ["전체","D+E(판단용)"] + (["F(봉인)"] if show_f else [])
        for key in keys:
            sub = j if key=="전체" else j[j.grp==key]
            if not len(sub): continue
            up=(sub.s_b>sub.s).sum(); dn=(sub.s_b<sub.s).sum()
            print(f"  {key:12s} 개선 {int(up):3d} / 퇴행 {int(dn):3d} / 동일 {len(sub)-int(up)-int(dn):4d}"
                  f"  p={sign_p(up,dn):.4f}  차이 {(sub.s_b.mean()-sub.s.mean())*100:+.2f}%p")

if __name__ == "__main__":
    main()
