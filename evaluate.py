# -*- coding: utf-8 -*-
"""예측 CSV와 손라벨 정답을 맞춰보고 점수를 냄.

항목별 부분점수를 주 지표로 보되, 완전일치도 같이 찍음. 채점이 연월일을
따로 채점하는 방식이라 월일만 맞아도 점수가 들어오기 때문임.

예측이 정답의 일부만 담고 있으면 정답 쪽도 거기에 맞춰 걸러냄. 안 그러면
안 돌린 사진까지 전부 오답으로 잡혀서 점수가 실제보다 낮게 나옴.

쓰는 법)  python evaluate.py 예측.csv 정답.csv
"""
import sys, pandas as pd

def norm(df):
    d = df.copy()
    d["image_id"] = d.image_id.astype(str).str.strip()
    # 숫자만으로 된 image_id는 6자리로 통일. 압축 배포 과정에서 일부 파일명이
    # 0-padding 없이(3345.jpg 등) 섞여 나와 있으면 병합이 안 돼서 그 건이
    # 통째로 채점에서 빠짐 — cust_0001 같은 비숫자 id는 안 건드림.
    digit = d.image_id.str.fullmatch(r"\d+", na=False)
    d.loc[digit, "image_id"] = d.loc[digit, "image_id"].str.zfill(6)
    for c in ("year","month","day"):
        s = d[c].astype(str).str.strip()
        w = {"year":4,"month":2,"day":2}[c]
        d[c] = s.where(~s.str.fullmatch(r"\d+", na=False), s.str.zfill(w))
    return d

def main(pred_path, gt_path, tag=""):
    pred = norm(pd.read_csv(pred_path, dtype=str, encoding="utf-8-sig").fillna("NONE"))
    gt   = norm(pd.read_csv(gt_path, dtype=str, encoding="utf-8-sig"))
    gt   = gt[gt.status == "완료"]                     # 보류 건 제외
    # 예측이 정답의 부분집합이면 그 부분만 채점 (부분 실험용)
    if len(pred) < len(gt):
        gt = gt[gt.image_id.isin(set(pred.image_id))]
        print(f"  [부분평가] 예측 {len(pred)}건에 맞춰 정답 {len(gt)}건으로 한정")
    m = gt.merge(pred, on="image_id", how="left", suffixes=("_t","_p")).fillna("NONE")

    ok_y = m.year_t  == m.year_p
    ok_m = m.month_t == m.month_p
    ok_d = m.day_t   == m.day_p
    exact = ok_y & ok_m & ok_d
    cov = (m.year_p != "NONE") | (m.month_p != "NONE")

    print(f"\n{'='*54}\n■ {tag or pred_path}   (정답 {len(m)}건)")
    print(f"{'='*54}")
    print(f"  Exact Match (연·월·일 전부) : {exact.mean()*100:6.2f}%   {exact.sum()}/{len(m)}")
    print(f"  연(year)  정확도            : {ok_y.mean()*100:6.2f}%")
    print(f"  월(month) 정확도            : {ok_m.mean()*100:6.2f}%")
    print(f"  일(day)   정확도            : {ok_d.mean()*100:6.2f}%")
    print(f"  응답률(Coverage)            : {cov.mean()*100:6.2f}%")
    got = m[cov]
    if len(got):
        print(f"  응답한 것 중 정확도          : {(exact & cov).sum()/len(got)*100:6.2f}%")

    # 난이도 태그별 (gt에 이미 tag_ 컬럼이 있으므로 재merge 불필요)
    tags = [c for c in m.columns if c.startswith("tag_")]
    if tags:
        mm = m.copy(); mm["exact"] = exact.values
        istrue = lambda col: mm[col].astype(str).str.lower().isin(["true","1"])
        print(f"\n  ── 난이도 태그별 Exact Match ──")
        for t in tags:
            sub = mm[istrue(t)]
            if len(sub) >= 3:
                print(f"    {t.replace('tag_',''):<12} {sub.exact.mean()*100:5.1f}%  (n={len(sub)})")
        ntag = sum(istrue(t).astype(int) for t in tags)
        easy = mm[ntag == 0]
        print(f"    {'태그없음(평이)':<12} {easy.exact.mean()*100:5.1f}%  (n={len(easy)})")

    return {"exact":exact.mean(),"y":ok_y.mean(),"m":ok_m.mean(),"d":ok_d.mean(),"cov":cov.mean(),"n":len(m)}

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "")
