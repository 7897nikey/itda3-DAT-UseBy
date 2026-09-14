#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""custom_data/labels.csv 무결성 검증 + 구성 요약.

라벨을 다 적었다고 생각한 시점에 돌린다. 여기서 걸리는 것들은 전부
"모델 오류"로 잘못 집계되기 전에 잡아야 하는 것들이다.

검사 항목
  - images/ 와 labels.csv 의 image_id 가 1:1 인가 (누락·유령 행)
  - year/month/day 형식 (4자리/2자리/NONE) 과 달력상 실재하는 날짜인가
  - final_date 가 세 필드와 모순되지 않는가
  - 연도 범위 (기본 2015~2035) — 오타·오독 라벨 잡기
  - status / tag_* 값이 허용된 값인가
  - 채우다 만 행 (전부 빈칸)
요약 출력
  - 제품군·태그·term 별 분포, 난이도 태그 커버리지

쓰는 법)
  python custom_data/tools/validate_labels.py custom_data
  python custom_data/tools/validate_labels.py custom_data --parser-cases
"""
import argparse
import csv
import sys
from calendar import monthrange
from collections import Counter
from pathlib import Path

NONE = "NONE"
OK_STATUS = {"완료", "보류"}
OK_BOOL = {"TRUE", "FALSE", "", "1", "0", "true", "false"}
YEAR_MIN, YEAR_MAX = 2015, 2035


def read_rows(path):
    if not path.exists():
        sys.exit(f"파일 없음: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_none(v):
    return (v or "").strip().upper() == NONE


def check_labels(root):
    errs, warns = [], []
    labels = read_rows(root / "labels.csv")
    img_ids = {p.stem for p in (root / "images").glob("*.jpg")}
    lab_ids = [(r.get("image_id") or "").strip() for r in labels]

    dup = [k for k, c in Counter(lab_ids).items() if c > 1]
    if dup:
        errs.append(f"image_id 중복: {dup}")
    missing = img_ids - set(lab_ids)
    ghost = set(lab_ids) - img_ids
    if missing:
        errs.append(f"이미지는 있는데 라벨 행이 없음 ({len(missing)}): {sorted(missing)[:8]}")
    if ghost:
        errs.append(f"라벨 행은 있는데 이미지가 없음 ({len(ghost)}): {sorted(ghost)[:8]}")

    tag_cols = [c for c in (labels[0].keys() if labels else []) if c.startswith("tag_")]
    done = 0
    for r in labels:
        i = (r.get("image_id") or "?").strip()
        y, m, d = [(r.get(k) or "").strip() for k in ("year", "month", "day")]
        fd = (r.get("final_date") or "").strip()
        st = (r.get("status") or "").strip()

        if not any([y, m, d, fd, st]):
            warns.append(f"{i}: 아직 안 채움")
            continue
        if st not in OK_STATUS:
            errs.append(f"{i}: status='{st}' (허용: 완료/보류)")
        if st == "보류":
            continue
        done += 1

        # 형식
        if not (is_none(y) or (y.isdigit() and len(y) == 4)):
            errs.append(f"{i}: year='{y}' 형식 오류 (4자리 또는 NONE)")
        for k, v in (("month", m), ("day", d)):
            if not (is_none(v) or (v.isdigit() and len(v) == 2)):
                errs.append(f"{i}: {k}='{v}' 형식 오류 (2자리 zero-fill 또는 NONE)")

        # 달력 유효성
        if not is_none(m) and m.isdigit() and not (1 <= int(m) <= 12):
            errs.append(f"{i}: month={m} 범위 밖")
        if not is_none(d) and d.isdigit():
            if not (1 <= int(d) <= 31):
                errs.append(f"{i}: day={d} 범위 밖")
            elif not is_none(m) and m.isdigit() and 1 <= int(m) <= 12:
                yy = int(y) if (not is_none(y) and y.isdigit()) else 2024
                if int(d) > monthrange(yy, int(m))[1]:
                    errs.append(f"{i}: {y}-{m}-{d} 는 존재하지 않는 날짜")
        if not is_none(y) and y.isdigit() and not (YEAR_MIN <= int(y) <= YEAR_MAX):
            warns.append(f"{i}: year={y} 가 {YEAR_MIN}~{YEAR_MAX} 밖 — 오독 아닌지 확인")

        # final_date 일관성
        want = NONE if all(is_none(v) for v in (y, m, d)) else f"{y or NONE}-{m or NONE}-{d or NONE}"
        if fd != want:
            errs.append(f"{i}: final_date='{fd}' 인데 필드 조합은 '{want}'")

        # 태그
        for t in tag_cols:
            if (r.get(t) or "").strip() not in OK_BOOL:
                errs.append(f"{i}: {t}='{r.get(t)}' (TRUE/FALSE)")

        # 규칙 상호검증
        if (r.get(_t := "tag_ymonly") or "").upper() == "TRUE" and not is_none(d):
            errs.append(f"{i}: tag_ymonly=TRUE 인데 day={d} 가 채워져 있음 (가이드 P2)")
        if not is_none(d) and is_none(m):
            warns.append(f"{i}: 일은 있는데 월이 NONE — 확인 필요")
        if not (r.get("notation_raw") or "").strip() and not is_none(y):
            warns.append(f"{i}: notation_raw 비어 있음 (파서 회귀 케이스를 못 만듦)")

    return labels, tag_cols, done, errs, warns


def summarize(root, labels, tag_cols, done):
    print(f"\n■ 구성 요약 (완료 {done}행 / 전체 {len(labels)}행)")
    meta_p = root / "meta.csv"
    if meta_p.exists():
        metas = read_rows(meta_p)
        for col in ("category", "print_type", "surface"):
            c = Counter((m.get(col) or "(미기입)").strip() for m in metas)
            print(f"  {col:<11}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        groups = {(m.get("group_id") or m.get("image_id")) for m in metas}
        print(f"  group_id   : {len(groups)}개 (같은 제품 다각도 촬영을 한 그룹으로 취급)")
    c = Counter((r.get("term") or "(미기입)").strip() for r in labels if (r.get("status") or "") == "완료")
    print(f"  term       : " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print("  난이도 태그 :")
    for t in tag_cols:
        n = sum(1 for r in labels if (r.get(t) or "").upper() == "TRUE")
        if n:
            print(f"     {t.replace('tag_',''):<13} {n}")
    untag = sum(1 for r in labels
                if (r.get("status") or "") == "완료"
                and not any((r.get(t) or "").upper() == "TRUE" for t in tag_cols))
    print(f"     {'태그없음(평이)':<13} {untag}")
    nonrate = sum(1 for r in labels
                  if (r.get("status") or "") == "완료" and is_none(r.get("year", "")) and is_none(r.get("month", "")))
    if done:
        print(f"  표기 없음/판독불가로 정답이 NONE 인 행: {nonrate} ({nonrate/done*100:.0f}%)")


def check_parser_cases(root):
    p = root / "parser_cases.csv"
    rows = read_rows(p)
    print(f"\n■ parser_cases.csv — {len(rows)}건")
    errs = []
    for r in rows:
        cid = r.get("case_id") or "?"
        if not (r.get("input_text") or "").strip():
            errs.append(f"{cid}: input_text 비어 있음")
        y, m, d = [(r.get(f"expect_{k}") or "").strip() for k in ("year", "month", "day")]
        want = NONE if all(is_none(v) for v in (y, m, d)) else f"{y or NONE}-{m or NONE}-{d or NONE}"
        if (r.get("expect_final_date") or "").strip() != want:
            errs.append(f"{cid}: expect_final_date 가 필드 조합('{want}')과 불일치")
    for e in errs:
        print("  ✗", e)
    if not errs and rows:
        print("  형식 이상 없음. 실제 파서 대조는 저장소 루트에서:")
        print("    python -c \"from date_parser import extract_expiry_fields as f; print(f('<input_text>'))\"")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="custom_data")
    ap.add_argument("--parser-cases", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).expanduser()

    labels, tag_cols, done, errs, warns = check_labels(root)
    print(f"■ labels.csv 검증 — {len(labels)}행")
    for w in warns:
        print("  △", w)
    for e in errs:
        print("  ✗", e)
    if not errs:
        print("  오류 없음.")
    summarize(root, labels, tag_cols, done)
    if a.parser_cases:
        errs += check_parser_cases(root)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
