# -*- coding: utf-8 -*-
"""동료가 뽑아온 라벨링 목록이 홀드아웃으로 쓸 만한지 검사한다.

홀드아웃은 "처음 보는 데이터"여야 값을 한다. 그런데 눈으로는 거를 수 없는
오염이 두 가지 있다.

  1) 이미 라벨된 904장과 겹침 — 중복 작업이고 홀드아웃도 아니다
  2) 기존 평가셋의 연사본 — 같은 상품을 연속으로 찍어 인덱스가 붙어 있다.
     사람 눈에는 다른 사진인데 모델에는 사실상 같은 사진이다.
     실제로 배치 B 148건이 이 경로로 학습에 샌 적이 있다.

2번을 잡으려고 대회 이미지 3,352장의 지각 해시(average hash, 16x16)를 미리
떠 뒀다. 후보와 기존 라벨본 사이의 최소 해밍거리가 임계값 이하면 연사본으로 본다.

쓰는 법)
    python check_selection.py 동료가_준_목록.csv
    python check_selection.py labels_G.csv          # 추천 300장 검사
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
THRESHOLD = 20  # 해밍거리. 실측 분포에서 중앙값이 61이라 20은 넉넉히 보수적이다.


def norm(s):
    s = str(s).strip()
    return s.zfill(6) if s.isdigit() else s


def read_ids(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "image_id" not in rows[0]:
        sys.exit(f"[실패] {path} 에 image_id 열이 없다.")
    return [norm(r["image_id"]) for r in rows]


def main(path):
    ids = read_ids(path)
    dup_in_file = len(ids) - len(set(ids))

    labels_master = os.path.join(ROOT, "labels", "labels_master.csv")
    with open(labels_master, encoding="utf-8-sig") as f:
        labeled = {norm(r["image_id"]) for r in csv.DictReader(f) if r["status"] == "완료"}

    img_dir = os.path.join(ROOT, "상품사진입니다")
    have = {norm(os.path.splitext(f)[0]) for f in os.listdir(img_dir)}

    excl_path = os.path.join(HERE, "excluded_near_duplicates.csv")
    near = {}
    if os.path.exists(excl_path):
        with open(excl_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                near[norm(r["image_id"])] = (r["해밍거리"], r["닮은_라벨본"])

    missing = [i for i in ids if i not in have]
    overlap = [i for i in ids if i in labeled]
    dupes = [i for i in ids if i in near]

    print(f"목록 {len(ids)}건 검사 — {path}\n")
    ok = True
    if dup_in_file:
        print(f"  [경고] 목록 안에 같은 id 가 {dup_in_file}건 중복"); ok = False
    if missing:
        print(f"  [실패] 상품사진입니다 폴더에 없는 id {len(missing)}건: {missing[:8]}"); ok = False
    if overlap:
        print(f"  [실패] 이미 라벨된 904장과 겹침 {len(overlap)}건: {overlap[:8]}")
        print("         → 홀드아웃이 아니다. 빼거나 교차검증용으로 따로 쓸 것."); ok = False
    if dupes:
        print(f"  [실패] 기존 평가셋의 연사본으로 의심 {len(dupes)}건")
        for i in dupes[:8]:
            d, b = near[i]
            print(f"         {i} ← {b} 와 해밍거리 {d}")
        ok = False
    if ok:
        print("  통과. 겹침 0 / 연사본 0 / 누락 0")

    n = len(set(ids))
    print(f"\n  이 크기로 잡을 수 있는 최소 개선폭(80% 검정력): 약 {2.8016 * 26.1 / (n ** 0.5):.1f}%p")
    print("  (F 에서 잰 장당 점수차 표준편차 26.1%p 기준)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "labels_G.csv")))
