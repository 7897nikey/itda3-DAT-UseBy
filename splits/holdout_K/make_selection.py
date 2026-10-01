# -*- coding: utf-8 -*-
"""홀드아웃 K (2026-10-01)

※ 경로는 로컬 작업 폴더 기준(labels/holdout_K, 상품사진입니다/). 저장소에는 결과물만 splits/holdout_K로 옮겨 두었다.
 — 김건우 단독 라벨링용 300장 선정.

holdout_G와 같은 방식: candidates_safe(904 라벨본 연사본 제외 2,413장)에서
GHI 300장을 빼고, GHI 300장과도 연사본(aHash 16x16 해밍거리 ≤20)인 것을 뺀 뒤
고정 시드로 셔플. 순서 자체가 무작위라 앞에서부터 몇 장만 라벨해도 무작위 표본이다.
뽑힌 목록 안의 연사본 쌍도 뒤쪽 것을 뺀다.
"""
import csv, os, random, hashlib
from PIL import Image, ImageOps
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(ROOT, "상품사진입니다")
files = {os.path.splitext(f)[0]: f for f in os.listdir(IMG)}
def norm(s): s = s.strip(); return s.zfill(6) if s.isdigit() else s
def ids(p): return [norm(r["image_id"]) for r in csv.DictReader(open(p, encoding="utf-8-sig"))]
def ahash(i):
    im = Image.open(os.path.join(IMG, files[i])).convert("L").resize((16, 16))
    px = list(im.getdata()); m = sum(px) / 256
    return sum(1 << k for k, v in enumerate(px) if v > m)
ham = lambda a, b: bin(a ^ b).count("1")
safe = ids(os.path.join(ROOT, "labels/holdout_G/candidates_safe.csv"))
ghi = set(ids(os.path.join(ROOT, "labels/holdout_G/labels_GHI_300.csv")))
pool = [i for i in safe if i not in ghi and i in files]
gh = [ahash(i) for i in ghi if i in files]
random.seed(20261001); random.shuffle(pool)
out, outh, dropped = [], [], 0
for i in pool:
    h = ahash(i)
    if min(ham(h, g) for g in gh) <= 20 or (outh and min(ham(h, g) for g in outh) <= 20):
        dropped += 1; continue
    out.append(i); outh.append(h)
    if len(out) == 300: break
with open(os.path.join(HERE, "ids_K300.csv"), "w", encoding="utf-8") as f:
    f.write("order,image_id,file\n")
    for n, i in enumerate(out, 1): f.write(f"{n},{i},{files[i]}\n")
print("pool", len(pool), "dropped near-dup", dropped, "selected", len(out),
      "sha", hashlib.sha256(",".join(out).encode()).hexdigest()[:16])
