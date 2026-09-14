#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""촬영 원본 -> custom_data/images 표준화.

하는 일
  1) EXIF 회전을 픽셀에 반영하고, EXIF(=GPS 포함) 전량 제거
  2) 긴 변을 max-side 이하로 축소 (확대는 안 함)
  3) JPEG 재인코딩 — 장당 용량 상한을 넘으면 품질을 단계적으로 내림
  4) 파일명을 cust_0001.jpg 로 표준화 (대회 데이터의 숫자 id와 섞이지 않게)
  5) labels.csv / meta.csv 뼈대를 만들고, 새 이미지 행만 append
     (이미 라벨을 적어둔 행은 절대 건드리지 않음)

쓰는 법)
  python custom_data/tools/prepare_images.py --src ~/Desktop/추가수집 --dst custom_data
  python custom_data/tools/prepare_images.py --src ... --dst ... --dry-run
"""
import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Pillow가 필요합니다:  pip install pillow")

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff"}

TAG_COLS = [
    "tag_dot", "tag_engrave", "tag_curved", "tag_lowcontrast",
    "tag_multidate", "tag_lotconfuse", "tag_relative", "tag_reference",
    "tag_ymonly", "tag_foreign", "tag_unreadable",
]
LABEL_COLS = (["image_id", "year", "month", "day", "final_date", "status",
               "term", "notation_raw"] + TAG_COLS + ["notes"])
META_COLS = ["image_id", "src_file", "group_id", "category", "print_type",
             "surface", "capture_note", "source", "collected_by", "collected_at",
             "orig_w", "orig_h", "orig_exif_orientation", "orig_bytes",
             "out_w", "out_h", "out_bytes"]
PARSER_COLS = ["case_id", "image_id", "input_text", "expect_year",
               "expect_month", "expect_day", "expect_final_date", "note"]


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # BOM 없는 utf-8 — 제출 스크립트(evaluate.py)와 동일한 규약
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def exif_orientation(img):
    try:
        ex = img.getexif()
        return str(ex.get(274, "")) if ex else ""
    except Exception:
        return ""


def save_jpeg(img, out_path, quality, max_bytes):
    """용량 상한을 넘으면 품질을 내려 재시도. 마지막까지 넘치면 그대로 저장.

    subsampling=0 (4:4:4) 고정 — 크로마 서브샘플링(4:2:0)은 색 경계를 뭉개는데,
    잉크젯 도트·각인 글자는 그 경계가 곧 획이다. 같은 용량이면 품질을 내리는 쪽이
    도트를 지키는 데 유리하다.
    """
    for q in [quality, 88, 85, 80, 75]:
        img.save(out_path, "JPEG", quality=q, subsampling=0,
                 optimize=True, progressive=True)
        if out_path.stat().st_size <= max_bytes:
            return q
    return q


def sweep(files, cur_side):
    """해상도·품질 조합별 총 용량을 표본으로 실측. 설정을 감으로 정하지 않기 위한 것."""
    import io
    import random
    random.seed(42)
    sample = random.sample(files, min(12, len(files)))
    n = len(files)
    orig = sum(f.stat().st_size for f in files) / 1e6
    print(f"원본 {n}장 / 합계 {orig:.1f}MB")
    print(f"\n{'긴변':>6} {'q':>4} {'장당평균':>10} {'최대':>9} {n:>3}장 환산")
    for s in (1280, 1600, 2048, 2560):
        for q in (88, 92):
            tot = []
            for f in sample:
                im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                w, h = im.size
                sc = min(1.0, s / max(w, h))
                im2 = im.resize((round(w * sc), round(h * sc)), Image.LANCZOS) if sc < 1 else im
                b = io.BytesIO()
                im2.save(b, "JPEG", quality=q, subsampling=0, optimize=True, progressive=True)
                tot.append(b.tell())
            mark = " ←현재" if s == cur_side and q == 92 else ""
            print(f"{s:>6} {q:>4} {sum(tot)/len(tot)/1024:>8.0f}KB "
                  f"{max(tot)/1024:>7.0f}KB {sum(tot)/len(tot)*n/1e6:>8.1f}MB{mark}")
    print("\nGitHub 제한은 '단일 파일' 100MB — 위 최대값과 비교하면 어느 조합도 여유가 크다.")
    print("실제 기준은 저장소 총량과 날짜 영역 해상도다. docs/COLLECTION_PROTOCOL.md §4 참고.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="촬영 원본 폴더")
    ap.add_argument("--dst", default="custom_data", help="custom_data 폴더 경로")
    ap.add_argument("--max-side", type=int, default=2048,
                    help="긴 변 상한 px (실측 근거는 docs/COLLECTION_PROTOCOL.md)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--max-mb", type=float, default=1.0, help="장당 용량 상한")
    ap.add_argument("--budget-mb", type=float, default=60.0,
                    help="images/ 폴더 총량 경고선 (GitHub 단일파일 100MB와 별개)")
    ap.add_argument("--prefix", default="cust_")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="파일을 쓰지 않고, 해상도·품질 조합별 총 용량만 실측해서 표로 출력")
    a = ap.parse_args()

    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    images_dir = dst / "images"
    max_bytes = int(a.max_mb * 1024 * 1024)

    dst_abs = dst.resolve()
    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTS
                   and not p.name.startswith(".")
                   and dst_abs not in p.resolve().parents)  # 출력 폴더를 다시 먹지 않게
    if not files:
        sys.exit(f"이미지를 못 찾음: {src}")

    if a.sweep:
        sweep(files, a.max_side)
        return

    labels = read_rows(dst / "labels.csv")
    metas = read_rows(dst / "meta.csv")
    known_src = {m.get("src_file", "") for m in metas}
    used_n = [int(m["image_id"].replace(a.prefix, "")) for m in metas
              if m.get("image_id", "").startswith(a.prefix)
              and m["image_id"].replace(a.prefix, "").isdigit()]
    next_n = max(used_n) + 1 if used_n else 1

    print(f"원본 {len(files)}장 / 이미 등록됨 {len(known_src)}장 / 출력 {images_dir}")
    if not a.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for p in files:
        rel = str(p.relative_to(src))
        if rel in known_src:
            print(f"  skip (등록됨) {rel}")
            continue

        try:
            img = Image.open(p)
        except Exception as e:
            print(f"  !! 열기 실패 {rel}: {e}")
            continue

        ow, oh = img.size
        orient = exif_orientation(img)
        obytes = p.stat().st_size

        img = ImageOps.exif_transpose(img)          # 회전을 픽셀에 반영
        if img.mode != "RGB":
            img = img.convert("RGB")                # EXIF/알파/ICC 제거 효과
        w, h = img.size
        if max(w, h) > a.max_side:
            s = a.max_side / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)

        image_id = f"{a.prefix}{next_n:04d}"
        out_path = images_dir / f"{image_id}.jpg"
        if a.dry_run:
            print(f"  [dry] {rel} -> {out_path.name}  {ow}x{oh}(or={orient or '-'}) -> {img.size}")
        else:
            q = save_jpeg(img, out_path, a.quality, max_bytes)
            ob = out_path.stat().st_size
            print(f"  {rel} -> {out_path.name}  {ow}x{oh} -> {img.size}  "
                  f"q{q}  {obytes/1e6:.1f}MB -> {ob/1e6:.2f}MB")
            labels.append({"image_id": image_id, "status": "",
                           **{t: "FALSE" for t in TAG_COLS}})
            metas.append({
                "image_id": image_id, "src_file": rel, "group_id": image_id,
                "collected_at": date.today().isoformat(),
                "orig_w": ow, "orig_h": oh, "orig_exif_orientation": orient,
                "orig_bytes": obytes, "out_w": img.size[0], "out_h": img.size[1],
                "out_bytes": ob,
            })
        next_n += 1
        added += 1

    if a.dry_run:
        print(f"\n[dry-run] 추가될 이미지 {added}장. 파일은 쓰지 않았음.")
        return

    write_rows(dst / "labels.csv", LABEL_COLS, labels)
    write_rows(dst / "meta.csv", META_COLS, metas)
    pc = dst / "parser_cases.csv"
    if not pc.exists():
        write_rows(pc, PARSER_COLS, [])

    sizes = [f.stat().st_size for f in images_dir.glob("*.jpg")]
    total, biggest = sum(sizes), (max(sizes) if sizes else 0)
    print(f"\n완료: 새 이미지 {added}장 / 총 {len(labels)}행")
    print(f"  images/ 총량 {total/1e6:.1f}MB · 최대 단일 파일 {biggest/1e6:.2f}MB "
          f"(GitHub 단일 파일 제한 100MB)")
    if total / 1e6 > a.budget_mb:
        print(f"  △ 총량이 경고선 {a.budget_mb}MB 를 넘음 — --max-side 1600 으로 재생성하거나")
        print("     원본은 외부 링크로 돌리는 것을 검토하세요 (--sweep 으로 조합별 용량 확인).")
    print("다음: labels.csv 의 year·month·day·final_date·status·term·notation_raw·tag_* 를 채우고")
    print("      python custom_data/tools/validate_labels.py custom_data")


if __name__ == "__main__":
    main()
