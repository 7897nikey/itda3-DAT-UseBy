# -*- coding: utf-8 -*-
"""OCR 텍스트 정규화 재시도 실험 (추가 OCR 없음. 캐시된 스윕 결과만 사용).

배경: 오류 분해 결과 '정답 숫자는 읽혔는데 파서가 못 살린' 경우가 41장(13.8%) 남았고,
      그 대부분이 토큰 사이 공백·중복 구분자·시각 혼입 같은 텍스트 표면 문제였다.
      OCR 을 다시 돌릴 필요 없이 텍스트만 손봐서 재파싱하면 되는지 잰다.

방식: 원문 파싱이 실패(또는 필드 결손)하면 정규화 후보를 순서대로 넣어보고 먼저 성공한 것을 쓴다.
      규칙을 하나씩 누적해가며 ablation 으로 효과를 분리한다.

사용법:
    python normalize_retry.py --sweep sweep_sorted --tag clahe__m15
"""
import argparse
import json
import math
import os
import re

import pandas as pd

from date_parser import extract_expiry_fields

FIELDS = ["year", "month", "day"]


# ── 정규화 규칙 ──────────────────────────────────────────────────────────
def r_squeeze(t):
    """숫자·구분자 사이 공백 제거, 중복 구분자 축약. '2026.07. .11' -> '2026.07.11'"""
    t = re.sub(r"(?<=[\d.\-/])\s+(?=[\d.\-/])", "", t)
    t = re.sub(r"([.\-/])[.\-/]+", r"\1", t)
    return t


def r_drop_time(t):
    """시각 표기 제거. '2026. .06.10 16:42' 의 16:42 가 날짜로 오인되는 것을 막는다."""
    return re.sub(r"\b\d{1,2}\s*[:：]\s*\d{2}(\s*[:：]\s*\d{2})?\b", " ", t)


def r_clean(t):
    """날짜와 무관한 글자 제거 (한글 키워드는 남긴다)."""
    return re.sub(r"[^0-9가-힣A-Za-z.\-/년월일 ]", " ", t)


def r_us_order(t):
    """MM/DD/YYYY 로 보이는 표기를 YYYY.MM.DD 로 바꾼다.

    실측 근거: '06/03/2027' 의 정답은 2027-06-03(6월 3일)인데 파서는 03월 06일로 읽었다.
    앞 두 자리가 12 이하이고 뒤 두 자리가 12 초과일 때만 적용하면 안전하지만,
    둘 다 12 이하인 모호한 경우까지 포함할지는 이 실험으로 판단한다.
    """
    def f(m):
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a <= 12 and 1 <= b <= 31:
            return f"{y}.{a:02d}.{b:02d}"
        return m.group(0)
    return re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", f, t)


def r_trim_extra(t):
    """마지막 조각이 3자리로 붙은 경우(2026.07.067) 끝 한 자리를 떼어 본다."""
    return re.sub(r"([.\-/]\d{2})\d\b", r"\1", t)


def r_year_prefix(t):
    """앞에 잡숫자가 붙은 연도를 잘라낸다. '112026.03.04' -> '2026.03.04'"""
    return re.sub(r"\b\d+(20\d{2})([.\-/])", r"\1\2", t)


# 순서가 중요하다: 공백을 먼저 지우면 '10 16:42' 가 '1016:42' 로 붙어 시각 제거가 실패한다.
RULES = [("drop_time", r_drop_time), ("squeeze", r_squeeze), ("clean", r_clean),
         ("year_prefix", r_year_prefix), ("us_order", r_us_order), ("trim_extra", r_trim_extra)]


def parse(text, kind):
    full = ("소비기한 " + text) if (kind == "crop" and text.strip()) else text
    try:
        p = extract_expiry_fields(full) if full.strip() else {}
    except Exception:
        p = {}
    return tuple(str(p.get(k, "NONE")) for k in FIELDS)


def n_filled(pred):
    return sum(v != "NONE" for v in pred)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep_sorted")
    ap.add_argument("--tag", default="clahe__m15")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--policy", default="fill", choices=["fill", "always"],
                    help="fill: 원문 파싱이 부실할 때만 재시도(보수적) / always: 정규화본을 먼저 믿음")
    a = ap.parse_args()

    g = pd.read_csv(a.splits, dtype=str)
    g = g[g["split"] != "excluded"].set_index("file")

    recs = []
    with open(os.path.join(a.sweep, a.tag + ".jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r["file"] in g.index:
                    r["text"] = " ".join(e[0] for e in r["ocr"]).strip()
                    recs.append(r)

    gold = {r["file"]: tuple(str(g.loc[r["file"]][k]) for k in FIELDS) for r in recs}

    def evaluate(rules):
        """정책에 따라 정규화 후보를 적용한다.

        fill  : 원문 파싱이 부실할 때만 재시도하고, 더 많이 채운 답만 채택 (악화 위험 없음)
        always: 규칙을 전부 적용한 텍스트를 먼저 파싱하고, 결손이 늘지 않으면 그걸 채택
                (완전하지만 틀린 답도 바로잡을 수 있는 대신 악화 가능)
        """
        out = {}
        for r in recs:
            best = parse(r["text"], r["kind"])
            if a.policy == "always" and rules:
                t = r["text"]
                for _, fn in rules:
                    t = fn(t)
                cand = parse(t, r["kind"])
                if n_filled(cand) >= n_filled(best):
                    best = cand
                out[r["file"]] = best
                continue
            if n_filled(best) < 3:
                t = r["text"]
                for _, fn in rules:
                    t = fn(t)
                    cand = parse(t, r["kind"])
                    if n_filled(cand) > n_filled(best):
                        best = cand
                        if n_filled(best) == 3:
                            break
            out[r["file"]] = best
        return out

    def score(pred):
        it = sum(sum(p == q for p, q in zip(pred[f], gold[f])) / 3.0 for f in pred) / len(pred)
        ex = sum(pred[f] == gold[f] for f in pred) / len(pred)
        an = sum(n_filled(pred[f]) > 0 for f in pred) / len(pred)
        return it, ex, an

    base = evaluate([])
    bi, be, ba = score(base)
    print(f"[{a.tag}] {len(recs)}장\n")
    print(f"{'규칙(누적)':<34} {'부분점수':>8} {'완전일치':>8} {'응답률':>7} {'개선/악화':>9} {'p':>7}")
    print(f"{'원문 그대로':<34} {bi:8.4f} {be:8.4f} {ba:7.4f} {'-':>9} {'-':>7}")

    for i in range(1, len(RULES) + 1):
        pred = evaluate(RULES[:i])
        it, ex, an = score(pred)
        b = sum(1 for f in pred if pred[f] == gold[f] and base[f] != gold[f])
        c = sum(1 for f in pred if pred[f] != gold[f] and base[f] == gold[f])
        名 = " + ".join(n for n, _ in RULES[:i])
        print(f"{名:<34} {it:8.4f} {ex:8.4f} {an:7.4f} {f'{b}/{c}':>9} {mcnemar(b, c):7.4f}")

    # 규칙별 단독 효과
    print("\n규칙 단독 효과 (원문 대비)")
    for name, fn in RULES:
        pred = evaluate([(name, fn)])
        it, ex, an = score(pred)
        b = sum(1 for f in pred if pred[f] == gold[f] and base[f] != gold[f])
        c = sum(1 for f in pred if pred[f] != gold[f] and base[f] == gold[f])
        print(f"  {name:<12} 부분점수 {it:.4f} ({(it - bi) * 100:+.1f}%p)  개선/악화 {b}/{c}  p={mcnemar(b, c):.4f}")

    full = evaluate(RULES)
    print("\n전체 규칙 적용 후에도 틀린 예시 (정답 | 출력 | 원문):")
    shown = 0
    for r in recs:
        f = r["file"]
        if full[f] != gold[f] and n_filled(full[f]) > 0 and shown < 10:
            print(f"  {'-'.join(gold[f]):<16} | {'-'.join(full[f]):<16} | {r['text'][:70]}")
            shown += 1


if __name__ == "__main__":
    main()
