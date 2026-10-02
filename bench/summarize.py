import json, glob, os, sys, collections
d = sys.argv[1]; R = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    o = json.load(open(f)); R[o["cfg"]].append(o)
ref = {"full": "seq_w1_t4", "fast": "seq_w1_t4_fast", "yolo": "seq_w1_t4_yolo"}
def task(c): return "yolo" if c.endswith("_yolo") else "fast" if c.endswith("_fast") else "full"
print("| 구성 | 처리량 ms/장 (회차별) | 현행 대비 | 한 장 지연 ms | CPU초/장 | 예측 동일 |")
print("|---|---|---|---|---|---|")
for c, os_ in R.items():
    b = R[ref[task(c)]]
    tp = [o["per_img_ms"] for o in os_]; m = sum(tp) / len(tp); bm = sum(o["per_img_ms"] for o in b) / len(b)
    same = all(o["pred"] == b[0]["pred"] for o in os_)
    print(f"| {c} | {m:.0f} ({', '.join(f'{x:.0f}' for x in tp)}) | {bm/m:.2f}배 | "
          f"{sum(o['latency_mean_ms'] for o in os_)/len(os_):.0f} | {sum(o['cpu_s'] for o in os_)/len(os_)/os_[0]['n']:.2f} | {'O' if same else 'X'} |")
