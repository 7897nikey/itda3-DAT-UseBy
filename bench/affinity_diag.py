# 리눅스에서 fork 한 작업 프로세스가 부모의 CPU 고정(affinity)을 물려받는지 확인
import os, sys, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def aff(): return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a"
print("시작 직후 부모:", aff())
import bench_par
g = bench_par.load_nb(4, False)
print("세션 생성 후 부모:", aff())
g["predict_one"](sorted(__import__("glob").glob("custom_data/images/*"))[0])
print("추론 1회 후 부모:", aff())
import threading
print("부모 스레드별 고정:", {t: sorted(os.sched_getaffinity(t)) for t in [int(x) for x in os.listdir(f"/proc/{os.getpid()}/task")]} if os.path.exists("/proc") else "n/a")
def child(q): q.put(aff())
for ctx in ("fork", "spawn"):
    c = mp.get_context(ctx); q = c.Queue(); p = c.Process(target=child, args=(q,)); p.start(); print(f"{ctx} 자식:", q.get()); p.join()
