# 리눅스에서 fork 한 작업 프로세스가 부모의 CPU 고정(affinity)을 물려받는지 확인
import os, sys, glob, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def aff():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a"


def child(q):
    q.put(aff())


def per_thread():
    d = f"/proc/{os.getpid()}/task"
    if not os.path.exists(d):
        return "n/a"
    return sorted({tuple(sorted(os.sched_getaffinity(int(t)))) for t in os.listdir(d)})


if __name__ == "__main__":   # spawn 자식이 이 파일을 다시 import 해도 아래가 돌지 않게
    print("시작 직후 부모:", aff())
    import bench_par
    g = bench_par.load_nb(4, False)
    print("세션 생성 후 부모:", aff())
    g["predict_one"](sorted(glob.glob("custom_data/images/*"))[0])
    print("추론 1회 후 부모(메인 스레드):", aff())
    print("부모 스레드들의 고정 집합:", per_thread())
    for ctx in ("fork", "spawn"):
        c = mp.get_context(ctx); q = c.Queue(); p = c.Process(target=child, args=(q,))
        p.start(); print(f"{ctx} 자식:", q.get(timeout=120)); p.join(timeout=60)
