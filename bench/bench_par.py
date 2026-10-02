# 실험용: CPU 병렬화 가설 검증 벤치마크 (speed-bench 브랜치 전용)
#
# 사용: python bench/bench_par.py <config> <이미지폴더> <장수> <결과json>
#   config 이름 규칙  {seq|proc|thr}_w{작업자수}_t{스레드수}[_nospin][_fast|_yolo]
#     seq   : 현행처럼 한 프로세스가 순서대로
#     proc  : 작업 프로세스 w개가 사진을 나눠 맡음
#     thr   : 한 프로세스 안에서 스레드 w개가 같은 세션을 나눠 씀 (onnxruntime 은 GIL 을 놓는다)
#     nospin: onnxruntime 스레드의 대기 회전(spin) 끄기
#     fast  : 1단계(크롭 OCR)만 — 가벼운 작업
#     yolo  : YOLO 검출만 — 더 가벼운 작업
import os, sys, json, time, glob, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(cfg):
    p = cfg.split("_")
    return dict(mode=p[0], w=int(p[1][1:]), t=int(p[2][1:]), nospin="nospin" in p,
                task="yolo" if "yolo" in p else "fast" if "fast" in p else "full")


def load_nb(t, nospin):
    """predict.ipynb cell 0~2 를 그대로 실행하고, 스레드 수·spin 만 바꾼다."""
    os.environ["OMP_NUM_THREADS"] = str(t)
    import onnxruntime as ort
    _SO = ort.SessionOptions
    class SO(_SO):
        def __init__(self):
            super().__init__()
            if nospin:
                self.add_session_config_entry("session.intra_op.allow_spinning", "0")
                self.add_session_config_entry("session.inter_op.allow_spinning", "0")
    ort.SessionOptions = SO
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    os.environ.setdefault("ITDA_INPUT_DIR", "/nonexistent")
    nb = json.load(open(os.path.join(ROOT, "predict.ipynb")))
    g = {"__name__": "nbmod"}
    for k, c in enumerate(nb["cells"][:3]):
        src = "".join(c["source"])
        if k == 1:
            src = src.replace("RAPID_CFG = _make_cfg()",
                "RAPID_CFG = _make_cfg()\n"
                f"_t=open(RAPID_CFG).read().replace('&intra_nums 4','&intra_nums {t}'); open(RAPID_CFG,'w').write(_t)")
            src = src.replace('det = ort.InferenceSession(YOLO_W, providers=["CPUExecutionProvider"])',
                f'_so=ort.SessionOptions(); _so.intra_op_num_threads={t}; _so.inter_op_num_threads=1\n'
                '        det = ort.InferenceSession(YOLO_W, sess_options=_so, providers=["CPUExecutionProvider"])')
            assert "_so=" in src and "_t=open" in src
        exec(src, g)
    g["cv2"].setNumThreads(t)
    return g


_G = None
_TASK = None


def _init(t, nospin, task):
    global _G, _TASK
    _G, _TASK = load_nb(t, nospin), task


def _one(p):
    c0 = time.process_time(); w0 = time.time()
    if _TASK == "yolo":
        img = _G["load_bgr"](p); _G["yolo_detect"](img); r = "-"
    else:
        r = _G["predict_one"](p, fast=(_TASK == "fast"))["final_date"]
    return r, time.time() - w0, time.process_time() - c0


def _warm(_):
    time.sleep(0.2)   # 모든 작업자가 기동을 마칠 때까지 붙잡아 둔다
    return os.getpid()


if __name__ == "__main__":
    cfg, inp, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    c = parse(cfg)
    paths = sorted(glob.glob(os.path.join(inp, "*")))[:n]
    t0 = time.time()
    if c["mode"] == "proc":
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(c["w"], mp_context=mp.get_context("spawn"),
                                 initializer=_init, initargs=(c["t"], c["nospin"], c["task"]))
        list(ex.map(_warm, range(c["w"] * 4)))             # 작업자 기동 대기
        t1 = time.time()
        res = list(ex.map(_one, paths, chunksize=1))
        t2 = time.time()
        cpu_total = sum(r[2] for r in res)
        ex.shutdown()
    else:
        _init(c["t"], c["nospin"], c["task"])
        t1 = time.time()
        if c["mode"] == "thr":
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(c["w"]) as ex:
                res = list(ex.map(_one, paths))
        else:
            res = [_one(p) for p in paths]
        t2 = time.time()
        cpu_total = None
    c_proc = time.process_time()
    wall = t2 - t1
    lat = sorted(r[1] for r in res)
    o = dict(cfg=cfg, n=len(paths), startup=t1 - t0, wall=wall, per_img_ms=wall / len(paths) * 1000,
             latency_p50_ms=lat[len(lat) // 2] * 1000, latency_mean_ms=sum(lat) / len(lat) * 1000,
             cpu_s=cpu_total if cpu_total is not None else c_proc, pred=[r[0] for r in res])
    json.dump(o, open(out, "w"), ensure_ascii=False)
    print(f"{cfg:24s} 처리량 {o['per_img_ms']:7.1f} ms/장 | 한 장 지연 평균 {o['latency_mean_ms']:7.1f} ms "
          f"| CPU {o['cpu_s']:6.1f}s | 기동 {o['startup']:.1f}s")
