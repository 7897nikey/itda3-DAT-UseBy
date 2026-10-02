# -*- coding: utf-8 -*-
"""predict.ipynb 의 병렬 실행용 작업 프로세스.

onnxruntime 은 fork 에 안전하지 않다. 부모가 세션을 만든 뒤 fork 한 자식이 새 세션을
만들면 죽는다(리눅스 x86 실측: 작업 프로세스 4개가 시작하자마자 전부 종료).
그래서 작업 프로세스는 spawn 으로 깨끗하게 띄우고, 거기서 predict.ipynb 의 cell 0~2 를
그대로 실행해 같은 함수(predict_one)를 만든다. 코드는 노트북 한 벌만 있고 여기서 복사하지 않는다.
모델은 1스레드로 다시 만든다 — 4코어를 프로세스 4개가 하나씩 나눠 쓰기 위해서다.
"""
import contextlib
import io
import json
import os
import tempfile

_NS = None


def init(base, threads=1):
    global _NS
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.chdir(base)
    nb = json.load(open(os.path.join(base, "predict.ipynb"), encoding="utf-8"))
    ns = {"__name__": "predict_worker"}
    with contextlib.redirect_stdout(io.StringIO()):   # 로드 메시지가 노트북 출력에 4번 섞이지 않게
        for c in nb["cells"][:3]:
            exec("".join(c["source"]), ns)

    # 노트북은 4스레드로 만든다. 작업 프로세스에서는 threads 로 다시 만든다.
    txt = open(ns["RAPID_CFG"], encoding="utf-8").read().replace("&intra_nums 4", f"&intra_nums {threads}")
    fd, cfg = tempfile.mkstemp(suffix=".yaml"); os.close(fd)
    open(cfg, "w", encoding="utf-8").write(txt)
    ns["ocr"] = ns["RapidOCR"](config_path=cfg)
    if ns["det"] is not None:
        ort = ns["ort"]
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        ns["det"] = ort.InferenceSession(ns["YOLO_W"], sess_options=so, providers=["CPUExecutionProvider"])
    ns["cv2"].setNumThreads(threads)
    _NS = ns


def work(path, fast):
    try:
        return _NS["predict_one"](path, fast=fast)
    except Exception as e:
        print(f"[WARN] {os.path.basename(path)} 처리 중 오류 → NONE: {e}")
        return dict(_NS["NONE4"])
