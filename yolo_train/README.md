# yolo_train — 레거시 학습 스크립트

> ⚠️ **이 폴더는 현재 배포 검출기(`weights/region_best.onnx`)를 만든 코드가 아닙니다.**
> 초기 실험 단계의 산출물이며, 클래스 구성부터 배포본과 다릅니다. 자체 라벨링 절차의
> 코드 근거로만 참고하세요(EXIF 회전 보정 규약은 `pipeline.py`의 `load_bgr()`과 동일).

## 레거시 vs 배포본

| | 이 폴더 (레거시) | 배포본 (`weights/region_best.onnx`) |
|---|---|---|
| 클래스 | 2개 — `EXP`/`MFG` | 4개 — `date`/`due`/`code`/`full` |
| 모델 | YOLOv8n | YOLO11n |
| 입력 크기 | 640 | 960 |
| 학습 epoch | 80 | — (별도 실행, 이 폴더에 스크립트 없음) |
| 실행 엔진 | `ultralytics` (torch) | `onnxruntime` (torch 불필요) |

## 파일

| 파일 | 내용 |
|---|---|
| `build_dataset.py` | 라벨링 툴 JSON을 읽어 EXIF 회전을 픽셀에 반영(`ImageOps.exif_transpose`)한 뒤 YOLO 포맷으로 변환·분할 |
| `train.py` | 2클래스(EXP/MFG) YOLOv8n 파인튜닝. `epochs=80`, `imgsz=640`, `device="cpu"` |
| `export_onnx.py` | 학습된 `.pt`를 ONNX로 내보내는 스크립트 |
| `predict_sample.py` | 학습된 모델로 샘플 이미지 추론 |
| `dataset/data.yaml` | YOLO 학습용 데이터셋 설정 |
| `sample_preds/` | 추론 샘플 이미지 |

## 주의

- `BASE` 경로가 로컬 절대경로로 하드코딩되어 있어(`build_dataset.py`, `train.py`), 그대로는 다른 환경에서 실행되지 않습니다.
- 4클래스 배포본을 만든 학습 스크립트는 이 저장소에 없습니다.
