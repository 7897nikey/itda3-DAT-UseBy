# ITDA 3rd 학술제 - 소비기한 추출 (한국외대 DAT · UseBy팀) 📌

본 저장소는 **제3회 ITDA 연합학술제** 예선 제출용 저장소입니다. 공식 제출 템플릿을 기반으로, 우리 팀이 실제로 구현한 파이프라인 내용을 반영해 두었습니다.

---

## 1. 대회 개요 및 과제 정의

- **주제**: OCR 기반 상품 소비기한 정보 추출 아키텍처 설계 및 도메인 활용 기획
- **주최**: 수도권 데이터사이언스 연합학회 ITDA (경희대 CODE, 서강대 INSIGHT, 성균관대 DScover, 인하대 IBAS, 한국외대 DAT)
- **입력 (Input)**: 상품 뒷면 이미지 (`ITDA_INPUT_DIR` 환경변수로 경로 주입)
- **출력 (Output)**: `submission.csv` (`ITDA_OUTPUT_PATH` 환경변수 경로에 저장)

### submission.csv 표준 스키마

| image_id | year | month | day | final_date |
| --- | --- | --- | --- | --- |
| 1 | 2026 | 05 | 29 | 2026-05-29 |
| 2 | NONE | NONE | NONE | NONE |

- `image_id` : 확장자를 제외한 이미지 파일명 (예: 1)
- `year` : 4자리 연도 문자열 (예: 2026, 미인식 시 NONE)
- `month` : 2자리 월 문자열 (예: 05, 미인식 시 NONE)
- `day` : 2자리 일 문자열 (예: 29, 미인식 시 NONE)
- `final_date` : 하이픈(-)으로 연결된 정규화 날짜 (예: 2026-05-29, 미인식 시 NONE)

---

## 2. 우리 팀 파이프라인 개요

두 단계로 구성된 하이브리드 구조입니다.

```
입력 이미지
  │
  ├─▶ ① YOLOv8n(ONNX) 으로 "소비기한" 문구 영역 탐지·크롭
  │      │
  │      ├─ 탐지 성공 → 크롭된 작은 영역만 EasyOCR로 인식 (빠르고 정확)
  │      │
  │      └─ 탐지 실패/저신뢰도 → ② 원본 전체 이미지를 리사이즈해 EasyOCR로 인식 (폴백)
  │
  └─▶ 인식된 텍스트를 date_parser.py(정규식 기반 규칙 파서)에 통과시켜
       year / month / day / final_date 추출
```

- **문자 인식(OCR)**: EasyOCR — 사전학습된 공개 모델을 그대로 사용(직접 학습 안 함)
- **영역 탐지**: YOLOv8n — 팀이 직접 라벨링한 450장(`labeling/`, 제출물 아님)으로 **파인튜닝**한 2클래스(EXP=소비기한, MFG=제조일자) 검출기. 대회 규정상 "사전학습 공개 모델의 파인튜닝"은 허용되는 방식입니다. 검증셋(45장) 기준 mAP50 0.924.
  - 학습·데이터셋 코드는 `yolo_train/`에 있으며, 학습 자체는 EasyOCR 실행 환경과 별도인 `yolo_venv`(로컬 전용, 저장소에 포함 안 됨)에서 진행했습니다. 채점에는 영향 없습니다.
  - 학습된 모델은 `weights/yolo_exp_mfg.onnx`로 내보내(export) 두었습니다. `torch`/`ultralytics` 없이 `onnxruntime`만으로 추론하며, 이는 EasyOCR가 요구하는 torch 버전과의 충돌을 피하기 위함입니다.
- **날짜 파싱**: `date_parser.py` — 정규식 기반 규칙 파서(새 모델 학습 아님). 부터/까지 범위, 제조일로부터 N개월 등 상대기간 계산, 2자리/4자리 연도, 압축 표기(DDMMYY), 영문 월(JAN~DEC) 등 실측으로 확인된 표기 패턴을 처리합니다.
- **실측 결과** (팀 자체 라벨링 450장 기준, 직전 전체 파이프라인 검증): 완전일치 약 44%, 평균 처리속도 약 2초/장(500장 기준 예산 2400초 대비 여유 있음).

---

## 3. 파일 및 저장소 구조

```
itda3-DAT-UseBy/
├── predict.ipynb            # 메인 추론 노트북 (운영진 채점용 필수)
├── date_parser.py           # 정규식 기반 날짜 파서 (predict.ipynb가 import함, 필수)
├── requirements.txt         # 실행 환경 패키지 목록 (필수)
├── README.md                # 본 문서
├── .gitignore
├── download_weights.sh      # EasyOCR 사전학습 가중치 다운로드 스크립트
├── weights/
│   ├── .gitkeep
│   ├── yolo_exp_mfg.onnx    # 팀이 라벨링·파인튜닝한 YOLO 모델 (직접 커밋됨, 다운로드 불필요)
│   ├── craft_mlt_25k.pth    # EasyOCR 가중치 (download_weights.sh로 받음, 커밋 안 됨)
│   └── korean_g2.pth        # EasyOCR 가중치 (download_weights.sh로 받음, 커밋 안 됨)
├── yolo_train/               # YOLO 학습·검증용 스크립트 (채점 대상 아님, 참고용)
│   ├── build_dataset.py     # 팀 라벨링 결과 → YOLO 학습 포맷 변환
│   ├── train.py             # yolov8n 파인튜닝
│   ├── export_onnx.py       # 학습된 모델을 ONNX로 내보내기
│   └── predict_sample.py    # 검증셋에 예측 그려서 눈으로 확인
└── labeling/                 # 팀 자체 라벨링 산출물 (제출물 아님, gitignore 대상)
```

---

## 4. 시작하기 및 실행 방법

### 1) 가상환경 구축 및 패키지 설치

```
git clone <본인 팀 저장소 URL>
cd <저장소 디렉토리>
pip install -r requirements.txt
```

### 2) 가중치 파일 설정

- **EasyOCR 가중치**: `download_weights.sh`를 실행하면 `./weights`에 자동으로 받아집니다(용량이 커서 Git에 커밋하지 않음).
  ```
  bash download_weights.sh
  ```
- **YOLO 가중치**(`weights/yolo_exp_mfg.onnx`): 팀이 직접 학습한 작은 모델(약 12MB)이라 저장소에 이미 커밋되어 있습니다. 별도로 받을 필요 없습니다.

### 3) 채점 재현성 검증 (운영진 채점 표준 명령어)

운영진은 Standard 4-Core vCPU 환경에서 아래 명령어를 실행하여 순차 실행(Run All) 및 채점을 진행합니다.

```
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
```

---

## 5. ⚠️ 채점 환경 필수 공지 (반드시 읽어주세요)

### 1) 팀 저장소 공개 범위

- 팀 저장소는 **Public** 으로 생성해 주세요.
- Private 으로 운영할 경우, 마감 전까지 운영진 계정 **`b9511242000-blip`** 을 Collaborator 로 초대해야 합니다. (Settings → Collaborators → Add people)
- 마감 시각 기준 운영진이 접근할 수 없는 저장소는 채점 대상에서 제외됩니다.

### 2) 채점 서버는 오프라인입니다

채점은 **인터넷이 차단된 Standard 4-Core vCPU 환경**에서 진행됩니다.

- EasyOCR, PaddleOCR 등 상당수 라이브러리는 최초 실행 시 가중치를 인터넷에서 **자동 다운로드** 합니다. 오프라인 환경에서는 이 단계가 실패해 실행 오류(정량 0점)가 발생합니다.
- 모든 가중치는 **노트북 실행 전에 로컬에 존재**해야 합니다.
  - `download_weights.sh` 는 채점 실행 **전에** 운영진이 1회 실행합니다. (EasyOCR 가중치용)
  - `weights/yolo_exp_mfg.onnx`는 저장소에 이미 커밋되어 있어 별도 다운로드 단계가 필요 없습니다.
  - `predict.ipynb` 의 Run All **도중에** 다운로드하는 코드는 동작하지 않습니다.

EasyOCR 사용 예시:

```python
reader = easyocr.Reader(
    ['ko', 'en'], gpu=False,
    model_storage_directory='./weights',
    download_enabled=False,   # 오프라인 강제
)
```

YOLO(ONNX) 사용 예시:

```python
import onnxruntime as ort
session = ort.InferenceSession('./weights/yolo_exp_mfg.onnx', providers=['CPUExecutionProvider'])
```

네트워크를 끄고 Run All 이 끝까지 돌아가면 통과입니다. 제출 전 반드시 한 번 검증해 보세요.

### 3) 환경 설치 시간은 속도 점수에 포함되지 않습니다

- `pip install -r requirements.txt` 및 `download_weights.sh` 소요 시간은 속도 점수(10점) 산정에서 **제외** 됩니다.
- 속도 점수는 `predict.ipynb` 의 Run All 실행 시간(최대 2400초)만으로 산정합니다.

---

## 6. 제출 전 필수 체크리스트

1. **CONFIG 셀 수정 금지**: `predict.ipynb` 최상단의 환경변수 주입 코드는 절대 변경하거나 값을 직접 하드코딩 대입하지 마세요.
2. **대화형 코드 제거**: 실행 중 사용자 입력을 대기하는 코드(`input()`, `getpass()` 등)가 있으면 실행이 중단되어 정량 0점 처리됩니다.
3. **인덱스 제외 저장**: CSV 저장 시 반드시 인덱스를 제외해야 합니다. (`df.to_csv(OUTPUT_PATH, index=False)`)
4. **결과 스키마 준수**: 누락된 컬럼이 없도록 `image_id, year, month, day, final_date` 5개 컬럼 스키마를 엄격히 지켜주세요.
5. **오프라인 실행 검증**: 네트워크 차단 상태에서 Run All 이 완주하는지 확인하세요.
6. **저장소 접근 권한**: Public 설정 또는 운영진 계정 Collaborator 초대를 완료하세요.
7. **`date_parser.py`와 `weights/yolo_exp_mfg.onnx`가 저장소에 커밋되어 있는지 확인**: 둘 다 `predict.ipynb`가 직접 참조하는 필수 파일입니다. 빠지면 `ImportError`/`FileNotFoundError`로 실행이 처음부터 실패합니다.
