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

**앞 단계가 실패한 것만 다음 단계로 넘기는 6단 폴백** 구조입니다. 보정을 전체 사진에 한꺼번에 걸면 원래 잘 읽히던 것까지 망가질 수 있는데, 실패분에만 걸면 그럴 일이 없습니다.

```
입력 이미지
  │
  ├─ [0] EXIF 회전 보정 — 세로로 찍었는데 파일엔 가로로 저장된 사진이 8.7%
  │       있음. 안 맞추면 글자가 누운 채로 들어가 인식률이 반토막 남.
  │
  ├─ [1] YOLO11n(4클래스: date/due/code/full)로 날짜 영역 탐지
  │       code(품목보고번호·전화번호처럼 날짜로 헷갈리기 쉬운 숫자)를 따로
  │       가르쳐서, 찾아낸 뒤 버리는 용도로 씀 — 날짜 오인식을 줄임.
  │
  ├─ [2] 그 영역만 원본 해상도에서 크롭 (여유분 45%) — 글자 크기 보존
  │
  ├─ [3] 6단 폴백으로 읽기 (실패한 것만 다음 단계로)
  │       1차  크롭 그대로 RapidOCR
  │       2차  크롭 + 사진 보정(CLAHE 대비 향상 + 끊긴 점 이어붙이기)
  │       3차  크롭 + 도트프린터 전용 보정(오츠 이진화 후 팽창)
  │       4차  date/due 타이트 크롭을 RapidOCR 자체 탐지 없이 바로 인식
  │             — 탐지 단계가 글자를 놓쳐 뒤로 못 넘어가던 건을 구제
  │       5차  원본 전체 사진
  │       6차  원본 전체 사진 + 사진 보정
  │
  │       연·월·일을 다 채운 답이 나오면 거기서 끊고, 아니면 끝까지 돌면서
  │       가장 많이 채운 답을 남김. 이른 단계의 '연-월만' 같은 부분 답이
  │       뒤 단계의 완전한 날짜를 막지 않게 하기 위함.
  │
  ├─ [4] 읽어낸 글자를 date_parser.py(정규식 규칙 파서)로 날짜 변환
  │       실패한 건에만 보강을 추가로 태움(date_parser_plus.py):
  │       별칭 통일(사용기한→소비기한) → OCR 오독 글자 교정 재시도 →
  │       구분자 없는 압축 표기 → 구분자 없는 전숫자 날짜 → 연-월만이라도 추출
  │
  └─ [5] 그래도 실패하면 월-일만이라도 제출

결과: image_id, year, month, day, final_date
```

- **문자 인식(OCR)**: [RapidOCR](https://github.com/RapidAI/RapidOCR) — `onnxruntime`만으로 동작, torch 불필요. 인식 모델은 한국어 특화 사전학습 공개 모델(`korean_PP-OCRv5_rec_mobile.onnx`)을 그대로 사용(직접 학습 안 함).
- **영역 탐지**: YOLO11n — **팀이 직접 박스를 라벨링한 자체 검출 데이터셋**으로 파인튜닝한 4클래스(date/due/code/full) 검출기. 라벨링 툴 결과를 EXIF 회전까지 반영해 YOLO 포맷으로 변환한 뒤 학습·검증으로 나눴고, 자체 검증 분할 기준 평균 mAP50 0.916. 원래 `.pt`(torch)로 내보냈던 걸 `region_best.onnx`로 다시 내보내 RapidOCR과 같은 onnxruntime 엔진으로 통일함 — torch 의존성이 완전히 사라지고 검출 속도가 7.1배 빨라짐(검출 결과는 동일).
- **날짜 파싱**: `date_parser.py` — 정규식 기반 규칙 파서(새 모델 학습 아님). 부터/까지 범위, 제조일로부터 N개월 등 상대기간 계산, 2자리/4자리 연도, 압축 표기(DDMMYY), 영문 월(JAN–DEC) 등 실측으로 확인된 표기 패턴을 처리한다. 그 위에 `date_parser_plus.py`가 실패건에만 보강 레이어를 얹는다(원래 맞던 건은 안 건드림).
- **실측 결과** (팀 손라벨 904장. 네트워크를 끊은 상태에서 `predict.ipynb`를 그대로 실행해 측정):

  | 집합 | n | 부분점수 | 완전일치 | 응답률 |
  |---|---|---|---|---|
  | 전체 | 904 | **85.07%** | 80.53% | 91.8% |
  | A–C (검출기 학습 + 개발에 노출됨) | 449 | 89.09% | 84.41% | 95.8% |
  | D+E (튜닝 판단용) | 303 | 79.43% | 74.26% | 87.5% |
  | **F (한 번도 보지 않음)** | 152 | **84.43%** | 81.58% | 88.8% |

  속도는 장당 0.56–0.63초 — 500장 환산 시 예산 2400초의 12–13%.

  **보고할 숫자는 전체 85.07%입니다.** 다만 A–C 449장은 검출기 박스 라벨링에 쓴 이미지이자
  개발 과정에서 실패 사례를 반복해 들여다보며 규칙을 고친 집단이라 점수가 부풀려져 있습니다.
  **D+E·F 455장은 검출기 학습에 한 장도 쓰지 않은 독립 구간**입니다. 그래서 배치 F를
  봉인해 두고 제출본이 확정된 뒤에만 열었습니다. 이번 개선의 효과가 튜닝에 쓴 D+E에서는
  +4.29%p였는데 한 번도 보지 않은 F에서는 +1.97%p였습니다. 퇴행은 양쪽 모두 0건이라
  방향은 확실하지만 크기는 절반 이하입니다. 규약은 [`splits/README_holdout.md`](splits/README_holdout.md)에 있고
  `split_eval.py`가 이를 강제합니다.

- **실행 견고성**: 시작하자마자 전부 `NONE`인 CSV를 깔고 50장마다 덮어쓰기 때문에
  마지막 장 직전에 프로세스가 죽어도 형식이 맞는 제출물이 남습니다. 추정 소요가 예산을
  넘으면 폴백을 끄는 시간 가드와, 크롭 긴 변을 4000px로 자르는 상한도 들어 있습니다.
  네트워크를 차단한 상태에서 7회 연속 완주를 확인했습니다(종료코드 0, 에러 0건).
- 자세한 실측 분석(어디서 시간이 쓰이는지, 뭘 시도했다가 소용없었는지, 남은 실패 원인 등)은 [`docs/rapidocr_notes.md`](docs/rapidocr_notes.md)에 정리되어 있음.

---

## 3. 파일 및 저장소 구조

```
itda3-DAT-UseBy/
├── predict.ipynb            # 메인 추론 노트북 (운영진 채점용 필수)
├── pipeline.py               # 검출-크롭-인식-파싱 전체를 묶은 본체
├── date_parser.py            # 정규식 기반 날짜 파서 (핵심, 필수)
├── date_parser_plus.py       # date_parser.py를 가져다 쓰고 실패분에 보강을 더하는 층
├── gap_fill.py                # date_parser.py가 못 잡는 표기 보강
├── ocr_normalize.py           # OCR이 헷갈린 글자(O/0, I/1 등) 교정
├── partial_rescue.py          # 완전 실패 시 월-일만이라도 건짐
├── preprocess.py              # 대비 향상(CLAHE) + 끊긴 글자 이어붙이기
├── config_ko.yaml             # RapidOCR 설정
├── evaluate.py                 # 예측 CSV와 정답 라벨을 맞춰 점수 계산 (팀 자체 검증용)
├── split_eval.py               # 홀드아웃 규약대로 집합을 갈라 채점 (F는 --show-f 필요)
├── splits/                     # 홀드아웃 규약과 분할 정의
├── requirements.txt           # 실행 환경 패키지 목록 (필수)
├── README.md                  # 본 문서
├── .gitignore
├── docs/
│   └── rapidocr_notes.md      # 파이프라인 설계·실측 분석 상세 노트
├── weights/
│   ├── .gitkeep
│   ├── region_best.onnx            # 팀이 학습한 YOLO11n 검출 모델 (직접 커밋)
│   └── korean_PP-OCRv5_rec_mobile.onnx  # RapidOCR 한국어 인식 모델 (공개 모델, 오프라인 대비 커밋)
└── custom_data/                # 가산점용 팀 자체 수집 데이터 (평가 전용, 학습 미사용)
    ├── images/                 # 직접 촬영한 소비기한 사진 97장 (cust_0001–cust_0098)
    ├── labels.csv              # ★ 정답 라벨 — image_id, year, month, day, final_date, status, 난이도 태그
    ├── meta.csv                # 수집 메타 (제품군, 표기 용어, 인쇄 방식, 원본 제원)
    ├── parser_cases.csv        # 이미지 없이 돌리는 날짜 파서 회귀 케이스
    ├── docs/                   # 라벨 판정 규칙 · 수집 설계 근거
    └── tools/                  # 전처리 · 라벨 검증 스크립트
```

`custom_data/`: 팀이 직접 촬영·라벨링한 자체 수집 데이터 **97장**. 정답 라벨은 `custom_data/labels.csv`, 사진은 `custom_data/images/`이며, 수집 설계와 라벨 판정 규칙은 `custom_data/docs/`에 있습니다. **학습에는 쓰지 않은 평가 전용 셋**입니다.

두 묶음으로 나뉘고 노리는 실패 모드가 다릅니다. **배치 1**(`cust_0069`–`cust_0098`, 30장)은 의약품 PTP·사용기한, 영양제, 화장품·앰플, 일본 `賞味期限`, 유럽 `BEST BEFORE DD/MM/YYYY` 등 **표기 체계** 다양성을 검증하고, **배치 2**(`cust_0001`–`cust_0068`, 67장)는 편의점 현장에서 찍은 뚜껑 각인·원형 곡면·도트 매트릭스·날짜+시각+로트 동시 인쇄 등 **촬영 조건** 다양성을 검증합니다. 성격이 다르므로 합산 점수로 보고하지 않습니다.

```bash
python pipeline.py --input_dir custom_data/images --output_path pred_custom.csv --weights weights/region_best.onnx
python evaluate.py pred_custom.csv custom_data/labels.csv "custom_data"
```

---

## 4. 시작하기 및 실행 방법

### 1) 가상환경 구축 및 패키지 설치

````
git clone https://github.com/7897nikey/itda3-DAT-UseBy.git
cd itda3-DAT-UseBy
pip install -r requirements.txt
````

### 2) 가중치 파일 설정

용량이 큰 모델 가중치 파일(`.pt`, `.pth`, `.safetensors` 등)은 Git에 직접 푸시하지 마시고, Google Drive, HuggingFace 링크 또는 Release Assets를 통해 `download_weights.sh` 스크립트 등으로 내려받도록 설정하세요.

### 3) 채점 재현성 검증 (운영진 채점 표준 명령어)

운영진은 Standard 4-Core vCPU 환경에서 아래 명령어를 실행하여 순차 실행(Run All) 및 채점을 진행합니다.

````
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
````

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
  - `download_weights.sh` 는 채점 실행 **전에** 운영진이 1회 실행합니다.
  - `predict.ipynb` 의 Run All **도중에** 다운로드하는 코드는 동작하지 않습니다.

EasyOCR 사용 예시:

````python
reader = easyocr.Reader(
    ['en'], gpu=False,
    model_storage_directory='./weights',
    download_enabled=False,   # 오프라인 강제
)
````

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
