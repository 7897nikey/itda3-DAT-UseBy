# custom_data — 팀 자체 수집 데이터셋 (UseBy팀)

팀이 직접 촬영·라벨링한 **98장**의 평가 전용 데이터셋입니다. 학습에는 쓰지 않습니다.
두 묶음으로 구성되며, 각각 **노리는 실패 모드가 다릅니다.**

| 묶음 | image_id | 장수 | 성격 |
|---|---|---|---|
| 배치 1 — 제품군 확장 | `cust_0069`~`cust_0098` | 30 | 의약품(PTP·사용기한) / 영양제 / 화장품 / 일본·유럽 수입품. 대회 제공 데이터에 없는 **표기 체계**(사용기한·제조번호·`賞味期限`·`BEST BEFORE DD/MM/YYYY`) |
| 배치 2 — 촬영 난조건 | `cust_0001`~`cust_0068` | 68 | 편의점 현장 촬영. 컵라면·캔 뚜껑 **각인**, 원형 **곡면**, 도트 매트릭스, 날짜+시각+로트 동시 인쇄, 매대 조명 반사 |

배치 1은 **무엇이 쓰여 있는가**(파서·키워드 사전)를, 배치 2는 **어떻게 찍혔는가**(검출·인식)를 대상으로 합니다.

## 폴더 구조

```
custom_data/
├── images/                  # 촬영 원본을 리사이즈·EXIF 정리한 이미지 (cust_0001.jpg …)
├── labels.csv               # ★ 정답 라벨 (image_id, year, month, day, final_date, status, tag_*)
├── meta.csv                 # 수집 메타데이터 (제품군, 표기 용어, 인쇄 방식, 촬영 조건, 원본 제원)
├── parser_cases.csv         # 이미지 없이 돌리는 날짜 파서 회귀 케이스 (원문 표기 → 기대 출력)
├── docs/
│   ├── LABELING_GUIDE.md    # 라벨 판정 규칙 (무엇을 정답으로 볼 것인가)
│   └── COLLECTION_PROTOCOL.md # 무엇을 왜 찍었는가 (수집 설계)
└── tools/
    ├── prepare_images.py    # EXIF 회전 반영 + GPS 제거 + 리사이즈 + 파일명 표준화
    └── validate_labels.py   # labels.csv 무결성·형식 검증
```

## labels.csv 스키마

저장소 루트의 `evaluate.py`가 **그대로 읽을 수 있는 형식**입니다.
(`submission.csv` 스키마 + `status` + 난이도 태그 `tag_*`)

| 컬럼 | 값 | 설명 |
|---|---|---|
| `image_id` | `cust_0001` | 확장자 제외 파일명 |
| `year` | `2026` / `NONE` | 4자리 |
| `month` | `05` / `NONE` | 2자리 |
| `day` | `29` / `NONE` | 2자리 |
| `final_date` | `2026-05-29` / `2026-05-NONE` / `NONE-05-29` / `NONE` | 하이픈 연결 |
| `status` | `완료` / `보류` | `보류`는 채점에서 제외 (evaluate.py 동작과 동일) |
| `term` | `소비기한` `유통기한` `사용기한` `EXP` `賞味期限` … | 포장에 실제로 쓰인 용어 |
| `notation_raw` | `26.09.11까지` | 포장의 원문 표기를 **보이는 그대로** |
| `tag_*` | `TRUE` / `FALSE` | 난이도 태그 (아래) |
| `notes` | 자유 | 판정이 갈린 이유 등 |

난이도 태그: `tag_dot`(잉크젯 도트) `tag_engrave`(각인·음각) `tag_curved`(곡면)
`tag_lowcontrast`(저대비·반사) `tag_multidate`(제조일·로트 등 다중 숫자)
`tag_lotconfuse`(로트/제품번호 혼동) `tag_relative`(제조일로부터 N개월·개봉 후)
`tag_reference`(상단 표기일 참조형) `tag_ymonly`(연-월만 표기) `tag_foreign`(외국어 표기)
`tag_unreadable`(사람 눈으로도 판독 불가)

## 사용법

```bash
# 1) 촬영 원본 정리 → images/ 생성 + labels.csv·meta.csv 뼈대 생성
python custom_data/tools/prepare_images.py --src <촬영원본폴더> --dst custom_data

# 2) 라벨 작성 후 형식 검증
python custom_data/tools/validate_labels.py custom_data

# 3) 채점 (저장소 루트의 기존 스크립트 그대로)
python pipeline.py --input_dir custom_data/images --output_path pred_custom.csv --weights <onnx>
python evaluate.py pred_custom.csv custom_data/labels.csv "custom_data OOD"

# 4) 이미지 없이 파서만 회귀 테스트
python custom_data/tools/validate_labels.py custom_data --parser-cases
```

## 데이터 관리 원칙

- **학습 사용 금지.** 이 폴더의 이미지는 YOLO·OCR 어느 단계의 학습에도 넣지 않습니다.
  (대회 제공 라벨 450장을 학습에서 제외한 것과 같은 이유 — 성능 수치의 독립성 확보)
- 촬영은 팀원이 직접, 개인 소유 제품 또는 매장에서 촬영 허가를 받은 제품만.
- 사람·주소·가격표 등 개인정보가 찍힌 사진은 제외했고, `prepare_images.py`가
  **EXIF의 GPS를 포함한 모든 메타데이터를 제거**합니다.
- 이미지 용량: 배치 2는 긴 변 2048px / JPEG 품질 92 / 크로마 서브샘플링 없음(4:4:4).
  원본 169MB → 약 34MB, 최대 단일 파일 약 0.7MB로 GitHub 단일 파일 100MB 제한 대비 여유가 큽니다.
  배치 1(901×1600, 총 5.1MB)은 이미 축소된 전송본이라 **재인코딩 없이 바이트 그대로** 옮겼습니다
  (JPEG를 다시 저장하면 손실만 얹힙니다). 전체 98장 약 39MB.
  조합별 실측 표와 해상도를 2048로 정한 근거는 `docs/COLLECTION_PROTOCOL.md` §4-1.
  총량을 더 줄여야 하면 `--max-side 1600` (약 22MB). 원본은 저장소에 넣지 않습니다.
