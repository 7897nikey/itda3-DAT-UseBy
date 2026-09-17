# sample/ — 검수용 최소 입력

운영진 검수 절차의 `ITDA_INPUT_DIR=./sample` 에 해당하는 폴더입니다.
대회 제공 이미지 중 **5장**을 골라 두었습니다. 파이프라인이 끝까지 도는지
확인하는 용도이고, 성능을 재는 용도가 아닙니다.

정답은 [`../docs/sample_expected.csv`](../docs/sample_expected.csv) 에 있습니다.
일부러 섞어 놓았습니다 — 5장 전부 맞히는 게 정상은 아니고, 실행이 오류 없이
끝나고 5행 5컬럼 CSV 가 나오면 통과입니다.

성능 수치를 확인하시려면 팀 손라벨 904건 기준 `labels_master.csv` 와
`split_eval.py` 를 쓰시면 됩니다.
