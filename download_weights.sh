#!/usr/bin/env bash
# EasyOCR 가중치를 ./weights 에 미리 받아둔다.
# 채점 실행(predict.ipynb Run All) 전에 운영진이 네트워크 있는 상태로 1회 실행.
# predict.ipynb 안의 easyocr.Reader(..., download_enabled=False)는 이 폴더에
# 이미 있는 파일만 쓰고 인터넷 접근을 시도하지 않는다.
set -euo pipefail

mkdir -p ./weights

python3 - <<'PY'
import easyocr
# download_enabled=True(기본값)로 한 번 생성하면 필요한 가중치를
# model_storage_directory에 전부 받아온다. 이후 predict.ipynb는
# 이 폴더를 오프라인으로 재사용한다.
easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory='./weights', download_enabled=True)
print("가중치 다운로드 완료: ./weights")
PY
