#!/usr/bin/env bash
# 가중치 준비 스크립트 (운영진 채점 실행 전 1회)
#
# 이 저장소는 가중치를 내려받지 않습니다. 두 파일 모두 저장소에 직접 커밋되어
# 있고 합쳐서 23MB입니다. 채점 서버가 인터넷이 차단된 환경이라, 실행 중에
# 무엇이든 받아오는 구조 자체를 두지 않는 쪽을 택했습니다.
#
# 그래서 이 스크립트가 하는 일은 내려받기가 아니라 '있어야 할 것이 있는지'
# 확인하는 것입니다. 네트워크를 전혀 쓰지 않으며, 파일이 누락되거나 내용이
# 바뀌었으면 여기서 멈춰 세웁니다. predict.ipynb 실행 도중에 발견하는 것보다
# 이 단계에서 걸리는 편이 낫기 때문입니다.
set -euo pipefail

cd "$(dirname "$0")"

EXPECT_REGION="c371e3c8346d0b3c67d51c5a157f72b79a8ad35202e467d6a53f7bddc81a55f6"
EXPECT_REC="cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

check() {
  local path="$1" expect="$2" label="$3"
  if [ ! -f "$path" ]; then
    echo "[실패] $label 이 없습니다: $path" >&2
    echo "       git lfs 없이 직접 커밋된 파일입니다. clone 이 온전한지 확인해 주세요." >&2
    exit 1
  fi
  local got
  got="$(sha256_of "$path")"
  if [ "$got" != "$expect" ]; then
    echo "[실패] $label 의 내용이 제출 시점과 다릅니다: $path" >&2
    echo "       기대 $expect" >&2
    echo "       실제 $got" >&2
    exit 1
  fi
  printf '[확인] %-38s %6.1fMB  sha256 일치\n' "$label" \
    "$(awk -v b="$(wc -c < "$path")" 'BEGIN{printf "%.1f", b/1048576}')"
}

echo "가중치 확인 (네트워크 사용 안 함)"
check "weights/region_best.onnx"             "$EXPECT_REGION" "YOLO11n 영역 검출기"
check "weights/korean_PP-OCRv5_rec_mobile.onnx" "$EXPECT_REC"  "RapidOCR 한국어 인식 모델"
echo "두 개 모두 준비됨. predict.ipynb 를 실행하셔도 됩니다."
