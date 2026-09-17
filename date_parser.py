# -*- coding: utf-8 -*-
"""
date_parser.py — 상품 이미지 OCR 텍스트에서 소비기한(year/month/day)을 추출하는 규칙 엔진.

ITDA OCR Challenge 1차 스크래치(A안: 전체 이미지 OCR + 정규식 후처리)의 파싱 단.
predict.ipynb 에서 `from date_parser import extract_expiry_fields` 로 불러 쓴다.

설계 원칙
---------
- "인쇄된 것만 읽는다. 없는 건 만들지 않는다" (운영진 확인 원칙) — 근거 숫자가
  전부 인쇄돼 있지 않으면(예: 기준 날짜 자체가 안 보임) 계산하지 않는다.
- 단, 제조일자와 기간(N개월/N년)이 **둘 다 인쇄**돼 있으면 그 합은 단순 산수이지
  "지어내는" 게 아니라고 보고 계산한다 — 팀 결정 (2026-09-09, RELATIVE_EXPR_AS_NONE
  = False로 전환). 실측 근거: `001958.jpg`(제조 2020.11.13 + 유통기한 제조일로부터
  18개월 → 2022-05-13), `001686.jpg`(제조 2020.11.27 + 제조일로부터 5년 →
  2025-11-27). 운영진 확인 전까지는 팀 자체 판단.
- 연도가 없어도 월/일만 있으면 채운다 — 운영진 확인 사례: `000995.jpg` 정답이
  `NONE | 02 | 18 | NONE-02-18` 이었음. year 따로, month/day 따로 판정한다.
- 확정 규칙: "부터 … 까지" 2단 표기(제조/포장 시각 → 소비기한)에서 정답은
  더 늦은(=까지에 해당하는) 날짜다. (`000081.jpg`, `000320.jpg` 검증)
- 확정 규칙: 구분자 없이 붙은 6자리(예: "050926")는 DD-MM-YY 순서. (운영진 확인)
- 확정 규칙: 아무 라벨 없이 날짜가 하나만 인쇄돼 있으면 그걸 소비기한으로 본다.
  (운영진 확인)

정책 스위치 (팀 회의로 아직 확정 안 된 항목)
-------------------------------------------
아래는 ITDA_컨텍스트.md §6 "미확정" 목록의 1번과 대응된다.
회의 결과가 나오면 로직을 다시 짤 필요 없이 이 값만 바꾸면 된다.

미확정2("소비기한이 제조일자보다 우선")는 스위치로 안 뒀다 — 이건 사실상
이견이 없는 항목이라, "EXP로 분류된 후보가 있으면 무조건 그것부터 쓰고,
전부 MFG로만 분류되면 아예 채택하지 않는다"는 규칙으로 코드에 고정해뒀다.

미확정3("제조일로부터 N개월 계산 여부")도 팀 결정으로 스위치를 껐다(계산함).
운영진이 나중에 "계산하지 말라"고 확인해주면 RELATIVE_EXPR_AS_NONE만 True로
되돌리면 된다 — 계산 로직 자체를 지울 필요 없음.
"""
import re
import os
from datetime import date, timedelta
import calendar

# ============================================================
# 정책 스위치
# ============================================================
TREAT_DISTRIB_AS_ANSWER = True   # 미확정1: 유통기한만 있으면 그 날짜를 정답으로 채택할지
RELATIVE_EXPR_AS_NONE = False    # 미확정3: "제조일로부터 N개월"류를 계산할지.
                                  # 팀 결정(2026-09-09)으로 계산하는 쪽으로 전환함.

# ============================================================
# 키워드
# ============================================================
KW_CONSUME = ["소비기한"]
KW_DISTRIB = ["유통기한"]
KW_MFG = ["제조일자", "제조년월일", "제조일", "제조", "PROD", "PRD", "MFG", "PR0D", "PR0"]  # PR0D/PR0: OCR이 PROD/PRO의 O를 0으로 자주 오독
KW_EXP_EN = ["EXP"]
ALL_KEYWORDS = KW_CONSUME + KW_DISTRIB + KW_MFG + KW_EXP_EN

# QA/설명용 — 최종 판정에는 관여하지 않고 explain()에서만 참고 정보로 노출
RELATIVE_PATTERNS = [
    r"제조일(로부터|자로부터|부터)?\s*\d+\s*(일|개월|주|년)",
    r"구입\s*후\s*\d+\s*(일|개월)\s*이내",
]

# 실제 계산에 쓰는 것 — N과 단위를 그룹으로 뽑아낸다.
_RELATIVE_PERIOD = re.compile(r"제조일(?:로부터|자로부터|부터)?\s*(\d+)\s*(일|주|개월|년)")
REFERENCE_PATTERNS = [
    r"상단\s*(에)?\s*(표기|기재|별도)",
    r"하단\s*(에)?\s*(표기|기재|별도)",
    r"별도\s*(표기|기재)",
    r"뒷면\s*(에)?\s*(표시|표기|기재)",
    r"표시일\s*(까지|전)",
    r"표기일\s*(까지|전)",
]

MONTH_NAME = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# ============================================================
# 날짜 후보 탐지 정규식
#   (?<!\d) / (?!\d) 경계를 반드시 건다 — 없으면 품목보고번호처럼 길게 이어지는
#   숫자열 중간을 잘라 가짜 날짜로 오탐한다
#   (실측 사례: "19930439002-406" -> "439002-40"을 날짜로 오인).
# ============================================================
# 구분자 문자 집합. "," 도 포함 — 실측: OCR이 "2025.05.19"의 마지막 점을
# 쉼표로 오독해 "202505,19"(연+월 붙고, 일 앞에만 쉼표)로 나온 사례.
# 쉼표를 널리 구분자로 인정해도 위험이 적은 이유: "1,000"처럼 콤마가 쓰이는
# 숫자는 보통 3자리씩 끊는데, 우리 패턴은 4자리 연도로 시작해야 매치되니까
# 겹칠 일이 거의 없음.
_SEP = r"[.\-/,]"

# 둘 중 최소 하나 자리는 진짜 구두점 구분자가 있어야 매치되게 해서, 완전히
# 구분자 없는 임의의 8자리 숫자열까지 날짜로 오인하는 건 막는다(4자리 연도가
# 2015~2035 안에 들어야 한다는 강한 앵커가 있긴 하지만, 그래도 안전판으로
# 둠). 나머지 한쪽 자리는 구두점(공백 허용)이든, 구두점 없이 공백만이든,
# 아예 없든(붙어있음) 다 허용 — "선택적"의 세 갈래.
_FLEX_GAP = r"(?:\s*" + _SEP + r"\s*|\s+)?"

_YMD_NUMERIC = re.compile(
    # 연-월 구분자가 점이 아니라 공백으로 오독되는 경우 허용
    # (실측: OCR이 "2026.08.04"를 "2026 08.04"로 읽음) → 첫 구분자는 선택적
    r"(?<!\d)(\d{4})" + _FLEX_GAP + r"(\d{1,2})\s*" + _SEP + r"\s*(\d{1,2})(?!\d)"
)
# 위 패턴의 반대 경우 — 연-월 구분자는 있는데 월-일 구분자가 없거나 공백뿐인
# 경우 (실측: "2025.0922 부터 2025.12211"은 구분자 자체가 없는 케이스,
# "유통기한 2021 . 01 14"는 연-월은 점인데 월-일은 공백만인 케이스 — 원래
# 여기서 연도 뒤에 공백 없이 구분자가 와야 한다고 짜여있어서 못 잡던 버그도
# 같이 고침(001219.jpg 실측)).
_YMD_NUMERIC_LOOSE_DAY = re.compile(
    r"(?<!\d)(\d{4})\s*" + _SEP + r"\s*(\d{1,2})" + _FLEX_GAP + r"(\d{1,2})(?!\d)"
)
# 2자리 연도는 4자리 연도보다 앵커가 약해서(연도 그럴듯함 판정 기준이 훨씬
# 헐겁게 통과됨), 구분자 자체를 생략할 수 있게 풀어주면 무관한 숫자열을
# 가짜 날짜로 오인하는 사례가 실측으로 나옴(001369.jpg: 영양정보 잡음
# "233.1A27 10.07"이 2027-10-07로 잘못 매치되어, 진짜 정답 25.09.21보다
# "더 늦은 날짜" 우선순위 규칙에 밀려 이겨버림). 그래서 두 구분자 모두
# 진짜 구두점은 유지하고, 앞뒤 공백만 허용한다(실측: 000022.jpg "25. 10.14").
_YMD_2DIGIT = re.compile(
    r"(?<!\d)(\d{2})\s*" + _SEP + r"\s*(\d{1,2})\s*" + _SEP + r"\s*(\d{1,2})(?!\d)"
)
# 위 패턴의 반대 경우 — 연-월 구분자는 있는데 월-일 구분자가 공백뿐인 경우
# (실측: 000022.jpg "25. 10.14", 000023.jpg "25. 10 10" — 기존엔 구분자
# 앞뒤로 공백을 전혀 허용 안 해서 둘 다 놓치고 있었음). 월-일 사이는
# _FLEX_GAP이 아니라 "완전히 붙어있는 경우는 제외"한 버전을 쓴다 — 4자리
# 연도 패턴과 달리 2자리 연도는 "붙어있는 월일"의 실측 근거가 없었고,
# 오히려 허용했더니 "16.42"(월1일6이 아니라 그냥 다른 숫자)를 "16.4.2"로
# 잘못 쪼개 읽는 회귀가 실측으로 남(000404.jpg, 001995.jpg). 연-월 자리는
# 여전히 진짜 구두점을 요구해서(2자리 연도는 4자리보다 우연히 맞을 확률이
# 높아 앵커가 약하므로) 오탐 위험을 관리한다.
_GAP_NO_GLUE = r"(?:\s*" + _SEP + r"\s*|\s+)"
_YMD_2DIGIT_LOOSE_DAY = re.compile(
    r"(?<!\d)(\d{2})\s*" + _SEP + r"\s*(\d{1,2})" + _GAP_NO_GLUE + r"(\d{1,2})(?!\d)"
)
_YMD_KOREAN = re.compile(
    r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)
# 영문 월(3글자) 표기 주변 구분자 — 공백/콤마/대시 중 뭐든 오거나 아예 없어도
# 됨(실측: "OCT19-2021"처럼 월 이름과 일자 사이 공백이 아예 없고, 일자와
# 연도 사이는 대시인 경우 확인됨, 001509.jpg). 월 이름 자체가 3글자 정확히
# 일치해야 하는 강한 앵커라 구분자를 느슨하게 풀어도 오탐 위험은 낮음.
_ENG_SEP = r"[\s,\-]*"
_DMY_ENGLISH = re.compile(
    r"(?<!\d)(\d{1,2})" + _ENG_SEP + r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)" + _ENG_SEP + r"(\d{4})(?!\d)",
    re.IGNORECASE,
)
# 월(영문 3글자) - 일 - 4자리연도 순서 (미국식, 예: "AUG 27 2021", "AUG 27, 2021").
# _DMY_ENGLISH(일-월-년)와 그룹 순서만 반대.
_MDY_ENGLISH = re.compile(
    r"(?<!\d)(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)" + _ENG_SEP + r"(\d{1,2})" + _ENG_SEP + r"(\d{4})(?!\d)",
    re.IGNORECASE,
)
# 연도 - 월(영문 3글자) - 일 순서 (예: "2021 AUG 27"). ISO 스타일 연도 선두 +
# 영문 월 표기 혼합형. 실측으로 세 순서(일-월-년/월-일-년/년-월-일) 전부
# 발견되어 셋 다 따로 둔다.
_YMD_ENGLISH = re.compile(
    r"(?<!\d)(\d{4})" + _ENG_SEP + r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)" + _ENG_SEP + r"(\d{1,2})(?!\d)",
    re.IGNORECASE,
)
# 일/월/4자리연도 순서 (예: "20/05/2026"). 확인된 DD-MM-YY(6자리 압축형) 규칙과
# 같은 순서로, 수입품 등에 흔한 국제식 표기. _YMD_NUMERIC(연도가 맨 앞, 4자리)
# 이나 _YMD_2DIGIT(연도가 2자리)과는 자릿수 조합이 달라 서로 안 겹친다.
# 구분자는 "./,-" 중 하나 또는 공백만(OCR이 구분자 자체를 못 읽고 띄어쓰기만
# 남기는 경우가 실측으로 확인됨)도 허용한다. 콤마도 여기 포함시킨 이유는
# _YMD_NUMERIC류와 달리 뒤에 항상 4자리 연도(_valid_ymd로 2015~2035 검증)라는
# 강한 앵커가 있어서 오탐 위험이 낮기 때문 — _MD_ONLY처럼 앵커가 약한 패턴과는
# 다르게 판단함. 완전히 붙어있는 경우(구분자 자체가 없는 "27032022")는 이
# 패턴이 아니라 아래 _DDMMYYYY_COMPACT(키워드 근처 한정)가 처리한다.
_DMY_NUMERIC_4Y = re.compile(
    r"(?<!\d)(\d{1,2})(?:\s*[.\-/,]\s*|\s+)(\d{1,2})(?:\s*[.\-/,]\s*|\s+)(\d{4})(?!\d)"
)
# 연-월-일 사이에 구분자가 아예 없고 공백만 있는 경우 (예: "2026 10 29") —
# 실측(000009.jpg): 큰 사진이라 전체이미지 방식은 cv2.resize 에러로 통째로
# 실패했는데, YOLO로 소비기한 영역만 크롭하고 나니 OCR이 "2026 10 29 0"로
# 거의 완벽하게 읽음. 근데 양쪽 다 구분자가 없어서(_DMY_NUMERIC_4Y와 달리
# 한쪽도 구두점이 없음) 오탐 위험이 더 크므로, 압축 6/8자리 패턴과 똑같이
# 기한류 키워드 근처(±30자)에서만 찾는다.
_YMD_SPACE_ONLY = re.compile(r"(?<!\d)(\d{4})\s+(\d{1,2})\s+(\d{1,2})(?!\d)")
# 연도 없는 월.일 (예: "02.18") — 영양정보 수치(예: "2.8g")나 보관온도 범위
# (예: "1-30℃", "0~10℃")와 구분하려고 뒤에 단위/기호가 오면 제외한다.
# "℃" 기호 자체를 OCR이 그냥 "C"/"c"로 읽는 경우가 많아 그것도 같이 막는다
# (실측: 001510.jpg "실온 보관(1-30℃)"를 EasyOCR이 "1-30C)"로 읽어서
# 01월30일처럼 보이는 가짜 날짜가 됨).
# 구분자에서 "-"는 아예 뺐다 — 도로명주소 지번("OO로 9-2", "OO길 12-3")이
# 키워드 근처(±40자)에 흔히 등장해서 "9-2"가 "9월 2일"로 오탐되는 사례를
# 실측으로 확인함(002992.jpg: "백삼로 9-2" → 09/02로 오판). 연도 없는
# 월-일 표기 자체가 실제 사례에서 "."/"/" 위주였고 "-"로 확인된 진짜 사례는
# 없어서, 주소 오탐을 막는 쪽이 이득이라고 판단.
_MD_ONLY = re.compile(
    r"(?<!\d)(0?[1-9]|1[0-2])[./](0?[1-9]|[12]\d|3[01])(?!\d)"
    r"(?!\s*(g|kg|mg|%|kcal|℃|도|°|[Cc]\)?))"
)
# 구분자 없이 붙어있는 8자리 (예: "27032022") — 위 _DMY_NUMERIC_4Y와 같은
# 일-월-년 순서(4자리 연도)인데 구분자 자체가 아예 없는 경우. 임의의 8자리
# 숫자는 품목보고번호 등과 안 겹치므로 6자리 압축형과 동일하게 기한류 키워드
# 근처(±30자)에서만 찾는다.
_DDMMYYYY_COMPACT = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")
# 구분자 없이 붙어있는 6자리 (예: "050926") — 운영진 확인: 이/월/년(DD-MM-YY)
# 순서로 인쇄된다. 팀 회의 지침대로 이 형식은 확인됨.
# 임의의 6자리 숫자는 전화번호/바코드/로트번호 조각과 구분이 안 되므로,
# _find_full_candidates 안에서 기한류 키워드 근처(±30자)에서만 찾는다 —
# 텍스트 전체에서 찾으면 오탐 위험이 너무 큼.
_DDMMYY_COMPACT = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")


# 이 데이터셋 성격상(최근 촬영된 상품 사진 + 최대 수년 상대기간 계산) 나올 수
# 있는 연도의 현실적인 범위. 실측 버그: 바코드/로트번호 숫자가 우연히 날짜
# 모양이 되면서 "0728년", "2037년"처럼 말도 안 되는 연도가 답으로 나온 적
# 있음 — Python의 date()는 이런 값도 "유효한 달력 날짜"라 그냥 통과시켜버림.
MIN_PLAUSIBLE_YEAR = 2015
MAX_PLAUSIBLE_YEAR = 2035


def _valid_ymd(y, m, d):
    if not (MIN_PLAUSIBLE_YEAR <= y <= MAX_PLAUSIBLE_YEAR):
        return False
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def _normalize_2digit_year(yy):
    # 2자리 연도는 이 대회 데이터 특성상 전부 최근 상품이라 20xx로 가정.
    # (1900년대로 해석해야 하는 케이스는 이 데이터셋 성격상 없다고 봄 — v1 가정)
    return 2000 + yy


# ── 2자리 연도 표기에서 어느 자리가 연도인가 ──
# "24/12/21" 은 연-월-일로 읽으면 2024-12-21, 일-월-연으로 읽으면 2021-12-24 다.
# 둘 다 달력상 유효해서 숫자만으로는 구분이 안 된다. 예전에는 이럴 때 무조건
# 연-월-일을 골랐는데, 실측 실패 122건 중 9건이 전부 이 지점에서 틀렸다.
#
# 손라벨의 연도 분포가 심하게 치우쳐 있다는 점을 쓴다. 상품 사진이 2025년
# 하반기에 찍혔으니 소비기한은 2021~2022(이미 지난 것)와 2026(살아 있는 것)에
# 몰리고, 2016·2017·2019·2024·2030 같은 해는 거의 나오지 않는다.
# 아래 분포는 A~C + D+E 741건에서만 뽑았다. 배치 F 와 홀드아웃 G 는 뺐다 —
# 판단에 쓰는 수치를 평가에 쓸 집합에서 만들면 안 되기 때문이다.
_YEAR_PRIOR = {
    2019: 0.0094, 2020: 0.0499, 2021: 0.3900, 2022: 0.2051, 2023: 0.0526,
    2024: 0.0067, 2025: 0.0513, 2026: 0.1849, 2027: 0.0418, 2028: 0.0081,
}
_YEAR_PRIOR_FLOOR = 0.002   # 분포에 없는 해(2016·2030 등)에 줄 최소값

# 뒤집으려면 상대 해석보다 이만큼은 그럴듯해야 한다. 1.0 으로 두면 근소한
# 차이에도 뒤집혀서 원래 맞던 것까지 건드린다. 실측으로 정한 값이다.
_ORDER_FLIP_RATIO = float(os.environ.get("ITDA_ORDER_FLIP_RATIO", "6.0"))


def _year_prior(y):
    return _YEAR_PRIOR.get(y, _YEAR_PRIOR_FLOOR)


def _prefer_dmy_by_year_prior(y_ymd, y_dmy):
    """연-월-일 해석과 일-월-연 해석이 둘 다 유효할 때 뒤집을지 판단."""
    return _year_prior(y_dmy) >= _ORDER_FLIP_RATIO * _year_prior(y_ymd)


# ── 라벨에 찍힌 "읽는 법" 안내 ──
# 소비기한 문구 근처가 아니라 라벨 설명문 어딘가에 표기 순서를 직접
# 알려주는 경우가 실제로 많다(예: 000379.jpg "읽는 법 (일월년 순)",
# 001338.jpg "일월년순까지", 002645.jpg "(년월일 순)", 000449.jpg
# "표기일-월-년 순", 002637.jpg "연월일춘"(순의 OCR 오독)). "순" 글자 자체는
# OCR이 숲/신/춘 등으로 잘못 읽는 사례가 많아 필수로 요구하지 않고, 순서
# 3글자 자체만 본다.
_ORDER_HINT_YMD = re.compile(r"[연년][.\-\s]?월[.\-\s]?일")
_ORDER_HINT_DMY = re.compile(r"일[.\-\s]?월[.\-\s]?[연년]")


def _detect_order_hint(text):
    """텍스트 전체에서 표기 순서 안내를 찾는다. 'ymd'/'dmy'/None 반환.
    둘 다 나오거나 둘 다 안 나오면 판단 불가로 보고 None — 이 경우 기존
    다수결(연-월-일 우선) 규칙을 그대로 쓴다. 2자리 연도 표기가 연-월-일/
    일-월-연 둘 다로 유효하게 해석되는 진짜 애매한 경우에만 이 힌트를 쓴다
    (범위상 한쪽으로만 해석 가능한 경우는 애초에 힌트가 필요 없음)."""
    has_ymd = bool(_ORDER_HINT_YMD.search(text))
    has_dmy = bool(_ORDER_HINT_DMY.search(text))
    if has_ymd and not has_dmy:
        return "ymd"
    if has_dmy and not has_ymd:
        return "dmy"
    return None


def _find_full_candidates(text):
    """연-월-일이 전부 있는 날짜 후보 리스트. [{start,end,year,month,day}, ...].
    겹치는 매치는 먼저 매치된 것 우선으로 제거."""
    spans = []
    for m in _YMD_NUMERIC.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))
    for m in _YMD_NUMERIC_LOOSE_DAY.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))
    order_hint = _detect_order_hint(text)  # 라벨 전체 기준, 후보마다 반복계산 안 함

    def _add_2digit_year_candidate(m):
        yy, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y_ymd = _normalize_2digit_year(yy)   # 연-월-일로 읽었을 때
        y_dmy = _normalize_2digit_year(d)    # 일-월-연으로 읽었을 때(순서 반전)
        ymd_ok = _valid_ymd(y_ymd, mo, d)
        dmy_ok = _valid_ymd(y_dmy, mo, yy)
        if ymd_ok and dmy_ok:
            # 둘 다 유효한 진짜 애매한 경우(두 수 모두 15~31 겹침구간).
            # 라벨에 읽는 법이 명시돼 있으면 그걸 따르고, 없으면 기존
            # 다수결(연-월-일 우선, "읽는법:년월일순" 실측 근거)을 유지한다.
            if order_hint == "dmy":
                spans.append((m.start(), m.end(), y_dmy, mo, yy))
            elif order_hint == "ymd":
                spans.append((m.start(), m.end(), y_ymd, mo, d))
            elif _prefer_dmy_by_year_prior(y_ymd, y_dmy):
                # 라벨에 읽는 법이 없을 때. 연-월-일로 읽은 연도가 이 데이터셋에
                # 거의 나오지 않는 해인데 뒤집으면 흔한 해가 되는 경우만 뒤집는다.
                spans.append((m.start(), m.end(), y_dmy, mo, yy))
            else:
                spans.append((m.start(), m.end(), y_ymd, mo, d))
        elif ymd_ok:
            spans.append((m.start(), m.end(), y_ymd, mo, d))
        elif dmy_ok:
            # 1차 해석(연-월-일)이 연도 유효범위(2015~2035)를 벗어나 무효일 때만
            # 순서를 뒤집어 재시도. 수입품은 일-월-연(DD.MM.YY)인 경우가 실측으로
            # 확인됨(001281.jpg: "05/08/25" → 연-월-일로 읽으면 2005년이라
            # 무효, 일-월-연으로 읽으면 2025-08-05로 유효 & 정답과 일치).
            spans.append((m.start(), m.end(), y_dmy, mo, yy))

    for m in _YMD_2DIGIT.finditer(text):
        _add_2digit_year_candidate(m)
    for m in _YMD_2DIGIT_LOOSE_DAY.finditer(text):
        _add_2digit_year_candidate(m)
    for m in _YMD_KOREAN.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))
    for m in _DMY_ENGLISH.finditer(text):
        d, mon_str, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        mo = MONTH_NAME.get(mon_str)
        if mo and _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))
    for m in _MDY_ENGLISH.finditer(text):
        mon_str, d, y = m.group(1).upper(), int(m.group(2)), int(m.group(3))
        mo = MONTH_NAME.get(mon_str)
        if mo and _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))
    for m in _YMD_ENGLISH.finditer(text):
        y, mon_str, d = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        mo = MONTH_NAME.get(mon_str)
        if mo and _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))
    for m in _DMY_NUMERIC_4Y.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            spans.append((m.start(), m.end(), y, mo, d))

    # 구분자 없는 6자리(DD-MM-YY)/8자리(DD-MM-YYYY) — 키워드 근처(±30자)에서만 탐색
    seen_windows = set()
    for kw in ALL_KEYWORDS:
        for idx in _find_all_occurrences(text, kw):
            ws, we = max(0, idx - 30), min(len(text), idx + len(kw) + 30)
            if (ws, we) in seen_windows:
                continue
            seen_windows.add((ws, we))
            for m in _DDMMYY_COMPACT.finditer(text[ws:we]):
                d, mo, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                y = _normalize_2digit_year(yy)
                if _valid_ymd(y, mo, d):
                    spans.append((ws + m.start(), ws + m.end(), y, mo, d))
            for m in _DDMMYYYY_COMPACT.finditer(text[ws:we]):
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if _valid_ymd(y, mo, d):
                    spans.append((ws + m.start(), ws + m.end(), y, mo, d))
            for m in _YMD_SPACE_ONLY.finditer(text[ws:we]):
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if _valid_ymd(y, mo, d):
                    spans.append((ws + m.start(), ws + m.end(), y, mo, d))

    spans.sort()
    dedup = []
    last_end = -1
    for s, e, y, mo, d in spans:
        if s >= last_end:
            dedup.append({"start": s, "end": e, "year": y, "month": mo, "day": d})
            last_end = e
    return dedup


def _nearest_before(before, keywords):
    """before(앞 window) 안에서 keywords 중 후보 날짜에 가장 가까운 것까지의 거리."""
    best = None
    for kw in keywords:
        i = before.rfind(kw)
        if i != -1:
            d = len(before) - i
            if best is None or d < best:
                best = d
    return best


def _nearest_after(after, keywords):
    """after(뒤 window) 안에서 keywords 중 후보 날짜에 가장 가까운 것까지의 거리."""
    best = None
    for kw in keywords:
        j = after.find(kw)
        if j != -1 and (best is None or j < best):
            best = j
    return best


def _classify_kind(text, start, end, window=20):
    """후보 날짜 앞/뒤 window자 안의 키워드로 종류(EXP/MFG/UNKNOWN) 판별.

    "라벨: 날짜"가 압도적으로 흔한 형태라서, **바로 앞에 있는 키워드를 최우선**으로
    본다. 뒤쪽 키워드는 앞에 아무 키워드도 없을 때만(예: "2020.10.29제조"처럼
    날짜 뒤에 라벨이 붙는 드문 형태) 보조로 쓴다.

    앞/뒤를 동급으로 취급하면 실패한다 — 실측 버그: "제조일자: 2025.09.09
    소비기한: 2026.09.08"에서 앞 날짜(2025.09.09)가 뒤에 있는 "소비기한"
    키워드와 더 가깝다는 이유로 EXP로 잘못 분류돼, 제조일자가 소비기한보다
    먼저 채택돼버렸음.
    """
    before = text[max(0, start - window):start]
    after = text[end:end + window]

    exp_kws = KW_CONSUME + KW_EXP_EN + (KW_DISTRIB if TREAT_DISTRIB_AS_ANSWER else [])

    exp_before = _nearest_before(before, exp_kws)
    mfg_before = _nearest_before(before, KW_MFG)
    if exp_before is not None and (mfg_before is None or exp_before < mfg_before):
        return "EXP"
    if mfg_before is not None and (exp_before is None or mfg_before <= exp_before):
        return "MFG"

    exp_after = _nearest_after(after, exp_kws)
    mfg_after = _nearest_after(after, KW_MFG)
    if exp_after is None and mfg_after is None:
        return "UNKNOWN"
    if exp_after is None:
        return "MFG"
    if mfg_after is None:
        return "EXP"
    return "EXP" if exp_after <= mfg_after else "MFG"


def _find_kkaji_adjacent(cands, text, window=5):
    """날짜 바로 뒤(몇 글자 이내)에 '까지'가 붙으면 그 날짜가 곧 마감일이라는
    뜻 — 한국어 문법상 '~까지'는 매우 강한 신호라서 키워드 근접도 휴리스틱보다
    우선 적용한다. "제조일자 소비기한 D1 D2 까지"처럼 라벨이 값보다 앞서 몰려
    나오는 표(表) 형태나, "D1제조 D2까지"처럼 날짜 뒤에 라벨이 붙는 형태 둘 다
    이걸로 해결된다 (근접도 휴리스틱만으로는 두 경우 다 오답 냄 — 실측:
    000686.jpg, 001229.jpg)."""
    for c in cands:
        after = text[c["end"]:c["end"] + window]
        if "까지" in after:
            return c
    return None


def _pick_final(cands, text):
    """여러 날짜 후보 중 '소비기한'에 해당하는 하나를 고른다."""
    if not cands:
        return None

    for c in cands:
        c["kind"] = _classify_kind(text, c["start"], c["end"])

    kkaji_hit = _find_kkaji_adjacent(cands, text)
    if kkaji_hit:
        return kkaji_hit

    # "부터" 단독이 아니라 "~로부터"(제조일로부터 등 상대 표기의 일부)인 경우는
    # 제외한다 — 안 그러면 "제조 2020.11.27 제조일로부터 5년까지"에서 "로부터"
    # 안의 "부터"를 진짜 "OO부터 OO까지" 구간 표기로 착각해서, 계산해야 할
    # 제조일자를 그대로 답으로 내버리는 사고가 난다 (실측 회귀 버그).
    has_buteo_kkaji = bool(re.search(r"(?<!로)부터", text)) and ("까지" in text)
    if has_buteo_kkaji:
        # 위 adjacency 매칭이 실패했다는 건 OCR이 레이아웃을 무시하고 텍스트를
        # 읽어 '부터'/'까지' 토큰이 실제 날짜 옆이 아니라 엉뚱한 곳(예: 문장
        # 맨 끝)에 몰려버렸다는 뜻이다 (실측: 000081.jpg). 이럴 땐 확인된
        # 규칙대로 더 늦은 날짜를 채택한다.
        return max(cands, key=lambda c: (c["year"], c["month"], c["day"]))

    # PREFER_CONSUME_OVER_MFG(미확정2)는 사실상 여기서 자동으로 지켜진다:
    # EXP로 분류된 후보가 하나라도 있으면 그걸 쓰고, 없을 때만 아래로 내려간다.
    exp_cands = [c for c in cands if c["kind"] == "EXP"]
    if exp_cands:
        return exp_cands[0]

    # "제조일로부터 N개월/년" 문구가 있으면 EXP 후보 판정보다는 약하지만,
    # "종류 모를 후보 1개 = 소비기한"이라는 아래 규칙보다는 먼저 확인해야 한다.
    # 이유(실측 버그, 001686.jpg/001958.jpg): OCR이 "제조"를 "제 "/"1조"처럼
    # 깨뜨리면 classify_kind가 MFG로도 못 알아보고 UNKNOWN으로 떨어지는데,
    # 그러면 아래 "단독 후보=소비기한" 규칙이 먼저 걸려서 제조일자를 그대로
    # 답으로 내버리고, 정작 뒤에 멀쩡히 살아있는 "제조일로부터 5년까지" 같은
    # 문구는 아예 확인도 안 하고 지나쳐버렸다. 라벨이 깨졌어도 상대 기간
    # 문구 자체는 온전한 경우가 많아서, 여기서 먼저 확인하는 게 더 안전하다.
    computed = _compute_from_relative(cands, text)
    if computed:
        return computed

    # 여기 도달 = EXP로 분류된 후보도, 계산 가능한 상대 기간도 없음.
    # 후보가 전부 MFG로 분류됐다면(=제조일자라고 확신) 절대 답으로 채택하지
    # 않는다 — 후보가 1개뿐이라고 무조건 받아쓰면 제조일자를 소비기한으로
    # 잘못 내는 사고가 난다.
    unknown_cands = [c for c in cands if c["kind"] == "UNKNOWN"]
    if len(unknown_cands) == 1:
        return unknown_cands[0]
    if len(unknown_cands) > 1:
        # 종류를 하나도 못 가른 다중 날짜: 소비기한이 제조일자보다 늦다는
        # 상식적 가정으로 가장 늦은 날짜를 채택. 휴리스틱이라 오탐 가능 —
        # 팀 검토 대상.
        return max(unknown_cands, key=lambda c: (c["year"], c["month"], c["day"]))

    return None


def _add_months(d, months):
    """월 단위 덧셈. 말일 초과 시 그 달의 마지막 날로 clamp 한다
    (예: 1/31 + 1개월 -> 2/28, 윤년이면 2/29)."""
    total_month0 = (d.month - 1) + months
    year = d.year + total_month0 // 12
    month = total_month0 % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


def _add_period(base_date, n, unit):
    if unit == "일":
        return base_date + timedelta(days=n)
    if unit == "주":
        return base_date + timedelta(weeks=n)
    if unit == "개월":
        return _add_months(base_date, n)
    if unit == "년":
        return _add_months(base_date, n * 12)
    return None


def _compute_from_relative(cands, text):
    """"제조일로부터 N개월/년" 같은 상대 기간 표기가 있으면, 후보 날짜(제조일로
    추정되는 것) + 기간을 계산해서 돌려준다. EXP로 확실히 분류된 후보는 이미
    앞에서 처리됐으니, 여기서는 MFG/UNKNOWN 후보를 기준일 후보로 본다 — OCR이
    "제조"를 "제 "/"1조"처럼 깨뜨리면 MFG로도 인식 못 하고 UNKNOWN으로 떨어지는
    경우가 실제로 있어서(001686.jpg, 001958.jpg), MFG만 보면 놓친다.

    RELATIVE_EXPR_AS_NONE=True로 되돌리면 이 함수는 항상 None을 반환하므로,
    계산이 필요없다는 쪽으로 정책이 바뀌어도 호출부를 지울 필요는 없다."""
    if RELATIVE_EXPR_AS_NONE:
        return None

    base_cands = [c for c in cands if c["kind"] != "EXP"]
    if not base_cands:
        return None

    m = _RELATIVE_PERIOD.search(text)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)

    # 상대 표기 문구에 가장 가까운 날짜를 기준(제조일)으로 삼는다.
    base = min(base_cands, key=lambda c: abs(c["start"] - m.start()))
    base_date = date(base["year"], base["month"], base["day"])

    result = _add_period(base_date, n, unit)
    if result is None:
        return None
    return {"year": result.year, "month": result.month, "day": result.day, "kind": "COMPUTED"}


def _find_all_occurrences(text, kw):
    """text.find(kw)는 첫 등장만 찾는다. 같은 키워드가 여러 번 나오는 라벨이
    흔해서(제조국명/제조일로부터 등) 전부 찾아야 한다."""
    idxs = []
    start = 0
    while True:
        idx = text.find(kw, start)
        if idx == -1:
            break
        idxs.append(idx)
        start = idx + 1
    return idxs


def _find_partial_md(text):
    """연도 없이 월.일만 있는 케이스. (운영진 확인 사례: 000995.jpg -> NONE|02|18)
    소비기한/유통기한/EXP 키워드 주변(±40자)에서만 찾는다 — 제조일자(MFG) 키워드
    주변은 일부러 뺐다. 안 그러면 제조일자의 월/일 조각이 "연도만 없는 소비기한"
    으로 잘못 새어들어온다 (실측 버그: 001686.jpg — 제조일자 "2020,11.27"에서
    콤마 때문에 연도 매칭이 깨지자, "11.27" 조각이 연도부재 케이스로 오인됨)."""
    windows = []
    for kw in KW_CONSUME + KW_DISTRIB + KW_EXP_EN:
        for idx in _find_all_occurrences(text, kw):
            windows.append((max(0, idx - 40), min(len(text), idx + len(kw) + 40)))
    for ws, we in windows:
        m = _MD_ONLY.search(text[ws:we])
        if m:
            mo, d = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return mo, d
    return None


def extract_expiry_fields(text):
    """OCR 원문 텍스트 -> {'year','month','day','final_date'} (전부 문자열, 미인식 NONE).

    submission.csv 스키마와 1:1 대응.
    """
    if not text:
        return {"year": "NONE", "month": "NONE", "day": "NONE", "final_date": "NONE"}

    full_cands = _find_full_candidates(text)
    chosen = _pick_final(full_cands, text)

    if chosen:
        year = f"{chosen['year']:04d}"
        month = f"{chosen['month']:02d}"
        day = f"{chosen['day']:02d}"
        return {"year": year, "month": month, "day": day,
                "final_date": f"{year}-{month}-{day}"}

    partial = _find_partial_md(text)
    if partial:
        mo, d = partial
        month = f"{mo:02d}"
        day = f"{d:02d}"
        return {"year": "NONE", "month": month, "day": day,
                "final_date": f"NONE-{month}-{day}"}

    return {"year": "NONE", "month": "NONE", "day": "NONE", "final_date": "NONE"}


def explain(text):
    """디버그/QA용: 왜 이 결과가 나왔는지 후보/판정 근거를 같이 보여준다.
    라벨링·검증 작업(팀원 담당) 할 때 이 함수로 이미지별 판정 이유를 확인하면 됨."""
    full_cands = _find_full_candidates(text)
    chosen = _pick_final([dict(c) for c in full_cands], text)
    return {
        "candidates": full_cands,
        "chosen": chosen,
        "has_relative_expr": any(re.search(p, text) for p in RELATIVE_PATTERNS),
        "has_reference_phrase": any(re.search(p, text) for p in REFERENCE_PATTERNS),
        "result": extract_expiry_fields(text),
    }
