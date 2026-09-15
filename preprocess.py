# -*- coding: utf-8 -*-
"""안 읽히는 사진을 읽히게 만드는 보정.

하는 일은 두 가지임. 사진을 잘게 나눠서 구역마다 따로 대비를 올리고,
그 다음에 끊어진 글자 점들을 이어붙임.

왜 이 두 개냐면, 못 읽은 사진들을 직접 열어보니 원인이 대부분 둘 중 하나였음.
포장이 어둡거나 빛이 반사돼서 글자와 배경 색이 비슷해진 경우, 그리고 잉크젯으로
점을 찍어 날짜를 인쇄해서 글자가 점선처럼 끊겨 있는 경우임.

여섯 가지 방법을 놓고 못 읽은 103장으로 비교했을 때 이 조합이 제일 나았음.
부분점수 기준 6.15%에서 27.51%로 올랐음.

사진 전체를 한꺼번에 밝히지 않고 구역별로 나눠서 올리는 이유는, 한꺼번에 올리면
원래 잘 보이던 부분이 하얗게 날아가기 때문임.
"""
import cv2
import numpy as np


def enhance(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 사진을 8x8 칸으로 나눠 칸마다 대비를 올림.
    # clipLimit 3.0 — D/E/F 신규 라벨 907장 기준 재검증해서 2.0보다 나음 확인.
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)

    # 끊긴 점을 이어붙임. 2x2로 작게 잡은 이유는 이보다 키우면 글자끼리
    # 붙어버려서 8과 3을 구분 못 하는 일이 생기기 때문임.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)

    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def binarize_dilate(img, k=5):
    """도트프린터 날짜 전용 보정. 오츠 이진화 후 팽창해서 점을 잇는다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.dilate(th, kernel, iterations=1)
    th = cv2.bitwise_not(th)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
