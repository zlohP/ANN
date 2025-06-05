import cv2
import numpy as np
import os


def load_multiple_handwritten_images(image_paths, label_lists):
    """여러 장의 손글씨 이미지에서 숫자를 잘라 학습용 배열로 만듦

    Args:
        image_paths: 이미지 경로 리스트
        label_lists: 각 이미지에 있는 숫자 정답 리스트 (예: [[4,1,3,...], [0,9,2,7], ...])

    Returns:
        x_my: shape=(N, 1, 28, 28), t_my: shape=(N,)
    """
    x_data = []
    t_data = []

    for img_path, labels in zip(image_paths, label_lists):
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours]
        rects.sort(key=lambda b: b[1])  # y좌표 기준 정렬 (세로 정렬)

        for i, rect in enumerate(rects):
            if i >= len(labels):  # 레이블 개수보다 contour 더 많을 경우 방지
                break
            x, y, w, h = rect
            if w < 10 or h < 10:
                continue
            margin = 20
            digit = img_gray[max(y - margin, 0):y + h + margin, max(x - margin, 0):x + w + margin]
            digit = cv2.resize(digit, (28, 28))
            digit = (digit < 70) * digit
            digit = digit.astype(np.float32) / 255.
            digit = digit.reshape(1, 28, 28)
            x_data.append(digit)
            t_data.append(labels[i])

    return np.array(x_data), np.array(t_data)
