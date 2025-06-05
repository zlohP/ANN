import numpy as np
import cv2

#이미지 하나에 대해 확장하여 총 7장의 변형된 이미지 반환(리스트)
def augment_image(img):
    augmented = []
    augmented.append(img)  # 원본 포함

    center = (14, 14)  # 이미지 중심

    # 회전 (±15도)
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (28, 28), borderValue=0)
        augmented.append(rotated)

    # 이동 (상하 ±2픽셀)
    for dx, dy in [(0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (28, 28), borderValue=0)
        augmented.append(shifted)

    # 확대/축소 (0.95배, 1.05배)
    for scale in [0.95, 1.05]:
        M = cv2.getRotationMatrix2D(center, 0, scale)
        scaled = cv2.warpAffine(img, M, (28, 28), borderValue=0)
        augmented.append(scaled)

    return augmented

#전체 x_train에 augment_image 반복 적용
def expand_dataset(x, t):
    x_aug = []
    t_aug = []

    for i in range(len(x)):
        img = x[i][0]  # (1, 28, 28) → (28, 28)
        augmented_imgs = augment_image(img)

        for aug in augmented_imgs:
            x_aug.append(aug[np.newaxis, :, :])  # 다시 (1, 28, 28)
            t_aug.append(t[i])

    x_aug = np.array(x_aug).astype(np.float32)
    t_aug = np.array(t_aug)

    return x_aug, t_aug
