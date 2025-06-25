import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

#이미지 하나에 대해 확장하여 변형된 이미지 반환(리스트)
def augment_image(img):
    augmented = [img]
    center = (14, 14)

    augmented.append(elastic_transform(img, alpha=36, sigma=6))
    augmented.append(elastic_transform(img, alpha=32, sigma=5))
    augmented.append(elastic_transform(img, alpha=40, sigma=7))

    # 회전
    for angle in [-15, -10, 10, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (28,28), borderValue=0))

    # 이동
    for dx, dy in [(0,-2),(0,2),(-2,0),(2,0)]:
        M = np.float32([[1,0,dx],[0,1,dy]])
        augmented.append(cv2.warpAffine(img, M, (28,28), borderValue=0))

    # 스케일
    for s in [0.9,0.95,1.05,1.1]:
        M = cv2.getRotationMatrix2D(center, 0, s)
        augmented.append(cv2.warpAffine(img, M, (28,28), borderValue=0))

    # 밝기 조절
    for alpha in [0.8,1.2]:
        tmp = img.astype(np.float32) * alpha
        augmented.append(np.clip(tmp,0,255).astype(np.uint8))

    # Gaussian 노이즈
    noise = np.random.normal(0,10,img.shape)
    tmp = img.astype(np.float32) + noise
    augmented.append(np.clip(tmp,0,255).astype(np.uint8))

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

def elastic_transform(img, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = img.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)