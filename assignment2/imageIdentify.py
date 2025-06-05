import matplotlib.pyplot as plt
from dataExpansion import augment_image  # 네가 만든 함수

# MNIST에서 임의로 하나 가져오기
from dataset.mnist import load_mnist
(x_train, t_train), _ = load_mnist(flatten=False)
img = x_train[588][0]  # (1, 28, 28) → (28, 28)

# 데이터 확장
augmented_imgs = augment_image(img)

# 시각화
plt.figure(figsize=(10, 2))
for i, aug in enumerate(augmented_imgs):
    plt.subplot(1, len(augmented_imgs), i+1)
    plt.imshow(aug, cmap='gray')
    plt.axis('off')
plt.suptitle("Augmented Samples", fontsize=14)
plt.show()
