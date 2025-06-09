import pickle
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터셋 로드
with open("TestDataSet3D.pkl", "rb") as f:
    x_data, t_data = pickle.load(f)

# 2. 시각화
num_images = len(x_data)
plt.figure(figsize=(15, 2))

for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(x_data[i], cmap='gray')
    plt.title(f"Label: {t_data[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
