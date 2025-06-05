# make_test_dataset_3d.py

import pickle
from handwrittenNumLoader import load_multiple_handwritten_images
import matplotlib.pyplot as plt

# 1. 손글씨 이미지 경로 리스트
image_paths = [
    "myDataset/image_59710386.jpg", "myDataset/image_0123456789 (2).jpg"
]

# 2. 각 이미지에 있는 숫자들의 정답 라벨 리스트
label_lists = [
    [5, 9, 7, 1, 0, 3, 8, 6], [0,1,2,3,4,5,6,7,8,9]

]

# 3. 숫자 이미지 자르고 (1,28,28) 형식으로 전처리
x_test, t_test = load_multiple_handwritten_images(image_paths, label_lists)


# 자른 숫자 이미지들이 잘 들어갔는지 확인
print(f"총 {len(x_test)}개의 숫자 이미지가 감지됨")

plt.figure(figsize=(15, 3))

for i in range(len(x_test)):
    plt.subplot(1, len(x_test), i + 1)
    plt.imshow(x_test[i][0], cmap='gray')  # x_test[i].shape = (1, 28, 28)
    plt.title(str(t_test[i]))
    plt.axis('off')

plt.tight_layout()
plt.show()
# 4. 데이터셋 저장
with open("TestDataSet3D.pkl", "wb") as f:
    pickle.dump((x_test, t_test), f)

print("저장 완료! TestDataSet3D.pkl 생성됨")
print(f"x_test shape: {x_test.shape}")  # → (N, 1, 28, 28)
print(f"t_test shape: {t_test.shape}")  # → (N,)
