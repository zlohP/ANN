import matplotlib.pyplot as plt
import pickle
import numpy as np
from simple_convnet import SimpleConvNet

with open('TestDataSet3D.pkl', 'rb') as f:
    x_test, t_test = pickle.load(f)

print("📏 x_test.shape:", x_test.shape)  # ← 이걸 꼭 확인해봐
# 예: (5, 1, 28, 28)

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("params.pkl")

plt.figure(figsize=(15, 3))
for i in range(len(x_test)):
    img = x_test[i].squeeze()  # (1,28,28) → (28,28)
    label = t_test[i]
    pred = np.argmax(network.predict(x_test[i:i+1]))

    plt.subplot(1, len(x_test), i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"L:{label} / P:{pred}")
    plt.axis('off')

plt.tight_layout()
plt.show()