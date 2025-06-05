# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from dataExpansion import expand_dataset
from handwrittenNumLoader import load_multiple_handwritten_images


#이미지 처리
#image_paths = ["image_59710386.jpg", "image_90142683.jpg"]
#label_lists = [[5,9,7,1,0,3,8,6],[9,0,1,4,2,6,3,8]]
image_paths = [
    "myDataset/image_59710386.jpg",
    "myDataset/image_0123456789 (2).jpg"
]

label_lists = [
    [5, 9, 7, 1, 0, 3, 8, 6],
    [0,1,2,3,4,5,6,7,8,9]
]

x_my, t_my = load_multiple_handwritten_images(image_paths, label_lists)
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

x_train = np.concatenate((x_train[:4986], x_my),axis=0)
t_train = np.concatenate((t_train[:4986], t_my),axis=0)
x_test = np.concatenate((x_test[:986], x_my),axis=0)
t_test = np.concatenate((t_test[:986], t_my),axis=0)

print("원래:", len(x_train))
x_train, t_train = expand_dataset(x_train, t_train)
print("확장 후:", len(x_train))

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  weight_decay_lambda=1e-4,
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
