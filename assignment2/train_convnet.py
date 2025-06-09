# coding: utf-8
import pickle
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from dataExpansion import expand_dataset
from handwrittenNumLoader import load_multiple_handwritten_images


with open('Dataset3D_1.pkl', 'rb') as f:
    datasetL = pickle.load(f)
x_train, t_train = datasetL
x_train = np.expand_dims(x_train, axis=1)

# 데이터 확장
x_aug, t_aug = expand_dataset(x_train, t_train)

# 새 파일로 저장
with open("Dataset3D_1_expanded.pkl", "wb") as f:
    pickle.dump((x_aug, t_aug), f)

"""with open('dataset/mnist.pkl', 'rb') as f:
    datasetL = pickle.load(f)
x_train2 = datasetL['train_img']
x_train2 = x_train2.reshape(-1, 28, 28)
x_train2 = np.expand_dims(x_train2, axis=1)
t_train2 = datasetL['train_label']


x_train = np.concatenate([x_train, x_train2[:(5000-len(x_train))]])
t_train = np.concatenate([t_train, t_train2[:(5000-len(t_train))]])"""

with open('TestDataSet3D.pkl', 'rb') as f:
	datasetL = pickle.load(f)
x_test, t_test = datasetL
x_test = np.expand_dims(x_test, axis=1)

max_epochs = 40

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='AdaGrad', optimizer_param={'lr': 0.001},
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
