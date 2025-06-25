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
from sklearn.model_selection import train_test_split

with open('Dataset3D_1.pkl', 'rb') as f:
    datasetL = pickle.load(f)
x_train, t_train = datasetL
x_train = np.expand_dims(x_train, axis=1)

# 데이터 확장
x_aug, t_aug = expand_dataset(x_train, t_train)


with open('DataSet3D_2.pkl', 'rb') as f:
	datasetL = pickle.load(f)
x_train2, t_train2 = datasetL
x_train2 = np.expand_dims(x_train2, axis=1)

x_aug2, t_aug2 = expand_dataset(x_train2, t_train2)

x_train = np.concatenate([x_aug, x_aug2])/255
t_train = np.concatenate([t_aug, t_aug2])


with open('testdata3D_N.pkl', 'rb') as f:
	datasetL = pickle.load(f)
x_test, t_test = datasetL


max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param={'filter_num1':16, 'filter_num2':32, 'filter_size':3, 'pad':1, 'stride':1},
                        hidden_size=64, output_size=10, weight_init_std=0.02)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='AdaGrad', optimizer_param={'lr': 0.009616019313344763},
                  weight_decay_lambda=3.3219737729913015e-08,
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
