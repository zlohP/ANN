import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
import pickle



with open('testdata3D_N.pkl', 'rb') as f:
	datasetL = pickle.load(f)
x_test, t_test = datasetL
idx = np.arange(x_test.shape[0])
np.random.shuffle(idx)
x_test = x_test[idx]
t_test = t_test[idx]
network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param={'filter_num1':16, 'filter_num2':32, 'filter_size':3, 'pad':1, 'stride':1},
                        hidden_size=100, output_size=10, weight_init_std=0.02)

# 매개변수 읽기
network.load_params("params.pkl")
print("Load Network Parameters!")

test_acc = network.accuracy(x_test, t_test)
print(test_acc*100, "%")