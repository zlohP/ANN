# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
from common.layers import BatchNormalization


class SimpleConvNet:
    """단순한 합성곱 신경망
    
    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num1':16, 'filter_num2':32, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=64, output_size=10, weight_init_std=0.02):
        filter_num1 = conv_param['filter_num1']
        filter_num2 = conv_param['filter_num2']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = input_size
        pool_output_size = int(filter_num2 * (conv_output_size // 4) ** 2)  # 7x7 map

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num1, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num1)
        self.params['W2'] = weight_init_std * np.random.randn(filter_num2, filter_num1, filter_size, filter_size)
        self.params['b2'] = np.zeros(filter_num2)
        self.params['W3'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        self.params['gamma1'] = np.ones(filter_num1)
        self.params['beta1'] = np.zeros(filter_num1)
        self.params['gamma2'] = np.ones(filter_num2)
        self.params['beta2'] = np.zeros(filter_num2)
        self.params['gamma3'] = np.ones(hidden_size)
        self.params['beta3'] = np.zeros(hidden_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['BN1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], filter_stride, filter_pad)
        self.layers['BN2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BN3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Dropout1'] = Dropout(0.3)
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, (Dropout,BatchNormalization)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, weight_decay_lambda=0):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x, train_flg = True)
        loss = self.last_layer.forward(y, t)
        #가중치 감소(L2 정규화)
        weight_decay = 0
        for i in range(1,5):
            W = self.params['W' + str(i)]
            weight_decay += 0.5 * weight_decay_lambda * np.sum(W**2)

        return loss + weight_decay

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        total = 0

        for i in range(0, x.shape[0], batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
            total += len(tx)

        return acc / total

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['gamma1'] = self.layers['BN1'].dgamma
        grads['beta1'] = self.layers['BN1'].dbeta
        grads['gamma2'] = self.layers['BN2'].dgamma
        grads['beta2'] = self.layers['BN2'].dbeta
        grads['gamma3'] = self.layers['BN3'].dgamma
        grads['beta3'] = self.layers['BN3'].dbeta
        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        params['running_mean1'] = self.layers['BN1'].running_mean
        params['running_var1'] = self.layers['BN1'].running_var
        params['running_mean2'] = self.layers['BN2'].running_mean
        params['running_var2'] = self.layers['BN2'].running_var
        params['running_mean3'] = self.layers['BN3'].running_mean
        params['running_var3'] = self.layers['BN3'].running_var

        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        self.layers['BN1'].running_mean = params['running_mean1']
        self.layers['BN1'].running_var = params['running_var1']
        self.layers['BN2'].running_mean = params['running_mean2']
        self.layers['BN2'].running_var = params['running_var2']
        self.layers['BN3'].running_mean = params['running_mean3']
        self.layers['BN3'].running_var = params['running_var3']

        self.layers['BN1'].gamma = self.params['gamma1']
        self.layers['BN1'].beta = self.params['beta1']
        self.layers['BN2'].gamma = self.params['gamma2']
        self.layers['BN2'].beta = self.params['beta2']
        self.layers['BN3'].gamma = self.params['gamma3']
        self.layers['BN3'].beta = self.params['beta3']

        self.layers['Conv1'].W = self.params['W1']
        self.layers['Conv1'].b = self.params['b1']
        self.layers['Conv2'].W = self.params['W2']
        self.layers['Conv2'].b = self.params['b2']
        self.layers['Affine1'].W = self.params['W3']
        self.layers['Affine1'].b = self.params['b3']
        self.layers['Affine2'].W = self.params['W4']
        self.layers['Affine2'].b = self.params['b4']
