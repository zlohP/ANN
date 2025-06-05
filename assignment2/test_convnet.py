from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SimpleConvNet()
network.load_params("params.pkl")

# 불러온 모델로 예측
accuracy = network.accuracy(x_test, t_test)
print("정확도:", accuracy*100, "%")
