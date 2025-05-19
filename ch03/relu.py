import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0,x)

x = np.arange(-5,5,0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-0.1,5)
plt.show()
