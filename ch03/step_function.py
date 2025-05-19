import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    '''y = x>0
    y = y.astype(int)
    return y'''
    return np.array(x>0, dtype=int)

x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y) #x,y 배열을 그래프로 그리기
plt.ylim(-0.1,1.1) #y축 범위 지정
plt.show()
