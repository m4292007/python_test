"""
python -m pip install -U pip
python -m pip install -U matplotlib
"""
from pylab import *

x = linspace(-1.6, 1.6, 10000)
f = lambda x: (sqrt(cos(x)) * cos(200 * x) + sqrt(abs(x)) - 0.7) * \
    pow((4 - x * x), 0.01)
plot(x, list(map(f, x)))
show()



import matplotlib.pyplot as plt  

plt.plot([1,2,3,4])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

import matplotlib.pyplot as plt  

plt.plot([1,2,3,4], [0,1,4,9], 'ro')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()


import numpy as np     # numpy객체를 np로 정의하고 호출한다.
import matplotlib.pyplot as plt

t = np.arange(0., 5., 0.2)     # 0이상 5미만까지 0.2 간격으로 배열을 만든다.
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')    # 설명 참고
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.01)

plt.plot(t1, f(t1))

plt.show()

import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

def g(t):
    return np.sin(np.pi*t)

t1 = np.arange(0.0, 5.0, 0.01)
t2 = np.arange(0.0, 5.0, 0.01)

plt.subplot(221)
plt.plot(t1, f(t1))

plt.subplot(222)
plt.plot(t2, g(t2))

plt.subplot(223)
plt.plot(t1, f(t1), 'r-')

plt.subplot(224)
plt.plot(t2, g(t2), 'r-')

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 랜덤 자료 생성
np.random.seed(20180107)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# 히스토그램 만들기
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
