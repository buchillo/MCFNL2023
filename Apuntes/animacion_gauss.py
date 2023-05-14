import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1001)

x_0 = 3 #centro de la gaussiana
s0 = 1 #spread = anchura de la gaussiana

c = 1 #velocidad a la que se mueve la gaussiana

for t in range(10):

    gauss = np.exp(- (x-x_0-c*t)**2 / (2*s0**2))

    plt.plot(x, gauss)
    plt.grid()
    plt.ylim(-0.1, 1.1)
    plt.xlim(x[0], x[-1])
    plt.pause(0.1)
    plt.cla()