import final_class
import numpy as np
import matplotlib.pyplot as plt

eps = 1
mu = 1
c = 1/np.sqrt(eps*mu)
L = 10

fd = final_class.FDTD_PML_2D()
dt = fd.dt


x_0 = L/2
s_0 = 0.25

xh, yh = np.meshgrid(fd.xDual, fd.yDual)

initialField = np.exp(-(pow(xh-x_0, 2) + pow(yh-x_0, 2))/(2*s_0**2))
b = mu*initialField

fd.h = initialField
fd.b = b

level = np.linspace(-0.1, 0.6, 51)

for _ in np.arange(0, 5, fd.dt):

    fd.step()
    plt.imshow(fd.h, cmap = 'viridis', interpolation='gaussian',vmin = -0.05, vmax = 0.4)
    plt.axis('off')
    #cb =plt.colorbar()
    plt.pause(0.0001)
    plt.cla()
    #cb.remove()