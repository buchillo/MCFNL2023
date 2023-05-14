import numpy as np
import matplotlib.pyplot as plt

#Constantes universales

eps = 1.0 #sistema natural: c0 = epsilon = mu = 1 --> el tiempo NO estará en segundos
mu  = 1.0
c0 = 1/np.sqrt(eps*mu)

CFL = 0.9 #parámetro de la condición CFL

#Posiciones del grid primario

x = np.linspace(0, 10, num = 101) #Cogemos 101 porque como empieza en 0, debemos coger 'uno de más' para que (p.e) vaya en 0.1
xDual = (x[1:] + x[:-1])/2 #Son las posiciones que están en el centro de cada lugar de la malla principal.

dx = x[1] - x[0]

#NOTA: el xDual tiene un nodo menos --> el campo mágnetico (que 'vive' en el dual) tendrá un punto menos

#Condición inicial de los campos

x0 = 5.0
s0 = 0.75 #Para evitar componentes significativas del campo en la frontera

e = np.exp(- (x-x0)**2 / (2*s0**2))
e[0] = 0.0 #condiciones de contorno
e[-1] = 0.0

h = np.zeros(xDual.shape) #xDual.shape te devuelve la dimension de xDual --> h tiene el mismo número de nodos que xDual

#Evolucion temporal

dt = CFL * dx/c0
tRange = np.arange(0, 10, dt) #np.arange es igual que np.linspace pero indicas el paso en lugar del número de nodos


for t in tRange:

    e[1:-1] = - dt/(dx*eps) * (h[1:]-h[:-1])+e[1:-1]
    h[:] = - dt/(dx*mu) * (e[1:]-e[:-1])+h[:]
    
    plt.plot(x, e, '.-')
    plt.plot(xDual, h, '.-')
    plt.ylim(-1.1, 1.1)
    plt.xlim(-0.1, 10.1)
    plt.pause(0.01)
    plt.cla()