import numpy as np
import matplotlib.pyplot as plt

#Posiciones del grid primario

L = 20
x = np.linspace(0, L, num = 121) #Cogemos 101 porque como empieza en 0, debemos coger 'uno de más' para que (p.e) vaya en 0.1
xDual = (x[1:] + x[:-1])/2 #Son las posiciones que están en el centro de cada lugar de la malla principal.

dx = x[1] - x[0]

#NOTA: el xDual tiene un nodo menos --> el campo mágnetico (que 'vive' en el dual) tendrá un punto menos

#Permitividades, conductividad y  velocidad de la luz. La mitad del espacio va a ser vacío y el otro no.

eps = np.ones(x.shape)
eps[x>=L/2] = 5.0

sigma = np.zeros(x.shape)
sigma[x>=L/2] = 0.5

mu  = 1.0
c0 = 1

CFL = 0.9 #parámetro de la condición CFL


#Condición inicial de los campos

x0 = 7.0
s0 = 0.5 #Para evitar componentes significativas del campo en la frontera

e = np.exp(- (x-x0)**2 / (2*s0**2))
e[0] = 0.0 #condiciones de contorno
e[-1] = 0.0

h = np.zeros(xDual.shape) #xDual.shape te devuelve la dimension de xDual --> h tiene el mismo número de nodos que xDual

#Evolucion temporal

dt = CFL * dx #Cuidado con esto porque he tenido que quitar c0 que estaba definido con eps
tRange = np.arange(0, 25, dt) #np.arange es igual que np.linspace pero indicas el paso en lugar del número de nodos


for t in tRange:
    
    alpha = sigma[1:-1]/2 + eps[1:-1]/dt
    beta = sigma[1:-1]/2 - eps[1:-1]/dt
    e[1:-1] = -  (h[1:]-h[:-1])/(alpha*dx) - beta[:]/alpha* e[1:-1]
    h[:] = - dt/(dx*mu) * (e[1:]-e[:-1])+h[:]

    plt.plot(x, e, '.-')
    plt.plot(xDual, h, '.-')
    plt.vlines(L/2, -2.1, 2.1) #Hace una línea donde cambia el medio
    plt.ylim(-2.1, 2.1)
    plt.xlim(-0.1, 20.1)
    plt.pause(0.01)
    plt.cla()