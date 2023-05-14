import numpy as np
import matplotlib.pyplot as plt

eps = 1
mu = 1
c = 1/np.sqrt(eps*mu)

class FDTD_PML_2D():

    def __init__(self, CFL=1.0, L =10, N=101):

        self.x = np.linspace(0, L, num=N)
        self.xDual = (self.x[1:] + self.x[:-1])/2

        self.y = np.linspace(0, L, num=N)
        self.yDual = (self.y[1:] + self.y[:-1])/2

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dt =  CFL / ( c * np.sqrt( 1 / self.dx**2 + 1 / self.dy**2 ) )

        self.ex = np.zeros((N, N-1))
        self.ey = (self.ex).transpose()
        self.h = np.zeros((N-1, N-1))
   

        self.Dx = np.zeros((N, N-1))
        self.Dy = np.zeros((N-1, N))
        self.b = np.zeros( (N-1, N-1) )

        self.Dxn = np.zeros((N, N-1))
        self.Dyn = np.zeros((N-1, N))
        self.bznew = np.zeros((N-1, N-1)) 

        def aux(x, Ny, par, yx):

            x0 = 1*L/10
            x1 = L-x0

            sig = np.zeros(x.shape)
            L0 = int(len(x)/2)
            sig[:L0] = np.where(x[:L0]<x0, 9*(x[:L0]-x0)**2, 0)
            sig[L0:] = np.where(x[L0:]>x1, 9*(x[L0:]-x1)**2, 0)

            if yx < 0 :
                sig = sig + np.zeros((Ny,1))
            else:
                sig = (sig + np.zeros((Ny,1))).transpose()

            alpha = par/self.dt - sig/2
            beta = par/self.dt + sig/2

            return alpha, beta
        
        self.alpha_xx,self.beta_xx = aux(self.xDual, N, eps,-1) # Sigma_x viviendo donde Ex
        self.alpha_yx,self.beta_yx = aux(self.y, N-1, eps, 1)  # Sigma_y viviendo donde Ex

        self.alpha_xy,self.beta_xy = aux(self.x, N-1, eps, -1) # Sigma_x viviendo donde Ey
        self.alpha_yy,self.beta_yy = aux(self.yDual, N, eps, 1)  # Sigma_y viviendo donde Ey

        self.alpha_xz,self.beta_xz = aux(self.xDual, N-1, mu,  -1) # Sigma_x viviendo donde Hz
        self.alpha_yz,self.beta_yz = aux(self.yDual, N-1, mu, 1) # Sigma_y viviendo donde Hz
   


    def step(self, N=101):     

        alpha_xx = self.alpha_xx
        alpha_xy = self.alpha_xy
        alpha_yx = self.alpha_yx
        alpha_yy = self.alpha_yy
        alpha_xz = self.alpha_xz
        alpha_yz = self.alpha_yz

        beta_xx = self.beta_xx
        beta_xy = self.beta_xy
        beta_yx = self.beta_yx
        beta_yy = self.beta_yy
        beta_xz = self.beta_xz
        beta_yz = self.beta_yz


        dx = self.dx
        dy = self.dy
        dt = self.dt
        

        ex = self.ex
        ey = self.ey
        h = self.h
 
        Dx = self.Dx
        Dy = self.Dy
        b = self.b

        Dxn = self.Dxn
        Dyn = self.Dyn
        bznew = self.bznew

        ex[:,0] = ex[:,-1] = 0
        ey[0, :] = ey[-1, :] = 0
        
        bznew[:,:] = alpha_yz[:,:]/beta_yz[:,:]*b[:,:] + 1/beta_yz[:,:]*( -(ey[:, 1:] - ey[:, :-1])/dx + (ex[1:, :] - ex[:-1, :])/dy )
        h[:,:] = alpha_xz[:,:]/beta_xz[:,:]*h[:,:] + 1/beta_xz[:,:]*( bznew[:,:] - b[:,:] )/dt

        b[:,:] = bznew[:,:]

        Dxn[1:-1,:] = Dx[1:-1,:] + dt/eps*( (h[1:,:] - h[:-1,:])/dy )
        ex[1:-1, :] = alpha_yx[1:-1,:]/beta_yx[1:-1,:] * ex[1:-1,:] + 1/beta_yx[1:-1,:]*( beta_xx[1:-1,:]*Dxn[1:-1,:] - alpha_xx[1:-1,:]*Dx[1:-1,:] )
        
        Dyn[:, 1:-1] = Dy[:,1:-1] - dt/eps*( (h[:,1:] - h[:,:-1])/dx )
        ey[:, 1:-1] = alpha_xy[:, 1:-1]/beta_xy[:, 1:-1] * ey[:,1:-1] + 1/beta_xy[:,1:-1]*( beta_yy[:,1:-1]*Dyn[:,1:-1] - alpha_yy[:,1:-1]*Dy[:,1:-1] )
        
        Dx[:,:] = Dxn
        Dy[:,:] = Dyn     
        
