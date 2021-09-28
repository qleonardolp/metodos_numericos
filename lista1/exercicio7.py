#///////////////////////////////////////////////////////
#// Leonardo Felipe Lima Santos dos Santos, 2021     ///
#// leonardo.felipe.santos@usp.br	_____ ___  ___   //
#// github/bitbucket qleonardolp	  |  | . \/   \  //
#////////////////////////////////	| |   \ \   |_|  //
#////////////////////////////////	\_'_/\_`_/__|    //
#///////////////////////////////////////////////////////

# Problema de transferencia de calor unidimensional transiente

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

lmb = 0.01      # Difusividade Termica
cond = 1.00     # Condutividade Termica
dt = 0.2       # Time step
dx = 0.4       # Length step x
dy = dx         # Length step y

A1 = 1 - 2*lmb*dt/(dx**2) - 2*lmb*dt/(dy**2)
A2 = lmb*dt/(dx**2)
A3 = lmb*dt/(dy**2)

print(A2)

ti = 0
tf = 100
xi = -1
xf =  1
yi = -1
yf =  1

tempo = np.arange(ti, tf+dt, dt)
pos_x = np.arange(xi, xf+dx, dx)
pos_y = np.arange(yi, yf+dy, dy)
Nx = np.size(pos_x)
Ny = np.size(pos_y)
Nt = np.size(tempo)

Tscale = 10
Tp = np.zeros([Nx*Ny, Nt]) # Temperatura
## C.I. T0_(x,y) = 10*exp^(-20(x^2 + y^2))
for j in range(Ny):
    for i in range(Nx):
        Tp[i+j*Nx,0] = Tscale*np.exp(-20*((pos_x[i])**2 + (pos_y[j])**2))
    #endfor    
#endfor

## C.C. (Neumann)
q_xi = np.zeros([Ny, Nt])
q_xf = np.zeros([Ny, Nt])
q_yi = np.zeros([Nx, Nt])
q_yf = np.zeros([Nx, Nt])

## Abordagem Explicita:
for k in range(Nt):
    if k == (Nt-1): # CC no tempo final
        Tp[0+0*Nx,k] = Tp[(0+2)+2*Nx,k]                     # (i,j == 0,0)
        Tp[0+(Ny-1)*Nx,k] = Tp[(0+2)+(Ny-3)*Nx,k]           # (i,j == 0,Ny-1)
        Tp[(Nx-1)+0*Nx,k] = Tp[((Nx-1)-2)+2*Nx,k]           # (i,j == Nx-1,0)
        Tp[(Nx-1)+(Ny-1)*Nx,k] = Tp[((Nx-1)-2)+(Ny-3)*Nx,k] # (i,j == Nx-1,Ny-1)
        for j in range(1,Ny-1):
            Tp[0+j*Nx,k] = Tp[0+2+j*Nx,k]           # (i == 0)
            Tp[Nx-1+j*Nx,k] = Tp[(Nx-1)-2+j*Nx,k]   # (i == Nx-1)
            for i in range(1,Nx-1):
                Tp[i+0*Nx,k] = Tp[i+2*Nx,k]           # (j == 0)
                Tp[i+(Ny-1)*Nx,k] = Tp[i+(Ny-3)*Nx,k] # (j == Ny-1)
    else:
        Tp[0+0*Nx,k] = Tp[(0+2)+2*Nx,k]                     # (i,j == 0,0)
        Tp[0+(Ny-1)*Nx,k] = Tp[(0+2)+(Ny-3)*Nx,k]           # (i,j == 0,Ny-1)
        Tp[(Nx-1)+0*Nx,k] = Tp[((Nx-1)-2)+2*Nx,k]           # (i,j == Nx-1,0)
        Tp[(Nx-1)+(Ny-1)*Nx,k] = Tp[((Nx-1)-2)+(Ny-3)*Nx,k] # (i,j == Nx-1,Ny-1)
        for j in range(1,Ny-1):
            Tp[0+j*Nx,k] = Tp[0+2+j*Nx,k]           # (i == 0)
            Tp[Nx-1+j*Nx,k] = Tp[(Nx-1)-2+j*Nx,k]   # (i == Nx-1)
            # CC antes do "for" principal pois a evolucao no tempo precisa carregar a info das CCs
            for i in range(1,Nx-1):
                Tp[i+0*Nx,k] = Tp[i+2*Nx,k]           # (j == 0)
                Tp[i+(Ny-1)*Nx,k] = Tp[i+(Ny-3)*Nx,k] # (j == Ny-1)
                Tp[i+j*Nx,k+1] = A1*Tp[i+j*Nx,k] + A2*(Tp[i+1 +j*Nx,k] + Tp[i-1 +j*Nx,k]) + A3*(Tp[i+(j+1)*Nx,k] + Tp[i+(j-1)*Nx,k])
    #endif
#endfor

Nt = Nt-1
SolExp_Tp0   = np.reshape(Tp[:,0], (Nx, Ny))
SolExp_Tp100 = np.reshape(Tp[:,Nt], (Nx, Ny))
SolExp_Tp25  = np.reshape(Tp[:,math.floor(Nt/4)], (Nx, Ny))
SolExp_Tp50  = np.reshape(Tp[:,math.floor(Nt/2)], (Nx, Ny))
SolExp_Tp75  = np.reshape(Tp[:,math.floor(3*Nt/4)], (Nx, Ny))
Nt = Nt+1

#Reiniciando Temperaturas
Tp = np.zeros([Nx*Ny, Nt]) # Temperatura
## C.I. T0_(x,y) = 10*exp^(-20(x^2 + y^2))
for j in range(Ny):
    for i in range(Nx):
        Tp[i+j*Nx,0] = Tscale*np.exp(-20*((pos_x[i])**2 + (pos_y[j])**2))
    #endfor    
#endfor

## Abordagem Implicita:
a_1 = 1 + 2*lmb*dt/(dx**2) + 2*lmb*dt/(dy**2)
a_x = lmb*dt/(dx**2)
a_y = lmb*dt/(dy**2)

# Diagonais principais da matrix A:
d1  =  a_1 * np.ones(Nx*Ny)
d1f = -a_x * np.ones(Nx*Ny-1)
d1t = d1f.copy()
dyf = -a_y * np.ones(Nx*Ny-Nx)
dyt = dyf.copy()

# Montando a matrix A do sistema linear:
A = np.diag(d1, 0)
A = A + np.diag(d1f,  1)
A = A + np.diag(d1t, -1)
A = A + np.diag(dyf,  Nx)
A = A + np.diag(dyt, -Nx)

# CC:
A[0,:] = 0
A[0,0] = -1
A[0,2] =  1

print(A[0,:])
print(A[Nx+1,:])


# invertemos fora do loop apenas uma vez para simplificacao computacional:
Ainv = np.linalg.inv(A)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
pos_x, pos_y = np.meshgrid(pos_x, pos_y)

# Plot the surface.
surf = ax.plot_surface(pos_x, pos_y, SolExp_Tp25, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0.00, Tscale)
ax.zaxis.set_major_locator(LinearLocator(5+1))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, format='%.3f')

plt.show(block=True)