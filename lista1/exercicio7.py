# Problema de transferencia de calor unidimensional transiente

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy.core.fromnumeric import mean, shape, size

lmb = 0.01      # Difusividade Termica
cond = 1.00     # Condutividade Termica
dt = 0.05       # Time step
dx = 0.04       # Length step x
dy = 0.04       # Length step y

A1 = lmb*dt/(dx**2)
A2 = 1 - 2*lmb*dt/(dx**2)
A3 = lmb*dt/(dx**2)

# Critério de estabilidade: 'A1' <= 0.5
print(A1)

ti = 0
tf = 100
xi = -1
xf =  1
yi = -1
yf =  1

tempo = np.arange(ti, tf+dt, dt)
pos_x = np.arange(xi, xf+dx, dx)
pos_y = np.arange(yi, yf+dy, dy)
Nx = size(pos_x)
Ny = size(pos_y)
Nt = size(tempo)

Tp = np.ones([Nx*Ny, Nt]) # Temperatura
## C.I. T0_(x,y) = 10*exp^(-20(x^2 + y^2))
for i in range(Ny):
    for j in range(Nx):
        Tp[j+i*Nx,0] = 10*np.exp(-20*((pos_x[j])**2 + (pos_y[i])**2))
    #endfor    
#endfor
#print(Tp[:,0])



## C.C. (Neumann)
q_xi = np.zeros([Nx, Nt])
q_xf = np.zeros([Nx, Nt])
q_yi = np.zeros([Ny, Nt])
q_yf = np.zeros([Ny, Nt])

SolExp_Tp = np.reshape(Tp[:,0], (Ny, Nx))
print(shape(SolExp_Tp))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
pos_x, pos_y = np.meshgrid(pos_x, pos_y)
print(shape(pos_x), shape(pos_y))

# Plot the surface.
surf = ax.plot_surface(pos_y, pos_x, SolExp_Tp, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0.00, 11.00)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show(block=True)