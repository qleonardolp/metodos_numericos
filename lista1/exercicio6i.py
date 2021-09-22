# Problema de transferencia de calor unidimensional transiente

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy.core.fromnumeric import shape, size

lmb = 0.01
cond = 1.00
dt = 0.01
dx = 0.02

A1 = lmb*dt/(dx**2)
A2 = 1 - 2*lmb*dt/(dx**2)
A3 = lmb*dt/(dx**2)

# Critério de estabilidade: 'A1' <= 0.5
print(A1)

ti = 0
tf = 30
xi = 0
xf = 1

Tp_t0 = 1   # C.I.

tempo = np.arange(ti, tf+dt, dt)
pos_x = np.arange(xi, xf+dx, dx)

Tp = np.zeros([size(pos_x), size(tempo)])
Tp[:,0] = Tp_t0

## C.C.
q_xf = np.zeros(size(tempo))
q_xi = np.zeros(size(tempo))
for i in range(size(tempo)):
    if tempo[i] <= 10:
        q_xi[i] = 1
    #endif
#endfor

## Abordagem Explicita:
for k in range(size(tempo)-1):
    for i in range(size(pos_x)-1):
        if i == 0:
            Tp[i,k] = 2*dx/cond*q_xi[k] + Tp[i+2,k] # CC em x=0
        if i == (size(pos_x)-1):
            Tp[i+1,k] = Tp[i-1,k]                   # CC em x=L
        else:
            Tp[i,k+1] = A1*Tp[i+1,k] + A2*Tp[i,k] + A3*Tp[i-1,k]
        #endif
    #endfor
#endfor

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
tempo, pos_x = np.meshgrid(tempo, pos_x)
surf = ax.plot_surface(pos_x, tempo, Tp, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
