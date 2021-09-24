# Problema de transferencia de calor unidimensional transiente

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy.core.fromnumeric import mean, shape, size
from numpy.core.shape_base import block

lmb = 0.01      # Difusividade Termica
cond = 1.00     # Condutividade Termica
dt = 0.003       # Time step
dx = 0.10       # Length step
dt = 0.01       # Time step
dx = 0.02       # Length step

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

Tp = np.ones([size(pos_x), size(tempo)]) * Tp_t0

## C.C. (Neumann)
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
            Tp[i,k] = 2*dx/cond*q_xi[k] + Tp[i+2,k]     # CC em x=0
        if i == (size(pos_x)-1):
            #Tp[i+1,k] = 2*dx/cond*q_xf[k] + Tp[i-1,k]   # CC em x=L (usando q_xf qql)
            Tp[i+1,k+1] = A2*Tp[i+1,k] + 2*A1*Tp[i,k]    # CC em x=L
        else:
            Tp[i,k+1] = A1*Tp[i+1,k] + A2*Tp[i,k] + A3*Tp[i-1,k]
        #endif
    #endfor
#endfor

# Sanity Check: Tp media em tf deve ser maior que Tp de t0,
# visto que calor entra na barra e apos 10s ela esta completamente adiabatica
print(mean(Tp[:,-1]))
SolExp_Tp = Tp.copy()

# Reiniciando as Temperaturas
Tp = np.ones([size(pos_x), size(tempo)]) * Tp_t0

## Abordagem Implicita:
# Montando a matrix A do sistema linear:
ofst = 1
d1 = -A1 * np.ones(size(pos_x)-ofst)
d2 =  A2 * np.ones(size(pos_x))
d3 = -A3 * np.ones(size(pos_x)-ofst)

A = np.diag(d2, 0)
A = A + np.diag(d1, ofst)
A = A + np.diag(d3,-ofst)
# CC:
A[0,:] =  0
A[0,0] =  1
A[0,2] = -1

A[-1, :] =  0
A[-1,-1] = A2
A[-1,-2] = -(A1 + A3)

print(np.linalg.det(A))

# Atalho para comentario de multiplas linhas: Shift + Alt + A
""" # A solucao do sistema linear inverte A. Como A é constante ao longo do tempo,
# invertemos fora do loop apenas uma vez para simplificacao computacional:
Ainv = np.linalg.inv(A)

for k in range(size(tempo)-1):
    b = Tp[:,k].copy()
    b[0]  = 2*dx/cond*q_xi[k+1]
    #b[-1] = 2*dx/cond*q_xf[k+1]
    Tp[:,k+1] = np.dot(Ainv,b)
#endfor
print(mean(Tp[:,-1])) """
SolImp_Tp = Tp.copy()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
tempo, pos_x = np.meshgrid(tempo, pos_x)

# Plot the surface.
surf = ax.plot_surface(pos_x, tempo, SolExp_Tp, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(1.00, 1.50)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show(block=False)

SolImp_Tp = SolExp_Tp.copy()                                # comentar aqui

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(pos_x, tempo, SolImp_Tp, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(1.00, 1.50)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show(block=True)