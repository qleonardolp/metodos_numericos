#///////////////////////////////////////////////////////
#// Leonardo Felipe Lima Santos dos Santos, 2021     ///
#// leonardo.felipe.santos@usp.br	_____ ___  ___   //
#// github/bitbucket qleonardolp	  |  | . \/   \  //
#////////////////////////////////	| |   \ \   |_|  //
#////////////////////////////////	\_'_/\_`_/__|    //
#///////////////////////////////////////////////////////

# Problema de transferencia de calor unidimensional transiente

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

lmb = 0.01      # Difusividade Termica
cond = 1.00     # Condutividade Termica
dt = 0.003       # Time step
dx = 0.05       # Length step
#dt = 0.01       # Time step
#dx = 0.02       # Length step

A1 = lmb*dt/(dx**2)
A2 = 1 - 2*lmb*dt/(dx**2)
A3 = A1

# Critério de estabilidade: 'A1' <= 0.5
print(A1)

ti = 0
tf = 30
xi = 0
xf = 1

Tp_t0 = 1   # C.I.

tempo = np.arange(ti, tf+dt, dt)
pos_x = np.arange(xi, xf+dx, dx)
Nx = np.size(pos_x)
Nt = np.size(tempo)

Tp = np.ones([Nx, Nt]) * Tp_t0

## C.C. (Neumann)
q_xf = np.zeros(Nt)
q_xi = np.zeros(Nt)
for i in range(Nt):
    if tempo[i] <= 10:
        q_xi[i] = 1
    #endif
#endfor

## Abordagem Explicita:
for k in range(Nt):
    if k == (Nt-1):
        Tp[0,k] = 2*dx/cond*q_xi[k] + Tp[0+2,k]     # CC em x=0, para t final
        Tp[Nx-1,k] = Tp[Nx-1-2,k]                   # CC em x=L, para t final
    else:
        Tp[0,k] = 2*dx/cond*q_xi[k] + Tp[0+2,k]     # CC em x=0
        Tp[Nx-1,k] = Tp[Nx-1-2,k]                   # CC em x=L (adiabatica)
        for i in range(1,Nx-1):
            if i == (Nx-2):
                Tp[i,k+1] = A2*Tp[i,k] + 2*A3*Tp[i-1,k]   # CC em x=L (adiabatica)
            else:
                Tp[i,k+1] = A1*Tp[i+1,k] + A2*Tp[i,k] + A3*Tp[i-1,k]
        #endfor
#endfor

# Sanity Check: Tp media em tf deve ser maior que Tp de t0,
# visto que calor entra na barra e apos 10s ela esta completamente adiabatica
print(np.mean(Tp[:,-1]))
SolExp_Tp = Tp.copy()

# Reiniciando as Temperaturas
Tp = np.ones([Nx, Nt]) * Tp_t0

## Abordagem Implicita:
A1 = lmb*dt/(dx**2)
A2 = 1 + 2*lmb*dt/(dx**2)
A3 = A1
# Montando a matrix A do sistema linear:
ofst = 1
d1 = -A1 * np.ones(Nx-ofst)
d2 =  A2 * np.ones(Nx)
d3 = -A3 * np.ones(Nx-ofst)

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

# Atalho para comentario de multiplas linhas: Shift + Alt + A
# A solucao do sistema linear inverte A. Como A é constante ao longo do tempo,
# invertemos fora do loop apenas uma vez para simplificacao computacional:
Ainv = np.linalg.inv(A)

for k in range(Nt-1):
    b = Tp[:,k].copy()
    b[0]  = 2*dx/cond*q_xi[k+1]
    #b[-1] = 2*dx/cond*q_xf[k+1]
    Tp[:,k+1] = np.dot(Ainv,b)
#endfor
print(np.mean(Tp[:,-1]))
SolImp_Tp = Tp.copy()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title('Solução Explícita')
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

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title('Solução Implícita')

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