#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Funcoes de forma para um elemento quadrilatero no plano \xi-\eta

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# Funcao de forma Bilinear
def shape_func_4q(idx, nodes, nodal_coordinates, xi, eta):
    val = 0.25*(1 + nodal_coordinates[idx][0]*xi)*(1 + nodal_coordinates[idx][1]*eta)*1 # nodic value
    return val

# Funcao de forma Qudratica
def shape_func_9q(idx, nodes, nodal_coordinates, xi, eta):
    Ni, Nj = 1,1
    if nodal_coordinates[idx][0] == 1:
        Ni = 0.5*((xi - 1)*xi)*1
    if nodal_coordinates[idx][0] == 2:
        Ni = (1 - xi*xi)*1
    if nodal_coordinates[idx][0] == 3:
        Ni = 0.5*((xi + 1)*xi)*1
    if nodal_coordinates[idx][1] == 1:
        Nj = 0.5*((eta - 1)*eta)*1
    if nodal_coordinates[idx][1] == 2:
        Nj = (1 - eta*eta)*1
    if nodal_coordinates[idx][1] == 3:
        Nj = 0.5*((eta + 1)*eta)*1
    return Ni*Nj


#--- Main Code ---#
xi  = np.linspace(-1, 1)
eta = np.linspace(-1, 1)
aproximacao = 'bilinear'
campo = np.zeros((np.size(xi), np.size(eta)))
no_number = 3 # cuidado: aproximacao linear so aceita No ate 4

if aproximacao == 'bilinear':
    Nodes = np.array([[-1, 1, 1, -1],[-1, -1, 1, 1]])
    nod_coords = {1:(-1,-1), 2:(1,-1), 3:(1,1), 4:(-1,1)}   # Xi-Eta table
    for i, x in enumerate(xi):
        for j, y in enumerate(eta):
            campo[i,j] = shape_func_4q(no_number, Nodes, nod_coords, x, y)

if aproximacao == 'quadratica':
    Nodes = np.array([[-1, 1, 1, -1,  0, 1, 0, -1, 0],
                      [-1, -1, 1, 1, -1, 0, 1,  0, 0]])
    nod_coords = {1:(1,1), 2:(3,1), 3:(3,3), 
                  4:(1,3), 5:(2,1), 6:(3,2), 
                  7:(2,3), 8:(1,2), 9:(2,2)}   # shape functions idx for the nine-node quadrilateral element
    for i, x in enumerate(xi):
        for j, y in enumerate(eta):
            campo[i,j] = shape_func_9q(no_number, Nodes, nod_coords, x, y)


# Plotting...
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
eixo_eta, eixo_xi = np.meshgrid(eta, xi)
surf = ax.plot_surface(eixo_xi, eixo_eta, campo,
                        cmap=cm.ocean, linewidth=0, antialiased=False)
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta$')
# Add a color bar which maps values to colors.
surf_label = '$\Theta_{' + str(no_number) + '}$'
fig.colorbar(surf, shrink=0.5, aspect=5, format='%.3f', label=surf_label)
plt.show()