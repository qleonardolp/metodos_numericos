#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Integral da funcao x^2 + y^2 via quadratura gaussiana.
# Versao errada:
# 0.0   3.113988467592265   3.472500366720472   3.472221229275364
# Versao corrigida:
# 1.40625   3.472222222222221   3.4722222222222237  3.4722222222222205
# Versao corrigida (N<4):
# 1.40625   3.472222222222222   3.472222222222224  3.472222222222221

from numpy.polynomial.legendre import leggauss
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

nod_coords = {1:(-1,-1), 2:(1,-1), 3:(1,1), 4:(-1,1)}   # Xi-Eta table 4Q (var global)

## Funcoes auxiliares

# Xi-Eta Mapping
def map4q(nodes, xi, eta):  # passar nodes contendo x ou y, nao o par (x,y)
    val = 0
    for i, var  in enumerate(nodes):
        val = 0.25*(1 + nod_coords[i+1][0]*xi)*(1 + nod_coords[i+1][1]*eta)*var
    return val

def f_de_xieta(xe, ye, xi, eta):
    f = (map4q(xe, xi, eta))**2 + (map4q(ye, xi, eta))**2
    return f

def jac_de_xieta(xe, ye, xi, eta):
    J = np.zeros((2,2))
    delMap_xi  = np.zeros(4)
    delMap_eta = np.zeros(4)
    for i in range(4):
        delMap_xi[i]  = 0.25*(1 + nod_coords[i+1][1]*eta)*nod_coords[i+1][0]
        delMap_eta[i] = 0.25*(1 + nod_coords[i+1][0]*xi)*nod_coords[i+1][1]
    J[0,0] = np.dot(delMap_xi, xe)
    J[0,1] = np.dot(delMap_xi, ye)
    J[1,0] = np.dot(delMap_eta, xe)
    J[1,1] = np.dot(delMap_eta, ye)
    return J
##

# Check functions:
Xe = np.array([0, 2, 3, -1])
Ye = np.array([0, 1, 2,  2])
xi  = np.linspace(-1,1)
eta = np.linspace(-1,1)
f_de_xy = np.zeros((50,50,3))

for i, x in enumerate(xi):
        for j, n in enumerate(eta):
            f_de_xy[i,j,0] = f_de_xieta(Xe, Ye, x, n)
            f_de_xy[i,j,1] = map4q(Xe, x, n)
            f_de_xy[i,j,2] = map4q(Ye, x, n)

# Plotting... nao sei pq so plot uma fronteira, quando na vdd eu esperava a superficie
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(f_de_xy[:,:,1], f_de_xy[:,:,2], f_de_xy[:,:,0], 
                                cmap=cm.ocean, edgecolor='black')
plt.show(block=False)

## Main Code ##

Xe = np.array([0, 2, 3, -1])
Ye = np.array([0, 1, 2,  2])
Nmax = 4
Integral = np.zeros(Nmax)

for N in range(Nmax):
    n_gp = N+1
    poly_ord = 2*n_gp - 1
    dim = poly_ord + 1

    # sample points and weights for Gauss-Legendre quadrature:
    xi, Wx_vec = leggauss(n_gp)
    eta, We_vec = leggauss(n_gp)

    for i, x in enumerate(xi):
        for j, n in enumerate(eta):
            detJ = np.linalg.det(jac_de_xieta(Xe, Ye, x, n))
            Integral[N] += Wx_vec[i]*We_vec[j]*detJ*f_de_xieta(Xe, Ye, x, n)
#endfor

print(Integral)

#quit()
#Plotting
plt.figure()
N = np.linspace(1,Nmax,Nmax)
plt.plot(N, Integral,'--b')
plt.xlabel(r'$\sqrt{n}$')
plt.ylabel(r'$\int x^2 + y^2 d\Omega$')
plt.grid()
plt.show(block=True)

quit()
plt.figure()
N = np.linspace(10,Nmax,Nmax-10)
plt.plot(N, Integral[10:],'--b')
plt.xlabel(r'$n$')
plt.ylabel(r'$\int x^2 + y^2 d\Omega$')
plt.grid()
plt.show(block=True)
