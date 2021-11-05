#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Integral da funcao x^2 + y^2 via quadratura gaussiana.

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

nod_coords = {1:(-1,-1), 2:(1,-1), 3:(1,1), 4:(-1,1)}   # Xi-Eta table 4Q (global)

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

# Plotting...
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(f_de_xy[:,:,1], f_de_xy[:,:,2], f_de_xy[:,:,0], 
                                cmap=cm.ocean, edgecolor='black')
plt.show(block=False)

## Main Code ##

Xe = np.array([0, 2, 3, -1])
Ye = np.array([0, 1, 2,  2])
# Veja até 40, pois comeca a divergir a partir dos N=35
Nmax = 20
Integral = np.zeros(Nmax)

for N in range(Nmax):
    n_gp = N+1
    poly_ord = 2*n_gp - 1
    dim = poly_ord + 1

    xi  = np.linspace(-1,1,n_gp)
    eta = np.linspace(-1,1,n_gp)

    P_vec = np.zeros(dim)
    Wx_vec = np.zeros(n_gp)
    We_vec = np.zeros(n_gp)
    Mx_mtx = np.zeros((n_gp,dim))
    Me_mtx = np.zeros((n_gp,dim))

    # construcao de ^P:
    for i in range(dim):
        P_vec[i] = (1**(i+1) - (-1)**(i+1))/(i+1)

    # construcao de [M]:
    for i in range(n_gp):
        for j in range(dim):
            Mx_mtx[i,j] = xi[i]**(j)

    Wx_vec = np.matmul( np.linalg.pinv(Mx_mtx.T), P_vec) # pseudoinversa pois M nao eh quadrada
    We_vec = Wx_vec.copy()

    for i, x in enumerate(xi):
        for j, n in enumerate(eta):
            detJ = np.linalg.det(jac_de_xieta(Xe, Ye, x, n))
            Integral[N] += Wx_vec[i]*We_vec[j]*detJ*f_de_xieta(Xe, Ye, x, n)

#endfor
print(Integral[-1]) #3.4722222101382374
#Plotting
plt.figure()
N = np.linspace(1,Nmax,Nmax)
plt.plot(N, Integral,'--b')
plt.xlabel(r'$n$')
plt.ylabel(r'$\int x^2 + y^2 d\Omega$')
plt.grid()
plt.show(block=False)

plt.figure()
N = np.linspace(10,Nmax,Nmax-10)
plt.plot(N, Integral[10:],'--b')
plt.xlabel(r'$n$')
plt.ylabel(r'$\int x^2 + y^2 d\Omega$')
plt.grid()
plt.show(block=True)
