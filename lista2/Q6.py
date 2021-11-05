#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Integral da funcao de Bessel (0) via quadratura gaussiana.
# Resultado pelo WolframAlpha: 1.3875672520098649871

import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt
## Main Code ##

Linf = 0
Lsup = 3
Integral = np.zeros(30)

for N in range(30):
    n_gp = N+1
    poly_ord = 2*n_gp - 1
    dim = poly_ord + 1

    x    = np.linspace(Linf, Lsup, n_gp)
    eta  = np.zeros(n_gp)
    fdex = np.zeros(n_gp)
    # linear mapping
    for i, xk in enumerate(x):
        eta[i]  = (2*xk - (Linf + Lsup))/(Lsup - Linf)
        fdex[i] = sp.jv(0,xk)   # Funcao de Bessel do primeiro tipo (J_0)

    P_vec = np.zeros(dim)
    W_vec = np.zeros(n_gp)
    M_mtx = np.zeros((n_gp,dim))

    # construcao de ^P:
    for i in range(dim):
        P_vec[i] = (1**(i+1) - (-1)**(i+1))/(i+1)

    # construcao de [M]:
    for i in range(n_gp):
        for j in range(dim):
            M_mtx[i,j] = eta[i]**(j)

    W_vec = np.matmul( np.linalg.pinv(M_mtx.T), P_vec) # pseudoinversa pois M nao eh quadrada

    Integral[N] = 0.5*(Lsup - Linf)*np.dot(W_vec, fdex)
#endfor

#Plotting
plt.figure()
plt.plot(np.linspace(0,30), Integral)
plt.grid()
plt.show()
