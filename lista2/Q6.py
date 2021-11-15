#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Integral da funcao de Bessel (0) via quadratura gaussiana.
# Resultado pelo WolframAlpha: 1.3875672520098649871
# Resultado para N=30          1.3875672520098639

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

    # sample points and weights for Gauss-Legendre quadrature:
    eta, W_vec  = np.polynomial.legendre.leggauss(n_gp) 
    fdex = np.zeros(n_gp)
    
    for i, eta_i in enumerate(eta):
        xk = 0.5*(Lsup + Linf) + 0.5*eta_i*(Lsup - Linf)
        fdex[i] = sp.jv(0,xk)   # Funcao de Bessel do primeiro tipo (J_0)

    Integral[N] = 0.5*(Lsup - Linf)*np.dot(W_vec, fdex)
#endfor

print(Integral[-1])

#Plotting
plt.figure()
N = np.linspace(1,30,30)
plt.plot(N, Integral,'--b')
plt.xlabel(r'$n$')
plt.ylabel(r'$\int J_0(x) dx$')
plt.grid()
plt.show(block=True)
quit()

plt.figure()
N = np.linspace(10,30,20)
plt.plot(N, Integral[10:],'--b')
plt.xlabel(r'$n$')
plt.ylabel(r'$\int J_0(x) dx$')
plt.grid()
plt.show(block=True)
