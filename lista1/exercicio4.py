# Exercicio para calcular aproximacoes da funcao de Bessel por diferencas finitas

import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plot

x = 5
dx = 0.005
erros = np.zeros(6)
approx = np.zeros(6)
solucao_exata = -sp.jv(1,x) # Bessel Funct Derivative, -J1(x) = dJ0(x)/dx

#print(sp.jv(0,x))

## Estimativas da derivada em torno de x=5
# a) Backward com 2 pts
approx[0] = ( sp.jv(0,x) - sp.jv(0,x-dx) )/dx

# b) Backward com 3 pts
approx[1] = ( 3*sp.jv(0,x) - 4*sp.jv(0,x-dx) + 1*sp.jv(0,x-2*dx)  )/(2*dx)

# c) Forward com 2 pts
approx[2] = ( sp.jv(0,x+dx) - sp.jv(0,x) )/dx

# d) Forward com 3 pts
approx[3] = ( -3*sp.jv(0,x) + 4*sp.jv(0,x+dx) - 1*sp.jv(0,x+2*dx)  )/(2*dx)

# e) Central com 2 pts
approx[4] = ( sp.jv(0,x+dx) - sp.jv(0,x-dx) ) / (2*dx)

# e) Central com 4 pts
approx[5] = ( -sp.jv(0,x+2*dx) + 8*sp.jv(0,x+dx) - 8*sp.jv(0,x-dx) + sp.jv(0,x-2*dx) ) / (12*dx)

erros = solucao_exata*np.ones(6) - approx

print(solucao_exata)
print(erros)

