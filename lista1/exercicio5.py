# Exercicio para calcular aproximacoes (FDM) de df/dx e d2f/dx2 para f(x) = e^(x)*sin(x)

import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt

## f(x) definida como uma função python para simplificar sintaxe
def fdex(x):
    return np.exp(x) * np.sin(x)

## Analiticamente temos:
def dfdex(x):
    return np.exp(x) * (np.sin(x) + np.cos(x))

def ddfdex(x):
    return np.exp(x) * np.cos(x) * 2

## Aproximacoes por FDM
def cent2_dfdex(x, dx):
    if not(dx == 0): # segurança
        return (-fdex(x-dx) + fdex(x+dx))/ (2*dx)
    else:
        dx = 0.001
        return (-fdex(x-dx) + fdex(x+dx))/ (2*dx)

def cent4_dfdex(x, dx):
    if not(dx == 0):
        return (fdex(x-2*dx) -8*fdex(x-dx) + 8*fdex(x+dx) - fdex(x+2*dx))/ (12*dx)
    else:
        dx = 0.001
        return (fdex(x-2*dx) -8*fdex(x-dx) + 8*fdex(x+dx) - fdex(x+2*dx))/ (12*dx)

def cent2_ddfdex(x, dx):    # na vdd central 3
    if not(dx == 0):
        return (fdex(x-dx) -2*fdex(x) + fdex(x+dx))/ (dx*dx)
    else:
        dx = 0.001
        return (fdex(x-dx) -2*fdex(x) + fdex(x+dx))/ (dx*dx)

def cent4_ddfdex(x, dx):    # na vdd central 5
    if not(dx == 0):
        return (-fdex(x-2*dx) +16*fdex(x-dx) -30*fdex(x) + 16*fdex(x+dx) - fdex(x+2*dx))/ (12*dx*dx)
    else:
        dx = 0.001
        return (-fdex(x-2*dx) +16*fdex(x-dx) -30*fdex(x) + 16*fdex(x+dx) - fdex(x+2*dx))/ (12*dx*dx)


x = 2
anl_dfdex = dfdex(x)
anl_ddfdex = ddfdex(x)

num_step = 100
step = np.logspace(-7,-1,num_step)
estC2dfdex = np.zeros(np.size(step))
estC4dfdex = estC2dfdex.copy()
estC2Ddfdex = estC2dfdex.copy()
estC4Ddfdex = estC2dfdex.copy()

for i in range(step.size):
    estC2dfdex[i] = cent2_dfdex(x,step[i])
    estC4dfdex[i] = cent4_dfdex(x,step[i])
    estC2Ddfdex[i] = cent2_ddfdex(x,step[i])
    estC4Ddfdex[i] = cent4_ddfdex(x,step[i])

##  Absolute Error Plots (log-log):

plt.figure()
plt.loglog(step, abs(anl_dfdex*np.ones(num_step) - estC2dfdex) )
plt.loglog(step, abs(anl_dfdex*np.ones(num_step) - estC4dfdex) )
plt.grid()
plt.show(block=False)


plt.figure()
plt.loglog(step, abs(anl_ddfdex*np.ones(num_step) - estC2Ddfdex) )
plt.loglog(step, abs(anl_ddfdex*np.ones(num_step) - estC4Ddfdex) )
plt.grid()
plt.show(block=True)


