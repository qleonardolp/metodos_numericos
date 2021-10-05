#///////////////////////////////////////////////////////
#// Leonardo Felipe Lima Santos dos Santos, 2021     ///
#// leonardo.felipe.santos@usp.br	_____ ___  ___   //
#// github/bitbucket qleonardolp	  |  | . \/   \  //
#////////////////////////////////	| |   \ \   |_|  //
#////////////////////////////////	\_'_/\_`_/__|    //
#///////////////////////////////////////////////////////

# Exercicio para calcular aproximacoes (FDM) de df/dx e d2f/dx2 para f(x) = e^(x)*sin(x)

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from math import sin, cos

## f(x) definida como uma função python para simplificar sintaxe
def fdex(x):
    return np.exp(x) * np.sin(x)

def fdex_dec(x):
    return Decimal(x).exp() * Decimal(sin(x))

x=2
print( format(fdex(x), '.20f') )
print( format(fdex_dec(x), '.20f') )

## Analiticamente temos:
def dfdex(x):
    return np.exp(x) * (np.sin(x) + np.cos(x))

def ddfdex(x):
    return np.exp(x) * np.cos(x) * 2

def dfdex_dec(x):
    return Decimal(x).exp() * Decimal(sin(x) + cos(x))

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

# d_fx 'C4' usando representacao decimal do Python
def cent4_dfdex_dec(x, dx):
    dx = Decimal(dx)
    if not(dx == 0):
        return (fdex_dec(x-2*dx) -8*fdex_dec(x-dx) + 8*fdex_dec(x+dx) - fdex_dec(x+2*dx))/ (12*dx)
    else:
        dx = 0.001
        return (fdex_dec(x-2*dx) -8*fdex_dec(x-dx) + 8*fdex_dec(x+dx) - fdex_dec(x+2*dx))/ (12*dx)

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
anl_dfdex_dec = dfdex_dec(x)

num_step = 500
step = np.logspace(-7,-1,num_step)
estC2dfdex = np.zeros(np.size(step))
estC4dfdex = estC2dfdex.copy()
estC2Ddfdex = estC2dfdex.copy()
estC4Ddfdex = estC2dfdex.copy()
errC4dfdex_dec = estC2dfdex.copy()

for i in range(step.size):
    estC2dfdex[i]     = cent2_dfdex(x,step[i])
    estC4dfdex[i]     = cent4_dfdex(x,step[i])
    estC2Ddfdex[i]    = cent2_ddfdex(x,step[i])
    estC4Ddfdex[i]    = cent4_ddfdex(x,step[i])
    errC4dfdex_dec[i] = anl_dfdex_dec - cent4_dfdex_dec(x,step[i])

##  Absolute Error Plots (log-log):

plt.figure()
plt.loglog(step, abs(anl_dfdex*np.ones(num_step) - estC2dfdex), label='C2' )
plt.loglog(step, abs(anl_dfdex*np.ones(num_step) - estC4dfdex), label='C4' )
plt.title('Gráfico log-log do erro das estimativas de f\'(2) em funçao de $\Delta x$')
plt.xlabel('$\Delta x$')
plt.ylabel('$|E|$')
plt.legend(loc='upper right')
plt.grid()
plt.show(block=False)


plt.figure()
plt.loglog(step, abs(anl_ddfdex*np.ones(num_step) - estC2Ddfdex), label='C2' )
plt.loglog(step, abs(anl_ddfdex*np.ones(num_step) - estC4Ddfdex), label='C4' )
plt.title('Gráfico log-log do erro das estimativas de f\'\'(2) em funçao de $\Delta x$')
plt.xlabel('$\Delta x$')
plt.ylabel('$|E|$')
plt.legend(loc='upper right')
plt.grid()
plt.show(block=False)

plt.figure()
plt.loglog(step, abs(anl_dfdex*np.ones(num_step) - estC4dfdex), label='C4' )
plt.loglog(step, abs(errC4dfdex_dec), label='C4 decimal' )
plt.title('Log-log do erro de f\'(2) C4 por $\Delta x$ usando python decimals')
plt.xlabel('$\Delta x$')
plt.ylabel('$|E|$')
plt.legend(loc='upper right')
plt.grid()
plt.show(block=True)


