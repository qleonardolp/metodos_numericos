#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

# elemento finito com aproximação cúbica

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block

def shape_funct(idx, domain, x):
    val = 1
    if np.size(domain) >= 3:
        for i, xd in enumerate(domain):
            if (i+1) != idx:
                val *= (x - xd)/(domain[idx-1] - xd)
    
    return val


x1 = 1
h  = 3
Dom = np.arange(x1, x1+h + 1)
val_nods = np.array([1, 2, 3, 2.5])
x = np.linspace(x1, x1+h, 100)
Campo = np.zeros(np.size(x))

plt.figure()
plt.plot(x, shape_funct(1,Dom,x), label='N1')
plt.plot(x, shape_funct(2,Dom,x), label='N2')
plt.plot(x, shape_funct(3,Dom,x), label='N3')
plt.plot(x, shape_funct(4,Dom,x), label='N4')
plt.legend(loc='upper right')
plt.grid()
plt.show(block=False)


for k, xi in enumerate(x):
    for i, node in enumerate(val_nods):
        Campo[k] += node*shape_funct(i+1,Dom,xi)

# Equivalente a:
""" 
Campo = (val_nods[0]*shape_funct(1,Dom,x) + val_nods[1]*shape_funct(2,Dom,x) + 
         val_nods[2]*shape_funct(3,Dom,x) + val_nods[3]*shape_funct(4,Dom,x) )
 """

plt.figure()
plt.plot(x, Campo, label='Campo')
plt.legend(loc='upper right')
plt.grid()
plt.show(block=True)

