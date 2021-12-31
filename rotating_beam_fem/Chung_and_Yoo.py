#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2022           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Simulação Elastodinamica de uma barra sob rotacao e carregamento na ponta ("Perna Robotica"), baseado em:
# CHUNG, J.; YOO, Hong Hee. Dynamic analysis of a rotating cantilever beam by using the finite element method (2002).
# CHUNG, Jintai; HULBERT, G. M.. A time integration algorithm for structural dynamics with improved numerical dissipation: the generalized-α method (1993).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse
from scipy.sparse import linalg as sprlinalg
## Class Definition

class fem1DRotatingBeam():
    def __init__(self, end_time, time_steps, rho):
        self.nodes = []
        self.elements = []
        self.nnodes = 0
        self.nelements = 0
        self.rho_infty = rho                #Ref 2 eq.(25)
        self.alp_m = (2*rho - 1)/(rho + 1)  #Ref 2 eq.(25)
        self.alp_f = (rho)/(rho + 1)
        self.gamma = 0.5 - self.alp_m + self.alp_f         #Ref 2 eq.(17)
        self.beta = 0.25*(1 - self.alp_m + self.alp_f)**2  #Ref 2 eq.(20)
        self.end_time   = end_time
        self.num_steps = time_steps
        self.time = np.linspace(0.0, self.end_time, self.num_steps)

    def createNodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)

    def createElements(self, connectivities, props):
        self.connectivities = connectivities
        self.properties = props
        for item in connectivities:
            element = RB2(item, props)
            self.elements.append(element)
        self.nelements = len(self.elements)

    def createBoundaryConds(self, Omg, Fx, Fy, Fz):
        # Series temporais de Omega, Fx, Fy, Fz
        self.Omega = Omg
        self.ps = Fx
        self.pv = Fy
        self.pw = Fz

    def solve(self):

        row_m = []
        col_m = []
        val_m = []
        row_k = []
        col_k = []
        val_k = []
        row_g = []
        col_g = []
        val_g = []
        row_s = []
        col_s = []
        val_s = []

        for element in self.elements:
            r, c, v = element.me(self.nodes)
            row_m.append(r)
            col_m.append(c)
            val_m.append(v)
            r, c, v = element.ke(self.nodes)
            row_k.append(r)
            col_k.append(c)
            val_k.append(v)
            r, c, v = element.ge(self.nodes)
            row_g.append(r)
            col_g.append(c)
            val_g.append(v)
            r, c, v = element.se(self.nodes)
            row_s.append(r)
            col_s.append(c)
            val_s.append(v)

        row_k = np.array(row_k, dtype='int').flatten()
        col_k = np.array(col_k, dtype='int').flatten()
        val_k = np.array(val_k, dtype='float').flatten()
        # Montagem da matriz global K
        K_g = sparse.csr_matrix((val_k, (row_k, col_k)), shape=((self.nnodes, self.nnodes)))
        K_g = K_g + K_g.T - sparse.diags(K_g.diagonal(), dtype='float')

        row_m = np.array(row_m, dtype='int').flatten()
        col_m = np.array(col_m, dtype='int').flatten()
        val_m = np.array(val_m, dtype='float').flatten()
        # Montagem da matriz global M
        M_g = sparse.csr_matrix((val_m, (row_m, col_m)), shape=((self.nnodes, self.nnodes)))
        M_g = M_g + M_g.T - sparse.diags(M_g.diagonal(), dtype='float')

        # Montagem do vetor fglobal
        f_g = np.zeros(self.nnodes)

        self.K_g = K_g
        self.M_g = M_g
        self.f_g = f_g

        Tp = np.zeros((self.num_steps, self.nnodes))
        Tp[0,:] = self.temp_begin # condicao de temperatura inicial

        Dt = self.end_time/self.num_steps # Delta_t
        condutivity = k_term
        Area = 1.000 # area unitaria

        #Metodo Implicito
        Ak1 = ((1/Dt)*M_g + K_g) # precisava converter para usar inv da numpy: ".toarray()"
        Ak1_inv = sprlinalg.inv(Ak1.tocsr())
        Ak = condutivity*Ak1_inv*(1/Dt)*M_g


        for k, t in enumerate (self.time[:-1]):
            # ajustando o vetor f_g de acordo com BC de Neumann no tempo:
            for nd in self.bcs_nodes:
                if(nd[1] == 1): #BC de Neumann
                    if (nd[3] > 0) and (t <= nd[3]):
                        f_g[nd[0]] = condutivity*Area*nd[2]
                    if (nd[3] > 0) and (t > nd[3]):
                        f_g[nd[0]] = 0
                    if (nd[3] == -1):
                        f_g[nd[0]] = condutivity*Area*nd[2]
            # metodo implicito:
            Tp[k+1,:] = Ak*Tp[k,:] + Ak1_inv*f_g

        self.T = Tp

    def plot(self):
        return False


class RB2():
    def __init__(self, nodes, props):
        #{'Area':crossArea, 'Young':E, 'MoIz':Izz, 'MoIy':Iyy, 'Density':density, 'L':beamLength, 'a':offset}
        self.enodes = nodes
        self.L   = props['L']
        self.offset = props['a']
        self.A   = props['Area']
        self.E   = props['Young']
        self.Iz  = props['MoIz']
        self.Iy  = props['MoIy']
        self.rho = props['Density']

    def me(self, coords): # constroi a matrix M do elemento
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        M = np.zeros((6,6))
        # Ref. 1, eq.(34), using Matlab symbolic math...(check script)

        M[0,0] = h/3
        M[0,3] = h/6

        M[1,1] = 13*h/35
        M[1,2] = 11*h*h/210
        M[1,4] = 9*h/70
        M[1,5] = -13*h*h/420

        M[2,2] = h*h*h/105
        M[2,4] = 13*h*h/420
        M[2,5] = -h*h*h/140

        M[3,3] = h/3

        M[4,4] = 13*h/35
        M[4,5] = -11*h*h/210

        M[5,5] = h*h*h/105

        M = (self.rho*self.A/self.L)*M

        n1 = 3*self.enodes[0]
        n2 = 3*self.enodes[1]

        #  s1  v1 tht1    s2  v2 tht2     
        # n11 n12  n13   n14 n15  n16  s1
        #     n22  n23   n24 n25  n26  v1 
        #          n33   n34 n35  n36  tht1
        #               ...
        #                n44 n45  n46  s2
        #                    n55  n56  v2
        #                         n66  tht2
        row_id = 6*[n1] + 5*[n1+1] + 4*[n1+2] + 3*[n2] + 2*[n2+1] + 1*[n2 + 2] 
        col_id = [n1, n1+1, n1+2, n2, n2+1, n2+2, 
                      n1+1, n1+2, n2, n2+1, n2+2,
                            n1+2, n2, n2+1, n2+2,
                                  n2, n2+1, n2+2,
                                      n2+1, n2+2,
                                            n2+2]   #OK
        values =   [M[0,0], M[0,1], M[0,2], M[0,3], M[0,4], M[0,5],
                            M[1,1], M[1,2], M[1,3], M[1,4], M[1,5],
                                    M[2,2], M[2,3], M[2,4], M[2,5],
                                            M[3,3], M[3,4], M[3,5],
                                                    M[4,4], M[4,5],
                                                            M[5,5]] #OK

        M = M + M.T - np.diag(M.diagonal())
        self.Me = M
        self.h = h

        return row_id, col_id, values

    def ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        Area = 1 # area unitaria

        Ke_11 = (self.k *Area / self.L) 
        Ke_12 = -Ke_11
        Ke_22 =  Ke_11
        
        # Ke = [[(n1, n1), (n1, n2)], [(n2, n1), (n2,n2)]]
        n1 = self.enodes[0]
        n2 = self.enodes[1]
        ind_rows = [n1, n1, n2]
        ind_cols = [n1, n2, n2]
        values = [Ke_11, Ke_12, Ke_22]
        
        return ind_rows, ind_cols, values
    
    def ge(self, coords): # constroi a matrix G do elemento
        x = coords[self.enodes, 0]
        # ...

        n1 = self.enodes[0]
        n2 = self.enodes[1]
        ind_rows = [n1, n1, n2]
        ind_cols = [n1, n2, n2]
        values = [Ke_11, Ke_12, Ke_22]
        
        return ind_rows, ind_cols, values
    
    def se(self, coords): # constroi a matrix S do elemento
        x = coords[self.enodes, 0]
        #...

        n1 = self.enodes[0]
        n2 = self.enodes[1]
        ind_rows = [n1, n1, n2]
        ind_cols = [n1, n2, n2]
        values = [Ke_11, Ke_12, Ke_22]
        
        return ind_rows, ind_cols, values


## Main Code ##

beamLength = 0.30 #[m]
t_end = 30.00     #[s]
time_steps = 150
nnodes = 16
offset = beamLength*5/100
Radius = 2*beamLength/70 # Euler-Bernoulli beam theory constrain with alpha = 70, see Ref 1 eq (43)
crossArea = np.pi*Radius*Radius
Izz = 0.25* np.pi*pow(Radius, 4)
Iyy = 0.25* np.pi*pow(Radius, 4)
# ABS Properties:
E = 27e6        # [Pa] (27 MPa)
density = 1.2e3 # [Kg/m^3]

spec_rds = 0.7 #Spectral Radius (rho_inf) [Ref 2]
# util para o caso 2D:
coords = np.zeros((nnodes, 2)) 
coords[:,0] = np.linspace(0.0, beamLength, nnodes)
# 2 nos por elemento:
connectivities = np.zeros((nnodes - 1, 2), dtype='int')
connectivities[:,0] = np.arange(0, nnodes-1)
connectivities[:,1] = np.arange(1, nnodes)

problem = fem1DRotatingBeam(t_end, time_steps, spec_rds)
problem.createNodes(coords) # array com coords dos nos
props = {'Area':crossArea, 'Young':E, 'MoIz':Izz, 'MoIy':Iyy, 'Density':density, 'L':beamLength, 'a':offset}
problem.createElements(connectivities, props) # array dos elementos, 
# em que cada elemento contem os ids dos seus nos, mas o objeto element nao contem a info das coords

# Boundary conditions (Dynamic Conditions)
t = np.linspace(0.0, t_end, time_steps)
Omg = 1.0*np.sin(0.5*t)
Fx = 0*t
Fy = 0*t
Fz = 0*t
problem.createBoundaryConds(Omg, Fx, Fy, Fz)

problem.solve()

problem.plot()