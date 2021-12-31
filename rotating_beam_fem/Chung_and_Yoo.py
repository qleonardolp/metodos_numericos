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

        rowsK = []
        colsK = []
        valuesK = []
        rowsM = []
        colsM = []
        valuesM = []

        for element in self.elements:
            r, c, v = element.Ke(self.nodes)
            rowsK.append(r)
            colsK.append(c)
            valuesK.append(v)
            r, c, v = element.Me(self.nodes)
            rowsM.append(r)
            colsM.append(c)
            valuesM.append(v)

        rowsK = np.array(rowsK, dtype='int').flatten()
        colsK = np.array(colsK, dtype='int').flatten()
        valuesK = np.array(valuesK, dtype='float').flatten()
        # Montagem da matriz global K
        Kglobal = sparse.csr_matrix((valuesK, (rowsK, colsK)), shape=((self.nnodes, self.nnodes)))
        Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')

        rowsM = np.array(rowsM, dtype='int').flatten()
        colsM = np.array(colsM, dtype='int').flatten()
        valuesM = np.array(valuesM, dtype='float').flatten()
        # Montagem da matriz global M
        Mglobal = sparse.csr_matrix((valuesM, (rowsM, colsM)), shape=((self.nnodes, self.nnodes)))
        Mglobal = Mglobal + Mglobal.T - sparse.diags(Mglobal.diagonal(), dtype='float')

        self.Kglobal = Kglobal
        self.Mglobal = Mglobal

        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(self.nnodes)
        
        self.Kglobal = Kglobal
        self.fglobal = fglobal

        Tp = np.zeros((self.num_steps, self.nnodes))
        Tp[0,:] = self.temp_begin # condicao de temperatura inicial

        Dt = self.end_time/self.num_steps # Delta_t
        condutivity = k_term
        Area = 1.000 # area unitaria

        #Metodo Implicito
        Ak1 = ((1/Dt)*Mglobal + Kglobal) # precisava converter para usar inv da numpy: ".toarray()"
        Ak1_inv = sprlinalg.inv(Ak1.tocsr())
        Ak = condutivity*Ak1_inv*(1/Dt)*Mglobal


        for k, t in enumerate (self.time[:-1]):
            # ajustando o vetor fglobal de acordo com BC de Neumann no tempo:
            for nd in self.bcs_nodes:
                if(nd[1] == 1): #BC de Neumann
                    if (nd[3] > 0) and (t <= nd[3]):
                        fglobal[nd[0]] = condutivity*Area*nd[2]
                    if (nd[3] > 0) and (t > nd[3]):
                        fglobal[nd[0]] = 0
                    if (nd[3] == -1):
                        fglobal[nd[0]] = condutivity*Area*nd[2]
            # metodo implicito:
            Tp[k+1,:] = Ak*Tp[k,:] + Ak1_inv*fglobal

        self.T = Tp

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        eixo_x, eixo_t = np.meshgrid(self.nodes[:,0], self.time)
        surf = ax.plot_surface(eixo_x, eixo_t, self.T,
                                cmap=cm.plasma, linewidth=0, antialiased=False)
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        plt.xlabel('x (m)')
        plt.ylabel('tempo (s)')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5, format='%.3f', label='Temperatura (°C)')
        plt.show()


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

    def Ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        self.L = abs(x[1] - x[0])
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

    def Me(self, coords):
        x = coords[self.enodes, 0]
        self.L = abs(x[1] - x[0])

        Ke_11 = 2/6*(self.rhocp * self.L) # Jacob Fish, pg 99
        Ke_12 = Ke_11/2
        Ke_22 = Ke_11
        
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