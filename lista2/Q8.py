#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## FEM: Transferencia de Calor em solidos (transiente)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse
from scipy.sparse import linalg as sprlinalg
## Class Definition

class fem1DHTTransient():
    def __init__(self, temp_init, end_time, time_steps):
        self.nodes = []
        self.elements = []
        self.nnodes = 0
        self.nelements = 0
        self.temp_begin = temp_init
        self.end_time   = end_time
        self.num_steps = time_steps
        self.time = np.linspace(0.0, self.end_time, self.num_steps)

    def createNodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)

    def createElements(self, connectivities):
        self.connectivities = connectivities
        for item in connectivities:
            element = HT2(item)
            self.elements.append(element)
        self.nelements = len(self.elements)

    def defineProperties(self, props):
        for element in self.elements:
            element.k = props['condutividade']
            element.rhocp = props['cap_termica_vol']

    def createBoundaryConds(self, conditions):
        self.bcs_nodes = [] # id/ tipo/ valor/ tempo
        eps = 0.000001
        for i,nd in enumerate(self.nodes):
            if nd[0] < eps: #(x=0)
                typ, val, time = conditions['E']
                self.bcs_nodes.append((i, typ, val, time))
            if nd[0] > (1-eps): #(x=1)
                typ, val, time = conditions['D']
                self.bcs_nodes.append((i, typ, val, time))

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

        #Metodo Implicito
        Ak1 = ((1/Dt)*Mglobal + Kglobal) # precisava converter para usar inv da numpy: ".toarray()"
        Ak1_inv = sprlinalg.inv(Ak1.tocsr())
        Ak = Ak1_inv*(1/Dt)*Mglobal

        condutivity = k_term
        Area = 1.000 # area unitaria

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


class HT2():
    def __init__(self, nodes):
        self.enodes = nodes
        self.rhocp = 0
        self.k = 0

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

Ti = 1.000      #[°C]
L_barra = 1.000 #[m]
t_end = 30.00   #[s]
k_term = 1.000   #[W/(m^2 K)]
cap_term_vol = 1.00*100.00   #[J/(m^3 K)]
time_steps = 150
nnodes = 16
# util para o caso 2D:
coords = np.zeros((nnodes, 2)) 
coords[:,0] = np.linspace(0.0, L_barra, nnodes)
# 2 nos por elemento:
connectivities = np.zeros((nnodes - 1, 2), dtype='int')
connectivities[:,0] = np.arange(0, nnodes-1)
connectivities[:,1] = np.arange(1, nnodes)

problem = fem1DHTTransient(Ti, t_end, time_steps)
problem.createNodes(coords) # array com coords dos nos
problem.createElements(connectivities) # array dos elementos, 
# em que cada elemento contem os ids dos seus nos, mas o objeto element nao contem a info das coords
props = {'condutividade':k_term, 'cap_termica_vol':cap_term_vol}
problem.defineProperties(props)

# Boundary conditions
qe  = 1.00
qd  = 0.00
# tipos: 0: Dirichlet / 1: Neumann 
# local: E: Esquerda / D: Direita
# tempo: -1: qql q seja t / >0: tempo ATÉ o qual a condicao eh valida
bcs = {'E':(1,qe,10), 'D':(1,qd,-1)}
problem.createBoundaryConds(bcs)

problem.solve()

problem.plot()