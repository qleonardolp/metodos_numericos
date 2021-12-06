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
from scipy.sparse.linalg import spsolve

## Class Definition

class fem1DHTTransient():
    def __init__(self, temp_init, end_time):
        self.nodes = []
        self.elements = []
        self.nnodes = 0
        self.nelements = 0
        self.temp_begin = temp_init
        self.end_time   = end_time
        self.num_steps = 300
        self.time = np.linspace(0.0, self.end_time, self.num_steps)

    def createNodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)

    def createElements(self, connectivities, condutividade, rhocp, area):
        self.connectivities = connectivities
        for k,item in enumerate(connectivities):
            element = HT2(item, condutividade[k], rhocp[k], area[k])
            self.elements.append(element)

        self.nelements = len(self.elements)

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

        rowsk = []
        colsk = []
        valuesk = []
        rowsm = []
        colsm = []
        valuesm = []
        area = []

        for element in self.elements:
            r, c, v = element.Ke(self.nodes)
            rowsk.append(r)
            colsk.append(c)
            valuesk.append(v)
            r, c, v = element.Me(self.nodes)
            rowsm.append(r)
            colsm.append(c)
            valuesm.append(v)
            area.append(element.A)

        rowsk = np.array(rowsk, dtype='int').flatten()
        colsk = np.array(colsk, dtype='int').flatten()
        valuesk = np.array(valuesk, dtype='float').flatten()
        rowsm = np.array(rowsm, dtype='int').flatten()
        colsm = np.array(colsm, dtype='int').flatten()
        valuesm = np.array(valuesm, dtype='float').flatten()

        # Montagem da matriz global K
        Kglobal = sparse.csr_matrix((valuesk, (rowsk, colsk)), shape=((self.nnodes, self.nnodes)))
        Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')
        # Montagem da matriz global M
        Mglobal = sparse.csr_matrix((valuesm, (rowsm, colsm)), shape=((self.nnodes, self.nnodes)))
        Mglobal = Mglobal + Mglobal.T - sparse.diags(Mglobal.diagonal(), dtype='float')

        self.Kglobal = Kglobal
        self.Mglobal = Mglobal

        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(self.nnodes)

        # 2) Aplicação das BCs no vetor fglobal
        # 3) Aplicar as condições de contorno na matriz Kglobal
        w = 1e20
        # usado para BCs de Dirichlet
        for nd in self.bcs_nodes:
            if nd[1] == 0: # B.C. de Dirichlet
                fglobal[nd[0]] = nd[2] * w
                Kglobal[nd[0], nd[0]] += w
        
        self.Kglobal = Kglobal
        self.fglobal = fglobal

        Tp = np.zeros((self.num_steps, self.nnodes))
        Tp[0,:] = self.temp_begin # condicao de temperatura inicial

        timestep_1 = self.num_steps # inverso de Delta_t

        #Metodo Implicito
        Ak1 = (timestep_1*Mglobal + Kglobal).toarray() # precisa converter para usar inv
        Ak1inv = np.linalg.inv(Ak1)
        Ak = np.matmul(Ak1inv, timestep_1*Mglobal.toarray()) # precisa converter para usar inv

        for k, t in enumerate (self.time[:-1]):
            # ajustando o vetor fglobal de acordo com BC de Dirichlet
            for nd in self.bcs_nodes:
                Ar = 0.10 # ajustar isso para usar area de cada elemento
                if (nd[1] == 1) and (nd[3] > 0) and (t <= nd[3]):
                    fglobal[nd[0]] = Ar*nd[2]
                if (nd[1] == 1) and (nd[3] > 0) and (t > nd[3]):
                    fglobal[nd[0]] = 0
                if (nd[1] == 1) and (nd[3] == -1):
                    fglobal[nd[0]] = Ar*nd[2]

            Tp[k+1,:] = np.matmul(Ak,Tp[k,:]) + np.matmul(Ak1inv, fglobal)

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
    def __init__(self, nodes, k, rhocp, area):
        self.enodes = nodes
        self.k = k
        self.rhocp = rhocp
        self.A = area

    def Ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        self.L = abs(x[1] - x[0])

        Ke_11 = (self.k * self.A / self.L) 
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
        
        # Ke = [[(n1, n1), (n1, n2)], [(n2, n1), (n2,n2)]]
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
#alpha = 1/(1.00*100) # k/rho*c [m^2/s]
condutividade_termica = 1.000            #[W/(m^2 K)]
cap_termica_vol = 1.00*100.00   #[J/(m^3 K)]
A = 0.100 # area da secao transversal
nnodes = 16
# util para o caso 2D:
coords = np.zeros((nnodes, 2)) 
coords[:,0] = np.linspace(0.0, L_barra, nnodes)
# 2 nos por elemento
connectivities = np.zeros((nnodes - 1, 2), dtype='int')
connectivities[:,0] = np.arange(0, nnodes-1)
connectivities[:,1] = np.arange(1, nnodes)

cond = condutividade_termica*np.ones((nnodes - 1, 1))
rhocp = cap_termica_vol*np.ones((nnodes - 1, 1))
areas = A*np.ones((nnodes - 1, 1))

problem = fem1DHTTransient(Ti, t_end)
problem.createNodes(coords) # array com coords dos nos
problem.createElements(connectivities, cond, rhocp, areas) 
# array dos elementos, em que cada elemento contem os ids dos seus nos, mas o objeto element nao contem a info das coords

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