#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Transferencia de Calor em solidos (estacionario)

import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

plt.style.use('dark_background') # comentar essa linha para ter as figuras com fundo branco

## FEM2DHT Class Definition

class fem2DHeaTransfer():
    def __init__(self):
        self.nnodes = 0
        self.nelements = 0

    def createNodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)

    def createElements(self, connectivities):
        self.elements = []
        self.connectivities = connectivities
        for item in connectivities:
            element = HT3(item)
            self.elements.append(element)

        self.nelements = len(self.elements)

    def defineProperties(self, props):
        for element in self.elements:
            element.props = props

    def createBoundaryConds(self, conditions, nodes, values):
        self.nds_left = []
        self.nds_right = []
        self.nds_top = []
        self.nds_bot = []
        self.vals_left = []
        self.vals_right = []
        self.vals_top = []
        self.vals_bot = []
        eps = 0.000001
        for k,elemt in enumerate(self.elements):
            if elemt.nodes[0] < eps: #(x=0)
                self.nds_left.append(elemt.nodes)
                self.vals_left.append(conditions['Esq'][1])
            if elemt.nodes[0] > 1-eps: #(x=1)
                self.nds_right.append(elemt.nodes)
                self.vals_right.append(conditions['Dir'][1])
            if elemt.nodes[1] < eps: #(y=0)
                self.nds_bot.append(elemt.nodes)
                self.vals_bot.append(conditions['Inf'][1])
            if elemt.nodes[1] > 1-eps: #(y=1)
                self.nds_top.append(elemt.nodes)
                self.vals_top.append(conditions['Sup'][1])

        self.bcs_nodes = nodes
        self.bcs_values = values

    def solve(self):

        rows = []
        cols = []
        values = []

        for element in self.elements:
            r, c, v = element.Kmatrix(self.nodes)
            rows.append(r)
            cols.append(c)
            values.append(v)

        rows = np.array(rows, dtype='int').flatten()
        cols = np.array(cols, dtype='int').flatten()
        values = np.array(values, dtype='float').flatten()

        Kglobal = sparse.csr_matrix((values, (rows, cols)), shape=((self.nnodes, self.nnodes)))
        Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')

        self.Kglobal = Kglobal
        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(self.nnodes)

        # 2) Aplicação das BCs no vetor fglobal
        # 3) Aplicar as condições de contorno na matriz Kglobal
        w = 1e20
        nbcs = len(self.bcs_nodes)
        for k in range(nbcs):
            node = self.bcs_nodes[k]
            fglobal[node] = self.bcs_values[k] * w
            Kglobal[node, node] += w

        self.Kglobal = Kglobal
        self.fglobal = fglobal
        # 4) Resolver o problema
        self.T = spsolve(Kglobal, fglobal)

    def plot(self):
        plt.figure()
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.axis('off')
        plt.axis('equal')

        plt.figure()
        plt.tripcolor(self.nodes[:,0], self.nodes[:,1], self.connectivities, self.T, shading='gouraud')
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.colorbar()
        plt.axis('off')
        plt.axis('equal')
        plt.title('Temperature')

        plt.show()



class HT3():
    def __init__(self, nodes):
        self.enodes = nodes
        self.props = 0.0

    def Kmatrix(self, coords):
        x = coords[self.enodes, 0]
        y = coords[self.enodes, 1]

        B = np.zeros((2,3))
        B[0][0] = y[1] - y[2]
        B[0][1] = y[2] - y[0]
        B[0][2] = y[0] - y[1]

        B[1][0] = x[2] - x[1]
        B[1][1] = x[0] - x[2]
        B[1][2] = x[1] - x[0]

        A = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])

        B = (1.0/(2*A)) * B

        K = self.props * np.matmul(B.transpose(), B) * A

        self.area = A
        self.centroid = [np.mean(x), np.mean(y)]

        ind_rows = [self.enodes[0], self.enodes[0], self.enodes[0], self.enodes[1], self.enodes[1], self.enodes[2]]
        ind_cols = [self.enodes[0], self.enodes[1], self.enodes[2], self.enodes[1], self.enodes[2], self.enodes[2]]
        values =   [K[0,0], K[0,1], K[0,2], K[1,1], K[1,2], K[2,2]]

        return ind_rows, ind_cols, values


## Main Code ##

problem = fem2DHeaTransfer()

mesh = meshio.read('./util/L2/ex1_mesh1_tri.msh')
coords = np.array(mesh.points[:,0:2])
connectivities = mesh.cells[-1].data

# Implementação do elemento Q4
# connectivities = ([0, 1, 2, 3])

# Geometry and mesh
problem.createNodes(coords) # array com coords dos nos
problem.createElements(connectivities) # array dos elementos, em que cada elemento contem os ids dos seus nos, mas o objeto element nao contem a info das coords

# Nao foi definido o material, usarei o mesmo da Q8:
alpha = 1/(1.00*100) # k/rho*c
problem.defineProperties(alpha)

# Boundary conditions
T1 = 30.0
T2 = 10.0
q  = 0.00
# tipo 0: Dirichlet | 1: Neumann
bcs = {'Esq':(0,T1), 'Dir':(0,T2), 'Sup':(1,q), 'Inf':(1,q)}
problem.createBoundaryConds(bcs, [0, 2, 4], [100, 50, 75])

problem.solve()

problem.plot()