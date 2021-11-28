#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Transferencia de Calor em solidos (estacionario)

import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss

#plt.style.use('dark_background') # comentar essa linha para ter as figuras com fundo branco

nod_coords = {1:(-1,-1), 2:(1,-1), 3:(1,1), 4:(-1,1)}   # Xi-Eta table 4Q (var global)

def dForma_dxi(eta):
    dN_dxi  = np.zeros(4)
    for i in range(4):
        dN_dxi[i] = 0.25*(1 + nod_coords[i+1][1]*eta)*nod_coords[i+1][0]
    return dN_dxi

def dForma_deta(xi):
    dN_deta  = np.zeros(4)
    for i in range(4):
        dN_deta[i] = 0.25*(1 + nod_coords[i+1][0]*xi)*nod_coords[i+1][1]
    return dN_deta

def jac_de_xieta(xe, ye, e, n):
    J = np.zeros((2,2))
    dN_xi  = dForma_dxi(n)
    dN_eta = dForma_deta(e)
    J[0,0] = np.dot(dN_xi, xe)
    J[0,1] = np.dot(dN_xi, ye)
    J[1,0] = np.dot(dN_eta, xe)
    J[1,1] = np.dot(dN_eta, ye)
    return J

## FEM2DHT Class Definition

class fem2DHeaTransfer():
    def __init__(self, method):
        self.nnodes = 0
        self.nelements = 0
        self.method = method

    def createNodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)

    def createElements(self, connectivities):
        self.elements = []
        self.connectivities = connectivities
        for item in connectivities:
            if self.method == 'tri':
                element = HT3(item)
            if self.method == 'quad':
                element = HT4(item)
            self.elements.append(element)

        self.nelements = len(self.elements)

    def defineProperties(self, props):
        for element in self.elements:
            element.props = props

    def createBoundaryConds(self, conditions, des_nodes, des_values):
        self.bcs_nodes = [] # id/ tipo/ valor
        eps = 0.000001
        aux = True
        if aux:
            for i,nd in enumerate(self.nodes):
                if nd[0] < eps: #(x=0)
                    typ, val = conditions['E']
                    self.bcs_nodes.append((i, typ, val))
                if nd[0] > (1-eps): #(x=1)
                    typ, val = conditions['D']
                    self.bcs_nodes.append((i, typ, val))
                if nd[1] < eps: #(y=0)
                    typ, val = conditions['I']
                    self.bcs_nodes.append((i, typ, val))
                if nd[1] > (1-eps): #(y=1)
                    typ, val = conditions['S']
                    self.bcs_nodes.append((i, typ, val))
            # pontos nos vertices satisfazem as condicoes 2x
            # remocao por pop considera a mudanca de id da remocao anterior...
            self.bcs_nodes.pop(1)
            self.bcs_nodes.pop(2)
            self.bcs_nodes.pop(3)
            self.bcs_nodes.pop(4)
            self.bcs_nodes.append((4, 0, 10.0)) # bug?
        else: # opcao para usar o modo original de BCs
            for k,nd in enumerate(des_nodes):
                self.bcs_nodes.append((nd, 0, des_values[k]))

    def solve(self):

        rows = []
        cols = []
        values = []

        for element in self.elements:
            r, c, v = element.Ke(self.nodes)
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
        # usado para BCs de Dirichlet
        for nd in self.bcs_nodes:
            if nd[1] == 0:
                fglobal[nd[0]] = nd[2] * w
                Kglobal[nd[0], nd[0]] += w
        
        self.Kglobal = Kglobal
        self.fglobal = fglobal
        # 4) Resolver o problema
        self.T = spsolve(Kglobal, fglobal)

    def plot(self):
        if self.method == 'tri':
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
            plt.title('Temperatura')

            plt.show()
        if self.method == 'quad':
            # remover o ponto do meio antes...
            X = np.delete(self.nodes[:,0], 4)
            Y = np.delete(self.nodes[:,1], 4)
            Temp = np.delete(self.T, 4)
            axs = plt.figure().add_subplot(projection='3d')
            #axs.scatter(self.nodes[:,0], self.nodes[:,1], self.T, s=1.2, c=self.T)
            axs.plot_trisurf(X, Y, Temp, cmap=cm.viridis, linewidth=4)
            axs.set_xlabel('x')
            axs.set_ylabel('y')
            axs.set_zlabel('Temperatura')
            plt.show()



class HT3():
    def __init__(self, nodes):
        self.enodes = nodes
        self.props = 0.0

    def Ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        y = coords[self.enodes, 1]

        B = np.zeros((2,3))
        B[0][0] = y[1] - y[2]
        B[0][1] = y[2] - y[0]
        B[0][2] = y[0] - y[1]

        B[1][0] = x[2] - x[1]
        B[1][1] = x[0] - x[2]
        B[1][2] = x[1] - x[0]

        Ae = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])

        B = (1.0/(2*Ae)) * B # Be, pg 160, eq 7.20

        K = self.props * np.matmul(B.transpose(), B) * Ae

        # formulas no Jacob, pg 155 e 156, 160

        self.area = Ae
        self.centroid = [np.mean(x), np.mean(y)]

        ind_rows = [self.enodes[0], self.enodes[0], self.enodes[0], self.enodes[1], self.enodes[1], self.enodes[2]]
        ind_cols = [self.enodes[0], self.enodes[1], self.enodes[2], self.enodes[1], self.enodes[2], self.enodes[2]]
        values =   [K[0,0], K[0,1], K[0,2], K[1,1], K[1,2], K[2,2]]

        return ind_rows, ind_cols, values

class HT4():
    def __init__(self, nodes):
        self.enodes = nodes
        self.props = 0.0

    def Ke(self, coords): # constroi a matrix K do elemento
        x = np.array(coords[self.enodes, 0], dtype='float').flatten()
        y = np.array(coords[self.enodes, 1], dtype='float').flatten()

        Bint = np.zeros((4,4))
        # sample points and weights for Gauss-Legendre quadrature:
        Ngp = 4
        xi, Wx = leggauss(Ngp)
        eta, We = leggauss(Ngp)

        for i, q in enumerate(xi):
            for j, n in enumerate(eta):
                Je_inv = np.linalg.inv( jac_de_xieta(x, y, q, n) )
                G = np.array([dForma_dxi(n), dForma_deta(q)])
                B = np.matmul(Je_inv, G)
                Bint += Wx[i]*We[j]*(2)* np.matmul(B.transpose(), B)

        K = self.props * Bint

        # conferir essa conta da area
        A1 = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])
        A2 = 0.5*(x[0]*y[3] + y[0]*x[2] + x[3]*y[2] - x[2]*y[3] - x[0]*y[2] - x[3]*y[0])

        self.area = A1 + A2 # area dos dois triangulos definidos pelo quadrilatero
        self.centroid = [np.mean(x), np.mean(y)]
        
        ind_rows = [self.enodes[0], self.enodes[0], self.enodes[0], self.enodes[0], 
                    self.enodes[1], self.enodes[1], self.enodes[1], self.enodes[2], 
                    self.enodes[2], self.enodes[3]]

        ind_cols = [self.enodes[0], self.enodes[1], self.enodes[2], self.enodes[3],
                    self.enodes[1], self.enodes[2], self.enodes[3], self.enodes[2], 
                    self.enodes[3], self.enodes[3]]

        values =   [K[0,0], K[0,1], K[0,2], K[0,3], 
                    K[1,1], K[1,2], K[1,3], K[2,2],
                    K[2,3], K[3,3]]

        return ind_rows, ind_cols, values


## Main Code ##

met = 'quad'   # tri ou quad
problem = fem2DHeaTransfer(met)
if met == 'tri':
    mesh = meshio.read('./util/L2/ex1_mesh1_tri.msh')
if met == 'quad':
    mesh = meshio.read('./util/L2/ex1_mesh1_quad.msh')
coords = np.array(mesh.points[:,0:2])
connectivities = mesh.cells[-1].data

# Implementacao do elemento Q4
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
# tipo 0: Dirichlet | 1: Neumann (E: Esq, D: Dir, Sup: Superior, Inf: Inferior)
bcs = {'E':(0,T1), 'D':(0,T2), 'S':(1,q), 'I':(1,q)}
problem.createBoundaryConds(bcs, [0, 2, 4], [10, 5, 7.5])

problem.solve()

problem.plot()