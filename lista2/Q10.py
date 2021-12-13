#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Equilibrio de Solidos (trelicas)

import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse
from scipy.sparse.linalg import spsolve

plt.style.use('dark_background') # comentar essa linha para ter as figuras com fundo branco

## FEM Class Definition

class femTrelica():
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
            element = Trelica(item)
            self.elements.append(element)

        self.nelements = len(self.elements)

    def defineProperties(self, Young, Area, Comp):
        for element in self.elements:
            element.young_modulus = Young
            element.comprimento = Comp
            element.area = Area

    def createBoundaryConds(self, bcs):
        # Lembrando:
        # bcs = {1:(0,1), 1:(0,2), 9:(0,2), 4:(1,(0,Fn4)), 6:(1,(0,Fn6))}
        Nos = bcs.keys()
        self.bcs_nodes = [] # id/ tipo/ valor(res)
        for nd in Nos:
            type, val = bcs[nd]
            self.bcs_nodes.append((nd, type, val))


    def solve(self, angulos_Ke):

        rows = []
        cols = []
        values = []

        for k, element in enumerate(self.elements):
            r, c, v = element.Ke(angulos_Ke[k])
            rows.append(r)
            cols.append(c)
            values.append(v)

        rows = np.array(rows, dtype='int').flatten()
        cols = np.array(cols, dtype='int').flatten()
        values = np.array(values, dtype='float').flatten()

        Kglobal = sparse.csr_matrix((values, (rows, cols)), shape=((2*self.nnodes, 2*self.nnodes)))
        Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')

        self.Kglobal = Kglobal
        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(self.nnodes*2)

        deleteId = []
        deleteId_forward = []
        for nd in self.bcs_nodes:
            if nd[1] == 0: # Restricao geometrica (Dirichlet)
                deleteId.append(2*nd[0])
                deleteId_forward.append(2*nd[0]+1)
            if nd[1] == 1: # Neumann
                fglobal[2*nd[0]]   = nd[2][0] #(Fx)
                fglobal[2*nd[0]+1] = nd[2][1] #(Fy)

        self.f = np.delete(fglobal, deleteId + deleteId_forward, 0)

        K = Kglobal.toarray()
        K = np.delete(K, deleteId + deleteId_forward, 0) # deleta linhas
        K = np.delete(K, deleteId + deleteId_forward, 1) # deleta colunas
        self.K = K

        self.Ured = spsolve(sparse.csr_matrix(self.K), self.f)

        # 2) Aplicação das BCs no vetor fglobal
        # 3) Aplicar as condições de contorno na matriz Kglobal
        
        self.Kglobal = Kglobal
        self.fglobal = fglobal
        # 4) Resolver o problema
        self.U = spsolve(Kglobal, fglobal)


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
        plt.title('Temperatura')
        x = np.zeros((self.nelements,1), dtype='float')
        y = x.copy()
        u = x.copy()
        v = x.copy()
        L = x.copy()
        for k,elmt in enumerate(self.elements):
            x[k], y[k] = elmt.centroid[0], elmt.centroid[1]
        
        for k,q_vec in enumerate(self.flux):
            u[k], v[k] = q_vec[0], q_vec[1]
            L[k] = np.linalg.norm([u[k], v[k]])
    
        plt.figure()
        plt.quiver(x, y, u, v, L, pivot='mid', cmap=cm.cool, 
                    headwidth=4.0, headlength=4, headaxislength=4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.colorbar(format='%.3f')
        plt.title('Fluxo de Calor')    
        plt.show()



class Trelica():
    def __init__(self, nodes):
        self.enodes = nodes
        self.young_modulus = 0.0
        self.comprimento = 0.0
        self.area = 0.0

    def Ke(self, ang): # constroi a matrix K do elemento
    
        K = np.zeros((4,4))

        c0 = np.cos(ang)
        s0 = np.sin(ang)
        K[0,0] =  c0*c0
        K[0,1] =  c0*s0
        K[0,2] = -c0*c0
        K[0,3] = -c0*s0
        K[1,1] =  s0*s0
        K[1,2] = -c0*s0
        K[1,3] = -s0*s0        
        K[2,2] =  c0*c0 
        K[2,3] =  c0*s0
        K[3,3] =  s0*s0

        K = K*(self.area*self.young_modulus)/self.comprimento

        n1 = 2*self.enodes[0]
        n2 = 2*self.enodes[1]
        # K é simetrica

        #  u1  v1        u2  v2     
        # n11 n12   ... n13 n14  u1
        #     n22       n23 n24  v1
        #     ...
        #               n33 n34  u2
        #                   n44  v2

        ind_rows = 4*[n1] + 3*[n1 + 1] + 2*[n2] + 1*[n2 + 1]
        ind_cols = [n1, n1+1, n2, n2+1, 
                        n1+1, n2, n2+1,
                              n2, n2+1,
                                  n2+1]
        values =   [K[0,0], K[0,1], K[0,2], K[0,3],
                            K[1,1], K[1,2], K[1,3],
                                    K[2,2], K[2,3],
                                            K[3,3]]

        return ind_rows, ind_cols, values

## Main Code ##

problem = femTrelica()

coords = np.arange(9)
connectivities = np.array([[0,1],[0,2],[1,2],[1,3],[2,3],
                           [2,4],[3,4],[3,5],[4,5],[4,6],
                           [5,6],[5,7],[6,7],[6,8],[7,8]])

angulos = np.array([1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2])*(np.pi/3)
comprimento = 0.3 # m

problem.createNodes(coords) # array com coords dos nos
problem.createElements(connectivities) # array dos elementos, 
# em que cada elemento contem os ids dos seus nos, 
# mas o objeto element nao contem a info das coords

# Material isotropico:
E = 72e9 # [Pa] (72 GPa)
D = 0.030 # [m]
Area = np.pi * D*D/4
problem.defineProperties(E, Area, comprimento)

Fn4 = -4000 #[N]
Fn6 = -2000 #[N]
# Condicoes de contorno: "No:(tipo,restricao)"
# tipo 0: Dirichlet | 1: Neumann | -1: Invalida
# restricao geometrica: 1: (dx = 0), 2: (dy = 0)

bcs = {0:(0,1), 0:(0,2), 8:(0,2), 4:(1,(0,Fn4)), 6:(1,(0,Fn6))}
problem.createBoundaryConds(bcs)

problem.solve(angulos)

problem.plot()