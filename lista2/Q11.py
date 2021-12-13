#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Equilibrio de Solidos (chapas)

import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

    def defineProperties(self, Young, Poisson):
        for element in self.elements:
            element.young_modulus = Young
            element.poisson_coeff = Poisson

    def createBoundaryConds(self, boundaries, bcNodes, bcValues):
        self.bcs_nodes = [] # id/ tipo/ valor
        eps = 0.000001
        for i,nd in enumerate(self.nodes):
            #(x=0)
            if (nd[0] < eps) and (boundaries['E'][0] != -1):
                typ, val = boundaries['E']
                self.bcs_nodes.append((i, typ, val))
            #(x=1)
            if (nd[0] > (1-eps)) and (boundaries['D'][0] != -1):
                typ, val = boundaries['D']
                self.bcs_nodes.append((i, typ, val))
            #(y=0)
            if (nd[1] < eps) and (boundaries['I'][0] != -1):
                typ, val = boundaries['I']
                self.bcs_nodes.append((i, typ, val))
            #(y=1)
            if (nd[1] > (1-eps)) and (boundaries['S'][0] != -1):
                typ, val = boundaries['S']
                self.bcs_nodes.append((i, typ, val))

        # pontos nos vertices satisfazem as condicoes 2x
        #self.bcs_nodes.append((4, 0, 1e-9)) # bug?

        # Neumann conditions by nodes:
        for k, nd in enumerate(bcNodes):
            self.bcs_nodes.append((nd, 1, bcValues[k]))


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

        Kglobal = sparse.csr_matrix((values, (rows, cols)), shape=((2*self.nnodes, 2*self.nnodes)))
        Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')

        self.Kglobal = Kglobal
        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(self.nnodes*2)

        deleteId = []
        deleteId_forward = []
        for nd in self.bcs_nodes:
            if nd[1] == 0:
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
        w = 1e20
        # Forcando BCs:
        for nd in self.bcs_nodes:
            if nd[1] == 0: # Dirichilet
                if nd[2] == 1: # em u
                    i = 2*nd[0]
                    np.delete(fglobal,i)
                    sparse.vstack([Kglobal[:i, :], Kglobal[i:, :]])
                    sparse.hstack([Kglobal[:, :i], Kglobal[:, i:]])
                if nd[2] == 2: # em v
                    i = 2*nd[0] + 1
                    np.delete(fglobal,i)
                    sparse.vstack([Kglobal[:i, :], Kglobal[i:, :]])
                    sparse.hstack([Kglobal[:, :i], Kglobal[:, i:]])
            
            if nd[1] == 1: # Neumann
                fglobal[2*nd[0]]   = nd[2][0] #(Fx)
                fglobal[2*nd[0]+1] = nd[2][1] #(Fy)
        
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



class HT3():
    def __init__(self, nodes):
        self.enodes = nodes
        self.young_modulus = 0.0
        self.poisson_coeff = 0.0

    def Ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        y = coords[self.enodes, 1]
        
        A = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])
        
        D = np.zeros((3,3))
        v = self.poisson_coeff
        E = self.young_modulus
        D[0,0] = 1
        D[0,1] = v
        D[1,0] = v
        D[1,1] = 1
        D[2,2] = (1 - v)/2

        D = D*(E/(1 - v*v))

        # no caso triangular o produto das matrizes dentro da integral
        # nao depende de x e y.
        B = np.zeros((3,6))
        B[0,0] = y[1] - y[2]
        B[0,2] = y[2] - y[0]
        B[0,4] = y[0] - y[1]

        B[1,1] = x[2] - x[1]
        B[1,3] = x[0] - x[2]
        B[1,5] = x[1] - x[0]

        B[2,0] = x[2] - x[1]
        B[2,1] = y[1] - y[2]
        B[2,2] = x[0] - x[2]
        B[2,3] = y[2] - y[0]
        B[2,4] = x[1] - x[0]
        B[2,5] = y[0] - y[1]


        B = (1.0/(2*A)) * B # Be, pg 160, eq 7.20

        K = np.matmul(np.matmul(B.transpose(), D), B) * A

        # K é simetrica 6x6

        # formulas no Jacob, pg 155 e 156, 160

        self.area = A
        self.b_matrix = B
        self.centroid = np.array([np.mean(x), np.mean(y)])

        n1 = 2*self.enodes[0]
        n2 = 2*self.enodes[1]
        n3 = 2*self.enodes[2]
        #  u1  v1        u2  v2      u3  v3     
        # n11 n12       n13 n14     n15 n16  u1
        #     n22       n23 n24     n25 n26  v1
        #                  ...  
        #               n33 n34     n35 n36  u2
        #                   n44     n45 n46  v2
        #                  ...  
        #                           n55 n56  u3
        #                               n66  v3

        ind_rows = 6*[n1] + 5*[n1 + 1] + 4*[n2] + 3*[n2 + 1] + 2*[n3] + 1*[n3 + 1]
        ind_cols = [n1, n1+1, n2, n2+1, n3, n3+1, 
                        n1+1, n2, n2+1, n3, n3+1,
                              n2, n2+1, n3, n3+1,
                                  n2+1, n3, n3+1,
                                        n3, n3+1,
                                            n3+1]
        values =   [K[0,0], K[0,1], K[0,2], K[0,3], K[0,4], K[0,5],
                            K[1,1], K[1,2], K[1,3], K[1,4], K[1,5],
                                    K[2,2], K[2,3], K[2,4], K[2,5],
                                            K[3,3], K[3,4], K[3,5],
                                                    K[4,4], K[4,5],
                                                            K[5,5]]

        return ind_rows, ind_cols, values

## Main Code ##

problem = fem2DHeaTransfer()
mesh = meshio.read('./util/L2/ex1_mesh1_tri.msh')
coords = np.array(mesh.points[:,0:2])
connectivities = mesh.cells[-1].data

problem.createNodes(coords) # array com coords dos nos
problem.createElements(connectivities) # array dos elementos, 
# em que cada elemento contem os ids dos seus nos, 
# mas o objeto element nao contem a info das coords

# Material isotropico:
E = 210e9 # Pa (210 GPa)
Nu = 0.3
problem.defineProperties(E, Nu)

Fx = 2000 #[N]
Fy = 4000 #[N]
# Condicoes de contorno: (tipo,restricao)
# tipo 0: Dirichlet | 1: Neumann | -1: Invalida
# restricao geometrica: 1: (dx = 0), 2: (dy = 0)
# E: Esq, D: Dir, Sup: Superior, Inf: Inferior
bcs = {'E':(0,1), 'D':(-1,0), 'S':(-1,0), 'I':(0,2)}
# Forca aplicada em x,y = 1,1 
problem.createBoundaryConds(bcs, [2], [(Fx, Fy)])

problem.solve()

problem.plot()