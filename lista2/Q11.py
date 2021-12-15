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

## FEM Class Definition

class femLinearElasticity():
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
        self.bcs_nodes = [] # id/ tipo/ valor(res)
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
        self.bcs_nodes.append((4, 0, 0)) # bug?

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
            if nd[1] == 0: # Restricao geometrica (Dirichlet)
                deleteId.append(2*nd[0])
                deleteId_forward.append(2*nd[0]+1)
            if nd[1] == 1: # Neumann
                fglobal[2*nd[0]]   = nd[2][0] #(Fx)
                fglobal[2*nd[0]+1] = nd[2][1] #(Fy)

        self.Kglobal = Kglobal
        self.fglobal = fglobal

        self.fred = np.delete(fglobal, deleteId + deleteId_forward, 0)

        K = Kglobal.toarray()
        K = np.delete(K, deleteId + deleteId_forward, 0) # deleta linhas
        K = np.delete(K, deleteId + deleteId_forward, 1) # deleta colunas
        self.Kred = K

        # 4) Resolver o problema
        self.Ured = spsolve(sparse.csr_matrix(self.Kred), self.fred)

        # 5) Montando de volta o vetor global U:
        insert_list_idx = []
        for nd in self.bcs_nodes:
            if nd[1] == 0: # Restricao geometrica (Dirichlet)
                insert_list_idx.append(2*nd[0])
                insert_list_idx.append(2*nd[0]+1)
        insert_list_idx.sort()
        insert_list_idx = list(dict.fromkeys(insert_list_idx))  # remove duplicatas

        self.U = np.zeros(self.nnodes*2)
        k = 0
        for i in range(self.nnodes*2):
            if k < len(insert_list_idx):
                if i == insert_list_idx[k]:
                    k = k+1
            else:
                self.U[i] = self.Ured[i-k]
        #enfor

        # 6) Obtendo forcas de reacao:
        self.Reacoes = []
        self.F = np.matmul(self.Kglobal.toarray(), self.U)
        for nd in self.bcs_nodes:
            if nd[1] == 1: # Forca aplicada
                self.Reacoes.append( self.F[2*nd[0]] )
                self.Reacoes.append( self.F[2*nd[0]+1] )
        print('Forças de Reação:')
        print(self.Reacoes) # rever sinal!

    #endmethod


    def plot(self):
        Uxy = self.U
        scale = 1e6
        normas = np.zeros((self.nnodes,1))
        u = Uxy[::2]
        v = Uxy[1::2]
        for i in range(self.nnodes):
            normas[i] = np.linalg.norm([u[i], v[i]])
        
        #Malha original:
        plt.figure()
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        #Malha deformada (escala de cor: norma de u_xy):
        #plt.tripcolor(self.nodes[:,0]+scale*u, self.nodes[:,1]+scale*v, self.connectivities, 1e6*normas, shading='gouraud')
        plt.triplot(self.nodes[:,0]+scale*u, self.nodes[:,1]+scale*v, self.connectivities, '-r', linewidth=0.5)
        #plt.colorbar(format='%.3f', label='Norma do vetor $u_{xy} [\mu$m]')
        plt.title('Chapa deformada em escala de cor')
        plt.axis('equal')
        plt.show(block=False)


        #Campo de deformacoes:
        scale = 1e6
        Uxy = scale*Uxy
        plt.figure()
        plt.quiver(self.nodes[:,0], self.nodes[:,1], Uxy[::2], Uxy[1::2], scale*normas, 
                   cmap=cm.spring, headwidth=2.0, headlength=3, headaxislength=3)
                  # use cm.winter para fundo branco
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(format='%.3f', label='Norma de $u_{xy} [\mu$m]')
        plt.title('Campo de Deformações $u_{xy} [\mu$m]')
        plt.show(block=True)



class HT3():
    def __init__(self, nodes):
        self.enodes = nodes
        self.young_modulus = 0.0
        self.poisson_coeff = 0.0

    def Ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        y = coords[self.enodes, 1]
        
        A = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])     #OK
        
        D = np.zeros((3,3))
        v = self.poisson_coeff
        E = self.young_modulus
        D[0,0] = 1
        D[0,1] = v
        D[1,0] = v
        D[1,1] = 1
        D[2,2] = (1 - v)/2

        D = D*(E/(1 - v*v))     #OK

        # no caso triangular o produto das matrizes dentro da integral
        # nao depende de x e y.
        B = np.zeros((3,6))
        B[0,0] = y[1] - y[2]     #OK
        B[0,2] = y[2] - y[0]     #OK
        B[0,4] = y[0] - y[1]     #OK

        B[1,1] = x[2] - x[1]     #OK
        B[1,3] = x[0] - x[2]     #OK
        B[1,5] = x[1] - x[0]     #OK

        B[2,0] = x[2] - x[1]     #OK
        B[2,1] = y[1] - y[2]     #OK
        B[2,2] = x[0] - x[2]     #OK
        B[2,3] = y[2] - y[0]     #OK
        B[2,4] = x[1] - x[0]     #OK
        B[2,5] = y[0] - y[1]     #OK

        # formulas no Jacob, pg 155 e 156, 160
        B = (1.0/(2*A)) * B # Be, pg 160, eq 7.20

        K = np.matmul(np.matmul(B.transpose(), D), B) * A     #OK
        # K é simetrica 6x6

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

        ind_rows = 6*[n1] + 5*[n1 + 1] + 4*[n2] + 3*[n2 + 1] + 2*[n3] + 1*[n3 + 1]  #OK
        ind_cols = [n1, n1+1, n2, n2+1, n3, n3+1, 
                        n1+1, n2, n2+1, n3, n3+1,
                              n2, n2+1, n3, n3+1,
                                  n2+1, n3, n3+1,
                                        n3, n3+1,
                                            n3+1]   #OK
        values =   [K[0,0], K[0,1], K[0,2], K[0,3], K[0,4], K[0,5],
                            K[1,1], K[1,2], K[1,3], K[1,4], K[1,5],
                                    K[2,2], K[2,3], K[2,4], K[2,5],
                                            K[3,3], K[3,4], K[3,5],
                                                    K[4,4], K[4,5],
                                                            K[5,5]] #OK

        return ind_rows, ind_cols, values

## Main Code ##

problem = femLinearElasticity()
mesh = meshio.read('./meshes/ex1_mesh1_tri.msh')
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
# Forca aplicada em x,y = 1,1 => Nó 2
problem.createBoundaryConds(bcs, [2], [(Fx, Fy)])

problem.solve()

problem.plot()