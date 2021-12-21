#///////////////////////////////////////////////////////
#// Leonardo Felipe L. S. dos Santos, 2021           ///
#// leonardo.felipe.santos@usp.br	                 ///
#// github/bitbucket qleonardolp	                 ///
#///////////////////////////////////////////////////////

## Equilibrio de Solidos (trelicas)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse
from scipy.sparse.linalg import spsolve

#plt.style.use('dark_background') # comentar essa linha para ter as figuras com fundo branco

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

        self.U = np.zeros(self.nnodes*2)
        UredListed = list(self.Ured)

        for i in range(self.nnodes*2):
            k = -(i+1)  # indo de tras para frente
            # se nao for o item da ultima posicao dessa lista:
            if (2*self.nnodes+k) != insert_list_idx[-1]:
                self.U[k] = UredListed.pop()
            # se for o id = item da ultima posicao dessa lista,
            # remove o ultimo item da lista:
            else:
                insert_list_idx.pop()

        # 6) Obtendo forcas de reacao:
        self.Reacoes = []
        self.F = np.matmul(self.Kglobal.toarray(), self.U)
        for nd in self.bcs_nodes:
            if nd[1] == 1: # Forca aplicada
                self.Reacoes.append( -self.F[2*nd[0]] )
                self.Reacoes.append( -self.F[2*nd[0]+1] )
        print('Forças de Reação (N):')
        print(self.Reacoes) # rever sinal!

        # 7) Obtendo forcas normais em cada barra:
        ndXDeformado = self.nodes[:,0] + self.U[::2]
        ndYDeformado = self.nodes[:,1] + self.U[1::2]
        ndNormal = []
        for elmt in self.elements:
            n1 = elmt.enodes[0]
            n2 = elmt.enodes[1]
            Dx = ndXDeformado[n1] - ndXDeformado[n2]
            Dy = ndYDeformado[n1] - ndYDeformado[n2]
            Length = np.sqrt(Dx*Dx + Dy*Dy)
            Forca = elmt.area*elmt.young_modulus*(Length - elmt.comprimento)/elmt.comprimento
            ndNormal.append(Forca)
        self.Normais = ndNormal
        print('Forças normais nas barras (N):')
        print(ndNormal)


    #endmethod

    def plot(self):
        Uxy = self.U
        scale = 2e2
        normas = np.zeros((self.nnodes,1))
        u = Uxy[::2]
        v = Uxy[1::2]
        for i in range(self.nnodes):
            normas[i] = np.linalg.norm([u[i], v[i]])

        plt.figure()
        plt.triplot(self.nodes[:,0], self.nodes[:,1], '-b', linewidth=0.6)
        masks = [False, False, True, False, False, True, False, False, True] # para esconder triangulos indesejados
        plt.triplot(self.nodes[:,0]+scale*u, self.nodes[:,1]+scale*v,'-r', linewidth=0.6, mask=masks)
        plt.axis('equal')

        scale = 1e6
        Uxy = scale*Uxy
        #plt.figure()
        plt.quiver(self.nodes[:,0], self.nodes[:,1], Uxy[::2], Uxy[1::2], scale*normas, 
                   cmap=cm.winter, headwidth=2.0, headlength=3, headaxislength=3)
                  # use cm.winter para fundo branco
                  # use cm.spring para fundo preto
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(format='%.3f', label='Norma do vetor $u_{xy} [\mu$m]')
        plt.title('Treliça Plana Deformada (x200 $u_{xy}$)')
        plt.show(block=False)

        scale = 2e2
        plt.figure()
        for k, elmt in enumerate(self.elements):
            n1 = elmt.enodes[0]
            n2 = elmt.enodes[1]
            if self.Normais[k] > 0:
                plt.plot([self.nodes[n1,0]+scale*u[n1], self.nodes[n2,0]+scale*u[n2]] ,
                         [self.nodes[n1,1]+scale*v[n1], self.nodes[n2,1]+scale*v[n2]], '-g', linewidth=0.6)
            if self.Normais[k] < 0:
                plt.plot([self.nodes[n1,0]+scale*u[n1], self.nodes[n2,0]+scale*u[n2]] ,
                         [self.nodes[n1,1]+scale*v[n1], self.nodes[n2,1]+scale*v[n2]], '--r', linewidth=0.6)
        plt.title('Treliça Deformada (x200 $u_{xy}$), Compressão (--) / Tração (-)')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=True)


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
comprimento = 0.3 # m

off_x = comprimento/2
off_y = comprimento * np.sqrt(3)/2
coords = np.array([[0.0,0.0],[off_x,off_y],[0.3,0.0],
                  [off_x+0.3,off_y],[0.6,0.0],[off_x+0.6,off_y],
                  [0.9,0.0],[off_x+0.9,off_y],[1.2,0.0]])
connectivities = np.array([[0,1],[0,2],[1,2],[1,3],[2,3],
                           [2,4],[3,4],[3,5],[4,5],[4,6],
                           [5,6],[5,7],[6,7],[6,8],[7,8]])

angulos = np.array([1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2])*(np.pi/3)

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
# Lembrar que é zero-based: No 1 = 0, No 9 = 8 ...
bcs = {0:(0,1), 0:(0,2), 8:(0,2), 3:(1,(0,Fn4)), 5:(1,(0,Fn6))}
problem.createBoundaryConds(bcs)

problem.solve(angulos)

problem.plot()