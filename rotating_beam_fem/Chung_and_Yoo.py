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
from numpy.lib.function_base import append
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
        self.delta_t = end_time/time_steps
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

    def createBoundaryConds(self, Omg, dOmg, Fx, Fy, Fz):
        # Series temporais de Omega, Fx, Fy, Fz
        self.Omega = Omg
        self.dOmega = dOmg
        self.ps = Fx
        self.pv = Fy
        self.pw = Fz
        # Rever BCs de Dirichlet e Neumann...

    def solveChordwise(self):

        rows = []
        cols = []
        val_m = []
        val_k = []
        val_g = []
        val_s = []
        rows_f = []
        val_f0s = []
        val_f1s = []
        val_f0v = []
        val_f1v = []


        for element in self.elements:
            r, c, v = element.me(self.nodes)
            rows.append(r)
            cols.append(c)
            val_m.append(v)
            val_k.append( element.ke(self.nodes) )
            val_g.append( element.ge(self.nodes) )
            val_s.append( element.se(self.nodes) )
            r, v = element.fe0s(self.nodes)
            rows_f.append(r)
            val_f0s.append(v)
            val_f1s.append(element.fe1s(self.nodes))
            val_f0v.append(element.fe0v(self.nodes))
            val_f1v.append(element.fe1v(self.nodes))

        rows = np.array(rows, dtype='int').flatten()
        cols = np.array(cols, dtype='int').flatten()
        # Montagem da matriz global M
        val_m = np.array(val_m, dtype='float').flatten()
        M_g = sparse.csr_matrix((val_m, (rows, cols)), shape=((self.nnodes*3, self.nnodes*3)))
        M_g = M_g + M_g.T - sparse.diags(M_g.diagonal(), dtype='float')
        # Montagem da matriz global K
        val_k = np.array(val_k, dtype='float').flatten()
        K_g = sparse.csr_matrix((val_k, (rows, cols)), shape=((self.nnodes*3, self.nnodes*3)))
        K_g = K_g + K_g.T - sparse.diags(K_g.diagonal(), dtype='float')
        # Montagem da matriz global G
        val_g = np.array(val_g, dtype='float').flatten()
        G_g = sparse.csr_matrix((val_g, (rows, cols)), shape=((self.nnodes*3, self.nnodes*3)))
        G_g = G_g - G_g.T + sparse.diags(G_g.diagonal(), dtype='float')
        # Montagem da matriz global S
        val_s = np.array(val_s, dtype='float').flatten()
        S_g = sparse.csr_matrix((val_s, (rows, cols)), shape=((self.nnodes*3, self.nnodes*3)))
        S_g = S_g + S_g.T - sparse.diags(S_g.diagonal(), dtype='float')

        # Montagem dos vetores fglobal
        rows_f = np.array(rows_f, dtype='int').flatten()
        val_f0s = np.array(val_f0s, dtype='float').flatten()
        fg_0s = sparse.csr_matrix((val_f0s, (rows_f, np.zeros(np.shape(rows_f)) )), shape=((self.nnodes*3, 1)))
        val_f1s = np.array(val_f1s, dtype='float').flatten()
        fg_1s = sparse.csr_matrix((val_f1s, (rows_f, np.zeros(np.shape(rows_f)) )), shape=((self.nnodes*3, 1)))
        val_f0v = np.array(val_f0v, dtype='float').flatten()
        fg_0v = sparse.csr_matrix((val_f0v, (rows_f, np.zeros(np.shape(rows_f)) )), shape=((self.nnodes*3, 1)))
        val_f1v = np.array(val_f1v, dtype='float').flatten()
        fg_1v = sparse.csr_matrix((val_f1v, (rows_f, np.zeros(np.shape(rows_f)) )), shape=((self.nnodes*3, 1)))

        self.M_g = M_g
        self.K_g = K_g
        self.G_g = G_g
        self.S_g = S_g

        d = np.zeros((self.nnodes*3, self.num_steps)) # deformacoes
        v = np.zeros((self.nnodes*3, 2))              # d/dt deformacoes
        a = np.zeros((self.nnodes*3, 2))              # d2/dt2 deformacoes

        M_inv = sprlinalg.inv(M_g.tocsr()) # M^-1, pre-calculada pois é constante no tempo
        M_inv = M_inv.toarray()
        SminusM = S_g - M_g
        dt = self.delta_t  # Delta t
        af = self.alp_f
        am = self.alp_m
        bet = self.beta
        gam = self.gamma
        Ps = self.ps
        Pv = self.pv
        Omg = self.Omega
        dotOmg = self.dOmega
        rhoA  = self.properties['Area']*self.properties['Density']
        of = self.properties['a']
        
        for k, t in enumerate (self.time[:-1]):
            omgSq_k = Omg[k]*Omg[k]
            # Applying eq.(34) and (37)
            if k == 0:
                # ajustando o vetor fg de acordo com [ps pv pw]:
                f = (Ps[k] + rhoA*omgSq_k*of)*fg_0s + (rhoA*omgSq_k)*fg_1s
                f = f + (Pv[k] - rhoA*dotOmg[k]*of)*fg_0v - (rhoA*dotOmg[k])*fg_1v
                C = 2*Omg[k]*G_g
                K = K_g + omgSq_k*SminusM + dotOmg[k]*G_g
                
                V =   -np.matmul(K.toarray(), d[:,k]).reshape((self.nnodes*3, 1))
                V = V -np.matmul(C.toarray(), v[:,0]).reshape((self.nnodes*3, 1))
                V = V + f.toarray()
                a[:,0] = np.matmul(M_inv, V).reshape((self.nnodes*3, ))        # a0
                a[:,1] = a[:,0]
            else:
                omgSq_k_1 = Omg[k-1]*Omg[k-1]
                # ajustando o vetor fg de acordo com [ps pv pw]:
                fk = (Ps[k] + rhoA*omgSq_k*of)*fg_0s + (rhoA*omgSq_k)*fg_1s
                fk = fk + (Pv[k] - rhoA*dotOmg[k]*of)*fg_0v - (rhoA*dotOmg[k])*fg_1v

                fk1 = (Ps[k-1] + rhoA*omgSq_k_1*of)*fg_0s + (rhoA*omgSq_k_1)*fg_1s
                fk1 = fk1 + (Pv[k-1] - rhoA*dotOmg[k-1]*of)*fg_0v - (rhoA*dotOmg[k-1])*fg_1v
                f = (1 - af)*fk + af*fk1

                C = 2*Omg[k]*G_g
                K = K_g + omgSq_k*SminusM + dotOmg[k]*G_g

                # The Generalized \alpha Algorithm ( Ref. 2 eq.(4)-(13) )
                d[:,k] = d[:,k-1] + dt*v[:,0] + dt*dt*((0.5 - bet)*a[:,0] + bet*a[:,1])
                v[:,1] = v[:,0] + dt*((1 - gam)*a[:,0] + gam*a[:,1])
                d_af = (1 - af)*d[:,k] + af*d[:,k-1]
                v_af = (1 - af)*v[:,1] + af*v[:,0]
                # k <- k+1:
                a[:,0] = a[:,1]
                v[:,0] = v[:,1]

                #V = f - C*v_af - K*d_af
                V = np.matmul(K.toarray(), d_af).reshape((self.nnodes*3, 1))
                V = V + np.matmul(C.toarray(), v_af).reshape((self.nnodes*3, 1))
                V = f.toarray() - V
                a_am = np.matmul(M_inv, V).reshape((self.nnodes*3, )) 
                a[:,1] = (a_am - am*a[:,0])/(1 - am)

        self.d = d

    def solveFlapwise(self):
        return False

    def plot(self):
        return False


class RB2():
    def __init__(self, nodes, props):
        #{'Area':crossArea, 'Young':E, 'MoIz':Izz, 'MoIy':Iyy, 'Density':density, 'L':beamLength, 'a':offset}
        self.enodes = nodes
        self.L   = props['L']
        self.Iz  = props['MoIz']
        self.Iy  = props['MoIy']
        self.offset = props['a']
        self.E   = props['Young']
        self.A   = props['Area']
        self.rhoA   = props['Area']*props['Density']

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

        M = self.rhoA*M

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
                                            n2+2]
        values =   [M[0,0], M[0,1], M[0,2], M[0,3], M[0,4], M[0,5],
                            M[1,1], M[1,2], M[1,3], M[1,4], M[1,5],
                                    M[2,2], M[2,3], M[2,4], M[2,5],
                                            M[3,3], M[3,4], M[3,5],
                                                    M[4,4], M[4,5],
                                                            M[5,5]] #OK

        M = M + M.T - np.diag(M.diagonal())
        self.m_mtx = M
        self.h = h

        return row_id, col_id, values

    def ke(self, coords): # constroi a matrix K do elemento
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        Ks = np.zeros((6,6))
        Kv = np.zeros((6,6))
        # Ref. 1, eq.(34)

        Ks[0,0] =  1/h
        Ks[0,3] = -Ks[0,0]
        Ks[3,0] = -Ks[0,0]
        Ks[3,3] =  Ks[0,0]

        Kv[1,1] = 12/(h*h*h)
        Kv[1,2] = 6/(h*h)
        Kv[1,4] = -12/(h*h*h)
        Kv[1,5] = 6/(h*h)

        Kv[2,2] = 4/(h)
        Kv[2,4] = -6/(h*h)
        Kv[2,5] = 2/(h)

        Kv[4,4] = 12/(h*h*h)
        Kv[4,5] = -6/(h*h)

        Kv[5,5] = 4/(h)

        K = (self.E*self.A/self.L)*Ks + (self.E*self.Iz/self.L)*Kv

        values =   [K[0,0], K[0,1], K[0,2], K[0,3], K[0,4], K[0,5],
                            K[1,1], K[1,2], K[1,3], K[1,4], K[1,5],
                                    K[2,2], K[2,3], K[2,4], K[2,5],
                                            K[3,3], K[3,4], K[3,5],
                                                    K[4,4], K[4,5],
                                                            K[5,5]]

        K = K + K.T - np.diag(K.diagonal())
        self.k_mtx = K
        return values
    
    def ge(self, coords): # constroi a matrix G do elemento
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        G = np.zeros((6,6))
        # Ref. 1, eq.(34)

        G[0,1] = -7*h/20
        G[0,2] = -(h*h)/20
        G[0,4] = -3*h/20
        G[0,5] =  h*h/30

        G[1,3] =  3*h/20

        G[2,3] =  h*h/30

        G[3,4] = -7*h/20
        G[3,5] =  h*h/20

        G = self.rhoA*G

        values =   [G[0,0], G[0,1], G[0,2], G[0,3], G[0,4], G[0,5],
                            G[1,1], G[1,2], G[1,3], G[1,4], G[1,5],
                                    G[2,2], G[2,3], G[2,4], G[2,5],
                                            G[3,3], G[3,4], G[3,5],
                                                    G[4,4], G[4,5],
                                                            G[5,5]] #OK

        G = G - G.T + np.diag(G.diagonal()) # skew-symmetric
        self.g_mtx = G
        return values
    
    def se(self, coords): # constroi a matrix S do elemento
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        a =  x.min()
        S0 = np.zeros((6,6))
        S1 = np.zeros((6,6))
        S2 = np.zeros((6,6))
        # Ref. 1, eq.(34)
        # Zero Order Mtx
        S0[1,1] = 6/(5*h)
        S0[1,2] = 1/10
        S0[1,4] = -6/(5*h)
        S0[1,5] = 1/10
        S0[2,2] = 2*h/15
        S0[2,4] = -1/10
        S0[2,5] = -h/30
        S0[4,4] = 6/(5*h)
        S0[4,5] = -1/10
        S0[5,5] = 2*h/15

        # First Order Mtx
        S1[1,1] = 6*a/(5*h) + 3/5
        S1[1,2] = a/10 + h/10
        S1[1,4] = -6*a/(5*h) - 3/5
        S1[1,5] = a/10
        S1[2,2] = (h*(4*a + h))/30
        S1[2,4] = -a/10 -h/10
        S1[2,5] = -(h*(2*a + h))/60
        S1[4,4] = (6*a)/(5*h) + 3/5
        S1[4,5] = -a/10
        S1[5,5] = (h*(4*a + 3*h))/30

        # Second Order Mtx
        S2[1,1] = (6*a*a)/(5*h) + (6*a)/5 + (12*h)/35
        S2[1,2] = a*a/10 + (a*h)/5 + h*h/14
        S2[1,4] = -(6*a)/5 -(12*h)/35 -(6*a*a)/(5*h)
        S2[1,5] = a*a/10 - h*h/35
        S2[2,2] = (h*(14*a*a + 7*a*h + 2*h*h))/105
        S2[2,4] = -a*a/10 -(a*h)/5 -h*h/14
        S2[2,5] = -(h*(7*a*a + 7*a*h + 3*h*h))/210
        S2[4,4] = (6*a*a)/(5*h) + (6*a)/5 + (12*h)/35
        S2[4,5] = h*h/35 - a*a/10
        S2[5,5] = (h*(14*a*a + 21*a*h + 9*h*h))/105


        S = (self.offset + 0.5*self.L)*self.L*S0 - (self.offset)*S1 - (0.5)*S2
        S = self.rhoA*S

        values =   [S[0,0], S[0,1], S[0,2], S[0,3], S[0,4], S[0,5],
                            S[1,1], S[1,2], S[1,3], S[1,4], S[1,5],
                                    S[2,2], S[2,3], S[2,4], S[2,5],
                                            S[3,3], S[3,4], S[3,5],
                                                    S[4,4], S[4,5],
                                                            S[5,5]]

        S = S + S.T - np.diag(S.diagonal())
        self.s_mtx = S
        return values

    def fe0s(self, coords): # 0-ord Ns fe vector
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        v = [h/2, 0, 0, h/2, 0, 0]
        n1 = 3*self.enodes[0]
        n2 = 3*self.enodes[1]
        rows = [n1] + [n1+1] + [n1+2] + [n2] + [n2+1] + [n2 + 2] 
        return rows, v

    def fe1s(self, coords): # 1-ord Ns fe vector
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        a =  x.min()
        v = [(h*(3*a + h))/6, 0, 0, (h*(3*a + 2*h))/6, 0, 0]
        return v

    def fe0v(self, coords): # 0-ord Nv fe vector
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        v = [0, h/2, h*h/12, 0, h/2, -h*h/12]
        return v

    def fe1v(self, coords): # 1-ord Nv fe vector
        x = coords[self.enodes, 0]
        h = abs(x[1] - x[0]) # element size
        a =  x.min()
        v = [0, (h*(10*a + 3*h))/20, (h*h*(5*a + 2*h))/60, 
             0, (h*(10*a + 7*h))/20, -(h*h*(5*a + 3*h))/60]
        return v


## Main Code ##

beamLength = 0.30 #[m]
t_end = 10.00     #[s]
time_steps = 200
nnodes = 16
offset = beamLength*5/100
Radius = 2*beamLength/70 # Euler-Bernoulli beam theory constrain with alpha = 70, see Ref 1 eq (43)
crossArea = np.pi*Radius*Radius
Izz = 0.25* np.pi*pow(Radius, 4)
Iyy = 0.25* np.pi*pow(Radius, 4)
# ABS Properties:
E = 27e6        # [Pa] (27 MPa)
density = 1.2e3 # [Kg/m^3]

spec_rds = 0.2 #Spectral Radius (rho_inf) [Ref 2]
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
dotOmg = 1.0*0.5*np.cos(0.5*t)
Fx = 0*t
Fy = 0*t
Fz = 0*t
problem.createBoundaryConds(Omg, dotOmg, Fx, Fy, Fz)

problem.solveChordwise()
problem.solveFlapwise()

problem.plot()