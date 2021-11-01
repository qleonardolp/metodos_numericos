import numpy as np
import matplotlib.pyplot as plt

# abstracao em OO para os nós e elementos...
# encapsulamento pdoe ser feito com "decorators"

class Node:
    #construction method
    def __init__(self, coords, label):
        self.coords = coords
        self.label = int(label)
    # veja que é possível definir um membro da classe a partir da funcao construtora

class Element:
    #construction method
    def __init__(self, nos_locais, props, material, nos_globais):
        self.nodes = []
        for nd in nos_locais:
            self.nodes.append(nos_globais[nd])

        # usando props/material como um dicionário:
        self.material = material
        self.heat_gen = props['s']
        self.area = props['A']

        # usando o 1° e último provavelmente para ter 
        # o maior comprimento do elemento
        coords0 = self.nodes[0].coords
        coords1 = self.nodes[-1].coords
        self.length = np.linalg.norm(np.array(coords1) 
                                    -np.array(coords0))

    # Element Matrix
    def create_element_matrix(self):
        num_nd = len(self.nodes)
        
        if (num_nd == 2):
            n1 = self.nodes[0].label
            n2 = self.nodes[1].label
            idx_row = [n1, n1, n2, n2]
            idx_col = [n1, n2, n1, n2]
            k = self.material['thermal_cond']
            A = self.area
            h = self.length
            elem_mtx = np.array([[1.0, -1.0], [-1.0, 1.0]]) * (h*A/L)

        if (num_nd == 3):
            n1 = self.nodes[0].label
            n2 = self.nodes[1].label
            n3 = self.nodes[2].label
            idx_row = [n1, n1, n1, n2, n2, n2, n3, n3, n3]
            idx_col = [n1, n2, n3, n1, n2, n3, n1, n2, n3]
            k = self.material['thermal_cond']
            A = self.area
            h = self.length
            elem_mtx = np.array([[7*A*k/(3*h), -8*A*k/(3*h), A*k/(3*h)], 
                              [-8*A*k/(3*h), 16*A*k/(3*h), -8*A*k/(3*h)], 
                              [A*k/(3*h), -8*A*k/(3*h), 7*A*k/(3*h)]])

        self.ke = elem_mtx
        return elem_mtx

    # Element 'Force' vector
    def create_element_vector(self):
        num_nd = len(self.nodes)
        if (num_nd == 2):
            n1 = self.nodes[0].label
            n2 = self.nodes[1].label
            idx_row = [n1, n2]
            s = self.heat_gen
            A = self.area
            h = self.length
            elem_vec = np.array([1.0, 1.0]) * (s*h*A/2.0)

        if (num_nd == 3):
            n1 = self.nodes[0].label
            n2 = self.nodes[1].label
            n3 = self.nodes[2].label
            idx_row = [n1, n2, n3]
            s = self.heat_gen
            A = self.area
            h = self.length
            elem_vec = np.array([A*h*s/6, 2*A*h*s/3, A*h*s/6])

        self.fe = elem_vec
        return elem_vec
        
    # printing method
    def __str__(self):
        string = 'N° de nós:' + str(len(self.nodes)) + '\n'
        for k, nd in enumerate(self.nodes):
            string += 'Nó ' + str(k+1) + ": " + str(nd.coords) + '\n'
        string += 'Comprimento ' + str(self.length)
        return string


class HeatTransfer1D:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.bcs = []

    def create_nodes(self, coords):
        for k, coord in enumerate(coords):
            new_nd = Node(coord, k)
            self.nodes.append(new_nd)

    def create_elements(self, connectivities, properties, materials):
        for connection in connectivities:
            new_elem = Element(connection, properties, materials, self.nodes)
            self.elements.append(new_elem)

    def create_boundary_conditions(self, bc):
        self.bcs.append(bc)
        
    def assembly_matrices(self):
        # Matriz K
        num_nd = len(self.nodes)
        K = np.zeros((num_nd, num_nd))
        f = np.zeros(num_nd)

        for element in self.elements:
            K_e = element.create_element_matrix()
            f_e = element.create_element_vector()

            num_nd = len(element.nodes)
            if (num_nd == 2):
                pos1 = element.nodes[0].label
                pos2 = element.nodes[1].label

                K[pos1, pos1] += K_e[0,0]
                K[pos1, pos2] += K_e[0,1]
                K[pos2, pos1] += K_e[1,0]
                K[pos2, pos2] += K_e[1,1]

                f[pos1] += f_e[0]
                f[pos2] += f_e[1]
            if (num_nd == 3):
                pos1 = element.nodes[0].label
                pos2 = element.nodes[1].label
                pos3 = element.nodes[2].label

                K[pos1, pos1] += K_e[0,0]
                K[pos1, pos2] += K_e[0,1]
                K[pos1, pos3] += K_e[0,2]

                K[pos2, pos1] += K_e[1,0]
                K[pos2, pos2] += K_e[1,1]
                K[pos2, pos3] += K_e[1,2]

                K[pos3, pos1] += K_e[2,0]
                K[pos3, pos2] += K_e[2,1]
                K[pos3, pos3] += K_e[2,2]

                f[pos1] += f_e[0]
                f[pos2] += f_e[1]
                f[pos3] += f_e[2]

        self.K = K
        self.f = f

    # Method to solve the problem
    def solve(self):
        for bc in self.bcs:
            # Penalization
            if (bc['type'] == 'T'):
                self.K[bc['node'], bc['node']] += 1e20
                self.f[bc['node']] = 1e20 * bc['value']

        T = np.linalg.solve(self.K, self.f)
        self.T = T
        return T
    



## "Main code"
L = 1.0
N = 5
x = np.linspace(0, L, N)
y = np.zeros(N)
z = np.zeros(N)
Ti = 20
Tf = 12

global_coords = np.column_stack((x, y, z))

# elementos quadraticos
Ne = int((N+1)/3)
connectivities = np.zeros((Ne, 3), dtype=int)
connectivities[:,0] = np.arange(0, N-1, 2)
connectivities[:,1] = connectivities[:,0] + 1
connectivities[:,2] = connectivities[:,0] + 2
print(connectivities)

properties = {'s': 2.00, 'A': 0.3} # dicionário, "key-value pairs"
material = {'density': 1.00, 'specific_heat': 1.00, 'thermal_cond': 0.1}

problem = HeatTransfer1D()
problem.create_nodes(global_coords)
problem.create_elements(connectivities, properties, material)
problem.assembly_matrices()
problem.create_boundary_conditions({'type':'T', 'node':0, 'value':Ti})
problem.create_boundary_conditions({'type':'T', 'node':N-1, 'value':Tf})
T = problem.solve()

# Reference solution
s = properties['s']
k = material['thermal_cond']
xref = np.linspace(0, L, 200)
Tref =  -s*xref**2.0/(2*k) + ((Tf - Ti)/L + s*L/(2*k))*xref + Ti

# Compare FEM solution with the reference
plt.close('all')
plt.plot(xref, Tref, 'k', label='Reference')
plt.plot(x, T, '--sb', label='FEM result')
plt.legend()
plt.xlabel('x')
plt.ylabel('Temperature')

plt.show()