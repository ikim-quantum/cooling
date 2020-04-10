###############################################
# Efficient simulation of algorithmic cooling #
# for noise-resilient circuits                #
# MIT License                                 #
# Galit Anikeeva and Isaac H. Kim             #
# Last update: 4/10/2020                      #
###############################################

import tensornetwork as tn
import numpy as np
import warnings
from scipy.stats import unitary_group
from pandas.core.common import flatten
from typing import Any, Tuple, List


def normalize_gate(gate):
    """
    Transforms a 4x4 unitary matrix into a 2x2x2x2 tensor.

    Comment: It will be good to verify that gate is indeed
             unitary.

    Args:
        gate(np.array): 4x4 matrix or a 2x2x2x2 tensor.

    Returns:
        np.array: 2x2x2x2 tensor
    """
    assert gate.shape == (4,4) or gate.shape == (2,2,2,2)
    if gate.shape == (4,4):
        return gate.T.reshape(2,2,2,2)
    else:
        return gate

    
def delete_traces_no_complaint(N):
    """
    Contracts all the edges that begin and end on the same
    node. (Let's call them as "internal" edges.)

    Args:
        N(tensornetwork.Node): Tensornetwork Node

    Returns:
        N(tensornetwork.Node): The same node, but after
                               contracting all the internal
                               edges.
    """
    has_trace_edges = False
    for edge in N.edges:
        if edge.is_trace():
            has_trace_edges = True
            break

    if has_trace_edges:
        N = tn.contract_trace_edges(N)
    return N


class MPS:
    """
    Matrix Product State(MPS) with periodic/cyclic boundary 
    condition. The physical degrees of freedoms are qubits.

    Attrs:
        n_qubits(int): Number of qubits in the chain.
        nodes(list): A list of nodes. Each node has three types 
                     of edges. 
                     1) Out edge: The edge(s) representing the 
                                  physical qubits.
                     2) Left edge: Internal edge. Connected to the 
                                   right edge of the left node.
                     3) Right edge: Internal edge. Connected to
                                    the left edge of the right node.
        cyclic(bool): True if the boundary condition is periodic.
                      False otherwise.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.nodes = []
        self.cyclic = True

        # Initialize each node.
        for i in range(n_qubits):
            n = np.array([1,0]).reshape((1,2,1))
            self.nodes.append(tn.Node(n, name = str(i)))

        # Connect each nodes to each other.
        # Combined with the initial condition for each nodes,
        # we obtain a |0...0> state.
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            self.nodes[i][2] ^ self.nodes[j][0]
    
    def out_edge(self, i):
        """
        Args:
            i(int): Index of the node.

        Returns:
            list: list of out edges of the i'th node.
        """
        return list(tn.get_all_dangling([self.nodes[i]]))[0]
    
    def left_edges(self, i):
        """
        Args:
            i(int): Index of the node.

        Returns:
            list: list of left edges of the i'th node.
        """
        ip = (i + self.n_qubits - 1) % self.n_qubits
        return list(tn.get_shared_edges(self.nodes[i], self.nodes[ip]))
    
    def right_edges(self, j):
        """
        Args:
            i(int): Index of the node.

        Returns:
            list: list of right edges of the i'th node.
        """
        jn = (j + 1) % self.n_qubits
        return list(tn.get_shared_edges(self.nodes[j], self.nodes[jn]))
            
    def apply_one_site_gate(self, i, gate):
        """
        Applies a single-qubit gate to the i'th qubit. The gate 
        has to be a 2x2 matrix.

        Args:
            i(int): Index of the qubit.
            gate(np.array): 2x2 matrix.
        """
        assert gate.shape == (2,2)
        gate = tn.Node(gate)
        gate[0] ^ self.out_edge(i)
        self.nodes[i] = self.nodes[i] @ gate
        
    def apply_consecutive_gates(self, i, gate):
        """
        Applies a two-qubit gate to the i'th and the (i+1)'th qubit. 
        The gate must be either a 4x4 matrix or a 2x2x2x2 tensor.
        
        Comment: In tn.split_node, it may make sense to use max_truncation_err
                 option instead of max_singular_values option, in case there
                 was a mistake in my note.

        Args:
            i(int): Index of the qubit.
            gate(np.array): 4x4 matrix or 2x2x2x2 tensor.
        """
        gate = normalize_gate(gate)
        j = (i + 1) % self.n_qubits
        
        # get dimension of inner bond between i and i+1
        r = self.right_edges(i)[0].dimension
        
        gate = tn.Node(gate)
        
        # connect i and i+1 to the gate
        self.out_edge(i) ^ gate[0]
        self.out_edge(j) ^ gate[1]
        
        # Get non-dangling edges for each side
        left_edges = self.left_edges(i) + [gate[2]]
        right_edges = self.right_edges(j) + [gate[3]]
        
        # Contract edges
        C = (self.nodes[i] @ self.nodes[j])
        D = C @ gate

        self.nodes[i], self.nodes[j], _ = tn.split_node(D, 
                                                     left_edges = left_edges, 
                                                     right_edges = right_edges, 
                                                     max_singular_values = 4*r, 
                                                     left_name = str(i), 
                                                     right_name = str(j))

    def apply_swap(self, i):
        """
        SWAP the i'th qubit with the (i+1)'th qubit.

        Args:
            int(i): Qubit index.
        """
        swap = np.array([[[[1., 0.],
                 [0., 0.]],
                [[0., 0.],
                 [1., 0.]]],
               [[[0., 1.],
                 [0., 0.]],
                [[0., 0.],
                 [0., 1.]]]])
        
        self.apply_consecutive_gates(i, swap)
        
    def apply_two_site_gate(self, i, j, gate):
        """
        Apply a two-qubit gate between i and j.
        
        Args:
            i,j(int): Index of the qubits
            gate(np.array): 4x4 or 2x2x2x2 matrix.
        """
        gate = normalize_gate(gate)
        assert i != j
        
        if i < j:
            t = i
            while (t + 1) != j:
                self.apply_swap(t)
                t += 1
                
            self.apply_consecutive_gates(t, gate)
            
            t -= 1
            while t >= i:
                self.apply_swap(t)
                t -= 1
        else:
            t = i - 1
            while t > j:
                self.apply_swap(t)
                t -= 1
                
            self.apply_consecutive_gates(j, gate)
            
            t += 1
            while t < i:
                self.apply_consecutive_gates(t)        
        
    def zero_overlap(self):
        """
        Computes the overlap between the MPS and all-0 state.
        Note that this is an amplitude, not amplitude squared.
        Also, this method destroys the MPS, so be careful in
        using it.

        Complexity of this operation is O(nD^2), where D is the
        maximum inner(left/right) bond dimension and n is the
        number of qubits.

        Returns:
            float: overlap between self and |0...0>.
        """
        for i in range(self.n_qubits):
            m = tn.Node(np.array([1,0]))
            self.out_edge(i) ^ m[0]
            self.nodes[i] = self.nodes[i] @ m
        
        single = self.nodes[0]
        for n in self.nodes[1:]:
            single = single @ n

        self.nodes = None
        
        return single.tensor
    
    def reduce_at_sites(self, sites, values = None):
        """
        Contracts MPS with |0> on the given sites. This
        reduces the MPS to a smaller MPS.
        
        The complexity of this method is O(len(sites)*D^2),
        where D is the maximum inner(left/right) bond 
        dimension.

        Args:
            sites(list): List of sites.
            values(list): Optional list, which stores the
                          value of the bit we want impose.
                          For instance, [0,1,0] would contract
                          the MPS with the state |010>. 
                          Default values of the list are 0.
        """
        if values is None:
            values = [0] * len(sites)
            
        sites = {s: v for s, v in zip(sites, values)}
        for s, v in sites.items():
            if v == 0:
                m = tn.Node(np.array([1,0]))
            elif v == 1:
                m = tn.Node(np.array([0,1]))
            elif v.shape == (2,):
                m = tn.Node(np.array(v))
            self.out_edge(s) ^ m[0]
            self.nodes[s] = self.nodes[s] @ m
        
        first_nonsite = None
        for i in range(self.n_qubits - 1):
            if i in sites:
                self.nodes[i + 1] = self.nodes[i] @ self.nodes[i + 1]
                self.nodes[i] = None
            elif first_nonsite is None:
                first_nonsite = i
        if (self.n_qubits - 1) in sites:
            self.nodes[first_nonsite] = self.nodes[-1] @ self.nodes[first_nonsite]
            self.nodes[-1] = None
        
        self.nodes = [node for node in self.nodes if node is not None]
        self.n_qubits = len(self.nodes)
        return self
    
    def get_norm(self):
        """
        Gets the norm of the MPS.

        Returns:
            float: norm of the MPS.
        """
        co_mps = self.copy(conjugate=True)
        for i in range(self.n_qubits):
            self.out_edge(i) ^ co_mps.out_edge(i)
        
        N = self.nodes[0] @ co_mps.nodes[0]
        
        for i in range(1, self.n_qubits):
            N = N @ self.nodes[i]
            N = N @ co_mps.nodes[i]
        
        N = delete_traces_no_complaint(N)
        
        return np.real(N.tensor)

    def get_transfer_matrix(self):
        """
        Obtain the transfer matrix defined with
        respect to the first node.

        Comment: Not sure what's going on here. Need
                 to check later.
        """
        warnings.warn("Need to check this method.", Warning)
        co_mps = self.copy(conjugate=True)
        for i in range(self.n_qubits):
            self.out_edge(i) ^ co_mps.out_edge(i)
        
        edgesA = self.left_edges(0) # should be 1
        edgesB = co_mps.left_edges(0) # should be 1

        for e in (edgesA + edgesB):
            e.disconnect()

        left = list(self.nodes[0].get_all_dangling()) + list(co_mps.nodes[0].get_all_dangling())
        right = list(self.nodes[-1].get_all_dangling()) + list(co_mps.nodes[-1].get_all_dangling())
        
        N = self.nodes[0] @ co_mps.nodes[0]
        
        for i in range(1, self.n_qubits):
            N = N @ self.nodes[i]
            N = N @ co_mps.nodes[i]
        
        # N = delete_traces_no_complaint(N)

        return N, left, right
    
    def probability_zero_at_sites(self, sites):
        """
        Args:
            sites(list): A list of qubits

        Returns:
            float: Probability that all the qubits in
                   sites is 0.
        """
        self.reduce_at_sites(sites)
        return self.get_norm()
    
    def get_full_state(self):
        """
        Collapses the whole MPS and gets the 2^n-dimensional
         state. Only for testing purposes.

        Returns:
            MPS: Post-collapse state.
        """
        N = self.nodes[0]
        for i in range(1, self.n_qubits):
            N = N @ self.nodes[i]

        self.nodes = None
        return N.tensor
    
    def shift_self(self, shift):
        """
        Shifts the MPS by shift.

        Args:
            shift(int): shift unit
        """
        self.nodes = self.nodes[shift:] + self.nodes[:shift]
        return self
        
    def reverse_self(self):
        """
        Flips the MPS order.
        """
        self.nodes = self.nodes[::-1]
        return self
    
    def copy(self, conjugate = False):
        """
        Copies the whole MPS.

        Returns:
            MPS: A deep copy of self.
        """
        node_dict, _ = tn.copy(self.nodes, conjugate = conjugate)
        mps_copy = MPS(0)
        mps_copy.n_qubits = self.n_qubits
        for i in range(mps_copy.n_qubits):
            rel_node = node_dict[self.nodes[i]]
            mps_copy.nodes.append(rel_node)
        
        return mps_copy
