import tensornetwork as tn

#instead of using the tensor network class 
#that was shown a couple weeks ago, 
#I figured something that already exists 
#would be faster, so using this going forward

from scipy.stats import unitary_group

import numpy as np

from pandas.core.common import flatten

from typing import Any, Tuple, List

swap = np.zeros((2,2,2,2)) #tensor for swap gate
swap[0,0,0,0] = 1
swap[1,1,1,1] = 1
swap[0,1,1,0] = 1
swap[1,0,0,1] = 1

def normalize_gate(gate):
    
    #transforms a unitary matrix in a 4x4 shape into a rank 4 tensor
    
    assert gate.shape == (4,4) or gate.shape == (2,2,2,2)
    if gate.shape == (4,4):
        return gate.T.reshape(2,2,2,2)
    else:
        return gate

def delete_traces_no_complaint(N):
    
    #constracts all self-edges in the node. 
    
    has_trace_edges = False
    for edge in N.edges:
        if edge.is_trace():
            has_trace_edges = True
            break

    if has_trace_edges:
        N = tn.contract_trace_edges(N)
    return N

class MPS:
    
    #cyclic MPS with the inner bond dimension being variable. it always
    #starts at |0>. this will keep the structure of the MPS roughly, but 
    #not quite, as described in the paper. the difference is, we do not 
    #keep separate eigenvalue diagonal matrices between the tensors. 
    #instead, these are just absorbed into the tensors. 
    
    def __init__(self, n_qubits):
        #the list where we're going to keep the tensors
        self.nodes = []
        self.n_qubits = n_qubits
        self.cyclic = True
        for i in range(n_qubits):
            n = np.array([1,0]).reshape((1,2,1))
            self.nodes.append(tn.Node(n, name = str(i)))
        
        for i in range(n_qubits):
            #here we connect the tensors
            j = (i + 1) % n_qubits
            self.nodes[i][2] ^ self.nodes[j][0]
    
    def out_edge(self, i):
        #for the i-th tensor, this returns the dangling edge
        return list(tn.get_all_dangling([self.nodes[i]]))[0]
    
    def left_edges(self, i):
        #returns the edge to the left
        ip = (i + self.n_qubits - 1) % self.n_qubits
        return list(tn.get_shared_edges(self.nodes[i], self.nodes[ip]))
    
    def right_edges(self, j):
        #returns the edge to the right
        jn = (j + 1) % self.n_qubits
        return list(tn.get_shared_edges(self.nodes[j], self.nodes[jn]))
            
    def apply_one_site_gate(self, i, gate):
        #applies one gate operation to a qubit
        assert gate.shape == (2,2)
        gate = tn.Node(gate)
        gate[0] ^ self.out_edge(i)
        self.nodes[i] = self.nodes[i] @ gate
        
    def apply_consecutive_gates(self, i, gate):
        
        #place a 2-gate operation on the i-th and i+1-st tensor. 
        #it essentially contracts the gate and the two tensors at
        #those positions and does an SVD decomposition. when doing
        #the SVD decomposition, we only take the top 4 r vectors, 
        #where r was the bond dimension between i and i+1, since
        #per the paper, the inner dimensionality would increase at
        #most by a factor of 4. 
        
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
        #applies swap to i and i+1
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
        
        #with a combination of swap gates, and the gate we
        #intend to apply, we decompose an operation on non-
        #consecutive gates as a series of operations on 
        #consecutive gates.
        
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
        
        #this operation destroys the state, i.e. you cannot
        #use the MPS after applying this. then this returns
        #the overlap between the state represented by the MPS
        #and |0>. 
        
        #the complexity of this is n*(D^2) where D is the
        #maximum inner bond dimension, because the biggest
        #contraction we do has a multiplication by D^2. 
        
        #note that in our case, D is constant, so this is just n. 
        
        for i in range(self.n_qubits):
            m = tn.Node(np.array([1,0]))
            self.out_edge(i) ^ m[0]
            self.nodes[i] = self.nodes[i] @ m
        
        single = self.nodes[0]
        for n in self.nodes[1:]:
            single = single @ n
        
        # Destroy the MPS
        self.nodes = None
        
        return single.tensor
    
    def reduce_at_sites(self, sites, values = None):
        
        #this applies the |0> to dangling edges provided, 
        #and reduces the MPS to a smaller MPS
        
        #the complexity of this is number of sites*D^2, 
        #where D is still the maximum inner bond dimension. 
        
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
        
        #gets the norm of the state represented by the MPS. 
        #then it destroys the state. 
        
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
        
        #gets the TF of the state represented by the MPS. 
        #then it destroys the state. 

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
        
        #gives the probability that when we measure, 
        #the specififed sites are 0. 
        
        self.reduce_at_sites(sites)
        return self.get_norm()
    
    def get_full_state(self):
        
        #collapses the whole MPS and gets the 2^n dimensional 
        #state. obviously very expensive and should only be 
        #used for testing purposes. 
        
        N = self.nodes[0]
        for i in range(1, self.n_qubits):
            N = N @ self.nodes[i]

        self.nodes = None
        return N.tensor
    
    def rotate(self, shift):
        
        #since we're working with a cycles, we can rotate the MPS. 
        #this operation moves the tensor #(shift) into the position 0, 
        #that is, it rotates the MPS #(shift) places backwards. 
        
        self.nodes = self.nodes[shift:] + self.nodes[:shift]
        return self
        
    def reverse(self):
        #flips the MPS order
        self.nodes = self.nodes[::-1]
        return self
    
    def copy(self, conjugate = False):
        
        #copies the whole MPS state and returns a new instance of
        #the same MPS with the same information. this is useful because
        #we have operations that destroy the state. 
        
        node_dict, _ = tn.copy(self.nodes, conjugate = conjugate)
        mps_copy = MPS(0)
        mps_copy.n_qubits = self.n_qubits
        for i in range(mps_copy.n_qubits):
            rel_node = node_dict[self.nodes[i]]
            mps_copy.nodes.append(rel_node)
        
        return mps_copy

def contract_tensors(left: Tuple[List[int], np.ndarray], right: Tuple[List[int], np.ndarray]):
    # Not implemented yet
    # Not important right now since ladders are automatically reduced
    return None

def reduce_circuit(circuit: List[Tuple[List[int], np.ndarray]]):
    n_gates = len(circuit)
    contracted = circuit[0]
    
    new_circuit = []
    for gate in circuit[1:]:
        new_contracted = contract_tensors(contracted, gate)
        if new_contracted is None:
            new_circuit.append(contracted)
            contracted = gate
        else:
            contracted = new_contracted
            
    new_circuit.append(contracted)

    return new_circuit

def compute_mps(n_qubits, circuit):
    
    #given a circuit, it simulates the circuit by
    #applying the gates of the circuit. 
    
    red_circuit = reduce_circuit(circuit)
    m = MPS(n_qubits)
    for pos, gate in red_circuit:
        assert (len(pos) in [1,2])
        if len(pos) == 1:
            m.apply_one_site_gate(pos[0], gate)
        else:
            m.apply_two_site_gate(pos[0], pos[1], gate)
    return m

def random_unitary_gate(n_qubits):
    
    #returns a random unitary operating on n qubits
    
    return unitary_group.rvs(2**n_qubits)

def mk_ladder(n_qubits, all_same = True):
    
    #returns a ladder of random gates

    if all_same:
        gate = random_unitary_gate(2)
        return [([i, i + 1], gate) for i in range(n_qubits - 1)]
    else:
        return [([i, i + 1], random_unitary_gate(2)) for i in range(n_qubits - 1)]

def reverse_ladder(circuit):
    
    #reverses a circuit
    
    rev_circuit = []
    for pos, gate in circuit[::-1]:
        rev_circuit.append((pos, np.array(np.matrix(gate).H)))
    return rev_circuit

def sample_ladder(n_qubits, all_same = True):
    
    #samples the circuit we need to simulate the proposed method
    
    circuit = mk_ladder(n_qubits, all_same = all_same)
    rev_circuit = reverse_ladder(circuit[:-1])
    total_circuit = circuit + rev_circuit
    return total_circuit
    
def sample_process(n_qubits, all_same = True):
    
    #takes the above and simulates it with an MPS
    
    total_circuit = sample_ladder(n_qubits, all_same = all_same)
    return compute_mps(n_qubits, total_circuit).zero_overlap()

def get_all_probabilities(m, all_same = True):
    
    #for an MPS, we compute the probability 
    #that its state will collapse to 0 after measuring.
    
    #the complexity of this is quadratic. 
    
    circuit = sample_ladder(n_qubits)
    m = compute_mps(n_qubits, circuit)
    probs = []
    for i in range(n_qubits):
        p = m.copy().probability_zero_at_sites([i])
        probs.append(np.real(p))
    return probs

def advance_merging_mps(N, mps, co_mps, m):
    
    #helper function for a faster sample_all_qubits
    
    for i in range(1,m + 1):
        mps.out_edge(i) ^ co_mps.out_edge(i)
        N = N @ mps.nodes[i]
        N = N @ co_mps.nodes[i]
    
    for t in [mps, co_mps]:
        t.nodes = [None] + t.nodes[(m+1):]
        t.n_qubits = len(t.nodes)
    
    return N

def copy_mps_co_mps(N, mps, co_mps):
    
    #for an MPS and its conjugate that are joined at the 0-th site, 
    #it takes the combined structure and copies it. 
    
    n_qubits = mps.n_qubits
    node_dict, _ = tn.copy([N]+ mps.nodes[1:] + co_mps.nodes[1:])
    N2 = node_dict[N]
    
    mpss = []
    for t in [mps, co_mps]:
        m = MPS(0)
        m.nodes = [None] + [node_dict[t.nodes[i]] for i in range(1, n_qubits)]
        m.n_qubits = len(m.nodes)
        mpss.append(m)
    
    mps2, co_mps2 = mpss
    
    return N2, mps2, co_mps2

def get_probabilities_helper(N, mps, co_mps):
    
    #mps and co_mps are joined at 0 
    #joint node at position 0 must be provided as N
    #using divide and conquer approach, for an MPS of size n, 
    #we reduce the problem of computing get_all_probabilities
    #into two instances of get_all_probabilities of size n/2
    #in linear time. 
    
    #overall, this implies a complexity of n*log(n). 
    
    if mps.n_qubits == 2:
        m = tn.Node(np.array([1,0]))
        mps.out_edge(1) ^ m[0]
        
        co_m = tn.Node(np.array([1,0]))
        co_mps.out_edge(1) ^ co_m[0]
        
        mps.nodes[1] = mps.nodes[1] @ m
        co_mps.nodes[1] = co_mps.nodes[1] @ co_m
        
        N = N @ mps.nodes[1]
        N = N @ co_mps.nodes[1]
        N = delete_traces_no_complaint(N)
        return [np.real(N.tensor)]
    
    n_qubits = mps.n_qubits
    
    m = (n_qubits - 1) // 2    
    N2, mps2, co_mps2 = copy_mps_co_mps(N, mps, co_mps)
    
    N = advance_merging_mps(N, mps, co_mps, m)
    first_probs = get_probabilities_helper(N, mps, co_mps)
    
    mps2.reverse().rotate(-1)
    co_mps2.reverse().rotate(-1)
    
    N2 = advance_merging_mps(N2, mps2, co_mps2, n_qubits - 1 - m)
    second_probs = get_probabilities_helper(N2, mps2, co_mps2)
    
    return second_probs[::-1] + first_probs

def get_all_probabilities_faster(mps: MPS):
    
    #wrapper function for the above function, for conovenience. 
    
    if mps.n_qubits == 1:
        return [mps.get_norm()]
    
    n_qubits = mps.n_qubits
    
    p0 = mps.copy().probability_zero_at_sites([0])
    
    co_mps = mps.copy(conjugate = True)
    co_mps.out_edge(0) ^ mps.out_edge(0)
    
    N = co_mps.nodes[0] @ mps.nodes[0]
    mps.nodes[0] = None
    co_mps.nodes[0] = None
    probs = get_probabilities_helper(N, mps, co_mps)
    return [p0] + probs

def sample_all_qubits(n_qubits):
    
    #samples the ladder, computes the MPS, 
    #and gets all the probabilities
    
    circuit = sample_ladder(n_qubits)
    m = compute_mps(n_qubits, circuit)
    return get_all_probabilities(m)

def sample_all_qubits_faster(n_qubits, all_same = True):
    
    #samples the ladder, computes the MPS, 
    #and gets all the probabilities, but faster. 
    
    circuit = sample_ladder(n_qubits, all_same = all_same)
    m = compute_mps(n_qubits, circuit)
    return get_all_probabilities_faster(m.copy())

def sample_top_k(n_qubits, k = 3):
    
    #the probability of getting 0,0,0 in the first qubits,
    #that is, the furthest ones from the edge of the ladder
    
    circuit = sample_ladder(n_qubits)
    m = compute_mps(n_qubits, circuit)
    return m.probability_zero_at_sites(list(range(k)))

def sample_block(left: List[int], right: List[int]):
    D = len(left)
    assert D == len(right)
    block = []
    for i in range(D):
        if i % 2 == 0:
            pairs = zip(left, right)
        else:
            pairs = zip(left[1:], right)

        for l, r in pairs:
            block.append(([l, r], random_unitary_gate(2)))
        
    return block



def sample_better_ladder(n_lines: int, D: int) -> List[List[Any]]:
    # total: n_lines and D qubits per line
    # D = 1 should be the same as previous case
    circuit = []
    for i in range(1, n_lines):
        l = list(range((i-1)*D, i*D))
        r = list(range(i*D, (i+1)*D))
        circuit.append(sample_block(l, r))

    return circuit

def sample_better_process(n_lines:int, D: int):
    circuit = sample_better_ladder(n_lines, D)
    except_last_part = list(sum(circuit[:-1], []))
    reversing = reverse_ladder(except_last_part)
    circuit = list(sum(circuit,[])) + reversing
    return circuit

def sample_better_ladder_faster_probabilities(n_qubits, D):
    c = sample_better_process(n_qubits, D)
    mps = compute_mps(n_qubits * D, c)
    return get_all_probabilities_faster(mps.copy())