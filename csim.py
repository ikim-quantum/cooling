###############################################
# Efficient simulation of algorithmic cooling #
# for noise-resilient circuits:               #
# Simulation part.                            #   
# MIT License                                 #
# Galit Anikeeva and Isaac H. Kim             #
# Last update: 4/10/2020                      #
###############################################

import tensornetwork as tn
import numpy as np
import warnings
from scipy.stats import unitary_group
from typing import Any, List
from mps import MPS, delete_traces_no_complaint



def compute_mps(n_qubits, circuit, depolarizing_noise = None):
    #given a circuit, it simulates the circuit by
    #applying the gates of the circuit.

    # Tensors for single-qubit Paulis
    Id = np.eye(2)
    sx = np.array([[0,1.],[1,0]])
    sy = np.array([[1,0.],[0,-1]])
    sz = np.array([[0,-1j],[1j,0]])

    
    red_circuit = reduce_circuit(circuit)
    m = MPS(n_qubits)
    for pos, gate in red_circuit:
        assert (len(pos) in [1,2])
        if len(pos) == 1:
            m.apply_one_site_gate(pos[0], gate)
        else:
            m.apply_two_site_gate(pos[0], pos[1], gate)
        
        if depolarizing_noise is not None:
            for p in pos:
                noise_gate = np.random.choice([None, sx, sy, sz], p = [1 - 3*depolarizing_noise/4, depolarizing_noise/4, depolarizing_noise/4, depolarizing_noise/4])
                if noise_gate is None:
                    continue
                m.apply_one_site_gate(p, noise_gate)

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

    n_qubits = m.n_qubits
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
    
    mps2.reverse_self().shift_self(-1)
    co_mps2.reverse_self().shift_self(-1)
    
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

def sample_all_qubits_faster(n_qubits, all_same = True, depolarizing_noise = None, times = None):
    
    #samples the ladder, computes the MPS, 
    #and gets all the probabilities, but faster. 
    if times is None:
        times = 1
    
    # ps.mean(axis=0)
    
    circuit = sample_ladder(n_qubits, all_same = all_same)
    ps = []
    for _ in range(times):
        m = compute_mps(n_qubits, circuit, depolarizing_noise = depolarizing_noise)
        ps.append(get_all_probabilities_faster(m.copy()))
    
    ps = np.mean(ps, axis = 0)
    return ps

def sample_all_qubits_faster_polarized(n_qubits, p = 0.,all_same = True):
    
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
