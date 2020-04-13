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
import scipy.linalg as la
from typing import Any, Tuple, List
from mps import MPS, delete_traces_no_complaint


def compute_mps(n_qubits, circuit, p_noise = None):
    """
    Given a circuit, it simulates the circuit by applying
    the gates of the circuit. The global state is initialized
    to |0...0>.

    Args:
        n_qubits(int): Number of qubits in the chain.
        circuit(list): A list of tuples, which are composed of
                       a list and np.ndarray.
        p_noise: Noise strength
    
    Returns:
        m(MPS): A MPS created after applying the circuit, with
                (potentially) randomized noise. In order to study
                the effect of noise reliably, one has to sample
                over many MPSs.
    """
    # Tensors for single-qubit Paulis
    sx = np.array([[0,1.],[1,0]])
    sy = np.array([[1,0.],[0,-1]])
    sz = np.array([[0,-1j],[1j,0]])

    m = MPS(n_qubits)
    for pos, gate in circuit:
        assert (len(pos) in [1,2])
        if len(pos) == 1:
            m.apply_one_site_gate(pos[0], gate)
        else:
            m.apply_two_site_gate(pos[0], pos[1], gate)
        
        if p_noise is not None:
            for p in pos:
                noise_gate = np.random.choice([None, sx, sy, sz], p = [1 - 3*p_noise/4, p_noise/4, p_noise/4, p_noise/4])
                if noise_gate is None:
                    continue
                m.apply_one_site_gate(p, noise_gate)
    return m


def random_unitary_gate(n_qubits):
    """
    Args:
        n_qubits(int): Number of qubits

    Returns:
        np.ndarray: A random unitary acting on n qubits, in a matrix form.
    """
    return unitary_group.rvs(2**n_qubits)


def apply_channel(u, o):
    v = np.kron(np.array([1,0]), np.eye(2))
    return v @ u.T.conjugate() @ np.kron(np.eye(2),o) @ u @ v.T


def transfer_matrix(u):
    """
    Map a 2x2 matrix O to 
    (<0| \otimes I)U^{\dagger}(I \otimes O)U(I\otimes |0>)
    Args:
        u(np.ndarray): 4x4 unitary matrix

    Returns:
        np.ndarray: Transfer matrix
    """
    # We will do the dumb thing and just compute each of the matrix entries.
    o00 = np.array([[1,0],[0,0]])
    o10 = np.array([[0,0],[1,0]])
    o01 = np.array([[0,1],[0,0]])
    o11 = np.array([[0,0],[0,1]])

    op00 = apply_channel(u,o00)
    op10 = apply_channel(u,o10)
    op01 = apply_channel(u,o01)
    op11 = apply_channel(u,o11)

    transfer = [[op00[0,0], op10[0,0], op01[0,0], op11[0,0]],
                [op00[1,0], op10[1,0], op01[1,0], op11[1,0]],
                [op00[0,1], op10[0,1], op01[0,1], op11[0,1]],
                [op00[1,1], op10[1,1], op01[1,1], op11[1,1]]]

    return np.array(transfer)


def mk_ladder(n_qubits, all_same = True):
    """
    Args:
        n_qubits(int): Number of qubits
        all_same(bool): If True, choose all the gates to be equal.
                        False otherwise.
    Returns:
        List: A list of random gates in a ladder-like form.
    """
    if all_same:
        gate = random_unitary_gate(2)
        tm = transfer_matrix(gate)
        ev, junk = la.eig(tm)
        ev_abs = abs(ev)
        ev_abs.sort()
        lambda1 = ev_abs[-2]
        c_length = -(1/np.log(lambda1))
        # print(c_length)
        return [([i, i + 1], gate) for i in range(n_qubits - 1)], c_length
    else:
        return [([i, i + 1], random_unitary_gate(2)) for i in range(n_qubits - 1)], 0


def invert_circuit(circuit):
    """
    Returns the inverse circuit.

    Args:
        circuit(List): A circuit

    Returns:
        List: The inverse circuit.
    """
    circuit_inv = []
    for pos, gate in circuit[::-1]:
        circuit_inv.append((pos, np.array(np.matrix(gate).H)))
    return circuit_inv


def sample_ladder(n_qubits, all_same = True):
    """
    Generate a random instance of a ladder-like circuit plus
    the rewinding protocol.

    Args:
        n_qubits(int): Number of qubits
        all_same(bool): If True, set all the gates to be identical.
                        If False, sample each gates independently.

    Returns:
        total_circuit(List): A ladder-like circuit concatenated
                             with its rewinding.
    """
    circuit, c_length = mk_ladder(n_qubits, all_same = all_same)
    rev_circuit = invert_circuit(circuit[:-1])
    total_circuit = circuit + rev_circuit
    return total_circuit, c_length


def sample_process(n_qubits, all_same = True):
    """
    Apply a rewinding protocol to a random ladder-like circuit
    and estimate the overlap with the all-0 state.

    Args:
        n_qubits(int): Number of qubits
        all_same(bool): If True, set all the gates to be identical.
                        If False, sample each gates independently.

    Returns:
        float: overlap with the |0...0> state.
    """
    total_circuit, c_length = sample_ladder(n_qubits, all_same = all_same)
    return compute_mps(n_qubits, total_circuit).zero_overlap()


def get_all_probabilities(m):
    """
    For a MPS m, compute the fidelity between the reduced density
    matrix of the i'th site with the |0> state for all i.

    Args:
        m(MPS): MPS

    Returns:
        List(float): A list of overlap between reduced density
                     matrices and |0>.
    """
    n_qubits = m.n_qubits
    probs = []
    for i in range(n_qubits):
        p = m.copy().probability_zero_at_sites([i])
        probs.append(np.real(p))
    return probs


def get_correlation(m, i, j):
    """
    For a MPS m, compute the Pearson correlation coefficient
    between qubit i and qubit j.

    Args:
        m(MPS): MPS
        i, j(int): Qubit indices

    Returns:
        float: Pearson correlation coefficient
    """
    # Probability of being in the 1 state for i and j.
    pi = 1-m.copy().probability_zero_at_sites([i])
    pj = 1-m.copy().probability_zero_at_sites([j])
    std_i = np.sqrt(pi * (1-pi))
    std_j = np.sqrt(pj * (1-pj))

    m_temp = m.copy()
    m_temp.reduce_at_sites([i,j], [1,1])
    cor = m_temp.get_norm() - pi*pj
    return cor/(std_i * std_j)


def sample_correlation(n_q, i, j, n_samples):
    """
    Samples correlation values.

    Args:
        n_q(int): Number of qubits
        i,j(int): Qubit indices
        n_samples(int): Number of samples

    Returns:
        list(float): Samples for Pearson correlation coefficients.
    """
    sam, c_length = sample_ladder(n_q)
    out = [get_correlation(compute_mps(n_q, sam), i, j) for k in range(n_samples)]
    return out


def correlated_flip_probability(n_q, k):
    """
    Generate a random MPS of length n_q and compute the probability that
    the first k qubits are in the |1> state.

    Args:
        n_q(int): Number of qubits
        k(int): First k qubits

    Returns:
        float: Probability that the first k bits are flipped.
    """
    sam, c_length = sample_ladder(n_q)
    mps_rand = compute_mps(n_q, sam)
    mps_rand.reduce_at_sites([i for i in range(k)], [1]*k)
    return mps_rand.get_norm()


def correlated_flip_probability_extra(n_q, k):
    """
    Generate a random MPS of length n_q and compute the probability that
    the first k qubits are in the |1> state.

    Args:
        n_q(int): Number of qubits
        k(int): First k qubits

    Returns:
        float: Probability that the first k bits are flipped.
        float: correlation length
    """
    sam, c_length = sample_ladder(n_q)
    mps_rand = compute_mps(n_q, sam)
    mps_rand.reduce_at_sites([i for i in range(k)], [1]*k)
    return mps_rand.get_norm(), c_length


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
    
    circuit, c_length = sample_ladder(n_qubits)
    m = compute_mps(n_qubits, circuit)
    return get_all_probabilities(m)

def sample_all_qubits_faster(n_qubits, all_same = True, depolarizing_noise = None, times = None):
    
    #samples the ladder, computes the MPS, 
    #and gets all the probabilities, but faster. 
    if times is None:
        times = 1
    
    # ps.mean(axis=0)
    
    circuit, c_length = sample_ladder(n_qubits, all_same = all_same)
    ps = []
    for _ in range(times):
        m = compute_mps(n_qubits, circuit, p_noise = depolarizing_noise)
        ps.append(get_all_probabilities_faster(m.copy()))
    
    ps = np.mean(ps, axis = 0)
    return ps

def sample_all_qubits_faster_polarized(n_qubits, p = 0.,all_same = True):
    
    #samples the ladder, computes the MPS, 
    #and gets all the probabilities, but faster. 
    
    circuit, c_length = sample_ladder(n_qubits, all_same = all_same)
    m = compute_mps(n_qubits, circuit)
    return get_all_probabilities_faster(m.copy())

def sample_top_k(n_qubits, k = 3):
    
    #the probability of getting 0,0,0 in the first qubits,
    #that is, the furthest ones from the edge of the ladder
    
    circuit, c_length = sample_ladder(n_qubits)
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
    reversing = invert_circuit(except_last_part)
    circuit = list(sum(circuit,[])) + reversing
    return circuit

def sample_better_ladder_faster_probabilities(n_qubits, D):
    c = sample_better_process(n_qubits, D)
    mps = compute_mps(n_qubits * D, c)
    return get_all_probabilities_faster(mps.copy())
