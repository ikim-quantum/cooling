import tensornetwork as tn
import numpy as np

from typing import List, Tuple

def contract_tensors(left: Tuple[List[int], np.ndarray], right: Tuple[List[int], np.ndarray]):
    pass

def get_reduced_circuit(circuit: List[Tuple[List[int], np.ndarray]]):
    n_gates = len(circuit)
    contracted = None
    for gate in circuit:
        if contracted is None:
            contracted = gate
        else:
            
    pass
    