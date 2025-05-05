from numba import jit
import numpy as np
from general_model import Model

@jit(nopython=True)
def flip_edge_proposal(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    edge_to_change = np.random.randint(0, num_nodes, 2)
    adj_matrix_proposed = adj_matrix.copy()
    adj_matrix_proposed[edge_to_change[0], edge_to_change[1]] = 1 - adj_matrix_proposed[edge_to_change[0], edge_to_change[1]]
    return adj_matrix_proposed

def renyi_prob_ratio(proposed_adj_matrix, current_adj_matrix, q, hamiltonian, lagrange_multipliers, avg_prob):
    current_h = hamiltonian(current_adj_matrix, lagrange_multipliers)
    proposed_h = hamiltonian(proposed_adj_matrix, lagrange_multipliers)
    return ((1 + (1 - q)/q * proposed_h) / (1 - (1 - q) * current_h)) ** (1 / (1 - q))

class Renyi(Model):
    def __init__(self, q, hamiltonian, lagrange_multipliers, precomputed_prob_ratios=None):
        self.q = q
        self.hamiltonian = hamiltonian
        self.lagrange_multipliers = lagrange_multipliers