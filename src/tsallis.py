import igraph as ig
from tqdm import tqdm
import numpy as np
from numba import jit
from .general_model import Model


# @jit(nopython=True)
# def flip_edge_proposal(adj_matrix):
#     num_nodes = adj_matrix.shape[0]
#     edge_to_change = np.random.randint(0, num_nodes, 2)
#     adj_matrix_proposed = adj_matrix.copy()
#     adj_matrix_proposed[edge_to_change[0], edge_to_change[1]] = 1 - adj_matrix_proposed[edge_to_change[0], edge_to_change[1]]
#     return adj_matrix_proposed

@jit(nopython=True)
def flip_edge_proposal(adj_matrix, directed: bool = True, self_loops_allowed: bool = True):
    num_nodes = adj_matrix.shape[0]
    adj_matrix_proposed = adj_matrix.copy()
    while True:
        i, j = np.random.randint(0, num_nodes, 2)
        if not self_loops_allowed and i==j:
            continue        
        adj_matrix_proposed[i, j] = 1 - adj_matrix_proposed[i, j]
        if not directed:
            adj_matrix_proposed[j, i] = adj_matrix_proposed[i, j]  # Ensure symmetry
        return adj_matrix_proposed

# @jit(nopython=True)
def tsallis_prob_ratio(proposed_adj_matrix, current_adj_matrix, q, hamiltonian, lagrange_multipliers):
    current_h = hamiltonian(current_adj_matrix, lagrange_multipliers)
    proposed_h = hamiltonian(proposed_adj_matrix, lagrange_multipliers)
    
    if q == 1.0:
        return np.exp(current_h - proposed_h)
    else:
        numerator = 1 - (q - 1) * proposed_h
        denominator = 1 - (q - 1) * current_h
        if numerator >= 0 and denominator >= 0:
            return ((1 - (q - 1) * proposed_h) / (1 - (q - 1) * current_h)) ** (1 / (q - 1))
        else:
            return 0

class Tsallis(Model):
    def __init__(self, q, hamiltonian, lagrange_multipliers, directed=True, self_loops_allowed=True):
        self.q = q
        self.hamiltonian = hamiltonian
        self.lagrange_multipliers = lagrange_multipliers
        self.directed = directed
        self.self_loops_allowed = self_loops_allowed

    def proposal_fn(self, graph):
        return flip_edge_proposal(graph, self.directed, self.self_loops_allowed)
    
    def transform_fn(self, adj_matrix):
        mode = 'undirected' if not self.directed else 'directed'
        loops = 'once' if self.self_loops_allowed else 'ignore'
        return ig.Graph.Adjacency(adj_matrix, mode=mode, loops=loops)

    def prob_ratio(self, proposed_adj_matrix, current_adj_matrix):
        return tsallis_prob_ratio(proposed_adj_matrix, current_adj_matrix, self.q, self.hamiltonian, self.lagrange_multipliers)

@jit(nopython=True)
def tsallis_precomputed_prob_ratio(q, lagrange_multipliers, num_links_proposed, num_links_current, precomputed_probs):
    if q==1:
        return np.exp(lagrange_multipliers[0]*(num_links_current - num_links_proposed))
    return precomputed_probs[num_links_proposed]/precomputed_probs[num_links_current]

@jit(nopython=True)
def flip_edge_proposal_num_links(adj_matrix, num_links_current):
    num_nodes = adj_matrix.shape[0]
    edge_to_change = np.random.randint(0, num_nodes, 2)
    adj_matrix_proposed = adj_matrix.copy()
    adj_matrix_proposed[edge_to_change[0], edge_to_change[1]] = 1 - adj_matrix_proposed[edge_to_change[0], edge_to_change[1]]
    return adj_matrix_proposed, num_links_current - 1 + 2*adj_matrix_proposed[edge_to_change[0], edge_to_change[1]]

class TsallisPrecomputed(Model):
    def __init__(self, q, hamiltonian, lagrange_multipliers, precomputed_probs):
        self.q = q
        self.hamiltonian = hamiltonian
        self.lagrange_multipliers = lagrange_multipliers
        self.precomputed_probs = precomputed_probs

    def proposal_fn(self, state):
        return flip_edge_proposal_num_links(state[0], state[1])
    
    def transform_fn(self, state):
        return ig.Graph.Adjacency(state[0])
    
    def prob_ratio(self, proposed_state, current_state):
        return tsallis_precomputed_prob_ratio(self.q, self.lagrange_multipliers, proposed_state[1], current_state[1], self.precomputed_probs)