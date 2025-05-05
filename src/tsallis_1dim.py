import igraph as ig
from tqdm import tqdm
import numpy as np
from numba import jit
from .general_model import Model


@jit(nopython=True)
def flip_bit_proposal(state):
    num_nodes = state.shape[0]
    bit_to_change = np.random.randint(0, num_nodes)
    state_proposed = state.copy()
    state_proposed[bit_to_change] = 1 - state_proposed[bit_to_change]
    return state_proposed

# @jit(nopython=True)
def tsallis_prob_ratio(proposed_state, current_state, q, hamiltonian, lagrange_multipliers):
    current_h = hamiltonian(current_state, lagrange_multipliers)
    proposed_h = hamiltonian(proposed_state, lagrange_multipliers)
    
    if q == 1.0:
        return np.exp(current_h - proposed_h)
    else:
        return ((1 - (q - 1) * proposed_h) / (1 - (q - 1) * current_h)) ** (1 / (q - 1))

class Tsallis1dim(Model):
    def __init__(self, q, hamiltonian, lagrange_multipliers):
        self.q = q
        self.hamiltonian = hamiltonian
        self.lagrange_multipliers = lagrange_multipliers

    def proposal_fn(self, state):
        return flip_bit_proposal(state)
    
    def transform_fn(self, state):
        return state

    def prob_ratio(self, proposed_state, current_state):
        return tsallis_prob_ratio(proposed_state, current_state, self.q, self.hamiltonian, self.lagrange_multipliers)