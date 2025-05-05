import numpy as np
from . import hamiltonians
from . import tsallis
from . import metropolis

def generate_tsallis_ensemble(q, lagrange_multipliers, num_nodes, num_samples, initial_num_edges, skip_first_samples=0, progress_bar=True):
    er_hamiltonian = hamiltonians.erdos_renyi_hamiltonian
    tsallis_model = tsallis.Tsallis(q, hamiltonian=er_hamiltonian, lagrange_multipliers=lagrange_multipliers)

    initial_adj_matrix = np.zeros((num_nodes, num_nodes), dtype = 'bool')
    random_positions = np.random.choice(num_nodes, size=(initial_num_edges, 2))
    initial_adj_matrix[random_positions[:, 0], random_positions[:, 1]] = 1
    
    steps_between_samples = num_nodes**2
    metropolis_sampler_tsallis = metropolis.Metropolis(tsallis_model, initial_adj_matrix, steps_between_samples=steps_between_samples, progress_bar=progress_bar)

    ensemble = metropolis_sampler_tsallis.sample(num_samples + skip_first_samples)
    ensemble = ensemble[skip_first_samples:]

    return ensemble

def generate_tsallis_undirected_configuration_ensemble(q, lagrange_multipliers, num_nodes, num_samples, initial_num_edges, skip_first_samples=0, steps_between_samples=None, progress_bar=True):
    tsallis_model = tsallis.Tsallis(q, hamiltonian=hamiltonians.undirected_configuration_model, lagrange_multipliers=lagrange_multipliers, directed=False, self_loops_allowed=False)
    
    initial_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    random_positions = np.random.choice(num_nodes * num_nodes, size=initial_num_edges, replace=False)
    initial_adj_matrix.flat[random_positions] = 1
    initial_adj_matrix = np.triu(initial_adj_matrix) + np.triu(initial_adj_matrix, 1).T

    if steps_between_samples is None:
        steps_between_samples = num_nodes**2
    
    metropolis_sampler = metropolis.Metropolis(tsallis_model, initial_adj_matrix, steps_between_samples=steps_between_samples, progress_bar=progress_bar)
    samples = metropolis_sampler.sample(num_samples + skip_first_samples)
    samples = samples[skip_first_samples:]
    
    return samples