from typing import List

import igraph as ig
from numba import jit
import numpy as np

@jit(nopython=True)
def erdos_renyi_hamiltonian(adj_matrix, lagrange_multipliers: List[float]) -> float:
    """"The graph Hamiltonian for the Erdos-Renyi model."""
    return np.sum(adj_matrix) * lagrange_multipliers[0]

@jit(nopython=True)
def erdos_renyi_hamiltonian_num_links(num_links, lagrange_multipliers: List[float]) -> float:
    """The graph Hamiltonian for the Erdos-Renyi model with precomputed number of links."""
    return num_links * lagrange_multipliers[0]

# @jit(nopython=True)
def configuration_model_hamiltonian(adj_matrix, lagrange_multipliers: List[tuple]) -> float:
    """The graph Hamiltonian for the configuration model using matrix operations."""
    out_degrees = np.sum(adj_matrix, axis=1)  # Sum along rows for out-degrees
    in_degrees = np.sum(adj_matrix, axis=0)   # Sum along columns for in-degrees
    
    # Convert lagrange multipliers to arrays more efficiently using array operations
    lagrange_array = np.array(lagrange_multipliers)
    out_fitness = lagrange_array[:, 0]  # First column contains out-fitnesses
    in_fitness = lagrange_array[:, 1]   # Second column contains in-fitnesses
    
    # Compute hamiltonian using array multiplication and sum
    hamiltonian = np.sum(out_degrees * out_fitness) + np.sum(in_degrees * in_fitness)
    
    return hamiltonian

@jit(nopython=True)
def undirected_configuration_model(adj_matrix, lagrange_multipliers, degrees=None):
    """Optimized version that can use precomputed degrees if provided."""
    n = adj_matrix.shape[0]
    
    if degrees is None:
        # Calculate degrees if not provided
        degrees = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                degrees[i] += adj_matrix[i, j]
    
    # Dot product of degrees and multipliers
    result = 0.0
    for i in range(n):
        result += degrees[i] * lagrange_multipliers[i]
    return result