import igraph as ig
from numba import jit
import numpy as np
from numpy.typing import NDArray
Vector = NDArray[np.float64] 

@jit(nopython=True)
def undirected_configuration_model_ensemble(lagrange_multipliers: Vector, num_nodes: int, num_samples: int):
    # Create NxN matrix where each element is sum of corresponding lagrange multipliers
    theta_matrix = lagrange_multipliers[:, np.newaxis] + lagrange_multipliers[np.newaxis, :]
    
    # Calculate connection probabilities
    prob_matrix = 1 / (1 + np.exp(theta_matrix))
    
    # Generate samples
    samples = np.random.random((num_samples, num_nodes, num_nodes)) < prob_matrix
    
    # Make matrices symmetric and remove diagonal
    for i in range(num_samples):
        samples[i] = np.triu(samples[i], 1) + np.triu(samples[i], 1).T
        
    return samples

def undirected_configuration_model_ensemble_igraph(lagrange_multipliers: Vector, num_nodes: int, num_samples: int):
    # Generate ensemble of adjacency matrices
    adj_matrices = undirected_configuration_model_ensemble(lagrange_multipliers, num_nodes, num_samples)
    
    # Convert each adjacency matrix to an igraph Graph object
    graphs = [ig.Graph.Adjacency(adj_matrix, mode="undirected", loops=False) for adj_matrix in adj_matrices]
    
    return graphs

def erdos_renyi_ensemble(lagrange_multiplier: float, num_nodes: int, num_samples: int):
    # Generate ensemble of adjacency matrices
    adj_matrices = np.random.binomial(1, 1 / (1 + np.exp(lagrange_multiplier)), (num_samples, num_nodes, num_nodes))
    return adj_matrices

def erdos_renyi_ensemble_igraph(lagrange_multiplier: float, num_nodes: int, num_samples: int, directed: bool = True, loops: bool = True):
    # Generate ensemble of adjacency matrices
    adj_matrices = erdos_renyi_ensemble(lagrange_multiplier, num_nodes, num_samples)
    
    # Convert each adjacency matrix to an igraph Graph object
    graphs = [ig.Graph.Adjacency(adj_matrix.tolist(), mode="undirected" if not directed else "directed", loops="once" if loops else "ignore") for adj_matrix in adj_matrices]
    
    return graphs