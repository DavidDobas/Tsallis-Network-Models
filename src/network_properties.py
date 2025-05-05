import numpy as np

def annd(graph, mode="all"):
    if mode=="all":
        new_graph = graph.as_undirected()
    else:
        new_graph = graph
    annd_array = np.zeros(len(new_graph.vs))
    for i in range(len(new_graph.vs)):
        vertex = new_graph.vs[i]
        neighbors = vertex.neighbors(mode=mode)
        annd_array[i] = np.nanmean([neighbor.degree(mode=mode, loops=False) for neighbor in neighbors if neighbor != vertex])
    return annd_array

def clustering_coeff(graph):
    result = graph.transitivity_local_undirected()
    return result

def average_property_over_degree(degrees, property_values, max_degree=None):
    if not max_degree:
        max_degree = np.max(degrees)
    averaged_values = np.zeros(max_degree + 1)
    degree_hist = np.zeros(max_degree + 1)
    for degree_seq, property_value_seq in zip(degrees, property_values):
        for degree, property_value in zip(degree_seq, property_value_seq):
            averaged_values[degree] += property_value
            degree_hist[degree] += 1
    averaged_values[degree_hist == 0] = np.nan
    # degree_hist[degree_hist==0] = 1
    averaged_values = averaged_values/degree_hist
    return averaged_values, degree_hist