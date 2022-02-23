import torch
import dgl
import networkx as nx
import math
import numpy as np

#Functions for generating example graphs

def generate_cycle(n):
    source_l = list(np.arange(0,n))
    #print(source_l)
    dest_l = list(np.arange(1,n))
    dest_l.append(0)
    #print(dest_l)
    g = dgl.graph((torch.tensor(source_l), torch.tensor(dest_l)))
    #g.add_edges(torch.tensor(dest_l), torch.tensor(source_l))
    return g

def generate_csl(n,s):
    g = generate_cycle(n)
    source_l = list(np.arange(0,n))
    dest_l = [(el + s)%n for el in source_l]
    g.add_edges(torch.tensor(source_l), torch.tensor(dest_l))
    #g.add_edges(torch.tensor(dest_l), torch.tensor(source_l))
    return g

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

#Counting the homomorphism numbers hom(C_k, G^v) for k=3 -> 10

def cycle_count_node_hom(graph, nodenum=0):
    A = dgl.to_bidirected(graph).adjacency_matrix().to_dense()
    cycle_counts_node = torch.zeros(8)
    for j in range(3,11):
        cycles = torch.matrix_power(A,j)
        cycle_counts_node[j-3] = cycles[nodenum,nodenum]
    return cycle_counts_node.long()

#Counting the subgraph isomorphism numbers sub(C_k, G^v) for k=3 -> 10

def cycle_count_node_sub(graph, nodenum=0):
    cycle_counts = torch.zeros((graph.number_of_nodes(),8))
    nx_graph = graph.to_networkx()
    nx_graph = nx_graph.to_undirected()
    neighbor_list = list(nx.all_neighbors(nx_graph,nodenum))
    #print(neighbor_list)
    cycle_counts_node = torch.zeros(8)
    for n in neighbor_list:
        #print(n)
        for j in range(3,11):
            #print(j)
            path_max = len(list(nx.algorithms.simple_paths.all_simple_paths(nx_graph, nodenum, n, cutoff=j-1)))
            #print(list(nx.algorithms.simple_paths.all_simple_paths(nx_graph, 0, n, cutoff=j-1)))
            path_min = len(list(nx.algorithms.simple_paths.all_simple_paths(nx_graph, nodenum, n, cutoff=j-2)))
            #print(list(nx.algorithms.simple_paths.all_simple_paths(nx_graph, 0, n, cutoff=j-2)))
            path_number = path_max-path_min
            #print("There are ", path_number, "paths of length ", j-1, "from node 0 to node ", n)
            cycle_counts_node[j-3] = cycle_counts_node[j-3] + path_number
    return cycle_counts_node.long()

#Counting the subgraph isomorphisms numbers sub(C_4^01, G^uv) for k=3 -> 10 w.r.t to an edge in the graph

def number_of_squares_edge_sub(graph, source, dest):
    nx_graph = graph.to_networkx()
    nx_graph = nx_graph.to_undirected()
    source_neighbor_list = list(nx.all_neighbors(nx_graph, source))
    dest_neighbor_list = list(nx.all_neighbors(nx_graph, dest))
    count = 0 
    for a in source_neighbor_list:
        for b in dest_neighbor_list:          
            if nx_graph.has_edge(a,b):
                if a != dest and b!= source:
                    #print(a,b)
                    count += 1
    return count

#Counting the subgraph isomorphisms numbers hom(C_4^01, G^uv) for k=3 -> 10 w.r.t to an edge in the graph

def number_of_squares_edge_hom(graph, source, dest):
    nx_graph = graph.to_networkx()
    nx_graph = nx_graph.to_undirected()
    source_neighbor_list = list(nx.all_neighbors(nx_graph, source))
    dest_neighbor_list = list(nx.all_neighbors(nx_graph, dest))
    count = 0 
    for a in source_neighbor_list:
        for b in dest_neighbor_list:          
            if nx_graph.has_edge(a,b):
                    #print(a,b)
                    count += 1
    return count




