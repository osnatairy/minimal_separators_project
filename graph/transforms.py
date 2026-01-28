import random
import networkx as nx
from typing import Tuple, List, Optional

from minimal_separators_project.graph.helpers import normalize_nodes_to_set



def singleton_reduction(H: nx.Graph, X, Y) -> Tuple[nx.Graph, str, str]:
    r"""
    Reduce group-separation (X,Y) to a single pair (x,y) while preserving separators:
    connect a chosen x to N_H(X) and a chosen y to N_H(Y), then delete X\{x}, Y\{y}.
    """
    Xs = normalize_nodes_to_set(X)
    Ys = normalize_nodes_to_set(Y)
    assert len(Xs) > 0 and len(Ys) > 0, "X and Y must be non-empty"

    x = next(iter(Xs))
    y = next(iter(Ys))

    H2 = H.copy()

    # Connect x to neighbors of all X
    N_X = set()
    for u in Xs:
        N_X |= set(H2.neighbors(u))
    N_X -= Xs
    for v in N_X:
        H2.add_edge(x, v)

    # Connect y to neighbors of all Y
    N_Y = set()
    for u in Ys:
        N_Y |= set(H2.neighbors(u))
    N_Y -= Ys
    for v in N_Y:
        H2.add_edge(y, v)

    # Remove other X/Y nodes
    H2.remove_nodes_from((Xs - {x}) | (Ys - {y}))
    return H2, x, y



def relabel_to_ints(H: nx.Graph, x, y):
    """Relabel nodes to integers for your enumerators; keep mappings and terminals."""
    id_map = {node: i for i, node in enumerate(H.nodes())}
    H_int = nx.relabel_nodes(H, id_map, copy=True)
    s, t = id_map[x], id_map[y]
    H_int.graph['st'] = (s, t)            # your scripts expect this
    return H_int, id_map, {i: n for n, i in id_map.items()}, s, t



# remove all the edges from x to any node in cv_set.
def remove_edges_from_x_to_cv(G, x, cv_set):

    if x not in G.nodes():
        raise ValueError(f"Node {x} does not exist in the graph")

    #copy the original graph
    G_new = G.copy()

    #Convert cv_set to a set if it is not already a set
    if not isinstance(cv_set, set):
        cv_set = set(cv_set)

    # Find all arcs from X to the group CV
    edges_to_remove = []
    for neighbor in G.neighbors(x):
        if neighbor in cv_set:
            edges_to_remove.append((x, neighbor))

    #remove the edges
    G_new.remove_edges_from(edges_to_remove)

    return G_new, edges_to_remove


# Node Group Extension - X
# Remove all edges from any node in X to any node in cv_set.
#Creates the proper back door graph G' by removing edges X → cv_set.
# Args:
#   G: NetworkX DiGraph - the original graph
#   X: a single vertex or a collection of vertices (set, list, tuple, or single vertex)
#   cv_set: the set of causal vertices
#
# Returns:
#   tuple: (G_new, edges_removed)
#       G_new: the new graph without the edges
#       edges_removed: the list of edges removed
# O(|X| × avg_degree)
def remove_edges_from_X_to_cv(G, X, cv_set):

    # Convert X to a set
    X_set = normalize_nodes_to_set(X)

    # Integrity check
    missing_nodes = X_set - set(G.nodes())
    if missing_nodes:
        raise ValueError(f"Nodes {missing_nodes} do not exist in the graph")

    # If X contains a single vertex, the original function is used for efficiency
    if len(X_set) == 1:
        x = next(iter(X_set))
        return remove_edges_from_x_to_cv(G, x, cv_set)

    # Copy of the original graph
    G_new = G.copy()

    # Convert cv_set to a set (if not already)
    if not isinstance(cv_set, set):
        cv_set = set(cv_set)

    # Finding all arcs from X to cv_set efficiently
    edges_to_remove = []

    # We go through each vertex in X and check its neighbors.
    for x in X_set:
        for neighbor in G.neighbors(x):
            if neighbor in cv_set:
                edges_to_remove.append((x, neighbor))

    # Removing all the arches at once (more effective than one by one)
    if edges_to_remove:
        G_new.remove_edges_from(edges_to_remove)

    return G_new, edges_to_remove


# A version optimized for large groups of X.
# uses a more efficient data structure for searching.
# O(|E|)
def remove_edges_from_X_to_cv_optimized(G, X, cv_set):

    # Convert X to a set
    X_set = normalize_nodes_to_set(X)

    # Integrity check
    missing_nodes = X_set - set(G.nodes())
    if missing_nodes:
        raise ValueError(f"Nodes {missing_nodes} do not exist in the graph")

    # Copy of the original graph
    G_new = G.copy()

    # Convert cv_set to a set (if not already)
    if not isinstance(cv_set, set):
        cv_set = set(cv_set)

    # If cv_set is empty, there is nothing to remove
    if not cv_set:
        return G_new, []

    # Finding arcs to remove efficiently
    # Instead of going over all neighbors, we only go over existing edge
    edges_to_remove = [
        (u, v) for u, v in G.edges()
        if u in X_set and v in cv_set
    ]

    # Removing the edges
    if edges_to_remove:
        G_new.remove_edges_from(edges_to_remove)

    return G_new, edges_to_remove



# A wrapper function that chooses the best method for remove edges from X to causal_vertices
# Creates a proper back door graph by removing edges from X to cv_set.
# Automatically chooses the most efficient method.
#
# Args:
#   G: NetworkX DiGraph
#   X: Source node(s)
#   cv_set: Set of causal nodes
#
# Returns:
#   tuple: (G_prime, removed_edges)
def create_proper_backdoor_graph(G, X, cv_set):

    # Convert X to a set
    X_set = normalize_nodes_to_set(X)

    # Choosing the method according to the size of the groups and the graph
    total_edges = G.number_of_edges()
    X_size = len(X_set)

    # If X is small relative to the graph, it is better to go over the neighbors
    if X_size * 10 < total_edges:
        return remove_edges_from_X_to_cv(G, X_set, cv_set)
    else:
        # Otherwise, it's better to go through all the edges .
        return remove_edges_from_X_to_cv_optimized(G, X_set, cv_set)




#create moral graph of G - meaning add an edge between each vertices pair with shared node (as descendant)
def create_moral_graph(dag):
    # Unintentional copy of existing edges
    moral_graph = nx.Graph()
    moral_graph.add_nodes_from(dag.nodes)

    # We will add all existing edges in an unintentional manner.
    for u, v in dag.edges:
        moral_graph.add_edge(u, v)

    # We will add edges between each pair of parents of each node
    for child in dag.nodes:
        parents = list(dag.predecessors(child))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral_graph.add_edge(parents[i], parents[j])

    return moral_graph



#force all the nodes in I to be in the adjustment set
#Connects each node in I to one node from X and one node from Y, according to the selection strategy.
#graph: NetworkX graph (can be directed or undirected)
#I: List of nodes in set I
#X: List of nodes in set X
#Y: List of nodes in set Y
#pick_strategy: How to choose the node from X and Y ('first', 'random', 'central')
def connect_I_to_XY(graph, I, X, Y, pick_strategy='first'):

    def pick_node(nodes):
        if pick_strategy == 'first':
            return nodes[0]
        elif pick_strategy == 'random':
            return random.choice(nodes)
        elif pick_strategy == 'central':
            # Assume the graph is connected; we will take the node with the highest centrality (for a small graph)
            centrality = nx.degree_centrality(graph)
            return max(nodes, key=lambda n: centrality.get(n, 0))
        else:
            raise ValueError("Unknown pick strategy")

    x_target = pick_node(X)
    y_target = pick_node(Y)

    for node in I:
        graph.add_edge(node, x_target)
        graph.add_edge(node, y_target)

    return graph

#for the forbidden vertices, create clique with its neighbors (Saturation)
def create_clique_on_neighbors(graph, nodes):

    edges_to_add = set()

    # Collect all the edges required to form a clique
    for node in nodes:
        neighbors = list(graph.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                edge = tuple(sorted([neighbors[i], neighbors[j]]))
                edges_to_add.add(edge)

    # Adding all edges
    graph.add_edges_from(edges_to_add)


# using to build graph from the read DAT file. (after export the data from it)
def build_G_from_mapped_edges(
    mapped_edges_unique,
    id_to_name=None,
    st=None,           # e.g. ("A","T") or ("V1","V7")
    ensure_all_nodes=True
):
    """
    Build a networkx.DiGraph from mapped edges and attach useful metadata.

    mapped_edges_unique: list[(src_id, dst_id)]
    id_to_name: optional dict[id -> original_name] for labels/visualization
    st: optional tuple (s, t) in the *mapped-id* space
    """
    G = nx.DiGraph()

    # 1) Add edges (this also adds endpoint nodes)
    G.add_edges_from(mapped_edges_unique)

    # 2) If you want nodes that have no edges to still exist in the graph:
    if ensure_all_nodes and id_to_name is not None:
        G.add_nodes_from(id_to_name.keys())

    # 3) Attach mapping as graph metadata (handy later)
    if id_to_name is not None:
        G.graph["id_to_name"] = dict(id_to_name)
        # If you also have name_to_id you can store it similarly:
        # G.graph["name_to_id"] = dict(name_to_id)

    # 4) Set s,t for your downstream pipeline
    if st is not None:
        s, t = st
        if s not in G or t not in G:
            raise ValueError(f"st nodes not in G: missing {[n for n in (s,t) if n not in G]}")
        G.graph["st"] = (s, t)

    # 5) Optional sanity check
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph is not a DAG (contains a directed cycle).")

    return G