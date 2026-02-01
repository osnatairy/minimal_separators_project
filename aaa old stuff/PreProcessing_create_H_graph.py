import networkx as nx
from collections import deque
import random
import matplotlib.pyplot as plt
import utils_ms as utils

#Converts input of nodes to a set, regardless of the input format.
# Args:
#   nodes: A single node or collection of nodes (set, list, tuple, string, or single node)
#
# Returns:
#   set: A set of nodes
#
# Examples:
# normalize_nodes_to_set(1) -> {1}
# normalize_nodes_to_set("A") -> {"A"}
# normalize_nodes_to_set([1, 2, 3]) -> {1, 2, 3}
# normalize_nodes_to_set({1, 2}) -> {1, 2}
# normalize_nodes_to_set("AB") -> {"AB"} # A single node named AB
def normalize_nodes_to_set(nodes):
    # If X is a single string and not a collection, turn it into a set
    if isinstance(nodes, str):
        return {nodes}
    # If it is neither a string nor a collection, it is assumed to be a singleton.
    elif not isinstance(nodes, (set, list, tuple)):
        return {nodes}
    else: # Guarantees it is a set (even if passed as a list or tuple)
        return set(nodes)

#find all the causal vertices in the graph
#those are the vertices with a path from x to Y and reverse.
#where the value of x and y is are specific nodes
def find_causal_vertices(G, x, y):
    descendants_x = set(nx.descendants(G, x))  # Anyone accessible from x
    ancestors_y = set(nx.ancestors(G, y))  # Anyone who can reach y

    # The intersection = vertices that are between x and y
    causal = descendants_x & ancestors_y

    # If there is a direct arc from x to y, add y
    if G.has_edge(x, y):
        causal.add(y)

    return causal

# Finds causal vertices between node sets X and Y.
# Efficient when |X| and |Y| are relatively small.
# will take O(|X| + |Y|) call functions, may cause big temp groups.
def find_causal_vertices_sets_v1(G, X, Y):

    # Unites all descendants of all nodes in X
    all_descendants_X = set()
    for x in X:
        all_descendants_X.update(nx.descendants(G, x))

    # Unites all predecessors of all nodes in Y
    all_ancestors_Y = set()
    for y in Y:
        all_ancestors_Y.update(nx.ancestors(G, y))

    # Intersection = the nodes that are between X and Y
    causal = all_descendants_X & all_ancestors_Y

    return causal


# Finds causal vertices between sets of nodes X and Y.
# More efficient for large graphs - uses one-time BFS search.
# O(|V| + |E|) for the BFS
def find_causal_vertices_sets_v2(G, X, Y):

    # Finding all nodes reachable from X
    reachable_from_X = set(X)  # Including X itself
    queue = deque(X)

    while queue:
        current = queue.popleft()
        for successor in G.successors(current):
            if successor not in reachable_from_X:
                reachable_from_X.add(successor)
                queue.append(successor)

    # Finding all nodes that can reach Y
    can_reach_Y = set(Y)  # Including Y itself
    queue = deque(Y)

    # Creating an inverse graph for backward search
    G_reversed = G.reverse()

    while queue:
        current = queue.popleft()
        for predecessor in G_reversed.successors(current):  # predecessors in the original graph
            if predecessor not in can_reach_Y:
                can_reach_Y.add(predecessor)
                queue.append(predecessor)

    # Intersection = the nodes between X and Y
    # Removing X and Y themselves from the result (as we only want the nodes in the middle)
    causal = (reachable_from_X & can_reach_Y) - set(X) - set(Y)

    return causal


# Runs the appropriate function to find causal vertices according to the input type
# A hybrid approach that chooses the better method based on the size of the groups.
# If X and Y are single nodes, uses the original function.
# Args:
#         G: NetworkX DiGraph
#         X: A single node or a collection of nodes (set, list, tuple, or single string)
#         Y: A single node or a collection of nodes (set, list, tuple, or single string)
# Returns:
#         set: the set of causal nodes between X and Y
def find_causal_vertices_sets_optimized(G, X, Y):

    # Convert X to a set
    X = normalize_nodes_to_set(X)

    # Same for Y
    Y = normalize_nodes_to_set(Y)

    # Checking if X and Y are unique nodes
    if len(X) == 1 and len(Y) == 1:
        x = next(iter(X))  # Removes the single braid from X
        y = next(iter(Y))  # Removes the single braid from Y
        return find_causal_vertices(G, x, y)

    # If the groups are small, use "find_causal_vertices_sets_v1"
    elif len(X) <= 5 and len(Y) <= 5:
        return find_causal_vertices_sets_v1(G, X, Y)
    else: # use find_causal_vertices_sets_v2
        return find_causal_vertices_sets_v2(G, X, Y)


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

#return the induced graph over I u X u Y - they are the nodes.
def get_induced_subgraph(graph, nodes):

    # Checking that all nodes exist in the graph
    missing_nodes = set(nodes) - set(graph.nodes())
    if missing_nodes:
        raise ValueError(f"The following nodes do not exist in the graph: {missing_nodes}")

    return graph.subgraph(nodes).copy()


# Creates an induced subgraph for the set of nodes V' = X ∪ Y ∪ I
#
# The induced subgraph includes all original edges that connect nodes within V' only.
#
# Parameters:
#   G: The original graph (dictionary of dictionaries)
#   X: A list/set of nodes or a single node
#   Y: A list/set of nodes or a single node
#   I: A list/set of nodes or a single node (MUST HAVE NODES)
#
# Returns:
# induced subgraph - a dictionary containing only the nodes from V' and the internal edges between them
def create_induced_subgraph(G, X, Y, I):

    # Convert X to a set
    X_set = normalize_nodes_to_set(X)
    Y_set = normalize_nodes_to_set(Y)
    I_set = normalize_nodes_to_set(I)

    # Creating V' = X ∪ Y ∪ Z
    V_prime = X_set.union(Y_set).union(I_set)

    # Creating the induced subgraph
    induced_subgraph = {}

    # Passing each intersection in a 'V'
    for node in V_prime:
        if node in G:  # Checking that the node exists in the original graph
            induced_subgraph[node] = {}

            # Only add edges to nodes that are also in the V'
            for neighbor, weight in G[node].items():
                if neighbor in V_prime:
                    induced_subgraph[node][neighbor] = weight

    return induced_subgraph


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


if __name__ == "__main__":
    #creat or load a graph
    g = nx.DiGraph()
    s,t = [0,0]

    print("Start preprocessing...")
    causal_vertices = find_causal_vertices(g,s,t)




