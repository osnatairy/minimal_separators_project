

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
    if nodes is None:
        return set()
    # If X is a single string and not a collection, turn it into a set
    if isinstance(nodes, str):
        return {nodes}
    # If it is neither a string nor a collection, it is assumed to be a singleton.
    elif not isinstance(nodes, (set, list, tuple)):
        return {nodes}
    else: # Guarantees it is a set (even if passed as a list or tuple)
        return set(nodes)



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
