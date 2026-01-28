import networkx as nx
from collections import deque

from minimal_separators_project.graph.helpers import normalize_nodes_to_set



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
