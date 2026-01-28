import networkx as nx
import itertools

def get_minimal_separators(G, u, v):
    """
    Finds all minimal s-t separators for vertices u and v.
    A set S is a minimal u-v separator if u and v are in different connected components of G - S,
    and no proper subset of S has this property.
    """
    separators = []
    nodes = set(G.nodes())
    candidates = list(nodes - {u, v})

    # Iterate through all possible subset sizes
    # Optimization: minimal separator size is at least connectivity(u,v)
    # But for small graphs, full enumeration is fine.

    for r in range(len(candidates) + 1):
        for S in itertools.combinations(candidates, r):
            S_set = set(S)

            # Check blockage
            G_sub = G.copy()
            G_sub.remove_nodes_from(S_set)
            if nx.has_path(G_sub, u, v):
                continue

            # Check minimality
            is_minimal = True
            for x in S_set:
                S_prime = S_set - {x}
                G_prime = G.copy()
                G_prime.remove_nodes_from(S_prime)
                if not nx.has_path(G_prime, u, v):
                    # Removing x still blocks -> S was not minimal because S_prime is a subset that blocks
                    is_minimal = False
                    break

            if is_minimal:
                separators.append(S_set)

    return separators
