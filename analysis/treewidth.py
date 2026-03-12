import networkx as nx
from itertools import combinations

def moral_graph_from_bn(bn) -> nx.Graph:
    """Build the moral graph of bn.g (DiGraph) as an undirected nx.Graph."""
    G = nx.Graph()
    G.add_nodes_from(bn.g.nodes())

    # 1) Add undirected version of all directed edges
    for u, v in bn.g.edges():
        G.add_edge(u, v)

    # 2) "Marry" co-parents: connect all pairs of parents of each node
    for child in bn.g.nodes():
        parents = list(bn.g.predecessors(child))
        for u, v in combinations(parents, 2):
            G.add_edge(u, v)

    return G

def min_fill_order(G: nx.Graph):
    """Greedy Min-Fill elimination order on an undirected graph."""
    H = G.copy()
    order = []
    while H.nodes:
        best_x = None
        best_fill = None

        for x in H.nodes:
            neigh = list(H.neighbors(x))
            # number of missing edges among neighbors
            missing = 0
            for u, v in combinations(neigh, 2):
                if not H.has_edge(u, v):
                    missing += 1
            if best_fill is None or missing < best_fill:
                best_fill = missing
                best_x = x

        # eliminate best_x: add fill edges then remove node
        neigh = list(H.neighbors(best_x))
        for u, v in combinations(neigh, 2):
            H.add_edge(u, v)
        H.remove_node(best_x)
        order.append(best_x)

    return order

def induced_width_by_order(G: nx.Graph, order):
    """Simulate VE fill-in on G using 'order'. Returns induced width = max |N(x)|."""
    H = G.copy()
    width = 0

    for x in order:
        if x not in H:
            continue
        neigh = list(H.neighbors(x))
        width = max(width, len(neigh))

        # fill-in: make neighbors a clique
        for u, v in combinations(neigh, 2):
            H.add_edge(u, v)

        H.remove_node(x)

    return width

def treewidth_upper_bound_ve(bn, heuristic="minfill"):
    """Compute a VE-style upper bound on treewidth of BN (no factors)."""
    Gm = moral_graph_from_bn(bn)

    if heuristic == "minfill":
        order = min_fill_order(Gm)
    elif heuristic == "alphabetic":
        order = sorted(Gm.nodes(), key=str)
    else:
        raise ValueError("heuristic must be 'minfill' or 'alphabetic'")

    w = induced_width_by_order(Gm, order)
    return w, order