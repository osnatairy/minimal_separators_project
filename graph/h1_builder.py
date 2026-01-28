import networkx as nx
from typing import List, Dict, Tuple, Any, Callable, Union

from minimal_separators_project.graph.helpers import normalize_nodes_to_set
from minimal_separators_project.graph.causal_vertices import find_causal_vertices_sets_optimized
from minimal_separators_project.graph.transforms import create_proper_backdoor_graph, create_moral_graph, connect_I_to_XY, create_clique_on_neighbors


def restrict_to_ancestors_dag(G: nx.DiGraph, keep: List) -> nx.DiGraph:
    """Keep only nodes that are ancestors of keep ∪ keep itself."""
    keep_set = normalize_nodes_to_set(keep)
    nodes = set(keep_set)
    for v in keep_set:
        nodes |= nx.ancestors(G, v)
    return G.subgraph(nodes).copy()


def forbidden_set(G: nx.DiGraph, X, Y) -> set:
    """
    Forbidden nodes for total-effect adjustment: descendants of nodes
    that lie on proper directed paths X->...->Y (and the path nodes themselves).
    (If you want the stricter/looser variant, tweak the union with `cp`.)
    """
    Xs = normalize_nodes_to_set(X)
    Ys = normalize_nodes_to_set(Y)
    cp = find_causal_vertices_sets_optimized(G, Xs, Ys)       # nodes between X and Y
    forb = set(cp)
    for v in cp:
        forb |= nx.descendants(G, v)
    # Don’t ever forbid X or Y themselves here
    return forb - Xs - Ys



def build_H1_from_DAG(G: nx.DiGraph, X, Y, R: List[str], I: List[str] = None) -> nx.Graph:
    r"""
    Implements (in the spirit of the paper) the transformation to H^1:
      1) Proper backdoor graph G' (delete first edge on X->Y paths).
      2) Restrict to ancestors of X∪Y∪I (optional).
      3) Moralize => undirected.
      4) Connect I to X and to Y.
      5) Saturate neighbors of: forbidden nodes, and unobserved nodes V\R; then remove those nodes.
    Returns an undirected graph H^1.
    """
    Xs = normalize_nodes_to_set(X)
    Ys = normalize_nodes_to_set(Y)
    Is = normalize_nodes_to_set(I) if I else set()

    # 1) Proper backdoor DAG
    cv = find_causal_vertices_sets_optimized(G, Xs, Ys)
    G_pbd, _removed = create_proper_backdoor_graph(G, Xs, cv)

    # 2) Ancestor restriction
    keep = set().union(Xs, Ys, Is)
    if keep:
        G_restricted = restrict_to_ancestors_dag(G_pbd, list(keep))
    else:
        G_restricted = G_pbd

    # 3) Moralize
    H = create_moral_graph(G_restricted)       # undirected

    # 4) Connect I to some X and some Y (only if I is nonempty)
    if Is:
        H = connect_I_to_XY(H, list(Is), list(Xs), list(Ys), pick_strategy='first')

    # 5) Saturate+remove forbidden and unobserved
    forb = forbidden_set(G, Xs, Ys)           # compute forb on the original DAG (safer)
    unobserved = set(H.nodes()) - set(R)

    to_remove = (forb | unobserved) & set(H.nodes())
    if to_remove:
        create_clique_on_neighbors(H, to_remove)   # saturation
        H.remove_nodes_from(to_remove)

    return H
