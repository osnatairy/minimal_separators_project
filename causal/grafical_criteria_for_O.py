from __future__ import annotations
from typing import Any, Iterable, List, Set, FrozenSet
import networkx as nx


########################################
######## FIND THE CRITERIA FOR O(X,Y) FROM HENKEL PAPER
########################################

Node = Any


def _as_set(v: Iterable[Node] | Node) -> Set[Node]:
    if isinstance(v, (set, frozenset)):
        return set(v)
    # treat strings as atomic nodes
    if isinstance(v, str):
        return {v}
    try:
        return set(v)  # iterable
    except TypeError:
        return {v}


def descendants(G: nx.DiGraph, nodes: Iterable[Node]) -> Set[Node]:
    """All descendants (reachable by directed paths) of any node in `nodes`."""
    nodes_set = _as_set(nodes)
    out: Set[Node] = set()
    for n in nodes_set:
        out |= nx.descendants(G, n)
    return out


def parents(G: nx.DiGraph, nodes: Iterable[Node]) -> Set[Node]:
    """All parents (immediate predecessors) of nodes in `nodes`."""
    nodes_set = _as_set(nodes)
    out: Set[Node] = set()
    for n in nodes_set:
        out |= set(G.predecessors(n))
    return out


def find_proper_causal_paths(G: nx.DiGraph, X: Iterable[Node], Y: Iterable[Node]) -> List[List[Node]]:
    """
    Find all 'proper causal paths' from X to Y:
      - directed paths following edge direction (->) only
      - start node is in X
      - no other node on the path (except the first) is in X
    Returns a list of node-lists (paths).
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Expected a DAG (directed acyclic graph).")

    Xs = _as_set(X)
    Ys = _as_set(Y)

    all_paths: List[List[Node]] = []

    # Enumerate directed simple paths from each x in X to each y in Y
    for x in Xs:
        for y in Ys:
            # networkx.all_simple_paths respects direction in DiGraph
            for path in nx.all_simple_paths(G, source=x, target=y):
                # Proper: only first node is allowed to be in X
                if any(node in Xs for node in path[1:]):
                    continue
                all_paths.append(path)

    return all_paths


def find_optimal_adjustment_set(G: nx.DiGraph, X: Iterable[Node], Y: Iterable[Node]) -> Set[Node]:
    """
    Implements the pseudo-code provided:

    1) cn = nodes on proper causal paths from X to Y, excluding X
    2) forbidden_nodes = descendants(cn) ∪ X
    3) parents_of_cn = parents(cn)
    4) O = parents_of_cn \ forbidden_nodes
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Expected a DAG (directed acyclic graph).")

    Xs = _as_set(X)
    Ys = _as_set(Y)

    # 1) causal nodes cn
    causal_paths = find_proper_causal_paths(G, Xs, Ys)
    cn: Set[Node] = set()
    for path in causal_paths:
        cn.update(path)
    cn -= Xs  # exclude X

    # 2) forbidden nodes
    forbidden_nodes = descendants(G, cn) | Xs

    # 3) parents of cn
    parents_of_cn = parents(G, cn)

    # 4) optimal set O
    O = parents_of_cn - forbidden_nodes
    return O


# -----------------------
# Example run
# -----------------------
if __name__ == "__main__":
    # Example 3.5-ish graph:
    # A -> X -> Y
    # B -> X, B -> Y
    # C -> Y
    # X -> D
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "X"),
        ("B", "X"),
        ("B", "Y"),
        ("C", "Y"),
        ("X", "Y"),
        ("X", "D"),
    ])

    X = {"X"}
    Y = {"Y"}
    O = find_optimal_adjustment_set(G, X, Y)

    print("Optimal adjustment set O:", O)
