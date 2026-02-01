import networkx as nx
from collections import deque
from typing import Iterable, Hashable, Set
"""
  Computes PCP(X,Y): all vertices on a path X→Y, excluding the vertices in X.
  - In a directed graph: v is included if v is reachable from X and can reach Y.

  Returns a list (not necessarily sorted).
  """
def find_causal_vertices_sets_v1(G, X, Y):

    all_descendants_X = set()
    for x in X:
        all_descendants_X.update(nx.descendants(G, x))

    all_ancestors_Y = set()
    for y in Y:
        all_ancestors_Y.update(nx.ancestors(G, y))

    causal = set(all_descendants_X) & set(all_ancestors_Y)
    # Include Y, and exclude X
    return (causal | set(Y)) - set(X)

"""
    Compute the forbidden set in a DAG:
    all descendants of the PCP set, including the PCP nodes themselves.
 The input: DAG, PCP (-set of nodes)
 The output: The forbidden set.
    """
def forbidden_set_from_pcp(DG, pcp):
    sources = [s for s in pcp if s in DG]
    visited = set(sources)
    q = deque(sources)

    while q:
        u = q.popleft()
        for v in DG.successors(u):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited


def test():
    DG = nx.DiGraph()
    DG.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 2), (2, 4)])
    causal_vertices = find_causal_vertices_sets_v1(DG, X=[0], Y=[4])
    print(sorted(causal_vertices))
    # Output: [1, 2, 3, 4]   (Y=4 is included, X=0 is excluded)

    print(forbidden_set_from_pcp(DG, causal_vertices))

    DG = nx.DiGraph()
    DG.add_edges_from([
        ("a","b"), ("b","c"), ("a","d"), ("d","e"), ("x","y")
    ])

    pcp = {"b"}
    print(forbidden_set_from_pcp(DG, pcp))
    # Output: {'b', 'c'}

test()