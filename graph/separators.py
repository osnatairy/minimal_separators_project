
from __future__ import annotations

import time
import multiprocessing

import networkx as nx
from collections import deque
from typing import Iterable, List, Optional, Tuple, Set, Any

from torchgen.api.cpp import return_names

# your graph transform
from graph.transforms import relabel_to_ints

# your enumeration package/module
import enum_algorithms




def decode_separators(seps_int, id_to_name):
    """Map integer separators back to original node names."""
    return [sorted(id_to_name[i] for i in S) for S in seps_int]



def run_enumerator(H_int: nx.Graph, s: int, t: int, which="smallminimalseps", K=25, limit_seconds=120):
    """
    Run your minimal-separator enumerator. Returns a list[list[int]].
    which: "RankedEnumSeps" (RankedEnumSeps) or "small" (SmallMinimalSeps, uses K)
    """
    H_int = H_int.copy()
    event = multiprocessing.Event()
    enum_algorithms.start_time = time.time()

    if which.lower() in ("small", "smallmin", "smallminimalseps"):
        seps, _stats, _total_time = enum_algorithms.SmallMinimalSeps(H_int, K or 10**9, event)
    else:
        H_int.graph['weighted'] = 0
        seps, _stats, _total_time = enum_algorithms.RankedEnumSeps(H_int, event)

    # Normalize to sorted lists of ints
    return [sorted(list(S)) for S in seps],_total_time


def get_closest_to_s_t(G: nx.Graph,s,t):
    #s = G.graph['st'][0]
    G.graph['st'] = (s, t)
    close_to_s =  enum_algorithms.MinimalstSepCloseToA(G, {s})

    H = G.copy()
    #s, t = H.graph['st']
    H.graph['st'] = (t, s)
    close_to_t = enum_algorithms.MinimalstSepCloseToA(H, {t})

    return close_to_s, close_to_t

def find_seperators(
    H: nx.Graph,
    s: str,
    t: str,
    which: str = "RankedEnumSeps",
    K: int = None
):
    # 3) Relabel to ints + enumerate
    H_int, name_to_id, id_to_name, s, t = relabel_to_ints(H, s, t)

    #seps_int = alg.start_algorithm(H_int)
    seps_int,time = run_enumerator(H_int, s, t, which=which, K=K)

    if len(seps_int) >= 1 and len(seps_int[0]) == 0:
        return []

    # 4) Back to names
    Z_named = decode_separators(seps_int, id_to_name)

    # Unique + stable order
    Z_unique = sorted({tuple(z) for z in Z_named})


    return [list(z) for z in Z_unique]





def parents_of(G: nx.DiGraph, x: Any) -> Set[Any]:
    """Return parents of x in a DAG (nodes with an edge -> x)."""
    return set(G.predecessors(x))


def Z_contains_parent_of_X(G: nx.DiGraph, Z: Iterable[Any], X: Any) -> Tuple[bool, Set[Any]]:
    """Check whether Z contains any parent of X."""
    Zset = set(Z)
    px = parents_of(G, X)
    hit = Zset & px
    return (len(hit) > 0, hit)


def _is_collider(G: nx.DiGraph, u: Any, v: Any, w: Any) -> bool:
    """
    v is a collider on path u - v - w iff u -> v and w -> v.
    (This is the standard collider definition in a DAG.)
    """
    return G.has_edge(u, v) and G.has_edge(w, v)


def _path_is_open_given_empty(G: nx.DiGraph, path: List[Any]) -> bool:
    """
    For d-separation with NO conditioning set (empty set),
    a path is open iff it contains NO colliders.
    """
    if len(path) <= 2:
        return True
    for i in range(1, len(path) - 1):
        if _is_collider(G, path[i - 1], path[i], path[i + 1]):
            return False
    return True


def shortest_open_path_Z_to_X(
        G: nx.DiGraph,
        Z: Iterable[Any],
        X: Any,
        max_len: int = 3
) -> Optional[List[Any]]:
    """
    Find a shortest *open* (d-connecting given empty set) path from any z in Z to X,
    of length <= max_len (length = number of edges).

    We search in the underlying undirected graph but test openness using directions.
    Returns the node-list path [z, ..., X] if found, else None.
    """
    Zset = set(Z)
    if X in Zset:
        return [X]

    # Underlying undirected adjacency for path search
    U = G.to_undirected()

    # Multi-source BFS over paths (explicitly tracking paths, limited by max_len)
    q = deque()
    seen = set()  # (node, path_length, prev_node) - light pruning

    for z in Zset:
        if z not in U:
            continue
        q.append([z])

    while q:
        path = q.popleft()
        v = path[-1]
        if len(path) - 1 > max_len:
            continue

        if v == X:
            if _path_is_open_given_empty(G, path):
                return path
            else:
                continue

        # Expand
        for nbr in U.neighbors(v):
            if nbr in path:  # keep simple paths
                continue
            new_path = path + [nbr]
            if len(new_path) - 1 > max_len:
                continue

            # Early pruning: if we've already been at (nbr) with same previous node and length, skip
            key = (nbr, len(new_path), v)
            if key in seen:
                continue
            seen.add(key)
            q.append(new_path)

    return None


def has_short_open_path_Z_to_X(
        G: nx.DiGraph,
        Z: Iterable[Any],
        X: Any,
        max_len: int = 3
) -> Tuple[bool, Optional[List[Any]]]:
    """Convenience wrapper returning boolean + witness path (if exists)."""
    p = shortest_open_path_Z_to_X(G, Z, X, max_len=max_len)
    return (p is not None, p)



#
# # -------------------------
# # Example usage:
# # -------------------------
# if __name__ == "__main__":
#     # Build a tiny DAG
#     G = nx.DiGraph()
#     G.add_edges_from([
#         ("P", "X"),  # P is a parent of X
#         ("Z", "W"),
#         ("W", "X"),  # Z -> W -> X gives an open path (no collider)
#         ("A", "M"),
#         ("B", "M"),
#         ("M", "X"),  # A -> M <- B is a collider at M on path A-M-B (blocked if empty)
#     ])
#
#     X = "X"
#     Z1 = {"P", "Z",
#           "A"}  # includes parent P; Z has open short path; A has no open path to X within len 3 without hitting collider?
#
#     has_parent, which = Z_contains_parent_of_X(G, Z1, X)
#     print("Z1 contains parent of X?", has_parent, "parents in Z1:", which)
#
#     ok, witness = has_short_open_path_Z_to_X(G, Z1, X, max_len=3)
#     print("Has short open path Z1 ⇝ X (len<=3)?", ok, "witness path:", witness)
