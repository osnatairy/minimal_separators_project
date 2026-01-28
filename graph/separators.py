
from __future__ import annotations

import time
import multiprocessing
from typing import Optional, List

import networkx as nx

# your graph transform
from minimal_separators_project.graph.transforms import relabel_to_ints

# your enumeration package/module
import enum_algorithms




def decode_separators(seps_int, id_to_name):
    """Map integer separators back to original node names."""
    return [sorted(id_to_name[i] for i in S) for S in seps_int]



def run_enumerator(H_int: nx.Graph, s: int, t: int, which="SmallMinimalSeps", K=6, limit_seconds=120):
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
        seps, _stats, _total_time = enum_algorithms.RankedEnumSeps(H_int, event)

    # Normalize to sorted lists of ints
    return [sorted(list(S)) for S in seps]



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
    seps_int = run_enumerator(H_int, s, t, which=which, K=K)

    # 4) Back to names
    Z_named = decode_separators(seps_int, id_to_name)

    # Unique + stable order
    Z_unique = sorted({tuple(z) for z in Z_named})


    return [list(z) for z in Z_unique]
