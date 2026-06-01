from __future__ import annotations

from typing import List, Optional
import networkx as nx

# BN
from bn.bayesian_network import BN

# graph building + transforms
from graph.h1_builder import build_H1_from_DAG
from graph.transforms import singleton_reduction, relabel_to_ints

# separators decoding (and/or enumeration)
from graph.separators import decode_separators, run_enumerator, get_closest_to_s_t
from graph.helpers import normalize_nodes_to_set
# your algorithm runner

from enum_algorithms import gen_important_small_seps
# visualization utilities (if you want visualize_g)
import utils as utils



def preprocess_H(
    bn: BN,
    X: List[str],
    Y: List[str],
    R: List[str],
    I: List[str] = None
):
    """
    Load bn -> build H^1 -> singleton reduction -> enumerate minimal separators -> decode to names.
    Returns: list[list[str]] of adjustment sets Z (variable names).
    """

    G = bn.g               # directed
    [s, t] = [X, Y]
    G.graph['st'] = (s, t)
    #utils.visualize_g(G)
    # 1) H^1
    H1 = build_H1_from_DAG(G, X=X, Y=Y, R=R, I=I)

    H1.graph['weighted'] = 0
    H1.graph['st'] = (s, t)
    #utils.visualize_g(H1)

    H = H1
    x = G.graph['st'][0]
    y = G.graph['st'][1]

    # 2) Singleton reduction
    if len(X) > 1 or len(Y) > 1:
        H, x, y = singleton_reduction(H1, X, Y)


    return H, x, y



def find_adjustment_sets_for_pair(
    G: nx.DiGraph,
    X: str,
    Y: str,
    which: str,
    R: List[str],
    I: Optional[List[str]] = None,
    get_closest_seps: bool = False,
    K = None
) -> List[List[str]]:
    """
    Runs your exact snippet for one (X,Y) and returns Z_sets (decoded names).
    """
    G.graph['st'] = (X, Y)
    #utils.visualize_g(G)
    H1 = build_H1_from_DAG(G, X=X, Y=Y, R=R, I=I)
    [s, t] = [X, Y]
    H1.graph['st'] = (s, t)
    utils.visualize_g(H1)

    H = H1
    x = G.graph['st'][0]
    y = G.graph['st'][1]

    X = normalize_nodes_to_set(X)
    Y = normalize_nodes_to_set(Y)

    # 2) Singleton reduction
    if len(X) > 1 or len(Y) > 1:
        H, x, y = singleton_reduction(H1, X, Y)

    # 3) Relabel to ints + enumerate
    H_int, name_to_id, id_to_name, s, t = relabel_to_ints(H, x, y)
    print("start looking for seperators")
    #seps_int = alg.start_algorithm(H_int)
    seps_int,total_time = run_enumerator(H_int, s, t, which, K=len(G.nodes))

    if get_closest_seps:
        closest = get_closest_to_s_t(H_int,s,t)
        closest = decode_separators(closest,id_to_name)

    # 4) Back to names
    Z_named = decode_separators(seps_int, id_to_name)

    # Unique + stable order
    Z_unique = sorted({tuple(z) for z in Z_named})

    Z_unique = [Z for Z in Z_unique if len(Z) >= 1]
    if get_closest_seps:
        return H,[list(z) for z in Z_unique],total_time,closest
    return H,[list(z) for z in Z_unique],total_time


def find_closest_seps_for_pair(
    G: nx.DiGraph,
    X: str,
    Y: str,
    which: str,
    R: List[str],
    I: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Runs your exact snippet for one (X,Y) and returns Z_sets (decoded names).
    """
    G.graph['st'] = (X, Y)
    #utils.visualize_g(G)
    H1 = build_H1_from_DAG(G, X=X, Y=Y, R=R, I=I)
    [s, t] = [X, Y]
    H1.graph['st'] = (s, t)
    #utils.visualize_g(H1)

    H = H1
    x = G.graph['st'][0]
    y = G.graph['st'][1]

    X = normalize_nodes_to_set(X)
    Y = normalize_nodes_to_set(Y)

    # 2) Singleton reduction
    if len(X) > 1 or len(Y) > 1:
        H, x, y = singleton_reduction(H1, X, Y)

    # 3) Relabel to ints + enumerate
    H_int, name_to_id, id_to_name, s, t = relabel_to_ints(H, x, y)


    closest = get_closest_to_s_t(H_int,s,t)
    closest = decode_separators(closest,id_to_name)

    return closest


def find_important_seps_for_pair(
        G: nx.DiGraph,
        X: str,
        Y: str,
        which: str,
        R: List[str],
        I: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Runs your exact snippet for one (X,Y) and returns Z_sets (decoded names).
    """
    G.graph['st'] = (X, Y)
    # utils.visualize_g(G)
    H1 = build_H1_from_DAG(G, X=X, Y=Y, R=R, I=I)
    [s, t] = [X, Y]
    H1.graph['st'] = (s, t)
    # utils.visualize_g(H1)

    H = H1
    x = G.graph['st'][0]
    y = G.graph['st'][1]

    X = normalize_nodes_to_set(X)
    Y = normalize_nodes_to_set(Y)

    # 2) Singleton reduction
    if len(X) > 1 or len(Y) > 1:
        H, x, y = singleton_reduction(H1, X, Y)

    # 3) Relabel to ints + enumerate
    H_int, name_to_id, id_to_name, s, t = relabel_to_ints(H, x, y)

    num_nodes = G.number_of_nodes()

    #We call the function with t,s (instead of s,t) because important separators are defined relative to the source node, and in our setting we want to prioritize separators that are closer to Y (the outcome), not X (the treatment).
    important = gen_important_small_seps(H_int, t, s, num_nodes)
    important = decode_separators(important, id_to_name)

    return important