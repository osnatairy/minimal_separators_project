from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import random
import networkx as nx
import utils_ms as utils_ms
import separator_components_hasse as adj

import linear_sem_variance_calc as lvcal
# --- your generator ---
from linear_Structural_Equation_Model import LinearSEM

# --- your existing pipeline functions (DO NOT re-define) ---
# assumes these exist exactly in your project:
# build_H1_from_DAG, relabel_to_ints, decode_separators, alg
from test_all import build_H1_from_DAG, relabel_to_ints, decode_separators, singleton_reduction
import minimal_seperators_algorithm as alg


def all_reachable_xy_pairs(G: nx.DiGraph) -> List[Tuple[str, str]]:
    """
    Returns all ordered pairs (X,Y) such that there is a directed path X -> ... -> Y.
    (Useful for 'total effect' queries; avoids meaningless pairs.)
    """
    nodes = list(G.nodes())
    pairs = []
    for x in nodes:
        # descendants are exactly nodes reachable by a directed path
        for y in nx.descendants(G, x):
            pairs.append((x, y))
    return pairs





def run_many_xy(
    sem: LinearSEM,
    *,
    R: Optional[List[str]] = None,
    I: Optional[List[str]] = None,
    mode: str = "reachable",          # "reachable" | "all"
    sample_k: Optional[int] = 20,     # None -> run all pairs
    seed: int = 123,
    test_mode: bool = False,
) -> Dict[Tuple[str, str], List[List[str]]]:
    """
    For the SEM's DAG, enumerate adjustment sets for many X,Y pairs.

    mode="reachable": only pairs with X -> ... -> Y directed path.
    mode="all": all ordered pairs X!=Y (can be many, and may include pairs with no causal path)

    sample_k:
      - if integer: randomly sample that many pairs
      - if None: run over all pairs in the selected mode
    """
    G = sem.G
    if R is None:
        R = list(G.nodes())
    if I is None:
        I = []

    rng = random.Random(seed)

    if mode == "reachable":
        pairs = all_reachable_xy_pairs(G)
    elif mode == "all":
        nodes = list(G.nodes())
        pairs = [(x, y) for x in nodes for y in nodes if x != y]
    else:
        raise ValueError("mode must be 'reachable' or 'all'")

    if not pairs:
        return {}

    if sample_k is not None:
        sample_k = min(sample_k, len(pairs))
        pairs = rng.sample(pairs, k=sample_k)

    results: Dict[Tuple[str, str], List[List[str]]] = {}
    results1: Dict[Tuple[str, ...], float] = {}

    for X, Y in pairs:
        H, Z_sets = find_adjustment_sets_for_pair(G, X, Y, R=R, I=I)

        forward, reverse = adj.cy_components_for_sets(H, Y[0], Z_sets)
        print(forward, reverse)
        # get Hass graph for the Z - the adjustment sets
        res = adj.hasse_from_cy_results(forward, reverse)
        print("***************************************")
        print(res)
        print("***************************************")

        if test_mode: # check if there any containment pairs
            curr_result = utils_ms.find_containment_pairs(res)
            if curr_result:
                results1[(X, Y)] = curr_result

        else: # find the asimptotic variance
            for Z in Z_sets:
                if len(Z) > 1:
                    Z_key = tuple(sorted(Z))
                    aVar = lvcal.example_compute_avar(sem, X=X, Y=Y, Z=Z)
                    results1[Z_key] = aVar
            results[(X, Y)] = results1

    return results1

'''
if __name__ == "__main__":
    # 1) Generate a linear SEM (your code)
    sem = make_linear_sem(
        n=20,
        edge_prob=0.22,
        beta_scale=1.0,
        sigma2_low=0.2,
        sigma2_high=1.0,
        node_prefix="V",
        seed_graph=1,
        seed_params=2
    )

    # 2) Run over many (X,Y) pairs and find adjustment sets
    results = run_many_xy(
        sem,
        R=list(sem.G.nodes()),
        I=[],
        mode="reachable",
        sample_k=20,
        seed=42
    )

    # 3) Print a compact summary
    scores = []
    for (X, Y), Z_sets in results.items():
        for Z in Z_sets:
            if len(Z) > 1:
                aVar = lvcal.example_compute_avar(sem, X=X, Y=Y, Z=Z)
                scores.append((aVar, Z))

    if len(scores) > 0:
        scores.sort()
        best_aVar, best_Z = scores[0]
        print("Best Z:", best_Z, "best aVar:", best_aVar)

'''