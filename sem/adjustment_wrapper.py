import random
import networkx as nx
from typing import Dict, List, Tuple, Optional

from sem.linear_sem import LinearSEM
from sem.variance import example_compute_avar
from pipelines.adjust_sets import find_adjustment_sets_for_pair

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs

# validation for dowhy seperator
from validation.dowhy_check import test_Z_with_dowhy
from i_o.utils import append_line

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
    file_name: str = "outputs/defualt.csv"
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

        forward, reverse = cy_components_for_sets(H, Y, Z_sets)
        print(forward, reverse)
        # get Hass graph for the Z - the adjustment sets
        res = hasse_from_cy_results(forward, reverse)
        print("***************************************")
        print(res)
        print("***************************************")

        # check if all the Z_sets are an adjustment set using DoWhy.
        #if H.number_of_edges() > 0:
            #test_Z_with_dowhy(G, X, Y, Z_sets)

        if test_mode: # check if there any containment pairs
            curr_result = find_containment_pairs(res)
            if curr_result:
                results1[(X, Y)] = curr_result



        else: # find the asimptotic variance
            for Z in Z_sets:
                if len(Z) > 1:
                    Z_key = tuple(sorted(Z))
                    aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
                    results1[Z_key] = aVar
            results[(X, Y)] = results1

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        h_num_nodes = H.number_of_nodes()
        h_num_edges = H.number_of_edges()

        sep_result = len(res["hasse_edges"])

        append_line(file_name,
                    str(seed)+","+
                    str(num_nodes)+","+
                    str(num_edges)+","+
                    str(X)+","+
                    str(Y)+","+
                    str(h_num_nodes)+","+
                    str(h_num_edges)+","+
                    str(len(Z_sets))+","+
                    str(sep_result)+"/n")
    return results1

