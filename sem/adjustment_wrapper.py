import random
import networkx as nx
from typing import Dict, List, Tuple, Optional

from sem.linear_sem import LinearSEM
from sem.variance import example_compute_avar
from pipelines.adjust_sets import find_adjustment_sets_for_pair

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs,extract_separator_containment_pairs, frozenset_to_str

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
    G: nx.DiGraph,
    *,
    mode: str = "reachable",          # "reachable" | "all"
    sample_k: Optional[int] = 20,     # None -> run all pairs
    seed: int = 123,
) -> Dict[Tuple[str, str], List[List[str]]]:
    """
    For the SEM's DAG, enumerate adjustment sets for many X,Y pairs.

    mode="reachable": only pairs with X -> ... -> Y directed path.
    mode="all": all ordered pairs X!=Y (can be many, and may include pairs with no causal path)

    sample_k:
      - if integer: randomly sample that many pairs
      - if None: run over all pairs in the selected mode
    """



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



    return pairs

