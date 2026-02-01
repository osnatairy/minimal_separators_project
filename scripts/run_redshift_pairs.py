from typing import Dict, List, Tuple, Iterable, Optional

import utils as utils

from graph.separators import find_seperators
from i_o.dot_loader import dot_to_mapping_and_edges

from graph.transforms import build_G_from_mapped_edges, relabel_to_ints
from graph.h1_builder import build_H1_from_DAG

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs


if __name__ == '__main__':

    path = "BN_DATA/amazon_redshift.dot"
    # Example usage after you re-upload:
    name_to_id_G, id_to_name_G, edges_mapped = dot_to_mapping_and_edges(path, scheme="letters")
    print(name_to_id_G)
    print(edges_mapped)

    s_t_list = [
        ("query_template", "compile_time"),
        ("query_template", "planning_time"),
        ("query_template", "execution_time"),
        ("query_template", "lock_wait_time"),
        ("query_template", "elapsed_time"),

        ("num_joins", "planning_time"),
        ("num_joins", "execution_time"),
        ("num_joins", "elapsed_time"),

        ("num_tables", "planning_time"),
        ("num_tables", "execution_time"),
        ("num_tables", "elapsed_time"),

        ("num_columns", "execution_time"),
        ("num_columns", "elapsed_time"),

        ("result_cache_hit", "execution_time"),
        ("result_cache_hit", "elapsed_time"),
    ]
    Iset = []
    R = id_to_name_G.keys()

    results: Dict[Tuple[str, str], List[List[str]]] = {}
    for s_t in s_t_list:
        X = s_t[0]
        Y = s_t[1]
        s = name_to_id_G[X]
        t = name_to_id_G[Y]
        # Example usage:
        # ---- Example ----
        # Suppose you mapped by letters and decided A is s and T is t:
        G = build_G_from_mapped_edges(edges_mapped, id_to_name=id_to_name_G, st=(s,t))

        G.graph['st'] = (s, t)
        #utils.visualize_g(G)
        # 1) H^1
        H1 = build_H1_from_DAG(G, X=s, Y=t, R=R, I=Iset)

        H1.graph['st'] = (s, t)
        #utils.visualize_g(H1)

        H = H1
        x = G.graph['st'][0]
        y = G.graph['st'][1]

        # # 2) Singleton reduction
        # if len(X) > 1 or len(Y) > 1:
        #     H, x, y = singleton_reduction(H1, X, Y)

        Z_sets = find_seperators(H, x, y, which="SmallMinimalSeps")

        #results[s_t] = Z_unique

        forward, reverse = cy_components_for_sets(H, Y[0], Z_sets)
        print(forward, reverse)
        # get Hass graph for the Z - the adjustment sets
        res = hasse_from_cy_results(forward, reverse)
        print("***************************************")
        print(res)
        print("***************************************")
        utils.visualize_g(G)
        utils.visualize_g(H)
        if not False:#all_empty(Z_unique):
            utils.visualize_g(G)
            utils.visualize_g(H)

        curr_result = find_containment_pairs(res)
        if curr_result:
            results[s_t] = Z_sets
            # print("containment pairs:", pairs[:10])
            # print("number of containments:", len(pairs))

    print(results)