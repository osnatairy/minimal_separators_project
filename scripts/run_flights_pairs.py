from typing import Dict, List, Tuple, Iterable, Optional

import utils as utils
from bn import BN
from graph.separators import find_seperators
from i_o.dot_loader import dot_to_mapping_and_edges
from i_o.json_loader import load_bn_from_json

from bn.pgmpy_adapter import bn_to_pgmpy_model

from causal.policies import static_do_policy
from causal.influence.estimator_bn import asymptotic_variance_for_Z

from graph.transforms import build_G_from_mapped_edges, relabel_to_ints
from graph.h1_builder import build_H1_from_DAG

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs

def relabel_bn(bn, name_to_id_G, BNClass):
    lower_name_to_id = {
        name.lower(): node_id
        for name, node_id in name_to_id_G.items()
    }

    new_bn = BNClass()

    # 1) variables + domains
    for old_var, domain in bn.domains.items():
        new_var = lower_name_to_id[old_var.lower()]
        new_bn.add_var(new_var, domain)

    # 2) edges
    for old_child in bn.domains.keys():
        new_child = lower_name_to_id[old_child.lower()]

        for old_parent in bn.parents(old_child):
            new_parent = lower_name_to_id[old_parent.lower()]
            new_bn.add_edge(new_parent, new_child)

    # 3) parent order
    for old_var in bn.domains.keys():
        old_order = bn.parent_order[old_var]
        new_var = lower_name_to_id[old_var.lower()]
        new_order = [lower_name_to_id[p.lower()] for p in old_order]
        new_bn.set_parent_order(new_var, new_order)

    # 4) CPTs
    for old_var, old_table in bn.cpts.items():
        new_var = lower_name_to_id[old_var.lower()]
        new_bn.set_cpt(new_var, old_table)

    return new_bn
if __name__ == '__main__':

    path = "BN_DATA/flights/flights.dot"
    # Example usage after you re-upload:
    name_to_id_G, id_to_name_G, edges_mapped = dot_to_mapping_and_edges(path, scheme="letters")
    bn = load_bn_from_json("BN_DATA/flights/bn__flights.json", BNClass=BN)

    bn_letters = relabel_bn(bn, name_to_id_G, BNClass=BN)

    print(name_to_id_G)
    print(edges_mapped)

    s_t_list = [
        ("WEATHER_DELAY" , "ARRIVAL_DELAY"),
        ("DEPARTURE_DELAY" , "ARRIVAL_DELAY"),
        ("WEATHER_DELAY" , "DEPARTURE_DELAY"),
        ("AIRLINE_DELAY" , "ARRIVAL_DELAY"),
        ("DISTANCE" , "ARRIVAL_DELAY"),

        ("SCHEDULED_DEPARTURE" , "DEPARTURE_DELAY"),
        ("TAXI_OUT" , "ARRIVAL_DELAY"),
        ("AIR_TIME" , "ARRIVAL_DELAY")
    ]
    Iset = []
    R = id_to_name_G.keys()

    results1: Dict[Tuple[str, ...], float] = {}
    results: Dict[Tuple[str, str], List[List[str]]] = {}

    for s_t in s_t_list:
        X = s_t[0]
        Y = s_t[1]
        print("======================================================")
        print(X,Y)



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
        for Z in Z_sets:
            for z in Z:
                print(id_to_name_G[z])
            print("---")
        #results[s_t] = Z_unique

        forward, reverse = cy_components_for_sets(H, Y[0], Z_sets)
        #print(forward, reverse)
        # get Hass graph for the Z - the adjustment sets
        res = hasse_from_cy_results(forward, reverse)
        print("***************************************")
        print(res)
        print("***************************************")

        if len(res['hasse_edges']) >0:
            print(s_t)
            utils.visualize_g(G)
            utils.visualize_g(H)

            L_vars = []  # נניח שהמדיניות תלויה ב-G,H (אפשר גם L_vars=[])
            # מדיניות סטטית do(I=1)
            policy_fn = static_do_policy(a_star=1)
            model, infer = bn_to_pgmpy_model(bn_letters)
            for Z in Z_sets:
                if len(Z) >= 1:
                    Z_key = tuple(sorted(Z))
                    sigma2 = asymptotic_variance_for_Z(
                        bn_letters,infer, s, t, Z, L_vars, policy_fn, None
                    )
                    results1[Z_key] = sigma2
                    print(Z_key , sigma2)
                    for node in Z_key:
                        print(node, id_to_name_G[node])
        #utils.visualize_g(G)
        #utils.visualize_g(H)
        # if not False:#all_empty(Z_unique):
        #     utils.visualize_g(G)
        #     utils.visualize_g(H)

        curr_result = find_containment_pairs(res)
        if curr_result:
            results[s_t] = Z_sets
            # print("containment pairs:", pairs[:10])
            # print("number of containments:", len(pairs))

    print(results)