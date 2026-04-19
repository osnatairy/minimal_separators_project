import utils as utils

from typing import Dict, List, Tuple, Optional

from graph.separators import Z_contains_parent_of_X
from graph.helpers import separator_with_min_variance

from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy
from sem.variance import example_compute_avar, conditional_variance_from_cov, sigma_from_sem

from i_o.utils import save_list_json, append_line, write_bucket_stats_to_csv


# analysis for the HASS diagram
from analysis.adjustment_hasse import cy_components_for_sets, hasse_from_cy_results, find_containment_pairs, \
    extract_separator_containment_pairs, frozenset_to_str, adjustment_set_exists, separator_is_subset

from pipelines.adjust_sets import find_adjustment_sets_for_pair

from graph.hankel_optimal_set import optimal_adjustment_set_O


from graph.generators import spanning_tree_then_orient
from causal.policies import static_do_policy
from bn.cpt import generate_bn_binary_logistic
from causal.influence.estimator_bn import asymptotic_variance_for_Z

from validation.dowhy_check import test_Z_with_dowhy

if __name__ == "__main__":

    import random

    N = 10
    nodes = [20,30,40, 50]
    prob_nodes = [0.07,0.1,0.15,0.2]#, 0.20, 0.25]  # , 0.5, 0.7]

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [node, int(node*0.3), 3, 1]:
                variance = f"{node}_{prob_node}_{k_roots}"
                seperators_file = f"outputs_bn/2026_04_13_exp2_bn_seperators_{variance}.csv"
                seperators_results = {}


                append_line(seperators_file,
                            "seed,X, Y, separator, len_sep, variance, type\n")
                for seed in range(N):
                    # 1) Generate a linear SEM (your code)

                    seed_graph, seed_params = utils.split_seeds(seed)

                    G = spanning_tree_then_orient(
                        n=node,
                        prob_edge=prob_node,
                        k_roots=k_roots,
                        node_prefix="V",
                        seed=seed_graph
                    )

                    bn = generate_bn_binary_logistic(G, seed=seed_params)


                    # 2) Run over many (X,Y) pairs and find adjustment sets
                    pairs = run_many_xy(
                        G,
                        mode="reachable",
                        seed=seed
                    )

                    R = list(G.nodes())
                    I = []

                    results: Dict[Tuple[str, str], List[List[str]]] = {}
                    results_minimal: Dict[Tuple[str, ...], float] = {}
                    results_non_minimal: Dict[Tuple[str, ...], float] = {}
                    results_optimal: Dict[Tuple[str, ...], float] = {}

                    for X, Y in pairs:
                        H, minimal_Z_sets = find_adjustment_sets_for_pair(G, X, Y,"smallminimalseps", R=R, I=I)

                        H, all_Z_sets = find_adjustment_sets_for_pair(G, X, Y, "RankedEnumSeps", R=R, I=I)

                        non_minimal = utils.get_non_minimal_separators(all_Z_sets,minimal_Z_sets)

                        # check if all the Z_sets are an adjustment set using DoWhy.
                        # if H.number_of_edges() > 0:
                        #dowhy_seps = test_Z_with_dowhy(G, X, Y, [])

                        o_x_y_seperator = optimal_adjustment_set_O(G, X, Y)
                        is_O_in_all_Z = adjustment_set_exists(all_Z_sets, list(sorted(o_x_y_seperator)))
                        is_O_in_minimal_Z = adjustment_set_exists(minimal_Z_sets, list(sorted(o_x_y_seperator)))

                        if is_O_in_minimal_Z:
                            print("is_O_in_minimal_Z")

                        if len(non_minimal) > 0:

                            optimal_seps = []
                            forward, reverse = cy_components_for_sets(H, Y, minimal_Z_sets)
                            # print(forward, reverse)
                            # get Hass graph for the Z - the adjustment sets
                            res = hasse_from_cy_results(forward, reverse)
                            # print("***************************************")
                            # print(res)
                            print("***************************************")

                            for pair in res['hasse_edges']:
                                temp_opt = reverse[pair[0]]
                                temp = tuple(sorted(temp_opt[0]))
                                optimal_seps.append(temp)


                            L_vars = []
                            # מדיניות סטטית do(I=1)
                            policy_fn = static_do_policy(a_star=1)

                            for Z in minimal_Z_sets:
                                if len(Z) >= 1:
                                    Z_key = tuple(sorted(Z))
                                    sigma2_m = asymptotic_variance_for_Z(
                                        bn, Y, X, Z, L_vars, policy_fn, None
                                    )


                                    lable = "minimal"
                                    if Z_key in optimal_seps:
                                        lable = "optimal"

                                    append_line(seperators_file,
                                                str(seed) + "," +
                                                str(X) + "," +
                                                str(Y) + "," +
                                                frozenset_to_str(Z_key) + "," +
                                                str(len(Z_key)) + "," +
                                                str(round(sigma2_m, 5)) + "," +
                                                lable
                                                )


                            for Z in non_minimal:
                                if len(Z) >= 1:
                                    Z_key = tuple(sorted(Z))
                                    sigma2_nm = asymptotic_variance_for_Z(
                                        bn, Y, X, Z, L_vars, policy_fn, None
                                    )
                                    append_line(seperators_file,
                                                str(seed) + "," +
                                                str(X) + "," +
                                                str(Y) + "," +
                                                frozenset_to_str(Z_key) + "," +
                                                str(len(Z_key)) + "," +
                                                str(round(sigma2_nm, 5)) + "," +
                                                "non_minimal"
                                                )


                            Z_key = tuple(sorted(o_x_y_seperator))
                            sigma2_henkel = asymptotic_variance_for_Z(
                                bn, Y, X, o_x_y_seperator, L_vars, policy_fn, None
                            )
                            append_line(seperators_file,
                                        str(seed) + "," +
                                        str(X) + "," +
                                        str(Y) + "," +
                                        frozenset_to_str(Z_key) + "," +
                                        str(len(Z_key)) + "," +
                                        str(round(sigma2_henkel, 5)) + "," +
                                        "henkel"
                                        )


    print("End of this script")
