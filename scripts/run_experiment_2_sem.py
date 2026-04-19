import utils as utils

from typing import Dict, List, Tuple, Optional

from graph.separators import Z_contains_parent_of_X
from graph.helpers import separator_with_min_variance

from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy
from sem.variance import example_compute_avar, conditional_variance_from_cov, sigma_from_sem

from i_o.utils import save_list_json, append_line, write_bucket_stats_to_csv
from i_o.json_loader import save_linear_sem

# analysis for the HASS diagram
from analysis.adjustment_hasse import cy_components_for_sets, hasse_from_cy_results, find_containment_pairs, \
    extract_separator_containment_pairs, frozenset_to_str, adjustment_set_exists, separator_is_subset

from pipelines.adjust_sets import find_adjustment_sets_for_pair

from graph.hankel_optimal_set import optimal_adjustment_set_O

from experiments.create_buckets_graphs import make_bucket_boxplots
from analysis.bucket_sep import bucket_separators_by_cy_layers, bucket_variance_statistics, \
    bucket_separators_by_y_connectivity, bucket_y_component

from validation.dowhy_check import test_Z_with_dowhy

if __name__ == "__main__":

    import random

    N = 10
    nodes = [20, 30, 40, 50]
    prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]#, 0.20, 0.25]  # , 0.5, 0.7]
    betas = [0.7]

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [node, int(node*0.3), 3, 1]:
                variance = f"{node}_{prob_node}_{k_roots}"
                seperators_file = f"outputs_sem/2026_04_13_exp2_sem_seperators_{variance}.csv"
                seperators_results = {}


                append_line(seperators_file,
                            "seed,X, Y, separator, len_sep, variance, type\n")
                for seed in range(N):
                    # 1) Generate a linear SEM (your code)
                    sem = make_linear_sem(
                        n=node,
                        edge_prob=prob_node,
                        # beta_scale=beta,
                        # sigma2_low=0.2,
                        # sigma2_high=0.9,
                        beta_scale = 1.0,
                        sigma2_low = 0.2,
                        sigma2_high = 1.0,
                        node_prefix="V",
                        seed=seed,
                        k_roots=k_roots
                    )

                    # 2) Run over many (X,Y) pairs and find adjustment sets
                    pairs = run_many_xy(
                        sem.G,
                        mode="reachable",
                        seed=seed
                    )

                    R = list(sem.G.nodes())
                    I = []

                    results: Dict[Tuple[str, str], List[List[str]]] = {}
                    results_minimal: Dict[Tuple[str, ...], float] = {}
                    results_non_minimal: Dict[Tuple[str, ...], float] = {}
                    results_optimal: Dict[Tuple[str, ...], float] = {}

                    for X, Y in pairs:
                        H, minimal_Z_sets = find_adjustment_sets_for_pair(sem.G, X, Y,"smallminimalseps", R=R, I=I)

                        H, all_Z_sets = find_adjustment_sets_for_pair(sem.G, X, Y, "RankedEnumSeps", R=R, I=I)

                        non_minimal = utils.get_non_minimal_separators(all_Z_sets,minimal_Z_sets)

                        # check if all the Z_sets are an adjustment set using DoWhy.
                        # if H.number_of_edges() > 0:
                        #dowhy_seps = test_Z_with_dowhy(sem.G, X, Y, [])

                        o_x_y_seperator = optimal_adjustment_set_O(sem.G, X, Y)
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



                            for Z in minimal_Z_sets:
                                if len(Z) >= 1:
                                    Z_key = tuple(sorted(Z))
                                    aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)

                                    lable = "minimal"
                                    if Z_key in optimal_seps:
                                        lable = "optimal"

                                    append_line(seperators_file,
                                                str(seed) + "," +
                                                str(X) + "," +
                                                str(Y) + "," +
                                                frozenset_to_str(Z_key) + "," +
                                                str(len(Z_key)) + "," +
                                                str(round(aVar, 5)) + "," +
                                                lable
                                                )


                            for Z in non_minimal:
                                if len(Z) >= 1:
                                    Z_key = tuple(sorted(Z))
                                    aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
                                    append_line(seperators_file,
                                                str(seed) + "," +
                                                str(X) + "," +
                                                str(Y) + "," +
                                                frozenset_to_str(Z_key) + "," +
                                                str(len(Z_key)) + "," +
                                                str(round(aVar, 5)) + "," +
                                                "non_minimal"
                                                )


                            Z_key = tuple(sorted(o_x_y_seperator))
                            aVar = example_compute_avar(sem, X=X, Y=Y, Z=o_x_y_seperator)
                            append_line(seperators_file,
                                        str(seed) + "," +
                                        str(X) + "," +
                                        str(Y) + "," +
                                        frozenset_to_str(Z_key) + "," +
                                        str(len(Z_key)) + "," +
                                        str(round(aVar, 5)) + "," +
                                        "henkel"
                                        )


    print("End of this script")
