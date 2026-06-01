from typing import Dict, List, Tuple, Optional

from graph.separators import Z_contains_parent_of_X
from graph.helpers import separator_with_min_variance

from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy
from sem.variance import example_compute_avar, conditional_variance_from_cov, sigma_from_sem, x_drain_two_sets

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

if __name__ == "__main__":

    import random

    N = 10
    nodes = [20, 30, 40, 50]
    prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]
    betas = [0.7]

    date = "2026_04_13"

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [node, int(node * 0.3), 3, 1]:

                variance = f"_{node}_{prob_node}_{k_roots}"

                file = f"outputs_sem/{date}_***_{variance}.csv"
                results = {}


                graph_path = "../outputs_sem/graph/"
                append_line(file,
                            "seed,X, Y, H_graph_nodes, H_graph_edges, num_seperator, num_contained_separators\n")
                for seed in range(N):
                    # 1) Generate a linear SEM (your code)
                    sem = make_linear_sem(
                        n=node,
                        edge_prob=prob_node,
                        # beta_scale=beta,
                        # sigma2_low=0.2,
                        # sigma2_high= 0.9,
                        beta_scale=1.0,
                        sigma2_low=0.2,
                        sigma2_high=1.0,
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

                    test_mode: bool = False
                    # file_name: str = "outputs_sem/defualt.csv"

                    R = list(sem.G.nodes())
                    I = []

                    results: Dict[Tuple[str, str], List[List[str]]] = {}
                    results1: Dict[Tuple[str, ...], float] = {}

                    for X, Y in pairs:
                        H, Z_sets = find_adjustment_sets_for_pair(sem.G, X, Y, "smallminimalseps", R=R, I=I)

                        forward, reverse = cy_components_for_sets(H, Y, Z_sets)
                        # print(forward, reverse)
                        # get Hass graph for the Z - the adjustment sets
                        res = hasse_from_cy_results(forward, reverse)
                        # print("***************************************")
                        # print(res)
                        # print("***************************************")

                        # check if all the Z_sets are an adjustment set using DoWhy.
                        # if H.number_of_edges() > 0:
                        # test_Z_with_dowhy(G, X, Y, Z_sets)

                        num_nodes = sem.G.number_of_nodes()
                        num_edges = sem.G.number_of_edges()

                        h_num_nodes = H.number_of_nodes()
                        h_num_edges = H.number_of_edges()

                        if test_mode:  # check if there is any containment pairs
                            curr_result = find_containment_pairs(res)
                            if curr_result:
                                results1[(X, Y)] = curr_result

                            sep_result = len(res["hasse_edges"])

                            append_line(file_name,
                                        str(seed) + "," +
                                        str(num_nodes) + "," +
                                        str(num_edges) + "," +
                                        str(X) + "," +
                                        str(Y) + "," +
                                        str(h_num_nodes) + "," +
                                        str(h_num_edges) + "," +
                                        str(len(Z_sets)) + "," +
                                        str(sep_result) + "/n")



                        else:  # find the asimptotic variance
                            for Z in Z_sets:
                                if len(Z) >= 1:
                                    Z_key = tuple(sorted(Z))
                                    aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
                                    results1[Z_key] = aVar

                                # append_line(file_name,
                                #             str(seed) + "," +
                                #             str(num_nodes) + "," +
                                #             str(num_edges) + "," +
                                #             str(X) + "," +
                                #             str(Y) + "," +
                                #             str(h_num_nodes) + "," +
                                #             str(h_num_edges) + "," +
                                #             ";".join(map(str, Z)) + "," +
                                #             str(round(aVar, 5)) + "/n")
                            results[(X, Y)] = results1

                            hankel_optimal = optimal_adjustment_set_O(sem.G, X, Y)
                            is_O_in_Z = adjustment_set_exists(Z_sets, list(sorted(hankel_optimal)))
                            # print(hankel_optimal)

                            if len(results1) > 0:
                                best_sep, v = separator_with_min_variance(results1)
                                is_O_in_optimal_z = separator_is_subset(best_sep, hankel_optimal)
                            else:
                                is_O_in_optimal_z = False

                            var_names, Sigma = sigma_from_sem(sem)

                            pair_adjustment = extract_separator_containment_pairs(res)

                            for hass in pair_adjustment:

                                var_x_given_z_out = conditional_variance_from_cov(
                                    Sigma, target=X, given=list(tuple(sorted(hass['outer_sep']))), var_names=var_names,
                                    ridge=1e-10)
                                var_x_given_z_in = conditional_variance_from_cov(
                                    Sigma, target=X, given=list(tuple(sorted(hass['inner_sep']))), var_names=var_names,
                                    ridge=1e-10)

                                var_y_given_xz_out = conditional_variance_from_cov(
                                    Sigma, target=Y, given=[X] + list(tuple(sorted(hass['outer_sep']))),
                                    var_names=var_names, ridge=1e-10
                                )
                                var_y_given_xz_in = conditional_variance_from_cov(
                                    Sigma, target=Y, given=[X] + list(tuple(sorted(hass['inner_sep']))),
                                    var_names=var_names,
                                    ridge=1e-10
                                )

                                drains = x_drain_two_sets(sem, X, hass['inner_sep'], hass['outer_sep'])

                                append_line(new_file_name,
                                            str(seed) + "," +
                                            str(X) + "," +
                                            str(Y) + "," +
                                            frozenset_to_str(hass['outer_sep']) + "," +
                                            str(len(hass['outer_sep'])) + "," +
                                            str(var_y_given_xz_out) + ',' +
                                            # str(var_x_given_z_out)+','+
                                            # frozenset_to_str(hass['outer_component']) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5)) + "," +
                                            frozenset_to_str(hass['inner_sep']) + "," +
                                            str(len(hass['inner_sep'])) + "," +
                                            str(var_y_given_xz_in) + ',' +
                                            # str(var_x_given_z_in) + ',' +
                                            # frozenset_to_str(hass['inner_component']) + "," +
                                            str(round(results1[tuple(sorted(hass['inner_sep']))], 5)) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                                results1[tuple(sorted(hass['inner_sep']))], 5)) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                                results1[tuple(sorted(hass['inner_sep']))], 5) < 0) + "," +
                                            str(len(hass['outer_sep']) - len(hass['inner_sep'])) + "," +
                                            str(drains['X-Drain(Z1)']) + "," +
                                            str(drains['X-Drain(Z2)']) + "," +
                                            str(drains['X-Drain(Z2)'] - drains['X-Drain(Z1)']) + ","
                                            # str(is_O_in_Z)+ ","+
                                            # str(is_O_in_optimal_z)
                                            )

                                print(f"seed: {seed}, X: {X}, Y: {Y}")
                                has_parent_outer_sep, which = Z_contains_parent_of_X(sem.G, hass['outer_sep'], X)
                                # print("outer_sep contains parent of X?", has_parent_outer_sep, "parents in outer_sep:", which)

                                has_parent_inner_sep, which_inner_sep = Z_contains_parent_of_X(sem.G, hass['inner_sep'],
                                                                                               X)
                                # print("inner_sep contains parent of X?", has_parent_inner_sep, "parents in inner_sep:", which)

                                if round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                        results1[tuple(sorted(hass['inner_sep']))], 5) < 0:
                                    curr_path = graph_path + "graph_" + str(seed) + "_" + X + "_" + Y + ".json"
                                    save_linear_sem(curr_path, sem, pairs, len(Z_sets))
                                    print("SOMETHING IS WRONG")

                        buckets = bucket_separators_by_cy_layers(H, Y, Z_sets)
                        print(buckets)

                        buckets1 = bucket_separators_by_y_connectivity(H, Y, Z_sets)
                        print(buckets)

                        something = bucket_y_component(buckets1, res['Z_to_component'])

                        bucket_variance_statistics(seed, X, Y, buckets1, results1, res['Z_to_component'],
                                                   buckets_results)
                        print("bucket statistics:")
                        # print(bucket_statistics)

                    if len(results) > 0:
                        seeds_to_keep.append(seed)

                save_list_json(seeds_to_keep, "../outputs_sem/seeds_to_keep.json")

                write_bucket_stats_to_csv(bucket_file, buckets_results, False)

    print("End of this script")
    # make_bucket_boxplots(
    #     bucket_file,
    #     variance=variance,
    #     #output_dir=f"bucket_statistics_SEM{variance}",
    #     mode="global",
    # )
    # make_bucket_boxplots(
    #     bucket_file,
    #     variance=variance,
    #     #output_dir=f"bucket_statistics_SEM{variance}",
    #     mode="per_run",
    # )
    '''
                # 3) Print a compact summary
                scores = []
                for (X, Y), Z_sets in results.items():
                    for Z in Z_sets:
                        if len(Z) > 1:
                            aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
                            scores.append((aVar, Z))

                if len(scores) > 0:
                    scores.sort()
                    best_aVar, best_Z = scores[0]
                    print("Best Z:", best_Z, "best aVar:", best_aVar)
                '''
