import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Dict, List, Tuple, Optional

from sem.adjustment_wrapper import run_many_xy
from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from i_o.utils import save_list_json, append_line,write_bucket_stats_to_csv

import utils as utils
# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs,extract_separator_containment_pairs, frozenset_to_str

from pipelines.adjust_sets import find_adjustment_sets_for_pair
from analysis.bucket_sep import bucket_separators_by_cy_layers, bucket_variance_statistics,bucket_separators_by_y_connectivity,bucket_y_component

from analysis.treewidth import treewidth_upper_bound_ve

#influance function calculation
from causal.influence.estimator_bn import asymptotic_variance_for_Z, asymptotic_variance_over_Z_sets, \
    asymptotic_variance_for_Z_old#,asymptotic_variance_over_Z_sets_parallel
from causal.policies import static_do_policy

from ex1_parallel_run import process_all_xy_pairs_parallel

from experiments.create_buckets_graphs import make_bucket_boxplots

if __name__ == "__main__":

    import random

    N = 10 #number of different seeds
    nodes = [20,30,40, 50]# different sizes of nodes in  a tree
    prob_nodes = [0.07, 0.1, 0.15, 0.2]  #the probability of an edge

    date = "2026_04_28"

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [node, int(node*0.3), 3, 1]:

                seeds_to_keep = []
                variance = f"_main_{node}_{prob_node}_{k_roots}"
                file_name = f"outputs_bn/{date}_seeds_data_{variance}.csv"
                bucket_file = f"outputs_bn/{date}_bucket_statistics_{variance}.csv"
                write_bucket_stats_to_csv(bucket_file,[])
                buckets_results = {}

                append_line(file_name,
                            "seed, graph_nodes, graph_edges,X, Y, H_graph_nodes, H_graph_edges, num_seperator, num_contained_separators\n")



                for seed in range(N):
                    # 1) Generate a BN (your code)

                    seed_graph, seed_params = utils.split_seeds(seed)

                    G = spanning_tree_then_orient(
                        n=node,
                        prob_edge=prob_node,
                        k_roots=k_roots,
                        node_prefix="V",
                        seed=seed_graph
                    )

                    bn = generate_bn_binary_logistic(G, seed=seed_params)

                    w, order = treewidth_upper_bound_ve(bn, heuristic="minfill")
                    print("VE-style treewidth upper bound =", w)
                    print("order =", order)

                    if w > 20:
                        continue

                    # עכשיו bn הוא אובייקט BN מלא:
                    print("nodes:", list(bn.g.nodes()))
                    print("edges:", list(bn.g.edges()))

                    # לדוגמה: לבדוק CPT של צומת ראשון
                    v0 = list(bn.g.nodes())[0]
                    print("parents of", v0, "=", bn.parents(v0))
                    for pkey, row in list(bn.cpts[v0].items())[:5]:
                        print("pa =", pkey, "->", row)


                    # 2) Run over many (X,Y) pairs and find adjustment sets
                    pairs = run_many_xy(
                        bn.g,
                        mode="reachable",
                        #sample_k=20,
                        seed=seed)
                    print("pairs:", pairs)

                    test_mode = False

                    R = list(bn.g.nodes())
                    I = []



                    results: Dict[Tuple[str, str], Dict[Tuple[str, ...], float]] = {}
                    t0_start = time.perf_counter()
                    pair_results = process_all_xy_pairs_parallel(
                        bn=bn,
                        seed=seed,
                        pairs=pairs,
                        R=R,
                        I=I,
                        test_mode=test_mode,
                    )

                    t0_end = time.perf_counter()
                    find_adjustment_sets_time = (t0_end - t0_start)
                    print(f"parallel XY pairs = {find_adjustment_sets_time:.6f} seconds")

                    for pair_result in pair_results:
                        X = pair_result["X"]
                        Y = pair_result["Y"]

                        print("seed=", seed, " X =", X, " Y =", Y)
                        #print(f"find_adjustment_sets_time = {pair_result['find_adjustment_sets_time']:.6f} seconds")
                        print(f"Z_sets = {len(pair_result['Z_sets'])}")

                        if test_mode:
                            test_mode_result = pair_result["test_mode_result"]
                            curr_result = test_mode_result["curr_result"]
                            sep_result = test_mode_result["sep_result"]

                            if curr_result:
                                results[(X, Y)] = curr_result

                            append_line(
                                file_name,
                                str(seed) + "," +
                                str(pair_result["num_nodes"]) + "," +
                                str(pair_result["num_edges"]) + "," +
                                str(X) + "," +
                                str(Y) + "," +
                                str(pair_result["h_num_nodes"]) + "," +
                                str(pair_result["h_num_edges"]) + "," +
                                str(len(pair_result["Z_sets"])) + "," +
                                str(sep_result)  + "\n"
                                #str(pair_result["find_adjustment_sets_time"]) + "\n"
                            )

                        else:
                            results1 = pair_result["results1"]
                            results[(X, Y)] = results1

                            # print(
                            #     f"calc_variance_time with improvement = {pair_result['calc_variance_time']:.6f} seconds")

                            pair_adjustment = extract_separator_containment_pairs(pair_result["res"])
                            new_file_name = file_name.replace(".csv", "_seperators.csv")
                            for hass in pair_adjustment:
                                append_line(
                                    new_file_name,
                                    str(seed) + "," +
                                    str(X) + "," +
                                    str(Y) + "," +
                                    frozenset_to_str(hass['outer_sep']) + "," +
                                    frozenset_to_str(hass['outer_component']) + "," +
                                    str(round(results1[tuple(sorted(hass['outer_sep']))], 5)) + "," +
                                    frozenset_to_str(hass['inner_sep']) + "," +
                                    frozenset_to_str(hass['inner_component']) + "," +
                                    str(round(results1[tuple(sorted(hass['inner_sep']))], 5)) + "," +
                                    str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                        results1[tuple(sorted(hass['inner_sep']))], 5)) +
                                    "\n"
                                )

                            bucket_input = pair_result["bucket_input"]
                            buckets1 = bucket_input["buckets1"]

                            bucket_variance_statistics(
                                seed,
                                X,
                                Y,
                                buckets1,
                                results1,
                                pair_result["res"]["Z_to_component"],
                                buckets_results
                            )


                    results: Dict[Tuple[str, str], List[List[str]]] = {}
                    results1: Dict[Tuple[str, ...], float] = {}

                    t1_start = time.perf_counter()
                    for X, Y in pairs:
                        print("seed= ",seed," X =", X, " Y =", Y)
                        t0_start = time.perf_counter()
                        H, Z_sets = find_adjustment_sets_for_pair(bn.g, X, Y,"smallminimalseps", R=R, I=I)

                        Z_sets = [Z for Z in Z_sets if len(Z) >= 1]

                        t0_end = time.perf_counter()
                        #print(f"Avg per run: {(t0_end - t0_start) /:.6f} seconds")
                        find_adjustment_sets_time = (t0_end - t0_start)
                        print(f"find_adjustment_sets_time = {find_adjustment_sets_time:.6f} seconds")
                        print(f"Z_sets = {len(Z_sets)}")
                        forward, reverse = cy_components_for_sets(H, Y, Z_sets)
                        #print(forward, reverse)
                        # get Hass graph for the Z - the adjustment sets
                        res = hasse_from_cy_results(forward, reverse)
                        #print("***************************************")
                        #print(res)
                        #print("***************************************")

                        # check if all the Z_sets are an adjustment set using DoWhy.
                        # if H.number_of_edges() > 0:
                        # test_Z_with_dowhy(G, X, Y, Z_sets)

                        num_nodes = bn.g.number_of_nodes()
                        num_edges = bn.g.number_of_edges()

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
                                        str(sep_result) +","+
                                        str(find_adjustment_sets_time)+ "/n")



                        else:  # find the asimptotic variance

                            L_vars = []  # נניח שהמדיניות תלויה ב-G,H (אפשר גם L_vars=[])

                            # מדיניות סטטית do(I=1)
                            policy_fn = static_do_policy(a_star=1)

                            tt1_start = time.perf_counter()
                            results1 = asymptotic_variance_over_Z_sets(
                                       bn, Y, X, Z_sets, L_vars, policy_fn, None)
                            tt1_end = time.perf_counter()
                            calc_variance_time = (tt1_end - tt1_start)
                            print(f"calc_variance_time with improvement = {calc_variance_time:.6f} seconds")
                            a_star = 1
                            tt1_start = time.perf_counter()
                            #
                            # results1 = asymptotic_variance_over_Z_sets_parallel(
                            #     bn=bn,
                            #     Y=Y,
                            #     A_name=X,
                            #     Z_sets=Z_sets,
                            #     L_vars=L_vars,
                            #     #policy_fn=policy_fn,
                            #     a_star=a_star,
                            #     value_map=None,
                            #     max_workers=2,  # אפשר לשנות
                            # )
                            #
                            # tt1_end = time.perf_counter()
                            calc_variance_time = tt1_end - tt1_start
                            #
                            # print(f"calc_variance_time parallel = {calc_variance_time:.6f} seconds")
                            # print(f"calc_variance_time parallel = {calc_variance_time / 60:.2f} minutes")

                            # t1_start = time.perf_counter()
                            # for Z in Z_sets:
                            #     if len(Z) >= 1:
                            #         Z_key = tuple(sorted(Z))
                            #
                            #         sigma2 = asymptotic_variance_for_Z_old(
                            #             bn, Y, X, Z, L_vars, policy_fn, None
                            #         )
                            #
                            #         results1[Z_key] = sigma2
                            #
                            #         append_line(file_name,
                            #                     str(seed) + "," +
                            #                     str(num_nodes) + "," +
                            #                     str(num_edges) + "," +
                            #                     str(X) + "," +
                            #                     str(Y) + "," +
                            #                     str(h_num_nodes) + "," +
                            #                     str(h_num_edges) + "," +
                            #                     ";".join(map(str, Z)) + "," +
                            #                     str(round(sigma2, 5)) + ","+
                            #                     str(calc_variance_time) + "/n")
                            # t1_end = time.perf_counter()
                            # all_time = float((t1_end - t1_start))
                            results[(X, Y)] = results1
                            pair_adjustment = extract_separator_containment_pairs(res)
                            new_file_name = file_name.replace(".csv", "_seperators.csv")

                            for hass in pair_adjustment:
                                append_line(new_file_name,
                                            str(seed) + "," +
                                            str(X) + "," +
                                            str(Y) + "," +
                                            frozenset_to_str(hass['outer_sep']) + "," +
                                            frozenset_to_str(hass['outer_component']) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5)) + "," +
                                            frozenset_to_str(hass['inner_sep']) + "," +
                                            frozenset_to_str(hass['inner_component']) + "," +
                                            str(round(results1[tuple(sorted(hass['inner_sep']))], 5)) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                                results1[tuple(sorted(hass['inner_sep']))], 5))
                                            )
                            #print(f"calc_variance_time all time = {all_time:.6f} seconds")
                        '''
                        buckets = bucket_separators_by_cy_layers(H, Y, Z_sets)
                        print(buckets)
                        utils.visualize_g(H)
                        buckets1 = bucket_separators_by_y_connectivity(H, Y, Z_sets)
                        print(buckets)
                        y_component_result = bucket_y_component(buckets1, res['Z_to_component'])
                        bucket_statistics = bucket_variance_statistics(buckets1, results1,y_component_result)
                        print("bucket statistics:")
                        print(bucket_statistics)
                        
                        
                        write_bucket_stats_to_csv(bucket_file, bucket_statistics, seed, X, Y, False)
                        '''
                        t1_bucket_start = time.perf_counter()
                        buckets = bucket_separators_by_cy_layers(H, Y, Z_sets)
                        print(buckets)

                        buckets1 = bucket_separators_by_y_connectivity(H, Y, Z_sets)
                        print(buckets)

                        something = bucket_y_component(buckets1, res['Z_to_component'])

                        bucket_variance_statistics(seed, X, Y, buckets1, results1, res['Z_to_component'],
                                                   buckets_results)
                        print("bucket statistics:")
                        # print(bucket_statistics)
                        t1_bucket_end = time.perf_counter()
                        print(f"time_bucket = {(t1_bucket_start-t1_bucket_end)/100}")





                    if len(results) > 0:
                        seeds_to_keep.append(seed)

                    t1_end = time.perf_counter()
                    find_adjustment_sets_time1 = (t1_end - t1_start)
                    print(f"regular XY pairs = {find_adjustment_sets_time1:.6f} seconds")



                save_list_json(seeds_to_keep, "outputs_bn/seeds_to_keep.json")

                write_bucket_stats_to_csv("regular_"+bucket_file, buckets_results, False)

                # make_bucket_boxplots(
                #     bucket_file,
                #     #output_dir=f"bucket_statistics_{variance}",
                #     mode="global",
                # )
                # make_bucket_boxplots(
                #     bucket_file,
                #     #output_dir=f"bucket_statistics_{variance}",
                #     mode="per_run",
                # )
    print("End of this script")

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
