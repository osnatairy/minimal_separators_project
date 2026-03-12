import time

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
from causal.influence.estimator_bn import asymptotic_variance_for_Z
from causal.policies import static_do_policy

from experiments.create_buckets_graphs import make_bucket_boxplots

if __name__ == "__main__":

    import random

    N = 2 #number of different seeds
    nodes = [25]#, 50, 100, 500] # different sizes of nodes in  a tree
    prob_nodes = [0.10]  #the probability of an edge


    for node in nodes:
        for prob_node in prob_nodes:


            seeds_to_keep = []
            variance = f"BN_main_{node}_{prob_node}"
            file_name = f"outputs/26_2_23_seeds_data_{variance}.csv"
            bucket_file = f"outputs/26_2_23_bucket_statistics_{variance}.csv"
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
                    k_roots=20,
                    node_prefix="V",
                    seed=seed_graph
                )

                bn = generate_bn_binary_logistic(G, seed=seed_params)

                w, order = treewidth_upper_bound_ve(bn, heuristic="minfill")
                print("VE-style treewidth upper bound =", w)
                print("order =", order)

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
                    sample_k=20,
                    seed=seed)
                print("pairs:", pairs)

                test_mode = False

                R = list(bn.g.nodes())
                I = []

                results: Dict[Tuple[str, str], List[List[str]]] = {}
                results1: Dict[Tuple[str, ...], float] = {}

                for X, Y in pairs:
                    t0_start = time.perf_counter()
                    H, Z_sets = find_adjustment_sets_for_pair(bn.g, X, Y, R=R, I=I)
                    t0_end = time.perf_counter()
                    #print(f"Avg per run: {(t0_end - t0_start) / 100:.6f} seconds")
                    find_adjustment_sets_time = (t0_end - t0_start) / 100

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

                        for Z in Z_sets:
                            if len(Z) >= 1:
                                Z_key = tuple(sorted(Z))
                                t1_start = time.perf_counter()
                                sigma2 = asymptotic_variance_for_Z(
                                    bn, Y, X, Z, L_vars, policy_fn, None
                                )
                                t1_end = time.perf_counter()
                                calc_variance_time = (t0_end - t0_start) / 100
                                results1[Z_key] = sigma2

                            append_line(file_name,
                                        str(seed) + "," +
                                        str(num_nodes) + "," +
                                        str(num_edges) + "," +
                                        str(X) + "," +
                                        str(Y) + "," +
                                        str(h_num_nodes) + "," +
                                        str(h_num_edges) + "," +
                                        ";".join(map(str, Z)) + "," +
                                        str(round(sigma2, 5)) + ","+
                                        str(calc_variance_time) + "/n")
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

            save_list_json(seeds_to_keep, "outputs/seeds_BN_to_keep.json")

            write_bucket_stats_to_csv(bucket_file, buckets_results, False)

            make_bucket_boxplots(
                bucket_file,
                #output_dir=f"bucket_statistics_{variance}",
                mode="global",
            )
            make_bucket_boxplots(
                bucket_file,
                #output_dir=f"bucket_statistics_{variance}",
                mode="per_run",
            )


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
