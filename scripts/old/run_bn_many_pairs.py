import time

from typing import Dict, Tuple

from sem.adjustment_wrapper import run_many_xy
from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from i_o.utils import save_list_json, append_line,write_bucket_stats_to_csv

import utils as utils
# analysis for the HASS diagram
from analysis.adjustment_hasse import extract_separator_containment_pairs, frozenset_to_str

from analysis.bucket_sep import bucket_variance_statistics

from analysis.treewidth import treewidth_upper_bound_ve

#influance function calculation

#from scripts.ex1_parallel_run import process_all_xy_pairs_parallel
from scripts.old.ex1_parallel_XY import process_xy_pair_bn, process_all_xy_pairs_parallel

if __name__ == "__main__":

    N = 10 #number of different seeds
    nodes = [20]#,30,40, 50]# different sizes of nodes in  a tree
    prob_nodes = [0.07, 0.1, 0.15, 0.2]  #the probability of an edge

    date = "2026_05_08"

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [int(node*0.3)]:# [node, int(node*0.3), 3, 1]:

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
                    '''
                    num_roots, roots = count_roots(G)
                    depth = graph_depth(G)
                    degree = degree_stats(G)
                    avg_distance_from_root = avg_distance_from_roots(G)
                    layer_level = level_layer(G)

                    # print("Number of roots:", num_roots)
                    # print("Roots:", roots)
                    print(f"{seed} ,{node} ,{prob_node} , {k_roots} , {num_roots}  ,{depth} ,{tuple(float(x) for x in degree)}  ,{str(avg_distance_from_root)}")#) level_layer: {layer_level}")
                    '''
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
                    # pair_results = process_all_xy_pairs_parallel(
                    #     bn=bn,
                    #     seed=seed,
                    #     pairs=pairs,
                    #     R=R,
                    #     I=I,
                    #     test_mode=test_mode,
                    # )

                    pair_results = process_all_xy_pairs_parallel(
                        model=bn,
                        seed=seed,
                        pairs=pairs,
                        R=R,
                        I=I,
                        worker_fn = process_xy_pair_bn,
                        test_mode=False,
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

                        elif len(pair_result['Z_sets']) > 0:

                            results1 = pair_result["results1"]
                            results[(X, Y)] = results1
                            closest = pair_result["closest"]

                            pair_adjustment = extract_separator_containment_pairs(pair_result["res"])
                            new_file_name = file_name.replace(".csv", "_seperators.csv")
                            for hass in pair_adjustment:
                                append_line(
                                    new_file_name,
                                    str(seed) + "," +
                                    str(X) + "," +
                                    str(Y) + "," +
                                    frozenset_to_str(hass['outer_sep']) + "," +
                                    str(len((hass['outer_sep']))) + "," +
                                    frozenset_to_str(hass['outer_component']) + "," +
                                    str(len((hass['outer_component']))) + "," +
                                    str(round(results1[tuple(sorted(hass['outer_sep']))], 5)) + "," +
                                    frozenset_to_str(hass['inner_sep']) + "," +
                                    str(len((hass['inner_sep']))) + "," +
                                    frozenset_to_str(hass['inner_component']) + "," +
                                    str(len((hass['inner_component']))) + "," +
                                    str(round(results1[tuple(sorted(hass['inner_sep']))], 5)) + "," +
                                    str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                        results1[tuple(sorted(hass['inner_sep']))], 5))
                                )

                            if not tuple(sorted(closest[0])) in results1:
                                print("problem")
                            if not tuple(sorted(closest[1])) in results1:
                                print("problem")

                            new_file_name2 = file_name.replace(".csv", "_farthest_closest.csv")
                            append_line(
                                new_file_name2,
                                str(seed) + "," +
                                str(X) + "," +
                                str(Y) + "," +
                                ";".join(closest[0]) + "," +
                                str(len((closest[0]))) + "," +
                                str(round(results1[tuple(sorted(closest[0]))], 5)) + "," +
                                ";".join(closest[1]) + "," +
                                str(len((closest[1]))) + "," +
                                str(round(results1[tuple(sorted(closest[1]))], 5)) + "," +
                                str(round(results1[tuple(sorted(closest[0]))], 5) - round(results1[tuple(sorted(closest[1]))], 5))
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

                save_list_json(seeds_to_keep, "outputs_bn/seeds_to_keep.json")

                write_bucket_stats_to_csv(bucket_file, buckets_results, False)

    print("End of this script")

print("End of this script")
