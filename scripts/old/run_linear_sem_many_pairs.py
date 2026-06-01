import time

from typing import Dict, List, Tuple

from graph.separators import Z_contains_parent_of_X

from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy
from sem.variance import compute_avar_many_Z  #x_drain_two_sets

from i_o.utils import save_list_json, append_line,write_bucket_stats_to_csv
from i_o.json_loader import save_linear_sem

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs,extract_separator_containment_pairs, frozenset_to_str

from pipelines.adjust_sets import find_adjustment_sets_for_pair

from analysis.bucket_sep import bucket_separators_by_cy_layers, bucket_variance_statistics,bucket_separators_by_y_connectivity,bucket_y_component

from scripts.old.ex1_parallel_XY import process_all_xy_pairs_parallel, process_xy_pair_sem

if __name__ == "__main__":

    N = 5
    nodes = [10,20]#,30,40, 50]
    prob_nodes =[0.1,0.07, 0.15, 0.2, 0.3]#
    betas = [0.7]

    date = "2026_05_08"

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [int(node*0.3)]:#[node, int(node*0.3), 3, 1]:



                seeds_to_keep = []
                variance = f"_{node}_{prob_node}_{k_roots}"
                bucket_file = f"outputs_sem/{date}_bucket_statistics_{variance}.csv"
                write_bucket_stats_to_csv(bucket_file, [])
                buckets_results = {}

                file_name = f"outputs_sem/{date}_seeds_data_main_{variance}.csv"

                new_file_name = file_name.replace(".csv", "_seperators.csv")
                append_line(new_file_name,
                            "seed," +
                            "X," +
                            "Y," +
                            'outer_sep,' +
                            "outer_sep len," +
                             #"var_y_given_xz_out," +
                            # 'var_x_given_z_out,'+
                            # 'outer_component,' +
                            "outer_sep var," +
                            'inner_sep,' +
                            'inner_sep len,' +
                            #'var_y_given_xz_in,' +
                            #'var_x_given_z_in,' +
                            # 'inner_component,' +
                            'inner_sep var,' +
                            "diff sep var," +
                            "diff sep var > 0,"
                            "diff size"  # + "," +
                            #'X-Drain_in' + "," +
                            #'X-Drain_out' + ","
                            #"diff-drain"
                            )

                graph_path = "../outputs_sem/graph/"

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
                        sem.g,
                        mode="reachable",
                        seed=seed
                    )

                    test_mode: bool = False

                    R = list(sem.g.nodes())
                    I = []

                    results: Dict[Tuple[str, str], List[List[str]]] = {}
                    results1: Dict[Tuple[str, ...], float] = {}

                    pair_results = process_all_xy_pairs_parallel(
                        model=sem,
                        seed=seed,
                        pairs=pairs,
                        R=R,
                        I=I,
                        worker_fn=process_xy_pair_sem,
                        test_mode=False,
                    )

                    for result in pair_results:
                        seed = result["seed"]
                        X = result["X"]
                        Y = result["Y"]

                        Z_sets = result["Z_sets"]
                        res = result["res"]
                        results1 = result["results1"]
                        closest = result["closest"]

                        if len(Z_sets) == 0:
                            continue

                        print(f"seed={seed}, X={X}, Y={Y}, |Z_sets|={len(Z_sets)}")
                        print(f"Number of adjustment sets: {len(Z_sets)}")
                        print(f"Number of Hasse edges: {len(res['hasse_edges'])}")

                        pair_adjustment = extract_separator_containment_pairs(res)
                        print(f"Number of containment pairs: {len(pair_adjustment)}")

                        if tuple(sorted(closest[0])) not in results1:
                            print("closest[0] missing in results1")

                        if tuple(sorted(closest[1])) not in results1:
                            print("closest[1] missing in results1")

                        for hass in pair_adjustment:
                            outer_sep = hass['outer_sep']
                            inner_sep = hass['inner_sep']

                            outer_key = tuple(sorted(outer_sep))
                            inner_key = tuple(sorted(inner_sep))

                            outer_var = round(results1[outer_key], 5)
                            inner_var = round(results1[inner_key], 5)

                            diff = outer_var - inner_var
                            diff_negative = diff < 0
                            diff_size = len(outer_sep) - len(inner_sep)

                            append_line(
                                new_file_name,
                                str(seed) + "," +
                                str(X) + "," +
                                str(Y) + "," +
                                frozenset_to_str(outer_sep) + "," +
                                str(len(outer_sep)) + "," +
                                str(outer_var) + "," +
                                frozenset_to_str(inner_sep) + "," +
                                str(len(inner_sep)) + "," +
                                str(inner_var) + "," +
                                str(diff) + "," +
                                str(diff_negative) + "," +
                                str(diff_size)
                            )

                            if diff_negative:
                                print("SOMETHING IS WRONG")
                                print(f"seed={seed}, X={X}, Y={Y}")
                                print(f"outer={outer_sep}, inner={inner_sep}")
                                print(f"outer_var={outer_var}, inner_var={inner_var}")

                                curr_path = graph_path + f"graph_{seed}_{X}_{Y}.json"
                                save_linear_sem(curr_path, sem, pairs, len(Z_sets))

                            has_parent_outer, which_outer = Z_contains_parent_of_X(
                                sem.g, outer_sep, X
                            )
                            has_parent_inner, which_inner = Z_contains_parent_of_X(
                                sem.g, inner_sep, X
                            )

                            # אם צריך debug:
                            # print("outer has parent:", has_parent_outer, which_outer)
                            # print("inner has parent:", has_parent_inner, which_inner)

                        if not tuple(sorted(closest[0])) in results1:
                            print("problem")
                            print(closest[0])
                        if not tuple(sorted(closest[1])) in results1:
                            print("problem")
                            print(closest[1])

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
                            str(round(results1[tuple(sorted(closest[0]))], 5) - round(
                                results1[tuple(sorted(closest[1]))], 5))
                        )

                        bucket_input = result["bucket_input"]
                        buckets1 = bucket_input["buckets1"]
                        Z_to_component = res["Z_to_component"]

                        bucket_variance_statistics(
                            seed,
                            X,
                            Y,
                            buckets1,
                            results1,
                            Z_to_component,
                            buckets_results
                        )

                        print("bucket statistics updated")
                        print("buckets1:", buckets1)


                        # כאן כל append_line
                        # כאן bucket_variance_statistics
                        # כאן save_linear_sem במקרה בעייתי



                    for X, Y in pairs:
                        H, Z_sets,total_time, closest = find_adjustment_sets_for_pair(sem.g, X, Y,"smallminimalseps", R=R, I=I, get_closest_seps=True)

                        Z_sets = [Z for Z in Z_sets if len(Z) >= 1]

                        forward, reverse = cy_components_for_sets(H, Y, Z_sets)
                        #print(forward, reverse)
                        # get Hass graph for the Z - the adjustment sets
                        res = hasse_from_cy_results(forward, reverse)
                        #print("***************************************")
                        #print(res)
                        #print("***************************************")


                        num_nodes = sem.g.number_of_nodes()
                        num_edges = sem.g.number_of_edges()

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

                            # var_names, Sigma = sigma_from_sem(sem)
                            #
                            # for Z in Z_sets:
                            #     if len(Z) >= 1:
                            #         # Z_key = tuple(sorted(Z))
                            #         # aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
                            #         # results1[Z_key] = aVar
                            #
                            #         Z_list = list(Z)
                            #         Z_key = tuple(sorted(Z_list))
                            #         aVar = avar_henckel_single_xy(
                            #             Sigma=Sigma,
                            #             X=X,
                            #             Y=Y,
                            #             Z=Z_list,
                            #             var_names=var_names,
                            #             ridge=1e-10,
                            #         )
                            #         results1[Z_key] = aVar


                            start = time.perf_counter()
                            results1 = compute_avar_many_Z(sem, X=X, Y=Y, Z_sets=Z_sets)
                            elapsed = time.perf_counter() - start
                            print(f"old process, time={elapsed:.3f} sec")

                            #results[(X, Y)] = results1


                            print(f"len of Z_sets={len(Z_sets)}")
                            print(f"X={X},Y={Y},seed={seed}")
                            # for n_workers in [2, 4]:#[1, 2, 4, 5, os.cpu_count() - 1]:
                            #     start = time.perf_counter()
                            #
                            #     results1 = compute_avar_many_Z_parallel(
                            #         sem=sem,
                            #         X=X,
                            #         Y=Y,
                            #         Z_sets=Z_sets,
                            #         ridge=1e-10,
                            #         max_workers=n_workers,
                            #         chunksize=20,
                            #     )
                            #
                            #     elapsed = time.perf_counter() - start
                            #     print(f"workers={n_workers}, time={elapsed:.3f} sec")
                            #
                            #     results[(X, Y)] = results1





                            pair_adjustment = extract_separator_containment_pairs(res)

                            if not tuple(sorted(closest[0])) in results1:
                                print("problem")
                            if not tuple(sorted(closest[1])) in results1:
                                print("problem")

                            for hass in pair_adjustment:

                                # var_x_given_z_out = conditional_variance_from_cov(
                                #     Sigma, target=X, given=list(tuple(sorted(hass['outer_sep']))), var_names=var_names, ridge=1e-10)
                                # var_x_given_z_in = conditional_variance_from_cov(
                                #     Sigma, target=X, given=list(tuple(sorted(hass['inner_sep']))), var_names=var_names, ridge=1e-10)
                                #
                                # var_y_given_xz_out = conditional_variance_from_cov(
                                #     Sigma, target=Y, given=[X] + list(tuple(sorted(hass['outer_sep']))), var_names=var_names, ridge=1e-10
                                # )
                                # var_y_given_xz_in = conditional_variance_from_cov(
                                #     Sigma, target=Y, given=[X] + list(tuple(sorted(hass['inner_sep']))), var_names=var_names,
                                #     ridge=1e-10
                                # )

                                #drains = x_drain_two_sets(sem, X, hass['inner_sep'], hass['outer_sep'])





                                append_line(new_file_name,
                                            str(seed) + "," +
                                            str(X) + "," +
                                            str(Y) + "," +
                                            frozenset_to_str(hass['outer_sep']) + "," +
                                            str(len(hass['outer_sep'])) + "," +
                                            #str(var_y_given_xz_out) + ',' +
                                            #str(var_x_given_z_out)+','+
                                            #frozenset_to_str(hass['outer_component']) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5)) + "," +
                                            frozenset_to_str(hass['inner_sep']) + "," +
                                            str(len(hass['inner_sep'])) + "," +
                                            #str(var_y_given_xz_in) + ',' +
                                            #str(var_x_given_z_in) + ',' +
                                            #frozenset_to_str(hass['inner_component']) + "," +
                                            str(round(results1[tuple(sorted(hass['inner_sep']))], 5)) + "," +
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                                results1[tuple(sorted(hass['inner_sep']))], 5))+ ","+
                                            str(round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                                results1[tuple(sorted(hass['inner_sep']))], 5) < 0)+ "," +
                                            str(len(hass['outer_sep']) - len(hass['inner_sep'])) + "," +
                                            #str(drains['X-Drain(Z1)'])+","+
                                            #str(drains['X-Drain(Z2)']) + ","+
                                            #str(drains['X-Drain(Z2)']- drains['X-Drain(Z1)']) + ","+
                                            str(closest[0])+ ","+
                                            str(round(results1[tuple(sorted(closest[0]))],5)) + ","+
                                            str(closest[1])+ "," +
                                            str(round(results1[tuple(sorted(closest[1]))], 5))

                                            )


                                print(f"seed: {seed}, X: {X}, Y: {Y}")
                                has_parent_outer_sep, which = Z_contains_parent_of_X(sem.g, hass['outer_sep'], X)
                                #print("outer_sep contains parent of X?", has_parent_outer_sep, "parents in outer_sep:", which)

                                has_parent_inner_sep, which_inner_sep = Z_contains_parent_of_X(sem.g, hass['inner_sep'], X)
                                #print("inner_sep contains parent of X?", has_parent_inner_sep, "parents in inner_sep:", which)

                                if round(results1[tuple(sorted(hass['outer_sep']))], 5) - round(
                                                    results1[tuple(sorted(hass['inner_sep']))], 5) < 0:
                                    curr_path = graph_path+"graph_"+str(seed)+"_"+X+"_"+Y+".json"
                                    save_linear_sem(curr_path, sem, pairs, len(Z_sets))
                                    print("SOMETHING IS WRONG")


                        buckets = bucket_separators_by_cy_layers(H, Y, Z_sets)
                        print(buckets)

                        buckets1 = bucket_separators_by_y_connectivity(H, Y, Z_sets)
                        print(buckets)

                        something = bucket_y_component(buckets1, res['Z_to_component'])

                        bucket_variance_statistics(seed, X, Y, buckets1, results1, res['Z_to_component'], buckets_results)
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
