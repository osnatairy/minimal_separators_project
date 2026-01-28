import os
import networkx as nx
import enum_algorithms
import random
import datetime

import utils_ms
from utils import (visualize_g, randomize_graph, add_weights, randomize_st, reset_random_seed,
                      gen_all_possible_st, find_non_minimal_st_sep, bail, timeout_manager, import_graph_from_gr)
from itertools import combinations
import time
import math
import multiprocessing
import json
import platform


machine_name = platform.node()
if machine_name == 'Osnat_Drien':
    pace_files_path = "C:\\Users\\Osnat\\Dropbox\\Treewidth-PACE-2017-instances-master\\gr\\exact\\"

def start_algorithm(g: nx.Graph):
    K= 5
    event = multiprocessing.Event()
    enum_algorithms.start_time = time.time()
    p = multiprocessing.Process(target=timeout_manager,
                                args=(event, enum_algorithms.start_time, 5))
    p.start()
    seperators1, t1, total_time1 = enum_algorithms.SmallMinimalSeps(g, K, event)
    if p.is_alive():
        p.terminate()
        p.join()

    return seperators1

    print('Listing all important minimal seperators smaller than some k')
    print("Run started at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Machine name: {machine_name}")

    with open('config_test_SmallMinimalSeps.json', 'r') as f:
        configuration = json.load(f)
    run_number = configuration['RUN_NUMBER']
    use_pace_file = configuration['USE_PACE_FILE']
    rand_st = configuration['RAND_ST'] # set to a tuple of two integers to force the st pair, or leave empty for random
    edge_prob = configuration['EDGE_PROB']
    weighted = configuration['WEIGHTED']
    num_nodes = configuration['NUM_NODES']
    K = configuration['K']
    num_edges = configuration['NUM_EDGES']
    full_coverage = configuration['FULL_COVERAGE']
    use_random_edges = configuration['USE_RANDOM_EDGES']
    max_iterations = configuration['MAX_ITERATIONS']
    visualize = configuration['VISUALIZE']
    limit_runtime = configuration['LIMIT_RUNTIME']
    skip1 = configuration['SKIP1'] #set to 1 to skip the SmallMinimalSeps algorithm
    skip2 = configuration['SKIP2'] #set to 1 to skip the Takata algorithm
    do_check_for_bad_seps = configuration['DO_CHECK_FOR_BAD_SEPS']
    result_filename = configuration['RESULT_FILENAME']
    # print configuration for reference
    for key in configuration:
        print(repr(key), ":", configuration[key])

    reset_random_seed(run_number)
    start = time.time()

    if full_coverage:
        print(f"Running through all applicable graphs with {num_nodes} nodes")
    else:
        print(f"Using a random graph")

    if weighted:
        print(f"Weighted graph is not supported yet")
        exit(-1)
    else:
        print(f"Graph is unweighted- minimum cardinality will be calculated")

    if edge_prob > 1 or edge_prob <= 0:
        print("Error: edge probability should be inside (0,1]")
        exit(-1)

    num_nodes = g.number_of_nodes()
    edges = list(combinations(range(num_nodes), 2))  # all pairs in N
    num_possible_edges = len(edges)

    print(f"Number of nodes: {len(g.nodes())}")
    nx.write_gml(g, "last_g.gml")  # save for next time

    total_count = 0
    if num_edges < 0:
        num_edges_iter = range(num_possible_edges)  # use graphs with all possible combinations of edges
    else:
        num_edges_iter = [num_edges]  # use only the graphs with the required number of edges

    for k in range(len(num_edges_iter)):
        if total_count == max_iterations:
            break

        num_edges = num_edges_iter[k]
        if edges == [None] or edges == None:
            edges_comb_iter = [0]  # not needed
            num_combinations = 1
        else:
            edges_comb_iter = combinations(edges, num_edges)
            num_combinations = math.comb(len(edges), num_edges)

        for n in range(num_combinations):
            if total_count == max_iterations:
                break

            edges_choice = edges_comb_iter.__next__()

            if full_coverage:
                g.clear_edges()
                g.add_edges_from(edges_choice)
                st_list = gen_all_possible_st(g)
            else:
                st_list = [0]  # not needed
            for st in st_list:
                if total_count == max_iterations:
                    break
                total_count += 1
                if full_coverage:
                    g.graph['st'] = st
                # visualize
                if visualize:
                    visualize_g(g)
                    print(f"Seperating s = {g.graph['st'][0]} from t = {g.graph['st'][1]}")
                # use the algorithm to enumerate separators

                # enum_algorithms.gen_cuts_recursive_result = []
                print(f"Run time limited to {limit_runtime} seconds")
                if not skip1:

                    event = multiprocessing.Event()
                    enum_algorithms.start_time = time.time()
                    p = multiprocessing.Process(target=timeout_manager,
                                                args=(event, enum_algorithms.start_time, 5))
                    p.start()
                    seperators1, t1, total_time1 = enum_algorithms.SmallMinimalSeps(g, K, event)
                    if p.is_alive():
                        p.terminate()
                        p.join()
                else:
                    seperators1 = []
                    t1 = []
                    total_time1 = None
                if seperators1 == None:
                    print(f"Problem with the first algorithm at count {total_count}")
                    exit(-1)

                if not skip2:
                    event = multiprocessing.Event()
                    enum_algorithms.start_time = time.time()
                    p = multiprocessing.Process(target=timeout_manager,
                                                args=(event, enum_algorithms.start_time, limit_runtime))
                    p.start()
                    seperators2, t2, total_time2 = enum_algorithms.ListMinSepTakata(g, event)
                    if p.is_alive():
                        p.terminate()
                        p.join()
                else:
                    seperators2 = []
                    t2 = []
                    total_time2 = None

                sep1 = set([frozenset(x) for x in seperators1])
                sep2 = set([frozenset(x) for x in seperators2])
                not_in_sep1 = sep2 - sep1
                not_in_sep2 = sep1 - sep2

                seperators1_len = []
                seperators2_len = []
                # print("seperators1")
                for s in seperators1:
                    seperators1_len.append(len(s))
                    # print(s)
                # print("seperators2")
                for s in seperators2:
                    seperators2_len.append(len(s))
                    # print(s)

                os.makedirs(os.path.dirname(result_filename), exist_ok=True)
                with open(result_filename, 'w') as f1:
                    json.dump(
                        [list(map(list, seperators1)), list(map(list, seperators2)), seperators1_len, seperators2_len,
                         t1, t2, K, g.graph['st'], total_time1, total_time2], f1, indent=2)

                nomatch = None
                if skip1 == 0 and skip2 == 0:
                    nomatch = 0
                    if not_in_sep1 or not_in_sep2:
                        print(f'Not in seperators1: {not_in_sep1}')
                        print(f'Not in seperators2: {not_in_sep2}')
                        nomatch = 1
                    if nomatch:
                        print(f"Seperators don't match at count {total_count}:")

                bad_seps = None

                if do_check_for_bad_seps:
                    # Checking if the sets are not separators or not minimal, if the algorithms are correct this should never happen
                    bad_seps = 0
                    s = find_non_minimal_st_sep(g, seperators1)
                    if len(s['NOT_SEP']) > 0 or len(s['NOT_MIN']) > 0:
                        print(f"in seperators1: non separators {s['NOT_SEP']}")
                        print(f"in seperators1: non minimal separators {s['NOT_SEP']}")
                        bad_seps = 1
                    s = find_non_minimal_st_sep(g, seperators2)
                    if len(s['NOT_SEP']) > 0 or len(s['NOT_MIN']) > 0:
                        print(f"in seperators2: non separators {s['NOT_SEP']}")
                        print(f"in seperators2: non minimal separators {s['NOT_SEP']}")
                        bad_seps = 1
                    if bad_seps:
                        print(f"Bad separators found at count {total_count}")
                    else:
                        print(f"No bad seperators")

    end = time.time()
    print(f'Run time: {end - start} seconds')
    if total_count == max_iterations:
        print(f"Maximum number of iterations of {max_iterations} reached")

    if bad_seps or nomatch:
        print(f"compare failed")
    else:
        print(f"compare ok")
    return list(map(list, seperators1))
