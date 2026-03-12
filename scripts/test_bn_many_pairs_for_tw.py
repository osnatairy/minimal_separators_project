import time

from typing import Dict, List, Tuple, Optional

from sem.adjustment_wrapper import run_many_xy
from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from i_o.utils import save_list_json, append_line,write_bucket_stats_to_csv

import utils as utils

from analysis.treewidth import treewidth_upper_bound_ve


if __name__ == "__main__":

    import random

    tw_file = f"outputs/11_3_23_tw2.csv"

    append_line(tw_file,
                "seed, nodes, prob_edges,edges, tw\n")

    N = 20 #number of different seeds
    nodes = [25, 50, 75] # different sizes of nodes in  a tree
    prob_nodes = [0.07,0.10,0.15]  #the probability of an edge


    for node in nodes:
        for prob_node in prob_nodes:


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
                #print("order =", order)

                # עכשיו bn הוא אובייקט BN מלא:
                #print("nodes:", list(bn.g.nodes()))
                #print("edges:", list(bn.g.edges()))

                append_line(tw_file,
                        str(seed)+","+
                        str(node)+","+
                        str(prob_node) + "," +
                        str(len(bn.g.edges())) + "," +
                        str(w)



                )
