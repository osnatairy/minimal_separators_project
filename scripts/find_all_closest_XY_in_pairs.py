import time
import os
import csv
import json
import utils

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Tuple, Dict, List

from collections import defaultdict
from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy,all_direct_xy_pairs
from pipelines.adjust_sets import find_closest_seps_for_pair

from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient

# Globals inside each worker process
_WORKER_G = None
_WORKER_R = None
_WORKER_I = None
_WORKER_SEED = None

def _init_build_indices_worker(G, seed):
    """
    Runs once per worker process.
    Stores the graph and fixed data locally inside the worker.
    """
    global _WORKER_G, _WORKER_R, _WORKER_I, _WORKER_SEED

    _WORKER_G = G
    _WORKER_R = list(G.nodes())
    _WORKER_I = []
    _WORKER_SEED = seed


def _compute_xy_separators(pair):
    """
    Worker function.
    Computes all separators for one (X, Y) pair.
    """
    X, Y = pair

    closest = find_closest_seps_for_pair(
        _WORKER_G,
        X,
        Y,
        "smallminimalseps",
        R=_WORKER_R,
        I=_WORKER_I,
    )

    xy_key = (_WORKER_SEED, X, Y)

    return xy_key, closest

def build_indices_for_seed_parallel(
    modelGraph,
    seed: int,
    pairs: Iterable[Tuple[str, str]],
    max_workers: int | None = None,
    chunksize: int = 1,
):
    """
    Builds in parallel:
      1. xy_to_zsets:  (seed, X, Y, total_time) -> [Z1, Z2, ...]
      2. zset_to_xys:  (seed, Z) -> [(X1,Y1), (X2,Y2), ...]

    Each (X, Y) pair is computed in parallel.
    """

    pairs = list(pairs)

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(pairs), max(1, cpu_count - 1))

    xy_to_zsets = {}
    zset_to_xys = defaultdict(list)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_build_indices_worker,
        initargs=(modelGraph.g, seed),
    ) as executor:

        for xy_key, normalized_zsets in executor.map(
            _compute_xy_separators,
            pairs,
            chunksize=chunksize
        ):
            xy_to_zsets[xy_key] = normalized_zsets

    return xy_to_zsets


def write_xy_to_zsets_csv(xy_to_zsets, filename):
    """
    Writes rows:
      seed, X, Y, num_values, values_json
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "X", "Y", "closest_X", "closest_Y"])

        for (seed, X, Y), closest in sorted(xy_to_zsets.items()):

            writer.writerow([
                seed,
                X,
                Y,
                closest[0],
                closest[1]
            ])


if __name__ == "__main__":

    import random

    N = 10
    nodes = [40, 50]#10,20,
    prob_nodes = [  0.3]#0.07, 0.1, 0.15,
    betas = [0.7]
    model = "sem" #"bn" #

    date = "2026_04_26"

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [int(node * 0.3)]:  # [node, int(node*0.3), 3, 1]:

                variance = f"_{model}_{node}_{prob_node}"#_{k_roots}"

                all_xy_to_zsets = {}
                all_zset_to_xys = defaultdict(list)


                for seed in range(1,11):

                    if model == "sem":
                    # 1) Generate a linear SEM (your code)
                        modelGraph = make_linear_sem(
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
                    else:
                        seed_graph, seed_params = utils.split_seeds(seed)
                        G = spanning_tree_then_orient(
                            n=node,
                            prob_edge=prob_node,
                            k_roots=k_roots,
                            node_prefix="V",
                            seed=seed_graph
                        )

                        modelGraph = generate_bn_binary_logistic(G, seed=seed_params)

                    # 2) Run over many (X,Y) pairs and find adjustment sets
                    pairs = run_many_xy(
                        modelGraph.g,
                        mode="reachable",
                        seed=seed
                    )
                    direct_edge = all_direct_xy_pairs(modelGraph.g)
                    workers = utils.choose_num_workers(len(pairs))
                    xy_to_zsets = build_indices_for_seed_parallel(
                        modelGraph=modelGraph,
                        seed=seed,
                        pairs=pairs,
                        max_workers=workers,
                        chunksize=1
                    )

                    all_xy_to_zsets.update(xy_to_zsets)

                write_xy_to_zsets_csv(all_xy_to_zsets, f"xy_to_zsets_with_time/closest_seps{variance}.csv")

    print("End of this script")
