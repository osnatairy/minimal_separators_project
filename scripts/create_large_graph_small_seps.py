#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx

from sem.adjustment_wrapper import run_many_xy
from pipelines.adjust_sets import find_adjustment_sets_for_pair


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path("large_dags_many_adjustment_sets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_GRAPHS_TO_SAVE = 5
N_NODES = 70

MAX_WORKERS = max(1, (os.cpu_count() or 2) - 4)

SEEDS = range(1, 2)#300)

# layered DAG parameters
NUM_LAYERS = 7
EDGE_PROBS = [0.20]#, 0.20]#, 0.25, 0.30]

# limit pairs so it does not explode
MAX_PAIRS_TO_TEST = 80

METHOD = "smallminimalseps"


# ============================================================
# DAG GENERATION
# ============================================================

def make_layered_dag_many_adjustments(
    n: int,
    num_layers: int,
    edge_prob: float,
    seed: int,
    node_prefix: str = "V",
) -> nx.DiGraph:
    """
    Creates a directed acyclic graph with layers.

    Edges only go from earlier layers to later layers.
    This keeps the graph acyclic and tends to create many
    alternative paths, hence many adjustment sets.
    """

    rng = random.Random(seed)

    G = nx.DiGraph()
    nodes = [f"{node_prefix}{i}" for i in range(n)]
    G.add_nodes_from(nodes)

    # split nodes into layers
    layers = [[] for _ in range(num_layers)]
    for i, v in enumerate(nodes):
        layers[i % num_layers].append(v)

    # add forward edges between layers
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            distance = j - i

            # mostly connect nearby layers, fewer long edges
            p = edge_prob / distance

            for u in layers[i]:
                for v in layers[j]:
                    if rng.random() < p:
                        G.add_edge(u, v)

    # guarantee some connectivity between consecutive layers
    for i in range(num_layers - 1):
        left = layers[i]
        right = layers[i + 1]

        for u in left:
            v = rng.choice(right)
            G.add_edge(u, v)

        for v in right:
            u = rng.choice(left)
            G.add_edge(u, v)

    assert nx.is_directed_acyclic_graph(G)

    return G


# ============================================================
# ADJUSTMENT SET COUNTING
# ============================================================

def compute_adjustment_sets_for_pair(args):
    G, seed, X, Y, path_len, num_paths = args

    R = list(G.nodes())
    I = []

    try:
        H, Z_sets, total_time = find_adjustment_sets_for_pair(
            G,
            X,
            Y,
            METHOD,
            R=R,
            I=I,
            get_closest_seps=False,
            #K=8,
        )

        if Z_sets is None or len(Z_sets) == 0:
            zsets = []
        else:
            zsets = sorted({
                tuple(sorted(Z))
                for Z in Z_sets
                if 1 <= len(Z) <= 5
            })

        return {
            "seed": seed,
            "X": X,
            "Y": Y,
            "path_len": path_len,
            "num_paths": num_paths,
            "time": total_time,
            "num_zsets": len(zsets),
            "zsets": zsets,
        }

    except Exception as e:
        return {
            "seed": seed,
            "X": X,
            "Y": Y,
            "path_len": path_len,
            "num_paths": num_paths,
            "time": None,
            "num_zsets": 0,
            "zsets": [],
            "error": str(e),
        }
'''
def compute_adjustment_sets_for_pair(args):
    G, seed, X, Y = args

    R = list(G.nodes())
    I = []

    try:
        H, Z_sets, total_time = find_adjustment_sets_for_pair(
            G,
            X,
            Y,
            METHOD,
            R=R,
            I=I,
            get_closest_seps=False,
            K = 8

        )

        if Z_sets is None:
            zsets = []
        else:
            zsets = sorted({
                tuple(sorted(Z))
                for Z in Z_sets
                if len(Z) >= 1
            })

        return {
            "seed": seed,
            "X": X,
            "Y": Y,
            "time": total_time,
            "num_zsets": len(zsets),
            "zsets": zsets,
        }

    except Exception as e:
        return {
            "seed": seed,
            "X": X,
            "Y": Y,
            "time": None,
            "num_zsets": 0,
            "zsets": [],
            "error": str(e),
        }
'''

def score_graph(G: nx.DiGraph, seed: int):
    raw_pairs = list(run_many_xy(G, mode="reachable", seed=seed))

    filtered_pairs, longest_path_len, min_path_threshold = filter_pairs_by_path_length_and_count(
        G,
        raw_pairs,
        min_fraction=0.5,
        min_paths=20,
        path_cap=200,
    )

    print(
        f"  longest_shortest_path={longest_path_len}, "
        f"min_path_threshold={min_path_threshold}, "
        f"candidate_pairs={len(filtered_pairs)}",
        flush=True,
    )

    random.Random(seed).shuffle(filtered_pairs)
    filtered_pairs = filtered_pairs[:MAX_PAIRS_TO_TEST]

    jobs = [
        (G, seed, X, Y, path_len, num_paths)
        for X, Y, path_len, num_paths in filtered_pairs
    ]

    results = []

    if jobs:
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
            futures = [ex.submit(compute_adjustment_sets_for_pair, job) for job in jobs]

            for fut in as_completed(futures):
                results.append(fut.result())

    # Keep only pairs with at least 50 small adjustment sets
    MIN_ZSETS_PER_PAIR = 20

    results = [
        r for r in results
        if r["num_zsets"] >= MIN_ZSETS_PER_PAIR
    ]

    total_zsets = sum(r["num_zsets"] for r in results)
    max_zsets_for_pair = max([r["num_zsets"] for r in results], default=0)
    useful_pairs = len(results)

    return {
        "seed": seed,
        "total_zsets": total_zsets,
        "max_zsets_for_pair": max_zsets_for_pair,
        "useful_pairs": useful_pairs,
        "pairs_tested": len(filtered_pairs),
        "longest_shortest_path": longest_path_len,
        "min_path_threshold": min_path_threshold,
        "results": results,
    }
'''
def score_graph(G: nx.DiGraph, seed: int):
    pairs = run_many_xy(G, mode="reachable", seed=seed)

    # sample pairs so evaluation is not too slow
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:MAX_PAIRS_TO_TEST]

    jobs = [(G, seed, X, Y) for X, Y in pairs]

    results = []

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
        futures = [ex.submit(compute_adjustment_sets_for_pair, job) for job in jobs]

        for fut in as_completed(futures):
            results.append(fut.result())

    total_zsets = sum(r["num_zsets"] for r in results)
    max_zsets_for_pair = max([r["num_zsets"] for r in results], default=0)
    useful_pairs = sum(1 for r in results if r["num_zsets"] > 0)

    return {
        "seed": seed,
        "total_zsets": total_zsets,
        "max_zsets_for_pair": max_zsets_for_pair,
        "useful_pairs": useful_pairs,
        "pairs_tested": len(pairs),
        "results": results,
    }
'''

# ============================================================
# SAVING
# ============================================================

def save_graph_json(path: Path, G: nx.DiGraph, stats: dict):
    payload = {
        "nodes": list(G.nodes()),
        "edges": [[u, v] for u, v in G.edges()],
        "stats": {
            "seed": stats["seed"],
            "total_zsets": stats["total_zsets"],
            "max_zsets_for_pair": stats["max_zsets_for_pair"],
            "useful_pairs": stats["useful_pairs"],
            "pairs_tested": stats["pairs_tested"],
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

'''
def save_adjustment_sets_csv(path: Path, stats: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "X", "Y", "time", "num_values", "len_z", "Z"])

        for r in stats["results"]:
            for z in r["zsets"]:
                writer.writerow([
                    r["seed"],
                    r["X"],
                    r["Y"],
                    r["time"],
                    r["num_zsets"],
                    len(z),
                    list(z),
                ])
'''

def save_adjustment_sets_csv(path: Path, stats: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed",
            "X",
            "Y",
            "path_len",
            "num_paths_capped",
            "time",
            "num_values",
            "len_z",
            "Z",
        ])

        for r in stats["results"]:
            for z in r["zsets"]:
                writer.writerow([
                    r["seed"],
                    r["X"],
                    r["Y"],
                    r["path_len"],
                    r["num_paths"],
                    r["time"],
                    r["num_zsets"],
                    len(z),
                    list(z),
                ])
# ============================================================
# ADDITIONAL HEURISTIC FUNCTION
# ============================================================
def longest_shortest_path_length_dag(G: nx.DiGraph) -> int:
    max_len = 0

    for source, dist_map in nx.all_pairs_shortest_path_length(G):
        for target, dist in dist_map.items():
            if source != target:
                max_len = max(max_len, dist)

    return max_len


def count_paths_capped(
    G: nx.DiGraph,
    X: str,
    Y: str,
    cap: int = 200,
) -> int:
    """
    Counts simple directed paths from X to Y, but stops at cap.
    This avoids exploding on very connected DAGs.
    """
    count = 0

    try:
        for _ in nx.all_simple_paths(G, source=X, target=Y):
            count += 1
            if count >= cap:
                return cap
    except nx.NetworkXNoPath:
        return 0

    return count


def filter_pairs_by_path_length_and_count(
    G: nx.DiGraph,
    pairs,
    min_fraction: float = 0.5,
    min_paths: int = 20,
    path_cap: int = 200,
):
    longest = longest_shortest_path_length_dag(G)
    threshold = max(1, int(longest * min_fraction))

    kept = []

    for X, Y in pairs:
        try:
            path_len = nx.shortest_path_length(G, X, Y)
        except nx.NetworkXNoPath:
            continue

        if path_len < threshold:
            continue

        num_paths = count_paths_capped(G, X, Y, cap=path_cap)

        if num_paths >= min_paths:
            kept.append((X, Y, path_len, num_paths))

    return kept, longest, threshold


# ============================================================
# MAIN SEARCH
# ============================================================

def main():
    best = []

    for seed in SEEDS:
        for edge_prob in EDGE_PROBS:
            graph_seed = seed * 1000 + int(edge_prob * 1000)

            print(f"Checking seed={seed}, edge_prob={edge_prob}", flush=True)

            G = make_layered_dag_many_adjustments(
                n=N_NODES,
                num_layers=NUM_LAYERS,
                edge_prob=edge_prob,
                seed=graph_seed,
            )

            stats = score_graph(G, seed=seed)

            print(
                f"  total_zsets={stats['total_zsets']}, "
                f"max_pair={stats['max_zsets_for_pair']}, "
                f"useful_pairs={stats['useful_pairs']}/{stats['pairs_tested']}",
                flush=True,
            )

            best.append((stats["total_zsets"], stats, G))
            best = sorted(best, key=lambda x: x[0], reverse=True)[:N_GRAPHS_TO_SAVE]

            # save current best every time
            for idx, (_, best_stats, best_G) in enumerate(best, start=1):
                graph_path = OUT_DIR / f"best_{idx}_graph_seed_{best_stats['seed']}_zsets_{best_stats['total_zsets']}.json"
                csv_path = OUT_DIR / f"best_{idx}_adjustment_sets_seed_{best_stats['seed']}_zsets_{best_stats['total_zsets']}.csv"

                save_graph_json(graph_path, best_G, best_stats)
                save_adjustment_sets_csv(csv_path, best_stats)

    print("Done.")
    print(f"Saved best graphs to: {OUT_DIR}")


if __name__ == "__main__":
    main()