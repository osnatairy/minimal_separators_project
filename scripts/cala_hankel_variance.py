#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import utils

from sem.linear_sem import make_linear_sem
from sem.variance import sigma_from_sem, avar_henckel_single_xy

from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from causal.influence.estimator_bn_sampling import asymptotic_variance_for_Z_from_samples
from causal.influence.estimator_bn import asymptotic_variance_for_Z_auto
from causal.policies import static_do_policy


# ============================================================
# CONFIG
# ============================================================
BN_VARIANCE_METHOD = "exact"
# אפשרויות:
# "sampling" = חישוב לפי דגימות
# "exact"    = חישוב מדויק


INPUT_DIR = Path("important_seps")
OUTPUT_DIR = Path(f"variance_hankel_seps_{BN_VARIANCE_METHOD}")

# True = לקרוא את כל הקבצים:
# hankel_seps_bn_*.csv
# hankel_seps_sem_*.csv
USE_ALL_MATCHING_FILES = False


# רלוונטי רק אם USE_ALL_MATCHING_FILES = False
MODEL = "bn"  # "sem" or "bn"
NODES_LIST = [ 30]#, 40, 50]
PROB_NODES = [ 0.2, 0.3]

HARD_CAP_WORKERS = 8
RESERVE_CPUS = 1

csv.field_size_limit(10**7)


# ============================================================
# File names
# ============================================================

FILENAME_RE = re.compile(
    r"^hankel_seps_(?P<model>bn|sem)_(?P<n_nodes>\d+)_(?P<p>\d+(?:\.\d+)?)\.csv$"
)


def parse_file_meta(path: Path) -> dict:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Bad file name: {path.name}")

    return {
        "model": MODEL,
        "n_nodes": int(m.group("n_nodes")),
        "prob_node": float(m.group("p")),
    }


def output_path_for(input_path: Path) -> Path:
    input_name = input_path.name.replace("_sem", f"_{MODEL}")
    return OUTPUT_DIR / f"variance_{input_name}"


def probability_tokens(p: float) -> list[str]:
    raw = str(p)
    fixed2 = f"{p:.2f}"
    fixed3 = f"{p:.3f}"
    return list(dict.fromkeys([
        raw,
        fixed2.rstrip("0").rstrip("."),
        fixed3.rstrip("0").rstrip("."),
        fixed2,
        fixed3,
    ]))


def collect_input_files() -> list[Path]:
    if USE_ALL_MATCHING_FILES:
        return sorted(
            p for p in INPUT_DIR.glob("hankel_seps_*.csv")
            if FILENAME_RE.match(p.name)
        )

    paths = []
    for n in NODES_LIST:
        for p in PROB_NODES:
            for token in probability_tokens(p):
                candidate = INPUT_DIR / f"hankel_seps_sem_{n}_{token}.csv"
                if candidate.exists():
                    paths.append(candidate)
                    break

    return paths


# ============================================================
# Parsing
# ============================================================

def parse_separator(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return tuple()

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = text

    if parsed is None:
        return tuple()

    if isinstance(parsed, (list, tuple, set)):
        return tuple(sorted(str(v).strip() for v in parsed if str(v).strip()))

    if isinstance(parsed, str):
        if ";" in parsed:
            return tuple(sorted(x.strip() for x in parsed.split(";") if x.strip()))
        if "," in parsed and not parsed.startswith("V"):
            return tuple(sorted(x.strip().strip("'\"") for x in parsed.split(",") if x.strip()))
        return (parsed.strip(),) if parsed.strip() else tuple()

    return tuple()


def read_hankel_tasks(input_path: Path) -> list[dict]:
    meta = parse_file_meta(input_path)
    k_roots = max(1, int(meta["n_nodes"] * 0.3))

    tasks = []

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"seed", "X", "Y", "hankel_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_path.name} missing columns: {missing}")

        for row_index, row in enumerate(reader, start=1):
            seed = int(row["seed"])
            X = str(row["X"]).strip()
            Y = str(row["Y"]).strip()
            Z = parse_separator(row["hankel_z"])

            if not Z:
                continue

            tasks.append({
                "source_file": input_path.name,
                "row_index": row_index,
                "model": meta["model"],
                "n_nodes": meta["n_nodes"],
                "prob_node": meta["prob_node"],
                "k_roots": k_roots,
                "seed": seed,
                "X": X,
                "Y": Y,
                "Z": Z,
            })

    return tasks


def group_tasks_by_seed(tasks: list[dict]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["seed"]].append(task)
    return list(grouped.values())


# ============================================================
# Model reconstruction
# ============================================================

def build_model_graph_for_seed(
    model: str,
    n_nodes: int,
    prob_node: float,
    seed: int,
    k_roots: int,
):
    if model == "sem":
        return make_linear_sem(
            n=n_nodes,
            edge_prob=prob_node,
            beta_scale=1.0,
            sigma2_low=0.2,
            sigma2_high=1.0,
            node_prefix="V",
            seed=seed,
            k_roots=k_roots,
        )

    if model == "bn":
        seed_graph, seed_params = utils.split_seeds(seed)

        G = spanning_tree_then_orient(
            n=n_nodes,
            prob_edge=prob_node,
            k_roots=k_roots,
            node_prefix="V",
            seed=seed_graph,
        )

        return generate_bn_binary_logistic(G, seed=seed_params)

    raise ValueError(f"Unsupported model: {model}")


# ============================================================
# Variance calculations
# ============================================================

def choose_n_samples(
    num_nodes: int,
    z_size: int,
    base_samples: int = 20_000,
    max_samples: int = 200_000,
) -> int:
    if num_nodes <= 10:
        samples = 20_000
    elif num_nodes <= 20:
        samples = 50_000
    elif num_nodes <= 30:
        samples = 100_000
    elif num_nodes <= 40:
        samples = 150_000
    else:
        samples = base_samples

    multiplier = min(2.5, 1 + 0.3 * z_size)
    return min(int(samples * multiplier), max_samples)


def compute_sem_variance(model_graph: Any, X: str, Y: str, Z: tuple[str, ...], var_names, Sigma) -> float:
    return avar_henckel_single_xy(
        Sigma=Sigma,
        X=X,
        Y=Y,
        Z=list(Z),
        var_names=var_names,
        ridge=1e-10,
    )


def compute_bn_variance_from_samples(
    model_graph: Any,
    df_samples,
    X: str,
    Y: str,
    Z: tuple[str, ...],
) -> float:
    return asymptotic_variance_for_Z_from_samples(
        df_samples=df_samples,
        bn=model_graph,
        Y=Y,
        A_name=X,
        Z_vars=list(Z),
        L_vars=[],
        policy_fn=static_do_policy(a_star=1),
        value_map=None,
    )


def compute_bn_variance_exact(
    model_graph: Any,
    X: str,
    Y: str,
    Z: tuple[str, ...],
) -> float:
    return asymptotic_variance_for_Z_auto(
        model_graph,
        Y,
        X,
        list(Z),
        [],
        static_do_policy(a_star=1),
        method="exact",
    )

def compute_seed_batch(seed_tasks: list[dict]) -> list[dict]:
    first = seed_tasks[0]

    model_name = first["model"]
    n_nodes = first["n_nodes"]
    prob_node = first["prob_node"]
    seed = first["seed"]
    k_roots = first["k_roots"]

    model_graph = build_model_graph_for_seed(
        model=model_name,
        n_nodes=n_nodes,
        prob_node=prob_node,
        seed=seed,
        k_roots=k_roots,
    )

    if model_name == "sem":
        var_names, Sigma = sigma_from_sem(model_graph)
        df_samples = None
    else:
        var_names = None
        Sigma = None

        if BN_VARIANCE_METHOD == "sampling":
            max_z_size = max(len(task["Z"]) for task in seed_tasks)
            n_samples = choose_n_samples(
                num_nodes=len(model_graph.g.nodes()),
                z_size=max_z_size,
            )

            df_samples = model_graph.sample(
                n_samples=n_samples,
                seed=seed,
            )

        elif BN_VARIANCE_METHOD == "exact":
            df_samples = None

        else:
            raise ValueError('BN_VARIANCE_METHOD must be "sampling" or "exact"')

    rows = []

    for task in seed_tasks:
        X = task["X"]
        Y = task["Y"]
        Z = task["Z"]

        try:
            if model_name == "sem":
                variance = compute_sem_variance(
                    model_graph=model_graph,
                    X=X,
                    Y=Y,
                    Z=Z,
                    var_names=var_names,
                    Sigma=Sigma,
                )
            else:
                if BN_VARIANCE_METHOD == "sampling":
                    variance = compute_bn_variance_from_samples(
                        model_graph=model_graph,
                        df_samples=df_samples,
                        X=X,
                        Y=Y,
                        Z=Z,
                    )

                elif BN_VARIANCE_METHOD == "exact":
                    variance = compute_bn_variance_exact(
                        model_graph=model_graph,
                        X=X,
                        Y=Y,
                        Z=Z,
                    )

                else:
                    raise ValueError('BN_VARIANCE_METHOD must be "sampling" or "exact"')

            status = "ok"
            error = ""

        except Exception as e:
            variance = math.nan
            status = "error"
            error = repr(e)

        rows.append({
            "source_file": task["source_file"],
            "model": model_name,
            "n_nodes": n_nodes,
            "prob_node": prob_node,
            "k_roots": k_roots,
            "seed": seed,
            "X": X,
            "Y": Y,
            "len_z": len(Z),
            "hankel_z": json.dumps(list(Z), ensure_ascii=False),
            "variance": variance,
            "status": status,
            "error": error,
        })

    return rows


# ============================================================
# Parallel execution
# ============================================================

def choose_num_workers(num_batches: int) -> int:
    cpu_count = os.cpu_count() or 1
    usable = max(1, cpu_count - RESERVE_CPUS)
    workers = min(num_batches, usable)

    if HARD_CAP_WORKERS is not None:
        workers = min(workers, HARD_CAP_WORKERS)

    return max(1, workers)


def write_output(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "model",
        "n_nodes",
        "prob_node",
        "k_roots",
        "seed",
        "X",
        "Y",
        "len_z",
        "hankel_z",
        "variance",
        "status",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_one_file(input_path: Path) -> None:
    tasks = read_hankel_tasks(input_path)

    output_path = output_path_for(input_path)

    if not tasks:
        print(f"No non-empty Hankel separators in {input_path.name}")
        write_output(output_path, [])
        return

    seed_batches = group_tasks_by_seed(tasks)
    workers = choose_num_workers(len(seed_batches))

    print(
        f"Processing {input_path.name}: "
        f"{len(tasks)} rows, {len(seed_batches)} seeds, workers={workers}"
    )

    rows = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_rows in executor.map(compute_seed_batch, seed_batches, chunksize=1):
            rows.extend(batch_rows)

    rows.sort(key=lambda r: (r["seed"], r["X"], r["Y"], r["hankel_z"]))

    write_output(output_path, rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    input_files = collect_input_files()

    if not input_files:
        raise FileNotFoundError(f"No hankel_seps_*.csv files found in {INPUT_DIR}")

    print("Input files:")
    for p in input_files:
        print(f"  - {p}")

    for input_path in input_files:
        process_one_file(input_path)

    print("Done.")


if __name__ == "__main__":
    main()