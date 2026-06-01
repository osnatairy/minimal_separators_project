#!/usr/bin/env python3
"""
Compute variance for the two closest separators per (seed, X, Y).

Input CSV format, for example:
    closest_seps_sem_10_0.1.csv

Expected columns:
    seed, X, Y, closest_X, closest_Y

where:
    closest_X = separator closer to X
    closest_Y = separator closer to Y

Output CSV format:
    seed, X, Y,
    closest_X, closest_X_len, closest_X_variance,
    closest_Y, closest_Y_len, closest_Y_variance,
    diff_variance_X_minus_Y,
    abs_diff_variance

The script is parallel by seed. Each worker rebuilds the model once per seed,
then computes all closest separator variances for that seed.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import utils

from sem.linear_sem import make_linear_sem
from sem.variance import sigma_from_sem, avar_henckel_single_xy

from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from causal.influence.estimator_bn_sampling import asymptotic_variance_for_Z_from_samples
from causal.policies import static_do_policy
from causal.influence.estimator_bn import asymptotic_variance_for_Z

csv.field_size_limit(10**7)
# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(r"seperators_per_graphs")
OUTPUT_DIR = Path(r"variance_per_closest_separators")

nodes_list = [10,20,30,40,50]
prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]

model = "sem"  # "sem" or "bn"

# If your graphs were generated with k_roots=int(0.3*n), keep this.
# If not, set K_ROOTS_MODE="fixed" and K_ROOTS_FIXED to the value you used.
K_ROOTS_MODE = "auto_30_percent"  # "auto_30_percent" or "fixed"
K_ROOTS_FIXED = 1


# ============================================================
# File names
# ============================================================

def prob_to_filename_part(prob: float) -> str:
    return str(prob)


def input_filename_for(model_name: str, node: int, prob_node: float) -> str:
    return f"closest_seps_{model_name}_{node}_{prob_to_filename_part(prob_node)}.csv"


def output_filename_for(model_name: str, node: int, prob_node: float) -> str:
    return f"variance_closest_seps_{model_name}_{node}_{prob_to_filename_part(prob_node)}.csv"


def output_long_filename_for(model_name: str, node: int, prob_node: float) -> str:
    return f"variance_closest_seps_{model_name}_{node}_{prob_to_filename_part(prob_node)}_long.csv"


def get_k_roots(node: int) -> int:
    if K_ROOTS_MODE == "auto_30_percent":
        return int(node * 0.3)
    if K_ROOTS_MODE == "fixed":
        return int(K_ROOTS_FIXED)
    raise ValueError("K_ROOTS_MODE must be 'auto_30_percent' or 'fixed'")


# ============================================================
# Parsing helpers
# ============================================================

def safe_parse_separator(value: Any) -> tuple[str, ...]:
    """
    Parse one separator from common CSV formats:
      - "['V6', 'V7']"
      - '["V6", "V7"]'
      - 'V6;V7'
      - 'V6'
      - [] / empty / nan
    """
    if value is None:
        return tuple()

    if isinstance(value, float) and math.isnan(value):
        return tuple()

    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(x).strip() for x in value if str(x).strip()))

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return tuple()

    # JSON / Python list
    if text.startswith("[") or text.startswith("(") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None

        if parsed is None:
            return tuple()
        if isinstance(parsed, (list, tuple, set)):
            return tuple(sorted(str(x).strip() for x in parsed if str(x).strip()))
        return tuple()

    # semicolon-separated or comma-separated fallback
    if ";" in text:
        return tuple(sorted(part.strip() for part in text.split(";") if part.strip()))

    if "," in text:
        return tuple(sorted(part.strip() for part in text.split(",") if part.strip()))

    return (text,)


def separator_to_json(sep: tuple[str, ...]) -> str:
    return json.dumps(list(sep), ensure_ascii=False)


def read_closest_tasks_from_file(
    input_path: Path,
    node: int,
    prob_node: float,
    model_name: str,
    k_roots: int,
) -> list[dict]:
    """
    Reads closest CSV and returns one task per XY row.
    Each task contains both closest_X and closest_Y separators.
    """
    tasks: list[dict] = []

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = {c.strip() for c in (reader.fieldnames or [])}

        required_cols = {"seed", "X", "Y", "closest_X", "closest_Y"}
        missing = required_cols - fieldnames
        if missing:
            raise ValueError(f"Missing columns in {input_path.name}: {missing}")

        for row in reader:
            seed = int(float(str(row["seed"]).strip()))
            X = str(row["X"]).strip()
            Y = str(row["Y"]).strip()

            closest_x = safe_parse_separator(row.get("closest_X"))
            closest_y = safe_parse_separator(row.get("closest_Y"))

            tasks.append(
                {
                    "model": model_name,
                    "node": node,
                    "prob_node": prob_node,
                    "k_roots": k_roots,
                    "seed": seed,
                    "X": X,
                    "Y": Y,
                    "closest_X": closest_x,
                    "closest_Y": closest_y,
                }
            )

    return tasks


def group_tasks_by_seed(tasks: list[dict]) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for task in tasks:
        grouped[int(task["seed"])].append(task)
    return list(grouped.values())


# ============================================================
# Graph reconstruction
# ============================================================

def build_model_graph_for_seed(
    model_name: str,
    node: int,
    prob_node: float,
    seed: int,
    k_roots: int,
):
    """Rebuild exactly the graph/model used when closest separators were found."""
    if model_name == "sem":
        return make_linear_sem(
            n=node,
            edge_prob=prob_node,
            beta_scale=1.0,
            sigma2_low=0.2,
            sigma2_high=1.0,
            node_prefix="V",
            seed=seed,
            k_roots=k_roots,
        )

    if model_name == "bn":
        seed_graph, seed_params = utils.split_seeds(seed)
        G = spanning_tree_then_orient(
            n=node,
            prob_edge=prob_node,
            k_roots=k_roots,
            node_prefix="V",
            seed=seed_graph,
        )
        return generate_bn_binary_logistic(G, seed=seed_params)

    raise ValueError("model must be 'sem' or 'bn'")


# ============================================================
# Variance calculation
# ============================================================

def choose_n_samples(
    num_nodes: int,
    z_size: int,
    base_samples: int = 20_000,
    max_samples: int = 200_000,
) -> int:
    if num_nodes <= 10:
        samples = 5_000
    elif num_nodes <= 20:
        samples = 20_000
    elif num_nodes <= 30:
        samples = 50_000
    elif num_nodes <= 40:
        samples = 100_000
    else:
        samples = base_samples

    multiplier = min(2.5, 1 + 0.3 * z_size)
    return min(int(samples * multiplier), max_samples)


def compute_sem_separator_variance(
    X: str,
    Y: str,
    Z: tuple[str, ...],
    var_names,
    Sigma,
    ridge: float = 1e-10,
) -> Optional[float]:
    if len(Z) == 0:
        return None

    return float(
        avar_henckel_single_xy(
            Sigma=Sigma,
            X=X,
            Y=Y,
            Z=list(Z),
            var_names=var_names,
            ridge=ridge,
        )
    )


def compute_bn_separator_variance_from_samples(
    model_graph: Any,
    df_samples,
    X: str,
    Y: str,
    Z: tuple[str, ...],
) -> Optional[float]:
    if len(Z) == 0:
        return None

    L_vars = []
    policy_fn = static_do_policy(a_star=1)

    sigma2 = asymptotic_variance_for_Z(
            bn=model_graph,
            #infer=infer,
            Y=Y,
            A_name=X,
            Z_vars=Z,
            L_vars=L_vars,
            policy_fn=policy_fn,
            value_map=None,
        )

    # sample_sigma = asymptotic_variance_for_Z_from_samples(
    #     df_samples=df_samples,
    #     bn=model_graph,
    #     Y=Y,
    #     A_name=X,
    #     Z_vars=list(Z),
    #     L_vars=L_vars,
    #     policy_fn=policy_fn,
    #     value_map=None,
    # )
    return sigma2


def _compute_seed_batch_closest_variances(seed_tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Computes variances for all closest-separator rows of one seed.
    Returns:
      - wide rows: one row per (seed,X,Y)
      - long rows: one row per separator role: closest_X / closest_Y
    """
    first = seed_tasks[0]

    model_name = first["model"]
    node = int(first["node"])
    prob_node = float(first["prob_node"])
    k_roots = int(first["k_roots"])
    seed = int(first["seed"])

    model_graph = build_model_graph_for_seed(
        model_name=model_name,
        node=node,
        prob_node=prob_node,
        seed=seed,
        k_roots=k_roots,
    )

    if model_name == "sem":
        var_names, Sigma = sigma_from_sem(model_graph)
        df_samples = None
    else:
        var_names, Sigma = None, None
        max_z_size = max(
            max(len(task["closest_X"]), len(task["closest_Y"]))
            for task in seed_tasks
        )
        graph_nodes = list(model_graph.g.nodes())
        n_samples = choose_n_samples(num_nodes=len(graph_nodes), z_size=max_z_size)
        df_samples = model_graph.sample(n_samples=n_samples, seed=seed)

    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    # Cache repeated separators within this seed/XY to avoid duplicate work.
    variance_cache: dict[tuple[str, str, tuple[str, ...]], Optional[float]] = {}

    for task in seed_tasks:
        X = task["X"]
        Y = task["Y"]
        z_x = task["closest_X"]
        z_y = task["closest_Y"]

        def compute_for(role: str, Z: tuple[str, ...]) -> Optional[float]:
            cache_key = (X, Y, Z)
            if cache_key in variance_cache:
                return variance_cache[cache_key]

            if model_name == "sem":
                val = compute_sem_separator_variance(
                    X=X,
                    Y=Y,
                    Z=Z,
                    var_names=var_names,
                    Sigma=Sigma,
                )
            else:
                val = compute_bn_separator_variance_from_samples(
                    model_graph=model_graph,
                    df_samples=df_samples,
                    X=X,
                    Y=Y,
                    Z=Z,
                )

            variance_cache[cache_key] = val
            return val

        var_x = compute_for("closest_X", z_x)
        var_y = compute_for("closest_Y", z_y)

        diff = None
        abs_diff = None
        if var_x is not None and var_y is not None:
            diff = var_x - var_y
            abs_diff = abs(diff)

        wide_rows.append(
            {
                "seed": seed,
                "X": X,
                "Y": Y,
                "closest_X": separator_to_json(z_x),
                "closest_X_len": len(z_x),
                "closest_X_variance": var_x,
                "closest_Y": separator_to_json(z_y),
                "closest_Y_len": len(z_y),
                "closest_Y_variance": var_y,
                "diff_variance_X_minus_Y": diff,
                "abs_diff_variance": abs_diff,
            }
        )

        for role, Z, variance in [
            ("closest_X", z_x, var_x),
            ("closest_Y", z_y, var_y),
        ]:
            long_rows.append(
                {
                    "seed": seed,
                    "X": X,
                    "Y": Y,
                    "role": role,
                    "Z": separator_to_json(Z),
                    "Z_len": len(Z),
                    "variance": variance,
                }
            )

    return wide_rows, long_rows


# ============================================================
# Workers
# ============================================================

def choose_num_workers(
    num_tasks: int,
    concurrent_runs: int = 1,
    reserve_cpus: int = 1,
    hard_cap: int | None = None,
) -> int:
    cpu_count = os.cpu_count() or 1
    usable = max(1, cpu_count - reserve_cpus)
    fair_share = max(1, usable // max(1, concurrent_runs))
    workers = min(num_tasks, fair_share)
    if hard_cap is not None:
        workers = min(workers, hard_cap)
    return max(1, workers)


# ============================================================
# Output
# ============================================================

def write_csv(output_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_one_node_prob_file(
    node: int,
    prob_node: float,
    model_name: str,
    input_dir: Path,
    output_dir: Path,
    concurrent_runs: int = 1,
    reserve_cpus: int = 1,
    hard_cap: int | None = None,
) -> None:
    k_roots = get_k_roots(node)

    input_path = input_dir / input_filename_for(model_name, node, prob_node)
    output_path = output_dir / output_filename_for(model_name, node, prob_node)
    output_long_path = output_dir / output_long_filename_for(model_name, node, prob_node)

    if not input_path.exists():
        print(f"Skipping missing file: {input_path}")
        return

    tasks = read_closest_tasks_from_file(
        input_path=input_path,
        node=node,
        prob_node=prob_node,
        model_name=model_name,
        k_roots=k_roots,
    )

    if not tasks:
        print(f"No closest separator tasks found in: {input_path.name}")
        return

    seed_batches = group_tasks_by_seed(tasks)
    max_workers = choose_num_workers(
        num_tasks=len(seed_batches),
        concurrent_runs=concurrent_runs,
        reserve_cpus=reserve_cpus,
        hard_cap=hard_cap,
    )

    print(
        f"Processing {input_path.name}: "
        f"{len(tasks)} XY rows, {len(seed_batches)} seed batches, workers={max_workers}"
    )

    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_wide, batch_long in executor.map(
            _compute_seed_batch_closest_variances,
            seed_batches,
            chunksize=1,
        ):
            wide_rows.extend(batch_wide)
            long_rows.extend(batch_long)

    wide_rows.sort(key=lambda r: (r["seed"], r["X"], r["Y"]))
    long_rows.sort(key=lambda r: (r["seed"], r["X"], r["Y"], r["role"]))

    write_csv(
        output_path,
        wide_rows,
        fieldnames=[
            "seed",
            "X",
            "Y",
            "closest_X",
            "closest_X_len",
            "closest_X_variance",
            "closest_Y",
            "closest_Y_len",
            "closest_Y_variance",
            "diff_variance_X_minus_Y",
            "abs_diff_variance",
        ],
    )

    write_csv(
        output_long_path,
        long_rows,
        fieldnames=["seed", "X", "Y", "role", "Z", "Z_len", "variance"],
    )

    print(f"Wrote: {output_path}")
    print(f"Wrote: {output_long_path}")


def run_all_closest_variance_calculations(
    nodes: list[int],
    probabilities: list[float],
    model_name: str,
    input_dir: Path,
    output_dir: Path,
    concurrent_runs: int = 1,
    reserve_cpus: int = 1,
    hard_cap: int | None = None,
) -> None:
    for node in nodes:
        for prob_node in probabilities:
            process_one_node_prob_file(
                node=node,
                prob_node=prob_node,
                model_name=model_name,
                input_dir=input_dir,
                output_dir=output_dir,
                concurrent_runs=concurrent_runs,
                reserve_cpus=reserve_cpus,
                hard_cap=hard_cap,
            )


if __name__ == "__main__":
    run_all_closest_variance_calculations(
        nodes=nodes_list,
        probabilities=prob_nodes,
        model_name=model,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        concurrent_runs=1,
        reserve_cpus=1,
        hard_cap=8,
    )
