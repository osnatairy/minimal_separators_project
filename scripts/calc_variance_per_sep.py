import os
import csv
import sys
import json
import ast
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable

#from Tools.scripts.freeze_modules import FROZEN

import utils

from bn.pgmpy_adapter import bn_to_pgmpy_model
from sem.linear_sem import make_linear_sem
from sem.variance import sigma_from_sem,avar_henckel_single_xy
from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from causal.influence.estimator_bn import asymptotic_variance_for_Z,asymptotic_variance_for_Z_auto,asymptotic_variance_over_Z_sets
#influance function calculation
from causal.influence.estimator_bn_sampling import asymptotic_variance_for_Z_from_samples

from causal.policies import static_do_policy


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(r"seperators_per_graphs")
OUTPUT_DIR = Path(r"variance_per_separators_samples")

N_SEEDS = 10


nodes_list = [10,20]#,20]  # לדוגמה: [10, 20, 30, 40, 50]
prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]

model = "bn"# or "bn""bn"  #

#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(10**7)

# ============================================================
# Worker globals
# ============================================================

#_WORKER_GRAPH_CACHE = {}
#_WORKER_MODEL = None


# def _init_variance_worker(model_name: str):
#     """
#     Runs once per worker process.
#     Each worker keeps a local cache of generated graphs by:
#         (model, node, prob_node, seed, k_roots)
#     """
#     #global _WORKER_GRAPH_CACHE, _WORKER_MODEL
#
#     #_WORKER_GRAPH_CACHE = {}
#     #_WORKER_MODEL = model_name


# ============================================================
# Parsing helpers
# ============================================================

def safe_parse_list(value):
    """
    Parses JSON/Python-like list safely.

    Example:
        '[["V1", "V2"], ["V3"]]'
    becomes:
        [["V1", "V2"], ["V3"]]
    """
    if value is None:
        return []

    value = str(value).strip()

    if value == "":
        return []

    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def prob_to_filename_part(prob: float) -> str:
    """
    Keeps file naming stable.

    Example:
        0.3 -> "0.3"
        0.07 -> "0.07"
    """
    return str(prob)


def input_filename_for(model: str,node: int, prob_node: float) -> str:
    """
    Example:
        seps_sem_40_0.3.csv
    """
    return f"seps_sem_{node}_{prob_to_filename_part(prob_node)}.csv"


def output_filename_for(model: str, node: int, prob_node: float) -> str:
    """
    Example:
        variance_seps_sem_40_0.3.csv
    """
    return f"variance_seps_{model}_{node}_{prob_to_filename_part(prob_node)}.csv"

def group_tasks_by_seed(tasks):
    """
    Groups all separator tasks by seed.
    Each group will be processed by one worker,
    so the graph for that seed is built only once.
    """
    grouped = defaultdict(list)

    for task in tasks:
        grouped[task["seed"]].append(task)

    return list(grouped.values())
# ============================================================
# Graph reconstruction
# ============================================================

def build_model_graph_for_seed(
    model: str,
    node: int,
    prob_node: float,
    seed: int,
    k_roots: int,
):
    """
    Rebuilds exactly the graph/model that was used when separators were found.
    """

    if model == "sem":
        modelGraph = make_linear_sem(
            n=node,
            edge_prob=prob_node,
            beta_scale=1.0,
            sigma2_low=0.2,
            sigma2_high=1.0,
            node_prefix="V",
            seed=seed,
            k_roots=k_roots,
        )
        return modelGraph

    else:
        seed_graph, seed_params = utils.split_seeds(seed)

        G = spanning_tree_then_orient(
            n=node,
            prob_edge=prob_node,
            k_roots=k_roots,
            node_prefix="V",
            seed=seed_graph,
        )

        modelGraph = generate_bn_binary_logistic(
            G,
            seed=seed_params,
        )

        return modelGraph


# def get_cached_model_graph(
#     model: str,
#     node: int,
#     prob_node: float,
#     seed: int,
#     k_roots: int,
# ):
#     """
#     Each worker process lazily builds a graph only once per seed.
#     """
#     global _WORKER_GRAPH_CACHE
#
#     key = (model, node, prob_node, seed, k_roots)
#
#     if key not in _WORKER_GRAPH_CACHE:
#         _WORKER_GRAPH_CACHE[key] = build_model_graph_for_seed(
#             model=model,
#             node=node,
#             prob_node=prob_node,
#             seed=seed,
#             k_roots=k_roots,
#         )
#
#     return _WORKER_GRAPH_CACHE[key]


# ============================================================
# Variance calculation placeholder
# ============================================================
def choose_n_samples(
    num_nodes: int,
    z_size: int,
    base_samples: int = 20_000,
    max_samples: int = 200_000,
) -> int:
    """
    Chooses sample size according to graph size and separator size.
    Smaller graphs and smaller Z get fewer samples.
    """

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

    # larger separators usually need more samples
    #samples *= max(1, 1+0.3*z_size)
    multiplier = min(2.5, 1 + 0.3 * z_size)
    samples = int(samples * multiplier)
    return min(samples, max_samples)

def compute_bn_separator_variance_from_samples(
    modelGraph: Any,
    df_samples,
    X: str,
    Y: str,
    Z: list[str],
):
    """
    Computes BN variance for one Z using pre-generated samples.
    The samples are generated once per seed/graph outside this function.
    """

    L_vars = []
    policy_fn = static_do_policy(a_star=1)

    sample_sigma = asymptotic_variance_for_Z_from_samples(
        df_samples=df_samples,
        bn=modelGraph,
        Y=Y,
        A_name=X,
        Z_vars=list(Z),
        L_vars=L_vars,
        policy_fn=policy_fn,
        value_map=None,
    )

    # sigma2 = asymptotic_variance_for_Z(
    #         bn=modelGraph,
    #         #infer=infer,
    #         Y=Y,
    #         A_name=X,
    #         Z_vars=Z,
    #         L_vars=L_vars,
    #         policy_fn=policy_fn,
    #         value_map=None,
    #     )

    #sigma_exact = asymptotic_variance_for_Z_auto(
    #        modelGraph, Y, X, Z, L_vars, policy_fn,
    #        method="exact",
    #    )

    return sample_sigma


# def compute_bn_separator_variance(
#     modelGraph: Any,
#     X: str,
#     Y: str,
#     Z: list[str],
#     infer,
#     n_samples: int,
# ):
#     """
#     Fill in the actual variance calculation here.
#
#     Parameters
#     ----------
#     modelGraph:
#         The SEM / BN model reconstructed for the relevant seed.
#
#     X, Y:
#         The XY pair.
#
#     Z:
#         One separator set.
#
#     Returns
#     -------
#     float
#         The variance value.
#     """
#
#     # ========================================================
#     # כאן למלא את חישוב השונות
#     # למשל:
#     # return compute_avar_for_single_Z(modelGraph, X, Y, Z)
#     # ========================================================
#
#
#
#      #CALCULATE REGULAR VARIANCE
#
#     L_vars = []
#     policy_fn = static_do_policy(a_star=1)
#
#     Z_key = tuple(sorted(Z))
#
#     '''
#     sigma_exact = asymptotic_variance_for_Z_auto(
#         modelGraph, Y, X, Z, L_vars, policy_fn,
#         method="exact",
#     )
#     '''
#     sigma_sampling = asymptotic_variance_for_Z_auto(
#         modelGraph, Y, X, Z, L_vars, policy_fn,
#         method="sampling",
#         n_samples=n_samples,
#         seed=1,
#     )
#
#
#
#     '''
#     sigma2 = asymptotic_variance_for_Z(
#         bn=modelGraph,
#         infer=infer,
#         Y=Y,
#         A_name=X,
#         Z_vars=Z_key,
#         L_vars=L_vars,
#         policy_fn=policy_fn,
#         value_map=None,
#     )
#     '''
#     return sigma_sampling

def compute_sem_separator_variance(
        modelGraph: Any,
        X: str,
        Y: str,
        Z: list[str],
        var_names, Sigma,
        ridge: float = 1e-10,
):
    """
    Fill in the actual variance calculation here.

    Parameters
    ----------
    modelGraph:
        The SEM / BN model reconstructed for the relevant seed.

    X, Y:
        The XY pair.

    Z:
        One separator set.

    Returns
    -------
    float
        The variance value.
    """

    # ========================================================
    # כאן למלא את חישוב השונות
    # למשל:
    # return compute_avar_for_single_Z(modelGraph, X, Y, Z)
    # ========================================================

    # CALCULATE REGULAR VARIANCE

    Z_list = list(Z)
    aVar = avar_henckel_single_xy(
        Sigma=Sigma,
        X=X,
        Y=Y,
        Z=Z_list,
        var_names=var_names,
        ridge=ridge,
    )

    return aVar


# ============================================================
# Read separators file and flatten tasks
# ============================================================

def read_separator_tasks_from_file(
    input_path: Path,
    node: int,
    prob_node: float,
    model: str,
    k_roots: int,
):
    """
    Reads one separators CSV file and creates flat tasks.

    Expected columns:
        seed, X, Y, total_time, num_values, values_json

    Also supports:
        seed, X, Y, num_values, values_json
    """

    tasks = []

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"seed", "X", "Y", "Z"}
        missing = required_cols - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Missing columns in {input_path.name}: {missing}")

        for row in reader:
            seed = int(row["seed"])
            X = row["X"]
            Y = row["Y"]

            Z = safe_parse_list(row["Z"])  # עכשיו זה מפריד אחד בלבד

            if Z is None or len(Z) == 0:
                continue

            Z_key = tuple(sorted(Z))

            tasks.append({
                "model": model,
                "node": node,
                "prob_node": prob_node,
                "k_roots": k_roots,
                "seed": seed,
                "X": X,
                "Y": Y,
                "Z": Z_key,
            })

    return tasks


# ============================================================
# Parallel worker
# ============================================================

# def _compute_one_separator_variance(task: dict):
#     """
#     Worker function.
#     Computes variance for one row:
#         seed, X, Y, Z
#     """
#
#     model = task["model"]
#     node = task["node"]
#     prob_node = task["prob_node"]
#     k_roots = task["k_roots"]
#     seed = task["seed"]
#     X = task["X"]
#     Y = task["Y"]
#     Z_key = task["Z"]
#
#     modelGraph = get_cached_model_graph(
#         model=model,
#         node=node,
#         prob_node=prob_node,
#         seed=seed,
#         k_roots=k_roots,
#     )
#
#     if modelGraph.name == "sem":
#         var_names, Sigma = sigma_from_sem(modelGraph)
#
#         variance = compute_sem_separator_variance(
#             modelGraph=modelGraph,
#             X=X,
#             Y=Y,
#             Z=list(Z_key),
#             var_names = var_names,
#             Sigma=Sigma,
#         )
#     else:
#         model, infer = bn_to_pgmpy_model(modelGraph)
#
#         variance = compute_bn_separator_variance(
#             modelGraph=modelGraph,
#             X=X,
#             Y=Y,
#             Z=list(Z_key),
#             infer=infer,
#         )
#
#     return {
#         "seed": seed,
#         "X": X,
#         "Y": Y,
#         "Z": json.dumps(list(Z_key)),
#         "variance": variance,
#     }
#

def _compute_seed_batch_variances(seed_tasks: list[dict]):
    """
    Computes variances for all separator tasks of one seed.

    The graph is built once for the seed, then reused for all rows.
    """
    first = seed_tasks[0]

    model = first["model"]
    node = first["node"]
    prob_node = first["prob_node"]
    k_roots = first["k_roots"]
    seed = first["seed"]

    modelGraph = build_model_graph_for_seed(
        model=model,
        node=node,
        prob_node=prob_node,
        seed=seed,
        k_roots=k_roots,
    )

    if modelGraph.name == "sem":
        var_names, Sigma = sigma_from_sem(modelGraph)
        df_samples = None
    else:
        var_names = None
        Sigma = None

        max_z_size = max(len(task["Z"]) for task in seed_tasks)

        n_samples = choose_n_samples(
            num_nodes=len(modelGraph.g.nodes()),
            z_size=max_z_size,
        )

        df_samples = modelGraph.sample(
            n_samples=n_samples,
            seed=seed,
        )

    rows = []

    for task in seed_tasks:
        X = task["X"]
        Y = task["Y"]
        Z_key = task["Z"]

        if modelGraph.name == "sem":
            #var_names, Sigma = sigma_from_sem(modelGraph)

            variance = compute_sem_separator_variance(
                modelGraph=modelGraph,
                X=X,
                Y=Y,
                Z=list(Z_key),
                var_names=var_names,
                Sigma=Sigma,
            )
        else:
            variance = compute_bn_separator_variance_from_samples(
                modelGraph=modelGraph,
                df_samples=df_samples,
                X=X,
                Y=Y,
                Z=list(Z_key),
            )

        rows.append({
            "seed": seed,
            "X": X,
            "Y": Y,
            "Z": json.dumps(list(Z_key)),
            "variance": variance,
        })

    return rows

# ============================================================
# Choose workers
# ============================================================

def choose_num_workers(
    num_tasks: int,
    concurrent_runs: int = 1,
    reserve_cpus: int = 1,
    hard_cap: int | None = None,
) -> int:
    """
    Chooses workers for one script run.
    Useful when several scenarios run in parallel.
    """

    cpu_count = os.cpu_count() or 1
    usable = max(1, cpu_count - reserve_cpus)
    fair_share = max(1, usable // max(1, concurrent_runs))

    workers = min(num_tasks, fair_share)

    if hard_cap is not None:
        workers = min(workers, hard_cap)

    return max(1, workers)


# ============================================================
# Write output
# ============================================================

def write_variance_results(output_path: Path, rows: list[dict]):
    """
    Writes:
        seed, X, Y, Z, variance
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed", "X", "Y", "Z", "variance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# Main processing for one file
# ============================================================

def process_one_node_prob_file(
    node: int,
    prob_node: float,
    model: str,
    input_dir: Path,
    output_dir: Path,
    concurrent_runs: int = 1,
    reserve_cpus: int = 1,
    hard_cap: int | None = None,
    chunksize: int = 20,
):
    """
    Reads one separators file and computes variance for all separators in parallel.
    """

    k_roots = int(node * 0.3)

    input_path = input_dir / input_filename_for(model,node, prob_node)
    output_path = output_dir / output_filename_for(model, node, prob_node)

    if not input_path.exists():
        print(f"Skipping missing file: {input_path}")
        return

    tasks = read_separator_tasks_from_file(
        input_path=input_path,
        node=node,
        prob_node=prob_node,
        model=model,
        k_roots=k_roots,
    )

    if not tasks:
        print(f"No separator tasks found in: {input_path.name}")
        write_variance_results(output_path, [])
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
        f"{len(tasks)} separator tasks, workers={max_workers}"
    )

    rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for rows_batch in executor.map(
                _compute_seed_batch_variances,
                seed_batches,
                chunksize=1,
        ):
            rows.extend(rows_batch)

    write_variance_results(output_path, rows)

    print(f"Wrote: {output_path}")


# ============================================================
# Run all scenarios
# ============================================================

def run_all_variance_calculations(
    nodes_list: list[int],
    prob_nodes: list[float],
    model: str,
    input_dir: Path,
    output_dir: Path,
    concurrent_runs: int = 1,
    reserve_cpus: int = 1,
    hard_cap: int | None = None,
    chunksize: int = 20,
):
    """
    Loops over all graph sizes and edge probabilities.
    For each file, computes variance for all separators in parallel.
    """

    for node in nodes_list:
        for prob_node in prob_nodes:
            process_one_node_prob_file(
                node=node,
                prob_node=prob_node,
                model=model,
                input_dir=input_dir,
                output_dir=output_dir,
                concurrent_runs=concurrent_runs,
                reserve_cpus=reserve_cpus,
                hard_cap=hard_cap,
                chunksize=chunksize,
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_all_variance_calculations(
        nodes_list=nodes_list,
        prob_nodes=prob_nodes,
        model=model,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,

        # אם את מריצה כמה סקריפטים במקביל על השרת:
        concurrent_runs=1,

        # להשאיר ליבה/כמה ליבות למערכת:
        reserve_cpus=1,

        # במחשב שלך אפשר לשים 8.
        # בשרת אפשר None, או למשל 32/64.
        hard_cap=8,

        chunksize=20,
    )