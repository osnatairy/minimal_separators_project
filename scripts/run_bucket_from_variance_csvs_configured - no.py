#!/usr/bin/env python3
"""
Compute separator buckets from variance-separator CSV files.

This version does NOT get a specific input file name from the user.
Instead, configure these variables near the top of the file:
    INPUT_DIR
    OUTPUT_CSV
    nodes_list
    prob_nodes
    model

For each combination of node/probability/model, the script looks for:
    variance_seps2_{model}_{node}_{prob}.csv

Each input CSV must contain columns equivalent to:
    seed, X, Y, Z/separator, variance

The script groups rows by (source_file, seed, X, Y). Each worker receives one
such group, rebuilds exactly one model graph G for that seed, builds exactly one
H graph for that X,Y pair, computes buckets, and returns one output row per
separator.

Project functions you must have importable:
    make_linear_sem
    utils.split_seeds
    spanning_tree_then_orient
    generate_bn_binary_logistic
    build_H1_from_DAG
    bucket_separators_by_y_connectivity
    cy_component

Adjust the import section below to match your project package names.
"""

from __future__ import annotations

import ast
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Experiment configuration
# Edit these variables instead of passing file names as command-line arguments.
# ---------------------------------------------------------------------------
INPUT_DIR = Path("./variance_per_separators_original")
OUTPUT_CSV = Path("./bucket_results.csv")

nodes_list = [10]  # for example: [10, 20, 30, 40, 50]
prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]
model = "bn"  # "bn" or "sem"

WORKERS = None  # None = use CPU count. Or set an int, for example 8.
K_ROOTS = 1

# ---------------------------------------------------------------------------
# TODO: adjust these imports to your project structure.
# ---------------------------------------------------------------------------
from analysis.bucket_sep import bucket_separators_by_y_connectivity
from analysis.adjustment_hasse import cy_component

# Examples only. Replace module names with the real modules in your project:
from sem.linear_sem import make_linear_sem
from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from graph.h1_builder import build_H1_from_DAG
import utils



Node = str
Separator = Tuple[Node, ...]


@dataclass(frozen=True)
class FileMeta:
    csv_path: Path
    graph_type: str
    n_nodes: int
    edge_probability: float


@dataclass(frozen=True)
class WorkItem:
    meta: FileMeta
    seed: int
    X: Node
    Y: Node
    rows: Tuple[Tuple[Separator, float], ...]
    k_roots: int


_FILENAME_RE = re.compile(
    r"^variance_seps2_(?P<graph_type>bn|sem)_(?P<n_nodes>\d+)_(?P<p>\d+(?:\.\d+)?)\.csv$"
)


def parse_file_meta(csv_path: Path) -> FileMeta:
    """Extract graph type, node count, and edge probability from the file name."""
    m = _FILENAME_RE.match(csv_path.name)
    if not m:
        raise ValueError(
            f"File name does not match expected pattern: {csv_path.name}. "
            "Expected for example: variance_seps2_bn_10_0.15.csv"
        )

    return FileMeta(
        csv_path=csv_path,
        graph_type=m.group("graph_type"),
        n_nodes=int(m.group("n_nodes")),
        edge_probability=float(m.group("p")),
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Support both clean column names and names with leading spaces."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "separator" in df.columns and "Z" not in df.columns:
        df = df.rename(columns={"separator": "Z"})
    if "seperator" in df.columns and "Z" not in df.columns:
        df = df.rename(columns={"seperator": "Z"})

    required = {"seed", "X", "Y", "Z", "variance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)}. Found {list(df.columns)}")

    return df


def parse_separator(value: object) -> Separator:
    """
    Parse a separator from common CSV formats:
      - '["V3", "V7"]'
      - "V3;V7"
      - empty / nan
      - single node, e.g. "V3"
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return tuple()

    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return tuple(sorted(str(v).strip() for v in parsed if str(v).strip()))

    if ";" in text:
        return tuple(sorted(part.strip() for part in text.split(";") if part.strip()))

    return (text,)


def read_work_items(csv_path: Path, k_roots: int) -> List[WorkItem]:
    """Read one CSV and create one WorkItem per (seed, X, Y)."""
    meta = parse_file_meta(csv_path)
    df = normalize_columns(pd.read_csv(csv_path))

    df["seed"] = df["seed"].astype(int)
    df["X"] = df["X"].astype(str).str.strip()
    df["Y"] = df["Y"].astype(str).str.strip()
    df["Z_parsed"] = df["Z"].apply(parse_separator)
    df["variance"] = df["variance"].astype(float)

    items: List[WorkItem] = []
    for (seed, X, Y), group in df.groupby(["seed", "X", "Y"], sort=True):
        rows = tuple((z, float(v)) for z, v in zip(group["Z_parsed"], group["variance"]))
        items.append(
            WorkItem(
                meta=meta,
                seed=int(seed),
                X=str(X),
                Y=str(Y),
                rows=rows,
                k_roots=k_roots,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Graph and H construction adapted to your experiment code.
# ---------------------------------------------------------------------------
def build_model_graph(model: str, node: int, prob_node: float, seed: int, k_roots: int):
    """
    Rebuild the original model graph from the file metadata and row seed.

    This is the direct adaptation of the code you sent:
      - sem: make_linear_sem(...)
      - bn: split seed, build DAG skeleton/orientation, then generate BN params
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
    elif model == "bn":
        seed_graph, seed_params = utils.split_seeds(seed)
        G = spanning_tree_then_orient(
            n=node,
            prob_edge=prob_node,
            k_roots=k_roots,
            node_prefix="V",
            seed=seed_graph,
        )
        modelGraph = generate_bn_binary_logistic(G, seed=seed_params)
    else:
        raise ValueError(f"Unsupported model type: {model}")

    return modelGraph


def build_graph_and_H(item: WorkItem):
    """
    Build exactly one model graph and exactly one H graph for this worker item.

    H depends on X and Y, so X/Y must be taken from the WorkItem, not only from
    the file name. R is all graph nodes, and I is the empty set.
    """
    modelGraph = build_model_graph(
        model=item.meta.graph_type,
        node=item.meta.n_nodes,
        prob_node=item.meta.edge_probability,
        seed=item.seed,
        k_roots=item.k_roots,
    )

    R = list(modelGraph.g.nodes())#get_graph_nodes(modelGraph)
    I = set()

    H = build_H1_from_DAG(modelGraph.g, X=item.X, Y=item.Y, R=R, I=I)
    return modelGraph, H


def process_one_graph(item: WorkItem) -> List[Dict[str, object]]:
    """
    Worker function.

    Important parallelism property:
      One WorkItem = one (file, seed, X, Y) group.
      Therefore this worker builds one modelGraph and one H only.
    """
    _, H = build_graph_and_H(item)

    separators = [z for z, _ in item.rows]
    variance_by_separator: Dict[FrozenSet[Node], float] = {
        frozenset(z): variance for z, variance in item.rows
    }

    buckets = bucket_separators_by_y_connectivity(H, item.Y, separators)
    y_component_by_separator = {
        frozenset(z): frozenset(cy_component(H, item.Y, z))
        for z in separators
    }

    out: List[Dict[str, object]] = []
    for bucket_index, bucket in enumerate(buckets, start=1):
        for sep in bucket:
            sep_key = frozenset(sep)
            sep_tuple = tuple(sorted(sep_key))
            out.append(
                {
                    "source_file": item.meta.csv_path.name,
                    "graph_type": item.meta.graph_type,
                    "n_nodes": item.meta.n_nodes,
                    "edge_probability": item.meta.edge_probability,
                    "seed": item.seed,
                    "X": item.X,
                    "Y": item.Y,
                    "bucket": bucket_index,
                    "separator": json.dumps(list(sep_tuple), ensure_ascii=False),
                    "separator_semicolon": ";".join(sep_tuple),
                    "variance": variance_by_separator[sep_key],
                    "y_component_len": len(y_component_by_separator[sep_key]),
                }
            )

    return out


def probability_to_file_tokens(probability: float) -> List[str]:
    """Return likely filename representations for a probability value."""
    raw = str(probability)
    fixed_2 = f"{probability:.2f}"
    fixed_3 = f"{probability:.3f}"
    trimmed_2 = fixed_2.rstrip("0").rstrip(".")
    trimmed_3 = fixed_3.rstrip("0").rstrip(".")

    # dict.fromkeys preserves order and removes duplicates
    return list(dict.fromkeys([raw, trimmed_2, trimmed_3, fixed_2, fixed_3]))


def find_csv_for_combination(input_dir: Path, model_name: str, node: int, prob_node: float) -> Optional[Path]:
    """Find the CSV file for one (model, node, probability) combination."""
    for prob_text in probability_to_file_tokens(prob_node):
        candidate = input_dir / f"variance_seps2_{model_name}_{node}_{prob_text}.csv"
        if candidate.exists():
            return candidate
    return None


def collect_csvs_from_config(
    input_dir: Path,
    model_name: str,
    nodes: Sequence[int],
    probabilities: Sequence[float],
) -> List[Path]:
    """Collect CSVs according to nodes_list, prob_nodes, and model."""
    csv_paths: List[Path] = []
    missing: List[str] = []

    for node in nodes:
        for prob_node in probabilities:
            csv_path = find_csv_for_combination(input_dir, model_name, node, prob_node)
            if csv_path is None:
                missing.append(f"variance_seps2_{model_name}_{node}_{prob_node}.csv")
            else:
                csv_paths.append(csv_path)

    if missing:
        print("Warning: these configured input files were not found:")
        for name in missing:
            print(f"  - {name}")

    return csv_paths


def run_from_config() -> None:
    """Run the full bucket computation using the constants at the top of the file."""
    if model not in {"bn", "sem"}:
        raise ValueError('model must be either "bn" or "sem"')

    csv_paths = collect_csvs_from_config(
        input_dir=INPUT_DIR,
        model_name=model,
        nodes=nodes_list,
        probabilities=prob_nodes,
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No configured CSV files were found in {INPUT_DIR}. "
            "Check INPUT_DIR, model, nodes_list, and prob_nodes."
        )

    work_items: List[WorkItem] = []
    for csv_path in csv_paths:
        work_items.extend(read_work_items(csv_path, k_roots=K_ROOTS))

    rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS or None) as executor:
        futures = [executor.submit(process_one_graph, item) for item in work_items]
        for future in as_completed(futures):
            rows.extend(future.result())

    result_df = pd.DataFrame(rows)
    sort_cols = ["source_file", "seed", "X", "Y", "bucket", "separator_semicolon"]
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(result_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_from_config()
