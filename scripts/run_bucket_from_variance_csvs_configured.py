#!/usr/bin/env python3
"""
Compute bucket CSVs and bucket plots from variance-per-separator CSV files.

Input files are expected to look like:
    variance_seps_bn_10_0.1.csv
    variance_seps_sem_20_0.3.csv

Each input CSV must contain at least:
    seed, X, Y, Z, variance

For each input file the script writes:
    1. A bucket statistics CSV:
       OUTPUT_DIR/csv/bucket_statistics_<input_stem>.csv

    2. Bucket plots using create_buckets_graphs.make_bucket_boxplots:
       OUTPUT_DIR/plots/<input_stem>/...

The bucket computation stays parallel:
    one WorkItem = one (source_file, seed, X, Y) group.
"""

from __future__ import annotations

import ast
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# Folder containing files such as variance_seps_bn_10_0.1.csv
INPUT_DIR = Path("./variance_per_separators")

# All outputs will be written under this folder.
OUTPUT_DIR = Path("./bucket_outputs_sem")
BUCKET_CSV_DIR = OUTPUT_DIR / "csv"
BUCKET_PLOTS_DIR = OUTPUT_DIR / "plots"
BUCKET_PLOTS_DIR = OUTPUT_DIR / "plots"

# If True: process every file in INPUT_DIR matching variance_seps_{bn|sem}_*.csv.
# This is usually what you want for the folder shown in your screenshot.
USE_ALL_MATCHING_FILES = True

# Used only when USE_ALL_MATCHING_FILES = False.
nodes_list = [10,20,30,40]#,50]
prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]
model = "sem"  # "bn" or "sem"

# Parallel workers for bucket computation. None = os.cpu_count().
WORKERS = None

# Must match the k_roots used when the original graphs were generated.
# Your current parallel scripts use int(node * 0.3), so AUTO is the safe default.
K_ROOTS_MODE = "auto_30_percent"  # "auto_30_percent" or "fixed"
K_ROOTS_FIXED = 1

# Plot generation mode passed to make_bucket_boxplots:
#   "per_run" = one combined variance/Y-component plot per (seed,X,Y), when enough data exists
#   "global"  = one combined plot for all runs in the file
#   "both"    = create both
PLOT_MODE = "per_run"
MAKE_PLOTS = True

# Optional combined CSV across all input files.
WRITE_COMBINED_CSV = True
COMBINED_OUTPUT_CSV = OUTPUT_DIR / "bucket_results_all_files.csv"


# ============================================================
# Project imports
# ============================================================

from analysis.bucket_sep import bucket_separators_by_y_connectivity
from analysis.adjustment_hasse import cy_component

from sem.linear_sem import make_linear_sem
from bn.cpt import generate_bn_binary_logistic
from graph.generators import spanning_tree_then_orient
from graph.h1_builder import build_H1_from_DAG

from experiments.create_buckets_graphs import make_bucket_boxplots

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
    r"^variance_seps_(?P<graph_type>bn|sem)_(?P<n_nodes>\d+)_(?P<p>\d+(?:\.\d+)?)\.csv$"
)


def parse_file_meta(csv_path: Path) -> FileMeta:
    """Extract graph type, node count, and edge probability from the file name."""
    m = _FILENAME_RE.match(csv_path.name)
    if not m:
        raise ValueError(
            f"File name does not match expected pattern: {csv_path.name}. "
            "Expected for example: variance_seps_bn_10_0.15.csv"
        )

    return FileMeta(
        csv_path=csv_path,
        graph_type=m.group("graph_type"),
        n_nodes=int(m.group("n_nodes")),
        edge_probability=float(m.group("p")),
    )


def k_roots_for_node(n_nodes: int) -> int:
    if K_ROOTS_MODE == "auto_30_percent":
        return max(1, int(n_nodes * 0.3))
    if K_ROOTS_MODE == "fixed":
        return int(K_ROOTS_FIXED)
    raise ValueError("K_ROOTS_MODE must be 'auto_30_percent' or 'fixed'.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Support both clean column names and names with leading spaces."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    aliases = {
        "separator": "Z",
        "seperator": "Z",
        "sep": "Z",
        "z": "Z",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

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
      - "V3"
      - empty / nan
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

    if "," in text and not text.startswith("V"):
        return tuple(sorted(part.strip().strip("'\"") for part in text.split(",") if part.strip()))

    return (text,)


def read_work_items(csv_path: Path) -> List[WorkItem]:
    """Read one CSV and create one WorkItem per (seed, X, Y)."""
    meta = parse_file_meta(csv_path)
    k_roots = k_roots_for_node(meta.n_nodes)

    df = normalize_columns(pd.read_csv(csv_path))
    df["seed"] = df["seed"].astype(int)
    df["X"] = df["X"].astype(str).str.strip()
    df["Y"] = df["Y"].astype(str).str.strip()
    df["Z_parsed"] = df["Z"].apply(parse_separator)
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")
    df = df.dropna(subset=["variance"])

    items: List[WorkItem] = []
    for (seed, X, Y), group in df.groupby(["seed", "X", "Y"], sort=True):
        rows = tuple((z, float(v)) for z, v in zip(group["Z_parsed"], group["variance"]))
        if not rows:
            continue
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


# ============================================================
# Graph and H construction
# ============================================================

def build_model_graph(graph_type: str, node: int, prob_node: float, seed: int, k_roots: int):
    """Rebuild the exact model graph for the given file metadata and seed."""
    if graph_type == "sem":
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

    if graph_type == "bn":
        seed_graph, seed_params = utils.split_seeds(seed)
        G = spanning_tree_then_orient(
            n=node,
            prob_edge=prob_node,
            k_roots=k_roots,
            node_prefix="V",
            seed=seed_graph,
        )
        return generate_bn_binary_logistic(G, seed=seed_params)

    raise ValueError(f"Unsupported graph_type: {graph_type}")


def get_dag(model_graph):
    """Return the NetworkX DAG from either SEM or BN project objects."""
    if hasattr(model_graph, "g"):
        return model_graph.g
    if hasattr(model_graph, "G"):
        return model_graph.G
    raise AttributeError("Model object has neither .g nor .G graph attribute.")


def build_H_for_item(item: WorkItem):
    model_graph = build_model_graph(
        graph_type=item.meta.graph_type,
        node=item.meta.n_nodes,
        prob_node=item.meta.edge_probability,
        seed=item.seed,
        k_roots=item.k_roots,
    )
    G = get_dag(model_graph)
    R = list(G.nodes())
    I = []
    return build_H1_from_DAG(G, X=item.X, Y=item.Y, R=R, I=I)


def process_one_xy_group(item: WorkItem) -> List[Dict[str, object]]:
    """
    Worker function.

    One WorkItem = one (source_file, seed, X, Y) group.
    The worker rebuilds one graph, builds one H, buckets the separators,
    and returns one output row per separator.
    """
    H = build_H_for_item(item)

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

            # If bucket function returns a separator that was not in the variance CSV,
            # skip it rather than crashing.
            if sep_key not in variance_by_separator:
                continue

            out.append(
                {
                    "source_file": item.meta.csv_path.name,
                    "graph_type": item.meta.graph_type,
                    "n_nodes": item.meta.n_nodes,
                    "edge_probability": item.meta.edge_probability,
                    "k_roots": item.k_roots,
                    "seed": item.seed,
                    "X": item.X,
                    "Y": item.Y,
                    "bucket": bucket_index,

                    # create_buckets_graphs.py expects separator to be semicolon-delimited.
                    "separator": ";".join(sep_tuple),
                    "separator_json": json.dumps(list(sep_tuple), ensure_ascii=False),
                    "variance": variance_by_separator[sep_key],
                    "y_component_len": len(y_component_by_separator[sep_key]),
                }
            )

    return out


# ============================================================
# Input collection
# ============================================================

def probability_to_file_tokens(probability: float) -> List[str]:
    raw = str(probability)
    fixed_2 = f"{probability:.2f}"
    fixed_3 = f"{probability:.3f}"
    trimmed_2 = fixed_2.rstrip("0").rstrip(".")
    trimmed_3 = fixed_3.rstrip("0").rstrip(".")
    return list(dict.fromkeys([raw, trimmed_2, trimmed_3, fixed_2, fixed_3]))


def find_csv_for_combination(input_dir: Path, model_name: str, node: int, prob_node: float) -> Optional[Path]:
    for prob_text in probability_to_file_tokens(prob_node):
        candidate = input_dir / f"variance_seps_{model_name}_{node}_{prob_text}.csv"
        if candidate.exists():
            return candidate
    return None


def collect_csvs_from_config(
    input_dir: Path,
    model_name: str,
    nodes: Sequence[int],
    probabilities: Sequence[float],
) -> List[Path]:
    csv_paths: List[Path] = []
    missing: List[str] = []

    for node in nodes:
        for prob_node in probabilities:
            csv_path = find_csv_for_combination(input_dir, model_name, node, prob_node)
            if csv_path is None:
                missing.append(f"variance_seps_{model_name}_{node}_{prob_node}.csv")
            else:
                csv_paths.append(csv_path)

    if missing:
        print("Warning: these configured input files were not found:")
        for name in missing:
            print(f"  - {name}")

    return csv_paths


def collect_input_csvs() -> List[Path]:
    if USE_ALL_MATCHING_FILES:
        paths = sorted(INPUT_DIR.glob("variance_seps_*.csv"))
        valid: List[Path] = []
        skipped: List[Path] = []
        for p in paths:
            if _FILENAME_RE.match(p.name):
                valid.append(p)
            else:
                skipped.append(p)
        if skipped:
            print("Skipping files with unsupported names:")
            for p in skipped:
                print(f"  - {p.name}")
        return valid

    if model not in {"bn", "sem"}:
        raise ValueError('model must be either "bn" or "sem"')

    return collect_csvs_from_config(
        input_dir=INPUT_DIR,
        model_name=model,
        nodes=nodes_list,
        probabilities=prob_nodes,
    )


# ============================================================
# Output and plots
# ============================================================

def write_bucket_csvs_by_source(result_df: pd.DataFrame) -> List[Path]:
    BUCKET_CSV_DIR.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    for source_file, sub in result_df.groupby("source_file", sort=True):
        stem = Path(str(source_file)).stem
        out_path = BUCKET_CSV_DIR / f"bucket_statistics_{stem}.csv"

        # Keep the columns create_buckets_graphs needs, plus metadata columns.
        cols = [
            "source_file",
            "graph_type",
            "n_nodes",
            "edge_probability",
            "k_roots",
            "seed",
            "X",
            "Y",
            "bucket",
            "separator",
            "separator_json",
            "variance",
            "y_component_len",
        ]
        sub = sub[cols].sort_values(["seed", "X", "Y", "bucket", "separator"])
        sub.to_csv(out_path, index=False)
        created.append(out_path)
        print(f"Wrote bucket CSV: {out_path} ({len(sub)} rows)")

    return created


def create_plots_for_bucket_csvs(bucket_csvs: List[Path]) -> None:
    if not MAKE_PLOTS:
        return

    BUCKET_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    modes = ["global", "per_run"] if PLOT_MODE == "both" else [PLOT_MODE]
    for csv_path in bucket_csvs:
        # Remove bucket_statistics_ prefix for cleaner file names.
        stem = csv_path.stem.replace("bucket_statistics_", "")
        out_dir = BUCKET_PLOTS_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for mode_name in modes:
            if mode_name not in {"global", "per_run"}:
                raise ValueError("PLOT_MODE must be 'global', 'per_run', or 'both'.")
            try:
                created = make_bucket_boxplots(
                    input_file=csv_path,
                    variance=stem,
                    output_dir=out_dir,
                    mode=mode_name,
                )
                print(f"Created {len(created)} {mode_name} plot(s) in {out_dir}")
            except ValueError as e:
                print(f"No plots created for {csv_path.name} in mode={mode_name}: {e}")


def run_from_config() -> None:
    csv_paths = collect_input_csvs()
    if not csv_paths:
        raise FileNotFoundError(
            f"No variance_seps_*.csv files were found in {INPUT_DIR}. "
            "Check INPUT_DIR or set USE_ALL_MATCHING_FILES=False and configure nodes/prob/model."
        )

    print("Input files:")
    for p in csv_paths:
        print(f"  - {p}")

    work_items: List[WorkItem] = []
    for csv_path in csv_paths:
        items = read_work_items(csv_path)
        work_items.extend(items)
        print(f"Loaded {len(items)} XY groups from {csv_path.name}")

    if not work_items:
        raise RuntimeError("No work items were created from the input CSV files.")

    rows: List[Dict[str, object]] = []
    print(f"Computing buckets for {len(work_items)} XY groups with workers={WORKERS or 'auto'}")

    with ProcessPoolExecutor(max_workers=WORKERS or None) as executor:
        futures = [executor.submit(process_one_xy_group, item) for item in work_items]
        for future in as_completed(futures):
            rows.extend(future.result())

    if not rows:
        raise RuntimeError("Bucket computation produced no rows.")

    result_df = pd.DataFrame(rows)
    sort_cols = ["source_file", "seed", "X", "Y", "bucket", "separator"]
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if WRITE_COMBINED_CSV:
        result_df.to_csv(COMBINED_OUTPUT_CSV, index=False)
        print(f"Wrote combined CSV: {COMBINED_OUTPUT_CSV} ({len(result_df)} rows)")

    bucket_csvs = write_bucket_csvs_by_source(result_df)
    create_plots_for_bucket_csvs(bucket_csvs)

    print("Done.")


if __name__ == "__main__":
    run_from_config()
