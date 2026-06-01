#!/usr/bin/env python3
"""
Compute containment-pair CSVs and summary plots from variance-per-separator CSV files.

Input files are expected to look like:
    variance_seps2_bn_10_0.1.csv
    variance_seps2_sem_20_0.3.csv

Each input CSV must contain at least:
    seed, X, Y, Z, variance

For each input file this script:
    1. Rebuilds the original graph for each seed.
    2. Builds H for each (seed, X, Y).
    3. Uses the separators from the variance CSV.
    4. Finds separator pairs with a containment relation in the Hasse structure.
    5. Writes detailed containment-pair rows.
    6. Writes Excel-friendly summary CSVs.
    7. Creates simple summary plots.

Parallelism:
    One WorkItem = one (source_file, seed, X, Y) group.
    Each worker rebuilds one graph and one H, then processes that XY pair.
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

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
nodes_list = [10,20,30,40,50]
prob_nodes = [0.07, 0.1, 0.15, 0.2, 0.3]
model = "bn"  # "bn" or "sem"

# Folder containing files such as variance_seps2_bn_10_0.1.csv
INPUT_DIR = Path("./variance_per_separators_samples-up")

# All outputs will be written under this folder.
OUTPUT_DIR = Path("./containment_outputs_samples-up")
DETAIL_CSV_DIR = OUTPUT_DIR / "csv"
SUMMARY_CSV_DIR = OUTPUT_DIR / "summary"
PLOTS_DIR = OUTPUT_DIR / f"plots_{model}"

# If True: process every file in INPUT_DIR matching variance_seps2_{bn|sem}_*.csv.
USE_ALL_MATCHING_FILES = False

# Used only when USE_ALL_MATCHING_FILES = False.


# Parallel workers. None = os.cpu_count().
WORKERS = None

# Must match the k_roots used when the original graphs were generated.
# Your current parallel scripts usually use int(node * 0.3).
K_ROOTS_MODE = "auto_30_percent"  # "auto_30_percent" or "fixed"
K_ROOTS_FIXED = 1

# Output switches.
WRITE_PER_FILE_DETAIL_CSV = True
WRITE_COMBINED_DETAIL_CSV = True
MAKE_SUMMARY_TABLES = True
MAKE_SUMMARY_PLOTS = True

COMBINED_DETAIL_CSV = OUTPUT_DIR / "containment_pairs_all_files.csv"

# ============================================================
# Project imports
# ============================================================

from analysis.adjustment_hasse import (
    cy_components_for_sets,
    hasse_from_cy_results,
    extract_separator_containment_pairs,
    frozenset_to_str,
)

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


#_FILENAME_RE = re.compile(
#    r"^(?P<graph_type>bn|sem)_(?P<n_nodes>\d+)_(?P<p>\d+(?:\.\d+)?)\.csv$"
#)
_FILENAME_RE = re.compile(
    r"^variance_seps_(?P<graph_type>bn|sem)_(?P<n_nodes>\d+)_(?P<p>\d+(?:\.\d+)?)\.csv$"
)


# ============================================================
# Input parsing
# ============================================================

def parse_file_meta(csv_path: Path) -> FileMeta:
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
    """Parse separator values from JSON, Python-list, semicolon, comma, or single-node cells."""
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
        # Deduplicate separators. If duplicated, keep the first variance.
        var_by_sep: Dict[Separator, float] = {}
        for z, v in zip(group["Z_parsed"], group["variance"]):
            if z and z not in var_by_sep:
                var_by_sep[z] = float(v)

        rows = tuple((z, v) for z, v in var_by_sep.items())
        if len(rows) < 2:
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


def get_graph(model_graph):
    if hasattr(model_graph, "g"):
        return model_graph.g
    if hasattr(model_graph, "G"):
        return model_graph.G
    raise AttributeError("Model graph has neither .g nor .G")


def build_H_for_item(item: WorkItem):
    model_graph = build_model_graph(
        graph_type=item.meta.graph_type,
        node=item.meta.n_nodes,
        prob_node=item.meta.edge_probability,
        seed=item.seed,
        k_roots=item.k_roots,
    )
    G = get_graph(model_graph)
    R = list(G.nodes())
    I = []
    H = build_H1_from_DAG(G, X=item.X, Y=item.Y, R=R, I=I)
    return G, H


# ============================================================
# Containment computation
# ============================================================

def sep_key(value) -> Tuple[str, ...]:
    return tuple(sorted(str(v) for v in value))


def process_one_xy(item: WorkItem) -> List[Dict[str, object]]:
    """Worker function: one source file + one seed + one X,Y pair."""
    G, H = build_H_for_item(item)

    separators = [tuple(z) for z, _ in item.rows if len(z) > 0]
    if len(separators) < 2:
        return []

    variance_by_separator: Dict[Tuple[str, ...], float] = {
        tuple(z): float(v) for z, v in item.rows if len(z) > 0
    }

    try:
        forward, reverse = cy_components_for_sets(H, item.Y, separators)
        hasse = hasse_from_cy_results(forward, reverse)
        containment_pairs = extract_separator_containment_pairs(hasse)
    except Exception as e:
        return [
            {
                "source_file": item.meta.csv_path.name,
                "graph_type": item.meta.graph_type,
                "n_nodes": item.meta.n_nodes,
                "edge_probability": item.meta.edge_probability,
                "k_roots": item.k_roots,
                "seed": item.seed,
                "X": item.X,
                "Y": item.Y,
                "status": "error",
                "error": repr(e),
            }
        ]

    out: List[Dict[str, object]] = []
    for pair_index, hass in enumerate(containment_pairs, start=1):
        outer_sep = sep_key(hass["outer_sep"])
        inner_sep = sep_key(hass["inner_sep"])

        if outer_sep not in variance_by_separator or inner_sep not in variance_by_separator:
            print("outer_sep or inner_sep are not in variance_by_separator")
            continue

        outer_var = variance_by_separator[outer_sep]
        inner_var = variance_by_separator[inner_sep]
        diff_var = outer_var - inner_var

        outer_component = sep_key(hass.get("outer_component", tuple()))
        inner_component = sep_key(hass.get("inner_component", tuple()))

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
                "pair_index": pair_index,
                "num_separators_for_xy": len(separators),
                "num_hasse_edges_for_xy": len(hasse.get("hasse_edges", [])),
                "outer_sep": frozenset_to_str(outer_sep),
                "outer_sep_json": json.dumps(list(outer_sep), ensure_ascii=False),
                "outer_sep_len": len(outer_sep),
                "outer_component": frozenset_to_str(outer_component),
                "outer_component_len": len(outer_component),
                "outer_variance": outer_var,
                "inner_sep": frozenset_to_str(inner_sep),
                "inner_sep_json": json.dumps(list(inner_sep), ensure_ascii=False),
                "inner_sep_len": len(inner_sep),
                "inner_component": frozenset_to_str(inner_component),
                "inner_component_len": len(inner_component),
                "inner_variance": inner_var,
                "diff_variance_outer_minus_inner": diff_var,
                "abs_diff_variance": abs(diff_var),
                "diff_variance_positive": diff_var > 0,
                "diff_variance_negative": diff_var < 0,
                "diff_sep_len_outer_minus_inner": len(outer_sep) - len(inner_sep),
                "diff_component_len_inner_minus_outer": len(inner_component) - len(outer_component),
                "variance_ratio_outer_div_inner": (outer_var / inner_var) if inner_var not in {0, 0.0} else math.nan,
                "status": "ok",
                "error": "",
            }
        )

    return out


# ============================================================
# File collection
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


def collect_csvs() -> List[Path]:
    if USE_ALL_MATCHING_FILES:
        csv_paths = sorted(INPUT_DIR.glob("variance_seps_*.csv"))
        return [p for p in csv_paths if _FILENAME_RE.match(p.name)]

    if model not in {"bn", "sem"}:
        raise ValueError('model must be either "bn" or "sem"')

    return collect_csvs_from_config(INPUT_DIR, model, nodes_list, prob_nodes)


# ============================================================
# Summary tables and plots
# ============================================================

def build_summary_tables(detail_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ok = detail_df[detail_df["status"] == "ok"].copy()
    if ok.empty:
        return {
            "summary_by_file": pd.DataFrame(),
            "summary_by_n_p": pd.DataFrame(),
            "summary_by_graph": pd.DataFrame(),
            "summary_by_xy": pd.DataFrame(),
            "summary_for_excel_long": pd.DataFrame(),
            "pivot_mean_diff_by_p": pd.DataFrame(),
        }

    group_cols = ["graph_type", "n_nodes", "edge_probability", "k_roots"]

    summary_by_file = (
        ok.groupby(["source_file"] + group_cols, as_index=False)
        .agg(
            num_containment_pairs=("diff_variance_outer_minus_inner", "size"),
            num_seeds=("seed", "nunique"),
            num_xy_pairs=("X", "count"),
            mean_diff_variance=("diff_variance_outer_minus_inner", "mean"),
            median_diff_variance=("diff_variance_outer_minus_inner", "median"),
            mean_abs_diff_variance=("abs_diff_variance", "mean"),
            positive_diff_rate=("diff_variance_positive", "mean"),
            negative_diff_rate=("diff_variance_negative", "mean"),
            mean_outer_variance=("outer_variance", "mean"),
            mean_inner_variance=("inner_variance", "mean"),
            mean_outer_sep_len=("outer_sep_len", "mean"),
            mean_inner_sep_len=("inner_sep_len", "mean"),
            mean_outer_component_len=("outer_component_len", "mean"),
            mean_inner_component_len=("inner_component_len", "mean"),
        )
        .sort_values(group_cols)
    )

    summary_by_n_p = (
        ok.groupby(group_cols, as_index=False)
        .agg(
            num_containment_pairs=("diff_variance_outer_minus_inner", "size"),
            num_files=("source_file", "nunique"),
            num_seeds=("seed", "nunique"),
            mean_diff_variance=("diff_variance_outer_minus_inner", "mean"),
            median_diff_variance=("diff_variance_outer_minus_inner", "median"),
            mean_abs_diff_variance=("abs_diff_variance", "mean"),
            positive_diff_rate=("diff_variance_positive", "mean"),
            negative_diff_rate=("diff_variance_negative", "mean"),
            mean_outer_variance=("outer_variance", "mean"),
            mean_inner_variance=("inner_variance", "mean"),
            mean_outer_sep_len=("outer_sep_len", "mean"),
            mean_inner_sep_len=("inner_sep_len", "mean"),
            mean_outer_component_len=("outer_component_len", "mean"),
            mean_inner_component_len=("inner_component_len", "mean"),
            mean_component_reduction=("diff_component_len_inner_minus_outer", "mean"),
        )
        .sort_values(group_cols)
    )

    summary_by_graph = (
        ok.groupby(group_cols + ["seed"], as_index=False)
        .agg(
            num_containment_pairs=("diff_variance_outer_minus_inner", "size"),
            num_xy_pairs=("X", "count"),
            mean_diff_variance=("diff_variance_outer_minus_inner", "mean"),
            median_diff_variance=("diff_variance_outer_minus_inner", "median"),
            mean_abs_diff_variance=("abs_diff_variance", "mean"),
            positive_diff_rate=("diff_variance_positive", "mean"),
            negative_diff_rate=("diff_variance_negative", "mean"),
        )
        .sort_values(group_cols + ["seed"])
    )

    summary_by_xy = (
        ok.groupby(group_cols + ["seed", "X", "Y"], as_index=False)
        .agg(
            num_containment_pairs=("diff_variance_outer_minus_inner", "size"),
            num_separators_for_xy=("num_separators_for_xy", "max"),
            num_hasse_edges_for_xy=("num_hasse_edges_for_xy", "max"),
            mean_diff_variance=("diff_variance_outer_minus_inner", "mean"),
            median_diff_variance=("diff_variance_outer_minus_inner", "median"),
            mean_abs_diff_variance=("abs_diff_variance", "mean"),
            positive_diff_rate=("diff_variance_positive", "mean"),
            negative_diff_rate=("diff_variance_negative", "mean"),
        )
        .sort_values(group_cols + ["seed", "X", "Y"])
    )

    # Long table, convenient for Excel pivot charts.
    summary_for_excel_long = summary_by_n_p.melt(
        id_vars=group_cols,
        value_vars=[
            "num_containment_pairs",
            "mean_diff_variance",
            "median_diff_variance",
            "mean_abs_diff_variance",
            "positive_diff_rate",
            "negative_diff_rate",
            "mean_outer_variance",
            "mean_inner_variance",
            "mean_outer_sep_len",
            "mean_inner_sep_len",
            "mean_outer_component_len",
            "mean_inner_component_len",
            "mean_component_reduction",
        ],
        var_name="metric",
        value_name="value",
    )

    pivot_mean_diff_by_p = summary_by_n_p.pivot_table(
        index=["graph_type", "n_nodes", "k_roots"],
        columns="edge_probability",
        values="mean_diff_variance",
        aggfunc="mean",
    ).reset_index()

    return {
        "summary_by_file": summary_by_file,
        "summary_by_n_p": summary_by_n_p,
        "summary_by_graph": summary_by_graph,
        "summary_by_xy": summary_by_xy,
        "summary_for_excel_long": summary_for_excel_long,
        "pivot_mean_diff_by_p": pivot_mean_diff_by_p,
    }


def save_summary_tables(detail_df: pd.DataFrame) -> Dict[str, Path]:
    SUMMARY_CSV_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_summary_tables(detail_df)
    paths: Dict[str, Path] = {}

    for name, df in tables.items():
        path = SUMMARY_CSV_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
        print(f"Wrote {len(df)} rows to {path}")

    return paths


def plot_metric_vs_probability(summary_by_n_p: pd.DataFrame, metric: str, ylabel: str, filename: str) -> Optional[Path]:
    if summary_by_n_p.empty or metric not in summary_by_n_p.columns:
        return None

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename

    plt.figure(figsize=(9, 5))
    for (graph_type, n_nodes), sub in summary_by_n_p.groupby(["graph_type", "n_nodes"]):
        sub = sub.sort_values("edge_probability")
        label = f"{graph_type}, n={int(n_nodes)}"
        plt.plot(sub["edge_probability"], sub[metric], marker="o", label=label)

    plt.xlabel("Edge probability")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs edge probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Wrote plot to {path}")
    return path


def make_summary_plots(detail_df: pd.DataFrame) -> List[Path]:
    tables = build_summary_tables(detail_df)
    summary = tables["summary_by_n_p"]
    created: List[Path] = []

    plot_specs = [
        ("mean_diff_variance", "Mean variance difference: outer - inner", "mean_diff_variance_vs_probability.png"),
        ("median_diff_variance", "Median variance difference: outer - inner", "median_diff_variance_vs_probability.png"),
        ("mean_abs_diff_variance", "Mean absolute variance difference", "mean_abs_diff_variance_vs_probability.png"),
        ("positive_diff_rate", "Rate where outer variance > inner variance", "positive_diff_rate_vs_probability.png"),
        ("num_containment_pairs", "Number of containment pairs", "num_containment_pairs_vs_probability.png"),
        ("mean_component_reduction", "Mean component gap: inner component len - outer component len", "mean_component_gap_vs_probability.png"),
    ]

    for metric, ylabel, filename in plot_specs:
        p = plot_metric_vs_probability(summary, metric, ylabel, filename)
        if p is not None:
            created.append(p)

    return created


# ============================================================
# Main run
# ============================================================

def process_input_file(csv_path: Path) -> pd.DataFrame:
    print(f"\nProcessing {csv_path.name}")
    items = read_work_items(csv_path)
    print(f"  Work items: {len(items)}")

    rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS or None) as executor:
        futures = [executor.submit(process_one_xy, item) for item in items]
        for future in as_completed(futures):
            rows.extend(future.result())

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "source_file", "graph_type", "n_nodes", "edge_probability", "k_roots",
                "seed", "X", "Y", "pair_index", "outer_sep", "inner_sep",
                "outer_variance", "inner_variance", "diff_variance_outer_minus_inner",
                "status", "error",
            ]
        )

    if "status" in df.columns:
        ok_sort = [c for c in ["source_file", "seed", "X", "Y", "pair_index"] if c in df.columns]
        if ok_sort:
            df = df.sort_values(ok_sort).reset_index(drop=True)

    if WRITE_PER_FILE_DETAIL_CSV:
        DETAIL_CSV_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DETAIL_CSV_DIR / f"containment_pairs_{csv_path.stem}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Wrote {len(df)} rows to {out_path}")

    return df


def run_from_config() -> None:
    csv_paths = collect_csvs()
    if not csv_paths:
        raise FileNotFoundError(
            f"No input CSV files were found in {INPUT_DIR}. "
            "Check INPUT_DIR or set USE_ALL_MATCHING_FILES=False with nodes_list/prob_nodes/model."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        all_dfs.append(process_input_file(csv_path))

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if WRITE_COMBINED_DETAIL_CSV:
        COMBINED_DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(COMBINED_DETAIL_CSV, index=False)
        print(f"\nWrote combined detail CSV with {len(combined)} rows to {COMBINED_DETAIL_CSV}")

    if MAKE_SUMMARY_TABLES:
        save_summary_tables(combined)

    if MAKE_SUMMARY_PLOTS:
        make_summary_plots(combined)

    print("\nDone.")


if __name__ == "__main__":
    run_from_config()
