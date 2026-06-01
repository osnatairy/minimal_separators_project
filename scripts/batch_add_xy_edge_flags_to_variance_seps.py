"""
Batch add edge-direction flags to variance-separators CSV files.

For every combination of graph size n and edge probability p, the script:
  1. Looks for:  variance_seps_sem_{n}_{p}.csv
  2. Rebuilds the SEM graph for each seed appearing in that CSV.
  3. Generates the XY pairs with run_many_xy, exactly like the original script.
  4. Checks whether the graph contains X -> Y and whether it contains Y -> X.
  5. Writes a separate output CSV for that same n,p, containing all original rows
     plus two Boolean columns:
       - edge_X_to_Y
       - edge_Y_to_X

Default usage, no arguments:
    python batch_add_xy_edge_flags_to_variance_seps.py

Expected default folder:
    containment_outputs_sem/csv/

Expected input filenames:
    variance_seps_sem_10_0.07.csv
    variance_seps_sem_10_0.1.csv
    ...
    variance_seps_sem_50_0.3.csv

Default output folder:
    containment_outputs_sem/csv_with_edge_flags/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy


DEFAULT_NODES = [10, 20, 30, 40, 50]
DEFAULT_EDGE_PROBS = [0.07, 0.1, 0.15, 0.2, 0.3]

# Keep the exact textual form used in filenames, e.g. 0.1 and not 0.10.
def prob_to_file_token(p: float) -> str:
    return str(p).rstrip("0").rstrip(".")


def get_graph_object(model_graph):
    """Support both modelGraph.g and modelGraph.G naming conventions."""
    if hasattr(model_graph, "g"):
        return model_graph.g
    if hasattr(model_graph, "G"):
        return model_graph.G
    return model_graph


def build_graph_for_seed(seed: int, n_nodes: int, edge_prob: float, k_roots: Optional[int]):
    """Rebuild the SEM graph exactly as in the supplied separator-generation code."""
    if k_roots is None:
        k_roots = int(n_nodes * 0.3)

    model_graph = make_linear_sem(
        n=n_nodes,
        edge_prob=edge_prob,
        beta_scale=1.0,
        sigma2_low=0.2,
        sigma2_high=1.0,
        node_prefix="V",
        seed=seed,
        k_roots=k_roots,
    )
    return get_graph_object(model_graph)


def build_pair_flags_for_seed(
    seed: int,
    n_nodes: int,
    edge_prob: float,
    k_roots: Optional[int],
) -> Dict[Tuple[int, str, str], Tuple[bool, bool]]:
    """
    Generate all XY pairs for one seed and compute direct-edge flags.

    Returns:
        {(seed, X, Y): (edge_X_to_Y, edge_Y_to_X)}
    """
    graph = build_graph_for_seed(seed, n_nodes, edge_prob, k_roots)
    pairs = run_many_xy(graph, mode="reachable", seed=seed)

    flags: Dict[Tuple[int, str, str], Tuple[bool, bool]] = {}
    for X, Y in pairs:
        flags[(seed, str(X), str(Y))] = (
            bool(graph.has_edge(X, Y)),
            bool(graph.has_edge(Y, X)),
        )
    return flags


def read_seeds_from_variance_file(path: Path) -> List[int]:
    seeds = pd.read_csv(path, usecols=["seed"])["seed"].dropna().astype(int).unique()
    return sorted(int(s) for s in seeds)


def write_pairs_file(
    pair_flags: Dict[Tuple[int, str, str], Tuple[bool, bool]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "X", "Y", "edge_X_to_Y", "edge_Y_to_X"])
        for (seed, X, Y), (edge_xy, edge_yx) in sorted(pair_flags.items()):
            writer.writerow([seed, X, Y, edge_xy, edge_yx])


def add_flags_to_variance_file(
    input_path: Path,
    output_path: Path,
    pair_flags: Dict[Tuple[int, str, str], Tuple[bool, bool]],
    chunksize: int,
) -> Set[Tuple[int, str, str]]:
    """Stream a large variance CSV and append edge flags to every row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    total_rows = 0
    missing_pairs: Set[Tuple[int, str, str]] = set()

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        required_cols = {"seed", "X", "Y"}
        missing_cols = required_cols.difference(chunk.columns)
        if missing_cols:
            raise ValueError(f"{input_path} is missing columns: {sorted(missing_cols)}")

        chunk["seed"] = chunk["seed"].astype(int)
        chunk["X"] = chunk["X"].astype(str)
        chunk["Y"] = chunk["Y"].astype(str)

        edge_xy_values = []
        edge_yx_values = []

        for seed, X, Y in chunk[["seed", "X", "Y"]].itertuples(index=False, name=None):
            key = (int(seed), X, Y)
            flags = pair_flags.get(key)
            if flags is None:
                missing_pairs.add(key)
                edge_xy_values.append(False)
                edge_yx_values.append(False)
            else:
                edge_xy_values.append(flags[0])
                edge_yx_values.append(flags[1])

        chunk["edge_X_to_Y"] = edge_xy_values
        chunk["edge_Y_to_X"] = edge_yx_values

        chunk.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        total_rows += len(chunk)

    print(f"    wrote {total_rows:,} separator rows -> {output_path}")
    return missing_pairs


def process_one_experiment(
    n_nodes: int,
    edge_prob: float,
    input_dir: Path,
    output_dir: Path,
    pairs_output_dir: Optional[Path],
    k_roots: Optional[int],
    chunksize: int,
) -> None:
    p_token = prob_to_file_token(edge_prob)
    input_path = input_dir / f"variance_seps_sem_{n_nodes}_{p_token}.csv"
    output_path = output_dir / f"variance_seps_sem_{n_nodes}_{p_token}_with_edge_flags.csv"

    if not input_path.exists():
        print(f"SKIP n={n_nodes}, p={p_token}: input file not found: {input_path}")
        return

    print(f"PROCESS n={n_nodes}, p={p_token}")
    print(f"    input: {input_path}")

    seeds = read_seeds_from_variance_file(input_path)
    print(f"    seeds: {seeds}")

    all_pair_flags: Dict[Tuple[int, str, str], Tuple[bool, bool]] = {}
    for seed in seeds:
        seed_flags = build_pair_flags_for_seed(seed, n_nodes, edge_prob, k_roots)
        all_pair_flags.update(seed_flags)
        print(f"    seed={seed}: generated {len(seed_flags):,} XY pairs")

    if pairs_output_dir is not None:
        pairs_output_path = pairs_output_dir / f"xy_pairs_sem_{n_nodes}_{p_token}_with_edge_flags.csv"
        write_pairs_file(all_pair_flags, pairs_output_path)
        print(f"    wrote XY pairs -> {pairs_output_path}")

    missing_pairs = add_flags_to_variance_file(
        input_path=input_path,
        output_path=output_path,
        pair_flags=all_pair_flags,
        chunksize=chunksize,
    )

    if missing_pairs:
        missing_path = output_dir / f"variance_seps_sem_{n_nodes}_{p_token}_missing_xy_pairs.csv"
        pd.DataFrame(
            [{"seed": s, "X": x, "Y": y} for s, x, y in sorted(missing_pairs)]
        ).to_csv(missing_path, index=False)
        print(f"    WARNING: {len(missing_pairs):,} XY pairs were in the variance file but not generated now")
        print(f"    wrote missing pairs -> {missing_path}")

    # Small verification summary for this output file.
    summary = pd.read_csv(output_path, usecols=["edge_X_to_Y", "edge_Y_to_X"])
    edge_xy_count = int(summary["edge_X_to_Y"].sum())
    edge_yx_count = int(summary["edge_Y_to_X"].sum())
    either_count = int((summary["edge_X_to_Y"] | summary["edge_Y_to_X"]).sum())
    print(
        f"    summary rows={len(summary):,}, "
        f"X->Y rows={edge_xy_count:,}, "
        f"Y->X rows={edge_yx_count:,}, "
        f"either rows={either_count:,}"
    )


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="variance_per_separators",
        help="Folder containing variance_seps_sem_{n}_{p}.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="variance_per_separators/flag_pairs",
        help="Folder where one output file per n,p will be written",
    )
    parser.add_argument(
        "--pairs-output-dir",
        default="variance_per_separators/xy_pairs_with_edge_flags",
        help="Folder for optional unique XY-pairs files. Use empty string '' to disable.",
    )
    parser.add_argument(
        "--nodes",
        default=",".join(map(str, DEFAULT_NODES)),
        help="Comma-separated graph sizes, default: 10,20,30,40,50",
    )
    parser.add_argument(
        "--edge-probs",
        default=",".join(map(str, DEFAULT_EDGE_PROBS)),
        help="Comma-separated edge probabilities, default: 0.07,0.1,0.15,0.2,0.3",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for reading large CSV files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pairs_output_dir = Path(args.pairs_output_dir) if args.pairs_output_dir else None

    nodes = parse_int_list(args.nodes)
    edge_probs = parse_float_list(args.edge_probs)

    print("Batch edge-flag generation")
    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"pairs_output_dir={pairs_output_dir}")
    print(f"nodes={nodes}")
    print(f"edge_probs={edge_probs}")

    for n_nodes in nodes:
        for edge_prob in edge_probs:
            process_one_experiment(
                n_nodes=n_nodes,
                edge_prob=edge_prob,
                input_dir=input_dir,
                output_dir=output_dir,
                pairs_output_dir=pairs_output_dir,
                k_roots=int(0.3*n_nodes),
                chunksize=args.chunksize,
            )

    print("Done")


if __name__ == "__main__":
    main()
