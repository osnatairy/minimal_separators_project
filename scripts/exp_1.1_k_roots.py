from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


HEADER_PREFIX = "seed"
EXPECTED_HEADER = [
    "seed",
    "X",
    "Y",
    "outer_sep",
    "outer_sep len",
    "var_y_given_xz_out",
    "outer_sep var",
    "inner_sep",
    "inner_sep len",
    "var_y_given_xz_in",
    "inner_sep var",
    "diff sep var",
    "diff sep var > 0",
    "diff size",
    "X-Drain_in",
    "X-Drain_out",
    "diff-drain",
]


def split_line(line: str) -> List[str]:
    line = line.strip()
    if "\t" in line:
        parts = [x.strip() for x in line.split("\t")]
    else:
        parts = [x.strip() for x in line.split(",")]
    return parts


def parse_filename_metadata(path: Path) -> Dict[str, Any]:
    """
    Extract n, p, and optionally explicit k from file name.
    Examples:
      2026_04_13_seeds_data_main__40_0.07_12_seperators.csv
      2026_04_13_seeds_data_main__30_0.2_beta07_seperators.csv
    """
    name = path.stem

    m = re.search(
        r"main__"
        r"(?P<n>\d+)_"
        r"(?P<p>\d+(?:\.\d+)?)_"
        r"(?P<suffix>[^_]+)"
        r"_seperators$",
        name,
    )
    if not m:
        raise ValueError(f"Could not parse file name: {path.name}")

    n = int(m.group("n"))
    p = float(m.group("p"))
    suffix = m.group("suffix")

    explicit_k = None
    if suffix.isdigit():
        explicit_k = int(suffix)

    return {"n": n, "p": p, "explicit_k": explicit_k, "suffix": suffix}


def default_k_sequence(n: int) -> List[int]:
    """
    According to your experiment description:
      first block  -> k = n
      second block -> k ≈ 30% of n
      third block  -> k = 3
      fourth block -> k = 1
    """
    k30 = max(1, int(round(0.3 * n)))
    return [n, k30, 3, 1]


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_experiment_file(path: Path) -> pd.DataFrame:
    meta = parse_filename_metadata(path)
    n = meta["n"]
    p = meta["p"]
    explicit_k = meta["explicit_k"]

    rows: List[Dict[str, Any]] = []
    block_idx = -1
    k_seq = default_k_sequence(n)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # New block starts whenever header appears again
            if line.startswith(HEADER_PREFIX):
                block_idx += 1
                continue

            parts = split_line(line)

            # Skip malformed rows
            if len(parts) < 11:
                continue

            # Pad missing trailing columns if needed
            if len(parts) < len(EXPECTED_HEADER):
                parts += [None] * (len(EXPECTED_HEADER) - len(parts))
            elif len(parts) > len(EXPECTED_HEADER):
                parts = parts[: len(EXPECTED_HEADER)]

            row = dict(zip(EXPECTED_HEADER, parts))

            if explicit_k is not None:
                k_root = explicit_k
            else:
                if block_idx < 0:
                    # If data begins before first explicit header, ignore it
                    continue
                k_root = k_seq[block_idx] if block_idx < len(k_seq) else None

            row["k_root"] = k_root
            row["n"] = n
            row["p"] = p
            row["source_file"] = path.name
            row["block_idx"] = block_idx + 1
            rows.append(row)

    df = pd.DataFrame(rows)

    numeric_cols = [
        "seed",
        "outer_sep len",
        "var_y_given_xz_out",
        "outer_sep var",
        "inner_sep len",
        "var_y_given_xz_in",
        "inner_sep var",
        "diff sep var",
        "diff size",
        "X-Drain_in",
        "X-Drain_out",
        "diff-drain",
        "k_root",
        "n",
        "p",
    ]
    df = coerce_numeric(df, numeric_cols)

    # Derived columns
    df["variance_gap"] = df["outer_sep var"] - df["inner_sep var"]
    df["component_gap"] = df["outer_sep len"] - df["inner_sep len"]
    df["component_gap_inner_minus_outer"] = df["inner_sep len"] - df["outer_sep len"]
    df["positive_variance_gap"] = df["variance_gap"] > 0

    return df


def load_all_files(input_dir: str | Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*_seperators.csv"))

    if not files:
        raise FileNotFoundError(f"No *_seperators.csv files found in {input_dir}")

    all_dfs = []
    for fp in files:
        try:
            df = parse_experiment_file(fp)
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No files could be parsed.")

    data = pd.concat(all_dfs, ignore_index=True)
    return data


def save_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby(["n", "p", "k_root"], as_index=False)
        .agg(
            rows=("variance_gap", "size"),
            mean_variance_gap=("variance_gap", "mean"),
            median_variance_gap=("variance_gap", "median"),
            mean_component_gap=("component_gap_inner_minus_outer", "mean"),
            median_component_gap=("component_gap_inner_minus_outer", "median"),
            positive_gap_rate=("positive_variance_gap", "mean"),
        )
        .sort_values(["n", "p", "k_root"])
    )

    summary.to_csv(out_dir / "kroot_summary.csv", index=False)
    df.to_csv(out_dir / "all_rows_parsed.csv", index=False)

    print("\n=== Summary by (n, p, k_root) ===")
    print(summary.to_string(index=False))


def plot_mean_variance_gap_by_k(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby(["n", "p", "k_root"], as_index=False)["variance_gap"]
        .mean()
        .rename(columns={"variance_gap": "mean_variance_gap"})
    )

    for n_value in sorted(summary["n"].dropna().unique()):
        sub = summary[summary["n"] == n_value].copy()

        plt.figure(figsize=(8, 5))
        for p_value in sorted(sub["p"].dropna().unique()):
            s = sub[sub["p"] == p_value].sort_values("k_root")
            plt.plot(
                s["k_root"],
                s["mean_variance_gap"],
                marker="o",
                label=f"p={p_value:g}",
            )

        plt.xlabel("k-root")
        plt.ylabel("Mean variance gap: Var(outer) - Var(inner)")
        plt.title(f"Effect of k-root on variance gap (n={int(n_value)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"variance_gap_vs_k_n{int(n_value)}.png", dpi=200)
        plt.close()


def plot_mean_component_gap_by_k(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby(["n", "p", "k_root"], as_index=False)["component_gap_inner_minus_outer"]
        .mean()
        .rename(columns={"component_gap_inner_minus_outer": "mean_component_gap"})
    )

    for n_value in sorted(summary["n"].dropna().unique()):
        sub = summary[summary["n"] == n_value].copy()

        plt.figure(figsize=(8, 5))
        for p_value in sorted(sub["p"].dropna().unique()):
            s = sub[sub["p"] == p_value].sort_values("k_root")
            plt.plot(
                s["k_root"],
                s["mean_component_gap"],
                marker="o",
                label=f"p={p_value:g}",
            )

        plt.xlabel("k-root")
        plt.ylabel("Mean component gap: |CC_Y(inner)| - |CC_Y(outer)|")
        plt.title(f"Effect of k-root on Y-component reduction (n={int(n_value)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"component_gap_vs_k_n{int(n_value)}.png", dpi=200)
        plt.close()


def plot_boxplots_variance_gap(df: pd.DataFrame, out_dir: Path) -> None:
    for (n_value, p_value), sub in df.groupby(["n", "p"]):
        sub = sub.dropna(subset=["k_root", "variance_gap"]).copy()
        if sub.empty:
            continue

        ks = sorted(sub["k_root"].unique())
        data = [sub.loc[sub["k_root"] == k, "variance_gap"].dropna().values for k in ks]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, tick_labels=[str(int(k)) for k in ks])
        plt.xlabel("k-root")
        plt.ylabel("Variance gap: Var(outer) - Var(inner)")
        plt.title(f"Variance gap distribution by k-root (n={int(n_value)}, p={p_value:g})")
        plt.tight_layout()
        plt.savefig(out_dir / f"boxplot_variance_gap_n{int(n_value)}_p{p_value:g}.png", dpi=200)
        plt.close()


def main() -> None:
    # שנהי כאן את הנתיב לתיקייה שבה שמרת את הקבצים
    input_dir = Path("outputs_sem/exp_1_1")
    out_dir = Path("outputs_sem/exp_1_1/kroot_plots")
    out_dir.mkdir(exist_ok=True)

    df = load_all_files(input_dir)

    save_summary_tables(df, out_dir)
    plot_mean_variance_gap_by_k(df, out_dir)
    plot_mean_component_gap_by_k(df, out_dir)
    plot_boxplots_variance_gap(df, out_dir)

    print(f"\nDone. Outputs were saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()