import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("./variance_per_closest_separators_real")
OUTPUT_DIR = Path("./closest_sep_outputs_sem")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "sem"   # "bn", "sem", or "all"


FILENAME_RE = re.compile(
    r"^variance_closest_seps_(?P<graph_type>bn|sem)_(?P<n_nodes>\d+)_(?P<p>\d+(?:\.\d+)?)\.csv$"
)


def parse_file_meta(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    return {
        "graph_type": m.group("graph_type"),
        "n_nodes": int(m.group("n_nodes")),
        "edge_probability": float(m.group("p")),
    }


def collect_files(input_dir: Path, model: str = "all"):
    paths = []

    for path in input_dir.glob("variance_closest_seps_*.csv"):
        meta = parse_file_meta(path)
        if meta is None:
            continue

        if model != "all" and meta["graph_type"] != model:
            continue

        paths.append((path, meta))

    return sorted(paths, key=lambda x: (x[1]["graph_type"], x[1]["n_nodes"], x[1]["edge_probability"]))


def read_all_closest_sep_files(input_dir: Path, model: str = "all"):
    all_dfs = []

    for path, meta in collect_files(input_dir, model):
        df = pd.read_csv(path)

        required = {
            "seed",
            "X",
            "Y",
            "closest_X_variance",
            "closest_Y_variance",
            "diff_variance_X_minus_Y",
            "abs_diff_variance",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        df["source_file"] = path.name
        df["graph_type"] = meta["graph_type"]
        df["n_nodes"] = meta["n_nodes"]
        df["edge_probability"] = meta["edge_probability"]

        df["closest_X_variance"] = pd.to_numeric(df["closest_X_variance"], errors="coerce")
        df["closest_Y_variance"] = pd.to_numeric(df["closest_Y_variance"], errors="coerce")
        df["diff_variance_X_minus_Y"] = pd.to_numeric(df["diff_variance_X_minus_Y"], errors="coerce")
        df["abs_diff_variance"] = pd.to_numeric(df["abs_diff_variance"], errors="coerce")

        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No matching files found in {input_dir}")

    return pd.concat(all_dfs, ignore_index=True)


def build_summary(df: pd.DataFrame):
    clean = df.dropna(subset=["diff_variance_X_minus_Y"]).copy()

    summary = (
        clean.groupby(["graph_type", "n_nodes", "edge_probability"], as_index=False)
        .agg(
            num_rows=("diff_variance_X_minus_Y", "size"),
            num_seeds=("seed", "nunique"),
            mean_diff=("diff_variance_X_minus_Y", "mean"),
            median_diff=("diff_variance_X_minus_Y", "median"),
            mean_abs_diff=("abs_diff_variance", "mean"),
            positive_rate=("diff_variance_X_minus_Y", lambda s: (s > 0).mean()),
            negative_rate=("diff_variance_X_minus_Y", lambda s: (s < 0).mean()),
            zero_rate=("diff_variance_X_minus_Y", lambda s: (s == 0).mean()),
            mean_closest_X_variance=("closest_X_variance", "mean"),
            mean_closest_Y_variance=("closest_Y_variance", "mean"),
        )
        .sort_values(["graph_type", "n_nodes", "edge_probability"])
    )

    return summary


def plot_trend(summary: pd.DataFrame, metric: str, ylabel: str, output_name: str):
    plt.figure(figsize=(9, 5))

    for (graph_type, n_nodes), sub in summary.groupby(["graph_type", "n_nodes"]):
        sub = sub.sort_values("edge_probability")
        plt.plot(
            sub["edge_probability"],
            sub[metric],
            marker="o",
            label=f"{graph_type}, n={n_nodes}",
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Edge probability")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs edge probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wrote plot: {out_path}")


def main():
    df = read_all_closest_sep_files(INPUT_DIR, MODEL)

    summary = build_summary(df)

    raw_path = OUTPUT_DIR / "closest_sep_all_rows.csv"
    summary_path = OUTPUT_DIR / "closest_sep_summary.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote raw rows: {raw_path}")
    print(f"Wrote summary: {summary_path}")

    plot_trend(
        summary,
        metric="mean_diff",
        ylabel="Mean variance difference: closest to X - closest to Y",
        output_name=f"mean_diff_closest_X_minus_Y_{MODEL}.png",
    )

    plot_trend(
        summary,
        metric="mean_abs_diff",
        ylabel="Mean absolute variance difference",
        output_name=f"mean_abs_diff_closest_X_minus_Y_{MODEL}.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()