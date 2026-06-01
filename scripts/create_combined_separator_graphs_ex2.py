#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

MODEL = "sem"  # "sem", "bn", or "all"

NODES_LIST = [30]#, 20, 30, 40, 50]
PROB_NODES = [0.1]#, 0.15, 0.2, 0.3]

HANKEL_DIR = Path("variance_hankel_seps_exact")
MINIMAL_DIR = Path("variance_per_separators")
IMPORTANCE_DIR = Path("important_seps_up")

OUTPUT_DIR = Path("combined_separator_graphs")
COMBINED_CSV_DIR = OUTPUT_DIR / "csv"
PLOTS_DIR = OUTPUT_DIR / "plots"

SHOW = False

PER_XY_CSV_DIR = OUTPUT_DIR / "per_xy_csv"
WRITE_PER_XY_CSVS = True

# ============================================================
# Helpers
# ============================================================

def prob_tokens(p: float) -> list[str]:
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


def find_existing_file(folder: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_files_for_combo(model: str, n: int, p: float):
    p_tokens = prob_tokens(p)

    minimal_patterns = []
    importance_patterns = []
    hankel_patterns = []

    for pt in p_tokens:
        minimal_patterns += [
            f"variance_seps_{model}_{n}_{pt}.csv",
            f"variance_seps2_{model}_{n}_{pt}.csv",
        ]

        importance_patterns += [
            f"important_seps_sem_{n}_{pt}.csv",
            f"importance_seps_{model}_{n}_{pt}.csv",
            f"seps_{model}_{n}_{pt}.csv",
            f"hankel_seps_{model}_{n}_{pt}.csv",  # למקרה שכך שמרת importance
        ]

        hankel_patterns += [
            f"variance_hankel_seps_{model}_{n}_{pt}.csv",
            f"variance_henkel_seps_{model}_{n}_{pt}.csv",
        ]

    minimal_file = find_existing_file(MINIMAL_DIR, minimal_patterns)
    importance_file = find_existing_file(IMPORTANCE_DIR, importance_patterns)
    hankel_file = find_existing_file(HANKEL_DIR, hankel_patterns)

    return minimal_file, importance_file, hankel_file


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_separator(value) -> tuple[str, ...]:
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

    if isinstance(parsed, (list, tuple, set)):
        return tuple(sorted(str(v).strip() for v in parsed if str(v).strip()))

    if isinstance(parsed, str):
        if ";" in parsed:
            return tuple(sorted(x.strip() for x in parsed.split(";") if x.strip()))
        if "," in parsed and not parsed.startswith("V"):
            return tuple(sorted(x.strip().strip("'\"") for x in parsed.split(",") if x.strip()))
        return (parsed.strip(),) if parsed.strip() else tuple()

    return tuple()


def sep_to_str(sep: tuple[str, ...]) -> str:
    return ";".join(sep)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find separator column. Tried {candidates}. Found {list(df.columns)}")


# ============================================================
# Readers
# ============================================================

def read_minimal_file(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))

    required = {"seed", "X", "Y", "variance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    z_col = find_col(df, ["Z", "separator", "sep", "minimal_z"])

    df["seed"] = df["seed"].astype(int)
    df["X"] = df["X"].astype(str).str.strip()
    df["Y"] = df["Y"].astype(str).str.strip()
    df["sep_key"] = df[z_col].apply(parse_separator)
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")

    df = df.dropna(subset=["variance"])
    df = df[df["sep_key"].apply(len) > 0]

    return pd.DataFrame({
        "seed": df["seed"],
        "X": df["X"],
        "Y": df["Y"],
        "separator": df["sep_key"].apply(sep_to_str),
        "sep_key": df["sep_key"],
        "len_sep": df["sep_key"].apply(len),
        "variance": df["variance"],
        "type": "minimal",
    })


def read_importance_file(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))

    required = {"seed", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    z_col = find_col(
        df,
        [
            "importance_z",
            "important_z",
            "importance_sep",
            "important_sep",
            "Z",
            "separator",
            "sep",
            "hankel_z",   # למקרה שהקובץ נכתב באותו פורמט כמו הקוד שלך
            "henkel_z",
        ],
    )

    df["seed"] = df["seed"].astype(int)
    df["X"] = df["X"].astype(str).str.strip()
    df["Y"] = df["Y"].astype(str).str.strip()
    df["sep_key"] = df[z_col].apply(parse_separator)

    df = df[df["sep_key"].apply(len) > 0]

    return pd.DataFrame({
        "seed": df["seed"],
        "X": df["X"],
        "Y": df["Y"],
        "separator": df["sep_key"].apply(sep_to_str),
        "sep_key": df["sep_key"],
        "len_sep": df["sep_key"].apply(len),
        "type": "importance",
    })


def read_hankel_file(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))

    required = {"seed", "X", "Y", "variance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    z_col = find_col(df, ["hankel_z", "henkel_z", "Z", "separator", "sep"])

    df["seed"] = df["seed"].astype(int)
    df["X"] = df["X"].astype(str).str.strip()
    df["Y"] = df["Y"].astype(str).str.strip()
    df["sep_key"] = df[z_col].apply(parse_separator)
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")

    df = df.dropna(subset=["variance"])
    df = df[df["sep_key"].apply(len) > 0]

    return pd.DataFrame({
        "seed": df["seed"],
        "X": df["X"],
        "Y": df["Y"],
        "separator": df["sep_key"].apply(sep_to_str),
        "sep_key": df["sep_key"],
        "len_sep": df["sep_key"].apply(len),
        "variance": df["variance"],
        "type": "hankel",
    })


# ============================================================
# Merge
# ============================================================

def attach_importance_variance(
    importance_df: pd.DataFrame,
    minimal_df: pd.DataFrame,
    model: str,
    n: int,
    p: float,
) -> pd.DataFrame:
    lookup = minimal_df[["seed", "X", "Y", "sep_key", "variance"]].copy()

    merged = importance_df.merge(
        lookup,
        on=["seed", "X", "Y", "sep_key"],
        how="left",
    )

    missing = merged[merged["variance"].isna()]

    if not missing.empty:
        print()
        print(f"WARNING: importance separators not found in minimal file for {model}, n={n}, p={p}:")
        for _, row in missing.iterrows():
            print(
                f"  seed={row['seed']}, X={row['X']}, Y={row['Y']}, "
                f"separator={row['separator']}"
            )

    merged = merged.dropna(subset=["variance"])

    return merged[
        ["seed", "X", "Y", "separator", "sep_key", "len_sep", "variance", "type"]
    ]

'''
def build_combined_for_combo(
    model: str,
    n: int,
    p: float,
    minimal_file: Path,
    importance_file: Path,
    hankel_file: Path,
) -> pd.DataFrame:
    minimal_df = read_minimal_file(minimal_file)
    importance_raw = read_importance_file(importance_file)
    importance_df = attach_importance_variance(importance_raw, minimal_df, model, n, p)
    hankel_df = read_hankel_file(hankel_file)

    combined = pd.concat(
        [minimal_df, importance_df, hankel_df],
        ignore_index=True,
    )

    combined["model"] = model
    combined["n_nodes"] = n
    combined["edge_probability"] = p

    # אם מפריד מופיע גם כ-minimal וגם כ-importance → נשאיר רק importance
    key_cols = ["model", "n_nodes", "edge_probability", "seed", "X", "Y", "sep_key"]

    importance_keys = set(
        combined.loc[combined["type"] == "importance", key_cols]
        .apply(tuple, axis=1)
    )

    combined = combined[
        ~(
                (combined["type"] == "minimal")
                & combined[key_cols].apply(tuple, axis=1).isin(importance_keys)
        )
    ]

    combined = combined.drop_duplicates(
        subset=["model", "n_nodes", "edge_probability", "seed", "X", "Y", "sep_key", "type"]
    )

    combined = combined.sort_values(
        ["model", "n_nodes", "edge_probability", "seed", "X", "Y", "type", "len_sep", "separator"]
    ).reset_index(drop=True)

    return combined
'''

def build_combined_for_combo(
    model: str,
    n: int,
    p: float,
    minimal_file: Path,
    importance_file: Path,
    hankel_file: Path,
) -> pd.DataFrame:
    minimal_df = read_minimal_file(minimal_file)
    importance_raw = read_importance_file(importance_file)
    importance_df = attach_importance_variance(importance_raw, minimal_df, model, n, p)
    hankel_df = read_hankel_file(hankel_file)

    # Keep Hankel rows only for (seed, X, Y) pairs that have an X-Y separator
    # in the minimal/importance files.
    xy_with_xy_separator = set(
        pd.concat(
            [
                minimal_df[["seed", "X", "Y"]],
                importance_df[["seed", "X", "Y"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .apply(tuple, axis=1)
    )

    hankel_df = hankel_df[
        hankel_df[["seed", "X", "Y"]]
        .apply(tuple, axis=1)
        .isin(xy_with_xy_separator)
    ]

    combined = pd.concat(
        [minimal_df, importance_df, hankel_df],
        ignore_index=True,
    )

    combined["model"] = model
    combined["n_nodes"] = n
    combined["edge_probability"] = p

    # If a separator appears both as minimal and importance, keep only importance.
    key_cols = ["model", "n_nodes", "edge_probability", "seed", "X", "Y", "sep_key"]

    importance_keys = set(
        combined.loc[combined["type"] == "importance", key_cols]
        .apply(tuple, axis=1)
    )

    combined = combined[
        ~(
            (combined["type"] == "minimal")
            & combined[key_cols].apply(tuple, axis=1).isin(importance_keys)
        )
    ]

    combined = combined.drop_duplicates(
        subset=[
            "model",
            "n_nodes",
            "edge_probability",
            "seed",
            "X",
            "Y",
            "sep_key",
            "type",
        ]
    )

    combined = combined.sort_values(
        [
            "model",
            "n_nodes",
            "edge_probability",
            "seed",
            "X",
            "Y",
            "type",
            "len_sep",
            "separator",
        ]
    ).reset_index(drop=True)

    return combined

# ============================================================
# Plotting
# ============================================================

def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")

'''
def plot_by_xy(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for (seed, x_node, y_node), group in df.groupby(["seed", "X", "Y"], sort=True):
        plt.figure(figsize=(8, 6))

        minimal_group = group[group["type"] == "minimal"]
        importance_group = group[group["type"] == "importance"]
        hankel_group = group[group["type"] == "hankel"]

        if not hankel_group.empty:
            plt.scatter(
                hankel_group["len_sep"],
                hankel_group["variance"],
                marker="*",
                s=260,
                color="yellow",
                edgecolors="black",
                label="hankel",
                zorder=1,
            )

        if not minimal_group.empty:
            plt.scatter(
                minimal_group["len_sep"],
                minimal_group["variance"],
                color="red",
                alpha=0.65,
                label="minimal",
                zorder=3,
            )

        if not importance_group.empty:
            plt.scatter(
                importance_group["len_sep"],
                importance_group["variance"],
                color="blue",
                alpha=0.9,
                label="importance",
                zorder=2,
            )



        plt.xlabel("Separator size")
        plt.ylabel("Variance")
        plt.title(f"seed={seed}, X={x_node}, Y={y_node}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"seed_{seed}_X_{safe_name(x_node)}_Y_{safe_name(y_node)}.png"
        plt.savefig(output_dir / filename, bbox_inches="tight", dpi=200)

        if SHOW:
            plt.show()
        else:
            plt.close()
'''

def plot_by_xy(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    min_separators_for_plot = 10

    for (seed, x_node, y_node), group in df.groupby(["seed", "X", "Y"], sort=True):

        # Do not create plot if this X,Y has fewer than 10 separators.
        # The data is still kept in the combined CSV and per-XY CSV.
        if len(group) < min_separators_for_plot:
            continue

        plt.figure(figsize=(8, 6))

        minimal_group = group[group["type"] == "minimal"]
        importance_group = group[group["type"] == "importance"]
        hankel_group = group[group["type"] == "hankel"]

        if not hankel_group.empty:
            plt.scatter(
                hankel_group["len_sep"],
                hankel_group["variance"],
                marker="*",
                s=260,
                color="yellow",
                edgecolors="black",
                alpha=0.75,
                label="hankel",
                zorder=1,
            )

        if not minimal_group.empty:
            plt.scatter(
                minimal_group["len_sep"],
                minimal_group["variance"],
                color="red",
                alpha=0.65,
                label="minimal",
                zorder=3,
                s = 10
            )

        if not importance_group.empty:
            plt.scatter(
                importance_group["len_sep"],
                importance_group["variance"],
                color="blue",
                alpha=0.9,
                label="importance",
                zorder=4,
                s = 10
            )

        plt.xlabel("Separator size")
        plt.ylabel("Variance")
        plt.title(f"seed={seed}, X={x_node}, Y={y_node}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"seed_{seed}_X_{safe_name(x_node)}_Y_{safe_name(y_node)}.png"
        plt.savefig(output_dir / filename, bbox_inches="tight", dpi=200)

        if SHOW:
            plt.show()
        else:
            plt.close()

def write_per_xy_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Writes one CSV per (seed, X, Y), in the old long format:
        seed, X, Y, separator, len_sep, variance, type
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = ["seed", "X", "Y", "separator", "len_sep", "variance", "type"]

    type_order = {
        "minimal": 1,
        "importance": 2,
        "hankel": 3,
    }

    df = df.copy()
    df["type_order"] = df["type"].map(type_order).fillna(99)

    for (seed, x_node, y_node), group in df.groupby(["seed", "X", "Y"], sort=True):
        group = group.sort_values(
            ["type_order", "len_sep", "separator"]
        )

        filename = (
            f"separators_seed_{seed}"
            f"_X_{safe_name(x_node)}"
            f"_Y_{safe_name(y_node)}.csv"
        )

        group[cols].to_csv(output_dir / filename, index=False)



# ============================================================
# Main
# ============================================================

def models_to_run() -> list[str]:
    if MODEL == "all":
        return ["bn", "sem"]
    if MODEL not in {"bn", "sem"}:
        raise ValueError('MODEL must be "bn", "sem", or "all"')
    return [MODEL]


def process_combo(model: str, n: int, p: float) -> pd.DataFrame | None:
    minimal_file, importance_file, hankel_file = find_files_for_combo(model, n, p)

    if minimal_file is None or importance_file is None or hankel_file is None:
        print(f"Skipping {model}, n={n}, p={p}: missing files")
        if minimal_file is None:
            print("  missing minimal file")
        if importance_file is None:
            print("  missing importance file")
        if hankel_file is None:
            print("  missing hankel file")
        return None

    print(f"Processing {model}, n={n}, p={p}")
    print(f"  minimal:    {minimal_file}")
    print(f"  importance: {importance_file}")
    print(f"  hankel:     {hankel_file}")

    combined = build_combined_for_combo(
        model=model,
        n=n,
        p=p,
        minimal_file=minimal_file,
        importance_file=importance_file,
        hankel_file=hankel_file,
    )

    COMBINED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = COMBINED_CSV_DIR / f"combined_separators_{model}_{n}_{p}.csv"

    combined_for_csv = combined.drop(columns=["sep_key"])
    combined_for_csv.to_csv(csv_path, index=False)

    print(f"  wrote CSV: {csv_path} ({len(combined_for_csv)} rows)")

    plot_dir = PLOTS_DIR / f"{model}_{n}_{p}"
    plot_by_xy(combined, plot_dir)

    print(f"  wrote plots to: {plot_dir}")

    if WRITE_PER_XY_CSVS:
        per_xy_dir = PER_XY_CSV_DIR / f"{model}_{n}_{p}"
        write_per_xy_csvs(combined, per_xy_dir)
        print(f"  wrote per-XY CSVs to: {per_xy_dir}")

    return combined


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model in models_to_run():
        for n in NODES_LIST:
            for p in PROB_NODES:
                result = process_combo(model, n, p)
                if result is not None:
                    all_results.append(result)

    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        all_csv = COMBINED_CSV_DIR / "combined_separators_all.csv"
        all_df.drop(columns=["sep_key"]).to_csv(all_csv, index=False)
        print(f"Wrote global combined CSV: {all_csv}")

    print("Done.")


if __name__ == "__main__":
    main()