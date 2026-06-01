from experiments.create_buckets_graphs import make_bucket_boxplots


import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(r"C:\Users\Osnat\Documents\PostDoc\result files - April 2026\output_sem")


def extract_parameters(file_path: Path):
    """
    מחלץ n, p, k מתוך שם קובץ מהצורה:
    ...__30_0.07_9_seperators.csv
    """
    match = re.search(r"__(\d+)_([0-9.]+)_(\d+)_", file_path.name)
    if not match:
        raise ValueError(f"שם קובץ לא תקין: {file_path.name}")

    n = int(match.group(1))
    p = float(match.group(2))
    k = int(match.group(3))

    return n, p, k

def is_valid_experiment(n, k):
    return k == int(0.3 * n)

def extract_edge_probability(file_path: Path) -> float:
    match = re.search(r"__20_([0-9.]+)_6_", file_path.name)
    if not match:
        raise ValueError(f"לא הצלחתי לחלץ את p מתוך שם הקובץ: {file_path.name}")
    return float(match.group(1))


# =========================================================
# חלק א' - קבצי seperators
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Osnat\Documents\PostDoc\result files - April 2026\output_sem")

def analyze_separators(data_dir: Path):
    files = sorted(data_dir.glob("*_seperators.csv"))

    frames = []

    for file_path in files:
        n, p, k = extract_parameters(file_path)

        # סינון לפי התנאי שלך
        if not is_valid_experiment(n, k):
            continue

        df = pd.read_csv(file_path, on_bad_lines="skip")

        if "diff sep var" not in df.columns:
            continue

        df["diff sep var"] = pd.to_numeric(df["diff sep var"], errors="coerce")

        df["n"] = n
        df["p"] = p

        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    summary = (
        all_data.groupby(["n", "p"], as_index=False)["diff sep var"]
        .mean()
        .rename(columns={"diff sep var": "mean_diff_sep_var"})
        .sort_values(["n", "p"])
    )

    return summary

# =========================================================
# חלק ב' - קבצי farthest_closest
# =========================================================
def analyze_farthest_closest(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_farthest_closest.csv"))
    if not files:
        raise FileNotFoundError("לא נמצאו קבצי *_farthest_closest.csv")

    columns = [
        "seed",
        "X",
        "Y",
        "sep_a",
        "sep_a_size",
        "sep_a_var",
        "sep_b",
        "sep_b_size",
        "sep_b_var",
        "diff_var",
    ]

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path, header=None, names=columns)
        #df["p"] = extract_edge_probability(file_path)
        #df["p"] = extract_parameters(file_path)
        n, p, k = extract_parameters(file_path)
        df["n"] = n
        df["p"] = p
        df["k"] = k
        df["source_file"] = file_path.name
        df["diff_var"] = pd.to_numeric(df["diff_var"], errors="coerce")
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    summary = (
        all_data.groupby(["n", "p"], as_index=False)["diff_var"]
        .mean()
        .rename(columns={"diff_var": "mean_diff_var"})
        .sort_values(["n", "p"])
    )
    return summary


# =========================================================
# ציור גרף
# =========================================================
def plot_summary(summary: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(summary[x_col], summary[y_col], marker="o")
    plt.xlabel("Edge probability (p)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_by_n(summary):
    plt.figure()

    for n in sorted(summary["n"].unique()):
        subset = summary[summary["n"] == n]
        plt.plot(subset["p"], subset["mean_diff_sep_var"], marker='o', label=f"n={n}")

    plt.xlabel("Edge probability (p)")
    plt.ylabel("Mean diff sep var")
    plt.title("Mean diff sep var vs p (for different n)")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    '''
    nodes = [20]  # [20,30,40, 50]# different sizes of nodes in  a tree
    prob_nodes = [0.07, 0.1, 0.15, 0.2]  # the probability of an edge
    betas = [0.7]
    types= ["bn"]#,"sem"]

    for type in types:
        for node in nodes:
            for prob_node in prob_nodes:
                for k_roots in [int(node*0.3)]:
                    variance = f"_main_{node}_{prob_node}_{k_roots}"#_beta07"

                    bucket_file = f"regular_outputs_{type}/2026_04_29_bucket_statistics_{variance}.csv"
                    output_path = f"outputs_{type}/"+bucket_file.split("/")[-1].replace(".csv", "")
                    # make_bucket_boxplots(
                    #     bucket_file,
                    #     variance=variance,
                    #     #output_dir=f"bucket_statistics_SEM{variance}",
                    #     mode="global",
                    # )
                    make_bucket_boxplots(
                        bucket_file,
                        variance=variance,
                        output_dir=output_path,
                        mode="per_run",
                    )
                    
    '''
    # =========================================================
    # הרצה
    # =========================================================
    summary_sep = analyze_separators(DATA_DIR)
    summary_fc = analyze_farthest_closest(DATA_DIR)

    print("\n=== SEPERATORS ===")
    print(summary_sep.to_string(index=False))

    print("\n=== FARTHEST_CLOSEST ===")
    print(summary_fc.to_string(index=False))

    #analyze_separators
    plot_by_n(summary_sep)

    plot_summary(
        summary_sep,
        x_col="p",
        y_col="mean_diff_sep_var",
        title="Mean diff sep var by edge probability",
        y_label="Mean diff sep var"
    )

    plot_summary(
        summary_fc,
        x_col="p",
        y_col="mean_diff_var",
        title="Mean variance difference by edge probability (closest/farthest to Y)",
        y_label="Mean difference in variance"
    )

    summary_sep.to_csv("summary_seperators_mean_diff_sep_var.csv", index=False)
    summary_fc.to_csv("summary_farthest_closest_mean_diff_var.csv", index=False)


