"""
Run from a folder that contains files named like:
containment_pairs_variance_seps_bn_20_0.1.csv

The script:
1. extracts n and edge probability p from each filename,
2. removes repeated header rows inside the CSVs,
3. computes experiment-level and run-level summaries,
4. writes CSV files that can be used as evidence for the insights.

Usage:
    python prove_containment_insights.py /path/to/csv_folder
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
DATA_DIR = 'containment_outputs_original_/csv'
folder = DATA_DIR#sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(os.path.join(folder, "containment_pairs_variance_seps_bn_*.csv")))
out_dir = os.path.join(folder, "containment_analysis_outputs")
os.makedirs(out_dir, exist_ok=True)

usecols = [
    "source_file", "seed", "X", "Y",
    "outer_sep_len", "outer_component_len", "outer_variance",
    "inner_sep_len", "inner_component_len", "inner_variance",
    "diff_variance_outer_minus_inner", "abs_diff_variance",
    "diff_variance_positive", "diff_variance_negative",
    "diff_sep_len_outer_minus_inner",
    "diff_component_len_inner_minus_outer",
    "variance_ratio_outer_div_inner", "status",
]

frames = []
for path in files:
    match = re.search(r"_bn_(\d+)_([0-9.]+)\.csv$", os.path.basename(path))
    if not match:
        continue
    n = int(match.group(1))
    p = float(match.group(2))

    # chunksize lets the same script handle larger 50-node files.
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=100_000, low_memory=False):
        chunk = chunk[chunk["source_file"].astype(str) != "source_file"].copy()
        chunk["file_n"] = n
        chunk["file_p"] = p
        frames.append(chunk)

if not frames:
    raise SystemExit("No matching CSV files were found.")

df = pd.concat(frames, ignore_index=True)

numeric_cols = [
    "seed", "outer_sep_len", "outer_component_len", "outer_variance",
    "inner_sep_len", "inner_component_len", "inner_variance",
    "diff_variance_outer_minus_inner", "abs_diff_variance",
    "diff_sep_len_outer_minus_inner", "diff_component_len_inner_minus_outer",
    "variance_ratio_outer_div_inner",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["diff_variance_positive", "diff_variance_negative"]:
    df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

df["run_id"] = (
    df["file_n"].astype(str) + "_" + df["file_p"].astype(str) + "_" +
    df["seed"].astype(str) + "_" + df["X"].astype(str) + "_" + df["Y"].astype(str)
)
df["diff"] = df["diff_variance_outer_minus_inner"]
df["delta_comp"] = df["diff_component_len_inner_minus_outer"]
df["delta_sep"] = df["diff_sep_len_outer_minus_inner"]

# Insight 1-4 evidence: experiment-level summary.
experiment_summary = df.groupby(["file_n", "file_p"]).agg(
    rows=("diff", "size"),
    runs=("run_id", "nunique"),
    positive_pairs=("diff_variance_positive", "sum"),
    negative_pairs=("diff_variance_negative", "sum"),
    mean_diff=("diff", "mean"),
    median_diff=("diff", "median"),
    mean_abs_diff=("abs_diff_variance", "mean"),
    median_abs_diff=("abs_diff_variance", "median"),
    mean_ratio=("variance_ratio_outer_div_inner", "mean"),
    median_ratio=("variance_ratio_outer_div_inner", "median"),
    mean_inner_component=("inner_component_len", "mean"),
    mean_outer_component=("outer_component_len", "mean"),
    mean_delta_component=("delta_comp", "mean"),
    median_delta_component=("delta_comp", "median"),
    same_separator_len_count=("delta_sep", lambda s: int((s == 0).sum())),
).reset_index()
experiment_summary["positive_rate"] = experiment_summary["positive_pairs"] / experiment_summary["rows"]
experiment_summary["negative_rate"] = experiment_summary["negative_pairs"] / experiment_summary["rows"]
experiment_summary["same_separator_len_rate"] = experiment_summary["same_separator_len_count"] / experiment_summary["rows"]
experiment_summary.to_csv(os.path.join(out_dir, "experiment_summary.csv"), index=False)

# Insight 5 evidence: run-level stability.
run_summary = df.groupby(["file_n", "file_p", "seed", "X", "Y"]).agg(
    rows=("diff", "size"),
    positive_rate=("diff_variance_positive", "mean"),
    mean_diff=("diff", "mean"),
    median_diff=("diff", "median"),
    mean_delta_component=("delta_comp", "mean"),
).reset_index()
run_level_summary = run_summary.groupby(["file_n", "file_p"]).agg(
    runs=("positive_rate", "size"),
    all_positive_runs=("positive_rate", lambda s: int((s == 1).sum())),
    mostly_positive_runs=("positive_rate", lambda s: int((s >= 0.9).sum())),
    all_negative_runs=("positive_rate", lambda s: int((s == 0).sum())),
    mean_run_positive_rate=("positive_rate", "mean"),
).reset_index()
run_level_summary["all_positive_run_rate"] = run_level_summary["all_positive_runs"] / run_level_summary["runs"]
run_level_summary["mostly_positive_run_rate"] = run_level_summary["mostly_positive_runs"] / run_level_summary["runs"]
run_level_summary.to_csv(os.path.join(out_dir, "run_level_summary.csv"), index=False)
run_summary.to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)

# Insight 6 evidence: extreme examples.
cols = [
    "file_n", "file_p", "seed", "X", "Y", "outer_component_len", "inner_component_len",
    "outer_sep_len", "inner_sep_len", "outer_variance", "inner_variance", "diff",
    "abs_diff_variance", "variance_ratio_outer_div_inner",
]
examples = {
    "largest_positive": df.nlargest(10, "diff")[cols].to_dict("records"),
    "largest_negative": df.nsmallest(10, "diff")[cols].to_dict("records"),
    "largest_absolute": df.nlargest(10, "abs_diff_variance")[cols].to_dict("records"),
}
with open(os.path.join(out_dir, "extreme_examples.json"), "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

# Optional: correlations, useful for checking whether component/separator-size gaps explain variance gaps.
correlations = []
for (n, p), group in df.groupby(["file_n", "file_p"]):
    def corr(a, b):
        if group[a].nunique(dropna=True) < 2 or group[b].nunique(dropna=True) < 2:
            return np.nan
        return group[[a, b]].corr().iloc[0, 1]
    correlations.append({
        "file_n": n,
        "file_p": p,
        "corr_diff_delta_component": corr("diff", "delta_comp"),
        "corr_diff_delta_separator": corr("diff", "delta_sep"),
    })
pd.DataFrame(correlations).to_csv(os.path.join(out_dir, "correlations.csv"), index=False)

print(f"Rows analyzed: {len(df):,}")
print(f"Experiments: {experiment_summary.shape[0]:,}")
print(f"Runs: {df['run_id'].nunique():,}")
print(f"Outputs written to: {out_dir}")
