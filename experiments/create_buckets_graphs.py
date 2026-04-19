import csv
import re
from pathlib import Path
from typing import List, Dict,Tuple,Iterable,Any
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FOLDER = "outputs/23_3_26_bucket_statistics"

def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def _parse_number_list(cell: str) -> List[float]:
    """
    Parse a 'list-like' CSV cell into list[float].
    Works for:
      - "0.12"
      - "0.12 0.13 0.10"
      - "0.12;0.13;0.10"
      - "[0.12, 0.13, 0.10]"  (commas are rare in CSV cells, but we handle it anyway)
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []

    # remove common wrappers
    s = s.strip("[](){}")

    # split on semicolon OR whitespace OR comma
    parts = re.split(r"[;\s,]+", s)
    vals = []
    for p in parts:
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return vals


def parse_bucket_stats_csv(file_path: str | Path,
                           separator_delim: str = ";") -> List[Dict[str, Any]]:
    """
    Parse bucket statistics CSV with header:
      seed,X,Y,bucket,separator,variance,y_component_len

    מחזירה רשימה של dict-ים, כל שורה עם:
      seed (int), X (str), Y (str),
      bucket (int),
      separator_str (str),
      separator_nodes (list[str]),
      variance (float),
      y_component_len (int)
    """
    file_path = Path(file_path)

    rows: List[Dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            if r is None:
                continue

            # ניקוי רווחים מהערכים (לא מהמפתחות)
            cleaned = {}
            for k, v in r.items():
                if k is None:
                    # לפעמים DictReader דוחף עמודות עודפות תחת key=None
                    continue
                k = k.strip() if isinstance(k, str) else k
                if isinstance(v, str):
                    v = v.strip()
                cleaned[k] = v

            # דילוג על שורות ריקות לגמרי (,,,,,,)
            if all((v is None) or (str(v).strip() == "") for v in cleaned.values()):
                continue

            seed_str = str(cleaned.get("seed", "")).strip()
            X = str(cleaned.get("X", "")).strip()
            Y = str(cleaned.get("Y", "")).strip()
            bucket_str = str(cleaned.get("bucket", "")).strip()
            sep_str = str(cleaned.get("separator", "")).strip()
            var_str = str(cleaned.get("variance", "")).strip()
            ylen_str = str(cleaned.get("y_component_len", "")).strip()

            # אם אחד מהשדות הקריטיים ריק → לדלג
            if not seed_str or not bucket_str or not var_str or not ylen_str:
                continue

            # המרה מספרית עם הגנה – אם לא מספרי, מדלגים על השורה
            try:
                seed = int(float(seed_str))
                bucket = int(float(bucket_str))
                variance = float(var_str)
                y_component_len = int(float(ylen_str))
            except ValueError:
                # זו בדיוק השגיאה שקיבלת – במקום להתרסק, פשוט נדלג על השורה הבעייתית
                continue

            if sep_str:
                separator_nodes = [
                    s for s in (part.strip() for part in sep_str.split(separator_delim))
                    if s
                ]
            else:
                separator_nodes = []

            rows.append(
                {
                    "seed": seed,
                    "X": X,
                    "Y": Y,
                    "bucket": bucket,
                    "separator_str": sep_str,
                    "separator_nodes": separator_nodes,
                    "variance": variance,
                    "y_component_len": y_component_len,
                }
            )

    return rows



def _boxplot_by_bucket(
    values_by_bucket: Dict[int, List[float]],
    title: str,
    ylabel: str,
    out_path: Path,
):
    #blue
    box_facecolor = "#DCEBFF"
    box_edgecolor = "#1F4E79"
    median_color = "#1F4E79"
    mean_color = "#1F4E79"
    whisker_color = "#1F4E79"
    cap_color = "#1F4E79"

    #green
    box_facecolor = "#DFF5F2"
    box_edgecolor = "#0F766E"
    median_color = "#0F766E"
    mean_color = "#0F766E"
    whisker_color = "#0F766E"
    cap_color = "#0F766E"

    #purple
    box_facecolor = "#EFE7FF"
    box_edgecolor = "#5B21B6"
    median_color = "#5B21B6"
    mean_color = "#5B21B6"
    whisker_color = "#5B21B6"
    cap_color = "#5B21B6"


    buckets = sorted(values_by_bucket.keys())
    data = [values_by_bucket[b] for b in buckets]
    labels = [f"bucket_{b}" for b in buckets]

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        whis=(0, 100),
        patch_artist=True,  # מאפשר צבע מילוי לקופסאות
        boxprops=dict(facecolor=box_facecolor, edgecolor=box_edgecolor, linewidth=1.2),
        medianprops=dict(color=median_color, linewidth=1.6),
        meanprops=dict(marker="D", markerfacecolor=mean_color, markeredgecolor=mean_color, markersize=5),
        whiskerprops=dict(color=whisker_color, linewidth=1.2),
        capprops=dict(color=cap_color, linewidth=1.2),
    )

    #plt.boxplot(data, labels=labels, showmeans=True, whis=(0, 100))
    plt.xlabel("bucket")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# פונקציה לבדיקת מגמה
from typing import Dict, List, Tuple
import numpy as np

def _bucket_summary_series(
    by_bucket: Dict[int, List[float]],
    buckets: List[int],
    agg: str = "median",
) -> List[float]:
    """Return one summary value per bucket (median/mean). Empty bucket -> np.nan."""
    out = []
    for b in buckets:
        vals = by_bucket.get(b, [])
        if not vals:
            out.append(np.nan)
            continue
        arr = np.asarray(vals, dtype=float)
        out.append(float(np.nanmedian(arr) if agg == "median" else np.nanmean(arr)))
    return out

def _is_non_decreasing(
    series: List[float],
    tol: float = 0.0,
    min_points: int = 3,
) -> bool:
    """
    Non-decreasing with tolerance:
    series[i+1] + tol >= series[i]
    Ignores nan entries.
    """
    s = [x for x in series if not (isinstance(x, float) and np.isnan(x))]
    if len(s) < min_points:
        return False
    for a, b in zip(s, s[1:]):
        if b + tol < a:
            return False
    return True

def detect_increasing_trend_both(
    var_by_bucket: Dict[int, List[float]],
    y_by_bucket: Dict[int, List[float]],
    agg: str = "median",
    tol_var: float = 0.0,
    tol_y: float = 0.0,
    min_points: int = 3,
) -> Tuple[bool, dict]:
    """
    Returns:
      (trend_ok, details)

    trend_ok is True iff both variance and y_component are non-decreasing across buckets.
    """
    buckets = sorted(set(var_by_bucket) | set(y_by_bucket))
    var_series = _bucket_summary_series(var_by_bucket, buckets, agg=agg)
    y_series   = _bucket_summary_series(y_by_bucket, buckets, agg=agg)

    var_ok = _is_non_decreasing(var_series, tol=tol_var, min_points=min_points)
    y_ok   = _is_non_decreasing(y_series,   tol=tol_y,   min_points=min_points)

    details = {
        "buckets": buckets,
        "agg": agg,
        "var_series": var_series,
        "y_series": y_series,
        "var_non_decreasing": var_ok,
        "y_non_decreasing": y_ok,
    }
    return (var_ok and y_ok), details

def _combined_boxplot_by_bucket(
    var_by_bucket: Dict[int, List[float]],
    y_by_bucket: Dict[int, List[float]],
    title: str,
    out_path: Path,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    buckets = sorted(set(var_by_bucket) | set(y_by_bucket))

    var_data = [var_by_bucket.get(b, []) for b in buckets]
    y_data   = [y_by_bucket.get(b, []) for b in buckets]

    positions = np.arange(1, len(buckets) + 1)
    offset = 0.2

    fig, ax_var = plt.subplots(figsize=(10, 5))
    ax_y = ax_var.twinx()  # ✅ ציר Y שני

    # --- Variance (Blue) על ציר שמאל ---
    ax_var.boxplot(
        var_data,
        positions=positions - offset,
        widths=0.35,
        showmeans=True,
        whis=(0, 100),
        patch_artist=True,
        boxprops=dict(facecolor="#DCEBFF", edgecolor="#1F4E79", linewidth=1.2),
        medianprops=dict(color="#1F4E79", linewidth=1.6),
        meanprops=dict(marker="D", markerfacecolor="#1F4E79", markeredgecolor="#1F4E79", markersize=5),
        whiskerprops=dict(color="#1F4E79", linewidth=1.2),
        capprops=dict(color="#1F4E79", linewidth=1.2),
    )

    # --- Y Component (Purple) על ציר ימין ---
    ax_y.boxplot(
        y_data,
        positions=positions + offset,
        widths=0.35,
        showmeans=True,
        whis=(0, 100),
        patch_artist=True,
        boxprops=dict(facecolor="#EFE7FF", edgecolor="#5B21B6", linewidth=1.2),
        medianprops=dict(color="#5B21B6", linewidth=1.6),
        meanprops=dict(marker="D", markerfacecolor="#5B21B6", markeredgecolor="#5B21B6", markersize=5),
        whiskerprops=dict(color="#5B21B6", linewidth=1.2),
        capprops=dict(color="#5B21B6", linewidth=1.2),
    )

    # X-axis
    ax_var.set_xticks(positions)
    ax_var.set_xticklabels([f"bucket_{b}" for b in buckets])
    ax_var.set_xlabel("bucket")

    # Y labels (נפרד!)
    ax_var.set_ylabel("Variance")
    ax_y.set_ylabel("Y Component")

    ax_var.set_title(title)

    # Grid על ציר השונות (כמו אצלך)
    ax_var.grid(True, axis="y", linestyle="--", linewidth=0.5)

    # Legend ידני
    legend_elements = [
        Patch(facecolor="#DCEBFF", edgecolor="#1F4E79", label="Variance"),
        Patch(facecolor="#EFE7FF", edgecolor="#5B21B6", label="Y Component"),
    ]
    #ax_var.legend(handles=legend_elements, loc="upper right")
    ax_var.legend(handles=legend_elements, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_bucket_boxplots(
    input_file: str | Path,
    variance: str,
    output_dir:str,
    mode: str = "global",   # "global" or "per_run"
) -> List[Path]:
    """
    mode="global": one variance plot + one y_component plot for all runs combined.
    mode="per_run": for each (seed,X,Y), create two plots.

    Returns list of created file paths.
    """
    rows = parse_bucket_stats_csv(input_file)
    if not rows:
        print(input_file)
        raise ValueError("No data parsed from file.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []

    if mode not in {"global", "per_run"}:
        raise ValueError("mode must be 'global' or 'per_run'.")

    def add_row_to_maps(r, var_map, y_map):
        var_map.setdefault(r["bucket"], []).append(r["variance"])
        y_map.setdefault(r["bucket"], []).append(r["y_component_len"])

    if mode == "global":
        var_by_bucket: Dict[int, List[float]] = {}
        y_by_bucket: Dict[int, List[float]] = {}
        for r in rows:
            add_row_to_maps(r, var_by_bucket, y_by_bucket)

        #p1 = output_dir / f"{variance}_boxplot_variance_by_bucket__ALL_RUNS.png"
        #p2 = output_dir / f"{variance}_boxplot_y_component_len_by_bucket__ALL_RUNS.png"

        p1 =  f"{variance}_boxplot_variance_by_bucket__ALL_RUNS.png"
        p2 =  f"{variance}_boxplot_y_component_len_by_bucket__ALL_RUNS.png"

        '''
        _boxplot_by_bucket(
            var_by_bucket,
            title="Variance by bucket (all runs combined)",
            ylabel="variance",
            out_path=p1,
        )
        _boxplot_by_bucket(
            y_by_bucket,
            title="Y-component size by bucket (all runs combined)",
            ylabel="len(y_component)",
            out_path=p2,
        )
        created.extend([p1, p2])
        return created'''
        p = output_dir / f"{variance}_variance_and_y_component__ALL_RUNS.png"

        _combined_boxplot_by_bucket(
            var_by_bucket,
            y_by_bucket,
            title="Variance and Y-component by bucket (all runs combined)",
            out_path=p,
            #agg="mean",  # או "median"
        )

        created.append(p)
        return created

    # mode == "per_run"
    # group by (seed,X,Y)
    groups: Dict[Tuple[int, str, str], List[dict]] = {}
    for r in rows:
        key = (r["seed"], r["X"], r["Y"])
        groups.setdefault(key, []).append(r)

    for (seed, X, Y), gr in groups.items():
        var_by_bucket: Dict[int, List[float]] = {}
        y_by_bucket: Dict[int, List[float]] = {}
        for r in gr:
            add_row_to_maps(r, var_by_bucket, y_by_bucket)

        # --- skip runs where every bucket has only 1 sample (no real boxplot) ---
        max_var_samples = max((len(v) for v in var_by_bucket.values()), default=0)
        max_y_samples = max((len(v) for v in y_by_bucket.values()), default=0)

        # אם בכל ה-buckets יש לכל היותר מופע אחד (גם ב-variance וגם ב-y_component)
        if max(max_var_samples, max_y_samples) <= 1:
            continue

        base = _safe_filename(f"seed={seed}_X={X}_Y={Y}")
        p1 = output_dir / f"{base}__variance_boxplot.png"
        p2 = output_dir / f"{base}__y_component_boxplot.png"

        # _boxplot_by_bucket(
        #     var_by_bucket,
        #     title=f"Variance by bucket (seed={seed}, X={X}, Y={Y})",
        #     ylabel="variance",
        #     out_path=p1,
        # )
        # _boxplot_by_bucket(
        #     y_by_bucket,
        #     title=f"Y-component size by bucket (seed={seed}, X={X}, Y={Y})",
        #     ylabel="len(y_component)",
        #     out_path=p2,
        # )

        trend_ok, _ = detect_increasing_trend_both(
            var_by_bucket,
            y_by_bucket,
            agg="median",  # או "mean"
            tol_var=0.0,  # אפשר למשל 1e-6 או 0.01 אם יש רעש
            tol_y=0.0,
            min_points=3
        )

        prefix = "TREND__" if trend_ok else ""

        p = output_dir / f"{prefix}{variance}_variance_and_y_component_(seed={seed}, X={X}, Y={Y}).png"
        _combined_boxplot_by_bucket(
            var_by_bucket,
            y_by_bucket,
            title="Variance and Y-component by bucket)",
            out_path=p,
        )


        created.extend([p1, p2])

    return created


def make_bucket_bar_plots(
    input_file: str | Path,
        variance:str,
    agg: str = "mean",  # "mean" or "median"
) -> Tuple[Path, Path]:
    """
    Builds TWO bar charts from the attached CSV:
      1) X-axis: bucket, Y-axis: aggregated y_component_len (across all rows)
      2) X-axis: bucket, Y-axis: aggregated variance (across all rows)

    By default aggregates with mean; can use median.

    Returns (y_component_bar_path, variance_bar_path).
    """
    rows = parse_bucket_stats_csv(input_file)
    if not rows:
        raise ValueError("No data parsed from file (check header/format).")

    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    # collect values by bucket
    y_by_bucket: Dict[int, List[float]] = {}
    v_by_bucket: Dict[int, List[float]] = {}
    for r in rows:
        b = r["bucket"]
        y_by_bucket.setdefault(b, []).append(r["y_component_len"])
        v_by_bucket.setdefault(b, []).append(r["variance"])

    buckets = sorted(set(y_by_bucket) | set(v_by_bucket))
    labels = [f"bucket_{b}" for b in buckets]

    def _agg(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        vals_sorted = sorted(vals)
        if agg == "median":
            n = len(vals_sorted)
            mid = n // 2
            return vals_sorted[mid] if (n % 2 == 1) else 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
        # default mean
        return sum(vals_sorted) / len(vals_sorted)

    y_vals = [_agg(y_by_bucket.get(b, [])) for b in buckets]
    v_vals = [_agg(v_by_bucket.get(b, [])) for b in buckets]

    # ---- Plot 1: y_component_len ----
    y_path = output_dir / f"bar_y_component_len_by_bucket_{variance}_{agg}.png"
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(buckets)), y_vals)
    plt.xticks(range(len(buckets)), labels, rotation=0)
    plt.xlabel("bucket")
    plt.ylabel(f"y_component_len ({agg})")
    plt.title(f"y_component_len by bucket ({agg}, all runs combined)")
    plt.tight_layout()
    plt.savefig(y_path, dpi=200)
    plt.close()

    # ---- Plot 2: variance ----
    v_path = output_dir / f"bar_variance_by_bucket_{variance}_{agg}.png"
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(buckets)), v_vals)
    plt.xticks(range(len(buckets)), labels, rotation=0)
    plt.xlabel("bucket")
    plt.ylabel(f"variance ({agg})")
    plt.title(f"variance by bucket ({agg}, all runs combined)")
    plt.tight_layout()
    plt.savefig(v_path, dpi=200)
    plt.close()

    return y_path, v_path


# Example:
# make_bucket_bar_plots(
#     "26_2_23_bucket_statistics_SEM_50_0.25_beta1.csv",
#     output_dir="bucket_statistics",
#     agg="mean",
# )