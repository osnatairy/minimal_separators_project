import os
import re
import json
import ast
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "seperators_per_graphs"  # <-- לשנות לתיקייה שלך
OUTPUT_EXCEL = "separators_analysis.xlsx"

FILE_PREFIX = "seps2_bn"
FILE_PREFIX = "seps2_bn"


# ============================================================
# Helpers
# ============================================================

def parse_file_metadata(filename: str):
    """
    Extracts number of nodes and edge probability from file name.

    Example:
        seps2_sem_40_0.3.csv
    returns:
        nodes = 40
        edge_prob = 0.3
    """
    pattern = r"seps2_bn_(\d+)_(\d+(?:\.\d+)?)\.csv$"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"File name does not match expected format: {filename}")

    nodes = int(match.group(1))
    edge_prob = float(match.group(2))

    return nodes, edge_prob


def safe_parse_list(value):
    """
    Parses values_json safely.
    Supports JSON strings and Python-like list strings.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if value == "":
        return []

    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def separator_lengths(zsets):
    """
    Given list of separators, returns list of their lengths.

    Example:
        [["V1", "V2"], ["V3"]]
    returns:
        [2, 1]
    """
    lengths = []

    for z in zsets:
        if z is None:
            continue

        if isinstance(z, (list, tuple, set)):
            lengths.append(len(z))
        else:
            lengths.append(1)

    return lengths


# ============================================================
# Load all files
# ============================================================

def load_all_separator_files(data_dir: str):
    """
    Loads all files that start with seps2_sem_ and end with .csv.

    Returns one row per XY pair.
    """
    data_dir = Path(data_dir)

    all_rows = []

    files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file()
        and f.name.startswith(FILE_PREFIX)
        and f.suffix.lower() == ".csv"
    ])

    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} starting with {FILE_PREFIX}")

    for file_path in files:
        nodes, edge_prob = parse_file_metadata(file_path.name)

        df = pd.read_csv(file_path)

        # Normalize possible column names
        if "values_json" not in df.columns:
            raise ValueError(f"Missing values_json column in {file_path.name}")

        if "num_values" not in df.columns:
            df["num_values"] = df["values_json"].apply(lambda x: len(safe_parse_list(x)))

        if "time" not in df.columns:
            # If your file does not include time, keep as NaN
            df["time"] = pd.NA

        required = ["seed", "X", "Y", "num_values", "values_json", "time"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns in {file_path.name}: {missing}")

        for _, row in df.iterrows():
            zsets = safe_parse_list(row["values_json"])
            lengths = separator_lengths(zsets)

            all_rows.append({
                "source_file": file_path.name,
                "nodes": nodes,
                "edge_prob": edge_prob,
                "seed": row["seed"],
                "X": row["X"],
                "Y": row["Y"],
                "time": row["time"],
                "num_separators": int(row["num_values"]),
                "avg_separator_length": sum(lengths) / len(lengths) if lengths else 0,
                "min_separator_length": min(lengths) if lengths else 0,
                "max_separator_length": max(lengths) if lengths else 0,
                "separator_lengths": lengths,
                "separators": zsets,
            })

    return pd.DataFrame(all_rows)


# ============================================================
# Build analysis tables
# ============================================================

def build_analysis_tables(pair_df: pd.DataFrame):
    """
    Builds summary tables:
      1. pair-level table
      2. graph-level summary
      3. file-level summary
      4. separator length distribution per graph
      5. separator length distribution per file
    """

    pair_df = pair_df.copy()

    pair_df["time"] = pd.to_numeric(pair_df["time"], errors="coerce")
    pair_df["num_separators"] = pd.to_numeric(pair_df["num_separators"], errors="coerce")
    pair_df["avg_separator_length"] = pd.to_numeric(pair_df["avg_separator_length"], errors="coerce")

    # --------------------------------------------------------
    # Graph-level summary: one row per graph/seed
    # --------------------------------------------------------
    graph_summary = (
        pair_df
        .groupby(["source_file", "nodes", "edge_prob", "seed"], dropna=False)
        .agg(
            num_xy_pairs=("X", "count"),
            avg_time_per_xy=("time", "mean"),
            median_time_per_xy=("time", "median"),
            max_time_per_xy=("time", "max"),
            time_all_xy=("time", "sum"),
            avg_num_separators_per_xy=("num_separators", "mean"),
            median_num_separators_per_xy=("num_separators", "median"),
            max_num_separators_for_xy=("num_separators", "max"),
            avg_separator_length_per_xy=("avg_separator_length", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # File-level summary: all 10 graphs together
    # --------------------------------------------------------
    file_summary = (
        graph_summary
        .groupby(["source_file", "nodes", "edge_prob"], dropna=False)
        .agg(
            num_graphs=("seed", "nunique"),
            total_xy_pairs=("num_xy_pairs", "sum"),
            avg_xy_pairs_per_graph=("num_xy_pairs", "mean"),
            avg_time_per_xy_all_graphs=("avg_time_per_xy", "mean"),
            median_graph_avg_time=("avg_time_per_xy", "median"),
            avg_time_per_graph=("time_all_xy", "mean"),
            avg_num_separators_per_xy=("avg_num_separators_per_xy", "mean"),
            avg_separator_length=("avg_separator_length_per_xy", "mean"),
            max_time_per_xy_overall=("max_time_per_xy", "max"),
            max_num_separators_for_xy_overall=("max_num_separators_for_xy", "max"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Separator length distribution per graph
    # --------------------------------------------------------
    length_rows = []

    for _, row in pair_df.iterrows():
        for length in row["separator_lengths"]:
            length_rows.append({
                "source_file": row["source_file"],
                "nodes": row["nodes"],
                "edge_prob": row["edge_prob"],
                "seed": row["seed"],
                "X": row["X"],
                "Y": row["Y"],
                "separator_length": length,
            })

    sep_lengths_df = pd.DataFrame(length_rows)

    if not sep_lengths_df.empty:
        sep_len_by_graph = (
            sep_lengths_df
            .groupby(["source_file", "nodes", "edge_prob", "seed", "separator_length"])
            .size()
            .reset_index(name="count")
        )

        sep_len_by_file = (
            sep_lengths_df
            .groupby(["source_file", "nodes", "edge_prob", "separator_length"])
            .size()
            .reset_index(name="count")
        )

        sep_len_by_file["percent"] = (
            sep_len_by_file
            .groupby(["nodes", "edge_prob"])["count"]
            .transform(lambda x: 100 * x / x.sum())
        )
    else:
        sep_len_by_graph = pd.DataFrame()
        sep_len_by_file = pd.DataFrame()

    return {
        "pair_level": pair_df,
        "graph_summary": graph_summary,
        "file_summary": file_summary,
        "sep_len_by_graph": sep_len_by_graph,
        "sep_len_by_file": sep_len_by_file,
    }



# ============================================================
# analyzed pivot table per len seps
# ============================================================
def add_separator_length_chart_by_nodes_prob(
    writer,
    sep_len_df,
    sheet_name="SepLen_Chart_Data",
    chart_sheet_name="SepLen_Charts",
    value_col="percent",
):
    """
    Creates pivot tables and clustered column charts for separator lengths.

    Expected sep_len_df columns:
        nodes
        edge_prob
        separator_length
        percent   OR count

    The output table format is:
        edge_prob | 1 | 2 | 3 | 4 | 5 | ...

    For each graph size 'nodes', a separate pivot table and chart are created.
    """

    workbook = writer.book

    if sep_len_df.empty:
        return

    df = sep_len_df.copy()

    df["edge_prob"] = pd.to_numeric(df["edge_prob"], errors="coerce")
    df["separator_length"] = pd.to_numeric(df["separator_length"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Create sheets
    data_ws = workbook.add_worksheet(sheet_name)
    chart_ws = workbook.add_worksheet(chart_sheet_name)

    writer.sheets[sheet_name] = data_ws
    writer.sheets[chart_sheet_name] = chart_ws

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1
    })

    row_start = 0
    chart_row = 1

    for nodes in sorted(df["nodes"].dropna().unique()):
        sub = df[df["nodes"] == nodes]

        pivot = (
            sub.pivot_table(
                index="edge_prob",
                columns="separator_length",
                values=value_col,
                aggfunc="sum",
                fill_value=0
            )
            .sort_index()
            .reset_index()
        )

        pivot.columns = [str(c) if c != "edge_prob" else "edge_prob" for c in pivot.columns]

        # Write title
        data_ws.write(row_start, 0, f"Nodes = {nodes}", header_fmt)
        row_start += 1

        # Write pivot table
        pivot.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=row_start,
            startcol=0,
            index=False
        )

        n_rows, n_cols = pivot.shape

        # Format headers
        for col in range(n_cols):
            data_ws.write(row_start, col, pivot.columns[col], header_fmt)

        # Create clustered column chart
        chart = workbook.add_chart({"type": "column"})

        categories = [
            sheet_name,
            row_start + 1,
            0,
            row_start + n_rows,
            0
        ]

        for col in range(1, n_cols):
            chart.add_series({
                "name":       [sheet_name, row_start, col],
                "categories": categories,
                "values":     [sheet_name, row_start + 1, col, row_start + n_rows, col],
            })

        chart.set_title({
            "name": f"Separator Length Distribution - Nodes {nodes}"
        })
        chart.set_x_axis({"name": "Edge probability"})
        chart.set_y_axis({"name": value_col})
        chart.set_legend({"position": "bottom"})

        chart_ws.insert_chart(chart_row, 1, chart, {
            "x_scale": 1.5,
            "y_scale": 1.4
        })

        row_start += n_rows + 4
        chart_row += 22

# ============================================================
# Write Excel
# ============================================================

def write_excel_report(tables: dict, output_path: str):
    """
    Writes all tables to an Excel file with formatting.
    """

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1
        })

        number_fmt = workbook.add_format({"num_format": "0.0000"})
        int_fmt = workbook.add_format({"num_format": "0"})
        text_wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        readme = pd.DataFrame({
            "Description": [
                "This workbook summarizes separator search results.",
                "Each source file contains several graphs identified by seed.",
                "Graph_Summary gives one row per seed/graph.",
                "File_Summary aggregates all graphs from the same source file.",
                "Pair_Level contains one row per XY pair.",
                "SepLen_By_Graph and SepLen_By_File contain separator length distributions.",
            ]
        })

        sheet_map = {
            "README": readme,
            "Pair_Level": tables["pair_level"],
            "Graph_Summary": tables["graph_summary"],
            "File_Summary": tables["file_summary"],
            "SepLen_By_Graph": tables["sep_len_by_graph"],
            "SepLen_By_File": tables["sep_len_by_file"],
        }

        for sheet_name, df in sheet_map.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            worksheet = writer.sheets[sheet_name]

            if df.empty:
                continue

            rows, cols = df.shape

            worksheet.freeze_panes(1, 0)
            #worksheet.autofilter(0, 0, rows, cols - 1)

            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)

                width = min(max(len(str(col_name)) + 2, 12), 40)

                if col_name in ["separators", "separator_lengths"]:
                    worksheet.set_column(col_idx, col_idx, 45, text_wrap_fmt)
                elif "time" in col_name or "avg" in col_name or "median" in col_name:
                    worksheet.set_column(col_idx, col_idx, width, number_fmt)
                elif "num" in col_name or "count" in col_name or "nodes" in col_name or "seed" in col_name:
                    worksheet.set_column(col_idx, col_idx, width, int_fmt)
                else:
                    worksheet.set_column(col_idx, col_idx, width)

            # Add Excel table
            worksheet.add_table(
                0,
                0,
                rows,
                cols - 1,
                {
                    "columns": [{"header": col} for col in df.columns],
                    "style": "Table Style Medium 2",
                }
            )

        # Optional charts
        add_charts(writer, tables)

        add_separator_length_chart_by_nodes_prob(
            writer,
            tables["sep_len_by_file"],
            value_col="percent"
        )

    print(f"Excel report written to: {output_path}")


def add_charts(writer, tables: dict):
    """
    Adds simple charts to the Excel workbook.
    """
    workbook = writer.book

    if not tables["file_summary"].empty:
        df = tables["file_summary"]
        sheet = writer.sheets["File_Summary"]

        chart = workbook.add_chart({"type": "column"})
        rows = len(df)

        source_file_col = df.columns.get_loc("source_file")
        avg_time_col = df.columns.get_loc("avg_time_per_xy_all_graphs")

        chart.add_series({
            "name": "Average time per XY",
            "categories": ["File_Summary", 1, source_file_col, rows, source_file_col],
            "values": ["File_Summary", 1, avg_time_col, rows, avg_time_col],
        })

        chart.set_title({"name": "Average Separator Search Time per XY"})
        chart.set_x_axis({"name": "File"})
        chart.set_y_axis({"name": "Average time"})
        chart.set_legend({"none": True})

        sheet.insert_chart("L2", chart)

    if not tables["sep_len_by_file"].empty:
        df = tables["sep_len_by_file"]
        sheet = writer.sheets["SepLen_By_File"]

        chart = workbook.add_chart({"type": "column"})
        rows = len(df)

        length_col = df.columns.get_loc("separator_length")
        count_col = df.columns.get_loc("count")

        chart.add_series({
            "name": "Separator length count",
            "categories": ["SepLen_By_File", 1, length_col, rows, length_col],
            "values": ["SepLen_By_File", 1, count_col, rows, count_col],
        })

        chart.set_title({"name": "Separator Length Distribution"})
        chart.set_x_axis({"name": "Separator length"})
        chart.set_y_axis({"name": "Count"})
        chart.set_legend({"none": True})

        sheet.insert_chart("H2", chart)


# ============================================================
# Main
# ============================================================

def main():
    pair_df = load_all_separator_files(DATA_DIR)

    tables = build_analysis_tables(pair_df)

    output_path = os.path.join(DATA_DIR, OUTPUT_EXCEL)
    write_excel_report(tables, output_path)

    print("\nSummary:")
    print(f"Number of XY rows: {len(tables['pair_level'])}")
    print(f"Number of graph rows: {len(tables['graph_summary'])}")
    print(f"Number of file rows: {len(tables['file_summary'])}")


if __name__ == "__main__":
    main()