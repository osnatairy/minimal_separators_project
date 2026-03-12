from pandas.api.types import is_numeric_dtype
from pandas import to_numeric
import pandas as pd
import networkx as nx
import streamlit as st
import numpy as np
from pathlib import Path
from typing import Optional

from causallearn.search.ConstraintBased import PC
from i_o.json_loader import save_bn_json

def convert_df_columns_snake_to_pascal_inplace(df: pd.DataFrame):
    """
    In-place conversion of all column names from snake_case to PascalCase.
    e.g. 'result_cache_hit' -> 'ResultCacheHit'.
    """
    df_copy = df.copy()
    def to_pascal_case(snake: str):
        # Split by underscores, capitalize each part, then join
        parts = snake.split("_")
        return "".join(word.capitalize() for word in parts)

    #df_copy.rename(columns=lambda col: to_pascal_case(col), inplace=True)
    return df_copy


def discover_causal_dag(df: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """Run PC on a DataFrame, keep only the fully-directed edges (color='black'), and return a DAG."""
    causal_graph = PC.pc(df.values, alpha=alpha, verbose=verbose)
    causal_graph.to_nx_graph()

    graph_int = causal_graph.nx_graph
    edges_to_remove = [(u, v) for u, v, d in graph_int.edges(data=True) if d.get('color') != 'b']
    graph_int.remove_edges_from(edges_to_remove)

    for _, _, data in graph_int.edges(data=True):
        data['color'] = 'black'

    mapping = {i: df.columns[i] for i in graph_int.nodes()}
    return nx.relabel_nodes(graph_int, mapping)

def preprocess_for_pc(df):

    selected_cols = [

        "MONTH",
        "DAY_OF_WEEK",
        "SCHEDULED_DEPARTURE",
        "DISTANCE",
        "AIR_TIME",
        "TAXI_OUT",
        "TAXI_IN",
        "DEPARTURE_DELAY",
        "ARRIVAL_DELAY",
        "WEATHER_DELAY",
        "AIRLINE_DELAY"

    ]

    df = df[selected_cols].copy()

    # טיפול בחסרים
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    print("Shape for PC:", df.shape)
    return df


def generate_dag_from_dataset(df, alpha):
    """
    Generate a DAG from the provided dataset.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        if not is_numeric_dtype(df_copy[col]):
            df_copy[col] = to_numeric(df_copy[col], errors='coerce')

    df_copy.dropna(inplace=True)
    df_copy = convert_df_columns_snake_to_pascal_inplace(df_copy)
    G = discover_causal_dag(df_copy, alpha=alpha)
    st.session_state.is_loading = False
    return G

def nx_dag_to_dot(
    G: nx.DiGraph,
    out_path: str,
    graph_name: str = "causal_dag",
    include_node_decls: bool = True,
) -> str:
    """
    Writes a NetworkX DiGraph (DAG) to a Graphviz DOT file like:

    digraph causal_dag {
        A;
        B;
        A -> B;
    }

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph (preferably a DAG).
    out_path : str
        Path to save the .dot file.
    graph_name : str
        Name of the digraph in the DOT header.
    include_node_decls : bool
        If True, emits explicit node declaration lines ("node;") before edges.

    Returns
    -------
    str
        The output path.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")

    # (לא חובה) בדיקת DAG
    if not nx.is_directed_acyclic_graph(G):
        # אפשר עדיין לכתוב DOT גם אם זה לא DAG, אבל לפי הדרישה שלך זה DAG
        raise ValueError("G is not a DAG (contains a directed cycle).")

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # כדי לקבל קובץ יציב בין ריצות: נמיין צמתים וקשתות לפי str
    nodes = sorted(G.nodes(), key=str)
    edges = sorted(G.edges(), key=lambda e: (str(e[0]), str(e[1])))

    def dot_id(x) -> str:
        """
        Graphviz IDs:
        - אם זה מזהה "נקי" (אותיות/ספרות/_) נשאיר בלי מרכאות
        - אחרת נוסיף מרכאות
        """
        s = str(x)
        #if s.replace("_", "").isalnum() and not s[0].isdigit():
            #return s
        return '"' + s.replace('"', '\\"') + '"'

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"digraph {graph_name} {{\n")

        if include_node_decls:
            for n in nodes:
                f.write(f"    {dot_id(n)};\n")
            f.write("\n")

        for u, v in edges:
            f.write(f"    {dot_id(u)} -> {dot_id(v)};\n")

        f.write("}\n")

    return out_path

if __name__ == "__main__":

    df = preprocess_for_pc(pd.read_csv("../BN_DATA/flights/flights_dataset.csv"))
    G = generate_dag_from_dataset(df, alpha=0.05)
    nx_dag_to_dot(G, "../BN_DATA/flights/flights.dot", graph_name="query_performance")

    print("done!")



