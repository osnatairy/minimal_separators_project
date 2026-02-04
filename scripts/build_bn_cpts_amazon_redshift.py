import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, OrderedDict

from pathlib import Path

from i_o.dot_loader import parse_dot_graph
from i_o.json_loader import load_dataframe,build_bn_json
from i_o.utils import topo_sort

from bn.discretize import discretize_columns
from bn.cpt import compute_cpt

# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--graph",default="/BN_DATA/amazon_redshift.dot", required=True, help="Path to DOT graph file")
    # ap.add_argument("--data", default="/BN_DATA/amazon_redshift_dataset.pkl", required=True, help="Path to data file (.pkl DataFrame or .csv)")
    # ap.add_argument("--out",default="/BN_DATA/bn_amazon_redshift.json",  required=True, help="Output JSON path")
    ap.add_argument("--bins", type=int, default=5, help="Number of quantile bins for numeric discretization")
    ap.add_argument("--numeric-policy", default="auto", choices=["auto", "always", "never"],
                    help="How to treat numeric columns: auto / always / never")
    ap.add_argument("--explicit-discrete", nargs="*", default=[],
                    help="List of variables to keep discrete even if numeric (e.g. 0/1 flags)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha (Dirichlet prior)")
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]

    GRAPH_PATH = ROOT / "BN_DATA" / "amazon_redshift.dot"
    DATA_PATH = ROOT / "BN_DATA" / "amazon_redshift_dataset.pkl"
    OUT_PATH = ROOT / "BN_DATA" / "bn_amazon_redshift.json"

    # 1) Parse graph
    nodes, edges, parents_map = parse_dot_graph(GRAPH_PATH)

    # 2) Load data
    df = load_dataframe(DATA_PATH)

    # 3) Discretize (make everything categorical)
    df_cat, domains = discretize_columns(
        df=df,
        nodes=nodes,
        bins=args.bins,
        numeric_policy=args.numeric_policy,
        explicit_discrete=args.explicit_discrete,
    )

    # 4) Topological order (אופציונלי, אבל שימושי)
    topo = topo_sort(nodes, edges)

    # 5) Compute CPTs
    cpts: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for node in topo:
        parents = parents_map.get(node, [])
        cpts[node] = compute_cpt(
            df_cat=df_cat,
            child=node,
            parents=parents,
            domains=domains,
            alpha=args.alpha,
        )

    # 6) Build and write JSON
    out_json = build_bn_json(
        nodes=nodes,
        edges=edges,
        parents_map=parents_map,
        domains=domains,
        cpts=cpts,
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"Wrote BN JSON to: {OUT_PATH}")


if __name__ == "__main__":
    main()