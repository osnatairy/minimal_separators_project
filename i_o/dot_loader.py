import string
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_pydot import read_dot

import re
from collections import defaultdict
from typing import Dict, List, Tuple

def normalize_node(n: str) -> str:
    n = str(n).strip()
    if n.startswith('"') and n.endswith('"'):
        n = n[1:-1]
    return n.strip()



def dot_to_mapping_and_edges(dot_path: str, scheme: str = "letters"):
    """
    scheme: "letters" -> A,B,C,...
            "v"       -> V1,V2,V3,...
    returns:
      name_to_id: dict[str,str]
      id_to_name: dict[str,str]
      mapped_edges: list[tuple[str,str]]
    """
    # תיקיית הפרויקט (minimal_separators_project)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # נתיב מלא לקובץ
    full_path = PROJECT_ROOT / dot_path

    raw = read_dot(full_path)  # usually MultiDiGraph
    # collect normalized nodes
    nodes = sorted({normalize_node(n) for n in raw.nodes()
                    if normalize_node(n) and normalize_node(n).lower() not in {"graph","node","edge"}})

    if scheme == "letters":
        if len(nodes) > 26:
            raise ValueError("Too many nodes for letters A-Z. Use scheme='v'.")
        ids = list(string.ascii_uppercase[:len(nodes)])
    elif scheme == "v":
        ids = [f"V{i}" for i in range(1, len(nodes)+1)]
    else:
        raise ValueError("scheme must be 'letters' or 'v'")

    name_to_id = dict(zip(nodes, ids))
    id_to_name = {v: k for k, v in name_to_id.items()}

    # map edges
    mapped_edges = []
    for u, v in raw.edges():
        uu, vv = normalize_node(u), normalize_node(v)
        if uu in name_to_id and vv in name_to_id:
            mapped_edges.append((name_to_id[uu], name_to_id[vv]))

    # remove duplicates while keeping order
    seen = set()
    mapped_edges_unique = []
    for e in mapped_edges:
        if e not in seen:
            seen.add(e)
            mapped_edges_unique.append(e)

    return name_to_id, id_to_name, mapped_edges_unique

def build_redshift_dag_from_dot(
    dot_path: str = "/mnt/data/amazon_redshift.dot",
    s: str = "result_cache_hit",
    t: str = "elapsed_time",
) -> nx.DiGraph:
    """
    Loads a DOT file into a networkx.DiGraph (DAG), normalizes node names,
    and sets G.graph['st'] = (s, t) so it can plug into your pipeline.
    """
    # read_dot often returns a MultiDiGraph with attributes; we convert to DiGraph
    from networkx.drawing.nx_pydot import read_dot

    raw = read_dot(dot_path)  # usually MultiDiGraph

    G = nx.DiGraph()
    # Normalize node names (strip quotes/whitespace) and ignore pydot artifacts
    def norm(n):
        n = str(n)
        if n.startswith('"') and n.endswith('"'):
            n = n[1:-1]
        return n.strip()

    # Add nodes (skip graph-level pseudo nodes if any)
    for n in raw.nodes():
        nn = norm(n)
        if nn and nn.lower() not in {"graph", "node", "edge"}:
            G.add_node(nn)

    # Add edges
    for u, v in raw.edges():
        uu, vv = norm(u), norm(v)
        if uu and vv:
            G.add_edge(uu, vv)

    # Sanity checks
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Loaded graph is not a DAG (contains a directed cycle).")

    if s not in G or t not in G:
        raise ValueError(f"s/t not in graph. Missing: "
                         f"{'s' if s not in G else ''} {'t' if t not in G else ''}".strip())

    G.graph["st"] = (s, t)
    return G



EDGE_RE = re.compile(r'^\s*"?([A-Za-z0-9_]+)"?\s*->\s*"?([A-Za-z0-9_]+)"?\s*;?\s*$')

def parse_dot_graph(dot_path: str) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, List[str]]]:
    """
    מחזיר:
      nodes: רשימת צמתים (בסדר הופעה)
      edges: רשימת קשתות (src, dst) בסדר הופעה
      parents: dict child -> [parents...] בסדר הופעה בקובץ
    """
    nodes_order: List[str] = []
    seen_nodes = set()

    edges: List[Tuple[str, str]] = []
    parents: Dict[str, List[str]] = defaultdict(list)

    with open(dot_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # דלג על שורות שאינן קשתות/הצהרות
            if not line or line.startswith("//"):
                continue

            # נסה לזהות קשת A -> B
            m = EDGE_RE.match(line)
            if m:
                src, dst = m.group(1), m.group(2)

                # שמירת סדר צמתים
                for n in (src, dst):
                    if n not in seen_nodes:
                        nodes_order.append(n)
                        seen_nodes.add(n)

                edges.append((src, dst))

                # שמירת סדר הורים לילד
                if src not in parents[dst]:
                    parents[dst].append(src)

                # ודא שהצמתים קיימים גם אם אין להם הורים
                parents.setdefault(src, parents.get(src, []))
                continue

            # אפשר גם לזהות הצהרת צומת בודדת כמו: query_template;
            # (בדוגמה שלך יש block של שמות צמתים בלי קשתות)
            if line.endswith(";"):
                token = line[:-1].strip().strip('"')
                # אם זו לא מילה של DOT כמו digraph / { / }
                if token and token not in ("digraph", "{", "}"):
                    if token not in seen_nodes:
                        nodes_order.append(token)
                        seen_nodes.add(token)
                    parents.setdefault(token, parents.get(token, []))

    # ודא שלכל node יש ערך ב-parents
    for n in nodes_order:
        parents.setdefault(n, [])

    return nodes_order, edges, dict(parents)