import string
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_pydot import read_dot

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
