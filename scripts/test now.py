import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

from pipelines.adjust_sets import find_adjustment_sets_for_pair

def make_exponential_separator_dag(k: int = 8) -> nx.DiGraph:
    """
    Creates a DAG G with 2^k minimal X-Y separators
    in its moral graph H.

    Structure:
        X -> A_i -> B_i -> Y

    for i = 1,...,k.
    """
    G = nx.DiGraph()

    X, Y = "X", "Y"
    G.add_nodes_from([X, Y])

    for i in range(1, k + 1):
        A = f"A{i}"
        B = f"B{i}"

        G.add_edge(X, A)
        G.add_edge(A, B)
        G.add_edge(B, Y)

    assert nx.is_directed_acyclic_graph(G)

    return G


def moral_graph(G: nx.DiGraph) -> nx.Graph:
    """
    Moralize a DAG:
    1. Drop directions.
    2. Connect parents of every node.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # skeleton
    H.add_edges_from(G.to_undirected().edges())

    # marry parents
    for v in G.nodes():
        parents = list(G.predecessors(v))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                H.add_edge(parents[i], parents[j])

    return H


def all_minimal_xy_separators_for_this_graph(k: int):
    """
    For this specific construction:
    every minimal separator chooses one node from each pair {A_i, B_i}.
    Total: 2^k separators.
    """
    choices = []

    for i in range(1, k + 1):
        choices.append((f"A{i}", f"B{i}"))

    separators = [set(choice) for choice in product(*choices)]
    return separators


def layout_for_exponential_graph(k: int):
    pos = {}

    pos["X"] = (0, 0)
    pos["Y"] = (3, 0)

    for i in range(1, k + 1):
        y = i - (k + 1) / 2
        pos[f"A{i}"] = (1, y)
        pos[f"B{i}"] = (2, y)

    return pos


def draw_G_and_H(G: nx.DiGraph):
    H = moral_graph(G)
    k = sum(1 for v in G.nodes() if v.startswith("A"))
    pos = layout_for_exponential_graph(k)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=900,
        arrows=True,
        arrowsize=18,
        font_size=9,
    )
    plt.title("DAG G")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    nx.draw(
        H,
        pos,
        with_labels=True,
        node_size=900,
        font_size=9,
    )
    plt.title("Moral graph H")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return H


# Example
k = 8

G = make_exponential_separator_dag(k)
Y = "X"
X = "Y"

R = list(G.nodes())
I = []

H, Z_sets, total_time = find_adjustment_sets_for_pair(
    G,
    X,
    Y,
    "smallminimalseps",
    R=R,
    I=I,
    get_closest_seps=False,
    #K=5,
)

zsets = sorted({
    tuple(sorted(Z))
    for Z in Z_sets
    if 1 <= len(Z) <= 5
})
H = draw_G_and_H(G)
for Z in Z_sets:
    print(Z)

print("Total Z_sets:", len(Z_sets))
separators = all_minimal_xy_separators_for_this_graph(k)

print("G is DAG:", nx.is_directed_acyclic_graph(G))
print("Number of minimal X-Y separators:", len(separators))
print("Expected:", 2 ** k)

print("\nFirst 5 separators:")
for S in separators[:5]:
    print(S)