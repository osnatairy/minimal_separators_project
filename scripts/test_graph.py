import random
import networkx as nx
import matplotlib.pyplot as plt
from pipelines.adjust_sets import find_adjustment_sets_for_pair

def make_handdrawn_style_dag(
    layer_sizes = (5, 6, 7, 8),
    # layer_sizes=(6, 8, 10, 12, 14, 16),
    band_width=1,
    seed=1,
    node_prefix="V",
):
    """
    Creates a DAG shaped like your drawing:

        X <- L0 <- L1 <- L2 <- ... <- Lk -> Y

    Sparse banded edges between layers, not complete bipartite.
    """

    rng = random.Random(seed)

    G = nx.DiGraph()

    X = f"{node_prefix}0"
    Y = f"{node_prefix}1"

    G.add_nodes_from([X, Y])

    next_id = 2
    layers = []

    # Create layers
    for size in layer_sizes:
        layer = []
        for _ in range(size):
            v = f"{node_prefix}{next_id}"
            next_id += 1
            G.add_node(v)
            layer.append(v)
        layers.append(layer)

    # First layer points into X
    for v in layers[0]:
        G.add_edge(v, X)

    # Banded connections between layers, directed toward X
    for i in range(len(layers) - 1):
        left = layers[i]
        right = layers[i + 1]

        for j, u in enumerate(left):
            center = int(j * len(right) / max(1, len(left)))

            for d in range(-band_width, band_width + 1):
                idx = center + d
                if 0 <= idx < len(right):
                    G.add_edge(right[idx], u)

    # Last layer points into Y
    for v in layers[-1]:
        G.add_edge(v, Y)

    assert nx.is_directed_acyclic_graph(G)

    return G, X, Y, layers


def draw_handdrawn_style_dag(G, X, Y, layers, save_path="handdrawn_style_dag.png", show=True):
    pos = {}

    pos[X] = (-1, 0)

    for i, layer in enumerate(layers):
        x = i
        offset = (len(layer) - 1) / 2
        for j, node in enumerate(layer):
            pos[node] = (x, offset - j)

    pos[Y] = (len(layers), 0)

    plt.figure(figsize=(14, 7))

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        alpha=0.45,
        width=1.2,
        connectionstyle="arc3,rad=0.08",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=420,
        node_color="white",
        edgecolors="black",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def draw_all_adjustment_sets(G, X, Y, layers, zsets, out_dir="debug_adjustment_sets"):
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, Z in enumerate(zsets, start=1):
        draw_adjustment_set_on_dag(
            G,
            X,
            Y,
            layers,
            set(Z),
            out_dir / f"adjustment_set_{i:03d}_size_{len(Z)}.png",
        )

    print(f"saved drawings to: {out_dir}")

# Example
if __name__ == "__main__":
    G, X, Y, layers = make_handdrawn_style_dag(
        layer_sizes=(3,6,9,12,16), #(5, 6, 7, 8),#( 5, 7, 9, 11, 13),
        band_width=2,
        seed=1,
    )

    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())
    print("X:", X)
    print("Y:", Y)
    print("layer sizes:", [len(layer) for layer in layers])

    draw_handdrawn_style_dag(
        G,
        X,
        Y,
        layers,
        save_path="handdrawn_style_dag.png",
        show=True,
    )

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

    print("X:", X)
    print("Y:", Y)

    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())
    print("num adjustment sets:", len(zsets))
    #print("first 10:")
    for z in zsets:
        print(z)