import networkx as nx
import utils as utils

from i_o.utils import append_line

from bn.cpt import generate_bn_binary_logistic
from graph.separators import decode_separators, run_enumerator
from graph.transforms import relabel_to_ints
from pipelines.adjust_sets import find_adjustment_sets_for_pair,find_important_seps_for_pair

from causal.influence.estimator_bn import asymptotic_variance_for_Z
from causal.policies import static_do_policy

from graph.hankel_optimal_set import optimal_adjustment_set_O

from analysis.adjustment_hasse import frozenset_to_str

def make_funnel_dag(
    layer_sizes=(3, 5, 7, 10, 14),
    node_prefix="V",
    x_name="X",
    y_name="Y",
    connect_mode="complete",  # "complete" or "matching_plus"
):
    """
    Creates a funnel-shaped DAG:

        X -> small layer -> ... -> large layer -> Y

    layer_sizes:
        sizes from X side to Y side.
        Example: (3, 5, 7, 10, 14)

    connect_mode:
        "complete"      = every node in layer i connects to every node in layer i+1
        "matching_plus" = sparser version, still connected

    Returns:
        G, X, Y, layers
    """

    G = nx.DiGraph()
    X = x_name
    Y = y_name

    G.add_nodes_from([X, Y])

    layers = []
    next_id = 0

    for i, size in enumerate(layer_sizes):
        layer = []
        for _ in range(size):
            v = f"{node_prefix}{next_id}"
            next_id += 1
            G.add_node(v)
            layer.append(v)
        layers.append(layer)

    # X -> first/smallest layer
    for v in layers[0]:
        G.add_edge(X, v)

    # layer i -> layer i+1
    for i in range(len(layers) - 1):
        left = layers[i]
        right = layers[i + 1]

        if connect_mode == "complete":
            for u in left:
                for v in right:
                    G.add_edge(u, v)

        elif connect_mode == "matching_plus":
            for j, u in enumerate(left):
                G.add_edge(u, right[j % len(right)])
                G.add_edge(u, right[(j + 1) % len(right)])

            for j, v in enumerate(right):
                G.add_edge(left[j % len(left)], v)

        else:
            raise ValueError("connect_mode must be 'complete' or 'matching_plus'")

    # largest layer -> Y
    for u in layers[-1]:
        G.add_edge(u, Y)

    assert nx.is_directed_acyclic_graph(G)

    return G, X, Y, layers



def make_funnel_dag_with_adjustment_sets(
    layer_sizes=(3, 5, 7, 10, 14),
    node_prefix="V",
    x_name="X",
    y_name="Y",
    connect_mode="complete",  # "complete" or "matching_plus"
):
    """
    Creates a funnel-shaped DAG with many possible adjustment sets.

    Structure:
        X <- L0 -> L1 -> L2 -> ... -> Lk -> Y

    So L0, and nodes downstream of it, create backdoor paths
    between X and Y.
    """

    G = nx.DiGraph()
    X = x_name
    Y = y_name

    G.add_nodes_from([X, Y])

    layers = []
    next_id = 0

    for size in layer_sizes:
        layer = []
        for _ in range(size):
            v = f"{node_prefix}{next_id}"
            next_id += 1
            G.add_node(v)
            layer.append(v)
        layers.append(layer)

    # First layer causes X: L0 -> X
    for v in layers[0]:
        G.add_edge(v, X)

    # Layers flow toward Y: L0 -> L1 -> ... -> Lk
    for i in range(len(layers) - 1):
        left = layers[i]
        right = layers[i + 1]

        if connect_mode == "complete":
            for u in left:
                for v in right:
                    G.add_edge(u, v)

        elif connect_mode == "matching_plus":
            for j, u in enumerate(left):
                G.add_edge(u, right[j % len(right)])
                G.add_edge(u, right[(j + 1) % len(right)])

            for j, v in enumerate(right):
                G.add_edge(left[j % len(left)], v)

        else:
            raise ValueError("connect_mode must be 'complete' or 'matching_plus'")

    # Last/largest layer causes Y: Lk -> Y
    for u in layers[-1]:
        G.add_edge(u, Y)

    assert nx.is_directed_acyclic_graph(G)

    return G, X, Y, layers


def make_funnel_graph_undirected(
    layer_sizes=(3, 5, 7, 10, 14),
    node_prefix="V",
    x_name="X",
    y_name="Y",
    connect_mode="complete",  # "complete" or "matching_plus"
):
    """
    Creates an undirected funnel-shaped graph:

        X -- small layer -- ... -- large layer -- Y

    layer_sizes are ordered from X side to Y side.
    """

    G = nx.Graph()
    X = x_name
    Y = y_name

    G.add_nodes_from([X, Y])

    layers = []
    next_id = 0

    for size in layer_sizes:
        layer = []
        for _ in range(size):
            v = f"{node_prefix}{next_id}"
            next_id += 1
            G.add_node(v)
            layer.append(v)
        layers.append(layer)

    # X connected to smallest layer
    for v in layers[0]:
        G.add_edge(X, v)

    # Connect consecutive layers
    for i in range(len(layers) - 1):
        left = layers[i]
        right = layers[i + 1]

        if connect_mode == "complete":
            for u in left:
                for v in right:
                    G.add_edge(u, v)

        elif connect_mode == "matching_plus":
            for j, u in enumerate(left):
                G.add_edge(u, right[j % len(right)])
                G.add_edge(u, right[(j + 1) % len(right)])

            for j, v in enumerate(right):
                G.add_edge(left[j % len(left)], v)

        else:
            raise ValueError("connect_mode must be 'complete' or 'matching_plus'")

    # Largest layer connected to Y
    for u in layers[-1]:
        G.add_edge(u, Y)

    return G, X, Y, layers

import matplotlib.pyplot as plt


def draw_funnel_graph(G, X, Y, layers, save_path=None, show=True):
    """
    Draws the funnel graph like the sketch:
    X on the left, Y on the right, layers vertically in between.
    """

    pos = {}

    # X on the left
    pos[X] = (-1, 0)

    # Layers in the middle
    for i, layer in enumerate(layers):
        x = i

        # center each layer vertically around y=0
        offset = (len(layer) - 1) / 2

        for j, node in enumerate(layer):
            y = offset - j
            pos[node] = (x, y)

    # Y on the right
    pos[Y] = (len(layers), 0)

    plt.figure(figsize=(12, 6))

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.45,
        width=1.2,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=500,
        edgecolors="black",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()




if __name__ == "__main__":


    G, X, Y, layers = make_funnel_dag_with_adjustment_sets(
        layer_sizes=(3, 6, 9, 12, 17),
        connect_mode="complete",
    )

    seed = 0

    seed_graph, seed_params = utils.split_seeds(seed)
    bn = generate_bn_binary_logistic(G, seed=seed_params)


    draw_funnel_graph(
        G,
        X,
        Y,
        layers,
        save_path="funnel_graph.png",
        show=True,
    )

    R = list(bn.g.nodes())
    I = []
    H, minimal_Z_sets,total_time = find_adjustment_sets_for_pair(bn.g, X, Y, "smallminimalseps", R=R, I=I)

    important_seps = find_important_seps_for_pair(bn.g, X, Y, "smallminimalseps", R=R, I=I)

    o_x_y_seperator = optimal_adjustment_set_O(G, X, Y)

    L_vars = []  # נניח שהמדיניות תלויה ב-G,H (אפשר גם L_vars=[])
    # מדיניות סטטית do(I=1)
    policy_fn = static_do_policy(a_star=1)

    L_vars = []
    # מדיניות סטטית do(I=1)
    policy_fn = static_do_policy(a_star=1)

    seperators_file = f"outputs_funnel/bn.csv"
    append_line(seperators_file,
                "seed,X, Y, separator, len_sep, variance, type\n")

    for Z in important_seps:
        if len(Z) >= 1:
            Z_key = tuple(sorted(Z))
            sigma2_nm = asymptotic_variance_for_Z(
                bn, Y, X, Z, L_vars, policy_fn, None
            )
            append_line(seperators_file,
                        str(seed) + "," +
                        str(X) + "," +
                        str(Y) + "," +
                        frozenset_to_str(Z_key) + "," +
                        str(len(Z_key)) + "," +
                        str(round(sigma2_nm, 5)) + "," +
                        "important"
                        )

    for Z in minimal_Z_sets:
        if len(Z) >= 1:
            Z_key = tuple(sorted(Z))

            if Z_key in important_seps:
                continue

            sigma2_m = asymptotic_variance_for_Z(
                bn, Y, X, Z, L_vars, policy_fn, None
            )

            lable = "minimal"

            append_line(seperators_file,
                        str(seed) + "," +
                        str(X) + "," +
                        str(Y) + "," +
                        frozenset_to_str(Z_key) + "," +
                        str(len(Z_key)) + "," +
                        str(round(sigma2_m, 5)) + "," +
                        lable
                        )



    Z_key = tuple(sorted(o_x_y_seperator))
    sigma2_henkel = asymptotic_variance_for_Z(
        bn, Y, X, o_x_y_seperator, L_vars, policy_fn, None
    )
    append_line(seperators_file,
                str(seed) + "," +
                str(X) + "," +
                str(Y) + "," +
                frozenset_to_str(Z_key) + "," +
                str(len(Z_key)) + "," +
                str(round(sigma2_henkel, 5)) + "," +
                "henkel"
                )

