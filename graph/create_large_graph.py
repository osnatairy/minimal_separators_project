import json
import random
from pathlib import Path
import networkx as nx


def make_deterministic_500_minimal_graph(node_prefix="V"):
    G = nx.DiGraph()

    X = f"{node_prefix}0"
    Y = f"{node_prefix}1"
    G.add_nodes_from([X, Y])

    path_lengths = [5, 5, 5, 4]  # 5*5*5*4 = 500
    next_id = 2

    for length in path_lengths:
        prev = X
        for _ in range(length):
            v = f"{node_prefix}{next_id}"
            next_id += 1
            G.add_node(v)
            G.add_edge(prev, v)
            prev = v
        G.add_edge(prev, Y)

    # Fill up to 70 nodes with isolated nodes
    while next_id < 70:
        G.add_node(f"{node_prefix}{next_id}")
        next_id += 1

    return G, X, Y


def save_linear_sem_json(path, G, X, Y, seed=1):
    rng = random.Random(seed)

    beta = {
        f"{u}->{v}": rng.uniform(-1.0, 1.0)
        for u, v in G.edges()
    }

    sigma2 = {
        node: rng.uniform(0.2, 1.0)
        for node in G.nodes()
    }

    payload = {
        "nodes": list(G.nodes()),
        "edges": [[u, v] for u, v in G.edges()],
        "beta": beta,
        "sigma2": sigma2,
        "len(pairs)": 1,
        "pairs": [[X, Y]],
        "Z_sets": 500,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


for seed in range(1, 6):
    G, X, Y = make_deterministic_500_minimal_graph()

    save_linear_sem_json(
        path=f"generated_many_sep_graphs/deterministic_500_minimal_seed_{seed}.json",
        G=G,
        X=X,
        Y=Y,
        seed=seed,
    )

    print(f"saved seed={seed}, X={X}, Y={Y}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")