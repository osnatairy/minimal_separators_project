import random
from typing import Tuple, Dict, List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 2) יצירת DAG רנדומלי בגודל n
# -----------------------------

def generate_random_dag(
    n: int,
    edge_prob: float = 0.2,
    node_prefix: str = "V",
    seed: int = 1
) -> nx.DiGraph:
    """
    יוצר DAG ע"י בחירת סדר טופולוגי קבוע מראש (V0..V(n-1))
    והוספת קשתות רק קדימה (i -> j כאשר i<j) בהסתברות edge_prob.

    יתרון: מובטח DAG (אין מחזורים).
    """
    if n < 2:
        raise ValueError("n חייב להיות לפחות 2")

    rng = np.random.default_rng(seed)

    nodes = [f"{node_prefix}{i}" for i in range(n)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # מוסיפים קשת i->j רק אם i<j
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                G.add_edge(nodes[i], nodes[j])

    # אופציונלי: לוודא שהגרף לא "ריק מדי"
    # אם אין שום קשתות, נכריח קשת אחת כדי שיהיה מה לעשות
    if G.number_of_edges() == 0:
        G.add_edge(nodes[0], nodes[-1])

    return G

def layered_dag(n: int,
                prob_edge: float,
                num_layers: int = 3,
                node_prefix: str = "V",
                seed: int | None = None,
                force_nonempty_layer0: bool = True
                ) -> Tuple[nx.DiGraph, Dict[int, int]]:
    """
    שיטה 1 — יצירת DAG על ידי שכבות.
    מחזירה (DiGraph, node_layer_map).
    ההגיון:
      - מקצים שכבה לכל צומת.
      - בונים 'backbone' שמחבר שכבות סמוכות.
      - מוודאים שלכל צומת בשכבה>0 יש לפחות הורה.
      - מוסיפים קשתות נוספות בהסתברות prob_edge רק מ-layer נמוך ל-high.
    """
    if seed is not None:
        random.seed(seed)

    if num_layers < 2:
        raise ValueError("num_layers must be at least 2")

    G = nx.DiGraph()
    nodes = [f"{node_prefix}{i}" for i in range(n)]#list(range(n))
    G.add_nodes_from(nodes)

    # הקצאת שכבות
    layers: Dict[int, List[int]] = {k: [] for k in range(num_layers)}
    node_layer: Dict[int, int] = {}
    for v in nodes:
        L = random.randrange(num_layers)
        node_layer[v] = L
        layers[L].append(v)

    if force_nonempty_layer0 and len(layers[0]) == 0:
        pick = random.choice(nodes)
        old = node_layer[pick]
        layers[old].remove(pick)
        node_layer[pick] = 0
        layers[0].append(pick)

    # backbone: חיבור בין שכבות סמוכות לא ריקות
    prev_nonempty = None
    for k in range(num_layers):
        if len(layers[k]) == 0:
            continue
        if prev_nonempty is None:
            prev_nonempty = k
            continue
        u = random.choice(layers[prev_nonempty])
        v = random.choice(layers[k])
        G.add_edge(u, v)
        prev_nonempty = k

    # ודא שלכל צומת בשכבה>0 יש לפחות הורה
    for v in nodes:
        k = node_layer[v]
        if k == 0:
            continue
        if G.in_degree(v) == 0:
            earlier_nodes = [u for l in range(0, k) for u in layers[l]]
            if earlier_nodes:
                u = random.choice(earlier_nodes)
                G.add_edge(u, v)
            else:
                # fallback לחיבור מ-layer 0
                u = random.choice(layers[0])
                G.add_edge(u, v)

    # הוספת קשתות נוספות (רק מ-layer נמוך ל-high)
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if node_layer[u] < node_layer[v] and not G.has_edge(u, v):
                if random.random() < prob_edge:
                    G.add_edge(u, v)

    # וידוא: אם נוצר somehow לא weakly-connected — נסה לחבר רכיבים
    if not nx.is_weakly_connected(G):
        comps = list(nx.weakly_connected_components(G))
        for i in range(1, len(comps)):
            prev_comp = list(comps[i-1])
            cur_comp = list(comps[i])
            connected = False
            for u in prev_comp:
                for v in cur_comp:
                    if node_layer[u] < node_layer[v]:
                        G.add_edge(u, v)
                        connected = True
                        break
                if connected:
                    break
            if not connected:
                # קו ברירת מחדל — חיבור אקראי
                u = random.choice(prev_comp)
                v = random.choice(cur_comp)
                # נבטיח שלא ניצור מעגל על ידי בדיקה (אבל בהתחשבות בשכבות זה לא אמור לקרות)
                if not nx.has_path(G, v, u):
                    G.add_edge(u, v)
                else:
                    G.add_edge(v, u)

    # בסיום: הגרף אמור להיות DAG ו-weakly connected
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Layered method produced a cycle — יש לבדוק לוגיקה")
    return G#, node_layer


def spanning_tree_then_orient(n: int,
                              prob_edge: float,
                              k_roots: int | None = None,
                              seed: int | None = None,
                              node_prefix: str = "V",
                              ) -> Tuple[nx.DiGraph, List[int]]:
    """
    שיטה 2 — בונים קודם עץ פורש (שרשור קשור אקראי), מוסיפים קשתות בלתי-מכוונות,
    ואז מכוונים את כל הקשתות לפי סדר (permutation) כדי לקבל DAG.
    מחזירה (DiGraph, order_list) — order_list הוא הסדר (topological order used).
    """
    if seed is not None:
        random.seed(seed)

    nodes =  [f"{node_prefix}{i}" for i in range(n)]#list(range(n))

    # יצירת עץ פורש פשוט: שרשור לפי permutation אקראית
    perm = nodes[:]
    random.shuffle(perm)
    undirected_edges = set()
    for i in range(1, n):
        a, b = perm[i-1], perm[i]
        undirected_edges.add(tuple(sorted((a, b))))

    # הוספת קשתות נוספות (בלתי-מכוונות) לפי prob_edge
    for i in range(n):
        for j in range(i+1, n):
            u = nodes[i]
            v = nodes[j]
            e = tuple(sorted((u, v)))
            if e in undirected_edges:
                continue
            if random.random() < prob_edge:
                undirected_edges.add(e)
            # if (u, v) in undirected_edges:
            #     continue
            # if random.random() < prob_edge:
            #     undirected_edges.add((u, v))

    # בחר סדר לכיוון (נשתמש ב-permutation אקראית)
    order = nodes[:]
    random.shuffle(order)

    # אם רוצים k_roots — בוחרים k צמתים שיהיו ראשונים בסדר
    if k_roots is not None and k_roots > 0:
        if k_roots > n:
            raise ValueError("k_roots cannot exceed n")
        roots = random.sample(nodes, k_roots)
        remaining = [x for x in order if x not in roots]
        order = roots + remaining

    index = {node: i for i, node in enumerate(order)}
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # הכוון כל קשת לפי הסדר: ממקום נמוך למקום גבוה
    for (u, v) in undirected_edges:
        if index[u] < index[v]:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)

    # בדיקות בסיסיות
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Orientation produced a cycle — יש לבדוק לוגיקה")
    if not nx.is_weakly_connected(G):
        raise RuntimeError("Resulting graph is not weakly connected — יש לבדוק")
    return G#, order


# פונקציות בדיקה מבוקשות (עטיפה ל-networkx)
def is_directed_acyclic_graph(G: nx.DiGraph) -> bool:
    """האם G הוא DAG?"""
    return nx.is_directed_acyclic_graph(G)


def is_weakly_connected(G: nx.DiGraph) -> bool:
    """האם G הוא weakly connected (אם מתעלמים מהכיוונים הוא connected)?"""
    return nx.is_weakly_connected(G)


# פונקציה עזר לציור
def draw_graph(G: nx.DiGraph, title: str = "DAG", figsize=(6, 4)):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos=pos, with_labels=True, arrows=True)
    plt.title(title)
    plt.show()


# דוגמה לשימוש
if __name__ == "__main__":
    # פרמטרים לדוגמה
    n = 30
    prob = 0.05
    num_layers = 5
    seed = 42
    k_roots = 3

    # שיטה 1
    G1, layers = layered_dag(n=n, prob_edge=prob, num_layers=num_layers, seed=seed)
    print("Method 1 - layered_dag:")
    print("Nodes:", G1.number_of_nodes(), "Edges:", G1.number_of_edges())
    print("Is DAG?", is_directed_acyclic_graph(G1))
    print("Is weakly connected?", is_weakly_connected(G1))
    draw_graph(G1, title="Layered DAG (method 1)")

    # שיטה 2
    G2, order = spanning_tree_then_orient(n=n, prob_edge=prob, k_roots=k_roots, seed=seed)
    print("\nMethod 2 - spanning_tree_then_orient:")
    print("Nodes:", G2.number_of_nodes(), "Edges:", G2.number_of_edges())
    print("Is DAG?", is_directed_acyclic_graph(G2))
    print("Is weakly connected?", is_weakly_connected(G2))
    draw_graph(G2, title="Spanning Tree + Orient (method 2)")
