# linear_Structural_Equation_Model.py
"""
מייצר מודל SEM ליניארי מכוון (Linear Gaussian SEM) בגודל n:
  - גרף מכוון acyclic (DAG) באמצעות networkx
  - מקדמי קשתות beta
  - שונויות רעש sigma2 לכל צומת
ושומר הכל לקובץ JSON כדי שניתן לשחזר.

הקוד מיועד לשימוש עם פייפליין כמו שלך:
  G (nx.DiGraph) -> build_H1_from_DAG(...) -> moralize/saturate -> enumerate separators
השלב הזה תלוי רק במבנה הגרף, אבל אנחנו שומרים גם את הליניאריות להמשך חישובי שונות.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
from minimal_separators_project.graph.generators import spanning_tree_then_orient, generate_random_dag


# -----------------------------
# 1) מודל נתונים: גרף + פרמטרים ליניאריים
# -----------------------------

@dataclass
class LinearSEM:
    """
    מייצג SEM ליניארי על DAG:
      X_v = sum_{u in Pa(v)} beta[u->v] * X_u + eps_v
      eps_v ~ N(0, sigma2[v]) בלתי תלוי בין צמתים
    """
    G: nx.DiGraph
    beta: Dict[str, float]      # מפתח: "u->v"
    sigma2: Dict[str, float]    # מפתח: node


# -----------------------------
# 3) דגימת פרמטרים ליניאריים (beta ו-sigma2)
# -----------------------------

def sample_linear_parameters(
    G: nx.DiGraph,
    beta_scale: float = 1.0,
    sigma2_low: float = 0.2,
    sigma2_high: float = 1.0,
    seed: int = 1
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    דוגם:
      - לכל קשת u->v:  beta ~ Normal(0, beta_scale^2)
      - לכל צומת v:    sigma2[v] ~ Uniform(sigma2_low, sigma2_high)

    beta_scale שולט על "עוצמת" הקשרים.
    sigma2 שולט על "רעש עצמי" בכל צומת.
    """
    rng = np.random.default_rng(seed)

    beta: Dict[str, float] = {}
    for u, v in G.edges():
        beta[f"{u}->{v}"] = float(rng.normal(loc=0.0, scale=beta_scale))

    sigma2: Dict[str, float] = {}
    for v in G.nodes():
        sigma2[v] = float(rng.uniform(low=sigma2_low, high=sigma2_high))

    return beta, sigma2


# -----------------------------
# 4) שמירה/טעינה JSON כדי לשחזר בדיוק את אותו מודל
# -----------------------------

def save_sem_to_json(sem: LinearSEM, path: str) -> None:
    """
    שומר את ה-DAG והפרמטרים.
    פורמט פשוט:
      nodes, edges, beta, sigma2
    """
    payload = {
        "nodes": list(sem.G.nodes()),
        "edges": [[u, v] for (u, v) in sem.G.edges()],
        "beta": sem.beta,
        "sigma2": sem.sigma2,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_sem_from_json(path: str) -> LinearSEM:
    """
    טוען את ה-SEM מה-JSON ומחזיר nx.DiGraph + פרמטרים.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    G = nx.DiGraph()
    G.add_nodes_from(d["nodes"])
    G.add_edges_from([tuple(e) for e in d["edges"]])

    return LinearSEM(G=G, beta=d["beta"], sigma2=d["sigma2"])


# -----------------------------
# 5) (אופציונלי) בניית מטריצות B ו-Omega וחישוב Sigma
# -----------------------------

def sem_to_B_Omega(sem: LinearSEM) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    בונה מטריצת B ומטריצת Omega (אלכסונית) מתוך sem.
    הגדרה:
      B[child, parent] = beta[parent->child]
      Omega = diag(sigma2)

    שימי לב: זה שימושי אם תרצי אחרי זה לחשב קו-וריאנס Sigma או שונויות.
    """
    nodes = list(sem.G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)

    B = np.zeros((p, p), dtype=float)
    for u, v in sem.G.edges():
        key = f"{u}->{v}"
        B[idx[v], idx[u]] = float(sem.beta[key])

    Omega = np.diag([float(sem.sigma2[n]) for n in nodes])
    return nodes, B, Omega


def covariance_from_B_Omega(B: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """
    Sigma = (I - B)^(-1) * Omega * (I - B)^(-T)
    """
    I = np.eye(B.shape[0])
    M = np.linalg.inv(I - B)
    Sigma = M @ Omega @ M.T
    return Sigma


# -----------------------------
# 6) פונקציה אחת שמייצרת הכל מהר
# -----------------------------

def make_linear_sem(
    n: int,
    edge_prob: float = 0.2,
    beta_scale: float = 1.0,
    sigma2_low: float = 0.2,
    sigma2_high: float = 1.0,
    num_layers: int = 5,
    node_prefix: str = "V",
    seed_graph: int = 1,
    seed_params: int = 2

) -> LinearSEM:
    """
    מייצר:
      - DAG רנדומלי בגודל n
      - פרמטרים ליניאריים (beta, sigma2)
    """
    G1 = generate_random_dag(n=n, edge_prob=edge_prob, node_prefix=node_prefix, seed=seed_graph)
    k_roots = 3
    G = spanning_tree_then_orient(n=n,
                    prob_edge=edge_prob,
                    k_roots=k_roots,
                    node_prefix=node_prefix,
                    seed=seed_graph)

    beta, sigma2 = sample_linear_parameters(
        G=G,
        beta_scale=beta_scale,
        sigma2_low=sigma2_low,
        sigma2_high=sigma2_high,
        seed=seed_params
    )
    return LinearSEM(G=G, beta=beta, sigma2=sigma2)

# אופציונלי: חישוב Sigma לשונות/קו-וריאנס:
def calc_varience(sem2) :
    nodes, B, Omega = sem_to_B_Omega(sem2)
    Sigma = covariance_from_B_Omega(B, Omega)
    print("Sigma shape:", Sigma.shape)


# -----------------------------
# 7) דוגמת שימוש שמראה התאמה לפייפליין שלך
# -----------------------------

def example_usage():
    """
    הדגמה בסיסית:
      1) יצירת SEM בגודל n
      2) שמירה ל-JSON
      3) טעינה מחדש
      4) הפקת G (nx.DiGraph) כדי להעביר לקוד שלך למציאת מפרידים
      5) (אופציונלי) חישוב Sigma
    """
    n = 20
    sem = make_linear_sem(
        n=n,
        edge_prob=0.25,
        beta_scale=1.0,
        sigma2_low=0.2,
        sigma2_high=1.0,
        node_prefix="V",
        seed_graph=1,
        seed_params=2
    )

    # שמירה לקובץ
    out_json = "linear_sem_n10.json"
    save_sem_to_json(sem, out_json)
    print("Saved SEM to:", out_json)

    # טעינה מחדש (שחזור מלא)
    sem2 = load_sem_from_json(out_json)

    # זה הגרף שתעבירי לפונקציות שלך:
    G = sem2.G
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    # לדוגמה: בחירת X,Y
    X = [f"V{n-3}"]
    Y = [f"V{n-1}"]
    R = list(G.nodes())  # נניח שכולם observed
    I = []               # אם אין אילוץ

    print("Example X,Y:", X, Y)
    '''
    # כאן את מחברת לקוד שלך:
    H1 = build_H1_from_DAG(G, X=X, Y=Y, R=R, I=I)
    H_int, name_to_id, id_to_name, s, t = relabel_to_ints(H1, X[0], Y[0])
    seps_int = alg.start_algorithm(H_int)
    Z_sets = decode_separators(seps_int, id_to_name)

    # sem = make_linear_sem(...)
    # Z_sets = ...  # output of your separator enumerator (list of lists)

    scores = []
    for Z in Z_sets:
        aVar = lvcal.example_compute_avar(sem, X=X, Y=Y, Z=Z)
        scores.append((aVar, Z))
    
    scores.sort()
    best_aVar, best_Z = scores[0]
    print("Best Z:", best_Z, "best aVar:", best_aVar)
    '''

    # אופציונלי: חישוב Sigma לשונות/קו-וריאנס:
    nodes, B, Omega = sem_to_B_Omega(sem2)
    Sigma = covariance_from_B_Omega(B, Omega)
    print("Sigma shape:", Sigma.shape)


if __name__ == "__main__":
    example_usage()
