import json
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Tuple, List

from graph.generators import spanning_tree_then_orient, generate_random_dag, layered_dag


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
    G1 = layered_dag(n=n, prob_edge=edge_prob, seed=seed_graph)
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
