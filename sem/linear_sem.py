import json
import numpy as np
import networkx as nx
import utils as utils
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
    for u, v in sorted(G.edges(), key=lambda e: (str(e[0]), str(e[1]))):
        beta[f"{u}->{v}"] = float(rng.normal(loc=0.0, scale=beta_scale))

    sigma2: Dict[str, float] = {}
    for v in sorted(G.nodes(), key=str):
        sigma2[v] = float(rng.uniform(low=sigma2_low, high=sigma2_high))

    return beta, sigma2


def sample_linear_parameters_stabilized(
        G: nx.DiGraph,
        beta_scale: float = 0.5,  # מומלץ להוריד ל-0.5
        sigma2_low: float = 0.2,
        sigma2_high: float = 1.0,
        max_var_threshold: float = 5.0,  # הסף המקסימלי לשונות של צומת
        seed: int = 1
) -> Tuple[Dict[str, float], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    beta: Dict[str, float] = {}
    sigma2: Dict[str, float] = {}

    # נשמור כאן את השונות המוערכת של כל צומת בזמן אמת
    node_variances: Dict[str, float] = {}

    # מעבר לפי סדר טופולוגי מבטיח שאנחנו מטפלים בהורים לפני הילדים
    topo_order = list(nx.topological_sort(G))

    for v in topo_order:
        # 1. דגימת שונות עצמית (Noise)
        s2 = float(rng.uniform(low=sigma2_low, high=sigma2_high))
        sigma2[v] = s2

        # 2. חישוב השונות שמגיעה מההורים
        parents = list(G.predecessors(v))
        if not parents:
            # צומת שורש - השונות שלו היא רק הרעש העצמי
            node_variances[v] = s2
            continue

        # דוגמים בטאות ראשוניות
        current_betas = {u: float(rng.normal(loc=0.0, scale=beta_scale)) for u in parents}

        # חישוב השונות המצטברת (קירוב ללא קו-וריאנס למטרת בקרה)
        incoming_var = sum((current_betas[u] ** 2) * node_variances[u] for u in parents)

        # 3. מנגנון הריסון (Taming)
        # אם השונות המצטברת גדולה מדי, ננרמל את הבטאות שנכנסות לצומת
        total_potential_var = incoming_var + s2
        if total_potential_var > max_var_threshold:
            # אם השונות הנכנסת זניחה, אין צורך בנרמול (מונע חילוק ב-0)
            if incoming_var < 1e-10:
                shrink_factor = 1.0
            else:
                # הגנה: אם s2 לבדו גדול מהסף, המונה יהיה 0 והבטאות יתאפסו
                numerator = max(0.0, max_var_threshold - s2)
                shrink_factor = np.sqrt(numerator / incoming_var)

            # עדכון הבטאות עם מקדם הריסון
            for u in parents:
                current_betas[u] *= shrink_factor

            # חישוב מחדש של השונות הנכנסת לאחר התיקון
            incoming_var = sum((current_betas[u] ** 2) * node_variances[u] for u in parents)

        # if total_potential_var > max_var_threshold:
        #     # מקדם תיקון כדי להחזיר את השונות לסף המותר
        #     shrink_factor = np.sqrt((max_var_threshold - s2) / incoming_var)
        #     for u in parents:
        #         current_betas[u] *= shrink_factor
        #
        #     incoming_var = sum((current_betas[u] ** 2) * node_variances[u] for u in parents)

        # שמירת הבטאות הסופיות והשונות המעודכנת
        for u in parents:
            beta[f"{u}->{v}"] = current_betas[u]

        node_variances[v] = incoming_var + s2

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
    seed: int = 1

) -> LinearSEM:
    """
    מייצר:
      - DAG רנדומלי בגודל n
      - פרמטרים ליניאריים (beta, sigma2)
    """
    seed_graph, seed_params = utils.split_seeds(seed)

    G1 = layered_dag(n=n, prob_edge=edge_prob, seed=seed_graph)
    k_roots = 3
    G = spanning_tree_then_orient(n=n,
                    prob_edge=edge_prob,
                    k_roots=k_roots,
                    node_prefix=node_prefix,
                    seed=seed_graph)

    # beta, sigma2 = sample_linear_parameters(
    #     G=G,
    #     beta_scale=beta_scale,
    #     sigma2_low=sigma2_low,
    #     sigma2_high=sigma2_high,
    #     seed=seed_params
    # )

    beta, sigma2 = sample_linear_parameters_stabilized(
        G=G,
        beta_scale=beta_scale,
        sigma2_low=sigma2_low,
        sigma2_high=sigma2_high,
        max_var_threshold=0.5,
        seed=seed_params
    )


    return LinearSEM(G=G, beta=beta, sigma2=sigma2)


def remove_edge_from_sem(sem: LinearSEM, u: str, v: str, strict: bool = True) -> None:
    """
    מסירה את הקשת u->v גם מהגרף וגם ממילון beta.

    strict=False: לא זורקת שגיאה אם הקשת/המפתח לא קיימים.
    strict=True: זורקת שגיאה אם משהו חסר (כדי לגלות אי-עקביות).
    """
    key = f"{u}->{v}"

    # 1) להסיר מהגרף
    if sem.G.has_edge(u, v):
        sem.G.remove_edge(u, v)
    else:
        if strict:
            raise KeyError(f"Edge {u}->{v} not found in sem.G")

    # 2) להסיר מה-beta
    if key in sem.beta:
        sem.beta.pop(key)
    else:
        if strict:
            raise KeyError(f"Key '{key}' not found in sem.beta")
