from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
from networkx.algorithms.dag import lexicographical_topological_sort

from itertools import product
import random
import math
import networkx as nx
import pandas as pd

from bn.bayesian_network import BN

# ----------------------------
# 5) חישוב CPT
# ----------------------------

def compute_cpt(
    df_cat: pd.DataFrame,
    child: str,
    parents: List[str],
    domains: Dict[str, List[Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    מחזיר CPT בפורמט:
      [
        {"parents": [...], "prob": {"val": p, ...}},
        ...
      ]

    סדר parents בכל רשומה חייב להיות לפי parents list (זה ה-parent_order של הילד).
    """
    child_states = domains[child]
    parent_states = [domains[p] for p in parents]

    # מקרה ללא הורים: P(child)
    if not parents:
        counts = df_cat[child].value_counts().reindex(child_states, fill_value=0).astype(float)
        probs = (counts + alpha) / (counts.sum() + alpha * len(child_states))
        return [{
            "parents": [],
            "prob": {str(k): float(probs.loc[k]) for k in child_states}
        }]

    # נבנה טבלת ספירות: index=קומבינציות הורים, columns=מצבי הילד
    counts = (
        df_cat
        .groupby(parents + [child], observed=True)
        .size()
        .unstack(child, fill_value=0)
        .reindex(columns=child_states, fill_value=0)
        .astype(float)
    )

    # נוודא שכל קומבינציות ההורים קיימות גם אם לא נצפו בדאטה (כדי שיהיו CPT מלאים)
    full_parent_index = pd.MultiIndex.from_product(parent_states, names=parents)
    counts = counts.reindex(full_parent_index, fill_value=0)

    # smoothing + נרמול לכל שורת הורים
    probs = (counts + alpha).div(counts.sum(axis=1) + alpha * len(child_states), axis=0)

    # המרה לפורמט הרשומות שבדוגמה
    cpt_rows: List[Dict[str, Any]] = []
    for parent_vals, row in probs.iterrows():
        if not isinstance(parent_vals, tuple):
            parent_vals = (parent_vals,)

        cpt_rows.append({
            "parents": list(parent_vals),
            "prob": {str(k): float(row.loc[k]) for k in child_states}
        })

    return cpt_rows



def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _clip(p: float, eps: float = 1e-3) -> float:
    return max(eps, min(1.0 - eps, p))

def generate_bn_binary_logistic(
    G: nx.DiGraph,
    seed: int | None = None,
    weight_scale: float = 1.2,
    bias_scale: float = 0.7,
    allow_negative: bool = True,
    eps: float = 1e-3,
) -> BN:
    """
    מקבל DAG (nx.DiGraph) ומחזיר BN:
      - כל המשתנים בינאריים עם domain [0, 1].
      - כל ה-CPT נקבעים דטרמיניסטית כתלות ב-(G, seed).
    """

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a DAG")

    # RNG לוקאלי, לא נוגעים ב-random הגלובלי
    rng = random.Random(seed)

    # 1) בונים BN עם אותם צמתים וקשתות
    bn = BN()
    for node in G.nodes():
        # domain קבוע
        bn.add_var(node, [0, 1])

    for u, v in G.edges():
        bn.add_edge(u, v)

    # 2) סדר טופולוגי דטרמיניסטי (לקסיקוגרפי לפי שם)
    topo_order = list(lexicographical_topological_sort(G, key=lambda x: str(x)))

    # 3) עבור כל צומת, יוצרים CPT לוגיסטי דטרמיניסטי
    for node in topo_order:
        parents = sorted(list(G.predecessors(node)))  # סדר הורים קבוע
        bn.set_parent_order(node, parents)

        # פרמטרים לוגיסטיים מה-RNG הלוקאלי
        b = rng.gauss(0.0, bias_scale)  # bias
        weights = {}
        for p in parents:
            wi = abs(rng.gauss(0.0, weight_scale))
            if allow_negative and rng.random() < 0.25:
                wi = -wi
            weights[p] = wi

        dom_child = bn.domain(node)  # [0, 1]
        cpt = {}

        if not parents:
            # ללא הורים: שורה יחידה
            z = b
            p1 = _clip(_sigmoid(z), eps)
            cpt[()] = {dom_child[0]: 1.0 - p1,
                       dom_child[1]: p1}
        else:
            # לכל השמת הורים
            parent_domains = [bn.domain(p) for p in parents]  # כולם [0,1]
            for pa_vals in product(*parent_domains):
                z = b
                # מניחים שהערך השני בדומיין הוא "1"
                for p, val in zip(parents, pa_vals):
                    dom_p = bn.domain(p)   # [0,1]
                    is_one = 1 if val == dom_p[1] else 0
                    z += weights[p] * is_one

                p1 = _clip(_sigmoid(z), eps)
                cpt[pa_vals] = {dom_child[0]: 1.0 - p1,
                                dom_child[1]: p1}

        bn.set_cpt(node, cpt, strict=True)

    return bn