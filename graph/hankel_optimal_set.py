from __future__ import annotations
from typing import Any, Iterable, Set
import networkx as nx

from graph.helpers import normalize_nodes_to_set  # :contentReference[oaicite:5]{index=5}


def _parents_of_set(G: nx.DiGraph, S: Iterable[Any]) -> Set[Any]:
    pa = set()
    for v in S:
        pa |= set(G.predecessors(v))
    return pa


def causal_nodes_cn(G: nx.DiGraph, X, Y) -> Set[Any]:
    """
    cn(X,Y,G): כל הצמתים שנמצאים על מסלולים מכוונים X -> ... -> Y,
    ללא X, ובאופן שמכיל גם את Y (אם הוא צאצא אפשרי של X).
    עבור DAG: זה בדיוק anc(Y) ∩ desc(X), ואז מוסיפים Y, ומסירים X.
    """
    Xs = normalize_nodes_to_set(X)
    Ys = normalize_nodes_to_set(Y)

    # כל מה שנגיש מ-X
    desc_X = set()
    for x in Xs:
        desc_X |= nx.descendants(G, x)

    # כל מה שיכול להגיע ל-Y
    anc_Y = set()
    for y in Ys:
        anc_Y |= nx.ancestors(G, y)

    # צמתים על איזשהו מסלול מכוון X -> ... -> Y
    mid = desc_X & anc_Y

    # לכלול את Y עצמו אם הוא באמת צאצא של X (כלומר desc_X מכיל אותו)
    cn = set(mid)
    cn |= (Ys & desc_X)

    # cn לא כולל את X
    cn -= Xs
    return cn


def forbidden_set_henckel(G: nx.DiGraph, X, Y) -> Set[Any]:
    """
    forb(X,Y,G) לפי Henckel et al.:
    forb = de(cn) ∪ X, כאשר בפועל de כולל גם את הצומת עצמו (כפי שמשתמע מהדוגמאות).
    לכן: forb = cn ∪ descendants(cn) ∪ X
    """
    Xs = normalize_nodes_to_set(X)
    cn = causal_nodes_cn(G, X, Y)

    forb = set(Xs) | set(cn)
    for v in cn:
        forb |= nx.descendants(G, v)
    return forb


def optimal_adjustment_set_O(G: nx.DiGraph, X, Y) -> Set[Any]:
    """
    O(X,Y,G) = pa(cn(X,Y,G)) \\ forb(X,Y,G)  (Henckel et al., Theorem/Definition סביב Section 3.4)
    מחזיר סט צמתים.
    """
    cn = causal_nodes_cn(G, X, Y)
    pa_cn = _parents_of_set(G, cn)
    forb = forbidden_set_henckel(G, X, Y)

    O = pa_cn - forb
    return O
