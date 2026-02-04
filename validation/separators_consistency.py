import networkx as nx
from networkx.drawing.nx_pydot import to_pydot


from typing import List, Dict, Any, FrozenSet, Iterable


def _to_frozenset_list(seps: List[List[str]]) -> List[FrozenSet[str]]:
    """
    פונקציית עזר: ממירה רשימה של רשימות לרשימה של frozenset,
    כדי שנוכל להשוות כמחזורים (ללא תלות בסדר).
    """
    return [frozenset(s) for s in seps]


def check_minseps_in_allseps(all_seps: List[List[str]],
                             min_seps: List[List[str]]) -> Dict[str, Any]:
    """
    בדיקה 1:
    האם כל מפריד ב-MinSeps קיים גם ב-AllSeps (כקבוצה, לא כסדר)?
    """
    all_sets = set(_to_frozenset_list(all_seps))
    min_sets = _to_frozenset_list(min_seps)

    missing = [
        sorted(list(m))
        for m in min_sets
        if m not in all_sets
    ]

    return {
        "ok": len(missing) == 0,
        "missing_minseps_not_in_allseps": missing,
    }


def find_non_minimal_minseps(all_seps: List[List[str]],
                             min_seps: List[List[str]]) -> Dict[str, Any]:
    """
    בדיקה 2:
    עבור כל מפריד ב-MinSeps, בודקת אם יש מפריד אחר ב-AllSeps
    שהוא תת-קבוצה ממש שלו (כלומר 'מוכל בו').
    אם יש כזה – המפריד ב-MinSeps אינו מינימלי.
    """
    all_sets = _to_frozenset_list(all_seps)
    min_sets = _to_frozenset_list(min_seps)

    non_minimal = {}

    for m in min_sets:
        smaller_subseps = [a for a in all_sets if a < m]  # a < m = תת-קבוצה ממש
        if smaller_subseps:
            key = tuple(sorted(m))
            non_minimal[key] = [sorted(list(a)) for a in smaller_subseps]

    return {
        "ok": len(non_minimal) == 0,
        "non_minimal_minseps": non_minimal,
    }


def find_allseps_contained_in_minseps(all_seps: List[List[str]],
                                      min_seps: List[List[str]]) -> Dict[str, Any]:
    """
    בדיקה 3:
    עבור כל מפריד ב-AllSeps, בודקת שאין מפריד ב-MinSeps
    שהוא על-קבוצה ממש שלו (כלומר מכיל אותו).
    אם יש כזה – זה מצביע על בעייתיות במינימליות הרשומה ב-MinSeps.
    """
    all_sets = _to_frozenset_list(all_seps)
    min_sets = _to_frozenset_list(min_seps)

    violations = {}

    for a in all_sets:
        containing_minseps = [m for m in min_sets if a < m]  # a < m = a מוכל ב-m
        if containing_minseps:
            key = tuple(sorted(a))
            violations[key] = [sorted(list(m)) for m in containing_minseps]

    return {
        "ok": len(violations) == 0,
        "allseps_contained_in_minseps": violations,
    }


def check_separators_consistency(all_seps: List[List[str]],
                                 min_seps: List[List[str]]) -> Dict[str, Any]:
    """
    פונקציה עוטפת שמריצה את כל שלוש הבדיקות ומחזירה אובייקט תוצאות אחד.
    """
    res1 = check_minseps_in_allseps(all_seps, min_seps)
    res2 = find_non_minimal_minseps(all_seps, min_seps)
    res3 = find_allseps_contained_in_minseps(all_seps, min_seps)

    return {
        "minseps_subset_of_allseps": res1,
        "minseps_really_minimal_vs_allseps": res2,
        "allseps_not_strictly_contained_in_minseps": res3,
    }

def contains_set(raw_sets: List[Iterable[str]], Z: Iterable[str]) -> bool:
    """בודק האם Z נמצאת ב-raw_sets (התאמה לפי קבוצה, בלי תלות בסדר/כפילויות)."""
    universe = {frozenset(s) for s in (set(r) for r in raw_sets)}
    return frozenset(set(Z)) in universe


# if check_not_sep = 1 then the function will return a list of sets that are not separators
# Otherwize it will return a list of sets that are separators but not minimal
def find_non_minimal_st_sep(g, separators, check_not_sep = 1):
    #result = {'NOT_MIN': [], 'NOT_SEP': []}
    result = []
    result_not_sep = []
    nonmin_index = []
    n = 0
    for sep in separators:
        h1 = g.copy()
        h1.remove_nodes_from(sep)
        if check_not_sep == 1:
            if nx.has_path(h1, g.graph['st'][0], g.graph['st'][1]):
                 result_not_sep.append(sep)  # not a separator
        for v in sep:
            h2 = g.copy()
            h2.remove_nodes_from(sep-{v})
            if not nx.has_path(h2, g.graph['st'][0], g.graph['st'][1]):
                #result['NOT_MIN'].append(sep) # not minimal (sep-v is a separator)
                result.append(sep) # not minimal
                nonmin_index.append(n)
                break
        n += 1
    return [result, result_not_sep, nonmin_index]
