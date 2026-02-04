


import networkx as nx
from typing import Iterable, Hashable, Set, Dict, Tuple, List, FrozenSet
from collections import defaultdict

Node = Hashable


from typing import Iterable, List, Tuple, Dict, Optional
def _normalize_family(family: Iterable[Iterable]) -> List[frozenset]:
    uniq = set()
    norm = []
    for s in family:
        fs = frozenset(s)
        if fs not in uniq:
            uniq.add(fs)
            norm.append(fs)
    return norm

def build_hasse_and_edges(family: Iterable[Iterable]):
    nodes = _normalize_family(family)
    nodes_sorted = sorted(nodes, key=lambda s: (len(s), sorted(map(str, s))))
    n = len(nodes_sorted)
    subset_edges = []
    for i in range(n):
        A = nodes_sorted[i]
        for j in range(i+1, n):
            B = nodes_sorted[j]
            if len(A) >= len(B):
                continue
            if A.issubset(B):
                subset_edges.append((A,B))
    hasse_edges = []
    by_size = {}
    for s in nodes_sorted:
        by_size.setdefault(len(s), []).append(s)
    for A,B in subset_edges:
        is_cover = True
        for k in range(len(A)+1, len(B)):
            for C in by_size.get(k, []):
                if A.issubset(C) and C.issubset(B):
                    is_cover = False
                    break
            if not is_cover:
                break
        if is_cover:
            hasse_edges.append((A,B))
    return nodes_sorted, hasse_edges


def s1_precedes_s2_by_y(Cy1: Iterable[Node], Cy2: Iterable[Node]) -> bool:
    """Return True iff C_y(H - S1) ⊆ C_y(H - S2)."""
    return set(Cy1).issubset(set(Cy2))

def cy_component(H: nx.Graph, Y: Hashable, Z: Iterable[Hashable]) -> Set[Hashable]:
    """
    רכיב הקשירות של Y לאחר הסרת Z: מחזיר את C_Y(H − Z) כסט צמתים.
    """
    H_und = H.to_undirected(as_view=False) if H.is_directed() else H.copy()
    Z = set(Z)
    H_und.remove_nodes_from([n for n in Z if n in H_und])
    if Y not in H_und:
        return set()
    return set(nx.node_connected_component(H_und, Y))

# מניחים שיש לך כבר:
# - build_hasse_and_edges(family)
# - cy_components_for_sets(...) שמחזירה forward_map, reverse_map

def hasse_from_cy_results(
    forward_map: Dict[FrozenSet, Set],
    reverse_map: Dict[FrozenSet, List[FrozenSet]] | None = None,
):
    """
    בונה דיאגרמת אסה על משפחת הרכיבים C_Y(H−Z) שנמצאו ע"י cy_components_for_sets.

    קלט:
      forward_map : dict[frozenset(Z)] -> set(component_nodes)
          מיפוי מכל Z לרכיב שהתקבל עבורו: C_Y(H−Z).
      reverse_map : אופציונלי dict[frozenset(component_nodes)] -> list[frozenset(Z)]
          אם כבר זמינה (מ-cy_components_for_sets), נשתמש בה כדי לקבץ Z-ים לכל רכיב.

    פלט:
      {
        "component_nodes": List[frozenset],             # רשימת רכיבי CY ייחודיים (ממוינים)
        "hasse_edges": List[Tuple[frozenset,frozenset]],# קשתות אסה בין רכיבים (כסטים)
        "edges_idx": List[Tuple[int,int]],              # קשתות לפי אינדקסים ברשימה שלמעלה
        "component_to_Zs": Dict[frozenset, List[frozenset]],  # איזה Z נתנו כל רכיב
        "Z_to_component": Dict[frozenset, frozenset],   # הרכיב של כל Z (נוח לשאילתות)
      }
    """
    # 1) מבטיחים ייצוג עקבי של הרכיבים כ-frozenset (כדי שישמשו כמפתחות)
    Z_to_component: Dict[FrozenSet, FrozenSet] = {
        frozenset(Z): frozenset(comp) for Z, comp in forward_map.items()
    }

    # 2) מפיקים רשימת רכיבים ייחודיים (בסדר הופעה יציב)
    unique_components: List[FrozenSet] = list(dict.fromkeys(Z_to_component.values()))

    # 3) בונים אסה על משפחת הרכיבים
    component_nodes, hasse_edges = build_hasse_and_edges(unique_components)

    # 4) קיבוץ Z-ים לכל רכיב (אם reverse_map לא ניתן – נגזור מ-forward_map)
    if reverse_map is not None:
        comp_to_Zs = {
            frozenset(comp): [frozenset(z) for z in Zs]
            for comp, Zs in reverse_map.items()
        }
    else:
        comp_to_Zs = defaultdict(list)
        for Z, comp in Z_to_component.items():
            comp_to_Zs[comp].append(Z)
        comp_to_Zs = dict(comp_to_Zs)

    # 5) קשתות גם לפי אינדקסים (נוח לציור)
    idx_of = {c: i for i, c in enumerate(component_nodes)}
    edges_idx = [(idx_of[A], idx_of[B]) for (A, B) in hasse_edges]

    return {
        "component_nodes": component_nodes,
        "hasse_edges": hasse_edges,
        "edges_idx": edges_idx,
        "component_to_Zs": comp_to_Zs,
        "Z_to_component": Z_to_component,
    }

def cy_components_for_sets(
    H: nx.Graph,
    Y: Hashable,
    sets_Z: Iterable[Iterable[Hashable]],
    *,
    normalize_keys: bool = True,
) -> Tuple[
    Dict[FrozenSet[Hashable], Set[Hashable]],
    Dict[FrozenSet[Hashable], List[FrozenSet[Hashable]]]
]:
    """
    לחשב C_Y(H − Z) עבור רשימת קבוצות Z שונות, ולהחזיר:
      1) forward_map:  Z -> C_Y(H−Z)
      2) reverse_map:  קומפוננטה (כ-frozenset) -> [רשימת Z-ים שהביאו אליה]

    פרמטרים
    --------
    H : nx.Graph / nx.DiGraph
        הגרף הלא-מכוון H (או מכוון — יומר אוטומטית לבלתי-מכוון).
    Y : Hashable
        צומת ה-outcome.
    sets_Z : Iterable[Iterable[Hashable]]
        כל קבוצות ה-Z (מועמדות להתאמה/הסרה).
    normalize_keys : bool
        אם True, מפתחי המפה יהיו frozenset(Z) כדי שיהיו ברי-השוואה וברי-שימוש כמפתחות.

    מחזיר
    -----
    forward_map : dict[frozenset(Z)] -> set(component_nodes)
    reverse_map : dict[frozenset(component_nodes)] -> list[frozenset(Z)]
    """
    # נוודא שנעבוד תמיד על H לא-מכוון (לפי ההגדרה מחשבים רכיבי קשירות בגרף לא-מכוון)
    H_und = H.to_undirected(as_view=False) if H.is_directed() else H

    forward_map: Dict[FrozenSet[Hashable], Set[Hashable]] = {}
    buckets: Dict[FrozenSet[Hashable], List[FrozenSet[Hashable]]] = defaultdict(list)

    for Z in sets_Z:
        Z_key = frozenset(Z) if normalize_keys else frozenset(Z)  # שומרים אחיד
        comp = cy_component(H_und, Y, Z_key)  # סט צמתים של C_Y(H−Z)
        forward_map[Z_key] = comp
        comp_key = frozenset(comp)            # קומפוננטה כ-frozenset לצורך קיבוץ
        buckets[comp_key].append(Z_key)

    reverse_map: Dict[FrozenSet[Hashable], List[FrozenSet[Hashable]]] = dict(buckets)
    return forward_map, reverse_map



#find if there are HASS pair in the adjustment set component
def find_containment_pairs(res):

    if len(res["hasse_edges"]) > 0:
        return True
    return False




def extract_separator_containment_pairs(res):
    comp_to_Zs = res["component_to_Zs"]   # component -> [Z,...]
    edges = res["hasse_edges"]            # [(comp_small, comp_big), ...]

    pairs = []
    for comp_small, comp_big in edges:
        Zs_small = comp_to_Zs.get(comp_small, [])
        Zs_big = comp_to_Zs.get(comp_big, [])

        # comp_small ⊆ comp_big  ==>  comp_big "מכיל" את comp_small
        # לכן: מפריד של comp_big = "המכיל", מפריד של comp_small = "המוכל"
        for z_big in Zs_big:
            for z_small in Zs_small:
                pairs.append({
                    "outer_sep": z_big,
                    "outer_component": comp_big,
                    "inner_sep": z_small,
                    "inner_component": comp_small,
                })
    return pairs


def frozenset_to_str(fs):
    return ";".join(sorted(fs))


##### EXAMPLE OF USE #####

'''
G = nx.Graph([("Y","A"), ("A","B"), ("B","X")])
Cy1 = component_y_after_removal(G, {"A"}, "Y")
Cy2 = component_y_after_removal(G, {"B"}, "Y")
print(Cy1, Cy2)                           # {'Y'} , {'A','Y'}
print(s1_precedes_s2_by_y(Cy1, Cy2))      # True


set_Z = [
    {"a"}, {"a","b"}, {"a","c"}, {"a","b","c"},
    {"d"}, {"d","e"}
]
nodes, hasse_edges = build_hasse_and_edges(set_Z)

print("Hasse edges:", hasse_edges)
# [
#  ({a}->{a,b}), ({a}->{a,c}), ({d}->{d,e}),
#  ({a,b}->{a,b,c}), ({a,c}->{a,b,c})
# ]

# H: גרף לא-מכוון
H = nx.Graph()
H.add_edges_from([
    ("Y","a"), ("a","b"), ("b","c"),
    ("a","e"), ("c","d")
])

Y = "Y"
Z_candidates = [
    {"b"},          # מסיר את b
    {"c"},          # מסיר את c
    {"b","c"},      # מסיר את b וגם c
    set(),          # לא מסיר כלום
]

forward, reverse = cy_components_for_sets(H, Y, Z_candidates)
print(forward, reverse)
# forward[ frozenset({"b"}) ]  -> {"Y","a","e"}
# forward[ frozenset({"c"}) ]  -> {"Y","a","b","e"}
# forward[ frozenset({"b","c"})] -> {"Y","a","e"}
# forward[ frozenset()]         -> {"Y","a","b","c","d","e"}

# reverse מקבץ Z-ים שנתנו אותו רכיב:
#   frozenset({"Y","a","e"})        -> [ {"b"}, {"b","c"} ]
#   frozenset({"Y","a","b","e"})    -> [ {"c"} ]
#   frozenset({"Y","a","b","c","d","e"}) -> [ ∅ ]
'''


