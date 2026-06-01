import networkx as nx
from collections import deque
from collections.abc import Iterable

from graph.helpers import normalize_nodes_to_set



#find all the causal vertices in the graph
#those are the vertices with a path from x to Y and reverse.
#where the value of x and y is are specific nodes
def find_causal_vertices(G, x, y):
    descendants_x = set(nx.descendants(G, x)) | {x} # Anyone accessible from x and x
    ancestors_y = set(nx.ancestors(G, y)) | {y} # Anyone who can reach y and y

    # The intersection = vertices that are between x and y
    causal = descendants_x & ancestors_y

    return causal

# Finds causal vertices between node sets X and Y.
# Efficient when |X| and |Y| are relatively small.
# will take O(|X| + |Y|) call functions, may cause big temp groups.
def find_causal_vertices_sets_v1(G, X, Y):

    # Unites all descendants of all nodes in X
    all_descendants_X = set()
    for x in X:
        all_descendants_X.update(nx.descendants(G, x))

    # Unites all predecessors of all nodes in Y
    all_ancestors_Y = set()
    for y in Y:
        all_ancestors_Y.update(nx.ancestors(G, y))

    # Intersection = the nodes that are between X and Y
    causal = all_descendants_X & all_ancestors_Y

    return causal


def normalize_nodes(nodes):
    if isinstance(nodes, str):
        return {nodes}
    return set(nodes)

# Finds causal vertices between sets of nodes X and Y.
# More efficient for large graphs - uses one-time BFS search.
# O(|V| + |E|) for the BFS
def find_causal_vertices_sets_v2(G, X, Y):

    X = normalize_nodes(X)
    Y = normalize_nodes(Y)

    # Finding all nodes reachable from X
    reachable_from_X = (X)  # Including X itself
    queue = deque(X)

    while queue:
        current = queue.popleft()
        for successor in G.successors(current):
            if successor not in reachable_from_X:
                reachable_from_X.add(successor)
                queue.append(successor)

    # Finding all nodes that can reach Y
    can_reach_Y = (Y)  # Including Y itself
    queue = deque(Y)

    # Creating an inverse graph for backward search
    G_reversed = G.reverse()

    while queue:
        current = queue.popleft()
        for predecessor in G_reversed.successors(current):  # predecessors in the original graph
            if predecessor not in can_reach_Y:
                can_reach_Y.add(predecessor)
                queue.append(predecessor)

    # Intersection = the nodes between X and Y
    # Removing X and Y themselves from the result (as we only want the nodes in the middle)
    causal = (reachable_from_X & can_reach_Y) - set(X) - set(Y)

    return causal#, (reachable_from_X | can_reach_Y)


# Runs the appropriate function to find causal vertices according to the input type
# A hybrid approach that chooses the better method based on the size of the groups.
# If X and Y are single nodes, uses the original function.
# Args:
#         G: NetworkX DiGraph
#         X: A single node or a collection of nodes (set, list, tuple, or single string)
#         Y: A single node or a collection of nodes (set, list, tuple, or single string)
# Returns:
#         set: the set of causal nodes between X and Y
def find_causal_vertices_sets_optimized(G, X, Y):

    # Convert X to a set
    X = normalize_nodes_to_set(X)

    # Same for Y
    Y = normalize_nodes_to_set(Y)

    # Checking if X and Y are unique nodes
    if len(X) == 1 and len(Y) == 1:
        x = next(iter(X))  # Removes the single braid from X
        y = next(iter(Y))  # Removes the single braid from Y
        return find_causal_vertices(G, x, y)

    #TODO: the functions below do not include X and Y in the returned answer.
    # If the groups are small, use "find_causal_vertices_sets_v1"
    elif len(X) <= 5 and len(Y) <= 5:
        return find_causal_vertices_sets_v1(G, X, Y)
    else: # use find_causal_vertices_sets_v2
        return find_causal_vertices_sets_v2(G, X, Y)





def _normalize_node_set(nodes):
    """
    Convert a single node or an iterable of nodes into a set of nodes.
    Strings are treated as single nodes, not iterables of characters.
    """
    if isinstance(nodes, str):
        return {nodes}
    if isinstance(nodes, Iterable):
        return set(nodes)
    return {nodes}


def _is_collider(G: nx.DiGraph, a, m, b) -> bool:
    """
    Return True iff along the path segment a - m - b,
    node m is a collider: a -> m <- b.
    """
    return G.has_edge(a, m) and G.has_edge(b, m)


def _path_is_open(G: nx.DiGraph, path, Z, ancestors_of_Z) -> bool:
    """
    A path is open iff every internal node satisfies the d-separation rules:
    - non-collider must NOT be in Z
    - collider must be in Z or have a descendant in Z
    """
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        m = path[i]
        b = path[i + 1]

        if _is_collider(G, a, m, b):
            # Collider opens only if m in Z or a descendant of m is in Z.
            # Equivalent test: m is an ancestor of some node in Z, or m itself is in Z.
            if m not in ancestors_of_Z and m not in Z:
                return False
        else:
            # Chain or fork is blocked if the middle node is conditioned on.
            if m in Z:
                return False

    return True


def is_valid_d_separator(G: nx.DiGraph, X, Y, Z):
    """
    Check whether Z is a valid separator between X and Y according to Pearl's d-separation
    rules in a DAG.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic graph.
    X, Y, Z : node or iterable of nodes
        X and Y are the endpoint node sets.
        Z is the conditioning / separator set.

    Returns
    -------
    is_valid : bool
        True iff Z d-separates X from Y.
    details : dict
        Useful debugging information.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph.")

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Pearl's d-separation criterion here is defined for DAGs.")

    X = _normalize_node_set(X)
    Y = _normalize_node_set(Y)
    Z = _normalize_node_set(Z)

    # Standard definition assumes disjoint sets
    if X & Y:
        raise ValueError("X and Y must be disjoint.")
    if X & Z:
        raise ValueError("X and Z must be disjoint.")
    if Y & Z:
        raise ValueError("Y and Z must be disjoint.")

    missing = (X | Y | Z) - set(G.nodes)
    if missing:
        raise ValueError(f"These nodes are not in G: {missing}")

    # Undirected skeleton: paths are checked ignoring arrow direction
    skeleton = G.to_undirected()

    # Precompute all ancestors of Z.
    # A collider is opened if it is in Z or has a descendant in Z.
    ancestors_of_Z = set()
    for z in Z:
        ancestors_of_Z |= nx.ancestors(G, z)

    open_paths = []
    blocked_paths = []

    for x in X:
        for y in Y:
            if not nx.has_path(skeleton, x, y):
                continue

            for path in nx.all_simple_paths(skeleton, x, y):
                if _path_is_open(G, path, Z, ancestors_of_Z):
                    open_paths.append(path)
                else:
                    blocked_paths.append(path)

    is_valid = len(open_paths) == 0

    return is_valid, {
        "X": X,
        "Y": Y,
        "Z": Z,
        "open_paths": open_paths,
        "blocked_paths_count": len(blocked_paths),
    }



import networkx as nx
from typing import Set, Iterable


def _to_set(x):
    return set(x) if not isinstance(x, set) else x


def _ancestors_inclusive(G: nx.DiGraph, nodes: Iterable[str]) -> Set[str]:
    nodes = _to_set(nodes)
    anc = set(nodes)
    for v in nodes:
        anc |= nx.ancestors(G, v)
    return anc


def _descendants_inclusive(G: nx.DiGraph, nodes: Iterable[str]) -> Set[str]:
    nodes = _to_set(nodes)
    desc = set(nodes)
    for v in nodes:
        desc |= nx.descendants(G, v)
    return desc


def _causal_vertices(G: nx.DiGraph, X: Set[str], Y: Set[str]) -> Set[str]:
    """
    כל הצמתים שנמצאים על מסלול מכוון מ-X ל-Y.
    ב-DAG זה בדיוק:
        de(X) ∩ an(Y)
    עם הכללה של X,Y עצמם.
    """
    deX = _descendants_inclusive(G, X)
    anY = _ancestors_inclusive(G, Y)
    return deX & anY


def _forbidden_vertices(G: nx.DiGraph, X: Set[str], Y: Set[str]) -> Set[str]:
    """
    forb_G(X,Y) = X ∪ de(cv_G(X,Y))
    """
    cv = _causal_vertices(G, X, Y)
    return X | _descendants_inclusive(G, cv)


def _proper_backdoor_graph(G: nx.DiGraph, X: Set[str], Y: Set[str]) -> nx.DiGraph:
    """
    מוחק את הקשת הראשונה של כל מסלול מכוון מ-X ל-Y:
        remove {x->u : x in X, u in cv_G(X,Y)}
    """
    H = G.copy()
    cv = _causal_vertices(G, X, Y)
    edges_to_remove = [(x, u) for x in X for u in G.successors(x) if u in cv]
    H.remove_edges_from(edges_to_remove)
    return H


def build_H1_graph(sem: "LinearSEM", X, Y, I=None) -> nx.Graph:
    """
    בונה את H1_{X,Y}(I,G) לפי המאמר:
      1. proper backdoor graph
      2. restricted to ancestors of X ∪ Y ∪ I
      3. moralization
      4. saturate forbidden nodes and remove them

    כאן מניחים שכל הצמתים observable.
    """
    G = sem.G
    X = _to_set(X if isinstance(X, (set, list, tuple)) else [X])
    Y = _to_set(Y if isinstance(Y, (set, list, tuple)) else [Y])
    I = _to_set([] if I is None else (I if isinstance(I, (set, list, tuple)) else [I]))

    # שלב 1: proper backdoor graph
    G_pbd = _proper_backdoor_graph(G, X, Y)

    # שלב 2: ancestors of X ∪ Y ∪ I
    anc = _ancestors_inclusive(G, X | Y | I)
    G_anc = G_pbd.subgraph(anc).copy()

    # שלב 3: moral graph
    H0 = nx.moral_graph(G_anc)

    # שלב 4: saturate forbidden vertices, then remove them
    forbidden = _forbidden_vertices(G, X, Y) & set(H0.nodes())

    H1 = H0.copy()
    for v in forbidden:
        if v not in H1:
            continue
        nbrs = list(H1.neighbors(v))
        # saturate neighborhood: make neighbors of v a clique
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                H1.add_edge(nbrs[i], nbrs[j])

    H1.remove_nodes_from(forbidden)

    return H1


def is_adjustment_set(sem: "LinearSEM", X: str, Y: str, Z: Iterable[str], I=None) -> bool:
    """
    מחזירה האם Z הוא adjustment set תקף עבור X,Y ב-DAG.

    תנאים שנבדקים:
    1. Z לא מכיל צמתים אסורים
    2. ב-H1, הקבוצה Z מפרידה בין X ל-Y

    הערה:
    - אם X או Y לא נמצאים ב-H1 (בגלל forbidden-removal), הפונקציה בונה
      צמתי עזר s,t שמחוברים לשכנים של X,Y בהתאמה, כמו במעבר של Theorem 4.
    """
    G = sem.G
    Z = _to_set(Z)

    if X not in G or Y not in G:
        raise ValueError("X or Y not in graph")
    if not Z.issubset(set(G.nodes())):
        raise ValueError("Z contains nodes not in graph")
    if X in Z or Y in Z:
        return False

    forbidden = _forbidden_vertices(G, {X}, {Y})
    if Z & forbidden:
        return False

    H1 = build_H1_graph(sem, X, Y, I=I)

    # אם X,Y קיימים ב-H1 נבדוק ישירות
    if X in H1 and Y in H1:
        H1_minus_Z = H1.copy()
        H1_minus_Z.remove_nodes_from(Z)
        return not nx.has_path(H1_minus_Z, X, Y)

    # אחרת: בונים גרף עזר עם s,t שמחוברים לשכנים של X,Y ב-H1
    H = H1.copy()
    s, t = "__s__", "__t__"
    H.add_node(s)
    H.add_node(t)

    # שכנים של X ב-H1: צמתים שלא אסורים, שהיו שכנים של X אחרי moralization/saturation
    # דרך מעשית ופשוטה: נבנה אותם מתוך H0 לפני מחיקת forbidden
    Xs = {X}
    Ys = {Y}
    Iset = _to_set([] if I is None else (I if isinstance(I, (set, list, tuple)) else [I]))

    G_pbd = _proper_backdoor_graph(G, Xs, Ys)
    anc = _ancestors_inclusive(G, Xs | Ys | Iset)
    G_anc = G_pbd.subgraph(anc).copy()
    H0 = nx.moral_graph(G_anc)

    forbidden_full = _forbidden_vertices(G, Xs, Ys) & set(H0.nodes())

    # saturate forbidden in H0
    H0_sat = H0.copy()
    for v in forbidden_full:
        if v not in H0_sat:
            continue
        nbrs = list(H0_sat.neighbors(v))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                H0_sat.add_edge(nbrs[i], nbrs[j])

    # שכנים לא-אסורים של X,Y
    x_neighbors = [u for u in H0_sat.neighbors(X) if u not in forbidden_full and u in H]
    y_neighbors = [u for u in H0_sat.neighbors(Y) if u not in forbidden_full and u in H]

    for u in x_neighbors:
        H.add_edge(s, u)
    for u in y_neighbors:
        H.add_edge(t, u)

    H_minus_Z = H.copy()
    H_minus_Z.remove_nodes_from(Z)

    # אם אין מסלול מ-s ל-t אז Z מפריד ולכן הוא adjustment set
    return not nx.has_path(H_minus_Z, s, t)

'''
sem = load_sem_from_json("graph_8_v4_v6.json")

Z1 = {"V0", "V14", "V15", "V16", "V17", "V3", "V5"}
Z2 = {"V0", "V15", "V17", "V2", "V3", "V5"}

print(is_adjustment_set(sem, "V4", "V6", Z1))
print(is_adjustment_set(sem, "V4", "V6", Z2))
'''