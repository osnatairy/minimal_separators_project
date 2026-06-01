import utils_ms as utils
import PreProcessing_create_H_graph as pp


# טסט קייסים מקיפים
def test_causal_vertices():
    print("🧪 טסט קייסים לפונקציות BFS ו-Causal Vertices\n")

    # טסט קייס 1: גרף ליניארי פשוט
    print("📋 טסט קייס 1: גרף ליניארי")
    G1 = nx.DiGraph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    [s, t] = [1, 5]
    G1.graph['st'] = (s, t)
    #utils.visualize_g(G1)
    print("גרף:", list(G1.edges()))
    print("קודקודים סיבתיים בין 1 ל-5:", pp.find_causal_vertices_sets_optimized(G1, 1, 5))
    print("תוצאה צפויה: {2, 3, 4}")
    print()

    # טסט קייס 2: גרף עם הסתעפות
    print("📋 טסט קייס 2: גרף עם הסתעפות")
    G2 = nx.DiGraph()
    # A → B → D → F
    # A → C → E → F
    G2.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'F'), ('E', 'F')])
    [s, t] = ['A', 'F']
    G2.graph['st'] = (s, t)
    #utils.visualize_g(G2)
    print("גרף:", list(G2.edges()))
    print("קודקודים סיבתיים בין A ל-F:", pp.find_causal_vertices_sets_optimized(G2, 'A', 'F'))
    print("תוצאה צפויה: שני המסלולים עוברים דרך B,D או C,E")
    print()

    # טסט קייס 3: גרף מורכב יותר
    print("📋 טסט קייס 3: גרף מורכב")
    G3 = nx.DiGraph()
    # 1 → 2 → 3 → 4 → 5
    #     ↓     ↓
    #     6 → 7 → 8
    #         ↓
    #         9
    G3.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (3, 7), (7, 8), (7, 9)])
    [s, t] = [1, 5]
    G3.graph['st'] = (s, t)
    #utils.visualize_g(G3)
    print("גרף:", list(G3.edges()))
    print("קודקודים סיבתיים בין 1 ל-5:", pp.find_causal_vertices_sets_optimized(G3, 1, 5))
    print("קודקודים סיבתיים בין 1 ל-8:", pp.find_causal_vertices_sets_optimized(G3, 1, 8))
    print("קודקודים סיבתיים בין 2 ל-9:", pp.find_causal_vertices_sets_optimized(G3, 2, 9))
    print()

    # טסט קייס 4: אין מסלול
    print("📋 טסט קייס 4: אין מסלול")
    G4 = nx.DiGraph()
    G4.add_edges_from([(1, 2), (3, 4)])  # שני רכיבים נפרדים
    [s, t] = [1, 4]
    G4.graph['st'] = (s, t)
    #utils.visualize_g(G4)
    print("גרף:", list(G4.edges()))
    print("קודקודים סיבתיים בין 1 ל-4:", pp.find_causal_vertices_sets_optimized(G4, 1, 4))
    print("תוצאה צפויה: set() (קבוצה ריקה)")
    print()

    # טסט קייס 5: מסלול ישיר (אין קודקודים באמצע)
    print("📋 טסט קייס 5: מסלול ישיר")
    G5 = nx.DiGraph()
    G5.add_edges_from([(1, 2)])
    [s, t] = [1, 2]
    G5.graph['st'] = (s, t)
    #utils.visualize_g(G5)
    print("גרף:", list(G5.edges()))
    print("קודקודים סיבתיים בין 1 ל-2:", pp.find_causal_vertices_sets_optimized(G5, 1, 2))
    print("תוצאה צפויה: set() (אין קודקודים באמצע)")
    print()

    # טסט קייס 6: גרף עם מסלולים מרובים
    print("📋 טסט קייס 6: מסלולים מרובים")
    G6 = nx.DiGraph()
    # X → A → Y
    # X → B → C → Y
    # X → D → Y
    G6.add_edges_from([('X', 'A'), ('A', 'Y'), ('X', 'B'), ('B', 'C'), ('C', 'Y'), ('X', 'D'), ('D', 'Y')])
    [s, t] = ['X', 'Y']
    G6.graph['st'] = (s, t)
    #utils.visualize_g(G6)
    print("גרף:", list(G6.edges()))
    print("קודקודים סיבתיים בין X ל-Y:", pp.find_causal_vertices_sets_optimized(G6, 'X', 'Y'))
    print("תוצאה צפויה: רק הקודקודים שנמצאים על כל המסלולים")
    print()

    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 5), (5, 4), (6, 7), (7, 4)])

    X = {1, 6}
    Y = {4}

    print("גרף דוגמה:", list(G.edges()))
    print(f"X = {X}, Y = {Y}")

    result_v1 = pp.find_causal_vertices_sets_v1(G, X, Y)
    result_v2 = pp.find_causal_vertices_sets_v2(G, X, Y)
    result_opt = pp.find_causal_vertices_sets_optimized(G, X, Y)

    print(f"תוצאה גישה 1: {result_v1}")
    print(f"תוצאה גישה 2: {result_v2}")
    print(f"תוצאה אופטימלית: {result_opt}")

    # בדיקה שכל הגישות נותנות אותה תוצאה
    assert result_v1 == result_v2 == result_opt, "התוצאות שונות בין הגישות!"
    print("✓ כל הגישות נותנות תוצאה זהה")


# דוגמאות לשימוש וטסטים
def test_remove_edges():
    print("🧪 טסטים לפונקציית הסרת קשתות\n")

    # טסט 1: גרף פשוט
    print("📋 טסט 1: גרף פשוט")
    G1 = nx.DiGraph()
    G1.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)])
    cv_set1 = {2, 3}
    x1 = 1

    print(f"גרף מקורי: {list(G1.edges())}")
    print(f"צומת X: {x1}")
    print(f"קבוצת CV: {cv_set1}")

    G1_new = pp.remove_edges_from_x_to_cv(G1, x1, cv_set1)
    print(f"גרף חדש: {list(G1_new.edges())}")
    print()

    # טסט 2: גרף מורכב יותר
    print("📋 טסט 2: גרף מורכב")
    G2 = nx.DiGraph()
    # X מחובר לכמה צמתים, חלקם ב-CV וחלקם לא
    G2.add_edges_from([('X', 'A'), ('X', 'B'), ('X', 'C'), ('A', 'D'), ('B', 'D'), ('C', 'E')])
    cv_set2 = {'A', 'C', 'F'}  # F לא קיים בגרף
    x2 = 'X'

    print(f"גרף מקורי: {list(G2.edges())}")
    print(f"צומת X: {x2}")
    print(f"קבוצת CV: {cv_set2}")

    G2_new = pp.remove_edges_from_x_to_cv(G2, x2, cv_set2)
    print(f"גרף חדש: {list(G2_new.edges())}")
    print()

    # טסט 3: אין קשתות להסיר
    print("📋 טסט 3: אין קשתות להסיר")
    G3 = nx.DiGraph()
    G3.add_edges_from([(1, 2), (2, 3), (3, 4)])
    cv_set3 = {3, 4}
    x3 = 1

    print(f"גרף מקורי: {list(G3.edges())}")
    print(f"צומת X: {x3}")
    print(f"קבוצת CV: {cv_set3}")

    G3_new = pp.remove_edges_from_x_to_cv(G3, x3, cv_set3)
    print(f"גרף חדש: {list(G3_new.edges())}")
    print()

    # טסט 4: X לא קיים בגרף
    print("📋 טסט 4: X לא קיים בגרף")
    G4 = nx.DiGraph()
    G4.add_edges_from([(1, 2), (2, 3)])
    cv_set4 = {2, 3}
    x4 = 5  # לא קיים

    print(f"גרף מקורי: {list(G4.edges())}")
    print(f"צומת X: {x4}")
    print(f"קבוצת CV: {cv_set4}")

    try:
        G4_new = pp.remove_edges_from_x_to_cv(G4, x4, cv_set4)
        print(f"גרף חדש: {list(G4_new.edges())}")
    except Exception as e:
        print(f"שגיאה: {e}")
    print()

def analyze_induced_subgraph_structure(G, induced_subgraph, X, Y, Z):
    """
    מנתח את המבנה של התת-גרף המושרה
    """
    X_set = set(X) if isinstance(X, (list, set)) else {X}
    Y_set = set(Y) if isinstance(Y, (list, set)) else {Y}
    Z_set = set(Z) if isinstance(Z, (list, set)) else {Z}
    V_prime = X_set.union(Y_set).union(Z_set)

    print(f"ניתוח התת-גרף המושרה:")
    print(f"קבוצת X: {X_set}")
    print(f"קבוצת Y: {Y_set}")
    print(f"קבוצת Z: {Z_set}")
    print(f"V' = X ∪ Y ∪ Z: {V_prime}")
    print(f"מספר צמתים בתת-גרף המושרה: {len(induced_subgraph)}")

    # ספירת קשתות בתת-גרף המושרה
    edge_count = sum(len(neighbors) for neighbors in induced_subgraph.values()) // 2
    print(f"מספר קשתות בתת-גרף המושרה: {edge_count}")

    # ספירת קשתות בגרף המקורי בין אותם צמתים
    original_edges = 0
    for node in V_prime:
        if node in G:
            for neighbor in G[node]:
                if neighbor in V_prime and node < neighbor:  # נמנע מספירה כפולה
                    original_edges += 1

    print(f"קשתות במקור בין הצמתים של V': {original_edges}")
    print(f"האם כל הקשתות שמורות: {'כן' if edge_count == original_edges else 'לא'}")

    return {
        'V_prime': V_prime,
        'induced_edges': edge_count,
        'original_edges': original_edges,
        'all_edges_preserved': edge_count == original_edges
    }

def test_induced_subgraph(G, X, Y, Z):
    induce_g = pp.create_induced_subgraph(G, X, Y, Z)
    analyze_induced_subgraph_structure(G,induce_g, X, Y, Z)


import networkx as nx
import unittest


class TestMoralGraph(unittest.TestCase):

    def test_empty_graph(self):
        """בדיקת גרף ריק"""
        dag = nx.DiGraph()
        moral = pp.create_moral_graph(dag)

        self.assertEqual(len(moral.nodes), 0)
        self.assertEqual(len(moral.edges), 0)

    def test_single_node(self):
        """בדיקת צומת בודד"""
        dag = nx.DiGraph()
        dag.add_node('A')
        moral = pp.create_moral_graph(dag)

        self.assertEqual(set(moral.nodes), {'A'})
        self.assertEqual(len(moral.edges), 0)

    def test_single_edge(self):
        """בדיקת קשת בודדת"""
        dag = nx.DiGraph()
        dag.add_edge('A', 'B')
        moral = pp.create_moral_graph(dag)

        self.assertEqual(set(moral.nodes), {'A', 'B'})
        self.assertEqual(set(moral.edges), {('A', 'B')})

    def test_linear_chain(self):
        """בדיקת שרשרת ליניארית: A → B → C → D"""
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D'}
        expected_edges = {('A', 'B'), ('B', 'C'), ('C', 'D')}

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_v_structure(self):
        """בדיקת V-structure: A → C ← B"""
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'C'), ('B', 'C')])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C'}
        expected_edges = {('A', 'C'), ('B', 'C'), ('A', 'B')}  # A-B נוסף במהלך moralization

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_multiple_v_structures(self):
        """בדיקת מספר V-structures: A → D ← B, C → D"""
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'D'), ('B', 'D'), ('C', 'D')])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D'}
        # כל הורי D יחוברו זה לזה
        expected_edges = {('A', 'D'), ('B', 'D'), ('C', 'D'),
                          ('A', 'B'), ('A', 'C'), ('B', 'C')}

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_complex_dag(self):
        """בדיקת DAG מורכב יותר"""
        dag = nx.DiGraph()
        dag.add_edges_from([
            ('A', 'C'), ('B', 'C'),  # V-structure עבור C
            ('C', 'E'), ('D', 'E'),  # V-structure עבור E
            ('E', 'F')
        ])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D', 'E', 'F'}
        expected_edges = {
            ('A', 'C'), ('B', 'C'),  # קשתות מקוריות
            ('C', 'E'), ('D', 'E'),  # קשתות מקוריות
            ('E', 'F'),  # קשת מקורית
            ('A', 'B'),  # moralization של הורי C
            ('C', 'D')  # moralization של הורי E
        }

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_diamond_structure(self):
        """בדיקת מבנה יהלום: A → B, A → C, B → D, C → D"""
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D'}
        expected_edges = {
            ('A', 'B'), ('A', 'C'),  # קשתות מקוריות
            ('B', 'D'), ('C', 'D'),  # קשתות מקוריות
            ('B', 'C')  # moralization של הורי D
        }

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_isolated_nodes(self):
        """בדיקת צמתים מבודדים"""
        dag = nx.DiGraph()
        dag.add_nodes_from(['A', 'B', 'C', 'D'])
        dag.add_edge('A', 'B')  # רק קשת אחת
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D'}
        expected_edges = {('A', 'B')}

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_star_structure(self):
        """בדיקת מבנה כוכב: A,B,C,D → E"""
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'E'), ('B', 'E'), ('C', 'E'), ('D', 'E')])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D', 'E'}
        # כל הורי E יחוברו זה לזה - זה יוצר clique מלא
        expected_edges = {
            ('A', 'E'), ('B', 'E'), ('C', 'E'), ('D', 'E'),  # קשתות מקוריות
            ('A', 'B'), ('A', 'C'), ('A', 'D'),  # moralization
            ('B', 'C'), ('B', 'D'), ('C', 'D')  # moralization
        }

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_no_moralization_needed(self):
        """בדיקת גרף שלא צריך moralization (עץ)"""
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E')])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {'A', 'B', 'C', 'D', 'E'}
        expected_edges = {('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E')}

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)

    def test_numeric_nodes(self):
        """בדיקה עם צמתים מספריים"""
        dag = nx.DiGraph()
        dag.add_edges_from([(1, 3), (2, 3), (3, 4)])
        moral = pp.create_moral_graph(dag)

        expected_nodes = {1, 2, 3, 4}
        expected_edges = {(1, 3), (2, 3), (3, 4), (1, 2)}

        self.assertEqual(set(moral.nodes), expected_nodes)
        self.assertEqual(set(moral.edges), expected_edges)


def run_visual_test():
    """פונקציה להרצת בדיקה ויזואלית"""

    print("=== Visual Test: Complex DAG ===")

    # יצירת DAG מורכב
    dag = nx.DiGraph()
    dag.add_edges_from([
        ('X', 'Z'), ('Y', 'Z'),  # V-structure
        ('Z', 'W'), ('U', 'W'), ('V', 'W'),  # עוד V-structure
        ('W', 'Q')
    ])
    dag.graph['st'] = ('X', 'W')
    #utils.visualize_g(dag)
    moral = pp.create_moral_graph(dag)
    moral.graph['st'] = ('X', 'W')
    #utils.visualize_g(moral)

    print(f"Original DAG nodes: {list(dag.nodes)}")
    print(f"Original DAG edges: {list(dag.edges)}")
    print(f"\nMoral graph nodes: {list(moral.nodes)}")
    print(f"Moral graph edges: {list(moral.edges)}")

    # הדפסת הקשתות שנוספו
    original_edges_undirected = set()
    for u, v in dag.edges:
        original_edges_undirected.add((min(u, v), max(u, v)))

    moral_edges_normalized = set()
    for u, v in moral.edges:
        moral_edges_normalized.add((min(u, v), max(u, v)))

    added_edges = moral_edges_normalized - original_edges_undirected
    print(f"\nEdges added during moralization: {list(added_edges)}")

run_visual_test()
if __name__ == "__main__":
    # הרצת כל הטסטים
    print("Running unit tests...")

    # TESTS
    #test_causal_vertices()
    #test_remove_edges()
    # test_induced_subgraph()

    unittest.main(argv=[''], verbosity=2, exit=False)

    print("\n" + "=" * 50)

    # הרצת הבדיקה הויזואלית
    run_visual_test()
