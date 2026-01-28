
from minimal_separators_project.sem.linear_sem import make_linear_sem, save_sem_to_json, load_sem_from_json, sem_to_B_Omega, covariance_from_B_Omega


# -----------------------------
# 7) דוגמת שימוש שמראה התאמה לפייפליין שלך
# -----------------------------

if __name__ == "__main__":
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
    X = [f"V{n - 3}"]
    Y = [f"V{n - 1}"]
    R = list(G.nodes())  # נניח שכולם observed
    I = []  # אם אין אילוץ

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