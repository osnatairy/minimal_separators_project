
from minimal_separators_project.sem.linear_sem import make_linear_sem
from minimal_separators_project.sem.adjustment_wrapper import run_many_xy
from minimal_separators_project.sem.variance import example_compute_avar


if __name__ == "__main__":
    # 1) Generate a linear SEM (your code)
    sem = make_linear_sem(
        n=20,
        edge_prob=0.22,
        beta_scale=1.0,
        sigma2_low=0.2,
        sigma2_high=1.0,
        node_prefix="V",
        seed_graph=1,
        seed_params=2
    )

    # 2) Run over many (X,Y) pairs and find adjustment sets
    results = run_many_xy(
        sem,
        R=list(sem.G.nodes()),
        I=[],
        mode="reachable",
        sample_k=20,
        seed=42
    )

    # 3) Print a compact summary
    scores = []
    for (X, Y), Z_sets in results.items():
        for Z in Z_sets:
            if len(Z) > 1:
                aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
                scores.append((aVar, Z))

    if len(scores) > 0:
        scores.sort()
        best_aVar, best_Z = scores[0]
        print("Best Z:", best_Z, "best aVar:", best_aVar)

