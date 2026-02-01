
from sem.linear_sem import make_linear_sem
from sem.adjustment_wrapper import run_many_xy
from sem.variance import example_compute_avar

from i_o.utils import save_list_json, append_line


if __name__ == "__main__":

    import random

    N = 10
    seeds_to_keep = []
    file_name = "outputs/seeds_data_20_25.csv"
    append_line(file_name,
                "seed, graph_nodes, graph_edges,X, Y, H_graph_nodes, H_graph_edges, num_seperator, num_contained_separators\n")
    for seed in range(N):
        # 1) Generate a linear SEM (your code)
        sem = make_linear_sem(
            n=20,
            edge_prob=0.25,
            beta_scale=1.0,
            sigma2_low=0.2,
            sigma2_high=1.0,
            node_prefix="V",
            seed_graph=seed,
            seed_params=2
        )



        # 2) Run over many (X,Y) pairs and find adjustment sets
        results = run_many_xy(
            sem,
            R=list(sem.G.nodes()),
            I=[],
            mode="reachable",
            sample_k=20,
            seed=seed,
            test_mode=True,
            file_name=file_name
        )

        if len(results) > 0:
            seeds_to_keep.append(seed)

    save_list_json(seeds_to_keep, "outputs/seeds_to_keep.json")
    '''
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
    '''
