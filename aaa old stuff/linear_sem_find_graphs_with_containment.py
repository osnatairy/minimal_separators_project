from typing import Dict, List, Any
from linear_Structural_Equation_Model import make_linear_sem
from linear_sem_adjustment_sets_wrapper import run_many_xy  # <-- change to real import

def scan_seeds_for_containment(
    seeds: List[int],
    n: int = 5,
    edge_prob: float = 0.2,
    beta_scale: float = 1.0,
    sigma2_low: float = 0.2,
    sigma2_high: float = 1.0,
    node_prefix: str = "V",
    sample_k: int = 20,
    run_many_xy_fn = None
) -> Dict[int, Any]:
    """
    For each seed in seeds:
      - build sem = make_linear_sem(seed_graph=seed, seed_params=seed+1, ...)
      - run run_many_xy_fn(sem, ...)
      - check all (X,Y) that it returned for containment among Z_sets
    Returns a dict mapping seed -> info for seeds that have containment.
    """
    found = {}
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        sem = make_linear_sem(
            n=n,
            edge_prob=edge_prob,
            beta_scale=beta_scale,
            sigma2_low=sigma2_low,
            sigma2_high=sigma2_high,
            node_prefix=node_prefix,
            seed_graph=seed,
            seed_params=seed + 1

        )

        if run_many_xy_fn is None:
            raise ValueError("You must pass your run_many_xy function as run_many_xy_fn")

        # run your routine that finds separators for many (X,Y) pairs
        results = run_many_xy_fn(
            sem,
            R=list(sem.G.nodes()),
            I=[],
            mode="reachable",
            sample_k=sample_k,
            seed=seed+1000,
            test_mode=True

        )
        if len(results) >0:
            found.appand(seed)
    return found






if __name__ == "__main__":
    # define seeds to try
    seeds_to_try = list(range(1, 101))  # try 100 seeds
    # you must provide your own run_many_xy function here

    results_found = scan_seeds_for_containment(
        seeds=seeds_to_try,
        n=20,
        edge_prob=0.22,
        beta_scale=1.0,
        sigma2_low=0.2,
        sigma2_high=1.0,
        node_prefix="V",
        sample_k=None,
        run_many_xy_fn=run_many_xy
    )
    print("Done. Seeds with containment:", list(results_found.keys()))