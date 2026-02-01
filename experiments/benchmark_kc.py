import time
import random
import networkx as nx
from statistics import mean, median, pstdev

def sample_st_pair(G):
    # דוגמה פשוטה: בוחרים שני צמתים שונים אקראית
    nodes = list(G.nodes)
    s, t = random.sample(nodes, 2)
    return s, t

def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0), out

def estimate_kc_for_graph(G, takata_all_seps_fn, small_min_seps_fn, k_max=50):
    s, t = sample_st_pair(G)

    # T_max = זמן Takata (או timeout קבוע אם זה מה שאת עושה)
    T_max, _ = time_call(takata_all_seps_fn, G, s, t)

    # מחפשים את k הראשון שעובר את T_max
    for k in range(1, k_max + 1):
        Tk, _ = time_call(small_min_seps_fn, G, s, t, k)
        if Tk > T_max:
            return k, T_max
    return None, T_max  # לא עבר עד k_max

def estimate_kc(n, p, trials=100, directed=False, seed=None,
                takata_all_seps_fn=None, small_min_seps_fn=None,
                k_max=50):
    rng = random.Random(seed)
    kc_vals = []
    tmax_vals = []

    for i in range(trials):
        # דגימה מ-G(n,p)
        if directed:
            G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**9), directed=True)
        else:
            G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**9), directed=False)

        kc, T_max = estimate_kc_for_graph(
            G,
            takata_all_seps_fn=takata_all_seps_fn,
            small_min_seps_fn=small_min_seps_fn,
            k_max=k_max
        )
        if kc is not None:
            kc_vals.append(kc)
        tmax_vals.append(T_max)

    return {
        "n": n,
        "p": p,
        "kc_mean": mean(kc_vals) if kc_vals else None,
        "kc_median": median(kc_vals) if kc_vals else None,
        "kc_std": pstdev(kc_vals) if len(kc_vals) > 1 else 0.0,
        "kc_samples": len(kc_vals),
        "Tmax_mean": mean(tmax_vals),
    }
