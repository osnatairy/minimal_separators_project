from typing import Dict, List, Tuple, Optional
import utils as utils

from i_o.json_loader import save_linear_sem,load_linear_sem

from sem.linear_sem import make_linear_sem, remove_edge_from_sem
from sem.variance import example_compute_avar
from pipelines.adjust_sets import find_adjustment_sets_for_pair
# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results

from validation.separators_consistency import find_non_minimal_st_sep

seed = 9

sem = make_linear_sem(
            n=20,
            edge_prob=0.25,
            beta_scale=1.0,
            sigma2_low=0.1,#.2,
            sigma2_high=0.5,#1.0,
            node_prefix="V",
            seed=seed
        )


#remove_edge_from_sem(sem, "V7", "V16")

X = "V10"
Y = "V11"

R = list(sem.G.nodes())
I = []

H, Z_sets = find_adjustment_sets_for_pair(sem.G, X, Y, R=R, I=I)

file_name = "outputs/graph_"+str(seed)+"_"+X+"_"+Y+".json"
save_linear_sem(file_name,sem)

file_name = "outputs/graph_H_"+str(seed)+"_"+X+"_"+Y+".json"
save_linear_sem(file_name,sem)


forward, reverse = cy_components_for_sets(H, Y, Z_sets)
#print(forward, reverse)
# get Hass graph for the Z - the adjustment sets
res = hasse_from_cy_results(forward, reverse)
#print("***************************************")
#print(res)
#print("***************************************")


H.graph['st'] = (X,Y)
utils.visualize_g(H)

sets_Z_sets = [set(fs) for fs in Z_sets]
s = find_non_minimal_st_sep(H, sets_Z_sets)

sem.G.graph['st'] = (X,Y)
utils.visualize_g(sem.G)

s = find_non_minimal_st_sep(sem.G, sets_Z_sets)
# if len(s['NOT_SEP']) > 0 or len(s['NOT_MIN']) > 0:
#     print(f"in seperators1: non separators {s['NOT_SEP']}")
#     print(f"in seperators1: non minimal separators {s['NOT_SEP']}")
#     bad_seps = 1


results: Dict[Tuple[str, ...], float] = {}

for Z in Z_sets:
    if len(Z) >= 1:
        Z_key = tuple(sorted(Z))
        aVar = example_compute_avar(sem, X=X, Y=Y, Z=Z)
        results[Z_key] = aVar


print(results)


