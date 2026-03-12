from typing import Dict, List, Tuple, Optional
import utils as utils

from i_o.json_loader import save_linear_sem,load_linear_sem

from sem.linear_sem import make_linear_sem, remove_edge_from_sem
from sem.variance import example_compute_avar,x_drain_two_sets, condition_number_of_Z, compare_separators_stability
from pipelines.adjust_sets import find_adjustment_sets_for_pair
# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, adjustment_set_exists,separator_is_subset

from validation.separators_consistency import find_non_minimal_st_sep

from graph.hankel_optimal_set import optimal_adjustment_set_O
from graph.helpers import separator_with_min_variance

seed = 1

sem = make_linear_sem(
            n=20,
            edge_prob=0.25,
            beta_scale=1.0,
            sigma2_low=0.2,#.2,
            sigma2_high=1.0,#0.5,#1.0,
            node_prefix="V",
            seed=seed
        )


#remove_edge_from_sem(sem, "V7", "V16")

X = "V9"
Y = "V17"

R = list(sem.G.nodes())
I = []

H, Z_sets = find_adjustment_sets_for_pair(sem.G, X, Y, R=R, I=I)

file_name = "outputs/graph_"+str(seed)+"_"+X+"_"+Y+".json"
save_linear_sem(file_name,sem, [],len(Z_sets))

file_name = "outputs/graph_H_"+str(seed)+"_"+X+"_"+Y+".json"
save_linear_sem(file_name,sem,[],len(Z_sets))


forward, reverse = cy_components_for_sets(H, Y, Z_sets)
#print(forward, reverse)
# get Hass graph for the Z - the adjustment sets
res = hasse_from_cy_results(forward, reverse)
#print("***************************************")
#print(res)
#print("***************************************")


H.graph['st'] = (X,Y)
#utils.visualize_g(H)

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

# דוגמה:
for hass_list in res['hasse_edges']:
    z_in = list(res['component_to_Zs'][hass_list[0]][0])
    z_out = list(res['component_to_Zs'][hass_list[1]][0])
    print(f"z1={z_in}, z2={z_out}")
    res1 = x_drain_two_sets(sem, X, z_in,z_out)
    print(f"res1={results[tuple(sorted(z_out))]-results[tuple(sorted(z_in))]}")
    print(res1)

    #info = condition_number_of_Z(sem, Z, warn_threshold=1e8)
    #print(info)
    info = compare_separators_stability(sem, z_in,z_out, warn_threshold=1e8)
    print(info)




#from bn import BN
#from i_o.json_loader import load_bn_from_json

#new_g = load_bn_from_json("BN_DATA/bn_example3_5.json", BNClass=BN)

hankel_optimal = optimal_adjustment_set_O(sem.G, X, Y)
print(adjustment_set_exists(Z_sets,list(sorted(hankel_optimal))))
print(hankel_optimal)
best_sep, v = separator_with_min_variance(results)
print(separator_is_subset(best_sep,hankel_optimal))


