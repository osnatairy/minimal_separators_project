
# do_example_run.py
# Usage example (assumes you have bn.py, loader.py, bn_12_nodes.json in the same folder).
from minimal_separators_project.bn import BN
from minimal_separators_project.io.json_loader import load_bn_from_json
from do_adjustment import interventional_distribution, expectation_under_do, variance_under_do, compare_across_Zsets

# 1) Load bn
bn = load_bn_from_json("../BN_DATA/bn_12_nodes.json", BNClass=BN)

# 2) Define your causal query
Y = "K"
X_assign = {"I": 1}
# Example adjustment candidates (you will pass the ones you found via your enumerator)
Z_sets = [["G","H"], ["E","F","G"], []]  # NOTE: [] may not be valid; for demo only

# 3) Compute for one Z
pY_do = interventional_distribution(bn, Y, X_assign, Z_sets[0])
print("P(K | do(I=1)) via Z=['G','H']:", pY_do)
print("E[K | do(I=1)] =", expectation_under_do(bn, Y, X_assign, Z_sets[0]))
print("Var(K | do(I=1)] =", variance_under_do(bn, Y, X_assign, Z_sets[0]))

# 4) Compare multiple Z candidates (should match if all are valid adjustment sets)
results, all_equal = compare_across_Zsets(bn, Y, X_assign, Z_sets)
print("All Z give same distribution?", all_equal)
for Z, dist in results:
    print("Z =", list(Z), "→", dist)
