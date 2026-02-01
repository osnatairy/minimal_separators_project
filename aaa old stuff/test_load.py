import networkx as nx
from minimal_separators_project.io import json_loader as load_network
from Baysien_Network import BN

bn = load_network.load_bn_from_json("../BN_DATA/bn_12_nodes.json", BNClass=BN)

# 1) sanity: must be a DAG
assert nx.is_directed_acyclic_graph(bn.g)

# 2) check a prior: P(A=1)
print("P(A=1) =", bn.marginal_prob({"A": 1}))

# 3) check a conditional from a CPT row: P(D=1 | A=1,B=0)
print("P(D=1 | A=1,B=0) =", bn.conditional({"D": 1}, {"A": 1, "B": 0}))

# 4) a marginal that requires summing: P(E=1) = sum_{a,b,c} P(E=1|b,c)P(b)P(c|a)P(a)
print("P(E=1) =", bn.marginal_prob({"E": 1}))

