import json
from pathlib import Path

def _coerce_prob_keys_to_domain_types(prob_dict, domain):
    """
    JSON keys come as strings; map them back to the type used in `domain`.
    If domain is numeric {0,1}, turn "0"/"1" -> 0/1.
    If domain is strings {"low","high"}, keep as strings.
    """
    # choose a representative element to detect type
    sample = domain[0] if len(domain) > 0 else None
    if isinstance(sample, int):
        return {int(k): float(v) for k, v in prob_dict.items()}
    elif isinstance(sample, float):
        return {float(k): float(v) for k, v in prob_dict.items()}
    else:
        # strings or other hashables – leave keys as-is
        return {k: float(v) for k, v in prob_dict.items()}

def load_bn_from_json(path, BNClass):
    """
    Load a Bayesian network from the JSON format above into an instance of BNClass.
    BNClass must implement: add_var, add_edge, set_parent_order, set_cpt.
    """

    # תיקיית הפרויקט (minimal_separators_project)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # נתיב מלא לקובץ
    full_path = PROJECT_ROOT / path

    with open(full_path, "r") as f:
        spec = json.load(f)

    # 1) Create an empty bn
    bn = BNClass()

    # 2) Add variables and domains
    for var, info in spec["variables"].items():
        domain = info["domain"]
        bn.add_var(var, domain)

    # 3) Add edges (parent -> child)
    for u, v in spec["edges"]:
        bn.add_edge(u, v)

    # 4) Set parent order (the canonical order for CPT indexing)
    for var, order in spec["parent_order"].items():
        bn.set_parent_order(var, order)

    # 5) Build CPTs for every variable
    #    Convert JSON rows (list) into: { parent_tuple -> { value: prob } }
    for var, rows in spec["cpts"].items():
        table = {}
        # we need the child domain to coerce prob keys
        child_domain = bn.domains[var]

        for row in rows:
            ptuple = tuple(row["parents"])   # ordered per parent_order[var]
            prob = _coerce_prob_keys_to_domain_types(row["prob"], child_domain)
            table[ptuple] = prob

        # This will also check normalization & coverage (if your set_cpt validates)
        bn.set_cpt(var, table)

    return bn