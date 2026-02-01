import json
import csv
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------- Helpers: load + topological order ----------

def load_bn_spec(json_path: str) -> Dict[str, Any]:
    """Load bn spec (variables, edges, parent_order, cpts) from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def topological_order(variables: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """Kahn topological sort for DAG (parent -> child edges)."""
    indeg = {v: 0 for v in variables}
    children = defaultdict(list)

    for u, v in edges:
        children[u].append(v)
        indeg[v] += 1

    q = deque([v for v in variables if indeg[v] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(variables):
        raise ValueError("Graph is not a DAG or has missing variables in edges.")
    return order


# ---------- Helpers: CPT parsing + sampling ----------

def _coerce_value_to_domain_type(value_as_str: str, domain: List[Any]) -> Any:
    """
    CPT prob keys are strings in JSON; domains can be int/str/etc.
    Convert key back into the domain's type.
    """
    if not domain:
        return value_as_str
    t = type(domain[0])
    # If already correct type, return
    if isinstance(value_as_str, t):
        return value_as_str
    # Try to cast (int/float/etc). If fails, keep string.
    try:
        return t(value_as_str)
    except Exception:
        return value_as_str


def build_cpt_lookup(spec: Dict[str, Any]) -> Dict[str, Dict[Tuple[Any, ...], Dict[Any, float]]]:
    """
    Build: cpt[var][parent_tuple] = {value: prob}
    Parent tuple order is spec["parent_order"][var]
    """
    variables = spec["variables"]
    cpts_spec = spec["cpts"]

    cpt = {}
    for var, rows in cpts_spec.items():
        domain = variables[var]["domain"]
        table = {}
        for row in rows:
            ptuple = tuple(row["parents"])  # already ordered per parent_order[var] in your format
            prob_map = {}
            for k_str, p in row["prob"].items():
                val = _coerce_value_to_domain_type(k_str, domain)
                prob_map[val] = float(p)
            # (אופציונלי) בדיקת נרמול
            s = sum(prob_map.values())
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"CPT for var={var}, parents={ptuple} does not sum to 1 (sum={s})")
            table[ptuple] = prob_map
        cpt[var] = table

    return cpt


def sample_from_distribution(rng: random.Random, dist: Dict[Any, float]) -> Any:
    """Draw a single value from a discrete distribution."""
    r = rng.random()
    cum = 0.0
    last_val = None
    for val, p in dist.items():
        cum += p
        last_val = val
        if r <= cum:
            return val
    # numerical fallback
    return last_val


def generate_observations(
    json_path: str,
    n_samples: int,
    seed: Optional[int] = None,
    *,
    include_partial: Optional[List[str]] = None,
    missing_value: Any = ""
) -> List[Dict[str, Any]]:
    """
    Generate i.i.d. samples from the bn.
    - include_partial: אם רוצים להחזיר רק תת-קבוצה של משתנים (למשל רק משתני תצפית),
      תני רשימת שמות משתנים. יתר המשתנים יושמטו (או אפשר לשים missing_value אם רוצים).
    """
    spec = load_bn_spec(json_path)
    vars_list = list(spec["variables"].keys())
    edges = [tuple(e) for e in spec["edges"]]
    parent_order = spec["parent_order"]
    domains = {v: spec["variables"][v]["domain"] for v in vars_list}
    cpt = build_cpt_lookup(spec)

    topo = topological_order(vars_list, edges)
    rng = random.Random(seed)

    samples: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        assignment: Dict[str, Any] = {}
        for var in topo:
            parents = parent_order[var]
            ptuple = tuple(assignment[p] for p in parents) if parents else tuple()
            dist = cpt[var].get(ptuple)
            if dist is None:
                raise KeyError(f"Missing CPT row for var={var} with parents={parents} values={ptuple}")
            assignment[var] = sample_from_distribution(rng, dist)

        if include_partial is not None:
            filtered = {v: assignment.get(v, missing_value) for v in include_partial}
            samples.append(filtered)
        else:
            samples.append(assignment)

    return samples


# ---------- CSV I/O ----------

def save_observations_to_csv(samples: List[Dict[str, Any]], csv_path: str) -> None:
    """Save samples (list of dicts) to CSV with stable column order."""
    if not samples:
        raise ValueError("No samples to save.")
    fieldnames = list(samples[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def load_observations_csv(csv_path: str, *, as_dataframe: bool = True):
    """
    Load observations from CSV.
    - אם pandas מותקן ו-as_dataframe=True => מחזיר DataFrame
    - אחרת => מחזיר list[dict]
    """
    if as_dataframe and pd is not None:
        return pd.read_csv(csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


# ---------- Example usage ----------

if __name__ == "__main__":
    filename = "bn_12_nodes"
    bn_json = "BN_DATA/"+filename+".json"   # דוגמה (אצלך זה יהיה הנתיב לקובץ שתעבירי)
    out_csv = "BN_DATA/observations/obs_"+filename+".csv"

    # 1) דגימה מלאה של כל המשתנים
    samples = generate_observations(bn_json, n_samples=1000, seed=101)

    # (אופציונלי) אם את רוצה רק משתני "תצפית" (למשל רק עלים/או subset שאת מגדירה):
    # obs_vars = ["J", "K", "L"]  # דוגמה
    # samples = generate_observations(bn_json, n_samples=1000, seed=123, include_partial=obs_vars)

    save_observations_to_csv(samples, out_csv)

    # 2) טעינה לשימוש בהמשך (למשל לפני חישוב influence function)
    data = load_observations_csv(out_csv, as_dataframe=True)
    print(data.head() if hasattr(data, "head") else data[:3])
