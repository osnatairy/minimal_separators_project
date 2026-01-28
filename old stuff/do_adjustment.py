
"""
do_adjustment.py
----------------
Utilities to compute P(Y | do(X=x)) via backdoor adjustment using your bn object.

Requirements on `bn`:
- bn.domains: Dict[str, List[Any]]
- bn.marginal_prob(fixed: Dict[str, Any]) -> float
- bn.conditional(target: Dict[str, Any], given: Dict[str, Any]) -> float
- bn._enumerate_assignments(vars_subset: Iterable[str]) -> Iterator[Dict[str, Any]]  (optional)
"""

from typing import Dict, List, Iterable, Any, Tuple
from itertools import product

def _enumerate_assignments_fallback(domains: Dict[str, List[Any]], vars_subset: Iterable[str]):
    """Local enumeration if bn does not expose _enumerate_assignments."""
    vars_subset = list(vars_subset)
    if not vars_subset:
        yield {}
        return
    grids = [domains[v] for v in vars_subset]
    for values in product(*grids):
        yield dict(zip(vars_subset, values))

def _enum(bn, vars_subset: Iterable[str]):
    """Use bn._enumerate_assignments if present; otherwise fallback."""
    if hasattr(bn, "_enumerate_assignments"):
        return bn._enumerate_assignments(vars_subset)
    return _enumerate_assignments_fallback(bn.domains, vars_subset)

def interventional_distribution(bn, Y: str, X_assign: Dict[str, Any], Z_vars: List[str]):
    """
    Compute P(Y | do(X=x)) by backdoor adjustment:
      if Z empty: P(Y|X=x)
      else:       sum_z P(Y|X=x,z) * P(z)

    Returns: dict { y_value: probability }
    """
    # Defensive: ensure keys/values are valid
    if Y not in bn.domains:
        raise ValueError(f"Unknown variable Y={Y}")
    for Xk, Xv in X_assign.items():
        if Xk not in bn.domains: raise ValueError(f"Unknown X var {Xk}")
        if Xv not in bn.domains[Xk]: raise ValueError(f"X value {Xk}={Xv} not in domain {bn.domains[Xk]}")
    for z in Z_vars:
        if z not in bn.domains: raise ValueError(f"Unknown Z var {z}")

    pY = {y: 0.0 for y in bn.domains[Y]}

    if not Z_vars:
        # observational conditional
        den = 0.0
        for y in bn.domains[Y]:
            p = bn.conditional({Y: y}, X_assign)
            pY[y] = float(p)
            den += pY[y]
        # Normalize defensively
        if den > 0:
            for y in pY:
                pY[y] /= den
        return pY

    # Sum over all joint Z assignments
    for z_assign in _enum(bn, Z_vars):
        Pz = float(bn.marginal_prob(z_assign))
        if Pz == 0.0:
            continue
        # P(Y | X=x, z)
        # Build evidence dict
        evidence = dict(X_assign)
        evidence.update(z_assign)
        # Collect conditional over Y values
        for y in bn.domains[Y]:
            Py_xz = float(bn.conditional({Y: y}, evidence))
            pY[y] += Py_xz * Pz

    # Defensive normalization (should already sum to ~1)
    s = sum(pY.values())
    if s > 0:
        for y in pY:
            pY[y] /= s
    return pY

def expectation_under_do(bn, Y: str, X_assign: Dict[str, Any], Z_vars: List[str], value_map: Dict[Any, float] = None) -> float:
    """
    E[Y | do(X=x)]. If Y is binary {0,1} and value_map is None, uses {0:0.0, 1:1.0}.
    For non-binary Y, provide `value_map` mapping each Y value to a numeric value.
    """
    if value_map is None:
        dom = set(bn.domains[Y])
        if dom == {0, 1}:
            value_map = {0: 0.0, 1: 1.0}
        else:
            raise ValueError("Provide value_map for non-binary Y.")
    pY = interventional_distribution(bn, Y, X_assign, Z_vars)
    return sum(float(value_map[y]) * float(p) for y, p in pY.items())

def variance_under_do(bn, Y: str, X_assign: Dict[str, Any], Z_vars: List[str], value_map: Dict[Any, float] = None) -> float:
    """
    Var(Y | do(X=x)) = E[Y^2 | do] - (E[Y | do])^2.
    For binary Y with implicit mapping {0:0,1:1}, this equals p*(1-p).
    """
    if value_map is None:
        dom = set(bn.domains[Y])
        if dom == {0, 1}:
            value_map = {0: 0.0, 1: 1.0}
        else:
            raise ValueError("Provide value_map for non-binary Y.")
    pY = interventional_distribution(bn, Y, X_assign, Z_vars)
    EY  = sum(float(value_map[y])**1 * float(p) for y, p in pY.items())
    EY2 = sum(float(value_map[y])**2 * float(p) for y, p in pY.items())
    return float(EY2 - EY**2)

def compare_across_Zsets(bn, Y: str, X_assign: Dict[str, Any], Z_sets: List[List[str]], atol: float = 1e-9):
    """
    Compute P(Y | do(X=x)) for multiple candidate adjustment sets Z.
    Returns:
      results: List[Tuple[Tuple[str,...], Dict[y,prob]]]
      all_equal: bool  (True iff all distributions are pairwise equal within atol)
    """
    results = []
    base = None
    all_equal = True

    for Z in Z_sets:
        pY = interventional_distribution(bn, Y, X_assign, Z)
        key = tuple(sorted(Z))
        results.append((key, pY))
        if base is None:
            base = pY
        else:
            # compare with base
            ys = bn.domains[Y]
            for y in ys:
                if abs(pY[y] - base[y]) > atol:
                    all_equal = False
                    break
    return results, all_equal
