from __future__ import annotations
from collections import defaultdict

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from minimal_separators_project.bn.bayesian_network import BN
from minimal_separators_project.bn.inference import variable_elimination_factor

def _enumerate_Z_assignments(bn: BN, Z_vars: List[str]):
    """מחזיר גנרטור של השמותות לכל צירוף ערכי Z (dict)."""
    yield from bn._enumerate_assignments(Z_vars)

def compute_PY_given_X_and_specific_Z(
    bn: BN, Y: str, X_name: str, x_val: Any, z_assign: Dict[str, Any]
) -> Dict[Any, float]:
    """
    מחשב את P(Y | X=x, Z=z) באמצעות VE:
      - Query:   [Y]
      - Evidence: {'X': x_val, **z_assign}
    מחזיר dict: { y_val -> prob } (מנורמל).
    """
    evidence = {X_name: x_val, **z_assign}
    f = variable_elimination_factor(bn, query_vars=Y, evidence=evidence)
    # הפקטור על Y בלבד, ננרמל זהיר:
    total = sum(f.table.values())
    py = {key[0]: (val / total if total > 0 else 0.0) for key, val in f.table.items()}
    return py



def backdoor_PY_doX(
    bn: BN, Y: str, X_name: str, x_val: Any, Z_vars: List[str]
) -> Dict[Any, float]:
    """
    מממש את הנוסחה:
      P(Y | do(X=x)) = sum_z P(Y | X=x, Z=z) * P(Z=z)
    החישוב תצפיתי בלבד (אין do ברשת; בדיוק לפי הדרישה של המנחה).
    מחזיר dict: { y_val -> prob } (סכום=1).
    """
    # 1) P(Z)
    PZ = VE.compute_joint_PZ_via_VE(bn, Z_vars)

    # 2) סכימה משוקללת על כל z של P(Y | X=x, Z=z) * P(Z=z)
    acc = defaultdict(float)
    # נעבור לפי אותו סדר של Z_vars כדי להתאים את הטופל במפתח של PZ
    for z_assign in _enumerate_Z_assignments(bn, Z_vars):
        z_key = tuple(z_assign[z] for z in Z_vars)
        w = PZ.get(z_key, 0.0)
        if w == 0.0:
            continue
        py = compute_PY_given_X_and_specific_Z(bn, Y, X_name, x_val, z_assign)
        for y_val, p in py.items():
            acc[y_val] += w * p

    # 3) נרמול זהיר (אמור להיות כבר 1 אם PZ תקין):
    total = sum(acc.values())
    if total > 0:
        for k in list(acc.keys()):
            acc[k] /= total

    return dict(acc)


def joint_PZ_via_VE(bn: BN, Z_vars: List[str]) -> Dict[Tuple, float]:
    """
    מחשב את ההתפלגות המשותפת P(Z1,...,Zn) על הרשת הנתונה (תצפיתית או מותערבת),
    באמצעות Variable Elimination. מחזיר dict: (z1,...,zn) -> prob.
    """
    if not Z_vars:
        return {(): 1.0}

    # מפיק פקטור על Z (ללא ראיות)
    fZ = variable_elimination_factor(bn, query_vars=list(Z_vars), evidence={})

    # הפקטור כבר מייצג את P(Z) עד נרמול זניח; נוודא שנרמול מלא:
    # נרמול "על כלום": סכום כל התאים = 1
    total = sum(fZ.table.values())
    if total > 0:
        table = {k: v / total for k, v in fZ.table.items()}
    else:
        table = {k: 0.0 for k in fZ.table.keys()}

    # ממפה מטופל->הסתברות, כאשר סדר המפתח הוא בדיוק סדר Z_vars
    return {key: table[key] for key in table}


# multiple P(Y|(X=x),Z) with P(Z)
def backdoor_do_from_grouped(PY_given_XZ_grouped, PZ):
    """
    קלט:
      - PY_given_XZ_grouped: מפה {(z_tuple): {y: p}}  (הפלט של compute_PY_given_X_and_Z)
      - PZ:                  מפה {(z_tuple): p}       (הפלט של joint_PZ_via_VE)
    פלט:
      - dict {y_val: prob} עבור P(Y | do(X=x))
    """
    acc = defaultdict(float)

    for z_tuple, py_map in PY_given_XZ_grouped.items():
        w = PZ.get(z_tuple, 0.0)    # משקל P(Z=z)
        if w == 0.0:
            continue
        for y_val, p in py_map.items():
            acc[y_val] += w * p     # סכום משוקלל על z

    # נרמול זהיר (אמור לסכום ל-1 אם PZ מנורמלת)
    total = sum(acc.values())
    if total > 0:
        for k in list(acc.keys()):
            acc[k] /= total

    return dict(acc)
