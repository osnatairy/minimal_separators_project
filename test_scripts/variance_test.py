#https://chatgpt.com/c/69c27fde-ee68-832f-a013-c6ea8335ccca

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Callable
from pathlib import Path
import csv
import math
from bn.bayesian_network import BN
from bn.pgmpy_adapter import bn_to_pgmpy_model
from bn.cpt import generate_bn_binary_logistic
from causal.influence.estimator_bn import (compute_b_and_var_Y_given_AZ, compute_PZ_and_PAZ,compute_f_A_given_Z,
                                           compute_pi_A_given_L,compute_weights_w,compute_m_Z,compute_chi_pi_Z,
                                           _enum_assignments)
from graph.generators import spanning_tree_then_orient
from sem.adjustment_wrapper import run_many_xy
from pipelines.adjust_sets import find_adjustment_sets_for_pair
from causal.policies import static_do_policy
import utils




def debug_compare_asymptotic_variances_static(
    bn: BN,
    Y: str,
    A_name: str,
    a_star: Any,
    Z_vars: List[str],
    value_map: Dict[Any, float] | None = None,
    out_dir: str = "debug_sigma_outputs",
    positivity_tol: float = 0.0,
) -> Dict[str, float]:
    """
    מחשבת ומשווה בין:
      1) asymptotic_variance_for_Z(...) שלך
      2) הנוסחה המפושטת להתערבות סטטית do(A=a_star)

    בנוסף:
      - שומרת ערכי ביניים לקבצי CSV
      - מחזירה מילון סיכום

    נוצרים הקבצים:
      - summary.csv
      - z_level.csv
      - az_level.csv

    הערות:
      - הפונקציה מניחה התערבות סטטית.
      - L_vars = [] כי המדיניות אינה תלויה ב-L.
      - אם יש positivity violation עבור a_star, תיזרק שגיאה.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, infer = bn_to_pgmpy_model(bn)

    # -----------------------------
    # 0) policy סטטית
    # -----------------------------
    def static_policy_fn(a_val, L_assign):
        return 1.0 if a_val == a_star else 0.0

    A_domain = list(bn.domains[A_name])

    # -----------------------------
    # 1) רכיבי בסיס
    # -----------------------------
    b_map, var_map = compute_b_and_var_Y_given_AZ(
        bn=bn,
        infer=infer,
        Y=Y,
        A_name=A_name,
        Z_vars=Z_vars,
        value_map=value_map,
    )

    PZ, PAZ = compute_PZ_and_PAZ(
        bn=bn,
        infer=infer,
        A_name=A_name,
        Z_vars=Z_vars,
    )

    f_map = compute_f_A_given_Z(PZ, PAZ)

    pi_map = compute_pi_A_given_L(
        bn=bn,
        A_name=A_name,
        Z_vars=Z_vars,
        L_vars=[],
        policy_fn=static_policy_fn,
    )

    w_map = compute_weights_w(pi_map, f_map)

    m_map = compute_m_Z(
        bn=bn,
        A_name=A_name,
        Z_vars=Z_vars,
        L_vars=[],
        b_map=b_map,
        policy_fn=static_policy_fn,
    )

    chi_general = compute_chi_pi_Z(PZ, m_map)

    # -----------------------------
    # 2) שונות לפי הפונקציה שלך
    #    sigma_general = term1 + term2
    # -----------------------------
    term1_general = 0.0
    az_rows = []

    for z_assign in _enum_assignments(bn, Z_vars):
        z_tuple = tuple(z_assign[z] for z in Z_vars)

        for a_val in A_domain:
            key = (a_val, z_tuple)
            p_az = float(PAZ.get(key, 0.0))
            p_z = float(PZ.get(z_tuple, 0.0))
            f_az = float(f_map.get(key, 0.0))
            pi_az = float(pi_map.get(key, 0.0))
            w_az = float(w_map.get(key, 0.0))
            b_az = float(b_map.get(key, 0.0))
            varY_az = float(var_map.get(key, 0.0))

            contrib_general_term1 = (w_az ** 2) * varY_az * p_az
            term1_general += contrib_general_term1

            az_rows.append({
                "a_val": a_val,
                "z_tuple": repr(z_tuple),
                "PZ": p_z,
                "PAZ": p_az,
                "f_A_given_Z": f_az,
                "pi_A_given_L": pi_az,
                "w_pi_over_f": w_az,
                "b_EY_given_AZ": b_az,
                "varY_given_AZ": varY_az,
                "general_term1_contribution": contrib_general_term1,
            })

    term2_general = 0.0
    z_rows = []

    for z_assign in _enum_assignments(bn, Z_vars):
        z_tuple = tuple(z_assign[z] for z in Z_vars)

        p_z = float(PZ.get(z_tuple, 0.0))
        m_z = float(m_map.get(z_tuple, 0.0))
        diff = m_z - chi_general
        contrib_general_term2 = (diff ** 2) * p_z
        term2_general += contrib_general_term2

        # עבור הנוסחה המפושטת:
        b_star_z = float(b_map[(a_star, z_tuple)])
        var_star_z = float(var_map[(a_star, z_tuple)])
        p_a_star_given_z = float(f_map[(a_star, z_tuple)])

        if p_z > 0.0 and p_a_star_given_z <= positivity_tol:
            raise ValueError(
                f"Positivity violated for z={z_tuple}: "
                f"P({A_name}={a_star} | Z=z)={p_a_star_given_z}"
            )

        simplified_var_part = 0.0 if p_z <= 0.0 else (var_star_z / p_a_star_given_z)
        simplified_bias_part = (b_star_z - chi_general) ** 2
        simplified_total_inside_brackets = simplified_var_part + simplified_bias_part
        simplified_contribution = p_z * simplified_total_inside_brackets

        z_rows.append({
            "z_tuple": repr(z_tuple),
            "PZ": p_z,
            "m_z_general": m_z,
            "chi_general": chi_general,
            "general_diff_m_minus_chi": diff,
            "general_term2_contribution": contrib_general_term2,
            "b_a_star_z": b_star_z,
            "varY_a_star_z": var_star_z,
            "P_Aeqastar_given_Z": p_a_star_given_z,
            "simplified_var_part_var_over_ps": simplified_var_part,
            "simplified_bias_part": simplified_bias_part,
            "simplified_inside_brackets": simplified_total_inside_brackets,
            "simplified_total_contribution": simplified_contribution,
        })

    sigma_general = term1_general + term2_general

    # -----------------------------
    # 3) שונות לפי הנוסחה המפושטת
    # -----------------------------
    chi_simplified = 0.0
    for z_tuple, pz in PZ.items():
        chi_simplified += float(pz) * float(b_map[(a_star, z_tuple)])

    sigma_simplified = 0.0
    for z_tuple, pz in PZ.items():
        if pz <= 0.0:
            continue

        p_a_given_z = float(f_map[(a_star, z_tuple)])
        if p_a_given_z <= positivity_tol:
            raise ValueError(
                f"Positivity violated for z={z_tuple}: "
                f"P({A_name}={a_star} | Z=z)={p_a_given_z}"
            )

        b_val = float(b_map[(a_star, z_tuple)])
        var_y = float(var_map[(a_star, z_tuple)])

        sigma_simplified += float(pz) * (
            var_y / p_a_given_z +
            (b_val - chi_simplified) ** 2
        )

    # -----------------------------
    # 4) קובץ סיכום
    # -----------------------------
    summary_rows = [{
        "Y": Y,
        "A_name": A_name,
        "a_star": a_star,
        "Z_vars": repr(list(Z_vars)),
        "chi_general": chi_general,
        "chi_simplified": chi_simplified,
        "chi_abs_diff": abs(chi_general - chi_simplified),
        "term1_general": term1_general,
        "term2_general": term2_general,
        "sigma_general": sigma_general,
        "sigma_simplified": sigma_simplified,
        "sigma_abs_diff": abs(sigma_general - sigma_simplified),
        "n_z_assignments": len(PZ),
        "n_az_assignments": len(PAZ),
    }]

    # -----------------------------
    # 5) כתיבה ל-CSV
    # -----------------------------
    def write_csv(path: Path, rows: List[Dict[str, Any]]):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_path / "summary.csv", summary_rows)
    write_csv(out_path / "z_level.csv", z_rows)
    write_csv(out_path / "az_level.csv", az_rows)

    return {
        "chi_general": float(chi_general),
        "chi_simplified": float(chi_simplified),
        "chi_abs_diff": float(abs(chi_general - chi_simplified)),
        "sigma_general": float(sigma_general),
        "sigma_simplified": float(sigma_simplified),
        "sigma_abs_diff": float(abs(sigma_general - sigma_simplified)),
        "term1_general": float(term1_general),
        "term2_general": float(term2_general),
        "out_dir": str(out_path),
    }


if __name__ == "__main__":
    print("start test variance")

    seed = 0

    seed_graph, seed_params = utils.split_seeds(seed)

    G = spanning_tree_then_orient(
        n=20,
        prob_edge=0.25,
        k_roots=20,
        node_prefix="V",
        seed=seed_graph
    )

    bn = generate_bn_binary_logistic(G, seed=seed_params)
    # 2) Run over many (X,Y) pairs and find adjustment sets
    pairs = run_many_xy(
        bn.g,
        mode="reachable",
        sample_k=20,
        seed=seed)
    print("pairs:", pairs)

    R = list(bn.g.nodes())
    I = []

    for X, Y in pairs:
        H, Z_sets = find_adjustment_sets_for_pair(bn.g, X, Y,"smallminimalseps", R=R, I=I)
        L_vars = []  # נניח שהמדיניות תלויה ב-G,H (אפשר גם L_vars=[])
        # מדיניות סטטית do(I=1)
        policy_fn = static_do_policy(a_star=1)

        for Z in Z_sets:

            result = debug_compare_asymptotic_variances_static(
                bn=bn,
                Y=Y,
                A_name=X,
                a_star=policy_fn,
                Z_vars=Z_sets,
                value_map={0: 0.0, 1: 1.0},
                out_dir="debug_sigma_Z1_Z2"
            )

    print(result)