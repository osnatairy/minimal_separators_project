from collections import defaultdict
from typing import Iterable, Any, Dict, Tuple

# BN
from minimal_separators_project.bn.bayesian_network import BN

# backdoor / causal
from minimal_separators_project.causal.backdoor import (
    joint_PZ_via_VE,
    _enumerate_Z_assignments,
    compute_PY_given_X_and_specific_Z,
)

# stats – moments
from minimal_separators_project.stats.moments import (
    expectation_from_acc,
    variance_from_acc,
)

# stats – results / summarize
from minimal_separators_project.stats.summarize import (
    store_result,
    canon_Z,
    canon_assign,
    factor_marginal_X,
    summarize_E_Var_do_over_X_for_Z_with_fX,
    attach_summary_to_results,
)




# calculate the variance of y|do(x)
#calculate_variance_y_do_x - old function name
def evaluate_variance_of_Z_sets_for_y_do_x(
    bn: BN,
    *,
    Y: str,
    X_name: str,
    x_values: Iterable[Any],
    Z_sets: Iterable[Iterable[str]],
) -> Dict[Tuple[str, ...], Dict]:

    results: Dict[Tuple[str, ...], Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]]] = {}

    if len(Z_sets) > 0:
        print("Adjustment sets (v                                                                                                                                                                                                                                                                                   ariable names):")
        for Z in Z_sets:
            Z_key = canon_Z(Z)
            print("  Z =", Z)

            #calculate P(Z) - OVER THE ORIGINAL bn.
            PZ = joint_PZ_via_VE(bn, Z)

            # iterate x's domain.
            for x_value in x_values:

                # 2) Weighted scheme for all z of P(Y | X=x, Z=z) * P(Z=z)
                acc = defaultdict(float)

                for z_assign in _enumerate_Z_assignments(bn, Z):
                    z_key = tuple(z_assign[z] for z in Z)
                    w = PZ.get(z_key, 0.0)
                    if w == 0.0:
                        continue
                    py = compute_PY_given_X_and_specific_Z(bn, Y, x, x_value, z_assign)
                    for y_val, p in py.items():
                        acc[y_val] += w * p

                # 3) Careful normalization (should already be 1 if PZ is correct):
                total = sum(acc.values())
                if total > 0:
                    for k in list(acc.keys()):
                        acc[k] /= total


                print(acc)
                expectation_x = expectation_from_acc(acc)
                variance_x = variance_from_acc(acc)
                store_result(results,Z_key,x_value,acc,expectation_x,variance_x)

            #חישוב תוחלת השונות עבור X
            #X_name = X[0]  # כאשר יש רק משתנה יחיד של  X
            fX = factor_marginal_X(bn, X_name)  # פקטור על [X], כלומר P(X)
            print(fX.vars)  # ['X']
            print(fX.table)  # { (0,): P(X=0), (1,): P(X=1), ... }



            x_key = canon_assign({X_name: x_value})
            #store_result(results, Z_key, x_key, acc, expectation_x, variance_x)
            summ = summarize_E_Var_do_over_X_for_Z_with_fX(results, Z_vars=Z, fX=fX)
            print("Z =", Z, "  E[Var(Y|do(X))] =", summ["E_Var_do_over_X"])
            attach_summary_to_results(results, summ)


            # # 2) מאוחר יותר, מעדכנים את ערך ה-do בפועל (למשל X=1):
                # G_1.set_do_value(x, x_value, inplace=True)
                # prob_map = VE.compute_PY_given_X_and_Z(bn, Y=y, Z_vars=Z, evidence_X={"X": x_value})
                #
                # # multiple P(Y | (X=x), Z) with P(Z)
                # PY_do = backdoor_do_from_grouped(prob_map, PZ)
                # print(PY_do)
