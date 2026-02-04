from __future__ import annotations

from typing import Any, Dict, Tuple


# BN + טעינה מ-JSON
from bn.bayesian_network import BN
from i_o.json_loader import load_bn_from_json  # אם יצרת קובץ טעינה; אחרת עדכני לנתיב האמיתי

# גרף -> H
from pipelines.adjust_sets import preprocess_H

# separators (מומלץ להחליף find_seperators)
from graph.separators import find_seperators

# validation for dowhy seperator
from validation.dowhy_check import test_Z_with_dowhy

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs

#influance function calculation
from causal.influence.estimator_bn import asymptotic_variance_for_Z
from causal.policies import static_do_policy

import causal.grafical_criteria_for_O as gcO

if __name__ == '__main__':

    #JSON = "BN_DATA/synthetic_bn_many_separators_cpt_07.json"   # your example file  #bn_12_nodes
    JSON = "BN_DATA/bn_amazon_redshift.json"

    # 0) Load DAG from JSON
    bn = load_bn_from_json(JSON, BNClass=BN)

    Iset = []
    R = bn.g.nodes
    X = ["num_joins"]  # treatments of interest
    Y = ["execution_time"]  # outcomes of interest
    K = 6


    H, x, y = preprocess_H(bn, X, Y, R, I=Iset)
    #Z_sets =  find_seperators(H,x, y, "SmallMinimalSeps", K)
    Z_sets = find_seperators(H, x, y, which="SmallMinimalSeps", K=K)


    O = gcO.find_optimal_adjustment_set(bn.g, X[0], Y[0])
    print("optimal(X,Y) is:" +str(O))


    # check if all the Z_sets are an adjustment set using DoWhy.
    test_Z_with_dowhy(bn.g,X[0],Y[0],Z_sets)

    #check for hass diagram with the adjustment set
    forward, reverse = cy_components_for_sets(H, Y[0], Z_sets)
    print(forward, reverse)
    # get Hass graph for the Z - the adjustment sets
    res = hasse_from_cy_results(forward, reverse)
    print("***************************************")
    print(res)
    print("***************************************")
    curr_result = find_containment_pairs(res)


    ############################### for calculate the variance of y|do(x) ############################
    # remove the edges enter to X. - calculate the new marginal distribution of X
    #G_1 = bn.remove_incoming_edges_do(x)

    #get all possible evidence for X.
    #x_values = G_1.domain(x)
    ##################################################################################################

    results: Dict[Tuple[str, ...], Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]]] = {}

    ######################### for calculate the variance throgh observations #########################
    L_vars = []  # נניח שהמדיניות תלויה ב-G,H (אפשר גם L_vars=[])

    # מדיניות סטטית do(I=1)
    policy_fn = static_do_policy(a_star=1)

    # df = gen.load_observations_csv("BN_DATA/observations/obs_bn_12_nodes-old.csv")

    results: Dict[Tuple[str, ...], float] = {}
    for Z in Z_sets:
        if len(Z) == 0:
            continue
        Z_key = tuple(sorted(Z))

    ###########################   for calculate the variance throgh observations #########################
    #     sigma2_hat, nuisance = ive.estimate_asymptotic_variance_pi_Z(
    #         df,
    #         bn=bn,
    #         Y=x,
    #         A=y,
    #         Z_vars=Z,
    #         L_vars=L_vars,
    #         policy_fn=policy_fn,
    #         alpha=0.0,  # אפשר 0.5/1.0 אם יש אפסים
    #         clip_w_max=None,  # אפשר למשל 50.0 במדגם קטן
    #     )
    #     results[Z_key] = sigma2_hat
    #
    # # 5) הדפסה / בחירת Z עם השונות הכי קטנה
    # best_Z = min(results, key=results.get)
    # print("Best Z:", best_Z, "sigma^2_hat:", results[best_Z])
    #


        sigma2 = asymptotic_variance_for_Z(
            bn, y, x, Z, L_vars, policy_fn, None
        )
        results[Z_key] = sigma2

    #results = evaluate_variance_of_Z_sets_for_y_do_x(Z_sets ....) #need to sent more variables

    print(results)
    # else calculate P(Y|X
