from concurrent.futures import ProcessPoolExecutor, as_completed
import utils as utils

# analysis for the HASS diagram
from analysis.adjustment_hasse import  cy_components_for_sets, hasse_from_cy_results, find_containment_pairs,extract_separator_containment_pairs, frozenset_to_str

from pipelines.adjust_sets import find_adjustment_sets_for_pair
from analysis.bucket_sep import bucket_separators_by_cy_layers, bucket_variance_statistics,bucket_separators_by_y_connectivity,bucket_y_component

#influance function calculation
from causal.influence.estimator_bn import asymptotic_variance_over_Z_sets

from causal.policies import static_do_policy

from sem.variance import compute_avar_many_Z


def process_all_xy_pairs_parallel(
    model,
    seed,
    pairs,
    R,
    I,
    worker_fn,
    test_mode=False,
):
    """
    Wrapper כללי להרצת זוגות (X,Y) במקביל.

    model      - אובייקט המודל (bn / sem)
    seed       - seed נוכחי
    pairs      - רשימת זוגות (X,Y)
    R, I       - פרמטרים קיימים שלך
    worker_fn  - פונקציה שמעבדת זוג XY אחד
    test_mode  - מועבר ל-worker
    """
    num_workers = utils.choose_num_workers(len(pairs))
    print(f"num_pairs = {len(pairs)}, chosen workers = {num_workers}")

    tasks = [
        (model, seed, X, Y, R, I, test_mode)
        for X, Y in pairs
    ]

    if num_workers == 1:
        return [worker_fn(task) for task in tasks]

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_fn, task) for task in tasks]

        for future in as_completed(futures):
            results.append(future.result())

    return results


def process_xy_pair_bn(task):
    """
    מחשבת את כל העבודה עבור זוג XY אחד:
    - מציאת קבוצות ההתאמה
    - חישוב Hasse
    - חישוב שונות לכל Z
    - חישובי bucket

    מחזירה dict עם כל מה שהתהליך הראשי צריך.
    """
    (
        bn,
        seed,
        X,
        Y,
        R,
        I,
        test_mode,
    ) = task


    H, Z_sets,time, closest = find_adjustment_sets_for_pair(
        bn.g, X, Y, "smallminimalseps", R=R, I=I,get_closest_seps=True
    )

    forward, reverse = cy_components_for_sets(H, Y, Z_sets)
    res = hasse_from_cy_results(forward, reverse)

    num_nodes = bn.g.number_of_nodes()
    num_edges = bn.g.number_of_edges()

    h_num_nodes = H.number_of_nodes()
    h_num_edges = H.number_of_edges()

    result = {
        "seed": seed,
        "X": X,
        "Y": Y,
        "H": H,
        "Z_sets": Z_sets,
        "res": res,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "h_num_nodes": h_num_nodes,
        "h_num_edges": h_num_edges,
        "results1": {},
        "bucket_input": None,
        "test_mode_result": None,
    }

    if test_mode:
        curr_result = find_containment_pairs(res)
        sep_result = len(res["hasse_edges"])

        result["test_mode_result"] = {
            "curr_result": curr_result,
            "sep_result": sep_result,
        }
        return result

    # מצב חישוב שונות
    L_vars = []
    policy_fn = static_do_policy(a_star=1)


    results1 = asymptotic_variance_over_Z_sets(
        bn, Y, X, Z_sets, L_vars, policy_fn, None
    )

    result["results1"] = results1
    result["closest"] = closest

    # נתונים לחישובי buckets
    buckets = bucket_separators_by_cy_layers(H, Y, Z_sets)
    buckets1 = bucket_separators_by_y_connectivity(H, Y, Z_sets)
    y_component_info = bucket_y_component(buckets1, res["Z_to_component"])

    result["bucket_input"] = {
        "buckets": buckets,
        "buckets1": buckets1,
        "y_component_info": y_component_info,
    }

    return result


def process_xy_pair_sem(task):
    (
        sem,
        seed,
        X,
        Y,
        R,
        I,
        test_mode,
    ) = task

    H, Z_sets,time, closest = find_adjustment_sets_for_pair(
        sem.g, X, Y, "smallminimalseps", R=R, I=I, get_closest_seps=True
    )

    Z_sets = [Z for Z in Z_sets if len(Z) >= 1]

    forward, reverse = cy_components_for_sets(H, Y, Z_sets)
    res = hasse_from_cy_results(forward, reverse)

    num_nodes = sem.g.number_of_nodes()
    num_edges = sem.g.number_of_edges()
    h_num_nodes = H.number_of_nodes()
    h_num_edges = H.number_of_edges()

    result = {
        "seed": seed,
        "X": X,
        "Y": Y,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "h_num_nodes": h_num_nodes,
        "h_num_edges": h_num_edges,
        "Z_sets": Z_sets,
        "res": res,
        "closest": closest,
        "results1": {},
        "bucket_input": None,
        "test_mode_result": None,
    }

    if test_mode:
        curr_result = find_containment_pairs(res)
        sep_result = len(res["hasse_edges"])

        result["test_mode_result"] = {
            "curr_result": curr_result,
            "sep_result": sep_result,
        }
        return result

    results1 = compute_avar_many_Z(sem, X=X, Y=Y, Z_sets=Z_sets)

    buckets = bucket_separators_by_cy_layers(H, Y, Z_sets)
    buckets1 = bucket_separators_by_y_connectivity(H, Y, Z_sets)
    y_component_info = bucket_y_component(buckets1, res["Z_to_component"])

    result["results1"] = results1
    result["bucket_input"] = {
        "buckets": buckets,
        "buckets1": buckets1,
        "y_component_info": y_component_info,
    }

    return result