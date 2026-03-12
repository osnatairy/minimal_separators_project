
import numpy as np
from typing import Any, List, Dict, Tuple, Optional

from sem.linear_sem import sem_to_B_Omega, covariance_from_B_Omega



# ---------- helpers: indexing + conditional variance ----------

def _index_map(var_names: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(var_names)}

def _submatrix(Sigma: np.ndarray, rows: List[int], cols: List[int]) -> np.ndarray:
    return Sigma[np.ix_(rows, cols)]


def conditional_variance_from_cov(
    Sigma: np.ndarray,
    target: str,
    given: List[str],
    var_names: List[str],
    ridge: float = 0.0,
) -> float:
    """
    Computes Var(target | given) from the joint covariance matrix Sigma.

    This equals the residual variance of the linear regression:
        target ~ given

    Formula:
        Var(A|B) = Var(A) - Cov(A,B) Var(B)^{-1} Cov(B,A)

    Parameters
    ----------
    Sigma : (p,p) ndarray
        Covariance matrix over var_names.
    target : str
        Name of the target variable A.
    given : list[str]
        Names of conditioning variables B.
    var_names : list[str]
        The ordering of variables in Sigma.
    ridge : float
        Optional small diagonal regularization added to Var(B) before inversion,
        useful if Var(B) is nearly singular.

    Returns
    -------
    float
        Conditional variance Var(target | given).
    """
    idx = _index_map(var_names)

    a = idx[target]
    var_a = float(Sigma[a, a])

    if not given:
        return var_a

    b_idx = [idx[g] for g in given]

    Sigma_ab = _submatrix(Sigma, [a], b_idx)          # shape (1, k)
    Sigma_ba = _submatrix(Sigma, b_idx, [a])          # shape (k, 1)
    Sigma_bb = _submatrix(Sigma, b_idx, b_idx)        # shape (k, k)

    if ridge > 0:
        Sigma_bb = Sigma_bb + ridge * np.eye(Sigma_bb.shape[0])

    #inv_Sigma_bb = np.linalg.inv(Sigma_bb)
    #cond_var = var_a - float(Sigma_ab @ inv_Sigma_bb @ Sigma_ba)

    middle = np.linalg.solve(Sigma_bb, Sigma_ba)
    cond_var = var_a - (Sigma_ab @ middle).item()

    # numerical safety
    return float(max(cond_var, 0.0))

# ---------- Convenience: compute Sigma from your LinearSEM ----------

def sigma_from_sem(sem) -> Tuple[List[str], np.ndarray]:
    """
    Uses your existing functions sem_to_B_Omega and covariance_from_B_Omega.
    Returns (var_names, Sigma).
    """
    # assumes you imported these from your generator file
    nodes, B, Omega = sem_to_B_Omega(sem)
    Sigma = covariance_from_B_Omega(B, Omega)
    return nodes, Sigma


# ---------- Henckel aVar for single X, single Y ----------

def avar_henckel_single_xy(
    Sigma: np.ndarray,
    X: str,
    Y: str,
    Z: List[str],
    var_names: List[str],
    ridge: float = 0.0,
) -> float:
    """
    Computes Henckel et al. asymptotic variance for OLS adjustment estimator
    of total effect of X on Y given adjustment set Z, in the single-X single-Y case:

        aVar = Var(Y | X, Z) / Var(X | Z)

    Parameters
    ----------
    Sigma : ndarray
        Covariance matrix over var_names.
    X, Y : str
        Treatment and outcome variable names (singletons).
    Z : list[str]
        Adjustment set variable names.
    var_names : list[str]
        Ordering used in Sigma.
    ridge : float
        Optional regularization for matrix inversions.

    Returns
    -------
    float
        aVar value (asymptotic variance, i.e., Var(sqrt(n)(hat-beta - beta)) ).
        For sample size n, approximate Var(hat-beta) ≈ aVar / n.
    """
    # Numerator: residual variance of Y after regressing on (X + Z)
    var_y_given_xz = conditional_variance_from_cov(
        Sigma, target=Y, given=[X] + list(Z), var_names=var_names, ridge=ridge
    )

    # Denominator: residual variance of X after regressing on Z
    var_x_given_z = conditional_variance_from_cov(
        Sigma, target=X, given=list(Z), var_names=var_names, ridge=ridge
    )

    if var_x_given_z <= 0:
        raise ValueError(
            f"Var({X} | Z) computed as {var_x_given_z}. "
            "This can happen if X is (almost) perfectly explained by Z "
            "or due to numerical issues. Try ridge>0 or check Sigma."
        )

    return float(var_y_given_xz / var_x_given_z)


# ---------- Example usage with your LinearSEM ----------

def example_compute_avar(sem, X: str, Y: str, Z: List[str], ridge: float = 1e-10):
    var_names, Sigma = sigma_from_sem(sem)
    aVar = avar_henckel_single_xy(Sigma, X=X, Y=Y, Z=Z, var_names=var_names, ridge=ridge)
    print("aVar =", aVar)
    return aVar



from typing import List, Dict, Tuple
from sem import variance as vcalc  # זה variance.py שלך

def x_drain(sem, X: str, Z: List[str], ridge: float = 1e-10) -> float:
    """
    X-Drain(Z) = 1 - Var(X|Z)/Var(X)
    """
    var_names, Sigma = vcalc.sigma_from_sem(sem)
    idx = {v: i for i, v in enumerate(var_names)}

    var_x = float(Sigma[idx[X], idx[X]])
    var_x_given_z = vcalc.conditional_variance_from_cov(
        Sigma, target=X, given=list(Z), var_names=var_names, ridge=ridge
    )

    return 1.0 - (var_x_given_z / var_x)

def x_drain_two_sets(
    sem,
    X: str,
    Z1: List[str],
    Z2: List[str],
    ridge: float = 1e-10
) -> Dict[str, float]:
    """
    Returns X-Drain for Z1 and Z2.
    """
    return {
        "X-Drain(Z1)": x_drain(sem, X, Z1, ridge=ridge),
        "X-Drain(Z2)": x_drain(sem, X, Z2, ridge=ridge),
    }



def condition_number_of_Z(
    sem: Any,
    Z,
    ridge: float = 1e-12,
    warn_threshold: float = 1e8,
    method: int | None = None,
) -> Dict[str, float | bool | int]:
    """
    Compute cond( Cov(Z) ) using np.linalg.cond and flag if it's "very high".

    Parameters
    ----------
    sem : LinearSEM
        Your SEM object.
    Z : iterable[str] | set/frozenset | list[...] (supports nested like [frozenset(...)] )
        Adjustment set variables.
    ridge : float
        Small diagonal jitter added to Cov(Z) to avoid singular matrices.
    warn_threshold : float
        Flag as ill-conditioned if cond >= warn_threshold.
    method : int | None
        Passed to np.linalg.cond (None -> 2-norm default). Common: 2, 1, np.inf.

    Returns
    -------
    dict with:
      - cond_Z: condition number
      - ill_conditioned: bool
      - size_Z: int
    """
    # normalize Z (same idea as before)
    if Z is None:
        Z_list: List[str] = []
    elif isinstance(Z, list) and len(Z) == 1 and isinstance(Z[0], (set, frozenset, tuple, list)):
        Z_list = list(Z[0])
    elif isinstance(Z, (set, frozenset, tuple)):
        Z_list = list(Z)
    else:
        Z_list = list(Z)

    if len(Z_list) == 0:
        return {"cond_Z": 1.0, "ill_conditioned": False, "size_Z": 0}

    var_names, Sigma = vcalc.sigma_from_sem(sem)
    idx = {v: i for i, v in enumerate(var_names)}

    z_idx = [idx[z] for z in Z_list]
    cov_Z = Sigma[np.ix_(z_idx, z_idx)].astype(float)

    # stabilize numerically
    cov_Z = cov_Z + ridge * np.eye(cov_Z.shape[0])

    cond_val = float(np.linalg.cond(cov_Z, p=method))
    return {
        "cond_Z": cond_val,
        "ill_conditioned": cond_val >= warn_threshold,
        "size_Z": int(cov_Z.shape[0]),
    }


def compare_separators_stability(sem, Z_in, Z_out, warn_threshold=1e8):
    """
    משווה את היציבות הנומרית (Condition Number) בין מפריד קרוב ל-Y (Z_in)
    לבין מפריד רחוק יותר (Z_out).
    """
    # חישוב עבור המפריד הקרוב ל-Y
    res_in = condition_number_of_Z(sem, Z_in, warn_threshold=warn_threshold)

    # חישוב עבור המפריד הרחוק מ-Y
    res_out = condition_number_of_Z(sem, Z_out, warn_threshold=warn_threshold)

    comparison = {
        "Z_in": {
            "cond": res_in["cond_Z"],
            "is_ill": res_in["ill_conditioned"]
        },
        "Z_out": {
            "cond": res_out["cond_Z"],
            "is_ill": res_out["ill_conditioned"]
        },
        "ratio_in_out": res_in["cond_Z"] / res_out["cond_Z"],
        "potential_numerical_issue": res_in["ill_conditioned"] or res_out["ill_conditioned"]
    }

    return comparison