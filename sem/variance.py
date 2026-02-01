
import numpy as np
from typing import List, Dict, Tuple, Optional

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
    cond_var = var_a - float(Sigma_ab @ middle)

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