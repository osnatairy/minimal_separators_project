import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

#load data and create graph G
G = nx.drawing.nx_pydot.read_dot("../BN_DATA/amazon_redshift.dot")

with open("../BN_DATA/amazon_redshift_dataset.pkl", "rb") as f:
    data = pickle.load(f)

#define experiments
experiments =  {("num_joins", "planning_time"): [
      ["num_tables", "result_cache_hit"],
      ["query_template"]
  ],

  ("num_tables", "execution_time"): [
      ["num_columns", "num_joins", "result_cache_hit"],
      ["query_template"]
  ],

  ("result_cache_hit", "elapsed_time"): [
      ["num_columns", "num_joins", "num_tables"],
      ["query_template"]
  ],}

#helper functions
def estimate_propensity(Z, A, eps=0.01, C=0.1, n_splits=5, random_state=0):
    """
    Estimate propensity scores e(Z) = P(A=1 | Z) using cross-fitted logistic regression.

    Parameters
    ----------
    Z : pd.DataFrame
        Covariate matrix used for adjustment.
    A : pd.Series or np.ndarray
        Binary treatment indicator (0/1).
    eps : float, optional (default=0.01)
        Clipping threshold to enforce overlap: e_hat in [eps, 1-eps].
    C : float, optional (default=0.1)
        Inverse regularization strength for logistic regression.
    n_splits : int, optional (default=5)
        Number of folds for cross-fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    e_hat : np.ndarray, shape (n,)
        Cross-fitted and clipped propensity score estimates.
    """
    n = len(A)
    e_hat = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    for train_idx, test_idx in kf.split(Z):
        model = LogisticRegression(
            max_iter=1000,
            penalty="l2",
            C=C,
            solver="lbfgs"
        )
        #print(Z)
        model.fit(Z.iloc[train_idx], A.iloc[train_idx])
        e_hat[test_idx] = model.predict_proba(Z.iloc[test_idx])[:, 1]

    return np.clip(e_hat, eps, 1 - eps)

def ess_by_group(A, e_hat):
    """
    Compute Effective Sample Size (ESS) for treated and control groups
    based on inverse-probability weights.

    Parameters
    ----------
    A : pd.Series or np.ndarray
        Binary treatment indicator.
    e_hat : np.ndarray
        Estimated propensity scores.

    Returns
    -------
    ess : dict
        Dictionary with ESS for treatment (1) and control (0).
    """
    ess = {}

    for a in [0, 1]:
        if a == 1:
            w = 1 / e_hat[A == 1]
        else:
            w = 1 / (1 - e_hat[A == 0])

        ess[a] = (w.sum() ** 2) / np.sum(w ** 2)

    return ess

def min_ess(ess_dict):
    return min(ess_dict.values())

def evaluate_propensity(Z,A):
  """
  Convenience function to diagnose propensity score quality:
  overlap, tails, ESS, and histogram.

  Parameters
  ----------
  Z : pd.DataFrame
      Covariates.
  A : pd.Series or np.ndarray
      Treatment indicator.

  Returns
  -------
  None
  """
  e_hat = estimate_propensity(Z, A)
  print(f"min propensity : {min(e_hat)}, max propensity: {max(e_hat)}, q0.05-q0.95:{np.quantile(e_hat,0.05), np.quantile(e_hat,0.95)}")
  print(f"ess: {ess_by_group(A,e_hat)}")

  plt.hist(e_hat)
  plt.title("propensity distribution")
  plt.show()

def estimate_mu(Z, Y, A, a, n_splits=5, random_state=0):
    """
    Estimate outcome regression mu_a(Z) = E[Y | A=a, Z]
    using cross-fitted models.

    Parameters
    ----------
    Z : pd.DataFrame
        Covariates.
    Y : pd.Series or np.ndarray
        Outcome variable.
    A : pd.Series or np.ndarray
        Treatment indicator.
    a : int {0,1}
        Treatment level for which to estimate the outcome model.
    n_splits : int, optional
        Number of folds for cross-fitting.
    random_state : int, optional
        Random seed.

    Returns
    -------
    mu_hat : np.ndarray, shape (n,)
        Cross-fitted predictions of mu_a(Z).
    """
    n = len(Y)
    mu_hat = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(Z):
        train_mask = (A.iloc[train_idx] == a)
        Z_train_a = Z.iloc[train_idx][train_mask]
        Y_train_a = Y.iloc[train_idx][train_mask]

        #model = LinearRegression()
        model = HistGradientBoostingRegressor(
          max_iter=100,
          learning_rate=0.1,
          max_depth=5,
          random_state=random_state
        )
        model.fit(Z_train_a, Y_train_a)

        mu_hat[test_idx] = model.predict(Z.iloc[test_idx])

    return mu_hat


def aipw_ate(Y, A, e_hat, mu1_hat, mu0_hat):
    """
    Compute the Augmented Inverse Probability Weighted (AIPW)
    estimate of the Average Treatment Effect (ATE).

    Parameters
    ----------
    Y : np.ndarray
        Observed outcomes.
    A : np.ndarray
        Treatment indicator.
    e_hat : np.ndarray
        Propensity score estimates.
    mu1_hat : np.ndarray
        Estimated E[Y | A=1, Z].
    mu0_hat : np.ndarray
        Estimated E[Y | A=0, Z].

    Returns
    -------
    float
        AIPW estimate of the ATE.
    """
    term1 = A * (Y - mu1_hat) / e_hat
    term0 = (1 - A) * (Y - mu0_hat) / (1 - e_hat)
    return np.mean(term1 - term0 + mu1_hat - mu0_hat)

def aipw_variance(Y, A, e_hat, mu1_hat, mu0_hat):
    """
    Estimate the asymptotic variance of the AIPW ATE estimator
    using the empirical influence function.

    Parameters
    ----------
    Y, A, e_hat, mu1_hat, mu0_hat : np.ndarray
        Same inputs as aipw_ate.

    Returns
    -------
    float
        Estimated variance of the ATE.
    """
    psi_hat = aipw_ate(Y, A, e_hat, mu1_hat, mu0_hat)

    phi = (
        A * (Y - mu1_hat) / e_hat
        - (1 - A) * (Y - mu0_hat) / (1 - e_hat)
        + mu1_hat - mu0_hat
        - psi_hat
    )

    return np.var(phi, ddof=1) / len(Y)


def diagnostic_mu_performance(Y, A, mu1_hat, mu0_hat):
    """
    Diagnostics for outcome regression models:
    fit quality and residual imbalance.

    Parameters
    ----------
    Y : np.ndarray
        Observed outcomes.
    A : np.ndarray
        Treatment indicator.
    mu1_hat : np.ndarray
        Predicted outcomes under treatment.
    mu0_hat : np.ndarray
        Predicted outcomes under control.

    Returns
    -------
    None
    """
    mu_factual = np.where(A == 1, mu1_hat, mu0_hat)
    residuals = Y - mu_factual

    results = {}

    results['mse_treated'] = mean_squared_error(Y[A==1], mu1_hat[A==1])
    results['mse_control'] = mean_squared_error(Y[A==0], mu0_hat[A==0])

    res_treated_mean = np.mean(residuals[A==1])
    res_control_mean = np.mean(residuals[A==0])
    results['residual_diff'] = res_treated_mean - res_control_mean

    r2_treated = r2_score(Y[A==1], mu1_hat[A==1])
    r2_control = r2_score(Y[A==0], mu0_hat[A==0])


    print(f"--- Outcome Model (mu) Diagnostics ---")
    print(f"R² (Treated Model): {r2_treated:.4f}")
    print(f"R² (Control Model): {r2_control:.4f}")
    print(f"MSE (Treated): {results['mse_treated']:.4f}")
    print(f"MSE (Control): {results['mse_control']:.4f}")
    print(f"Residual Bias (Treated - Control): {results['residual_diff']:.4f}")

def evaluate_adjustment_set_full(
    Y, Z, A,
    eps=0.01, C=0.1
):
    """
    Full evaluation of an adjustment set using AIPW:
    - Propensity estimation + ESS
    - Outcome regression diagnostics
    - ATE and variance estimation

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable.
    Z : pd.DataFrame
        Adjustment covariates.
    A : np.ndarray
        Treatment indicator.
    eps : float, optional
        Propensity clipping threshold.
    C : float, optional
        Regularization strength for propensity model.

    Returns
    -------
    dict
        Dictionary with ATE, variance, and ESS diagnostics.
    """
    # nuisance A
    e_hat = estimate_propensity(Z, A, eps=eps)
    ess = ess_by_group(A, e_hat)

    # nuisance B
    mu1_hat = estimate_mu(Z, Y, A, a=1)
    mu0_hat = estimate_mu(Z, Y, A, a=0)

    diagnostic_mu_performance(Y,A,mu1_hat,mu0_hat)
    ate = aipw_ate(Y, A, e_hat, mu1_hat, mu0_hat)
    var = aipw_variance(Y, A, e_hat, mu1_hat, mu0_hat)

    return {
        "ATE": ate,
        "Variance": var,
        "ESS_min": min_ess(ess),
        "ESS_treated": ess[1],
        "ESS_control": ess[0]
    }

    #return results

#experiment 1
A = "num_joins"
Y = "planning_time"
Z_near = ["num_tables", "result_cache_hit"]
Z_far = ["query_template"]

data[A+'_binary'] = data[A].apply(lambda x: x % 2 == 0 ).astype(int)

evaluate_propensity(data[Z_near],data[A+'_binary'])
evaluate_propensity(data[Z_far],data[A+'_binary'])

print(evaluate_adjustment_set_full(data[Y],data[Z_near],data[A+'_binary']))
print(evaluate_adjustment_set_full(data[Y],data[Z_far],data[A+'_binary']))

#experiment 2
A = "num_tables"
Y = "execution_time"
Z_near = ["num_columns", "num_joins", "result_cache_hit"]
Z_far = ["query_template"]

data[A+'_binary'] = data[A].apply(lambda x: x % 2 == 0 ).astype(int)

evaluate_propensity(data[Z_near],data[A+'_binary'])
evaluate_propensity(data[Z_far],data[A+'_binary'])

print(evaluate_adjustment_set_full(data[Y],data[Z_near],data[A+'_binary']))
print(evaluate_adjustment_set_full(data[Y],data[Z_far],data[A+'_binary']))

#experiment 3
A = "result_cache_hit"
Y = "elapsed_time"
Z_near = ["num_columns", "num_joins", "num_tables"]
Z_far = ["query_template"]

#data[A+'_binary'] = data[A].apply(lambda x: x % 2 == 0 ).astype(int)

evaluate_propensity(data[Z_near],data[A])
evaluate_propensity(data[Z_far],data[A])

print(evaluate_adjustment_set_full(data[Y],data[Z_near],data[A]))
print(evaluate_adjustment_set_full(data[Y],data[Z_far],data[A]))


