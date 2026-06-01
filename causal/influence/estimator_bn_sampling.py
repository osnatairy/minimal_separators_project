from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from bn.bayesian_network import BN
from causal.policies import PolicyFn


def _default_value_map(bn: BN, Y: str, value_map):
    if value_map is not None:
        return value_map
    if set(bn.domains[Y]) == {0, 1}:
        return {0: 0.0, 1: 1.0}
    raise ValueError("Y אינו בינארי. ספקי value_map.")


def asymptotic_variance_for_Z_sampling(
    bn: BN,
    Y: str,
    A_name: str,
    Z_vars: List[str],
    L_vars: List[str],
    policy_fn: PolicyFn,
    value_map: Dict[Any, float] | None = None,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> float:
    value_map = _default_value_map(bn, Y, value_map)

    df = bn.sample(n_samples=n_samples, seed=seed).copy()
    df["_Y_num"] = df[Y].map(value_map).astype(float)

    group_cols = [A_name] + list(Z_vars)
    z_cols = list(Z_vars)

    # b(a,z) = E[Y | A=a,Z=z]
    # varY(a,z) = Var(Y | A=a,Z=z)
    stats_az = (
        df.groupby(group_cols, dropna=False)["_Y_num"]
        .agg(["mean", "var", "size"])
        .reset_index()
    )
    stats_az["var"] = stats_az["var"].fillna(0.0)

    # P(A,Z)
    stats_az["p_az"] = stats_az["size"] / len(df)

    # P(Z)
    stats_z = (
        df.groupby(z_cols, dropna=False)
        .size()
        .reset_index(name="z_size")
    )
    stats_z["p_z"] = stats_z["z_size"] / len(df)

    # merge כדי לקבל P(Z) ליד כל (A,Z)
    merged = stats_az.merge(stats_z, on=z_cols, how="left")

    # f(a|z) = P(A,Z)/P(Z)
    merged["f"] = merged["p_az"] / merged["p_z"]

    # pi(a|L)
    def compute_pi(row):
        l_assign = {l: row[l] for l in L_vars}
        return float(policy_fn(row[A_name], l_assign))

    merged["pi"] = merged.apply(compute_pi, axis=1)

    # w = pi / f
    merged["w"] = np.where(merged["f"] > 0, merged["pi"] / merged["f"], 0.0)

    # term1 = sum_{a,z} w^2 Var(Y|A,Z) P(A,Z)
    term1 = float(((merged["w"] ** 2) * merged["var"] * merged["p_az"]).sum())

    # m(z) = sum_a pi(a|L(z)) b(a,z)
    merged["pi_b"] = merged["pi"] * merged["mean"]

    m_df = (
        merged.groupby(z_cols, dropna=False)["pi_b"]
        .sum()
        .reset_index(name="m")
    )

    m_df = m_df.merge(stats_z[z_cols + ["p_z"]], on=z_cols, how="left")

    # chi = E[m(Z)]
    chi = float((m_df["m"] * m_df["p_z"]).sum())

    # term2 = sum_z (m(z)-chi)^2 P(Z=z)
    term2 = float((((m_df["m"] - chi) ** 2) * m_df["p_z"]).sum())

    return max(term1 + term2, 0.0)


def asymptotic_variance_for_Z_from_samples(
    df_samples: pd.DataFrame,
    bn: BN,
    Y: str,
    A_name: str,
    Z_vars: List[str],
    L_vars: List[str],
    policy_fn: PolicyFn,
    value_map: Dict[Any, float] | None = None,
) -> float:
    """
    Computes asymptotic variance using an existing samples DataFrame.
    This avoids calling bn.sample(...) separately for every Z.
    """
    value_map = _default_value_map(bn, Y, value_map)

    df = df_samples.copy()
    df["_Y_num"] = df[Y].map(value_map).astype(float)

    group_cols = [A_name] + list(Z_vars)
    z_cols = list(Z_vars)

    stats_az = (
        df.groupby(group_cols, dropna=False)["_Y_num"]
        .agg(["mean", "var", "size"])
        .reset_index()
    )
    stats_az["var"] = stats_az["var"].fillna(0.0)
    stats_az["p_az"] = stats_az["size"] / len(df)

    stats_z = (
        df.groupby(z_cols, dropna=False)
        .size()
        .reset_index(name="z_size")
    )
    stats_z["p_z"] = stats_z["z_size"] / len(df)

    merged = stats_az.merge(stats_z, on=z_cols, how="left")
    merged["f"] = merged["p_az"] / merged["p_z"]

    def compute_pi(row):
        l_assign = {l: row[l] for l in L_vars}
        return float(policy_fn(row[A_name], l_assign))

    merged["pi"] = merged.apply(compute_pi, axis=1)
    merged["w"] = np.where(merged["f"] > 0, merged["pi"] / merged["f"], 0.0)

    term1 = float(((merged["w"] ** 2) * merged["var"] * merged["p_az"]).sum())

    merged["pi_b"] = merged["pi"] * merged["mean"]

    m_df = (
        merged.groupby(z_cols, dropna=False)["pi_b"]
        .sum()
        .reset_index(name="m")
    )

    m_df = m_df.merge(stats_z[z_cols + ["p_z"]], on=z_cols, how="left")

    chi = float((m_df["m"] * m_df["p_z"]).sum())

    term2 = float((((m_df["m"] - chi) ** 2) * m_df["p_z"]).sum())

    return max(term1 + term2, 0.0)