from typing import Dict, List, Tuple, Callable, Any

from minimal_separators_project.bn.bayesian_network import BN         # רשת בייסיאנית שלך :contentReference[oaicite:2]{index=2}
from minimal_separators_project.causal.policies import PolicyFn

# ---------- עזר: איטרציה על השמותות של Z ----------

def _enum_assignments(bn: BN, vars_subset: List[str]):
    """
    מחזיר גנרטור של כל הצירופים האפשריים של המשתנים ברשימת vars_subset,
    בפורמט של dict: {var: value}.
    משתמש ב- _enumerate_assignments של ה-bn אם קיים.
    """
    if hasattr(bn, "_enumerate_assignments"):
        yield from bn._enumerate_assignments(vars_subset)
    else:
        # fallback קטן אם אי פעם תרצי להשתמש בזה על bn אחר
        from itertools import product
        vars_subset = list(vars_subset)
        domains = [bn.domains[v] for v in vars_subset]
        for values in product(*domains):
            yield dict(zip(vars_subset, values))



# ---------- 1. b(A,Z;P) ו-Var(Y | A,Z) ----------

def compute_b_and_var_Y_given_AZ(
    bn: BN,
    Y: str,
    A_name: str,
    Z_vars: List[str],
    value_map: Dict[Any, float] | None = None,
) -> Tuple[Dict[Tuple[Any, Tuple], float], Dict[Tuple[Any, Tuple], float]]:
    """
    מחשבת את:
      b(a,z)   = E[Y | A=a, Z=z]
      varY(a,z) = Var(Y | A=a, Z=z)

    מחזירה שני מילונים:
      b_map[(a, z_tuple)]   -> מספר
      var_map[(a, z_tuple)] -> מספר

    זהו החלק b(A,Z;P) והחלק Var(Y|A,Z) הדרוש בנוסחת השונות.
    """

    # מיפוי ערכי Y לערך מספרי (למשל 0/1)
    if value_map is None:
        domY = set(bn.domains[Y])
        # במקרה בינארי סטנדרטי
        if domY == {0, 1}:
            value_map = {0: 0.0, 1: 1.0}
        else:
            raise ValueError("Y אינו בינארי. ספק/י value_map שממפה כל ערך Y למספר.")

    def y_num(y):
        return float(value_map[y])

    b_map: Dict[Tuple[Any, Tuple], float] = {}
    var_map: Dict[Tuple[Any, Tuple], float] = {}

    A_domain = bn.domains[A_name]

    for z_assign in _enum_assignments(bn, Z_vars):
        z_tuple = tuple(z_assign[z] for z in Z_vars)

        for a_val in A_domain:
            evidence = {A_name: a_val, **z_assign}

            # התפלגות Y | A=a, Z=z
            probs = {}
            for y in bn.domains[Y]:
                p = bn.conditional({Y: y}, evidence)
                probs[y] = float(p)

            # E[Y | A=a,Z=z]
            EY = sum(y_num(y) * probs[y] for y in probs)
            EY2 = sum((y_num(y) ** 2) * probs[y] for y in probs)
            varY = EY2 - EY ** 2

            key = (a_val, z_tuple)
            b_map[key] = EY
            var_map[key] = max(varY, 0.0)  # הגנה מספרית קטנה

    return b_map, var_map



# ---------- 2. P(Z) ו- P(A,Z) ----------

def compute_PZ_and_PAZ(
    bn: BN,
    A_name: str,
    Z_vars: List[str],
) -> Tuple[Dict[Tuple, float], Dict[Tuple[Any, Tuple], float]]:
    """
    מחשבת את:
      P(Z=z)
      P(A=a, Z=z)

    מחזירה:
      PZ[z_tuple]            -> P(Z=z)
      PAZ[(a_val, z_tuple)]  -> P(A=a, Z=z)
    """
    PZ: Dict[Tuple, float] = {}
    PAZ: Dict[Tuple[Any, Tuple], float] = {}

    A_domain = bn.domains[A_name]

    for z_assign in _enum_assignments(bn, Z_vars):
        z_tuple = tuple(z_assign[z] for z in Z_vars)

        # P(Z=z)
        Pz = bn.marginal_prob(z_assign)
        PZ[z_tuple] = float(Pz)

        # P(A=a, Z=z) = P(Z=z, A=a)
        for a_val in A_domain:
            fixed = {A_name: a_val, **z_assign}
            p_az = bn.marginal_prob(fixed)
            PAZ[(a_val, z_tuple)] = float(p_az)

    return PZ, PAZ


# ---------- 3. f(A|Z) – Propensity score ----------

def compute_f_A_given_Z(
    PZ: Dict[Tuple, float],
    PAZ: Dict[Tuple[Any, Tuple], float],
) -> Dict[Tuple[Any, Tuple], float]:
    """
    מחשבת את f(A|Z) מתוך P(A,Z) ו-P(Z):
      f(a|z) = P(A=a, Z=z) / P(Z=z)

    מחזירה:
      f_map[(a_val, z_tuple)] -> f(a|z).
    """
    f_map: Dict[Tuple[Any, Tuple], float] = {}
    for (a_val, z_tuple), p_az in PAZ.items():
        pz = PZ[z_tuple]
        if pz <= 0.0:
            f_map[(a_val, z_tuple)] = 0.0
        else:
            f_map[(a_val, z_tuple)] = p_az / pz
    return f_map


# ---------- 4. π(A|L) – פונקציית ההתערבות ----------

# כאן אנחנו מניחים שהמשתמש נותן פונקציית מדיניות:
#   policy_fn(a_val, L_assign_dict) -> π(a|L)
# למשל עבור do(A = a_star) סטטי:
#   π(a|L) = 1 אם a == a_star, אחרת 0.
# פונקצית הפוליסי נמצאת בקובץ policies

def compute_pi_A_given_L(
    bn: BN,
    A_name: str,
    Z_vars: List[str],
    L_vars: List[str],
    policy_fn: PolicyFn,
) -> Dict[Tuple[Any, Tuple], float]:
    """
    מחשבת π(a|L(z)) לכל זוג (a,z).

    קלט:
      - bn: רשת
      - A_name: שם הטיפול (A)
      - Z_vars: רשימת שמות Z (שכוללת את L)
      - L_vars: רשימת שמות L ⊆ Z
      - policy_fn: פונקציה π(a, L_assign) -> הסתברות

    פלט:
      pi_map[(a_val, z_tuple)] -> π(a|L(z))
    """
    pi_map: Dict[Tuple[Any, Tuple], float] = {}
    A_domain = bn.domains[A_name]

    for z_assign in _enum_assignments(bn, Z_vars):
        z_tuple = tuple(z_assign[z] for z in Z_vars)
        L_assign = {ℓ: z_assign[ℓ] for ℓ in L_vars}

        for a_val in A_domain:
            pi = policy_fn(a_val, L_assign)
            pi_map[(a_val, z_tuple)] = float(pi)

    return pi_map


# ---------- 5. המשקל w = π/f ----------

def compute_weights_w(
    pi_map: Dict[Tuple[Any, Tuple], float],
    f_map: Dict[Tuple[Any, Tuple], float],
) -> Dict[Tuple[Any, Tuple], float]:
    """
    מחשבת את המשקל:
      w(a,z) = π(a|L(z)) / f(a|z)
    """
    w_map: Dict[Tuple[Any, Tuple], float] = {}
    for key, pi_val in pi_map.items():
        f_val = f_map.get(key, 0.0)
        if f_val <= 0.0:
            w_map[key] = 0.0
        else:
            w_map[key] = pi_val / f_val
    return w_map


# ---------- 6. m(z) = E_{πZ*}[b(A,Z)|Z=z] ----------

def compute_m_Z(
    bn: BN,
    A_name: str,
    Z_vars: List[str],
    L_vars: List[str],
    b_map: Dict[Tuple[Any, Tuple], float],
    policy_fn: PolicyFn,
) -> Dict[Tuple, float]:
    """
    מחשבת:
      m(z) = Σ_a b(a,z) * π(a|L(z))

    מחזירה:
      m_map[z_tuple] -> m(z).
    """
    m_map: Dict[Tuple, float] = {}
    A_domain = bn.domains[A_name]

    for z_assign in _enum_assignments(bn, Z_vars):
        z_tuple = tuple(z_assign[z] for z in Z_vars)
        L_assign = {ℓ: z_assign[ℓ] for ℓ in L_vars}

        m_val = 0.0
        for a_val in A_domain:
            b_val = b_map[(a_val, z_tuple)]
            pi_val = policy_fn(a_val, L_assign)
            m_val += b_val * pi_val

        m_map[z_tuple] = m_val

    return m_map



# ---------- 7. χ_{π,Z}(P;G) = E[m(Z)] ----------

def compute_chi_pi_Z(
    PZ: Dict[Tuple, float],
    m_map: Dict[Tuple, float],
) -> float:
    """
    מחשבת את הגודל הסיבתי:
      χ_{π,Z}(P;G) = E[ m(Z) ] = Σ_z m(z) * P(Z=z)
    """
    chi = 0.0
    for z_tuple, m_val in m_map.items():
        pz = PZ.get(z_tuple, 0.0)
        chi += m_val * pz
    return chi


# ---------- 8. השונות האסימפטוטית σ^2_{π,Z}(P) ----------

def asymptotic_variance_for_Z(
    bn: BN,
    Y: str,
    A_name: str,
    Z_vars: List[str],
    L_vars: List[str],
    policy_fn: PolicyFn,
    value_map: Dict[Any, float] | None = None,
) -> float:
    """
    פונקציה ראשית שמחשבת את השונות האסימפטוטית σ^2_{π,Z}(P) עבור קבוצת התאמה Z נתונה:

      σ^2 = E_P[ ψ^2 ] =
             Σ_{a,z} w(a,z)^2 Var(Y|A=a,Z=z) P(A=a,Z=z)
           + Σ_z    (m(z) - χ)^2 P(Z=z)

    קלט:
      - bn: רשת בייסיאנית דיסקרטית
      - Y:   שם משתנה התוצאה
      - A_name: שם משתנה הטיפול A (למשל אותו X במאמר)
      - Z_vars: רשימת שמות בקבוצת ההתאמה Z
      - L_vars: תת־קבוצה של Z שעליהם תלויה המדיניות π(A|L)
      - policy_fn: פונקציית מדיניות π(a, L_assign)
      - value_map: מיפוי של ערכי Y לערכים מספריים (אם Y לא {0,1})

    פלט:
      - מספר ממשי: σ^2_{π,Z}(P)
    """

    # 1) b(a,z) ו-varY(a,z)
    b_map, var_map = compute_b_and_var_Y_given_AZ(
        bn, Y, A_name, Z_vars, value_map=value_map
    )

    # 2) P(Z) ו-P(A,Z)
    PZ, PAZ = compute_PZ_and_PAZ(bn, A_name, Z_vars)

    # 3) f(A|Z)
    f_map = compute_f_A_given_Z(PZ, PAZ)

    # 4) π(A|L)
    pi_map = compute_pi_A_given_L(bn, A_name, Z_vars, L_vars, policy_fn)

    # 5) w = π/f
    w_map = compute_weights_w(pi_map, f_map)

    # 6) m(z)
    m_map = compute_m_Z(bn, A_name, Z_vars, L_vars, b_map, policy_fn)

    # 7) χ_{π,Z}(P;G)
    chi = compute_chi_pi_Z(PZ, m_map)

    # 8) חישוב σ^2 לפי הנוסחה המפורקת
    #    σ^2 = Σ_{a,z} w^2 varY P(A,Z) + Σ_z (m(z)-χ)^2 P(Z=z)

    term1 = 0.0
    for key, p_az in PAZ.items():
        a_val, z_tuple = key
        w = w_map[key]
        varY = var_map[key]
        term1 += (w ** 2) * varY * p_az

    term2 = 0.0
    for z_tuple, m_val in m_map.items():
        diff = m_val - chi
        pz = PZ[z_tuple]
        term2 += (diff ** 2) * pz

    sigma2 = term1 + term2
    return float(max(sigma2, 0.0))  # הגנה קטנה מפני שליליות נומרית



# ---------- 9. פונקציה על פני כמה קבוצות Z ----------

def asymptotic_variance_over_Z_sets(
    bn: BN,
    Y: str,
    A_name: str,
    Z_sets: List[List[str]],
    L_vars: List[str],
    policy_fn: PolicyFn,
    value_map: Dict[Any, float] | None = None,
) -> Dict[Tuple[str, ...], float]:
    """
    מחשבת σ^2_{π,Z}(P) עבור כמה קבוצות התאמה Z שונות.

    מחזירה:
      results[tuple(sorted(Z))] = σ^2 עבור אותה קבוצה.
    """
    results: Dict[Tuple[str, ...], float] = {}
    for Z in Z_sets:
        Z_key = tuple(sorted(Z))
        sigma2 = asymptotic_variance_for_Z(
            bn, Y, A_name, Z, L_vars, policy_fn, value_map=value_map
        )
        results[Z_key] = sigma2
    return results
