
def canon_Z(Z_vars: List[str]) -> Tuple[str, ...]:
    """מפתח קנוני ל־Z: טופל שמות ממוין, יציב והאשאבל."""
    return tuple(sorted(Z_vars))

# --- מפתחות קנוניים, אם עוד לא הוגדרו כאן ---
def canon_assign(assign: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """הופך השמה dict (ל-X) לטופל ממוין של זוגות (שם,ערך) כדי לשמש כמפתח יציב."""
    return tuple(sorted(assign.items()))


def store_result(results, z_key, x_key, pY_do: Dict[Any, float], E: float, Var: float):

    if z_key not in results:
        results[z_key] = {}
    results[z_key][x_key] = {
        "pY_do": dict(pY_do),
        "E": float(E),
        "Var": float(Var),

    }

def attach_summary_to_results(results, summary, key_name="__summary__"):
    """
    מוסיף/מעדכן סיכום לתוך results[Z_key][__summary__] בלי לדרוס את מה שקיים.
    """
    Z_key   = summary["Z_key"]
    payload = {
        "X_name":          summary["X_name"],
        "E_Var_do_X":      summary["E_Var_do_over_X"],  # המספר היחיד
        "weights":         summary["weights"],
        "count_X":         summary["count_X"],
    }

    block = results.setdefault(Z_key, {})           # אם אין בלוק כזה—ייווצר
    block.setdefault(key_name, {})                  # אם אין __summary__—ייווצר ריק
    block[key_name].update(payload)                 # מוסיף/מעדכן רק את השדות האלו



def factor_marginal_X(bn, X: str, evidence: dict | None = None) -> VE.Factor:
    """
    מחזיר Factor על [X] שבו table[(x,)] = P(X=x | evidence).
    אם evidence=None → P(X=x) התצפיתי.
    """
    print("good for a single X. if there are multiple variables of X we need different function")
    evidence = evidence or {}
    f =VE.variable_elimination_factor(bn, query_vars=[X], evidence=evidence)
    # לוודא נרמול (אמור להיות כבר):
    s = sum(f.table.values())
    if s > 0:
        f = VE.Factor(f.vars, {k: v/s for k, v in f.table.items()})
    return f



def summarize_E_Var_do_over_X_for_Z_with_fX(
    results: Dict[
        Tuple[str, ...],                                        # Z_key = tuple of Z names
        Dict[Union[Tuple[Tuple[str, Any], ...], Any], Dict[str, Any]]  # X_key (קנוני או ערך גלם) -> metrics
    ],
    Z_vars: List[str],
    fX,                                 # Factor על ['X'] שבו table[(x,)] = P(X=x)
    missing: str = "error"              # "error" | "skip" — מה לעשות אם חסר משקל ל-X_key
) -> Dict[str, Any]:
    """
    מחשב:  E_pi[ Var(Y | do(X=x)) ]  כאשר pi(x) = P(X=x) נלקח מתוך הפקטור fX.

    פרמטרים:
    ---------
    results : dict
        המבנה שבו שמרת לכל Z ולכל X:
        results[Z_key][X_key] = { "pY_do": {...}, "E": float, "Var": float, ... }
        הערה: X_key יכול להיות קנוני (כמו (('I',1),)) או 'ערך גלם' (למשל 1).
    Z_vars : list[str]
        שמות משתני Z (למשל ["G","H"]).
    fX : Factor
        פקטור על ['X'] שבו table[(x,)] = P(X=x) (מוחזר מ-factor_marginal_X).
    missing : "error" | "skip"
        אם יש ב-results ערך X שאין לו משקל בפקטור — האם לזרוק שגיאה או לדלג.

    פלט:
    ----
    dict עם:
      {
        "Z_key": Tuple[str, ...],
        "X_name": str,
        "E_Var_do_over_X": float,           # התוחלת המשוקללת של השונויות
        "weights": Dict[X_key, float],      # המשקלים שבהם השתמשנו (במפתח קנוני)
        "count_X": int                      # כמה ערכי X נספרו מה-results
      }
    """
    # 1) הפקת משקלים pi(x) מהפקטור fX
    if not hasattr(fX, "vars") or fX.vars != [fX.vars[0]]:
        # הגנה רכה: מצפים לפקטור על ['X'] בלבד
        pass
    X_name = fX.vars[0]                     # שם משתנה הטיפול (למשל "I")
    weights = { canon_assign({X_name: key[0]}): float(p) for key, p in fX.table.items() }

    # נרמול עדין (ליתר ביטחון)
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}

    # 2) שליפת הבלוק של Z
    Z_key = tuple(sorted(Z_vars))
    inner = results.get(Z_key, {})
    if not inner:
        raise ValueError(f"No entries in results for Z={Z_vars}. ודאי שמילאת results עבור Z זה.")

    # 3) ממוצע משוקלל של Var(Y | do(X=x))
    evar = 0.0
    used_w = {}  # לשקיפות
    counted = 0

    for x_key_in, metrics in inner.items():
        # תמיכה גם במפתח קנוני וגם ב"ערך גלם" (למשל 0/1)
        if isinstance(x_key_in, tuple) and x_key_in and isinstance(x_key_in[0], tuple):
            x_key = x_key_in                              # כבר קנוני
        else:
            x_key = canon_assign({X_name: x_key_in})      # ממירים לערך קנוני

        if x_key not in weights:
            if missing == "skip":
                continue
            raise KeyError(f"Missing weight for X assignment {dict(x_key)} (אין pi(x)).")

        if "Var" not in metrics or metrics["Var"] is None:
            raise ValueError(f"Missing Var for Z={Z_vars}, X={dict(x_key)}")

        w = weights[x_key]
        evar += float(metrics["Var"]) * w
        used_w[x_key] = w
        counted += 1

    return {
        "Z_key": Z_key,
        "X_name": X_name,
        "E_Var_do_over_X": float(evar),
        "weights": used_w,
        "count_X": counted
    }
