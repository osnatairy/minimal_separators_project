from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# ---------- VE כללי שמחזיר פקטור על query_vars ----------

def variable_elimination_factor(
    bn: "bn",
    query_vars: List[str],
    evidence: Dict[str, object],
    elimination_order: List[str] = None
) -> Factor:
    """
    VE שמחזיר פקטור על query_vars לאחר צמצום ראיות וסכימה על כל שאר המשתנים.
    - query_vars: המשתנים שצריכים להישאר בפלט (למשל ['Y','Z1','Z2'])
    - evidence: ראיות מוצבות (למשל {'X': x})
    """
    # 1) בניית פקטורים מכל ה-CPTs
    factors = build_factors_from_bn(bn)

    # 2) צמצום לפי ראיות (הצבה של ערכים ידועים)
    factors = [f.restrict(evidence) for f in factors]

    # 3) קביעת קבוצת הסרה (hidden): כל מה שלא בשאילתה ולא בראיות
    all_vars = set(bn.g.nodes())
    hidden = [v for v in all_vars if v not in set(query_vars) and v not in set(evidence.keys())]

    # אפשר לשים היוריסטיקה לבחירת סדר; כאן נבחר נאיבי: כפי שהם ברשימה
    order = elimination_order if elimination_order is not None else hidden

    # 4) חיסור משתנים אחד-אחד
    for z in order:
        with_z = [f for f in factors if z in f.vars]
        without_z = [f for f in factors if z not in f.vars]
        if with_z:
            prod = with_z[0]
            for f in with_z[1:]:
                prod = prod.multiply(f)
            summed = prod.sum_out(z)
            factors = without_z + [summed]
        # אם אין פקטורים עם z, פשוט ממשיכים

    # 5) כפל אחרון לקבלת פקטור על query_vars (ואולי תת-קבוצה שלהם)
    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    # בד"כ הפקטור כבר על subset הכולל את query_vars; אם יש עוד משתנים, נוכל לסכום אותם החוצה
    for v in list(result.vars):
        if v not in query_vars:
            result = result.sum_out(v)

    return result

