from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

# ----------------------------
# 3) דיסקרטיזציה (מספרי -> bins)
# ----------------------------

def qcut_safe(series: pd.Series, q: int) -> pd.Categorical:
    """
    דיסקרטיזציה לקוונטילים. אם יש הרבה ערכים זהים (ties) qcut עלול להיכשל,
    אז אנחנו מאפשרים duplicates='drop' (פחות bins בפועל).
    """
    labels = [f"B{i+1}" for i in range(q)]
    try:
        cat = pd.qcut(series, q=q, labels=labels, duplicates="drop")
        return cat.astype("category")
    except Exception:
        # fallback: equal-width (לא אידיאלי אבל עדיף מכלום)
        bins = min(q, series.nunique())
        labels2 = [f"B{i+1}" for i in range(bins)]
        cat = pd.cut(series, bins=bins, labels=labels2)
        return cat.astype("category")



def discretize_columns(
    df: pd.DataFrame,
    nodes: List[str],
    bins: int,
    numeric_policy: str,
    explicit_discrete: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    """
    הופך כל משתנה לרכיב קטגוריאלי (Discrete) כדי שאפשר יהיה לחשב CPT.

    numeric_policy:
      - "auto": אם עמודה מספרית ויש לה הרבה ערכים ייחודיים => binning
      - "always": כל עמודה מספרית => binning
      - "never": לא עושה binning (מסוכן אם יש אלפי ערכים)

    explicit_discrete: רשימה של משתנים שתרצה להשאיר כ-discrete גם אם הם מספריים,
                       למשל 0/1 או ספירות קטנות.

    מחזיר:
      df_cat: דאטה אחרי דיסקרטיזציה והמרה ל-category
      domains: dict var -> domain list (הערכים האפשריים אחרי דיסקרטיזציה)
    """
    df2 = df.copy()
    domains: Dict[str, List[Any]] = {}

    explicit_discrete = set(explicit_discrete or [])

    for var in nodes:
        if var not in df2.columns:
            raise KeyError(f"Graph node '{var}' not found as a column in the data file.")

        s = df2[var]

        # החלטה: האם לעשות binning?
        is_numeric = pd.api.types.is_numeric_dtype(s)
        nunique = s.nunique(dropna=True)

        do_binning = False
        if is_numeric and var not in explicit_discrete:
            if numeric_policy == "always":
                do_binning = True
            elif numeric_policy == "auto":
                # סף פשוט: אם יש "המון" ערכים, זה לא CPT-friendly
                do_binning = (nunique > 30)
            elif numeric_policy == "never":
                do_binning = False
            else:
                raise ValueError("numeric_policy must be one of: auto / always / never")

        if do_binning:
            df2[var] = qcut_safe(s, q=bins)
        else:
            # לא עושה binning: פשוט הופך לקטגוריה
            # אם זה מספרי רציף עם אלפי ערכים – זה עלול לפוצץ את ה-CPT.
            df2[var] = s.astype("category")

        # הגדרת domain (רשימת הערכים האפשריים)
        # שים לב: אנחנו שומרים את הערכים כפי שהם מופיעים ב-category (למשל "B1"... או מספרים/מחרוזות)
        cat = df2[var].astype("category")
        df2[var] = cat
        domains[var] = list(cat.cat.categories)

    return df2, domains

