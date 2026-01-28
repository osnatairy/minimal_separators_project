
def expectation_from_acc(
        acc: Dict[Any, float],
        value_map: Union[Dict[Any, float], Callable[[Any], float], None] = None,
        strict: bool = False,
        tol: float = 1e-9,
) -> float:
    """
    מחשבת את התוחלת E[Y] מתוך התפלגות acc: {outcome -> prob}.

    פרמטרים:
    - acc: dict שממפה ערכי Y להסתברות שלהם (לא חייב לסכם בדיוק ל-1; תתבצע נרמול עדין).
    - value_map:
        * None: אם כל הערכים הם מספריים (int/float/bool) – ייעשה שימוש ישיר בערכים.
        * dict: מיפוי מערכים לקטגוריות → ערך מספרי (למשל {"low":0,"mid":1,"high":2}).
        * callable: פונקציה שמקבלת outcome ומחזירה ערך מספרי.
    - strict: אם True – תיזרק שגיאה אם סכום ההסתברויות לא ~1; אם False – נרמול אוטומטי.
    - tol: טולרנס להשוואת סכום להסתברות 1.

    מחזירה:
    - float: התוחלת E[Y].
    """
    if not acc:
        raise ValueError("acc ריק – אין על מה לחשב תוחלת.")

    # אימות הסתברויות
    total_p = float(sum(acc.values()))
    if total_p <= 0:
        raise ValueError("סכום ההסתברויות ב-acc צריך להיות חיובי.")
    if abs(total_p - 1.0) > tol:
        if strict:
            raise ValueError(f"סכום ההסתברויות הוא {total_p}, שאינו קרוב ל-1 (strict=True).")
        # נרמול עדין
        acc_norm = {k: float(v) / total_p for k, v in acc.items()}
    else:
        acc_norm = {k: float(v) for k, v in acc.items()}

    # פונקציית המרה לערך מספרי
    def _to_num(y):
        if value_map is None:
            # אם כל הערכים מספריים – השתמש ישירות
            if all(isinstance(k, (int, float, bool)) for k in acc_norm.keys()):
                return float(y)
            else:
                raise ValueError(
                    "Y אינו מספרי. ספק value_map (dict או פונקציה) שממפה כל outcome לערך מספרי."
                )
        elif callable(value_map):
            return float(value_map(y))
        else:  # dict
            if y not in value_map:
                raise KeyError(f"לא נמצא מיפוי ב-value_map עבור outcome={y!r}")
            return float(value_map[y])

    # חישוב התוחלת
    expectation = 0.0
    for y, p in acc_norm.items():
        expectation += _to_num(y) * p
    return expectation



def variance_from_acc(
    acc: Dict[Any, float],
    value_map: Union[Dict[Any, float], Callable[[Any], float], None] = None,
    strict: bool = False,
    tol: float = 1e-9,
) -> float:
    """
    מחשבת Var(Y) מתוך התפלגות acc: {outcome -> prob}.
    אם Y אינו מספרי, ספק/י value_map (dict או פונקציה) שממפה כל outcome לערך מספרי.

    acc לא חייב לסכם בדיוק ל-1; תתבצע נרמול עדין (אלא אם strict=True).
    """

    if not acc:
        raise ValueError("acc ריק – אין על מה לחשב שונות.")

    total_p = float(sum(acc.values()))
    if total_p <= 0:
        raise ValueError("סכום ההסתברויות ב-acc צריך להיות חיובי.")
    if abs(total_p - 1.0) > tol:
        if strict:
            raise ValueError(f"סכום ההסתברויות הוא {total_p}, שאינו קרוב ל-1 (strict=True).")
        acc_norm = {k: float(v) / total_p for k, v in acc.items()}
    else:
        acc_norm = {k: float(v) for k, v in acc.items()}

    # פונקציה להמרת outcomes לערכים מספריים
    def to_num(y):
        if value_map is None:
            # אם כל הערכים עצמם מספריים (כולל bool), נשתמש בהם ישירות
            if all(isinstance(k, (int, float, bool)) for k in acc_norm.keys()):
                return float(y)
            else:
                raise ValueError("Y אינו מספרי. ספק/י value_map (dict או פונקציה) למיפוי מספרי.")
        elif callable(value_map):
            return float(value_map(y))
        else:  # dict
            if y not in value_map:
                raise KeyError(f"לא נמצא מיפוי ב-value_map עבור outcome={y!r}")
            return float(value_map[y])

    # חישוב E[Y] ו-E[Y^2] בלולאה אחת
    EY = 0.0
    EY2 = 0.0
    for y, p in acc_norm.items():
        v = to_num(y)
        EY  += v * p
        EY2 += (v * v) * p

    var = EY2 - EY * EY
    # יציבות נומרית: לחסום שליליות זעירה
    return max(var, 0.0)

