from typing import Any, Dict, Callable


# ---------- 4. π(A|L) – פונקציית ההתערבות ----------

# כאן אנחנו מניחים שהמשתמש נותן פונקציית מדיניות:
#   policy_fn(a_val, L_assign_dict) -> π(a|L)
# למשל עבור do(A = a_star) סטטי:
#   π(a|L) = 1 אם a == a_star, אחרת 0.

PolicyFn = Callable[[Any, Dict[str, Any]], float]

def static_do_policy(a_star: Any) -> PolicyFn:
    """
    מחזיר פונקציית מדיניות סטטית עבור do(A=a_star):
      π(a|L) = 1 אם a == a_star, אחרת 0.
    """
    def _pi(a_val, L_assign):
        return 1.0 if a_val == a_star else 0.0
    return _pi
