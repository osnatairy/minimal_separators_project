from collections import defaultdict
from typing import Dict, List, Tuple, Iterable


class Factor:
    """
    פקטור הסתברותי על רשימת משתנים self.vars (בסדר קבוע),
    עם טבלה self.table: { tuple(values_by_self.vars): prob }.
    """
    def __init__(self, vars: List[str], table: Dict[Tuple, float]):
        self.vars = list(vars)           # סדר קבוע של משתנים בפקטור
        self.table = dict(table)         # מפה ממפתח טופל (לפי self.vars) לערך הסתברות

    # ---------- פעולות בסיס ----------

    def restrict(self, evidence: Dict[str, object]) -> "Factor":
        """
        מצמצם את הפקטור לפי ראיות: מקבע ערכים למשתנים ידועים ומסיר אותם מהפקטור.
        evidence: dict כמו {'X':1, 'Z2':0}
        """
        keep_positions = []
        new_vars = []
        for i, v in enumerate(self.vars):
            if v not in evidence:
                keep_positions.append(i)
                new_vars.append(v)

        new_table = {}
        for key, p in self.table.items():
            ok = True
            for i, v in enumerate(self.vars):
                if v in evidence and key[i] != evidence[v]:
                    ok = False
                    break
            if not ok:
                continue
            # משאירים רק את ערכי המשתנים שלא הוצבו
            new_key = tuple(key[i] for i in keep_positions) if keep_positions else ()
            new_table[new_key] = new_table.get(new_key, 0.0) + p

        return Factor(new_vars, new_table)

    def multiply(self, other: "Factor") -> "Factor":
        """
        כפל פקטורים: מחבר לפי משתנים משותפים ויוצר טבלה חדשה.
        """
        a_vars = self.vars
        b_vars = other.vars
        shared = [v for v in a_vars if v in b_vars]

        a_pos = {v: i for i, v in enumerate(a_vars)}
        b_pos = {v: i for i, v in enumerate(b_vars)}

        new_vars = a_vars + [v for v in b_vars if v not in a_vars]
        new_table = {}

        for a_key, a_p in self.table.items():
            # שליפת ערכי המשתנים המשותפים מ-a
            shared_vals = {v: a_key[a_pos[v]] for v in shared}
            for b_key, b_p in other.table.items():
                # בדיקת עקביות בערכים המשותפים
                good = True
                for v in shared:
                    if b_key[b_pos[v]] != shared_vals[v]:
                        good = False
                        break
                if not good:
                    continue

                # בניית מפתח חדש: תחילה a_key ואז ערכי b למשתנים שלא ב-a
                new_key = list(a_key)
                for v in b_vars:
                    if v not in a_pos:
                        new_key.append(b_key[b_pos[v]])
                new_key = tuple(new_key)
                new_table[new_key] = new_table.get(new_key, 0.0) + (a_p * b_p)

        return Factor(new_vars, new_table)

    def sum_out(self, var: str) -> "Factor":
        """
        סכימה (חיסור משתנה) על var: מסיר את var ומסכם את ההסתברויות על כל ערכיו.
        """
        if var not in self.vars:
            return self
        pos = self.vars.index(var)
        new_vars = [v for v in self.vars if v != var]
        acc = defaultdict(float)
        for key, p in self.table.items():
            new_key = key[:pos] + key[pos+1:]
            acc[new_key] += p
        return Factor(new_vars, dict(acc))

    def normalize_by(self, group_vars: List[str]) -> "Factor":
        """
        נרמול בתוך קבוצות: לכל קיבוע של group_vars סכום הערכים יהיה 1.
        שימושי מאוד כשצריך לנרמל על Y לכל ערכי Z.
        """
        pos = {v: i for i, v in enumerate(self.vars)}
        group_pos = [pos[v] for v in group_vars]
        totals = defaultdict(float)
        for key, p in self.table.items():
            gkey = tuple(key[i] for i in group_pos) if group_pos else ()
            totals[gkey] += p

        new_table = {}
        for key, p in self.table.items():
            gkey = tuple(key[i] for i in group_pos) if group_pos else ()
            Z = totals[gkey]
            new_table[key] = (p / Z) if Z > 0 else 0.0

        return Factor(self.vars, new_table)

    def as_dict_grouped(self, group_vars: List[str]) -> Dict[Tuple, Dict[object, float]]:
        """
        מייצר מילון: group_vars -> {ערכי משתנה שארית -> הסתברות}
        לדוגמה אם self.vars = ['Y','Z1','Z2'] ו-group_vars=['Z1','Z2'],
        הפלט: {(z1,z2) : {y: prob, ...}, ...}
        """
        pos = {v: i for i, v in enumerate(self.vars)}
        # נזהה את המשתנה היחיד שנותר מקבוצה vars \ group_vars (כאן נרצה שזה יהיה Y)
        rest_vars = [v for v in self.vars if v not in group_vars]
        if len(rest_vars) != 1:
            # אפשר להרחיב לפי צורך; כאן אנו מניחים מקרה של "Y בהינתן Z"
            raise ValueError("as_dict_grouped מניח פקטור על משתנה אחד ועוד קבוצת group_vars.")
        target = rest_vars[0]
        tpos = pos[target]
        gpos = [pos[v] for v in group_vars]

        out = defaultdict(dict)
        for key, p in self.table.items():
            gkey = tuple(key[i] for i in gpos) if gpos else ()
            tval = key[tpos]
            out[gkey][tval] = p
        return dict(out)


# ---------- בניית פקטורים מתוך bn ----------

def build_factors_from_bn(bn: "bn") -> List[Factor]:
    """
    יוצר רשימת פקטורים f(V, parents(V)) מה-bn.
    לכל צומת V: vars = [V] + parents(V) (בסדר קבוע), table לפי ה-CPTs.
    """
    factors = []
    for v in bn.g.nodes():
        parents = bn.parents(v)  # הסדר נקבע ע"י parent_order שב-bn
        vars_order = [v] + parents

        table = {}
        # על כל שורת ה-CPT (מפתח הוא tuple של ערכי הורים לפי הסדר ב-parents)
        if parents:
            for pkey, row in bn.cpts[v].items():
                for val in bn.domains[v]:
                    key = (val,) + tuple(pkey)  # תואם ל-[v] + parents
                    table[key] = row[val]
        else:
            # ללא הורים: pkey ריק () ויש שורה אחת שמגדירה P(v)
            for pkey, row in bn.cpts[v].items():  # יהיה רק pkey = ()
                for val in bn.domains[v]:
                    key = (val,)
                    table[key] = row[val]
        factors.append(Factor(vars_order, table))
    return factors


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


# ---------- עטיפה: P(Y | (X=x), Z) עם נרמול על Y לכל ערכי Z ----------

def compute_PY_given_X_and_Z(
    bn: "bn",
    Y: str,
    Z_vars: Iterable[str],
    evidence_X: Dict[str, object]
) -> Dict[Tuple, Dict[object, float]]:
    """
    מחשב את P(Y | (X=x), Z...) עבור כל שילוב של ערכי Z (בלי להציב להם ערכים).
    - bn: מופע bn
    - Y: שם צומת היעד (למשל 'Y')
    - Z_vars: איטראבל של שמות צמתים Z (למשל ['Z1','Z2'])
    - evidence_X: מילון ראיות שמכיל לפחות {'X': x}

    הפלט: מילון ממופה לפי ערכי Z (טופל) -> מילון התפלגות על Y.
    לדוגמה: { (z1,z2): {y0: p0, y1: p1}, ... } כאשר לכל (z1,z2) הסכום על y הוא 1.
    """
    Z_vars = list(Z_vars)
    if Y in Z_vars:
        raise ValueError("Y לא יכול להופיע גם בקבוצת Z.")
    if not evidence_X:
        raise ValueError("חייבת להינתן ראיה עבור X, למשל {'X': x}.")

    # שלב VE: פקטור על [Y] + Z
    query_vars = [Y] + Z_vars
    f = variable_elimination_factor(bn, query_vars=query_vars, evidence=evidence_X)

    # נרמול על Y לכל קיבוע של Z (Y|Z)
    f_norm = f.normalize_by(group_vars=Z_vars)

    # המרה למבנה: (ערכי Z) -> {y: prob}
    result = f_norm.as_dict_grouped(group_vars=Z_vars)
    return result


# ---------- דוגמת שימוש (לא חובה להריץ כאן) ----------
if __name__ == "__main__":
    # נניח ש- bn הוא מופע ה-bn שלך, מוגדר עם domains, parent_order ו-cpts.
    # לדוגמה:
    # prob_map = compute_PY_given_X_and_Z(bn, Y="Y", Z_vars=["Z1","Z2"], evidence_X={"X": 1})
    # print(prob_map)
    #
    # פלט לדוגמה:
    # {
    #   (0,0): {'0': 0.2, '1': 0.8},
    #   (0,1): {'0': 0.1, '1': 0.9},
    #   (1,0): {'0': 0.25, '1': 0.75},
    #   (1,1): {'0': 0.35, '1': 0.65},
    # }
    pass
