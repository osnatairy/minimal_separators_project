from itertools import product
import numpy as np

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def bn_to_pgmpy_model(bn):
    """
    ממיר אובייקט BN שלך למודל pgmpy מסוג DiscreteBayesianNetwork,
    כולל כל ה-CPDs, ומחזיר (model, infer).

    למה צריך model + infer?
    - model: מייצג את הגרף והפרמטרים (CPDs)
    - infer: אובייקט שמבצע שאילתות כמו P(Y | evidence) באמצעות Variable Elimination
    """
    # 1) בונים מודל pgmpy עם אותם edges
    model = DiscreteBayesianNetwork(list(bn.g.edges()))

    # 2) מוסיפים גם nodes (למקרה שיש isolated nodes)
    model.add_nodes_from(list(bn.g.nodes()))

    cpds = []

    for var in bn.g.nodes():
        # הורים בסדר קבוע: בדיוק כמו שאתה משתמשת כדי לאנדקס CPT
        parents = bn.parents(var)  # משתמש ב-parent_order אם קיים
        var_states = bn.domains[var]
        var_card = len(var_states)

        # state_names מאפשר ל-pgmpy "להכיר" את הערכים המקוריים שלך (0/1 וכו')
        # חשוב: הסדר כאן חייב להתאים לסדר השורות שנייצר במטריצת values.
        state_names = {var: list(var_states)}

        if parents:
            parent_states = [bn.domains[p] for p in parents]
            evidence_card = [len(bn.domains[p]) for p in parents]
            for p, states in zip(parents, parent_states):
                state_names[p] = list(states)

            # 3) בונים את מטריצת values:
            # shape = (var_card, prod(evidence_card))
            num_cols = int(np.prod(evidence_card))
            values = np.zeros((var_card, num_cols), dtype=float)

            # העמודות ב-TabularCPD מסודרות לפי cartesian product של מצבי ההורים,
            # בסדר ההורים כפי שמופיע ב-evidence (כאן: parents).
            # לכן נבנה מיפוי: parent_tuple -> column_index
            parent_assignments = list(product(*parent_states))
            col_index = {pt: i for i, pt in enumerate(parent_assignments)}

            # עכשיו נמלא את הטבלה מתוך bn.cpts[var]:
            # bn.cpts[var] הוא: { parent_tuple -> {value: prob} }
            for pt, row in bn.cpts[var].items():
                j = col_index[pt]
                for state_i, x_val in enumerate(var_states):
                    values[state_i, j] = float(row[x_val])

            cpd = TabularCPD(
                variable=var,
                variable_card=var_card,
                values=values,
                evidence=parents,
                evidence_card=evidence_card,
                state_names=state_names,
            )
        else:
            # משתנה ללא הורים: values היא עמודה אחת (var_card x 1)
            values = np.zeros((var_card, 1), dtype=float)
            row = bn.cpts[var][()]  # CPT ללא הורים אצלך נשמר תחת key=()
            for state_i, x_val in enumerate(var_states):
                values[state_i, 0] = float(row[x_val])

            cpd = TabularCPD(
                variable=var,
                variable_card=var_card,
                values=values,
                state_names=state_names,
            )

        cpds.append(cpd)

    # 4) מוסיפים את כל ה-CPDs למודל ובודקים עקביות
    model.add_cpds(*cpds)

    # check_model בודק:
    # - שכל משתנה מקבל CPD
    # - שה-CPDs מנורמלים
    # - שהמבנה תואם להורים
    assert model.check_model()

    # 5) בונים אובייקט אינפרנס (Variable Elimination)
    infer = VariableElimination(model)  # :contentReference[oaicite:2]{index=2}

    return model, infer



def EY_and_VarY_from_pgmpy(infer, bn, Y, evidence, value_map=None):
    """
    מחשב E[Y|evidence] ו-Var(Y|evidence) בעזרת pgmpy.
    - infer: VariableElimination
    - bn: ה-BN שלך (בשביל domains/value_map)
    - evidence: dict כמו {"A":0, "Z1":1, ...}
    """

    # מפה מערכי Y לערכים מספריים (ברירת מחדל: 0/1)
    if value_map is None:
        domY = bn.domains[Y]
        if set(domY) == {0, 1}:
            value_map = {0: 0.0, 1: 1.0}
        else:
            raise ValueError("Y אינו בינארי. ספק/י value_map.")

    # query מחזיר פקטור/תוצאה שמכילה את ההתפלגות על Y
    q = infer.query(variables=[Y], evidence=evidence, show_progress=False)
    factorY = q[Y]  # התפלגות על Y

    # probability לכל מצב, באותו סדר של bn.domains[Y] (כי השתמשנו state_names)
    probs = [float(factorY.get_value(**{Y: y})) for y in bn.domains[Y]]

    y_vals = [float(value_map[y]) for y in bn.domains[Y]]
    EY = sum(p * v for p, v in zip(probs, y_vals))
    EY2 = sum(p * (v**2) for p, v in zip(probs, y_vals))
    VarY = EY2 - EY**2

    return EY, max(VarY, 0.0)
