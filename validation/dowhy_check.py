from networkx.drawing.nx_pydot import to_pydot
from dowhy import CausalModel
import pandas as pd

from validation.set_utils import sort_and_dedup_subsets

from validation.separators_consistency import (
    contains_set,
    check_separators_consistency,
)


# נניח שיש לך G: nx.DiGraph עם קשתות מכוונות
# ויש לך שמות צמתים: X (treatment) ו-Y (outcome)
def test_Z_with_dowhy(G, X, Y, Z_sets):
    # 1) ממירים את גרף ה-NX ל-DOT
    dot = to_pydot(G).to_string()

    # >>> פתרון השגיאה: DataFrame ריק עם כל הצמתים כעמודות
    df_empty = pd.DataFrame(columns=list(G.nodes()))

    model = CausalModel(
        data=df_empty,
        treatment=X,
        outcome=Y,
        graph=dot
    )

    # 2) יוצרים מודל סיבתי (אין חובה ל-data בשלב הזיהוי)
    #model = CausalModel(data=None, treatment=X, outcome=Y, graph=dot)

    # 3) מזהים עם exhaustive-search כדי לקבל את כל סטי ה-backdoor התקפים
    identified = model.identify_effect(method_name="exhaustive-search",optimize_backdoor=True)

    # 4) מציגים/שולפים את כל הסטים
    print(identified)

    bd = identified.backdoor_variables  # dict כמו בצילום
    raw_sets = list(bd.values())
    sorted_raw_sets = sort_and_dedup_subsets(raw_sets)

    for z in Z_sets:
        exists = contains_set(raw_sets, z)
        print(f"{z}: {exists}")

    check_separators_consistency(raw_sets, Z_sets)
    # לעתים שימושי:
    # print(identified.__str__(show_all_backdoor_sets=True))

    # חלק מהגרסאות מאפשרות לגשת לאוסף הסטים כך:
    #all_sets = identified.estimands.get("backdoor", {}).get("backdoor_sets", [])
    #print("All valid backdoor sets:", all_sets)



