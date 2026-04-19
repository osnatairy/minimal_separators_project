import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_separators_by_run(csv_path, output_dir=None, show=True):
    """
    קורא קובץ CSV של separators ומפיק גרף לכל הרצה (seed, X, Y).

    פרמטרים:
    ----------
    csv_path : str
        נתיב לקובץ ה-CSV.
    output_dir : str | None
        אם מוגדר, ישמור את הגרפים כתמונות בתיקייה הזו.
    show : bool
        האם להציג את הגרפים על המסך.

    מחזיר:
    -------
    None
    """

    # קריאת הקובץ
    df = pd.read_csv(csv_path)

    # ניקוי שמות עמודות (כי אצלך יש רווחים בתחילת חלק מהשמות)
    df.columns = df.columns.str.strip()

    # בדיקה שהעמודות הדרושות קיימות
    required_cols = {"seed", "X", "Y", "len_sep", "variance", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"חסרות עמודות בקובץ: {missing}")

    # ניקוי ערכי type
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # אם צריך לשמור קבצים - ניצור תיקייה
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # קיבוץ לפי הרצה
    grouped = df.groupby(["seed", "X", "Y"])

    for (seed, x_node, y_node), group in grouped:
        plt.figure(figsize=(8, 6))

        # כוכב צהוב: henkel
        henkel_group = group[group["type"] == "henkel"]

        if not henkel_group.empty:
            plt.scatter(
                henkel_group["len_sep"],
                henkel_group["variance"],
                marker="*",
                s=250,
                color="yellow",
                edgecolors="black",
                label="henkel",
                zorder=1
            )

        # נקודות כחולות: optimal
        blue_types = ["optimal"]
        blue_group = group[group["type"].isin(blue_types)]

        # נקודות אדומות: minimal / non minimal
        red_types = ["minimal", "non_minimal"]
        red_group = group[group["type"].isin(red_types)]

        if not blue_group.empty:
            plt.scatter(
                blue_group["len_sep"],
                blue_group["variance"],
                label="optimal",
                alpha=0.8,
                zorder=3

            )

        if not red_group.empty:
            plt.scatter(
                red_group["len_sep"],
                red_group["variance"],
                color="red",
                label="minimal / non minimal",
                alpha=0.8,
                zorder=2

            )

        plt.xlabel("Separator size")
        plt.ylabel("Variance")
        plt.title(f"seed={seed}, X={x_node}, Y={y_node}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # שמירה
        if output_dir is not None:
            filename = f"seed_{seed}_X_{x_node}_Y_{y_node}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")

        # תצוגה
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    nodes = [20,30,40]
    prob_nodes = [0.07, 0.1, 0.15, 0.2,0.3]  # , 0.20, 0.25]  # , 0.5, 0.7]

    for node in nodes:
        for prob_node in prob_nodes:
            for k_roots in [node, int(node*0.3), 3, 1]:
                variance = f"{node}_{prob_node}_{k_roots}"

                csv_path = f"outputs_bn/2026_04_13_exp2_bn_seperators_{variance}.csv"
                #csv_path = "outputs_bn/2026_04_09_exp2_bn_seperators_20_0.15.csv"
                output_dir = csv_path.split("/")[0]+"/exp_2/"+csv_path.split("/")[-1].replace(".csv", "")
                plot_separators_by_run(csv_path, output_dir, show=False)
                #plot_separators_by_run("outputs_bn/2026_04_09_exp2_bn_seperators_20_0_15.csv", output_dir="outputs_bn/exp2", show=False)