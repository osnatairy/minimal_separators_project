from typing import Iterable, List, Tuple


def sort_and_dedup_subsets(raw_sets: Iterable[Iterable]) -> List[List[str]]:
    """
    קלט: raw_sets - iterable של איטרבלים (למשל רשימות של תווים/שמות)
    פלט: רשימת רשימות ממויינת:
      - כל תת-רשימה ממוינת (sorted)
      - הרשימה ממוינת לפי (|subset|, lexicographic tuple)
      - כפילויות מוסרות (שמור על first-seen order לפני המיון)
    """
    # 1) נרמול: המרת כל אלמנט לטאפל של איברים ממוינים
    normalized = [tuple(sorted(map(str, s))) for s in raw_sets]

    # 2) הסרת כפילויות תוך שמירה על סדר הופעה ראשון
    seen = {}
    uniq = []
    for t in normalized:
        if t not in seen:
            seen[t] = True
            uniq.append(t)

    # 3) מיון לפי (גודל, סדר לקסיקוגרפי)
    uniq.sort(key=lambda t: (len(t), t))

    # 4) החזרה כרשימת רשימות (ממוינות בתוך וכל אחת ממוינת כבר)
    return [list(t) for t in uniq]


def all_empty(container):
    return all(len(x) == 0 for x in container)
