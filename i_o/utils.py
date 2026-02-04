from __future__ import annotations
import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional,Iterable


def save_list_json(items: Iterable[Any], filepath: str | Path) -> Path:
    """
    Save an iterable (e.g., list) to a JSON file.
    Returns the resolved Path to the saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON צריך משהו שניתן לסריאליזציה; לכן ממירים ל-list
    data = list(items)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path.resolve()

def append_line(filename, line):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(line + "\n")



# ----------------------------
# 4) טופולוגיה ו-parent_order
# ----------------------------

def topo_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    מיון טופולוגי פשוט (Kahn).
    מחזיר סדר צמתים כך שהורים באים לפני ילדים.
    """
    indeg = {n: 0 for n in nodes}
    children = defaultdict(list)

    for u, v in edges:
        children[u].append(v)
        indeg[v] += 1

    queue = [n for n in nodes if indeg[n] == 0]
    out = []

    # נשמר סדר יציב לפי nodes (כדי להיות דטרמיניסטי)
    queue = [n for n in nodes if indeg[n] == 0]

    while queue:
        n = queue.pop(0)
        out.append(n)
        for ch in children.get(n, []):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                queue.append(ch)

    if len(out) != len(nodes):
        raise ValueError("Graph has a cycle or disconnected parsing issue; cannot topo-sort.")

    return out
