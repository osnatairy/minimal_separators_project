from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Iterable


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
