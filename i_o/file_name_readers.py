from pathlib import Path
from typing import Optional
from typing import Iterable, Dict, List, Sequence, Set

def list_filenames(
    directory: str | Path,
    suffix: Optional[str] = None,
    stem_only: bool = False
) -> List[str]:
    """
    Return a list of filenames in a directory.

    Parameters
    ----------
    directory : str or Path
        Path to the directory.
    suffix : str, optional
        File suffix to filter by (e.g. '.json').
    stem_only : bool
        If True, return filename without suffix.

    Returns
    -------
    List[str]
        List of filenames.
    """
    directory = Path(directory)

    files = [
        f.stem if stem_only else f.name
        for f in directory.iterdir()
        if f.is_file() and (suffix is None or f.suffix == suffix)
    ]

    return sorted(files)

from typing import Iterable, Tuple, List

def compare_file_lists(
    list1: Iterable[str],
    list2: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """
    Compare two lists of filenames.

    Returns
    -------
    only_in_list1 : List[str]
        Files that appear only in list1.
    only_in_list2 : List[str]
        Files that appear only in list2.
    """
    set1 = set(list1)
    set2 = set(list2)

    only_in_1 = sorted(set1 - set2)
    only_in_2 = sorted(set2 - set1)

    return only_in_1, only_in_2




def uniques_per_list(*lists: Iterable[str]) -> Dict[int, List[str]]:
    """
    For each input list i, return the items that appear only in list i
    and in no other lists.

    Returns
    -------
    dict: {i: sorted_unique_items_only_in_list_i}
    """
    sets: List[Set[str]] = [set(lst) for lst in lists]
    union_all = set().union(*sets) if sets else set()

    out: Dict[int, List[str]] = {}
    for i, s in enumerate(sets):
        others_union = union_all - s
        out[i] = sorted(s - others_union)
    return out





files_a = list_filenames("../scripts/outputs/graph1", suffix=".json")
files_b = list_filenames("../scripts/outputs/graph2", suffix=".json")
files_c = list_filenames("../scripts/outputs/graph3", suffix=".json")



only_ab, only_ba = compare_file_lists(files_a, files_b)
only_ac, only_ca = compare_file_lists(files_a, files_c)
only_bc, only_cb = compare_file_lists(files_b, files_c)

print("Only in A:", only_ab)
print("Only in B:", only_ba)

print("Only in A:", only_ac)
print("Only in C:", only_ca)

print("Only in B:", only_bc)
print("Only in C:", only_cb)





