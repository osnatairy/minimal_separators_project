
from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Sequence, Set, FrozenSet, Tuple

import networkx as nx

from analysis.adjustment_hasse import cy_component

def bucket_separators_by_y_connectivity(
    G: Any,
    Y: Hashable,
    separators_list: Sequence[Iterable[Hashable]],
) -> List[List[FrozenSet[Hashable]]]:
    """
    Partition separators into buckets by 'closeness to Y' using inclusion of
    Cy(G - S) sets.

    Bucket 1 = separators whose Cy(G - S) is minimal by *strict inclusion* among
    all remaining separators. Then remove them and repeat.

    Parameters
    ----------
    G : Any
        Your graph object. Passed through to cy_component as-is.
    Y : Hashable
        The target node (or can be whatever your cy_component expects as Y).
    separators_list : Sequence[Iterable[Hashable]]
        List of separators; each separator is an iterable of nodes.
    cy_component : callable
        Helper function: cy_component(G, Y, S) -> Iterable of nodes in Cy(G - S).
        You said you already have it.

    Returns
    -------
    List[List[FrozenSet[Hashable]]]
        Buckets, each bucket is a list of separators (as frozensets).
        Earlier bucket index => closer to Y (smaller Cy-set under inclusion).

    Notes
    -----
    - If two separators induce exactly the same Cy-set, they will end up in the
      same bucket (neither is a *strict* subset of the other).
    - Complexity is O(m * cost(cy_component) + m^2 * subset_cost),
      where m = number of separators. If V is large, consider representing
      Cy-sets as bitsets for speed.
    """

    # Canonicalize separators so they are hashable and comparable
    seps: List[FrozenSet[Hashable]] = [frozenset(S) for S in separators_list]

    # 1) Compute R[S] = Cy(G - S)
    R: Dict[FrozenSet[Hashable], FrozenSet[Hashable]] = {}
    for S in seps:
        comp_nodes = cy_component(G, Y, S)  # should return iterable of nodes
        R[S] = frozenset(comp_nodes)

    # 2) Iteratively peel off minimal elements under strict inclusion
    unassigned: Set[FrozenSet[Hashable]] = set(seps)
    buckets: List[List[FrozenSet[Hashable]]] = []

    while unassigned:
        current_bucket: List[FrozenSet[Hashable]] = []

        # Optional micro-optimization: check smaller Cy-sets first
        # (helps find strict subsets earlier)
        candidates = sorted(unassigned, key=lambda s: len(R[s]))

        for S in candidates:
            is_minimal = True
            RS = R[S]

            # Only sets strictly smaller than RS can be strict subsets.
            # Because candidates is sorted by size, we can stop once len >= len(RS).
            for T in candidates:
                if T == S:
                    continue
                RT = R[T]
                if len(RT) >= len(RS):
                    # no strict subset possible beyond this point
                    break
                if RT < RS:  # strict subset
                    is_minimal = False
                    break

            if is_minimal:
                current_bucket.append(S)

        # Safety: should never be empty, but guard against bugs in cy_component
        if not current_bucket:
            raise RuntimeError(
                "No minimal separators found in this iteration. "
                "Check that cy_component returns consistent sets."
            )

        buckets.append(current_bucket)
        unassigned.difference_update(current_bucket)

    return buckets


from typing import Any, Callable, Dict, Hashable, Iterable, List, FrozenSet, Set, Tuple


def bucket_separators_by_cy_layers(
    G: Any,
    Y: Hashable,
    separators_list: Iterable[Iterable[Hashable]],
) -> Dict[FrozenSet[Hashable], List[FrozenSet[Hashable]]]:
    """
    Partition separators into buckets (layers) by minimality under *strict inclusion*
    of R(S) := C_Y(G - S).

    Process:
      - Compute R(S) for every separator S.
      - Repeatedly take all separators whose R(S) is minimal among the remaining ones
        (i.e., there is no remaining T with R(T) ⊂ R(S)).
      - Each iteration forms one bucket (one layer).

    Return format:
      dict mapping key = R(S) (as a frozenset of nodes)
                 value = list of separators (each as a frozenset) that are in the
                         SAME *layer* (bucket) and share that exact R(S).

    Notes:
      - If multiple separators induce the same R(S), they appear together under the
        same key.
      - Different layers can also contain different R(S) keys (so the dict is not
        explicitly "bucket index -> ..."). If you also need bucket indices, tell me
        and I’ll return an ordered list of dicts instead.
    """

    # Canonicalize separators so they are hashable
    seps: List[FrozenSet[Hashable]] = [frozenset(S) for S in separators_list]

    # Compute R(S) for all S
    R_of: Dict[FrozenSet[Hashable], FrozenSet[Hashable]] = {}
    for S in seps:
        R_of[S] = frozenset(cy_component(G, Y, S))

    # We'll build an *ordered* list of layers, each layer is a list of separators
    unassigned: Set[FrozenSet[Hashable]] = set(seps)
    layers: List[List[FrozenSet[Hashable]]] = []

    while unassigned:
        # Sort remaining by |R(S)| so we only test strict subsets that can exist
        remaining = sorted(unassigned, key=lambda s: len(R_of[s]))
        current_layer: List[FrozenSet[Hashable]] = []

        for i, S in enumerate(remaining):
            RS = R_of[S]
            is_minimal = True

            # Only candidates with strictly smaller |R| can be strict subsets
            for j in range(i):  # because remaining is sorted by len(R)
                T = remaining[j]
                if R_of[T] < RS:  # strict subset
                    is_minimal = False
                    break

            if is_minimal:
                current_layer.append(S)

        if not current_layer:
            raise RuntimeError(
                "No minimal separators found in an iteration. "
                "Check that cy_component returns consistent sets."
            )

        layers.append(current_layer)
        unassigned.difference_update(current_layer)

    # Now: you asked for a dict keyed by R(S) -> list of separators in *that bucket*.
    # Ambiguity: keys repeat across layers? In practice with this layering,
    # the same exact R(S) cannot appear in two different layers (because if R equal,
    # they'd be minimal together when first available). So we can safely flatten.
    buckets_by_R: Dict[FrozenSet[Hashable], List[FrozenSet[Hashable]]] = {}

    for layer in layers:
        for S in layer:
            RS = R_of[S]
            buckets_by_R.setdefault(RS, []).append(S)

    return buckets_by_R


from statistics import mean, median
from typing import Dict, List, Any


def bucket_variance_statistics(
        seed: int,
        X: str,
        Y: str,
    buckets: List[List[Any]],
    variance_by_separator: Dict[Any, float],
    y_component_result: List[List[Any]],
    buckets_results : Dict[int, List[Any]]
):
    """
    For each bucket, compute:
      - number of separators
      - sum of variances
      - average variance
      - median variance

    Parameters
    ----------
    buckets : list of lists
        buckets[i] is a list of separator identifiers in bucket i
    variance_by_separator : dict
        maps separator identifier -> variance (float)

    Returns
    -------
    list of dicts
        One dict per bucket, with summary statistics.
        Bucket index i corresponds to buckets[i].
    """



    for bucket_index, bucket in enumerate(buckets, start=1):
        bucket_key = tuple(sorted(bucket))
        variances = [
            variance_by_separator[tuple(sorted(s))]
            for s in bucket_key
            if tuple(sorted(s)) in variance_by_separator
        ]
        y_component = [
            len(y_component_result[s])
            for s in bucket
            if s in y_component_result
        ]

        y_component = {
            key: len(value)
            for key, value in y_component_result.items()
        }

        if not variances:
            print("Bucket {} has no variance".format(bucket_key))
            # stats = {
            #     "bucket": bucket_index,
            #     "num_separators":0,
            #     "variance_list": ",".join([]),
            #     "variance_max": 0,
            #     "variance_sum": 0,
            #     "variance_mean": 0,
            #     "variance_median": 0,
            #     "y_component_list": ",".join([]),
            #     "y_component_max": 0,
            #     "y_component_sum": 0,
            #     "y_component_mean": 0,
            #     "y_component_median": 0,
            # }
        else:
            for sep in bucket:
                stats = {
                    "seed": seed,
                    "X": X,
                    "Y": Y,
                    "bucket": bucket_index,
                    #"num_separators": len(variances),
                    "seperator": tuple(sorted(sep)),
                    "variance": variance_by_separator[tuple(sorted(sep))],
                    #"variance_max": max(variance_by_separator.values()),
                    #"variance_min": min(variance_by_separator.values()),
                    #"variance_sum": sum(variance_by_separator.values()),
                    #"variance_mean": mean(variance_by_separator.values()),
                    #"variance_median": median(variances),
                    "y_component_len": y_component[sep],
                    #"y_component_max": max(y_component.values()),
                    #"y_component_min": min(y_component.values()),
                    #"y_component_sum": sum(y_component.values()),
                    #"y_component_mean": mean(y_component.values()),
                    #"y_component_median": median(y_component.values()),

                }

                #buckets_results.append(stats)
                buckets_results.setdefault(bucket_index, []).append(stats)

    #return results


def bucket_y_component(buckets:Dict[FrozenSet[Hashable], List[FrozenSet[Hashable]]],
                        y_component_by_z: Dict[FrozenSet[Hashable], FrozenSet[Hashable]]):
    bucket_y_component = []
    for bucket in buckets:
        temp = []
        for Z in bucket:
            temp.append(len(y_component_by_z[Z]))
        bucket_y_component.append(temp)

    return bucket_y_component



