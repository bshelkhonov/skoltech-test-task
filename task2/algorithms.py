import typing as tp

from sortedcontainers import SortedSet
import numpy as np

from graph import Graph


def find_max_independent_set_approximately(g: Graph) -> set[tp.Any]:
    ind_set: set[tp.Any] = set()

    degrees: SortedSet[tuple[int, tp.Any]] = SortedSet()

    for v in g.vertices:
        degrees.add((len(g.adjacent_set[v]), v))

    while len(degrees) > 0:
        _, vertex = degrees[0]
        del degrees[0]

        ind_set.add(vertex)

        for adj in g.adjacent_set[vertex]:
            degrees.discard((len(g.adjacent_set[adj]), adj))

    return ind_set


def find_max_independent_set(g: Graph) -> set[tp.Any]:
    vertex_to_index: dict[tp.Any, int] = dict()
    index_to_vertex: dict[int, tp.Any] = dict()

    for index, v in enumerate(g.vertices):
        vertex_to_index[v] = index
        index_to_vertex[index] = v

    antineighbors_mask: list[int] = [(2 ** len(g)) - 1] * len(g)
    for index, v in enumerate(g.vertices):
        for u in g.adjacent_set[v]:
            u_index = vertex_to_index[u]
            antineighbors_mask[index] ^= (1 << u_index)

    dp: list[bool] = [False] * (2 ** len(g))
    dp[0] = True

    max_mask_result = (0, 0)

    for mask in range(1, 2 ** len(g)):
        v = int(np.log2(mask))
        mask_prev = mask ^ (1 << v)
        dp[mask] = dp[mask_prev] and (antineighbors_mask[v] & mask_prev) == mask_prev

        if dp[mask]:
            max_mask_result = max(max_mask_result, (bin(mask).count("1"), mask))

    max_mask = max_mask_result[1]
    ind_set: set[tp.Any] = set()
    for i in range(len(g)):
        if ((max_mask >> i) & 1) == 1:
            ind_set.add(index_to_vertex[i])

    return ind_set
