import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Graph:

    _LAYOUT_ITERATIONS = 100
    _INITIAL_TEMP = 9.0
    _TEMP_DECREASE = _INITIAL_TEMP / _LAYOUT_ITERATIONS
    _MARGIN = 0.02

    def __init__(self) -> None:
        self._adjacency_set: dict[int, set[tp.Any]] = dict()
        self._vertex_to_info: dict[tp.Any, tp.Any] = dict()

    def __len__(self) -> int:
        return len(self._adjacency_set)

    @property
    def vertices(self) -> list[tp.Any]:
        return list(self._adjacency_set.keys())

    @property
    def adjacent_set(self) -> dict[int, set[tp.Any]]:
        return self._adjacency_set

    @property
    def info(self) -> tp.Any:
        return self._vertex_to_info

    def add_vertex(self, vertex_label: tp.Any, info: tp.Any = None) -> None:
        if vertex_label in self._adjacency_set:
            raise ValueError(f"The vertex '{vertex_label}' already exists")

        self._adjacency_set[vertex_label] = set()
        self._vertex_to_info[vertex_label] = info

    def add_edge(self, vertex_a: tp.Any, vertex_b: tp.Any) -> None:
        if vertex_a not in self._adjacency_set:
            raise ValueError(f"The vertex '{vertex_a}' doesn't exist")
        if vertex_b not in self._adjacency_set:
            raise ValueError(f"The vertex '{vertex_b}' doesn't exist")
        if vertex_b in self._adjacency_set[vertex_a]:
            raise ValueError(
                f"The edge '{vertex_a} - {vertex_b}' already exists")

        self._adjacency_set[vertex_a].add(vertex_b)
        self._adjacency_set[vertex_b].add(vertex_a)

    @staticmethod
    def _get_repulsive_force(norm: float, coeff: float) -> float:
        return coeff ** 2 / norm

    @staticmethod
    def _get_attraction_force(norm: float, coeff: float) -> float:
        return norm ** 2 / coeff

    def _get_graph_layout(self) -> dict[tp.Any, tuple[float, float]]:
        width = 1
        height = 1

        coords: dict[tp.Any, npt.NDArray[np.float64]] = dict()
        forces: dict[tp.Any, npt.NDArray[np.float64]] = dict()

        for vertex in self._adjacency_set:
            coords[vertex] = np.random.uniform(0, 1, size=2) * np.array([width, height])

        coeff = np.sqrt(width * height / len(self._adjacency_set))
        temp = self._INITIAL_TEMP

        for _ in range(self._LAYOUT_ITERATIONS):
            for v in self._adjacency_set:
                forces[v] = np.zeros(2, dtype=float)
                for u in self._adjacency_set:
                    if u == v:
                        continue
                    vector = coords[v] - coords[u]
                    distance = np.linalg.norm(vector)

                    if np.isclose(distance, 0):
                        continue

                    repulsive_force = self._get_repulsive_force(distance, coeff) * vector / distance  # type: ignore
                    forces[v] += repulsive_force

            for v in self._adjacency_set:
                for u in self._adjacency_set[v]:
                    vector = coords[u] - coords[v]

                    distance = np.linalg.norm(vector)

                    if np.isclose(distance, 0):
                        continue

                    attraction_force = self._get_attraction_force(distance, coeff) * vector / distance  # type: ignore
                    forces[v] += attraction_force

            for v in self._adjacency_set:
                force_norm = np.linalg.norm(forces[v])
                coords[v] += np.random.uniform(-0.02, 0.02, size=2)

                if np.isclose(force_norm, 0):
                    continue

                coords[v] += forces[v] / force_norm * min(force_norm, temp)  # type: ignore
                np.clip(coords[v], [0, 0], [width, height], out=coords[v])

            temp -= self._TEMP_DECREASE

        result: dict[tp.Any, tuple[float, float]] = dict()

        for v in coords:
            result[v] = (coords[v][0], coords[v][1])

        return result

    def draw(self, figsize: tp.Optional[tuple[int, int]] = None) -> None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([])
        ax.set_yticks([])

        coords = self._get_graph_layout()

        xlim = [1.0 + self._MARGIN, -self._MARGIN]
        ylim = [1.0 + self._MARGIN, -self._MARGIN]

        for vertex in coords:
            x, y = coords[vertex]
            xlim[0] = min(xlim[0], x - 0.02)
            xlim[1] = max(xlim[1], x + 0.02)

            ylim[0] = min(ylim[0], y - 0.02)
            ylim[1] = max(ylim[1], y + 0.02)

            ax.text(x, y, str(vertex), ha="center", va="center", bbox=dict(boxstyle=f"circle,pad={0.4}", fc="#7dc4fa"))

        for vertex in self._adjacency_set:
            x_v, y_v = coords[vertex]

            for adj in self._adjacency_set[vertex]:
                x_adj, y_adj = coords[adj]
                ax.plot([x_v, x_adj], [y_v, y_adj], c="red", zorder=0)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    @staticmethod
    def generate_random_graph(n_vertices: int, p: float = 0.5) -> "Graph":
        graph = Graph()

        for v in range(1, n_vertices + 1):
            graph.add_vertex(v)

        for v in range(1, n_vertices + 1):
            for u in range(v + 1, n_vertices + 1):
                if np.random.random() < p:
                    graph.add_edge(v, u)

        return graph
