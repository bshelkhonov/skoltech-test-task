import itertools

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from algorithms import find_max_independent_set, find_max_independent_set_approximately
from graph import Graph


def test_heuristic() -> None:
    np.random.seed(42)

    graph_sizes = [3, 10, 20]
    probas = [0.1, 0.3, 0.6, 0.8]

    with open("results/heuristic.txt", "w") as f:
        timer = tqdm(desc="Heuristic test", total=len(graph_sizes) * len(probas))

        for size, p in itertools.product(graph_sizes, probas):
            graph = Graph.generate_random_graph(size, p)
            graph.draw((12, 12))
            plt.savefig(f"./images/heuristic-size={size}-p={p}.png")
            plt.close()

            max_ind_set = find_max_independent_set_approximately(graph)
            f.write(f"Max independent set for graph size={size} & p={p}: {max_ind_set}\n")
            timer.update()

        elapsed = timer.format_dict["elapsed"]
        f.write(f"Elapsed {elapsed:.2f} seconds")


def test_precise_algorithm() -> None:
    np.random.seed(42)

    graph_sizes = [3, 10, 20]
    probas = [0.1, 0.3, 0.6, 0.8]

    with open("results/precise.txt", "w") as f:
        timer = tqdm(desc="Precise test", total=len(graph_sizes) * len(probas))

        for size, p in itertools.product(graph_sizes, probas):
            graph = Graph.generate_random_graph(size, p)
            graph.draw((12, 12))
            plt.savefig(f"./images/precise-size={size}-p={p}.png")
            plt.close()

            max_ind_set = find_max_independent_set(graph)
            f.write(f"Max independent set for graph size={size} & p={p}: {max_ind_set}\n")
            timer.update()

        elapsed = timer.format_dict["elapsed"]
        f.write(f"Elapsed {elapsed:.2f} seconds")


if __name__ == "__main__":
    test_heuristic()
    test_precise_algorithm()
