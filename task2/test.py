import itertools
import os
import typing as tp

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from algorithms import find_max_independent_set, find_max_independent_set_approximately
from graph import Graph


def test_algorithm(params: dict[str, tp.Any]) -> None:
    np.random.seed(42)

    graph_sizes = [3, 10, 20]
    probas = [0.1, 0.3, 0.6, 0.8]

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./test_images"):
        os.makedirs("./test_images")

    prev_time = 0.0
    with open("./results/" + params["result_file"], "w") as f:
        timer = tqdm(desc=params["test_name"], total=len(graph_sizes) * len(probas))

        for size, p in itertools.product(graph_sizes, probas):
            graph = Graph.generate_random_graph(size, p)
            graph.draw((12, 12))
            plt.savefig(f"./test_images/{params['test_name']}-size={size}-p={p}.png")
            plt.close()

            max_ind_set = params["algorithm"](graph)
            f.write(f"Max independent set for graph size={size} & p={p}: {max_ind_set}, ")
            timer.update()

            elapsed = timer.format_dict["elapsed"] - prev_time
            prev_time = timer.format_dict["elapsed"]
            f.write(f"elapsed {elapsed:.2f} seconds\n")
        f.write(f"Total time {timer.format_dict['elapsed']:.2f} seconds\n")


def test_heuristic() -> None:
    test_algorithm({
        "result_file": "heuristic_results.txt",
        "test_name": "heuristic",
        "algorithm": find_max_independent_set_approximately,
    })


def test_precise_algorithm() -> None:
    test_algorithm({
        "result_file": "precise_results.txt",
        "test_name": "precise",
        "algorithm": find_max_independent_set,
    })


if __name__ == "__main__":
    test_heuristic()
    test_precise_algorithm()
