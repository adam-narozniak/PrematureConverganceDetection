import numpy as np
from cec2017 import simple
from cec2017 import hybrid
import reproductions
import successions
import logging
import matplotlib.pyplot as plt
import pandas as pd
import sys
import premature_convergence_algorithms
import crossovers
import mutations
from data_collector import DataCollector
import pathlib
import plotter
from cec2017 import functions

logger = logging.getLogger("PrematureConvergenceDetection")


def evolutionary_algorithm(population, n_iterations, mutation_probability, crossover_probability, cost_function,
                           mutation_strength, reproduction_fnc=reproductions.roulette_wheel,
                           succession_fnc=successions.elite, n_elite=None):
    """
    Evolutionary algorithm based on holland algorithm.

    Args:
        population:
        n_iterations:
        mutation_probability:
        crossover_probability:
        cost_function:
        mutation_strength:
        reproduction_fnc: pointer to reproduction (selection) function
        succession_fnc: pointer to succession function
        n_elite:

    Returns:

    """
    logger.info("Generic algorithm started")
    if population.ndim != 2:
        raise Exception(f"Wrong format of population")

    population_size = population.shape[0]
    n_features = population.shape[1]

    if population_size < 1:
        raise Exception(f"Population should be positive number, not {population_size}")
    if n_features < 1:
        raise Exception(f"Number of features should be positive number, not {population_size}")

    scores = evaluate(population, cost_function)
    best_individual_value, best_individual_idx = find_best_score(scores)
    best_individual_features = population[best_individual_idx]
    data_collector = DataCollector(n_features, n_iterations)
    data_collector.add_metrics(0, population, scores, best_individual_features, best_individual_value)
    for iteration in range(1, n_iterations + 1):
        # if premature_convergence_algorithms.naive_stop(population, scores, bests):
        #     break

        children = reproduction_fnc(population, scores)
        # genetic operations
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability,
                                                       mutation_strength, crossover=True)
        scores_mutants = evaluate(mutants_and_crossovered, cost_function)
        iteration_best_value, best_individual_in_iteration_idx = find_best_score(scores_mutants)
        feature_best_iteration = mutants_and_crossovered[best_individual_in_iteration_idx]
        # logger.info(
        #     f"{'Best in iteration.':20} score: {iteration_best_value:5.02e}") # features: {feature_best_iteration};

        # choose best individual
        if best_individual_value > iteration_best_value:
            best_individual_value = iteration_best_value
            best_individual_features = feature_best_iteration
        if succession_fnc == successions.elite:
            population, scores = succession_fnc(population, mutants_and_crossovered, scores, scores_mutants, 0.05)
        else:  # succession.generative
            population, scores = succession_fnc(population, mutants_and_crossovered, scores, scores_mutants)
        data_collector.add_metrics(iteration, population, scores, best_individual_features, best_individual_value)
        # logger.info(
        #     f"{'Best overall.':20} score: {best_individual_value:5.02e}")  # features: {best_individual_features};
        # logger.info(f"Iteration {iteration:3d}/{n_iterations} completed.")
    logger.info("Generic algorithm stopped")
    return data_collector


def evaluate(population, cost_function):
    """Evaluate each individual in population based on give cost function.
    Args:
        population:
        cost_function: pointer to cec2017 function

    """
    evaluation_scores = np.zeros([population.shape[0], ])
    for row_idx in range(population.shape[0]):
        evaluation_scores[row_idx] = cost_function(population[row_idx])
    return evaluation_scores


def find_best_score(scores):
    """Find best score and corresponding index.
    """
    scores_pd = pd.Series(scores)
    my_min = scores_pd.min(axis=0)
    return my_min, scores_pd.idxmin(axis=0)


def mutate_and_crossover(population, mutation_probability, crossover_probability, mutation_strength, crossover=False):
    """Perform genetic operations.

    Args:
        crossover: if True crossover will be performed otherwise it will be skipped
    """
    rng = np.random.default_rng()
    population = mutations.mutate(population, mutation_probability, mutation_strength, rng)

    # crossover (pl.krzyżowanie jednopunktowe)
    if crossover is False:
        pass
    else:
        population = crossovers.crossover(population, crossover_probability, rng)
    return population


def initialize_population(n_features, population_size):
    """ Initialize population, meet cec2017 criteria which are: uniform distribution within search range
    [-100, 100]^n_features.(pl. posiew równomierny)

    Args:
        n_features: number of dimensions
        population_size: size of population
    """
    rng = np.random.default_rng()
    population = rng.uniform(-100, 100, n_features * population_size).reshape(population_size, n_features)
    return population


def prepare_logging():
    """In order to generate information how algorithm performs."""
    logger = logging.getLogger("PrematureConvergenceDetection")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def plot_results(bests, std_x0, mean_x0):
    plt.plot(list(range(1, len(bests) + 1)), bests)
    plt.title("best scores")
    plt.savefig("./plot_best_score.jpg")
    plt.figure()
    plt.plot(list(range(1, len(std_x0) + 1)), std_x0)
    plt.title("std x0")
    plt.savefig("./plot_std_x0.jpg")
    plt.figure()
    plt.title("mean x0")
    plt.plot(list(range(1, len(mean_x0) + 1)), mean_x0)
    plt.savefig("./plot_mean_x0.jpg")


def check_on_one_fnc(cost_function, name):
    logger.info(f"start: {name}")
    n_features = 10
    population_size = 1000
    n_iterations = 200
    mutation_probability = 0.2
    crossover_probability = 0.6
    mutation_strength = 10
    population = initialize_population(n_features, population_size)
    data_collector = evolutionary_algorithm(population, n_iterations, mutation_probability,
                                            crossover_probability, cost_function,
                                            mutation_strength)
    results = data_collector.save_data(pathlib.Path.cwd()/"data"/"all_fnc"/f"{name}.csv")
    plotter.plot_std_on_best_x(results, name)
    plotter.plot_mean_on_best_x(results, name)
    plotter.plot_best_individual_value_vs_std_x(results, name)
    plotter.plot_best_individual_value_vs_mean_x(results, name)

def prepare_gid_search():
    """Return search params"""
    population_size = [100, 300, 1000, 3000]
    mutation_probability = [0.1, 0.2, 0.3]
    mutation_strength = [1, 3, 10, 30]
    crossover_probability = [0.2, 0.5, 0.7]
    search_params = np.array(
        np.meshgrid(population_size, mutation_probability, mutation_strength, crossover_probability)).T.reshape(-1, 4)
    search_params = pd.DataFrame(search_params, columns=["population_size", "mutation_probability", "mutation_strength",
                                                         "crossover_probability"])
    return search_params


def run_grid_search(search_params):
    n_iterations = 200
    successions_fnc = successions.elite
    reproduction_fnc = reproductions.roulette_wheel
    for index, row in search_params.iterrows():
        population = initialize_population(10, int(row["population_size"]))
        evolutionary_algorithm(population, n_iterations, row["mutation_probability"], row["crossover_probability"],
                               simple.f1, row["mutation_strength"])


if __name__ == '__main__':
    prepare_logging()
    # search_params = prepare_gid_search()
    # run_grid_search(search_params)
    fnc_names = ['f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16',
                 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30']
    for fnc_name, fnc in zip(fnc_names, functions.all_functions):
        check_on_one_fnc(fnc, fnc_name)

