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
                           mutation_strength, reproduction_fnc=reproductions.rank_selection,
                           succession_fnc=successions.elite, n_elite=None):
    """Analogical verion like in main but for the purpose of comparison best scores."""
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

    # cmp variants of looking back step and percent to compare
    look_back_by = [10, 25, 50]
    count_as_stuck_if_ratio_is_less_than = [1.025, 1.05, 1.1]
    variants = pd.DataFrame(np.zeros((len(look_back_by), len(count_as_stuck_if_ratio_is_less_than))),
                            columns=count_as_stuck_if_ratio_is_less_than, index=look_back_by)

    for iteration in range(1, n_iterations + 1):
        if (variants == 0).any().any():
            premature_convergence_algorithms.naive_stop_cmp(data_collector, variants, iteration - 1)

        children = reproduction_fnc(population, scores)
        # genetic operations
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability,
                                                       mutation_strength, crossover=True)
        scores_mutants = evaluate(mutants_and_crossovered, cost_function)
        iteration_best_value, best_individual_in_iteration_idx = find_best_score(scores_mutants)
        feature_best_iteration = mutants_and_crossovered[best_individual_in_iteration_idx]
        # choose best individual
        if best_individual_value > iteration_best_value:
            best_individual_value = iteration_best_value
            best_individual_features = feature_best_iteration
        if succession_fnc == successions.elite:
            population, scores = succession_fnc(population, mutants_and_crossovered, scores, scores_mutants, 0.1)
        else:  # succession.generative
            population, scores = succession_fnc(population, mutants_and_crossovered, scores, scores_mutants)
        data_collector.add_metrics(iteration, population, scores, best_individual_features, best_individual_value)
        logger.info(
            f"{'Best overall.':20} score: {best_individual_value:5.02e}")  # features: {best_individual_features};
        logger.info(f"Iteration {iteration:3d}/{n_iterations} completed.")
    logger.info("Generic algorithm stopped")
    return data_collector, variants


def evolutionary_algorithm_for_stds(population, n_iterations, mutation_probability, crossover_probability,
                                    cost_function,
                                    mutation_strength, reproduction_fnc=reproductions.rank_selection,
                                    succession_fnc=successions.elite, n_elite=None):
    """Analogical version as in main but with purpose of comparison of stds"""
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

    # cmp variants of looking back step and percent to compare
    thresholds = [2, 0.1, 0.001]
    variants = pd.Series(np.zeros((len(thresholds))), index=thresholds)

    for iteration in range(1, n_iterations + 1):
        if (variants == 0).any():
            premature_convergence_algorithms.stds_cmp(data_collector, variants, iteration - 1)

        children = reproduction_fnc(population, scores)
        # genetic operations
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability,
                                                       mutation_strength, crossover=True)
        scores_mutants = evaluate(mutants_and_crossovered, cost_function)
        iteration_best_value, best_individual_in_iteration_idx = find_best_score(scores_mutants)
        feature_best_iteration = mutants_and_crossovered[best_individual_in_iteration_idx]
        # choose best individual
        if best_individual_value > iteration_best_value:
            best_individual_value = iteration_best_value
            best_individual_features = feature_best_iteration
        if succession_fnc == successions.elite:
            population, scores = succession_fnc(population, mutants_and_crossovered, scores, scores_mutants, 0.1)
        else:  # succession.generative
            population, scores = succession_fnc(population, mutants_and_crossovered, scores, scores_mutants)
        data_collector.add_metrics(iteration, population, scores, best_individual_features, best_individual_value)
        logger.info(
            f"{'Best overall.':20} score: {best_individual_value:5.02e}")  # features: {best_individual_features};
        logger.info(f"Iteration {iteration:3d}/{n_iterations} completed.")
    logger.info("Generic algorithm stopped")
    return data_collector, variants


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
    """Find best score and corresponding index."""
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
    population_size = 500
    n_iterations = 200
    mutation_probability = 0.2
    crossover_probability = 0.6
    mutation_strength = 10
    population = initialize_population(n_features, population_size)
    data_collector, variants = evolutionary_algorithm(population, n_iterations, mutation_probability,
                                                      crossover_probability, cost_function,
                                                      mutation_strength)
    data_collector.save_data(pathlib.Path.cwd() / "data" / "all_fnc" / f"{name}.csv")
    results = data_collector.results
    my_plotter = plotter.Plotter(results, variants, name)
    variants.to_csv(pathlib.Path.cwd() / "data" / "all_fnc" / f"stop_in_{name}.csv")
    my_plotter.plot_cmp_bests_variants()

def check_on_one_fnc_for_stds(cost_function, name):
    logger.info(f"start: {name}")
    n_features = 10
    population_size = 500
    n_iterations = 200
    mutation_probability = 0.2
    crossover_probability = 0.6
    mutation_strength = 10
    population = initialize_population(n_features, population_size)
    data_collector, variants = evolutionary_algorithm_for_stds(population, n_iterations, mutation_probability,
                                                      crossover_probability, cost_function,
                                                      mutation_strength)
    data_collector.save_data(pathlib.Path.cwd() / "data" / "all_fnc" / f"{name}.csv")
    results = data_collector.results
    my_plotter = plotter.Plotter(results, variants, name)
    variants.to_csv(pathlib.Path.cwd() / "data" / "all_fnc" / f"stop_in_{name}.csv")
    my_plotter.plot_cmp_stds_variants()


if __name__ == '__main__':
    prepare_logging()
    fnc_names = ['f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16',
                 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30']
    every_other_fnc_name = fnc_names[::2]
    every_other_fnc = functions.all_functions[::2]
    for run in range(1, 25):
        for fnc_name, fnc in zip(every_other_fnc_name, every_other_fnc):
            check_on_one_fnc_for_stds(fnc, f"{fnc_name}_run{run:02d}")
