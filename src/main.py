import numpy as np
from cec2017 import simple
import reproductions
import successions
import logging
import matplotlib.pyplot as plt
import pandas as pd
import sys
import premature_convergence_algorithms

logger = logging.getLogger("PrematureConvergenceDetection")


def genetic_algorithm(population, iterations, mutation_probability, crossover_probability, cost_function,
                      mutation_strength):
    """Genetic algorithm based on holland algorithm."""
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
    bests = []
    bests.append(best_individual_value)
    for iteration in range(1, iterations + 1):
        # if premature_convergence_algorithms.naive_stop(population, scores, bests):
        #     break

        children = reproductions.roulette_wheel(population, scores)

        # don't get why it's grouped together, seems like they are separate operations
        # crossover not always during evolutionary algs
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability,
                                                       mutation_strength)

        scores_mutants = evaluate(mutants_and_crossovered, cost_function)
        iteration_best_value, best_individual_in_iteration_idx = find_best_score(scores_mutants)
        feature_best_iteration = mutants_and_crossovered[best_individual_in_iteration_idx]
        logger.info(
            f"{'Best in iteration.':20} features: {feature_best_iteration}; score: {iteration_best_value:5.2f}")
        if best_individual_value > iteration_best_value:
            best_individual_value = iteration_best_value
            best_individual_features = feature_best_iteration
        population, scores = successions.elite(population, mutants_and_crossovered, scores, scores_mutants, 0.1)
        logger.info(f"{'Best overall.':20} features: {best_individual_features}; score: {best_individual_value:5.2f}")
        logger.info(f"Iteration {iteration:3d}/{iterations} completed.")
        bests.append(best_individual_value)
    return bests


def evaluate(population, cost_function):
    """Evaluate each individual in population based on give cost function.
    Args:
        cost_function: pointer to cec2017 function
        """
    evaluation_scores = np.zeros([population.shape[0], ])
    for row_idx in range(population.shape[0]):
        evaluation_scores[row_idx] = cost_function(population[row_idx])
    return evaluation_scores


def find_best_score(scores):
    scores_pd = pd.Series(scores)
    my_min = scores_pd.min(axis=0)

    return my_min, scores_pd.idxmin(axis=0)


def mutate_and_crossover(population, mutation_probability, crossover_probability, mutation_strength,
                         feature_crossover_probability=0.5):
    population_size = population.shape[0]
    n_features = population.shape[1]
    rng = np.random.default_rng()
    # mutate
    mutation_uniform = rng.uniform(0, 1, population_size)
    normal_noise_for_mutation = rng.standard_normal(population_size *
                                                    n_features).reshape(population_size, n_features) * mutation_strength
    population[mutation_probability > mutation_uniform] += normal_noise_for_mutation[
        mutation_probability > mutation_uniform]
    # crossover (pl.krzyżowanie równomierne)
    crossover_uniform = rng.uniform(0, 1, population_size)

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


if __name__ == '__main__':
    prepare_logging()
    n_features = 2
    population_size = 10000
    iterations = 5000
    mutation_probability = 0.3
    crossover_probability = 0.6
    mutation_strength = 10

    population = initialize_population(n_features, population_size)
    bests = genetic_algorithm(population, iterations, mutation_probability, crossover_probability, simple.f1,
                              mutation_strength)

    plt.plot(list(range(1, len(bests) + 1)), bests)
    plt.savefig("./plot1.jpg")
