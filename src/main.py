import numpy as np
from cec2017 import simple
import reproductions
import successions
import logging
import matplotlib.pyplot as plt
import premature_convergence_algorithms

def genetic_algorithm(population, iterations, mutation_probability, crossover_probability, cost_function,
                      mutation_strength):
    """Genetic algorithm based on holland algorithm."""
    logging.info("Generic algorithm started")
    if population.ndim != 2:
        raise Exception(f"Wrong format of population")

    population_size = population.shape[0]
    n_features = population.shape[1]

    if population_size < 1:
        raise Exception(f"Population should be positive number, not {population_size}")
    if n_features < 1:
        raise Exception(f"Number of features should be positive number, not {population_size}")

    scores = evaluate(population, cost_function)
    best_individual = find_best_score(scores)
    bests = []
    bests.append(best_individual)
    for iteration in range(1,iterations+1):
        if premature_convergence_algorithms.naive_stop(population, scores, bests):
            break

        children = reproductions.roulette_wheel(population, scores)

        # don't get why it's grouped together, seems like they are separate operations
        # crossover not always during evolutionary algs
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability,
                                                       mutation_strength)

        scores_mutants = evaluate(mutants_and_crossovered, cost_function)
        best_individual_in_iteration = find_best_score(scores_mutants)

        best_individual = best_individual if best_individual < best_individual_in_iteration \
            else best_individual_in_iteration
        population, scores = successions.generative(population, mutants_and_crossovered, scores, scores_mutants)
        logging.info(f"Iteration {iteration:3d}/{iterations} completed, best individual score: {best_individual:.02e}")
        bests.append(best_individual)
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
    return scores.min(axis=0)





def mutate_and_crossover(population, mutation_probability, crossover_probability, mutation_strength, feature_crossover_probability=0.5):
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
    logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    prepare_logging()
    n_features = 10
    population_size = 100
    iterations = 500
    mutation_probability = 0.1
    crossover_probability = 0.6
    mutation_strength = 1

    population = initialize_population(n_features, population_size)
    bests = genetic_algorithm(population, iterations, mutation_probability, crossover_probability, simple.f1,
                      mutation_strength)

    plt.plot(list(range(1, len(bests)+1)), bests)
    plt.savefig("./plot1.jpg")