import numpy as np
from cec2017 import simple
from cec2017 import hybrid
from cec2017 import composition
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


logger = logging.getLogger("PrematureConvergenceDetection")


def evolutionary_algorithm(population, n_iterations, mutation_probability, crossover_probability, cost_function,
                           mutation_strength, reproduction_fnc=reproductions.rank_selection,
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
    stopped_in_iteration_naive = -1
    score_when_stopped_naive = -1
    stopped_in_iteration_std = -1
    stopped_in_iteration_std_1 = -1
    score_when_stopped_std = -1
    score_when_stopped_std_1 = -1
    stuck_std = False
    stuck_naive = False
    stuck_std_1 = False
    for iteration in range(1, n_iterations + 1):
        if not stuck_naive and premature_convergence_algorithms.naive_stop(data_collector, iteration - 1, 10, 1.001):
            stopped_in_iteration_naive = iteration - 1
            score_when_stopped_naive = best_individual_value
            stuck_naive = True
        if not stuck_std and premature_convergence_algorithms.naive_stop(data_collector, iteration - 1, 25, 1.001):
            stopped_in_iteration_std = iteration - 1
            score_when_stopped_std = best_individual_value
            stuck_std = True

        if not stuck_std_1 and premature_convergence_algorithms.naive_stop(data_collector, iteration - 1, 50, 1.001):
            stopped_in_iteration_std_1 = iteration - 1
            score_when_stopped_std_1 = best_individual_value
            stuck_std_1 = True

        children = reproduction_fnc(population, scores)
        # genetic operations
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability,
                                                       mutation_strength, crossover=False)
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
    return stopped_in_iteration_std, score_when_stopped_std, best_individual_value, stopped_in_iteration_naive,\
           score_when_stopped_naive, data_collector, score_when_stopped_std_1, stopped_in_iteration_std_1


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


def check_on_one_fnc(cost_function, name, stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_when_stopped_std_1, stop_in_iteration_std_1):
    logger.info(f"start: {name}")
    n_features = 10
    population_size = 500
    n_iterations = 200
    mutation_probability = 0.2
    crossover_probability = 0.6
    mutation_strength = 5
    population = initialize_population(n_features, population_size)
    stopped_in_iteration_std, score_when_stopped_std, best_individual_value, stopped_in_iteration_naive, \
    score_when_stopped_naive, data_collector, score_when_stopped_std_1, stopped_in_iteration_std_1 = \
        evolutionary_algorithm(population, n_iterations, mutation_probability,
                                                    crossover_probability, cost_function,
                                                    mutation_strength)
    stop_itr_for_naive.append(stopped_in_iteration_naive)
    stop_value_for_naive.append(score_when_stopped_naive)
    stop_itr_for_std.append(stopped_in_iteration_std)
    stop_value_for_std.append(score_when_stopped_std)
    best_individual.append(best_individual_value)
    stop_when_stopped_std_1.append(score_when_stopped_std_1)
    stop_in_iteration_std_1.append(stopped_in_iteration_std_1)


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


def count_TR(diff):
    TP_1 = 0
    FP_1 = 0
    TP_2 = 0
    FP_2 = 0
    TP_05 = 0
    FP_05 = 0
    for i in range(len(diff)):
        if diff[i] < 0.02:
            TP_2 += 1
        else:
            FP_2 += 1
        if diff[i] < 0.01:
            TP_1 += 1
        else:
            FP_1 += 1
        if diff[i] < 0.005:
            TP_05 += 1
        else:
            FP_05 += 1
    return TP_05, FP_05, TP_1, FP_1, TP_2, FP_2


if __name__ == '__main__':
    prepare_logging()
    stop_itr_for_naive = []
    stop_value_for_naive = []
    stop_itr_for_std = []
    stop_value_for_std = []
    best_individual = []
    stop_itr_for_std_1 = []
    stop_value_for_std_1 = []
    for i in range(25):
        check_on_one_fnc(simple.f1, f"{1}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(simple.f3, f"{3}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(simple.f5, f"{5}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(simple.f7, f"{7}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(simple.f9, f"{9}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(hybrid.f11, f"{11}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(hybrid.f13, f"{13}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(hybrid.f15, f"{15}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(hybrid.f17, f"{17}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(hybrid.f19, f"{19}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(composition.f21, f"{21}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(composition.f23, f"{23}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(composition.f25, f"{25}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(composition.f28, f"{28}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    for i in range(25):
        check_on_one_fnc(composition.f30, f"{30}_run{i}", stop_itr_for_naive, stop_value_for_naive, stop_itr_for_std,
                         stop_value_for_std, best_individual, stop_value_for_std_1, stop_itr_for_std_1)
    difference_for_naive = []
    difference_for_std = []
    difference_for_std_1 = []
    null_values_for_std_1 = 0
    null_values_for_naive = 0
    null_values_for_std = 0
    for i in range(len(best_individual)):
        if stop_value_for_naive[i] == -1:
            null_values_for_naive += 1
        else:
            difference_for_naive.append(abs(stop_value_for_naive[i] - best_individual[i]) / best_individual[i])
        if stop_value_for_std[i] == -1:
            null_values_for_std += 1
        else:
            difference_for_std.append(abs(stop_value_for_std[i] - best_individual[i]) / best_individual[i])
        if stop_value_for_std_1[i] == -1:
            null_values_for_std_1 += 1
        else:
            difference_for_std_1.append((abs(stop_value_for_std_1[i] - best_individual[i]) / best_individual[i]))

    print("For naive:")

    print(np.mean(difference_for_naive))
    print(null_values_for_naive)
    print(np.mean(stop_itr_for_naive))
    TP_05, FP_05, TP_1, FP_1, TP_2, FP_2 = count_TR(difference_for_naive)
    print(TP_05, FP_05, TP_1, FP_1, TP_2, FP_2)
    print("For std:")

    print(np.mean(difference_for_std))
    print(null_values_for_std)
    print(np.mean(stop_itr_for_std))
    TP_05, FP_05, TP_1, FP_1, TP_2, FP_2 = count_TR(difference_for_std)
    print(TP_05, FP_05, TP_1, FP_1, TP_2, FP_2)

    print("For std_1:")

    print(np.mean(difference_for_std_1))
    print(null_values_for_std_1)
    print(np.mean(stop_itr_for_std_1))
    TP_05, FP_05, TP_1, FP_1, TP_2, FP_2 = count_TR(difference_for_std_1)
    print(TP_05, FP_05, TP_1, FP_1, TP_2, FP_2)
