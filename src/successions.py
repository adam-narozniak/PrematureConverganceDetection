import pandas as pd
import numpy as np


def generative(population, mutants, scores, scores_mutants):
    """
    Replace current population with mutants.

    Returns:
        (np.ndarray): new population size: (m_individuals, n_features),
            (np.ndarray): corresponding scores size:(m_individuals, 1)
    """
    return mutants, scores_mutants


def elite(population, mutants, scores, scores_mutants, elite_n):
    """
    Replace current population with n best form this population and the rest from mutants.
    Args:
        elite_n: if int then than n best, if float then take n * 100% best

    Returns:
        (np.ndarray): new population size: (m_individuals, n_features),
            (np.ndarray): corresponding scores size:(m_individuals, 1)
    """
    population_size = population.shape[0]
    n_features = population.shape[1]
    if type(elite_n) == int:
        pass
    elif type(elite_n) == float:
        elite_n = int(population_size * elite_n)
    else:
        raise Exception(f"elite_n should be int or float")
    if elite_n > population_size:
        raise Exception(f"elite_n should be smaller than population size")
    mutants_and_scores = pd.DataFrame(mutants)
    mutants_and_scores = pd.concat([mutants_and_scores, pd.Series(scores_mutants, name="scores")], axis=1).sort_values(
        by="scores", ascending=True)
    population_and_scores = pd.DataFrame(population)
    population_and_scores = pd.concat([population_and_scores, pd.Series(scores, name="scores")], axis=1).sort_values(
        by="scores", ascending=True)
    elite_population_m = population_size - elite_n
    elite_population = population_and_scores.iloc[:elite_population_m, 0:n_features]
    elite_mutants = mutants_and_scores.iloc[:elite_n, 0:n_features]
    new_population = np.concatenate([elite_population.values, elite_mutants.values], axis=0)
    new_scores = np.concatenate([population_and_scores.iloc[:elite_population_m, n_features:].values,
                                 mutants_and_scores.iloc[:elite_n, n_features:].values], axis=0).reshape(-1)
    return new_population, new_scores
