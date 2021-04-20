import numpy as np
import pandas as pd


def roulette_wheel(population, scores):
    """Basic roulette wheel reproduction (selection). (pl. reprodukcja proporcjonalna/ruletkowa)

    Firstly computes probabilities of choice, then preforms selection.
    Note:
        For minimization problem.

    """

    scores_scaled = (scores - scores.min()) / (scores.max() - scores.min() + np.finfo(np.float32).eps)
    scores_minimization = 1 - scores_scaled
    probabilities = scores_minimization / scores_minimization.sum()
    probabilities = np.nan_to_num(probabilities, nan=probabilities.max())
    # create random generator (new numpy code should follow this way of using random module)
    rng = np.random.default_rng()
    selected_idx = rng.choice(population.shape[0], population.shape[0], p=probabilities, axis=0)
    selected = population[selected_idx]
    return selected


def rank_selection(population, scores):
    """Rank selection. (pl. reprodukcja rangowa)"""
    population_size = population.shape[0]
    n_features = population.shape[1]
    scores_scaled = (scores - scores.min()) / (scores.max() - scores.min() + np.finfo(np.float32).eps)
    scores_minimization = 1 - scores_scaled
    # scores_minimization = np.sort(scores_minimization)[::-1]
    population_and_scores = pd.DataFrame(population)
    scores_series = pd.Series(scores_minimization, name="scores")
    population_and_scores = pd.concat([population_and_scores, scores_series], axis=1)
    population_and_scores = population_and_scores.sort_values(by="scores", ascending=False)
    fitness = np.linspace(1, 0.1, population_size, endpoint=True)
    probabilities = fitness / fitness.sum()
    rng = np.random.default_rng()
    selected_idx = rng.choice(population_size, population_size, p=probabilities, axis=0)
    selected = population_and_scores.iloc[selected_idx, 0:n_features].values
    return selected
