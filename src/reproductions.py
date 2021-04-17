import numpy as np


def roulette_wheel(population, scores):
    """Basic roulette wheel reproduction (selection). (pl. reprodukcja proporcjonalna/ruletkowa)

    Firstly computes probabilities of choice, then preforms selection.
    Note:
        For minimization problem.

    """
    scores_scaled = (scores - scores.min())/(scores.max() - scores.min())
    scores_minimization = 1 - scores_scaled
    probabilities = scores_minimization / scores_minimization.sum()
    # create random generator (new numpy code should follow this way of using random module)
    rng = np.random.default_rng()
    selected_idx = rng.choice(population.shape[0], population.shape[0], p=probabilities, axis=0)
    selected = population[selected_idx]
    return selected

