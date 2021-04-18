"""This is our project assignment."""
import logging
import numpy as np

logger = logging.getLogger("PrematureConvergenceDetection")


def naive_stop(population, scores, bests, iteration):
    """
    Naive looking 25 iterations back

    Args:
        population:
        scores:
        iteration: number of previous iteration
    """
    if len(bests) > 50 and bests[-25] / bests[-1] < 1.05:
        logger.info("algorithm stuck in local optimum")
        return True
    return False


def individual_outside_std(population, data_collector, factor, iteration):
    """
    Classify as futile when there are no individuals outside factor * std. Answers question if we should stop algorithm.
    Args:
        population:
        data_collector:
        factor:
        iteration: number of previous iteration
    Returns:
        True when stuck in optimum
        False when still exploring
    """
    means = data_collector.means[iteration]
    upper_bound = means + factor * data_collector.stds[iteration]
    lower_bound = means - factor * data_collector.stds[iteration]
    greater_than_upper = np.array(upper_bound < population.max(axis=0)).any()
    smaller_than_lower = np.array(lower_bound > population.min(axis=0)).any()
    # check if there are any individuals greater or smaller than factor * std
    if greater_than_upper or smaller_than_lower:
        return False  # there are still exploring individuals
    else:
        logger.info("algorithm stuck in local optimum")
        return True
