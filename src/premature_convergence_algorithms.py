"""This is our project assignment."""
import logging
import numpy as np

logger = logging.getLogger("PrematureConvergenceDetection")


def naive_stop(data_collector, iteration, look_back, ratio):
    """
    Naive looking 25 iterations back

    Args:
        population:
        scores:
        iteration: number of previous iteration
    """

    if iteration > look_back and data_collector.best_scores[iteration - look_back] / data_collector.best_scores[
        iteration] < ratio:
        logger.info("algorithm stuck in local optimum")
        return True
    return False


def naive_stop_cmp(data_collector, variants, iteration):
    """
    Take many variants into account.

    Args:
        data_collector:
        variants: possibilities to check (row look by, column ratio)
    """
    for ratio in variants.columns.values:
        for look_back_by in variants.index.values:
            if variants.loc[look_back_by, ratio] != 0:
                continue
            if iteration > look_back_by and data_collector.best_scores[iteration - look_back_by] / \
                    data_collector.best_scores[iteration] < ratio:
                variants.loc[look_back_by, ratio] = iteration
                logger.info(f"alogrithm stuck based on criteria ratio: {ratio} and look_back_by{look_back_by}")
                variants.loc[look_back_by, ratio] = iteration


def individual_outside_std(population, data_collector, factor, iteration):
    """
    Classify as futile when there are no individuals outside factor > std. Answers question if we should stop algorithm.
    Args:
        population:
        data_collector:
        factor:
        iteration: number of previous iteration
    Returns:
        True when stuck in optimum
        False when still exploring
    """
    are_all_std_less_than_param = True
    stds = np.std(population, axis=0)
    for row in stds:
        if row > factor:
            are_all_std_less_than_param = False
    if are_all_std_less_than_param:
        logger.info("algorithm stuck in local optimum")
        return True
    else:
        return False
