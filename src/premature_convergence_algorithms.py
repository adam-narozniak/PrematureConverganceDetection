"""This is our project assignment."""
import logging
import numpy as np

logger = logging.getLogger("PrematureConvergenceDetection")



def naive_stop(data_collector, iteration):
    """
    Naive looking 25 iterations back

    Args:
        population:
        scores:
        iteration: number of previous iteration
    """
    back_by = 25
    if iteration > back_by and data_collector.best_scores[iteration - back_by] / data_collector.best_scores[iteration] < 1.1:
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
            if iteration > look_back_by and data_collector.best_scores[iteration - look_back_by] / data_collector.best_scores[iteration]  < ratio:
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
    if data_collector.stds[iteration].all() < factor:
        logger.info("algorithm stuck in local optimum")
        return True
    else:
        return False
