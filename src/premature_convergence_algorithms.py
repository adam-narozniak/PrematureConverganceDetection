"""This is our project assignment."""
import logging

import numpy as np


def naive_stop(population, scores, bests):
    """
    Naive looking 25 iterations back

    Args:
        population:
        scores:
    """


    if len(bests) > 50 and bests[-25] / bests[-1] < 1.05:
        logging.info("algorithm stuck in local optimum")
        return True
    return False
