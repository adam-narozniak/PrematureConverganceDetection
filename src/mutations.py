def mutate(population, mutation_probability, mutation_strength, rng):
    """
    Add normal noise to mutation. Cut to clip to -100, 100. This is our search range.
    Args:
        population:
        mutation_probability:
        mutation_strength: (sigma), factor for multiplication

    Returns:
        mutated probability
    """
    population_size = population.shape[0]
    n_features = population.shape[1]

    # mutate
    mutation_uniform = rng.uniform(0, 1, population_size)
    normal_noise_for_mutation = rng.standard_normal(population_size *
                                                    n_features).reshape(population_size, n_features) * mutation_strength
    population[mutation_probability > mutation_uniform] += normal_noise_for_mutation[
        mutation_probability > mutation_uniform]
    population = population.clip(min=-100, max=100)
    return population
