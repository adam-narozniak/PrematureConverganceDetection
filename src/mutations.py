def mutate(population, mutation_probability, mutation_strength, rng):
    population_size = population.shape[0]
    n_features = population.shape[1]

    # mutate
    mutation_uniform = rng.uniform(0, 1, population_size)
    normal_noise_for_mutation = rng.standard_normal(population_size *
                                                    n_features).reshape(population_size, n_features) * mutation_strength
    population[mutation_probability > mutation_uniform] += normal_noise_for_mutation[
        mutation_probability > mutation_uniform]
    return population
