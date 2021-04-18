import numpy as np
import math as m


def crossover(population, crossover_probability, rng):
    """Perform crossover operation (pl. krzyżownie jednopunktowe)"""
    population_size = population.shape[0]
    n_features = population.shape[1]
    crossover_uniform = rng.uniform(0, 1, population_size)

    for i in range(m.floor(population_size / 2)):
        if crossover_probability > crossover_uniform[i]:
            genome_crossover_point = np.random.randint(1, high=n_features)
            temp = population[i][:genome_crossover_point]
            population[i][:genome_crossover_point] = population[population_size-i-1][:genome_crossover_point]
            population[population_size-i-1][:genome_crossover_point] = temp
            # for j in range(0, genome_crossover_point):
            #     swapper = population[i][j]
            #     population[i][j] = population[population_size - i - 1][j]
            #     population[population_size - i - 1][j] = swapper
    return population

# testy, potem do usuniecia
if __name__ == '__main__':
    rng = np.random.default_rng()
    population = rng.uniform(-100, 100, 5 * 4).reshape(4, 5)
    print(population)
    print('')
    population = crossover(population, 1, rng)
    print(population)
