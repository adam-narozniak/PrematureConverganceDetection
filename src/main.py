from src.cec17_test_func.cec17_functions import cec17_test_func


# minimalization
def holland(population, iterations, mutation_probability, crossover_probability):
    if population.ndims != 2:
        raise Exception(f"Wrong format of population")
    population_length = population.shape[0]
    n_features = population.shape[1]
    if population_length < 1:
        raise Exception(f"Population should be positive number, not {population_length}")
    if n_features < 1:
        raise Exception(f"Number of features should be positive number, not {population_length}")

    scores = evaluate(population, cost_function)
    best = find_best(population, scores)

    for iteration in range(iterations):
        '''if stop_criterion_satisfied(population, scores):
            break
        # during lectures third parameter was passed - length, no use of that
        children = reproduce(population, scores)

        # don't get why it's grouped together, seems like they are separate operations
        # crossover not always during evolutionary algs
        mutants_and_crossovered = mutate_and_crossover(children, mutation_probability, crossover_probability)

        scores_mutants = evaluate(mutants_and_crossovered, cost_function)
        iteration_best = find_best(mutants_and_crossovered, scores)

        best = best if best < iteration_best else iteration_best
        population, scores = succession(population, mutants_and_crossovered, scores, scores_mutants)
'''
    # and what it returns???


def evaluate(population, cost_function):
    # x: Solution vector
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # nx: Number of dimensions
    nx = 10
    # mx: Number of objective functions
    mx = 1
    # func_num: Function number
    func_num = 1
    # Pointer for the calculated fitness
    f = [0]
    cec17_test_func(x, f, nx, mx, 10)
    print(f[0])


def find_best(scores):
    pass


def reproduce(population, scores):
    pass


def succession(population , mutants, scores, scores_mutants):
    pass


def cost_function():
    pass


if __name__ == '__main__':
    print("Hello word")
    evaluate(0, 0)

