from src.cec17_test_func.cec17_functions import cec17_test_func


class Population:
    def __init__(self, size_of_population, number_of_features):
        size = size_of_population
        n_feature = number_of_features


# co dalej?

# minimization
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
    best = find_best(scores)

    for iteration in range(iterations):
        if stop_criterion_satisfied(population, scores):
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

    # and what it returns???


def evaluate(population, cec2017_func_number):
    scores = []
    dimension = population.shape[1]
    mx = 1  # number of function to calculate
    for individual in population:
        placeholder_for_score = [0]
        cec17_test_func(individual, placeholder_for_score, dimension,
                        mx, cec2017_func_number)
        scores.append(placeholder_for_score[0])
    return scores


def find_best(scores):
    index_of_best_individual = 0
    for i in range(scores):
        if scores[i] < scores[index_of_best_individual]:
            index_of_best_individual = i
    return index_of_best_individual


def reproduce(population, scores):
    pass


def succession(population, mutants, scores, scores_mutants):
    pass


def cost_function():
    pass


def stop_criterion_satisfied(population, scores):
    pass


def mutate_and_crossover(children, mutation_probability, crossover_probability):
    pass


if __name__ == '__main__':
    print("Hello word")
    evaluate(0, 0)
