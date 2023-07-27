import random
from abc import abstractmethod
from math import sqrt

from genetic_algorithm.construction_heuristic import SOLUTION_SIZE
from genetic_algorithm.solution_representation import SolutionRepresentation


class MutationOperator:

    @abstractmethod
    def __init__(self, mutation_rate=None, solution_size=SOLUTION_SIZE):
        if mutation_rate is None:
            mutation_rate = 1 / solution_size
        self.mutation_rate = mutation_rate

        self.solution_indices = [x for x in range(solution_size)]

    @abstractmethod
    def mutate(self, population) -> None:
        pass

    def index_list_by_chance(self, probability, n):
        mu = probability * n
        sigma = sqrt(n * probability * (1 - probability))
        num_bits = round(random.gauss(mu, sigma))
        if num_bits < 0:
            num_bits = 0
        return random.sample(self.solution_indices, num_bits)


class SwapMutation(MutationOperator):
    
    def __init__(self, mutation_rate=None, solution_size=SOLUTION_SIZE):
        super(SwapMutation, self).__init__(mutation_rate, solution_size)

    def mutate(self, population: list[SolutionRepresentation]) -> None:
        n = len(self.solution_indices)

        for individual in population:
            indices = self.index_list_by_chance(self.mutation_rate, n)
            for i in indices:
                random_gene_index = random.randint(0, len(individual.solution_representation) - 1)
                val = individual.solution_representation[i]
                individual.solution_representation[i] = individual.solution_representation[random_gene_index]
                individual.solution_representation[random_gene_index] = val


class ZeroMutation(MutationOperator):
    
    def __init__(self, mutation_rate=None, solution_size=SOLUTION_SIZE):
        super(ZeroMutation, self).__init__(mutation_rate, solution_size)

    def mutate(self, population: list[SolutionRepresentation]) -> None:
        n = len(self.solution_indices)

        for individual in population:
            indices = self.index_list_by_chance(self.mutation_rate, n)
            for i in indices:
                individual.solution_representation[i] = 0


class FactorMutation(MutationOperator):

    def __init__(self, min_factor=0.9, max_factor=1.1, mutation_rate=None):
        super(FactorMutation, self).__init__(mutation_rate)
        self.max_factor = max_factor
        self.min_factor = min_factor

    def mutate(self, population) -> None:
        n = len(self.solution_indices)

        for individual in population:
            indices = self.index_list_by_chance(self.mutation_rate, n)
            for i in indices:
                factor = random.uniform(self.min_factor, self.max_factor)
                individual.solution_representation[i] *= factor


class BinomialDistributionMutationOperator(MutationOperator):
    def __init__(self, norm_sigma=0.01, mutation_rate=None):
        super(BinomialDistributionMutationOperator, self).__init__(mutation_rate)
        self.norm_sigma = norm_sigma

    def mutate(self, population) -> None:
        n = len(self.solution_indices)

        for individual in population:
            indices = self.index_list_by_chance(self.mutation_rate, n)
            for i in indices:
                individual.solution_representation[i] = random.gauss(individual.solution_representation[i],
                                                                     self.norm_sigma)
