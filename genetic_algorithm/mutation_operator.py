import random
from abc import abstractmethod

from genetic_algorithm.construction_heuristic import SOLUTION_SIZE
from genetic_algorithm.solution_representation import SolutionRepresentation


class MutationOperator:

    @abstractmethod
    def __init__(self, mutation_rate=1 / SOLUTION_SIZE):
        self.mutation_rate = mutation_rate

    @abstractmethod
    def mutate(self, population) -> None:
        pass


class SwapMutation(MutationOperator):

    def mutate(self, population: list[SolutionRepresentation]) -> None:
        for individual in population:
            for i in range(len(individual.solution_representation)):
                if random.random() < self.mutation_rate:
                    random_gene_index = random.randint(0, len(individual.solution_representation) - 1)
                    val = individual.solution_representation[i]
                    individual.solution_representation[i] = individual.solution_representation[random_gene_index]
                    individual.solution_representation[random_gene_index] = val


class ZeroMutation(MutationOperator):

    def mutate(self, population: list[SolutionRepresentation]) -> None:
        for individual in population:
            for i in range(len(individual.solution_representation)):
                if random.random() < self.mutation_rate:
                    individual.solution_representation[i] = 0


class FactorMutation(MutationOperator):

    def __init__(self, min_factor=0.9, max_factor=1.1, mutation_rate=1 / SOLUTION_SIZE):
        super(FactorMutation, self).__init__(mutation_rate)
        self.max_factor = max_factor
        self.min_factor = min_factor

    def mutate(self, population) -> None:
        for individual in population:
            for i in range(len(individual.solution_representation)):
                if random.random() < self.mutation_rate:
                    factor = random.uniform(self.min_factor, self.max_factor)
                    individual.solution_representation[i] *= factor


class NormalDistributionMutationOperator(MutationOperator):
    def __init__(self, norm_sigma=0.01, mutation_rate=1 / SOLUTION_SIZE):
        super(NormalDistributionMutationOperator, self).__init__(mutation_rate)
        self.norm_sigma = norm_sigma

    def mutate(self, population) -> None:
        for individual in population:
            for i in range(len(individual.solution_representation)):
                if random.random() < self.mutation_rate:
                    individual.solution_representation[i] = random.gauss(individual.solution_representation[i],
                                                                         self.norm_sigma)
