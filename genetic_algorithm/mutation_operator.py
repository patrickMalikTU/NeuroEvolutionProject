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
    pass


class ZeroMutation(MutationOperator):

    def mutate(self, population: list[SolutionRepresentation]) -> None:
        for individual in population:
            for i in range(len(individual.solution_representation)):
                if random.random() < self.mutation_rate:
                    individual.solution_representation[i] = 0


class FactorMutation(MutationOperator):
    pass
