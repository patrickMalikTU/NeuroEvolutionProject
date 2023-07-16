import random
from abc import abstractmethod
from math import floor

from genetic_algorithm.solution_representation import SolutionRepresentation


class CrossoverOperator:

    @abstractmethod
    def __init__(self, crossover_rate):
        self.crossover_rate = crossover_rate

    @abstractmethod
    def crossover(self, population: list[SolutionRepresentation]) -> list[SolutionRepresentation]:
        pass


class UniformCrossoverOperator(CrossoverOperator):

    def __init__(self, crossover_rate, uniform_prob=0.5):
        super(UniformCrossoverOperator, self).__init__(crossover_rate)
        self.uniform_prob = uniform_prob

    def crossover(self, population: list[SolutionRepresentation]) -> list[SolutionRepresentation]:
        fitness_calculator = population[0].fitness_calculator
        n_crossovers = floor((self.crossover_rate * len(population)) / 2)

        crossed_over = []

        for _ in range(n_crossovers):
            parents = random.sample(population, 2)

            first_representation = parents[0].solution_representation
            second_representation = parents[1].solution_representation

            first_result = []
            second_result = []

            for i in range(len(first_representation)):
                if random.random() < self.uniform_prob:
                    first_result.append(first_representation[i])
                    second_result.append(second_representation[i])
                else:
                    first_result.append(second_representation[i])
                    second_result.append(first_representation[i])

            crossed_over += [
                SolutionRepresentation(first_result, fitness_calculator),
                SolutionRepresentation(second_result, fitness_calculator)
            ]

        return crossed_over
