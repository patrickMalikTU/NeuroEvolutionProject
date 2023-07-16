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


class NPointCrossoverOperator(CrossoverOperator):

    def __init__(self, crossover_rate, crossover_n=2):
        super(NPointCrossoverOperator, self).__init__(crossover_rate)
        self.crossover_n = crossover_n

    def crossover(self, population: list[SolutionRepresentation]) -> list[SolutionRepresentation]:
        fitness_calculator = population[0].fitness_calculator
        n_crossovers = floor((self.crossover_rate * len(population)) / 2)

        crossed_over = []

        for _ in range(n_crossovers):
            parents = random.sample(population, 2)

            first_representation = parents[0].solution_representation
            second_representation = parents[1].solution_representation

            indices = random.sample(range(0, len(first_representation)), self.crossover_n)
            indices.sort()

            first_result = []
            second_result = []

            prev_index = 0
            toggle = True
            for index in indices:
                if toggle:
                    first_result += first_representation[prev_index:index]
                    second_result += second_representation[prev_index:index]
                else:
                    first_result += second_representation[prev_index:index]
                    second_result += first_representation[prev_index:index]

                prev_index = index
                toggle = not toggle

            if toggle:
                first_result += first_representation[prev_index:]
                second_result += second_representation[prev_index:]
            else:
                first_result += second_representation[prev_index:]
                second_result += first_representation[prev_index:]

            crossed_over += [
                SolutionRepresentation(first_result, fitness_calculator),
                SolutionRepresentation(second_result, fitness_calculator)
            ]

        return crossed_over


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
