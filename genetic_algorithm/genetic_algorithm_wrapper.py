from typing import Type

from genetic_algorithm.fitness_calculator import FitnessCalculator
from genetic_algorithm.solution_representation import SolutionRepresentation
from run_environment import algorithm_wrapper


class GeneticAlgorithmWrapper(algorithm_wrapper):

    def __init__(self, generations, population_size, construction_heuristic_behaviour, selection_behaviour,
                 crossover_behaviour, mutation_behaviour, fitness_calculator: Type[FitnessCalculator]):
        super(GeneticAlgorithmWrapper, self).__init__()
        self.fitness_calculator = fitness_calculator
        self.generations = generations
        self.population_size = population_size
        self.mutation_behaviour = mutation_behaviour
        self.crossover_behaviour = crossover_behaviour
        self.selection_behaviour = selection_behaviour
        self.construction_heuristic_behaviour = construction_heuristic_behaviour

    def train(self, train_data, validation_data):
        fitness_calculator = self.fitness_calculator(train_data)
        population = self.__create_population(self.population_size, fitness_calculator)

        total_best_individual = None

        for _ in range(self.generations):
            self.__calculate_population_fitness(population)
            best, selected = self.selection_behaviour.select(population)

            if total_best_individual is None or best.fitness < total_best_individual.fitness:
                total_best_individual = best

            crossed_over = self.crossover_behaviour.crossover(population)

            pass

    def __create_population(self, population_size, fitness_calculator) -> list[SolutionRepresentation]:
        return [
            SolutionRepresentation(
                self.construction_heuristic_behaviour.construct_solution(),
                fitness_calculator
            ) for _ in range(population_size)
        ]

    def __calculate_population_fitness(self, population: list[SolutionRepresentation]):
        for individual in population:
            individual.calculate_fitness()
