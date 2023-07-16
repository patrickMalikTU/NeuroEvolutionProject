import random
from typing import Type

import torch

from genetic_algorithm.fitness_calculator import FitnessCalculator
from genetic_algorithm.solution_representation import SolutionRepresentation
from neural_network.network import LeNetFromPaper
from run_environment import algorithm_wrapper


class GeneticAlgorithmWrapper(algorithm_wrapper):

    def test(self, test_data):
        return self.__validation_run(test_data)

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

        self.total_best_individual = None
        self.current_population = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_current_model(self):
        weights = self.total_best_individual.solution_representation
        return LeNetFromPaper.network_from_weight_list(weights)

    def train(self, train_data, validation_data):

        self.__training_run(train_data)
        error = self.__validation_run(validation_data)

        return [error]

    def __training_run(self, training_data):
        fitness_calculator = self.fitness_calculator(training_data)

        if self.current_population is None:
            self.current_population = self.__create_population(self.population_size, fitness_calculator)
        population = self.current_population

        for gen in range(self.generations):
            print(f'generation: {gen}')

            self.__calculate_population_fitness(population)
            print('fitness calculated')
            best, selected = self.selection_behaviour.select(population)

            if self.total_best_individual is None or best.fitness < self.total_best_individual.fitness:
                self.total_best_individual = best

            crossed_over = self.crossover_behaviour.crossover(selected)
            self.mutation_behaviour.mutate(crossed_over)

            population = self.__create_next_generation(crossed_over, population, self.total_best_individual)

        self.current_population = population

    def __validation_run(self, validation_data):
        model = self.get_current_model()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            error = 100 - (100 * correct / total)

        return error

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

    def __create_next_generation(self, children, last_generation, best_so_far):
        next_pop = children

        if len(next_pop) < self.population_size:
            next_pop.append(SolutionRepresentation(best_so_far.solution_representation, best_so_far.fitness_calculator,
                                                   best_so_far.fitness))

        missing_pop = self.population_size - len(next_pop)
        if missing_pop > 0:
            next_pop += random.sample(last_generation, missing_pop)

        return next_pop
