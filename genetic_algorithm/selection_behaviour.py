import random
from abc import abstractmethod

from genetic_algorithm.solution_representation import SolutionRepresentation


class SelectionOperator:

    @abstractmethod
    def __init__(self):
        pass

    def select(self, population) -> (SolutionRepresentation, list[SolutionRepresentation]):
        pass

    def get_best_fitness_solution(self, population):
        best = min(population, key=lambda x: x.fitness)
        return SolutionRepresentation(best.solution_representation, best.fitness_calculator, best.fitness)


class TournamentSelection(SelectionOperator):

    def __init__(self, tournament_size):
        self.tournament_size = tournament_size

    def select(self, population):
        population_size = len(population)
        new_pop = []

        for _ in range(population_size):
            tournament = random.sample(population, self.tournament_size)
            winner = min(tournament, key=lambda x: x.fitness)
            new_pop.append(SolutionRepresentation(winner.solution_representation, winner.fitness_calculator))

        return self.get_best_fitness_solution(population), new_pop
