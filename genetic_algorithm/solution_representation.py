class SolutionRepresentation:

    def __init__(self, solution_representation, fitness_calculator, fitness=None):
        self.fitness = fitness
        self.fitness_calculator = fitness_calculator
        self.solution_representation = solution_representation

    def calculate_fitness(self):
        if self.fitness is None:
            self.fitness = self.fitness_calculator.calculate_fitness(self)

        return self.fitness
