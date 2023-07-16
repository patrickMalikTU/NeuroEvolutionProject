from abc import abstractmethod

import torch

from neural_network.network import LeNetFromPaper


class FitnessCalculator:

    @abstractmethod
    def __init__(self, fitness_data):
        self.fitness_data = fitness_data

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def calculate_fitness(self, solution_representation: list[float]):
        pass


class AccuracyFitnessCalculator(FitnessCalculator):

    def calculate_fitness(self, solution_representation: list[float]):
        model = LeNetFromPaper.network_from_weight_list(solution_representation)

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.fitness_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = (100 * correct / total)

        return accuracy


class ErrorFitnessCalculator(FitnessCalculator):

    def calculate_fitness(self, solution_representation: list[float]):
        model = LeNetFromPaper.network_from_weight_list(solution_representation)

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.fitness_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = (100 * correct / total)

        return 100 - accuracy


class MeanSquaredErrorOnClasses(FitnessCalculator):

    def calculate_fitness(self, solution_representation: list[float]):
        raise NotImplementedError('not yet implemented')
