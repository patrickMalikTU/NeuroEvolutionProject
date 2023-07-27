import random
from abc import abstractmethod

from neural_network.network import LeNetFromPaper
from util import state_dict_to_list

SOLUTION_SIZE = 61706


class ConstructionHeuristicOperator:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def construct_solution(self):
        pass


class NNInitializationConstructionHeuristic(ConstructionHeuristicOperator):

    def construct_solution(self):
        return state_dict_to_list(LeNetFromPaper().state_dict())


class UniformConstructionHeuristic(ConstructionHeuristicOperator):

    def __init__(self, min_value=-1., max_value=1.):
        self.min_value = min_value
        self.max_value = max_value

    def construct_solution(self):
        return [random.uniform(self.min_value, self.max_value) for _ in range(SOLUTION_SIZE)]


class NormalConstructionHeuristic(ConstructionHeuristicOperator):

    def __init__(self, mean=0., std_dev=0.2):
        self.mean = mean
        self.std_dev = std_dev

    def construct_solution(self):
        return [random.gauss(self.mean, self.std_dev) for _ in range(SOLUTION_SIZE)]
