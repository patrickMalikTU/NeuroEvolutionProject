import random
from math import floor

from data import DataProvider, Dataset
from genetic_algorithm.construction_heuristic import NormalConstructionHeuristic
from genetic_algorithm.crossover_operator import UniformCrossoverOperator
from genetic_algorithm.fitness_calculator import ErrorFitnessCalculator
from genetic_algorithm.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from genetic_algorithm.mutation_operator import FactorMutation
from genetic_algorithm.selection_behaviour import TournamentSelection


def main():
    train, test = DataProvider.provide_data_set(Dataset.MNIST)

    train_loaded = [x for x in train]

    sample_dict = {
        '0.1': [__sample_dataset(train_loaded, 0.1) for _ in range(10)],
        '0.25': [__sample_dataset(train_loaded, 0.25) for _ in range(10)],
        '0.5': [__sample_dataset(train_loaded, 0.5) for _ in range(10)]
    }

    algorithm_wrapper_genetic = GeneticAlgorithmWrapper(
        10,
        20,
        NormalConstructionHeuristic(),
        TournamentSelection(3),
        UniformCrossoverOperator(0.8),
        FactorMutation(mutation_rate=.05),
        ErrorFitnessCalculator
    )

    algorithm_wrapper_genetic.train(train, train)

    result_dict_train_1 = {
        '0.1': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.1']],
        '0.25': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.25']],
        '0.5': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.5']]
    }

    algorithm_wrapper_genetic.train(train, train)

    result_dict_train_2 = {
        '0.1': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.1']],
        '0.25': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.25']],
        '0.5': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.5']]
    }

    algorithm_wrapper_genetic.train(train, train)

    result_dict_train_3 = {
        '0.1': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.1']],
        '0.25': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.25']],
        '0.5': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.5']]
    }

    print(result_dict_train_1)
    print(result_dict_train_2)
    print(result_dict_train_3)

    history = [
        [(result_dict_train_1['0.1'][x], result_dict_train_2['0.1'][x], result_dict_train_3['0.1'][x]) for x in
         range(len(result_dict_train_2['0.1']))],
        [(result_dict_train_1['0.25'][x], result_dict_train_2['0.25'][x], result_dict_train_3['0.25'][x]) for x in
         range(len(result_dict_train_2['0.25']))],
        [(result_dict_train_1['0.5'][x], result_dict_train_2['0.5'][x], result_dict_train_3['0.5'][x]) for x in
         range(len(result_dict_train_2['0.5']))]
    ]

    print('history 0.1')
    print(history[0])
    print('history 0.25')
    print(history[1])
    print('history 0.5')
    print(history[2])


def __sample_dataset(dataset, rate):
    return random.sample(dataset, floor(rate * len(dataset)))


if __name__ == '__main__':
    main()
