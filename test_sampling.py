import random
from math import floor

from torch.utils.data import DataLoader

from data import DataProvider, Dataset
from genetic_algorithm.construction_heuristic import NormalConstructionHeuristic
from genetic_algorithm.crossover_operator import UniformCrossoverOperator
from genetic_algorithm.fitness_calculator import ErrorFitnessCalculator
from genetic_algorithm.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from genetic_algorithm.mutation_operator import FactorMutation
from genetic_algorithm.selection_behaviour import TournamentSelection


def main():
    train, test = DataProvider.provide_data_set(Dataset.MNIST)
    train_loader = DataLoader(train, batch_size=64)

    train_loaded = [x for x in train_loader]

    sample_dict = {
        '0.005': [__sample_dataset(train_loaded, 0.1) for _ in range(10)],
        '0.01': [__sample_dataset(train_loaded, 0.25) for _ in range(10)],
        '0.05': [__sample_dataset(train_loaded, 0.5) for _ in range(10)]
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

    current_train = train_loaded[0]
    algorithm_wrapper_genetic.train(train_loader, train_loader)
    print('train 1 done..')

    result_dict_train_1 = {
        '0.005': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.005']],
        '0.01': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.01']],
        '0.05': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.05']]
    }

    current_train = train_loaded[0]
    algorithm_wrapper_genetic.train(train_loader, train_loader)
    print('train 2 done..')

    result_dict_train_2 = {
        '0.005': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.005']],
        '0.01': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.01']],
        '0.05': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.05']]
    }

    current_train = train_loaded[0]
    algorithm_wrapper_genetic.train(train_loader, train_loader)
    print('train 3 done..')

    result_dict_train_3 = {
        '0.005': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.005']],
        '0.01': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.01']],
        '0.05': [algorithm_wrapper_genetic.test(x) for x in sample_dict['0.05']]
    }

    print(result_dict_train_1)
    print(result_dict_train_2)
    print(result_dict_train_3)

    history = [
        [(result_dict_train_1['0.005'][x], result_dict_train_2['0.005'][x], result_dict_train_3['0.005'][x]) for x in
         range(len(result_dict_train_2['0.005']))],
        [(result_dict_train_1['0.01'][x], result_dict_train_2['0.01'][x], result_dict_train_3['0.01'][x]) for x in
         range(len(result_dict_train_2['0.01']))],
        [(result_dict_train_1['0.05'][x], result_dict_train_2['0.05'][x], result_dict_train_3['0.05'][x]) for x in
         range(len(result_dict_train_2['0.05']))]
    ]

    print('history 0.005')
    print(history[0])
    print('history 0.01')
    print(history[1])
    print('history 0.05')
    print(history[2])


def __sample_dataset(dataset, rate):
    return random.sample(dataset, floor(rate * len(dataset)))


if __name__ == '__main__':
    main()
