# Load in relevant libraries, and alias where appropriate
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from data import DataProvider, Dataset
from genetic_algorithm.construction_heuristic import NormalConstructionHeuristic
from genetic_algorithm.crossover_operator import FastUniformCrossoverOperator
from genetic_algorithm.fitness_calculator import CrossEntropyLossFitnessCalculator
from genetic_algorithm.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from genetic_algorithm.mutation_operator import BinomialDistributionMutationOperator
from genetic_algorithm.selection_behaviour import TournamentSelection
from neural_network.le_net_algorithm_wrapper import LeNetAlgorithmWrapper


def main():
    k = 5
    batch_size = 64
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    train_dataset, test_dataset = DataProvider.provide_data_set(Dataset.MNIST)

    # tuning results

    # MNIST Fashion
    # Configuration(values={
    #   'epochs': 42,
    #   'learning_rate': 0.007156576937659228,
    # })

    # MNIST
    # LeNetAlgorithmWrapper(0.043564649850770354, 45)

    algorithm_wrapper_le_net = LeNetAlgorithmWrapper(0.007156576937659228, 15)
    algorithm_wrapper_genetic = GeneticAlgorithmWrapper(
        150,
        150,
        NormalConstructionHeuristic(),
        TournamentSelection(4),
        FastUniformCrossoverOperator(crossover_rate=0.8),
        BinomialDistributionMutationOperator(),
        CrossEntropyLossFitnessCalculator
    )

    # loss_list = []
    error_list = []
    acc_list = []

    algorithm_wrapper = algorithm_wrapper_le_net

    begin = time.perf_counter_ns()

    # k-fold crossvalidation testing
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

        validation_sampler = SubsetRandomSampler(val_idx)
        validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)

        error_list_results = algorithm_wrapper.train(train_loader, validation_loader)
        # loss_list_results, error_list_results = algorithm_wrapper.train(train_loader, validation_loader)
        # loss_list += loss_list_results
        error_list += error_list_results

        acc_list.append(algorithm_wrapper.test(train_loader))

    # test results
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    error = algorithm_wrapper.test(test_loader)
    print(f'{100 - error}% accuracy')

    time_taken_in_ns = time.perf_counter_ns() - begin

    print(f'time taken in s: {time_taken_in_ns / 1000000000}')
    print(f'time taken in m: {time_taken_in_ns / 60000000000}')

    # plot data
    # plot_loss_and_error(loss_list, error_list)
    plt.plot(acc_list, label='train error')
    plt.plot(error_list, label='validation error')
    plt.legend()
    plt.show()


def plot_loss_and_error(loss_list, error_list):
    if len(loss_list) != len(error_list):
        raise ValueError('lists must be equally long')

    epoch_list = [x + 1 for x in range(len(loss_list))]
    plt.plot(epoch_list, error_list, label='validation error')
    plt.plot(epoch_list, loss_list, label='trainings loss')
    plt.show()


if __name__ == '__main__':
    main()
