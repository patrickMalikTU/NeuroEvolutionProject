# Load in relevant libraries, and alias where appropriate
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from data import DataProvider, Dataset
from genetic_algorithm.construction_heuristic import NormalConstructionHeuristic
from genetic_algorithm.crossover_operator import UniformCrossoverOperator
from genetic_algorithm.fitness_calculator import ErrorFitnessCalculator
from genetic_algorithm.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from genetic_algorithm.mutation_operator import FactorMutation
from genetic_algorithm.selection_behaviour import TournamentSelection
from neural_network.le_net_algorithm_wrapper import LeNetAlgorithmWrapper


def main():
    k = 5
    batch_size = 64
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    train_dataset, test_dataset = DataProvider.provide_data_set(Dataset.MNIST)

    algorithm_wrapper_le_net = LeNetAlgorithmWrapper(0.001, 10)
    algorithm_wrapper_genetic = GeneticAlgorithmWrapper(
        50,
        50,
        NormalConstructionHeuristic(),
        TournamentSelection(3),
        UniformCrossoverOperator(0.8),
        FactorMutation(mutation_rate=.05),
        ErrorFitnessCalculator
    )

    # loss_list = []
    error_list = []

    algorithm_wrapper = algorithm_wrapper_genetic

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

    # test results
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    error = algorithm_wrapper.test(test_loader)
    print(f'{100 - error}% accuracy')

    # plot data
    # plot_loss_and_error(loss_list, error_list)
    plt.plot(error_list)
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
