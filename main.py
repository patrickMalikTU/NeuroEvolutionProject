# Load in relevant libraries, and alias where appropriate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from neural_network.le_net_algorithm_wrapper import LeNetAlgorithmWrapper
from neural_network.network import LeNetFromPaper
from util import state_dict_to_list


def main():
    k = 10
    batch_size = 64
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()
                                               ]),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize((32, 32)),
                                                  transforms.ToTensor()]),
                                              download=True)

    algorithm_wrapper = LeNetAlgorithmWrapper(0.001, 10)

    loss_list = []
    error_list = []

    # k-fold crossvalidation testing
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

        validation_sampler = SubsetRandomSampler(val_idx)
        validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)

        loss_list_results, error_list_results = algorithm_wrapper.train(train_loader, validation_loader)
        loss_list += loss_list_results
        error_list += error_list_results

    # test results
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    error = algorithm_wrapper.test(test_loader)
    print(f'{100 - error}% accuracy')

    # plot data
    plot_loss_and_error(loss_list, error_list)


def plot_loss_and_error(loss_list, error_list):
    if len(loss_list) != len(error_list):
        raise ValueError('lists must be equally long')

    epoch_list = [x + 1 for x in range(len(loss_list))]
    plt.plot(epoch_list, error_list, label='validation error')
    plt.plot(epoch_list, loss_list, label='trainings loss')
    plt.show()


def test():
    # Define relevant variables for the ML task
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 2

    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()
                                               ]),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize((32, 32)),
                                                  transforms.ToTensor()]),
                                              download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    model = LeNetFromPaper().to(device)

    print(model)

    # Setting the loss function
    cost = nn.CrossEntropyLoss()

    # Setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[2, 5, 8, 12], gamma=0.5)

    # this is defined to print how many steps are remaining when training
    total_step = len(train_loader)

    current_step = 0
    step_list = []
    loss_list = []
    error_list = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 400 == 0:
                loss_value = loss.item()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss_value))

                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    error = 100 - (100 * correct / total)

                    # print('Accuracy of the network on the 10000 test images: {} %'.format(accuracy))

                current_step += i + 1
                step_list.append(current_step)
                loss_list.append(loss_value)
                error_list.append(error)

        # scheduler.step()

    # plt.plot(step_list, error_list)
    # plt.plot(step_list, loss_list)
    # plt.show()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        error = 100 - (100 * correct / total)

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 - error))

    weight_list = state_dict_to_list(model.state_dict())
    loaded_model = LeNetFromPaper.network_from_weight_list(weight_list)

    print(weight_list)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = loaded_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        error = 100 - (100 * correct / total)

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 - error))


if __name__ == '__main__':
    test()
