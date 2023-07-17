from enum import Enum

import torchvision
import torchvision.transforms as transforms


class Dataset(Enum):
    MNIST = 0
    MNIST_FASHION = 1


class DataProvider:

    @staticmethod
    def provide_data_set(dataset: Dataset):
        if dataset == Dataset.MNIST:
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

            return train_dataset, test_dataset

        if dataset == Dataset.MNIST_FASHION:
            train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                              train=True,
                                                              transform=transforms.Compose([
                                                                  transforms.Resize((32, 32)),
                                                                  transforms.ToTensor()
                                                              ]),
                                                              download=True)

            test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                             train=False,
                                                             transform=transforms.Compose([
                                                                 transforms.Resize((32, 32)),
                                                                 transforms.ToTensor()]),
                                                             download=True)

            return train_dataset, test_dataset

        raise ValueError('invalid dataset specified')
