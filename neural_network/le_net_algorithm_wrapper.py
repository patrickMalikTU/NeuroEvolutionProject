import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_network.network import LeNetFromPaper
from run_environment import algorithm_wrapper


class LeNetAlgorithmWrapper(algorithm_wrapper):

    def __init__(self, learning_rate, epochs):
        super(LeNetAlgorithmWrapper, self).__init__()
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LeNetFromPaper().to(self.device)
        self.cost = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.epochs = epochs

    def train(self, train_data: DataLoader, validation_data: DataLoader):
        loss_list = []
        error_list = []
        for epoch in range(self.epochs):
            # training run
            loss_list.append(self.__training_run(train_data))

            # validation run
            error_list.append(self.__validation_run(validation_data))

        return error_list

    def __training_run(self, training_data):
        for i, (images, labels) in enumerate(training_data):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.cost(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def __validation_run(self, validation_data):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            error = 100 - (100 * correct / total)

        return error

    def test(self, test_data: DataLoader):
        return self.__validation_run(test_data)

    def get_current_model(self):
        return self.model
