from abc import abstractmethod


class algorithm_wrapper:

    @abstractmethod
    def train(self, train_data, validation_data):
        pass

    @abstractmethod
    def test(self, test_data):
        pass

    @abstractmethod
    def get_current_model(self):
        pass
