import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, Float, Configuration, Integer
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from smac import HyperparameterOptimizationFacade, RunHistory
from smac import Scenario
from torch.utils.data import SubsetRandomSampler, DataLoader

from data import Dataset, DataProvider
from neural_network.le_net_algorithm_wrapper import LeNetAlgorithmWrapper


class LeNetSmacWrapper():

    def __init__(self):
        super(LeNetSmacWrapper, self).__init__()
        self.train_data, self.test_data = DataProvider.provide_data_set(Dataset.MNIST_FASHION)

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        learning_rate = Float("learning_rate", (0, 0.5), default=0.001)
        cs.add_hyperparameters([learning_rate])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        classifier = LeNetAlgorithmWrapper(config['learning_rate'], epochs=15)

        k = 5
        splits = KFold(n_splits=k, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(self.train_data)))):
            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            train_loader = DataLoader(self.train_data, batch_size=64, sampler=train_sampler)

            validation_sampler = SubsetRandomSampler(val_idx)
            validation_loader = DataLoader(self.train_data, batch_size=64, sampler=validation_sampler)
            classifier.train(train_loader, validation_loader)

        test_loader = torch.utils.data.DataLoader(dataset=self.test_data,
                                                  batch_size=64,
                                                  shuffle=True)

        error = classifier.test(test_loader)

        return error

def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    plt.figure()

    # Plot ground truth
    x = list(np.linspace(-5, 5, 100))
    y = [xi * xi for xi in x]
    plt.plot(x, y)

    # Plot all trials
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x = config["x"]
        y = v.cost  # type: ignore
        plt.scatter(x, y, c="blue", alpha=0.1, zorder=9999, marker="o")

    # Plot incumbent
    plt.scatter(incumbent["x"], incumbent["x"] * incumbent["x"], c="red", zorder=10000, marker="x")

    plt.show()


if __name__ == "__main__":
    model = LeNetSmacWrapper()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, n_trials=50)

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

    print(smac.intensifier.get_incumbent().get_dictionary())
 
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    # Let's plot it too
    plot(smac.runhistory, incumbent)
