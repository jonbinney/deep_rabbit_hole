"""
Performs hyperparameter optimization to train the model according to train.py
It uses Ray Tune

TODO:
 - Implement actual interesting parameters
 - Implement early termination
"""
from functools import partial
from ray import tune
from ray import train
from train import do_training


params = {
    # Some parameters are variable, for searching optimization
    "n_epochs": tune.grid_search([2, 5]),
    "dropout_p": tune.grid_search([0.1, 0.2]),
    # Others are fixed
    "train_dataset_dir": "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/datasets/water_train_set4",
    "test_dataset_dir": "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/datasets/water_test_set5",
    "annotations_file": "filtered.csv",
    "learning_rate": 1e-3,
    "crop_box": None,
    "report_fn": train.report
}

def train_adapter(config):
    return do_training(**config)

if __name__ == '__main__':
    analysis = tune.run(
        train_adapter,
        resources_per_trial={"gpu": 1},  # Ray Tune will run as many parallel experiments as <avaliable GPUs> / <gplus per experiment>
        config=params,
        mode="min",
        num_samples=3,
    )