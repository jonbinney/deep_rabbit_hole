"""
Performs hyperparameter optimization to train the model according to train.py
It uses Ray Tune

TODO:
 - Implement early termination
"""
from functools import partial
from ray import tune
from ray import train
from train import do_training


params = {
    # Some parameters are variable, for searching optimization
    "dropout_p": tune.grid_search([0, 0.2, 0.3, 0.5]),
    "crop_box": tune.grid_search([
        None,  # The whole image
        [130, 275, 140, 140], # A small square around the skimmer sink hole
        [112, 16, 180, 790]   # A rectangle taking most of the pool edge
        ]),
    "n_conv_layers": tune.grid_search([2, 3, 5]),
    "channel_multiplier": tune.grid_search([1.5, 2, 3, 4]),
    "conv_kernel_size": tune.grid_search([4, 5, 7, 9]),
    # Others are fixed
    "n_epochs": 40,
    "train_dataset_dir": "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/datasets/water_train_set4",
    "test_dataset_dir": "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/datasets/water_test_set5",
    "annotations_file": "filtered.csv",
    "learning_rate": 1e-3,
    "report_fn": train.report
}

def train_adapter(config):
    return do_training(**config)

if __name__ == '__main__':
    analysis = tune.run(
        train_adapter,
        # Ray Tune will run as many parallel experiments as <avaliable GPUs> / <GPUs per trial>
        # Fractional values are valid
        resources_per_trial={"gpu": 0.5},
        config=params,
        mode="min",
        num_samples=3,
    )