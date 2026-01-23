import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: F821

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from config import AlphaZeroPlayConfig, AlphaZeroSelfPlayConfig, Config, load_config_and_setup_run


def create_alphazero(
    config: Config,
    sub_config: Optional[AlphaZeroPlayConfig | AlphaZeroSelfPlayConfig],
    training_mode: bool,
) -> AlphaZeroAgent:
    mcts_n = config.alphazero.mcts_n
    mcts_ucb_c = config.alphazero.mcts_c_puct
    # TODO temperature

    if config.alphazero.network.type == "mlp":
        nn_type = "mlp"
        nn_mask_training_predictions = config.alphazero.network.mask_training_predictions
        # Those 2 are not needed for ml
        nn_resnet_num_blocks = None
        nn_resnet_num_channels = 32
    elif config.alphazero.network.type == "resnet":
        nn_type = "resnet"
        nn_mask_training_predictions = config.alphazero.network.mask_training_predictions
        nn_resnet_num_blocks = config.alphazero.network.num_blocks
        nn_resnet_num_channels = config.alphazero.network.num_channels

    else:
        raise ValueError(f"Unknown nn_type {config.alphazero.network.type}")

    if isinstance(sub_config, AlphaZeroPlayConfig):
        if sub_config.mcts_n is not None:
            mcts_n = sub_config.mcts_n
        if sub_config.mcts_c_puct is not None:
            mcts_ucb_c = sub_config.mcts_c_puct

    if isinstance(sub_config, AlphaZeroSelfPlayConfig):
        mcts_noise_epsilon = sub_config.mcts_noise_epsilon
        mcts_noise_alpha = sub_config.mcts_noise_alpha
    else:
        mcts_noise_epsilon = 0.25
        mcts_noise_alpha = None

    params = AlphaZeroParams(
        mcts_n=mcts_n,
        mcts_ucb_c=mcts_ucb_c,
        training_mode=training_mode,
        nn_type=nn_type,
        nn_mask_training_predictions=nn_mask_training_predictions,
        nn_resnet_num_blocks=nn_resnet_num_blocks,
        nn_resnet_num_channels=nn_resnet_num_channels,
        mcts_noise_epsilon=mcts_noise_epsilon,
        mcts_noise_alpha=mcts_noise_alpha,
    )
    return AlphaZeroAgent(
        config.quoridor.board_size,
        config.quoridor.max_walls,
        config.quoridor.max_steps,
        params=params,
    )


config = load_config_and_setup_run("deep_quoridor/experiments/B5W3/demo.yaml", "/Users/amarcu/code/deep_rabbit_hole")

az = create_alphazero(config, config.benchmarks[0].jobs[0].alphazero, True)
print(az.__dict__)
