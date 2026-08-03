"""Assemble the /api/config payload (board dims + AlphaZero defaults) from a
loaded UserConfig. The relevant fields live across the alphazero + self_play
sections, so we gather them into one flat 'defaults' block for the UI."""

from v2.config import UserConfig


def build_config_view(cfg: UserConfig) -> dict:
    sp = cfg.self_play
    spa = sp.alphazero  # Optional[AlphaZeroSelfPlayConfig]
    return {
        "board_size": cfg.quoridor.board_size,
        "max_walls": cfg.quoridor.max_walls,
        "max_steps": cfg.quoridor.max_steps,
        "defaults": {
            "mcts_n": cfg.alphazero.mcts_n,
            "mcts_c_puct": cfg.alphazero.mcts_c_puct,
            "temperature": spa.temperature if spa else None,
            "mcts_noise_epsilon": spa.mcts_noise_epsilon if spa else 0.0,
            "mcts_noise_alpha": spa.mcts_noise_alpha if spa else None,
            "leaf_parallelism": sp.leaf_parallelism,
            "virtual_loss": sp.virtual_loss,
            "mcts_worker_threads": sp.mcts_worker_threads,
        },
    }
