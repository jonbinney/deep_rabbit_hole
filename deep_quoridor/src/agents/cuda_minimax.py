import random
import time
from dataclasses import dataclass, fields
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np

from numba import cuda
from quoridor import Action, ActionEncoder, Player, Quoridor, WallAction, MoveAction, construct_game_from_observation
import quoridor_cuda as qcuda
from utils import SubargsBase

from agents.core import Agent
from agents.simple import SimpleParams, compute_wall_weight

# We use a large reward to encourage the agent to win the game, but we can't use infinity
# because then multiplying by a discount factor won't decrease the reward.
WINNING_REWARD = 1e6


@dataclass
class CudaMinimaxParams(SimpleParams):
    nick: Optional[str] = None
    max_depth: int = 2
    branching_factor: int = 20
    wall_sigma: float = 0.5
    discount_factor: float = 0.99
    cuda_batch_size: int = 128
    use_cuda: bool = True


def encode_action(action: Action) -> np.ndarray:
    """
    Encode an action into a numpy array format for CUDA processing.
    
    Returns:
        A tuple with (action_type, action_data) where:
        - action_type: 0 for move, 1 for wall
        - action_data: numpy array with action parameters
            For moves: [dest_row, dest_col, 0, prev_row, prev_col]
            For walls: [row, col, orientation, 0, 0]
    """
    action_data = np.zeros(5, dtype=np.int32)
    
    if isinstance(action, MoveAction):
        action_data[0] = action.destination[0]  # dest_row
        action_data[1] = action.destination[1]  # dest_col
        # action_data[3] and [4] will be filled in later with previous position
        return qcuda.ACTION_MOVE, action_data
    elif isinstance(action, WallAction):
        action_data[0] = action.position[0]  # row
        action_data[1] = action.position[1]  # col
        action_data[2] = int(action.orientation)  # orientation
        return qcuda.ACTION_WALL, action_data
    else:
        raise ValueError(f"Unknown action type: {action}")


def sample_actions(game: Quoridor, n: int, wall_sigma: float | None = None) -> List[Action]:
    """
    Choose a sample of valid actions for the current player.
    
    Starts by including all valid move actions, then adds a random sample of valid wall actions.
    """
    actions = game.get_valid_move_actions()
    if len(actions) > n:
        actions = random.sample(actions, n)
    
    if len(actions) < n:
        num_wall_actions = n - len(actions)
        wall_actions = game.get_valid_wall_actions()
        if len(wall_actions) > num_wall_actions:
            if wall_sigma is None:
                wall_actions = random.sample(wall_actions, num_wall_actions)
            else:
                position_1 = game.board.get_player_position(Player.ONE)
                position_2 = game.board.get_player_position(Player.TWO)
                wall_weights = [compute_wall_weight(wall, position_1, position_2, wall_sigma) for wall in wall_actions]
                wall_actions = np.random.choice(
                    wall_actions,
                    size=num_wall_actions,
                    replace=False,
                    p=wall_weights / np.sum(wall_weights),
                )
        
        actions.extend(wall_actions)
    
    return actions


def evaluate_with_cuda(game: Quoridor, actions: List[Action], player: Player, 
                      opponent: Player, max_depth: int, discount_factor: float) -> List[float]:
    """
    Evaluate a list of actions using CUDA parallelization.
    
    Returns:
        List of evaluation scores for each action
    """
    # Setup basic information needed for all evaluations
    grid = game.board._grid
    board_size = game.board.board_size
    p1_pos = game.board.get_player_position(Player.ONE)
    p2_pos = game.board.get_player_position(Player.TWO)
    p1_target = game.get_goal_row(Player.ONE)
    p2_target = game.get_goal_row(Player.TWO)
    walls_remaining = [
        game.board.get_walls_remaining(Player.ONE),
        game.board.get_walls_remaining(Player.TWO)
    ]
    current_player = int(game.get_current_player())
    is_p1_turn = (game.get_current_player() == Player.ONE)
    
    # Encode all actions for CUDA
    action_types = []
    action_data = []
    
    for action in actions:
        action_type, data = encode_action(action)
        # For move actions, store the previous position for undoing
        if action_type == qcuda.ACTION_MOVE:
            pos = game.board.get_player_position(game.get_current_player())
            data[3] = pos[0]  # prev_row
            data[4] = pos[1]  # prev_col
        
        action_types.append(action_type)
        action_data.append(data)
    
    # Prepare batch data
    batch_size = len(actions)
    grids = [grid.copy() for _ in range(batch_size)]
    p1_positions = [p1_pos for _ in range(batch_size)]
    p2_positions = [p2_pos for _ in range(batch_size)]
    p1_targets = [p1_target for _ in range(batch_size)]
    p2_targets = [p2_target for _ in range(batch_size)]
    walls_remaining_batch = [walls_remaining for _ in range(batch_size)]
    current_players = [current_player for _ in range(batch_size)]
    board_sizes = [board_size for _ in range(batch_size)]
    max_depths = [max_depth - 1 for _ in range(batch_size)]  # -1 because we're applying action
    is_p1_turns = [int(is_p1_turn) for _ in range(batch_size)]
    discount_factors = [discount_factor for _ in range(batch_size)]
    
    # Run batch evaluation
    evaluations = qcuda.batch_minimax(
        grids, p1_positions, p2_positions,
        p1_targets, p2_targets, walls_remaining_batch,
        action_types, action_data, current_players,
        board_sizes, max_depths, is_p1_turns, discount_factors
    )
    
    return evaluations


def choose_action_cuda(
    game: Quoridor,
    player: Player,
    opponent: Player,
    max_depth: int,
    branching_factor: int,
    wall_sigma: float | None,
    discount_factor: float,
    cuda_batch_size: int = 128,
) -> Tuple[Optional[Action], float]:
    """
    CUDA-accelerated minimax search to find the best action.
    """
    # Check terminal states
    if game.check_win(player):
        return None, WINNING_REWARD
    elif game.check_win(opponent):
        return None, -WINNING_REWARD
    elif max_depth == 0:
        return None, game.player_distance_to_target(opponent) - game.player_distance_to_target(player)
    
    # Sample actions to evaluate
    actions = sample_actions(game, branching_factor, wall_sigma)
    if not actions:
        return None, 0
    
    # Process actions in batches if needed
    if len(actions) > cuda_batch_size:
        all_values = []
        for i in range(0, len(actions), cuda_batch_size):
            batch = actions[i:i+cuda_batch_size]
            batch_values = evaluate_with_cuda(game, batch, player, opponent, max_depth, discount_factor)
            all_values.extend(batch_values)
        values = np.array(all_values)
    else:
        values = evaluate_with_cuda(game, actions, player, opponent, max_depth, discount_factor)
    
    # Select the best action
    if game.get_current_player() == player:
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    
    return actions[best_idx], values[best_idx]


def choose_action_cpu(
    game: Quoridor,
    player: Player,
    opponent: Player,
    max_depth: int,
    branching_factor: int,
    wall_sigma: float | None,
    discount_factor: float,
) -> Tuple[Optional[Action], float]:
    """
    CPU fallback implementation of minimax search.
    Identical to the original implementation but with Numba optimization.
    """
    if game.check_win(player):
        return None, WINNING_REWARD
    elif game.check_win(opponent):
        return None, -WINNING_REWARD
    elif max_depth == 0:
        return None, game.player_distance_to_target(opponent) - game.player_distance_to_target(player)
    
    actions = sample_actions(game, branching_factor, wall_sigma)
    assert len(actions) > 0, "There are no valid actions for the current player."
    
    chosen_action = None
    chosen_value = -float("inf") if game.get_current_player() == player else float("inf")
    
    for action in actions:
        current_player_before_action = game.get_current_player()
        position_before_action = game.board.get_player_position(current_player_before_action)
        
        # Apply the action to the game
        game.step(action)
        
        # Recursively call this function to choose the next player's action
        _, value = choose_action_cpu(
            game, player, opponent, max_depth - 1, 
            branching_factor, wall_sigma, discount_factor
        )
        value = discount_factor * value
        
        if current_player_before_action == player:
            if value >= chosen_value:
                chosen_value = value
                chosen_action = action
        else:
            if value <= chosen_value:
                chosen_value = value
                chosen_action = action
        
        # Undo the action to put the game back to its original state
        if isinstance(action, WallAction):
            game.board.remove_wall(current_player_before_action, action.position, action.orientation)
        else:
            game.board.move_player(current_player_before_action, position_before_action)
        game.set_current_player(current_player_before_action)
    
    return chosen_action, chosen_value


class CudaMinimaxAgent(Agent):
    def __init__(self, params=None, **kwargs):
        super().__init__()
        self.params = params or CudaMinimaxParams()
        self.board_size = kwargs["board_size"]
        self.action_encoder = ActionEncoder(self.board_size)
        
        # Check if CUDA is available and initialize
        self.cuda_available = qcuda.is_cuda_available()
        if self.params.use_cuda and self.cuda_available:
            print("CUDA is available - initializing CUDA accelerated minimax")
            try:
                # Run a small test to ensure CUDA is working correctly
                test_grid = np.zeros((5, 5), dtype=np.int32)
                test_positions = [(0, 0)]
                test_targets = [4]
                test_board_sizes = [2]
                qcuda.evaluate_positions(
                    [test_grid], test_positions, test_positions,
                    test_targets, test_targets, [0], [[10, 10]], test_board_sizes
                )
                print("CUDA initialization successful")
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU implementation")
                self.cuda_available = False
        else:
            if self.params.use_cuda and not self.cuda_available:
                print("CUDA requested but not available - using CPU implementation")
            else:
                print("Using CPU implementation as requested")
    
    @classmethod
    def params_class(cls):
        return CudaMinimaxParams
    
    def name(self):
        if self.params.nick:
            return self.params.nick
        
        param_strings = []
        for f in fields(self.params):
            value = getattr(self.params, f.name)
            if f.name != "nick" and value != f.default:
                param_strings.append(f"{f.name}={getattr(self.params, f.name)}")
        
        cuda_prefix = "cuda_" if self.params.use_cuda and self.cuda_available else ""
        return f"{cuda_prefix}minimax " + ",".join(param_strings)
    
    def start_game(self, game, player_id):
        self.player_id = player_id
    
    def get_action(self, observation):
        action_mask = observation["action_mask"]
        observation = observation["observation"]
        
        game, player, opponent = construct_game_from_observation(observation, self.player_id)
        
        start_time = time.time()
        
        # Choose between CUDA and CPU implementation
        if self.params.use_cuda and self.cuda_available:
            chosen_action, chosen_value = choose_action_cuda(
                game,
                player,
                opponent,
                self.params.max_depth,
                self.params.branching_factor,
                self.params.wall_sigma,
                self.params.discount_factor,
                self.params.cuda_batch_size,
            )
        else:
            chosen_action, chosen_value = choose_action_cpu(
                game,
                player,
                opponent,
                self.params.max_depth,
                self.params.branching_factor,
                self.params.wall_sigma,
                self.params.discount_factor,
            )
        
        end_time = time.time()
        
        if chosen_action is None:
            raise ValueError("No valid action found")
        
        assert game.is_action_valid(chosen_action), "The chosen action is not valid."
        action_id = self.action_encoder.action_to_index(chosen_action)
        assert action_mask[action_id] == 1, "The action is not valid according to the action mask."
        
        print(f"Action selection took {end_time - start_time:.4f} seconds")
        return action_id