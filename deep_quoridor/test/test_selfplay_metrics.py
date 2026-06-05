import math

from v2.selfplay_metrics import aggregate_records


def _record(**kw):
    base = dict(
        model_version=3, pid=1, sims=0, terminal_wins=0, truncations=0,
        max_depth=0, sum_depth=0, moves=0, sum_root_entropy=0.0,
        sum_top_move_frac=0.0, sum_nodes=0, sum_internal_nodes=0,
        games_generated=0, unique_full=0, unique_opening=0,
    )
    base.update(kw)
    return base


def test_aggregate_combines_two_processes():
    r1 = _record(
        sims=100, terminal_wins=10, truncations=5, max_depth=12, sum_depth=400,
        moves=20, sum_root_entropy=20.0, sum_top_move_frac=10.0, sum_nodes=600,
        sum_internal_nodes=300, games_generated=2, unique_full=2, unique_opening=1,
    )
    r2 = _record(
        sims=300, terminal_wins=30, truncations=15, max_depth=18, sum_depth=1200,
        moves=60, sum_root_entropy=66.0, sum_top_move_frac=36.0, sum_nodes=1800,
        sum_internal_nodes=900, games_generated=6, unique_full=5, unique_opening=2,
    )
    agg = aggregate_records([r1, r2])

    sims, moves = 400, 80
    assert agg["selfplay/terminal_sim_frac"] == 40 / sims
    assert agg["selfplay/truncation_sim_frac"] == 20 / sims
    assert agg["selfplay/max_tree_depth"] == 18
    assert agg["selfplay/mean_tree_depth"] == 1600 / sims
    assert agg["selfplay/root_visit_entropy"] == 86.0 / moves
    assert agg["selfplay/root_visit_perplexity"] == math.exp(86.0 / moves)
    assert agg["selfplay/top_move_visit_frac"] == 46.0 / moves
    assert agg["selfplay/mean_nodes_per_search"] == 2400 / moves
    assert agg["selfplay/mean_branching"] == (2400 - moves) / 1200
    assert agg["selfplay/games_generated"] == 8
    assert agg["selfplay/unique_games_full"] == 7
    assert agg["selfplay/unique_games_opening"] == 3
    assert agg["selfplay/unique_frac_full"] == 7 / 8
    assert agg["selfplay/unique_frac_opening"] == 3 / 8


def test_aggregate_skips_empty():
    assert aggregate_records([_record(moves=0, sims=0)]) is None
